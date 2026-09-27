"""
상품 텍스트 임베딩 만들기 — 추천 실험의 '텍스트 임베딩 베이스라인'용

상품명 + 특징(features) + 설명(description) 앞부분을 문장 임베딩 모델로 벡터화해 저장합니다.
대상: interactions_u5.parquet에 등장하는 상품 (추천 실험의 상품 전체)

사용:
  python build_text_emb.py                                  # 기본: BAAI/bge-small-en-v1.5 (384차원, 빠름)
  python build_text_emb.py --model BAAI/bge-base-en-v1.5    # 더 큰 모델 (768차원)
  python build_text_emb.py --max_chars 1000 --batch 128
결과:
  data_skincare/text_emb.npy          (상품 수 × 차원, L2 정규화)
  data_skincare/text_emb_asins.parquet (행 순서에 맞는 parent_asin)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

DATA = "./data_skincare"


def to_text(v):
    """리스트/배열/문자열 모두 문자열로"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, str):
        return v
    try:
        return " ".join(str(x) for x in list(v) if x)
    except TypeError:
        return str(v)


def item_text(row, max_chars):
    title = to_text(row.get("title"))
    feats = to_text(row.get("features"))
    desc = to_text(row.get("description"))
    text = f"{title}. {feats} {desc}".strip()
    return text[:max_chars] if text else ""


def encode(texts, model_name, batch, device):
    """sentence-transformers가 있으면 사용, 없으면 transformers + mean pooling"""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, device=device)
        return m.encode(texts, batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    except ImportError:
        import torch
        from tqdm import tqdm
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device).eval()
        out = []
        for s in tqdm(range(0, len(texts), batch)):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                h = mdl(**enc).last_hidden_state
            if "bge" in model_name.lower():          # bge는 CLS 토큰 사용
                e = h[:, 0]
            else:                                     # 그 외는 mean pooling
                msk = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
            e = torch.nn.functional.normalize(e, dim=-1)
            out.append(e.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max_chars", type=int, default=1500, help="상품당 텍스트 최대 글자 수")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inter = pd.read_parquet(os.path.join(DATA, "interactions_u5.parquet"), columns=["parent_asin"])
    asins = sorted(inter["parent_asin"].unique())
    items = pd.read_parquet(os.path.join(DATA, "items_norm.parquet"))
    cols = [c for c in ["parent_asin", "title", "features", "description"] if c in items.columns]
    items = items[cols].drop_duplicates("parent_asin").set_index("parent_asin")
    texts = [item_text(items.loc[a].to_dict(), args.max_chars) if a in items.index else "" for a in asins]
    n_empty = sum(1 for t in texts if not t)
    print(f"상품 {len(asins):,}개, 텍스트 없음 {n_empty:,}개, 평균 길이 {np.mean([len(t) for t in texts]):.0f}자")
    print(f"모델 {args.model}, 장치 {device}")

    t0 = time.time()
    emb = encode([t if t else "unknown product" for t in texts], args.model, args.batch, device)
    emb[[i for i, t in enumerate(texts) if not t]] = 0.0          # 텍스트 없는 상품은 0 벡터
    print(f"임베딩 {emb.shape}, {time.time() - t0:.0f}초")

    np.save(os.path.join(DATA, "text_emb.npy"), emb)
    pd.DataFrame({"parent_asin": asins}).to_parquet(os.path.join(DATA, "text_emb_asins.parquet"))
    print(f"저장: {DATA}/text_emb.npy, text_emb_asins.parquet")


if __name__ == "__main__":
    main()
