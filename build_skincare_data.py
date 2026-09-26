"""
스킨케어 실험용 데이터 만들기
1) 메타에서 Skin Care 상품만 추출 → items.parquet
   - segment: Face / Body / Sunscreens... (카테고리 3단계)
   - LLM 입력: title, features, description, 대표 이미지 URL(MAIN)
   - 정답(details): Item Form, Skin Type, Product Benefits, Active Ingredients
2) 정답 속성의 원본 값 분포 출력 → 정규화 어휘 설계용
3) 리뷰에서 인증구매 + 스킨케어 상품만 추출 → interactions.parquet
   - 유저 k개 이상 필터 후 규모를 segment별로 출력

사용법: python build_skincare_data.py --meta meta_Beauty_and_Personal_Care.jsonl \
          --review Beauty_and_Personal_Care.jsonl --out_dir ./data_skincare --min_user 5
필요: pip install pandas pyarrow
"""
import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

GT_KEYS = ["Item Form", "Skin Type", "Product Benefits", "Active Ingredients"]


def stream(path):
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def get_details(m):
    d = m.get("details")
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            d = {}
    return d if isinstance(d, dict) else {}


def main_image(m):
    """MAIN 이미지의 large URL (없으면 첫 이미지)"""
    imgs = m.get("images")
    if isinstance(imgs, dict):  # dict of lists 형식
        variants, larges = imgs.get("variant") or [], imgs.get("large") or []
        for v, u in zip(variants, larges):
            if v == "MAIN" and u:
                return u
        return larges[0] if larges else None
    if isinstance(imgs, list) and imgs:  # list of dict 형식
        for im in imgs:
            if im.get("variant") == "MAIN":
                return im.get("large") or im.get("hi_res")
        return imgs[0].get("large") or imgs[0].get("hi_res")
    return None


def join_text(x):
    if isinstance(x, list):
        return " ".join(str(s) for s in x if s)
    return x or ""


def split_values(v):
    """'Dry, Oily & Combination' → ['dry', 'oily', 'combination']"""
    parts = re.split(r",|/|;|&|\band\b", str(v).lower())
    return [p.strip() for p in parts if p.strip()]


def build_items(meta_path):
    rows = []
    for m in stream(meta_path):
        cats = m.get("categories") or []
        if len(cats) < 2 or cats[1] != "Skin Care":
            continue
        d = get_details(m)
        rows.append({
            "parent_asin": m.get("parent_asin"),
            "segment": cats[2] if len(cats) > 2 else "Other",
            "title": m.get("title") or "",
            "features": join_text(m.get("features")),
            "description": join_text(m.get("description")),
            "image_url": main_image(m),
            "store": m.get("store"),
            "price": m.get("price"),
            **{f"gt_{k.lower().replace(' ', '_')}": d.get(k) for k in GT_KEYS},
        })
    df = pd.DataFrame(rows).drop_duplicates("parent_asin")
    # price에 '—', 'None' 같은 문자열이 섞여 있어 숫자로 변환 (실패는 NaN)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # 정답/텍스트 컬럼은 문자열로 통일 (리스트 등 섞인 타입 방지)
    for c in [c for c in df.columns if c.startswith("gt_")] + ["store"]:
        df[c] = df[c].map(lambda x: None if x is None else str(x))
    return df


def report_gt(items):
    print(f"\n=== 스킨케어 상품 {len(items):,}개 ===")
    print(items["segment"].value_counts().to_string())
    for k in GT_KEYS:
        col = f"gt_{k.lower().replace(' ', '_')}"
        filled = items[col].notna()
        print(f"\n[{k}] 채움률 전체 {filled.mean():.1%}", end="")
        for seg in ["Face", "Body"]:
            s = items["segment"] == seg
            print(f" | {seg} {items.loc[s, col].notna().mean():.1%}", end="")
        cnt = Counter(v for x in items.loc[filled, col] for v in split_values(x))
        print(f"\n  고유 값(분리 후): {len(cnt):,}개, 상위 40:")
        for v, n in cnt.most_common(40):
            print(f"    {n:7,}  {v}")


def build_interactions(review_path, item_ids):
    rows = []
    for r in stream(review_path):
        if not r.get("verified_purchase"):
            continue
        a = r.get("parent_asin")
        if a in item_ids:
            rows.append((r.get("user_id"), a, r.get("rating"), r.get("timestamp")))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    # 같은 유저-상품 중복은 가장 이른 것만
    return df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--out_dir", default="./data_skincare")
    ap.add_argument("--min_user", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    items = build_items(args.meta)
    items.to_parquet(os.path.join(args.out_dir, "items.parquet"), index=False)
    report_gt(items)

    print("\n리뷰 읽는 중 (전체 파일, 수 분 걸림)...")
    inter = build_interactions(args.review, set(items["parent_asin"]))
    inter.to_parquet(os.path.join(args.out_dir, "interactions_all.parquet"), index=False)

    uc = inter["user_id"].value_counts()
    kept = inter[inter["user_id"].isin(uc[uc >= args.min_user].index)]
    kept = kept.merge(items[["parent_asin", "segment"]], on="parent_asin")
    kept.to_parquet(os.path.join(args.out_dir, f"interactions_u{args.min_user}.parquet"), index=False)

    t = pd.to_datetime(inter["timestamp"], unit="ms")
    print(f"\n=== 상호작용 (인증구매, 중복 제거) ===")
    print(f"전체: {len(inter):,}건, 유저 {inter['user_id'].nunique():,}, 상품 {inter['parent_asin'].nunique():,}")
    print(f"기간: {t.min().date()} ~ {t.max().date()}")
    print(f"유저 {args.min_user}개 이상: {len(kept):,}건, 유저 {kept['user_id'].nunique():,}, "
          f"상품 {kept['parent_asin'].nunique():,}")
    print("segment별 상호작용:")
    print(kept["segment"].value_counts().to_string())


if __name__ == "__main__":
    main()