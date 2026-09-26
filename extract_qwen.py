"""
Qwen3-VL-2B-Instruct(로컬)로 상품 대표 사진에서 속성 추출 → 두 가지 확인
  1) 정답 있는 상품(eval_sample 1,000개): 사진으로 뽑은 값이 판매자 값과 맞나?
  2) 빈칸 상품(missing_sample 100개): 비어 있던 속성을 몇 개나 채웠나? (+ 직접 확인용 CSV)

사용법
  # 추출 (중간에 끊겨도 다시 실행하면 이어서 진행)
  python extract_qwen.py --model_path ../fashion/Qwen3-VL-2B-Instruct --mode image
  # 평가
  python extract_qwen.py --eval --mode image

  --mode image : 사진만 / text : 텍스트만 / both : 사진+텍스트 (나중에 비교용)
필요: pip install -U transformers accelerate pillow  (Qwen3-VL은 transformers 4.57 이상)
"""
import argparse
import json
import os
import re

import pandas as pd

from prepare_eval_sample import ACTIVE_RX, BENEFIT_RX, FORM_RX, SKIN_RX

DATA = "./data_skincare"
FORMS = [l for l, _ in FORM_RX]
SKINS = [l for l, _ in SKIN_RX]
BENEFITS = [l for l, _ in BENEFIT_RX]
ACTIVES = [l for l, _ in ACTIVE_RX]


# ── 프롬프트 ────────────────────────────────────────────────────
def build_prompt(mode, row):
    src = {"image": "the product photo",
           "multi": "several photos of the product (front, back, sides, details)",
           "text": "the product text below",
           "both": "the product photo and the product text below"}[mode]
    prompt = f"""You are extracting attributes of a skincare product from {src}.

Step 1. In "visible_text", copy the short phrases you can actually read that mention
the product form, skin type, benefits or ingredients (up to 40 words). If you can read nothing useful, use "".
Step 2. Fill the other fields ONLY from what is written in visible_text. Do NOT guess.
If visible_text does not mention something, use null or [].

Choose values ONLY from these lists:
- form (one value or null): {FORMS}
- skin_type (list): {SKINS}
- benefits (list): {BENEFITS}
- active_ingredients (list): {ACTIVES}

Return one JSON object only, with keys in this order:
visible_text, form, skin_type, benefits, active_ingredients"""
    if mode in ("text", "both"):
        text = f"Title: {row['title']}\nFeatures: {row['features']}\nDescription: {row['description']}"
        prompt += "\n\nProduct text:\n" + text[:2000]
    return prompt


# ── 결과 파싱 ──────────────────────────────────────────────────
# 1) 모델이 "dry, sensitive"처럼 문자열로 줘도, "all day hydration"처럼 어휘 밖 표현을 줘도
#    정답 정규화와 같은 규칙으로 매핑
# 2) *_g: 읽은 문구(visible_text)에 근거가 있는 라벨만 남긴 버전 (환각 제거용)
from prepare_eval_sample import BENEFIT_C, ACTIVE_C, FORM_C, SKIN_C, norm_form, norm_multi

# 근거 검증용 피부타입 규칙: 'all day' 같은 문구에 all이 걸리지 않도록 더 엄격하게
SKIN_GROUND_C = [(lab, re.compile(rx)) for lab, rx in [
    ("all", r"\b(all|any|every)\s+(skin|types?)"),
    ("dry", r"\b(dry|dehydrated)\b"),
    ("oily", r"\boily\b"),
    ("combination", r"\b(combination|combo)\b"),
    ("normal", r"\bnormal\b"),
    ("sensitive", r"\b(sensitive|irritated|reactive|eczema|rosacea)\b"),
    ("acne_prone", r"\b(acne|blemish(ed)?|break-?outs?)"),
    ("mature", r"\b(mature|aging|ageing|wrinkl\w*)"),
]]


def _to_text(v):
    if v is None:
        return None
    return ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)


def parse(raw):
    d, truncated = None, False
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
    if d is None:
        # 생성 길이 한도로 JSON이 잘린 경우: 읽은 문구만 살려서 규칙으로 라벨링
        vt = re.search(r'"visible_text"\s*:\s*"(.*)', raw, re.S)
        if not vt:
            return None
        d = {"visible_text": vt.group(1).replace("\\n", " ")}
        truncated = True

    evidence = str(d.get("visible_text") or d.get("evidence") or "")
    ev = evidence.lower().replace("_", " ")
    ground = {
        "skin": {lab for lab, rx in SKIN_GROUND_C if rx.search(ev)},
        "benefit": {lab for lab, rx in BENEFIT_C if rx.search(ev)},
        "active": {lab for lab, rx in ACTIVE_C if rx.search(ev)},
    }
    form_in_text = [lab for lab, rx in FORM_C if rx.search(ev)]

    if truncated:  # 모델 판단 없이 읽은 글자 기준 규칙 결과
        form = form_in_text[0] if form_in_text else None
        labels = {c: sorted(v) for c, v in ground.items()}
    else:
        form = norm_form(_to_text(d.get("form")))
        form = None if form == "other" else form
        labels = {
            "skin": norm_multi(_to_text(d.get("skin_type")), SKIN_C) or [],
            "benefit": norm_multi(_to_text(d.get("benefits")), BENEFIT_C) or [],
            "active": norm_multi(_to_text(d.get("active_ingredients")), ACTIVE_C) or [],
        }

    out = {"form": form, "form_g": form if form in form_in_text else None,
           "evidence": evidence[:300], "truncated": truncated}
    for c, v in labels.items():
        out[c] = v
        out[f"{c}_g"] = [x for x in v if x in ground[c]]
    return out


# 줄바꿈으로 단어가 끊겨도 잡히도록 '공백 제거 버전'에서도 찾을 긴 성분명
LONG_ACTIVES = {"retinol", "niacinamide", "hyaluronic_acid", "salicylic_acid", "glycolic_acid",
                "peptides", "ceramide", "squalane", "collagen"}


def label_by_rules(text):
    """모델 없이 읽은 글자만 보고 규칙으로 라벨 (OCR + 규칙 기준선). parse()와 같은 형식."""
    ev = (text or "").lower().replace("_", " ")
    compact = re.sub(r"\s+", "", ev)
    labels = {
        "skin": sorted({lab for lab, rx in SKIN_GROUND_C if rx.search(ev)}),
        "benefit": sorted({lab for lab, rx in BENEFIT_C if rx.search(ev)}),
        "active": sorted({lab for lab, rx in ACTIVE_C
                          if rx.search(ev) or (lab in LONG_ACTIVES and rx.search(compact))}),
    }
    forms = [lab for lab, rx in FORM_C if rx.search(ev)]
    form = forms[0] if forms else None
    out = {"form": form, "form_g": form, "evidence": (text or "")[:300], "truncated": False}
    for c, v in labels.items():
        out[c] = v
        out[f"{c}_g"] = v
    return out


# ── 추출 ────────────────────────────────────────────────────────
def load_model(path):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as VLM
    except ImportError:
        raise SystemExit("transformers가 낮은 버전이에요: pip install -U transformers")
    try:
        model = VLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    except TypeError:  # 구버전 transformers
        model = VLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    p = next(model.parameters())
    print(f"모델 로딩: dtype={p.dtype}, device={p.device}, "
          f"GPU 메모리 {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    return model, AutoProcessor.from_pretrained(path)


def generate(model, proc, prompt, imgs):
    import torch
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=imgs if imgs else None,
                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def run_extract(args):
    from PIL import Image
    from tqdm import tqdm

    samples = pd.concat([
        pd.read_parquet(os.path.join(DATA, "eval_sample.parquet")).assign(split="eval"),
        pd.read_parquet(os.path.join(DATA, "missing_sample.parquet")).assign(split="missing"),
    ])
    out_path = os.path.join(DATA, f"pred_{args.mode}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["parent_asin"] for l in f}
    todo = samples[~samples["parent_asin"].isin(done)]
    print(f"전체 {len(samples)}, 완료 {len(done)}, 남음 {len(todo)}")

    model, proc = load_model(args.model_path)

    multi_paths = {}
    if args.mode == "multi":
        u = pd.read_parquet(os.path.join(DATA, "image_urls_hires.parquet"))
        u = u[u["ok"]].sort_values(["parent_asin", "order"])
        multi_paths = u.groupby("parent_asin")["path"].apply(list).to_dict()

    with open(out_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            if args.mode in ("image", "both"):
                paths = [row["img_path"]]
            elif args.mode == "multi":
                paths = multi_paths.get(row["parent_asin"], [row["img_path"]])
            else:
                paths = []
            imgs = []
            for p in paths:
                im = Image.open(p).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                imgs.append(im)
            raw = generate(model, proc, build_prompt(args.mode, row), imgs)
            rec = {"parent_asin": row["parent_asin"], "raw": raw, "parsed": parse(raw)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ── 평가 ────────────────────────────────────────────────────────
def as_list(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    return [v] if isinstance(v, str) else list(v)


def gt_labels(v):
    return [x for x in as_list(v) if x != "other"]


def multi_scores(df, col):
    """판매자 정답이 있는 상품만: 답한 비율 / 정밀도 / 재현율"""
    g = df[df[f"gt_{col}"].map(bool)]
    if len(g) == 0:
        return "정답 없음"
    tp = fp = fn = answered = 0
    for p, t in zip(g[f"pred_{col}"], g[f"gt_{col}"]):
        p, t = set(p), set(t)
        answered += bool(p)
        tp, fp, fn = tp + len(p & t), fp + len(p - t), fn + len(t - p)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return f"n={len(g):4d}  답함 {answered / len(g):5.1%}  정밀도 {prec:5.1%}  재현율 {rec:5.1%}"


def run_eval(args):
    pred_path = os.path.join(DATA, f"pred_{args.mode}.jsonl")
    preds = {}
    with open(pred_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            preds[r["parent_asin"]] = r["parsed"]
    parse_fail = sum(v is None for v in preds.values())
    print(f"=== mode={args.mode}  예측 {len(preds)}개, JSON 파싱 실패 {parse_fail}개 ===")

    def attach(df):
        df = df[df["parent_asin"].isin(preds)].copy()
        p = df["parent_asin"].map(lambda a: preds[a] or {})
        df["pred_form"] = p.map(lambda d: d.get("form"))
        for c in ["skin", "benefit", "active"]:
            df[f"pred_{c}"] = p.map(lambda d, c=c: d.get(c, []))
            df[f"gt_{c}"] = df[c].map(gt_labels)
        df["gt_form"] = df["form"].map(lambda v: v if isinstance(v, str) and v != "other" else None)
        df["evidence"] = p.map(lambda d: d.get("evidence", ""))
        return df

    ev = attach(pd.read_parquet(os.path.join(DATA, "eval_sample.parquet")))

    # 고해상도 여부: 선택된 사진이 전부 hi_res인 상품 = 고해상도 상품
    hires_path = os.path.join(DATA, "image_urls_hires.parquet")
    if os.path.exists(hires_path):
        u = pd.read_parquet(hires_path)
        if "is_hires" in u.columns:
            all_hires = u.groupby("parent_asin")["is_hires"].all()
            ev["hires"] = ev["parent_asin"].map(all_hires).fillna(False)
            print(f"고해상도 상품: {ev['hires'].mean():.1%} ({ev['hires'].sum()}/{len(ev)})")

    groups = [("Face", ev[ev.segment == "Face"]), ("Body", ev[ev.segment == "Body"]), ("전체", ev)]
    if "hires" in ev.columns:
        groups += [("고해상도만", ev[ev["hires"]]), ("저해상도 포함", ev[~ev["hires"]])]

    print("\n[1] 정답 있는 상품: 사진으로 뽑은 값이 판매자 값과 맞나")
    for seg, g in groups:
        print(f"\n  <{seg}>")
        f = g[g["gt_form"].notna()]
        ans = f["pred_form"].notna()
        acc = (f.loc[ans, "pred_form"] == f.loc[ans, "gt_form"]).mean() if ans.any() else 0
        print(f"  제형     n={len(f):4d}  답함 {ans.mean():5.1%}  답한 것 중 정확도 {acc:5.1%}")
        for c, name in [("skin", "피부타입"), ("benefit", "효능"), ("active", "성분")]:
            print(f"  {name:6s} {multi_scores(g, c)}")
        g2 = g[g["gt_skin"].map(lambda t: t != ["all"])]
        print(f"  피부타입(정답이 'all'뿐인 상품 제외) {multi_scores(g2, 'skin')}")

    ms = attach(pd.read_parquet(os.path.join(DATA, "missing_sample.parquet")))
    print("\n[2] 빈칸 상품: 비어 있던 속성을 몇 개나 채웠나")
    for seg, g in [("Face", ms[ms.segment == "Face"]), ("Body", ms[ms.segment == "Body"]), ("전체", ms)]:
        print(f"  <{seg}> n={len(g)}  피부타입 채움 {g['pred_skin'].map(bool).mean():5.1%}  "
              f"효능 채움 {g['pred_benefit'].map(bool).mean():5.1%}  "
              f"제형 {g['pred_form'].notna().mean():5.1%}  성분 {g['pred_active'].map(bool).mean():5.1%}")

    review = ms[["parent_asin", "segment", "title", "img_path",
                 "pred_form", "pred_skin", "pred_benefit", "pred_active", "evidence"]]
    review_path = os.path.join(DATA, f"missing_review_{args.mode}.csv")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    print(f"\n빈칸 상품 직접 확인용: {review_path}  (사진 열어보고 채운 값이 맞는지 체크)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image", "multi", "text", "both"], default="image")
    ap.add_argument("--max_side", type=int, default=768, help="사진 긴 변 최대 픽셀 (multi는 1024 이상 권장)")
    ap.add_argument("--model_path", default="./Qwen3-VL-2B-Instruct")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    run_eval(args) if args.eval else run_extract(args)
