"""
1단계: 정답 정규화 / 2단계: 샘플링 + 대표 이미지 다운로드

입력: ./data_skincare/items.parquet, ./data_skincare/interactions_u5.parquet
출력: ./data_skincare/items_norm.parquet   (전체 스킨케어 상품 + 정규화된 속성)
      ./data_skincare/eval_sample.parquet  (정답 있는 상품: Face/Body 각 500 → 추출 정확도 측정용)
      ./data_skincare/missing_sample.parquet (피부타입·효능이 비어 있는 상품 100 → 빈칸 상품 품질 점검용)
      ./images_eval/{parent_asin}.jpg

정규화 결과 표기
  None = 판매자가 아예 안 채움 / [] 또는 'other' = 채웠지만 정한 어휘에 안 들어감 / 값 = 정규화 성공

사용법: pip install requests
        python prepare_eval_sample.py --n_per_seg 500 --n_missing 100
"""
import argparse
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

DATA = "./data_skincare"
IMG_DIR = "./images_eval"

# ── 정규화 어휘 (위에서부터 우선 매칭) ─────────────────────────────
FORM_RX = [  # 제형: 1개만 선택
    ("mask_sheet", r"\b(sheets?|masks?|patch(es)?|pads?|wipes?|strips?|cloths?)\b"),
    ("spray_mist", r"\b(spray|mist|aerosol)\b"),
    ("foam", r"\b(foam(ing)?|mousse|lather)\b"),
    ("serum", r"\b(serum|drops?|ampoule|essence|concentrate)\b"),
    ("clay_scrub", r"\b(clay|mud|paste|scrub|polish)\b"),
    ("stick", r"\bsticks?\b"),
    ("powder", r"\b(powder|grains?|granules?)\b"),
    ("bar_soap", r"\b(bars?|soaps?)\b"),
    ("balm", r"\b(balm|ointment|salve)\b"),
    ("oil", r"\boils?\b"),
    ("gel", r"\b(gel|jelly)\b"),
    ("lotion", r"\b(lotion|milk|emulsion)\b"),
    ("cream", r"\b(creams?|creamy|cr[eè]me|butter)\b"),
    ("liquid", r"\b(liquid|wash|water|toner|solution)\b"),
]
SKIN_RX = [  # 피부타입: 복수
    ("all", r"\b(all|any|every|universal|default)\b"),
    ("dry", r"\b(dry|dehydrated)\b"),
    ("oily", r"\boily\b"),
    ("combination", r"\b(combination|combo)\b"),
    ("normal", r"\b(normal|balanced)\b"),
    ("sensitive", r"\b(sensitive|irritated|reactive|eczema|rosacea)\b"),
    ("acne_prone", r"\b(acne|blemish(ed)?|break-?outs?|problem)"),
    ("mature", r"\b(mature|aging|ageing|wrinkl\w*)"),
]
BENEFIT_RX = [  # 효능: 복수
    ("hydrating", r"(moistur|hydrat)"),
    ("anti_aging", r"(anti[- ]?ag|aging|ageing|firm|wrinkl|lift|plump|tighten|rejuven|fine line)"),
    ("exfoliating", r"(exfoliat|peel|resurfac)"),
    ("soothing", r"(sooth|calm|gentle|redness|anti[- ]?inflam)"),
    ("smoothing", r"(smooth|soften|texture)"),
    ("cleansing", r"(cleans|\bclean\b|detox|purif|\bpores?\b)"),
    ("nourishing", r"(nourish|replenish|repair|revital|restor|barrier)"),
    ("brightening", r"(bright|whiten|radian|glow|lighten|dark spot|even (skin )?tone)"),
    ("antioxidant", r"antioxid"),
    ("sun_protection", r"(\buv\b|ultra-?violet|\bspf\b|sun ?protect|sunscreen)"),
    ("acne_care", r"(acne|blemish|breakout|oil control|mattif)"),
]
ACTIVE_RX = [  # 대표 활성 성분 15개: 복수 (베이스 성분은 무시)
    ("retinol", r"(retin(ol|al|oid|yl)|vitamin a\b)"),
    ("niacinamide", r"(niacinamide|vitamin b3)"),
    ("vitamin_c", r"(vitamin c\b|ascorb)"),
    ("hyaluronic_acid", r"hyaluron"),
    ("salicylic_acid", r"(salicyl|\bbha\b)"),
    ("glycolic_acid", r"(glycolic|\baha\b)"),
    ("lactic_acid", r"lactic"),
    ("peptides", r"peptide"),
    ("ceramide", r"ceramide"),
    ("squalane", r"squal[ae]ne"),
    ("centella", r"(centella|\bcica\b|madecass)"),
    ("tea_tree", r"tea tree"),
    ("shea_butter", r"\bshea\b"),
    ("collagen", r"collagen"),
    ("snail_mucin", r"snail"),
]
_c = lambda rules: [(lab, re.compile(rx)) for lab, rx in rules]
FORM_C, SKIN_C, BENEFIT_C, ACTIVE_C = map(_c, [FORM_RX, SKIN_RX, BENEFIT_RX, ACTIVE_RX])


def clean(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() in ("", "None"):
        return None
    return str(raw).lower().replace("_", " ")


def norm_form(raw):
    t = clean(raw)
    if t is None:
        return None
    for part in re.split(r",|/|;|&", t):
        for lab, rx in FORM_C:
            if rx.search(part):
                return lab
    return "other"


def norm_multi(raw, rules):
    t = clean(raw)
    if t is None:
        return None
    return [lab for lab, rx in rules if rx.search(t)]


def normalize(items):
    items["form"] = items["gt_item_form"].map(norm_form)
    items["skin"] = items["gt_skin_type"].map(lambda x: norm_multi(x, SKIN_C))
    items["benefit"] = items["gt_product_benefits"].map(lambda x: norm_multi(x, BENEFIT_C))
    items["active"] = items["gt_active_ingredients"].map(lambda x: norm_multi(x, ACTIVE_C))
    return items


def report_norm(items):
    print("\n=== 정규화 결과 (판매자가 채운 것 중 어휘로 정리된 비율) ===")
    specs = [("form", "gt_item_form"), ("skin", "gt_skin_type"),
             ("benefit", "gt_product_benefits"), ("active", "gt_active_ingredients")]
    for col, raw in specs:
        filled = items[col].notna()
        mapped = items[col].map(lambda v: v is not None and v != "other" and v != [])
        print(f"\n[{col}] 채움 {filled.mean():.1%} → 정규화 성공 {mapped[filled].mean():.1%}")
        vals = Counter()
        for v in items.loc[mapped, col]:
            vals.update(v if isinstance(v, list) else [v])
        print("  분포:", ", ".join(f"{k} {n:,}" for k, n in vals.most_common()))
        unmapped = items.loc[filled & ~mapped, raw].astype(str).str.lower().value_counts().head(15)
        if len(unmapped):
            print("  정리 안 된 원본 상위 15 (어휘 보강 참고):")
            for k, n in unmapped.items():
                print(f"    {n:6,}  {k[:80]}")


def has_label(v):
    return v is not None and v != "other" and v != []


def download(row):
    path = os.path.join(IMG_DIR, f"{row['parent_asin']}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, True
    try:
        import requests
        r = requests.get(row["image_url"], timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            with open(path, "wb") as f:
                f.write(r.content)
            return path, True
    except Exception:
        pass
    return path, False


def attach_images(df):
    os.makedirs(IMG_DIR, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(download, df.to_dict("records")))
    df["img_path"] = [p for p, _ in res]
    df["img_ok"] = [ok for _, ok in res]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_seg", type=int, default=500)
    ap.add_argument("--n_missing", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = normalize(pd.read_parquet(os.path.join(DATA, "items.parquet")))
    report_norm(items)
    items.to_parquet(os.path.join(DATA, "items_norm.parquet"), index=False)

    inter = pd.read_parquet(os.path.join(DATA, "interactions_u5.parquet"))
    items["n_inter"] = items["parent_asin"].map(inter["parent_asin"].value_counts()).fillna(0).astype(int)

    base = items[(items["n_inter"] > 0) & items["segment"].isin(["Face", "Body"])
                 & items["image_url"].notna()]
    gt_any = base[["form", "skin", "benefit"]].apply(lambda r: any(has_label(v) for v in r), axis=1)
    blank = base["skin"].isna() & base["benefit"].isna()

    eval_df = pd.concat([
        base[gt_any & (base["segment"] == s)].sample(args.n_per_seg, random_state=args.seed)
        for s in ["Face", "Body"]])
    miss_df = pd.concat([
        base[blank & (base["segment"] == s)].sample(args.n_missing // 2, random_state=args.seed)
        for s in ["Face", "Body"]])

    print(f"\n=== 평가 샘플 {len(eval_df):,}개: 세그먼트별 정답 보유 수 ===")
    for s, g in eval_df.groupby("segment"):
        print(f"  {s}: " + ", ".join(f"{c} {g[c].map(has_label).sum()}" for c in ["form", "skin", "benefit", "active"]))

    print("\n이미지 다운로드 중...")
    eval_df, miss_df = attach_images(eval_df), attach_images(miss_df)
    print(f"  평가 샘플 성공률 {eval_df['img_ok'].mean():.1%}, 결측 샘플 성공률 {miss_df['img_ok'].mean():.1%}")

    eval_df.to_parquet(os.path.join(DATA, "eval_sample.parquet"), index=False)
    miss_df.to_parquet(os.path.join(DATA, "missing_sample.parquet"), index=False)
    print("저장 완료")


if __name__ == "__main__":
    main()
