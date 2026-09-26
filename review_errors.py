"""
오답 직접 검수용: OCR+규칙(고해상도 전체 기준)이 틀린 사례를 속성별로 뽑아 CSV로 저장

- 오답 = 예측했는데 판매자 정답에 없는 라벨 (피부타입·효능·성분), 또는 제형이 정답과 다름
- 각 오답마다: 규칙이 걸린 문구(앞뒤 문맥), 그 문구가 나온 사진 경로
- 엑셀에서 사진을 열어보고 '판정' 칸에 A / B / C 입력
    A = 정답이 불완전 (사진에 실제로 그런 내용이 있음 → 사실상 맞춘 것)
    B = 규칙이 틀림 (사용법·전성분·주의 문구·부정 표현 등에 잘못 걸림)
    C = 애매함

사용법: python review_errors.py --n 10
출력:   data_skincare/error_review.csv
"""
import argparse
import json
import os
import random

import pandas as pd

from extract_per_image import DATA, IDX, OCR_PRED, aggregate, attach, load_samples
from extract_qwen import ACTIVE_C, BENEFIT_C, FORM_C, SKIN_GROUND_C, label_by_rules

RULES = {"skin": dict(SKIN_GROUND_C), "benefit": dict(BENEFIT_C),
         "active": dict(ACTIVE_C), "form": dict(FORM_C)}
NAMES = {"skin": "피부타입", "benefit": "효능", "active": "성분", "form": "제형"}


def find_evidence(images, attr, label, width=60):
    """라벨이 걸린 첫 사진과 그 주변 문구"""
    rx = RULES[attr].get(label)
    for path, text in images:
        low = text.lower().replace("_", " ")   # 길이가 같아 원문 위치와 대응됨
        m = rx.search(low) if rx else None
        if m:
            s, e = max(0, m.start() - width), min(len(text), m.end() + width)
            return path, "…" + text[s:e] + "…"
    return (images[0][0] if images else ""), "(공백 제거 매칭 등으로 걸림)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="속성별로 뽑을 오답 수")
    ap.add_argument("--with_active", action="store_true", help="성분 오답도 포함")
    args = ap.parse_args()
    random.seed(42)

    idx = pd.read_parquet(IDX)
    idx = idx[idx["ok"] & idx["is_hires"]]          # 메인 조합: 고해상도 전체
    ocr = {}
    with open(OCR_PRED, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            ocr[tuple(r["key"])] = r["text"]
    idx = idx[[(a, k) in ocr for a, k in zip(idx.parent_asin, idx.order)]].sort_values(["parent_asin", "order"])

    images, agg = {}, {}
    for a, g in idx.groupby("parent_asin"):
        texts = [ocr[(a, k)] for k in g["order"]]
        images[a] = list(zip(g["path"], texts))
        agg[a] = aggregate([label_by_rules(t) for t in texts])

    ev, _ = load_samples()
    e = attach(ev, agg)

    attrs = ["form", "skin", "benefit"] + (["active"] if args.with_active else [])
    rows = []
    for attr in attrs:
        cands = []
        for r in e.itertuples():
            if attr == "form":
                if r.gt_form and r.pred_form and r.pred_form != r.gt_form:
                    cands.append((r, r.pred_form, r.gt_form))
            else:
                gt = list(getattr(r, f"gt_{attr}"))
                if not gt:
                    continue
                for lab in sorted(set(getattr(r, f"pred_{attr}")) - set(gt)):
                    cands.append((r, lab, ", ".join(gt)))
        print(f"{NAMES[attr]} 오답 후보 {len(cands)}개 → {min(args.n, len(cands))}개 추출")
        for r, lab, gt in random.sample(cands, min(args.n, len(cands))):
            path, snip = find_evidence(images[r.parent_asin], attr, lab)
            rows.append({"속성": NAMES[attr], "상품ID": r.parent_asin, "세그먼트": r.segment,
                         "상품명": r.title[:80], "예측": lab, "판매자 정답": gt,
                         "걸린 문구": snip, "사진": path, "판정(A/B/C)": "", "메모": ""})

    out = os.path.join(DATA, "error_review.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}  (총 {len(rows)}개)")
    print("판정: A=정답이 불완전(사진에 실제로 있음)  B=규칙이 틀림  C=애매")


if __name__ == "__main__":
    main()
