"""
PaddleOCR로 테스트 상품 사진의 글자 읽기  ※ .venv_paddle 가상환경에서 실행

- Qwen 추출과 '같은 상품'을 사용 (extract_per_image.select_products)
- 사진마다 저장: 합친 텍스트, 크기 태그 텍스트, 원본 줄 목록, 신뢰도, 좌표
- 합치는 규칙: 신뢰도 낮은 줄 제거 → 칼럼 나누기 → 위→아래, 왼→오른 순 정렬
             → 줄 끝 하이픈은 붙여서 잇기 → 나머지는 공백으로 잇기
- 중간에 끊겨도 다시 실행하면 이어서 진행

사용법
  python ocr_paddle.py --n_per_seg 150 --n_missing 50
평가 (이후, 어느 환경이든)
  python extract_per_image.py --eval --source ocr
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from extract_per_image import DATA, IDX, OCR_PRED, select_products


def to_boxes(res):
    """[x1, y1, x2, y2] 목록. rec_boxes가 없으면 다각형 좌표에서 계산"""
    try:
        boxes = res["rec_boxes"]
        if boxes is not None and len(boxes):
            return np.asarray(boxes, dtype=float).tolist()
    except Exception:
        pass
    polys = res["rec_polys"]
    return [[float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
            for p in (np.asarray(q) for q in polys)]


def assemble(texts, scores, boxes, min_score):
    items = []
    for t, s, b in zip(texts, scores, boxes):
        if s < min_score or not str(t).strip():
            continue
        x1, y1, x2, y2 = b
        items.append({"t": str(t).strip(), "x1": x1, "x2": x2, "yc": (y1 + y2) / 2, "h": y2 - y1})
    if not items:
        return "", ""

    med_h = max(float(np.median([it["h"] for it in items])), 1.0)
    # 칼럼 나누기: 줄 시작 x가 크게 벌어진 지점을 칼럼 경계로 (앞·뒷면이 나란히 찍힌 사진 대응)
    xs = sorted(it["x1"] for it in items)
    width = max(it["x2"] for it in items) - xs[0] + 1
    bounds = [(a + b) / 2 for a, b in zip(xs, xs[1:]) if b - a > 0.25 * width]
    col = lambda it: sum(it["x1"] > bd for bd in bounds)
    items.sort(key=lambda it: (col(it), round(it["yc"] / med_h), it["x1"]))

    text = "\n".join(it["t"] for it in items)
    text = re.sub(r"-\n(?=[a-z])", "", text)   # "Niacin-\namide" → "Niacinamide"
    text = text.replace("\n", " ")
    tagged = "\n".join(("[BIG] " if it["h"] >= 1.5 * med_h else "") + it["t"] for it in items)
    return text, tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_seg", type=int, default=150)
    ap.add_argument("--n_missing", type=int, default=50)
    ap.add_argument("--min_score", type=float, default=0.5, help="이 신뢰도 미만인 줄은 버림")
    args = ap.parse_args()

    from paddleocr import PaddleOCR
    from tqdm import tqdm

    idx = pd.read_parquet(IDX)
    idx = idx[idx["ok"]]
    keep = select_products(args.n_per_seg, args.n_missing)
    idx = idx[idx["parent_asin"].isin(keep)].sort_values(["parent_asin", "order"])

    done = set()
    if os.path.exists(OCR_PRED):
        with open(OCR_PRED, encoding="utf-8") as f:
            done = {tuple(json.loads(l)["key"]) for l in f}
    todo = idx[[(a, k) not in done for a, k in zip(idx.parent_asin, idx.order)]]
    print(f"상품 {len(keep)}개 / 사진 {len(idx):,}장, 완료 {len(idx) - len(todo):,}장, 남음 {len(todo):,}장")

    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False)
    t0, n, n_err = time.time(), 0, 0
    with open(OCR_PRED, "a", encoding="utf-8") as f:
        for r in tqdm(todo.itertuples(), total=len(todo)):
            try:
                res = list(ocr.predict(r.path))[0]
                texts = [str(t) for t in res["rec_texts"]]
                scores = [float(s) for s in res["rec_scores"]]
                boxes = to_boxes(res)
            except Exception:
                texts, scores, boxes = [], [], []
                n_err += 1
            text, tagged = assemble(texts, scores, boxes, args.min_score)
            f.write(json.dumps({"key": [r.parent_asin, int(r.order)], "text": text, "tagged": tagged,
                                "texts": texts, "scores": scores, "boxes": boxes}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    if n:
        print(f"완료: {n}장, 평균 {(time.time() - t0) / n:.2f}초/장, 에러 {n_err}장")


if __name__ == "__main__":
    main()
