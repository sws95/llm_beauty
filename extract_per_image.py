"""
사진 한 장씩 추출 → 상품별로 합쳐서 평가 (사진 조합별 비교까지 한 번에)

1) --download : 테스트 1,100개 상품의 사진 전부 받기 (hi_res 우선, 없으면 저해상도 + is_hires 표시)
2) --extract  : 사진 한 장씩 Qwen3-VL-2B로 추출 (고해상도부터 처리, 끊겨도 이어서 진행)
3) --eval     : 상품별로 합쳐서 정확도 평가. 추가 추출 없이 아래 조합을 모두 비교
      - 대표 사진만
      - 대표 + 랜덤 2장
      - 고해상도 전체      ← 메인 결과 (상한선)
      - 전체(저해상도 포함) ← 참고용 (저해상도는 글씨가 안 읽혀 오류 가능성 높음)
   합치는 규칙: 피부타입·효능·성분은 합집합, 제형은 가장 많이 나온 값

사용법 (extract_qwen.py, prepare_eval_sample.py와 같은 폴더)
  python extract_per_image.py --download
  python extract_per_image.py --extract --model_path C:\\...\\Qwen3-VL-2B-Instruct
  python extract_per_image.py --eval
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from extract_qwen import DATA, build_prompt, generate, gt_labels, label_by_rules, load_model, multi_scores, parse

META = "meta_Beauty_and_Personal_Care.jsonl"
IDX = os.path.join(DATA, "image_index.parquet")
PRED = os.path.join(DATA, "pred_per_image.jsonl")
OCR_PRED = os.path.join(DATA, "ocr_paddle.jsonl")
IMG_DIR = "images_all"
ATTRS = [("skin", "피부타입"), ("benefit", "효능"), ("active", "성분")]


def load_samples():
    return (pd.read_parquet(os.path.join(DATA, "eval_sample.parquet")),
            pd.read_parquet(os.path.join(DATA, "missing_sample.parquet")))


def select_products(n_per_seg, n_missing):
    """세그먼트별 n_per_seg + 빈칸 상품 n_missing (고정 시드). Qwen·OCR이 같은 상품을 쓰도록 공용."""
    ev, ms = load_samples()
    keep = set()
    for seg in ["Face", "Body"]:
        e, m = ev[ev.segment == seg], ms[ms.segment == seg]
        keep |= set(e["parent_asin"].sample(min(n_per_seg, len(e)), random_state=42))
        keep |= set(m["parent_asin"].sample(min(n_missing // 2, len(m)), random_state=42))
    return keep


# ── 1) 다운로드 ─────────────────────────────────────────────────
def run_download():
    import requests
    ev, ms = load_samples()
    targets = set(ev["parent_asin"]) | set(ms["parent_asin"])

    rows = []
    with open(META, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            a = m["parent_asin"]
            if a not in targets:
                continue
            imgs = m["images"]
            if isinstance(imgs, dict):
                imgs = [dict(zip(imgs.keys(), v)) for v in zip(*imgs.values())]
            imgs = [im for im in imgs if im.get("hi_res") or im.get("large")]
            imgs.sort(key=lambda im: im.get("variant") != "MAIN")  # 대표 사진을 0번으로
            for k, im in enumerate(imgs):
                rows.append((a, k, im.get("variant"), im.get("hi_res") or im.get("large"),
                             bool(im.get("hi_res"))))

    idx = pd.DataFrame(rows, columns=["parent_asin", "order", "variant", "url", "is_hires"])
    idx["path"] = [os.path.join(IMG_DIR, f"{a}_{k}.jpg") for a, k in zip(idx.parent_asin, idx.order)]
    os.makedirs(IMG_DIR, exist_ok=True)

    def dl(r):
        if os.path.exists(r.path) and os.path.getsize(r.path) > 0:
            return True
        try:
            resp = requests.get(r.url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and resp.content:
                with open(r.path, "wb") as out:
                    out.write(resp.content)
                return True
        except Exception:
            pass
        return False

    with ThreadPoolExecutor(16) as ex:
        idx["ok"] = list(ex.map(dl, idx.itertuples()))
    idx.to_parquet(IDX, index=False)

    per = idx.groupby("parent_asin")
    print(f"사진 {len(idx):,}장 (상품당 평균 {per.size().mean():.1f}장), 다운로드 성공 {idx['ok'].mean():.1%}")
    print(f"고해상도 사진 비율 {idx['is_hires'].mean():.1%}, "
          f"고해상도 사진이 1장 이상인 상품 {per['is_hires'].any().mean():.1%}")


# ── 2) 사진별 추출 ──────────────────────────────────────────────
def run_extract(args):
    from PIL import Image
    from tqdm import tqdm

    idx = pd.read_parquet(IDX)
    idx = idx[idx["ok"]]
    if args.hires_only:
        idx = idx[idx["is_hires"]]

    # 상품 수 줄이기: 세그먼트별 n_per_seg + 빈칸 상품 n_missing (고정 시드)
    keep = select_products(args.n_per_seg, args.n_missing)
    idx = idx[idx["parent_asin"].isin(keep)]

    done = set()
    if os.path.exists(PRED):
        with open(PRED, encoding="utf-8") as f:
            done = {tuple(json.loads(l)["key"]) for l in f}
    todo = idx[[(a, k) not in done for a, k in zip(idx.parent_asin, idx.order)]]
    # 상품 단위로 끝까지 처리 → 중간에 멈춰도 끝난 상품은 온전히 평가 가능
    todo = todo.sort_values(["parent_asin", "order"])
    print(f"상품 {len(keep)}개 / 사진 {len(idx):,}장, 완료 {len(idx) - len(todo):,}장, 남음 {len(todo):,}장")

    model, proc = load_model(args.model_path)
    prompt = build_prompt("image", None)
    with open(PRED, "a", encoding="utf-8") as f:
        for r in tqdm(todo.itertuples(), total=len(todo)):
            try:
                im = Image.open(r.path).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                raw = generate(model, proc, prompt, [im])
            except Exception as e:
                raw = f"ERROR: {e}"
            rec = {"key": [r.parent_asin, int(r.order)], "raw": raw, "parsed": parse(raw)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ── 3) 상품별 합치기 + 평가 ─────────────────────────────────────
SUFFIX = ""  # "--grounded"면 "_g": 읽은 문구에 근거가 있는 라벨만 사용


def aggregate(parsed_list):
    parsed_list = [p for p in parsed_list if p]
    forms = [p.get("form" + SUFFIX) for p in parsed_list if p.get("form" + SUFFIX)]
    agg = {"form": Counter(forms).most_common(1)[0][0] if forms else None}
    for c, _ in ATTRS:
        agg[c] = sorted({x for p in parsed_list for x in p.get(c + SUFFIX, [])})
    return agg


CONFIGS = {
    "대표 사진만": lambda g: g[g["order"] == 0],
    "대표 + 랜덤 2장": lambda g: pd.concat([
        g[g["order"] == 0],
        g[g["order"] > 0].sample(min(2, int((g["order"] > 0).sum())), random_state=42)]),
    "고해상도 전체 (메인)": lambda g: g[g["is_hires"]],
    "전체, 저해상도 포함 (참고)": lambda g: g,
}


def attach(df, agg):
    df = df[df["parent_asin"].isin(agg)].copy()
    df["pred_form"] = df["parent_asin"].map(lambda a: agg[a]["form"])
    df["gt_form"] = df["form"].map(lambda v: v if isinstance(v, str) and v != "other" else None)
    for c, _ in ATTRS:
        df[f"pred_{c}"] = df["parent_asin"].map(lambda a, c=c: agg[a][c])
        df[f"gt_{c}"] = df[c].map(gt_labels)
    return df


def print_scores(df, indent="    "):
    f = df[df["gt_form"].notna()]
    if len(f):
        ans = f["pred_form"].notna()
        acc = (f.loc[ans, "pred_form"] == f.loc[ans, "gt_form"]).mean() if ans.any() else 0
        print(f"{indent}제형     n={len(f):4d}  답함 {ans.mean():5.1%}  답한 것 중 정확도 {acc:5.1%}")
    else:
        print(f"{indent}제형     정답 없음")
    for c, name in ATTRS:
        print(f"{indent}{name:6s} {multi_scores(df, c)}")


def fill_rate(s):
    return s.map(bool).astype(float).mean() if len(s) else float("nan")


def run_eval(args):
    idx = pd.read_parquet(IDX)
    idx = idx[idx["ok"]]
    pred = {}
    if args.source in ("ocr", "ocr2"):  # OCR로 읽은 글자 + 규칙 (v1 / v2)
        from rules_v2 import label_v2
        with open(OCR_PRED, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                pred[tuple(r["key"])] = (label_by_rules(r["text"]) if args.source == "ocr"
                                         else label_v2(r.get("tagged", "")))
    else:
        with open(PRED, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                pred[tuple(r["key"])] = parse(r["raw"])   # 파서가 바뀌어도 재추출 없이 반영
    print(f"=== source={args.source}{' (근거 검증)' if SUFFIX else ''} ===")
    idx = idx[[(a, k) in pred for a, k in zip(idx.parent_asin, idx.order)]].copy()
    idx["parsed"] = [pred[(a, k)] for a, k in zip(idx.parent_asin, idx.order)]

    # 사진이 전부 처리된 상품만 평가 (일부만 처리된 상품은 합친 결과가 왜곡됨)
    full = pd.read_parquet(IDX)
    n_all = full[full["ok"]].groupby("parent_asin").size()
    n_done = idx.groupby("parent_asin").size()
    complete = n_done.index[n_done.values == n_all.reindex(n_done.index).values]
    idx = idx[idx["parent_asin"].isin(complete)]
    print(f"사진이 모두 처리된 상품 {len(complete)}개로 평가 (사진 {len(idx):,}장)")
    n_trunc = sum(bool(p and p.get("truncated")) for p in idx["parsed"])
    print(f"추출된 사진 {len(idx):,}장, 파싱 실패 {idx['parsed'].isna().sum()}장, "
          f"출력 잘림(읽은 문구로 복구) {n_trunc}장")

    ev, ms = load_samples()
    groups = dict(tuple(idx.groupby("parent_asin")))
    titles = {**dict(zip(ev["parent_asin"], ev["title"])), **dict(zip(ms["parent_asin"], ms["title"]))}
    if args.source == "ocr2":
        from rules_v2 import title_form, title_skin

    main_agg = None
    for name, sel in CONFIGS.items():
        agg = {}
        for a, g in groups.items():
            s = sel(g)
            if len(s):
                agg[a] = aggregate(s["parsed"].tolist())
                if args.source == "ocr2":   # 제형은 상품명 우선, 피부타입은 상품명 + OCR 합집합
                    agg[a]["form"] = title_form(titles.get(a)) or agg[a]["form"]
                    agg[a]["skin"] = sorted(set(agg[a]["skin"]) | set(title_skin(titles.get(a))))
        if name.startswith("고해상도"):
            main_agg = agg
        e = attach(ev, agg)
        print(f"\n===== {name}  (평가 상품 {len(e)}개) =====")
        for seg in ["Face", "Body"]:
            print(f"  <{seg}>")
            print_scores(e[e.segment == seg])
        m = attach(ms, agg)
        if len(m):
            print(f"  <빈칸 상품 {len(m)}개> 채움률  피부타입 {fill_rate(m['pred_skin']):5.1%}  "
                  f"효능 {fill_rate(m['pred_benefit']):5.1%}  "
                  f"제형 {m['pred_form'].notna().astype(float).mean():5.1%}  "
                  f"성분 {fill_rate(m['pred_active']):5.1%}")
        else:
            print("  <빈칸 상품> 평가 가능한 상품 없음")

    # 정보가 대표 사진 밖에서만 나온 비율 (고해상도 전체 기준)
    print("\n===== 대표 사진에는 없고 다른 사진에서만 나온 정보 (고해상도 전체 기준) =====")
    hi = idx[idx["is_hires"]]
    for c, name in ATTRS:
        only_else = total = 0
        for a, g in hi.groupby("parent_asin"):
            main = aggregate(g[g["order"] == 0]["parsed"].tolist())[c]
            allv = aggregate(g["parsed"].tolist())[c]
            if allv:
                total += 1
                only_else += bool(set(allv) - set(main))
        print(f"  {name}: 값이 나온 상품 {total}개 중 {only_else / max(total, 1):.1%}는 대표 사진 외 사진에서 추가로 나옴")

    # 빈칸 상품 직접 확인용 (메인 조합 기준 + 사진별 근거)
    ev_txt = idx.groupby("parent_asin").apply(
        lambda g: " | ".join(f"[{o}] {(p or {}).get('evidence', '')}" for o, p in zip(g["order"], g["parsed"])))
    m = attach(ms, main_agg)
    m["evidence_by_image"] = m["parent_asin"].map(ev_txt)
    out = os.path.join(DATA, f"missing_review_{args.source}{SUFFIX}.csv")
    m[["parent_asin", "segment", "title", "pred_form", "pred_skin", "pred_benefit",
       "pred_active", "evidence_by_image"]].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n빈칸 상품 직접 확인용: {out}  (사진은 images_all/상품ID_번호.jpg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--model_path", default="./Qwen3-VL-2B-Instruct")
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--hires_only", action="store_true", help="고해상도 사진만 추출")
    ap.add_argument("--n_per_seg", type=int, default=500, help="Face/Body 각각 처리할 상품 수")
    ap.add_argument("--n_missing", type=int, default=100, help="빈칸 상품 수 (Face/Body 반반)")
    ap.add_argument("--grounded", action="store_true", help="평가 시 읽은 문구에 근거가 있는 라벨만 사용")
    ap.add_argument("--source", choices=["qwen", "ocr", "ocr2"], default="qwen",
                    help="평가할 결과: Qwen VLM / OCR+규칙 v1 / OCR+규칙 v2")
    args = ap.parse_args()
    if args.grounded:
        SUFFIX = "_g"
    if args.download:
        run_download()
    if args.extract:
        run_extract(args)
    if args.eval:
        run_eval(args)
