"""
텍스트 LLM으로 속성 판단 (사진 없이, 상품당 1회)  ※ 전역 파이썬 환경(Qwen이 돌던 곳)에서 실행

역할 분담
- LLM: 제형·피부타입·효능 (문맥 해석이 필요한 속성)
- 규칙 v2: 성분 (이름이 그대로 적혀 있어 규칙이 강함. LLM은 전성분을 통째로 나열하는 문제가 있었음)

입력: 상품명 + 판매자 특징 + 설명 + OCR 문구(사용법·주의 구역 제외, [BIG]=큰 글씨)
검증(라벨 단위): 근거 문구가 입력에 실제로 있고, 그 문구 안에 라벨 키워드가 있어야 인정
복구: 출력이 잘려 JSON이 깨져도 완성된 필드는 꺼내서 사용

사용법
  python llm_text.py --run  --model_path "C:\\...\\Qwen3-VL-2B-Instruct"   (끊겨도 이어서)
  python llm_text.py --eval
  python llm_text.py --review --n 10
  (--structured 를 붙이면 outlines로 enum 스키마를 강제한 모드. 결과는 llm_text_enum.jsonl에 따로 저장)
"""
import argparse
import json
import os
import random
import re

import pandas as pd

import extract_per_image as epi
from extract_per_image import DATA, IDX, OCR_PRED, attach, fill_rate, load_samples, print_scores, select_products
from extract_qwen import BENEFITS, FORMS, SKINS, _to_text, load_model
from prepare_eval_sample import BENEFIT_C, SKIN_C, norm_form, norm_multi
from rules_v2 import BENEFIT2, FORM2, SKIN2, label_v2, split_sections, title_form

OUT = os.path.join(DATA, "llm_text.jsonl")
FIELDS = [("form", "form"), ("skin", "skin_type"), ("benefit", "benefits")]
LABEL_RX = {"form": dict(FORM2), "skin": dict(SKIN2), "benefit": dict(BENEFIT2)}


def clip(s, n):
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def load_ocr():
    ocr = {}
    with open(OCR_PRED, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            ocr[tuple(r["key"])] = r.get("tagged", "")
    return ocr


def hires_index(asins=None):
    idx = pd.read_parquet(IDX)
    idx = idx[idx["ok"] & idx["is_hires"]]
    if asins is not None:
        idx = idx[idx["parent_asin"].isin(asins)]
    return idx.sort_values(["parent_asin", "order"])


# ── 입력 만들기 ────────────────────────────────────────────────
def product_inputs(n_per_seg, n_missing):
    ev, ms = load_samples()
    items = pd.concat([ev, ms]).drop_duplicates("parent_asin").set_index("parent_asin")
    ocr = load_ocr()
    inputs = {}
    for a, g in hires_index(select_products(n_per_seg, n_missing)).groupby("parent_asin"):
        claim = []
        for k in g["order"]:
            c, _ = split_sections(ocr.get((a, int(k)), ""), negation=False)
            claim += [("[BIG] " if big else "") + l for l, big in c]
        seen = set()
        claim = [l for l in claim if not (l in seen or seen.add(l))]   # 사진 간 중복 줄 제거
        row = items.loc[a]
        inputs[a] = {"title": clip(row["title"], 200), "features": clip(row["features"], 800),
                     "description": clip(row["description"], 800), "package": clip(" | ".join(claim), 1500)}
    return inputs


def build_prompt(x):
    return f"""You are labeling a skincare product for a recommender system.
Use ONLY the product information below. Do not guess from general knowledge.

Rules:
- form: the physical form of the product (one value or null).
- skin_type: ONLY if the text says which skin types it is for. Use "all" ONLY if it says "all skin types".
  If the text does not mention skin types, use [].
- benefits: effects the product CLAIMS. Ignore directions for use and warnings.
- "oil-free", "does not contain sunscreen" mean the product does NOT have that.
- evidence: for each field, a list of exact phrases COPIED from the product information.
  Do not write label names as evidence. Keep the answer short.

Allowed values:
- form: {FORMS}
- skin_type: {SKINS}
- benefits: {BENEFITS}

Example (a different product; do not copy its values):
Product information: "Lip Balm Stick SPF 15 | soothes and repairs chapped lips | for sensitive skin"
Answer: {{"form": "stick", "skin_type": ["sensitive"], "benefits": ["soothing", "nourishing", "sun_protection"],
"evidence": {{"form": ["Lip Balm Stick"], "skin_type": ["for sensitive skin"],
"benefits": ["soothes and repairs chapped lips", "Stick SPF 15"]}}}}

Product information:
Title: {x['title']}
Seller bullet points: {x['features']}
Description: {x['description']}
Text on package photos ([BIG] = large print): {x['package']}

Answer with one JSON object only, in the same format as the example."""


def source_text(x):
    return " ".join(x.values())


# ── 구조화 출력(enum) 모드 ─────────────────────────────────────
OUT_ENUM = os.path.join(DATA, "llm_text_enum.jsonl")


def build_schema():
    """허용값은 Literal(enum)로, 목록 길이는 max_length로 강제. 근거(evidence)를 먼저 쓰게 맨 앞에 둠"""
    from typing import Annotated, List, Literal, Optional
    from pydantic import BaseModel, Field, StringConstraints

    Quote = Annotated[str, StringConstraints(max_length=100)]
    Form = Literal[tuple(FORMS)]
    Skin = Literal[tuple(SKINS)]
    Benefit = Literal[tuple(BENEFITS)]

    class Evidence(BaseModel):
        form: List[Quote] = Field(max_length=2)
        skin_type: List[Quote] = Field(max_length=3)
        benefits: List[Quote] = Field(max_length=4)

    class Attributes(BaseModel):
        evidence: Evidence
        form: Optional[Form]
        skin_type: List[Skin] = Field(max_length=4)
        benefits: List[Benefit] = Field(max_length=5)

    return Attributes


def build_prompt_enum(x):
    return f"""You are labeling a skincare product for a recommender system.
Use ONLY the product information below. Do not guess from general knowledge.

First write "evidence": exact phrases COPIED from the product information for each field.
Then choose the labels based ONLY on that evidence.
- form: the physical form of the product, or null.
- skin_type: ONLY if the text says which skin types it is for. "all" ONLY if it says "all skin types".
- benefits: effects the product CLAIMS. Ignore directions and warnings.
- "oil-free", "does not contain sunscreen" mean the product does NOT have that.

Product information:
Title: {x['title']}
Seller bullet points: {x['features']}
Description: {x['description']}
Text on package photos ([BIG] = large print): {x['package']}"""


# ── 파싱 (잘린 출력 복구) + 라벨 단위 근거 검증 ───────────────────
def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def supported(quote, src_norm, src_tokens):
    q = _norm(quote)
    if not q:
        return False
    if q in src_norm:
        return True
    toks = q.split()
    return sum(t in src_tokens for t in toks) / len(toks) >= 0.8


def _quotes(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else [str(q) for q in v]


def _field(seg, key):
    """seg에서 key의 값을 꺼냄. 목록이 중간에 잘렸으면 완성된 문자열 항목까지만"""
    mm = re.search(rf'"{key}"\s*:\s*(\[[^\]]*\]|"[^"]*"|null)', seg)
    if mm:
        try:
            return json.loads(mm.group(1))
        except Exception:
            pass
    mm = re.search(rf'"{key}"\s*:\s*\[([^\]]*)$', seg, re.S)
    if mm:
        return re.findall(r'"([^"]*)"', mm.group(1))
    return None


def loose_json(raw):
    """정상 JSON이면 그대로, 잘려서 깨졌으면 완성된 필드·항목만 꺼냄"""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if m:
        try:
            return json.loads(m.group(0)), False
        except Exception:
            pass
    head = re.split(r'"evidence"\s*:', raw or "", maxsplit=1)
    d = {key: v for _, key in FIELDS if (v := _field(head[0], key)) is not None}
    if not d:
        return None, True
    evd = {}
    if len(head) > 1:
        evd = {key: v for _, key in FIELDS if (v := _field(head[1], key)) is not None}
    d["evidence"] = evd
    return d, True


def parse_llm(raw, src):
    d, truncated = loose_json(raw)
    if d is None:
        return None
    evd = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    form = norm_form(_to_text(d.get("form")))
    labels = {
        "form": None if form == "other" else form,
        "skin": norm_multi(_to_text(d.get("skin_type")), SKIN_C) or [],
        "benefit": norm_multi(_to_text(d.get("benefits")), BENEFIT_C) or [],
    }
    src_norm = _norm(src)
    src_tokens = set(src_norm.split())

    out = {"evidence": {}, "truncated": truncated}
    for k, key in FIELDS:
        good = [q for q in _quotes(evd.get(key)) if supported(q, src_norm, src_tokens)]
        qtext = " ".join(good).lower().replace("_", " ")
        out["evidence"][k] = " / ".join(_quotes(evd.get(key)))[:200]
        rx = LABEL_RX[k]
        # _q: LLM의 라벨은 버리고, LLM이 골라준 (실제 있는) 근거 문구에 규칙을 적용해 라벨 생성
        if k == "form":
            f = labels["form"]
            out["form"] = f
            out["form_g"] = f if f and rx.get(f) and rx[f].search(qtext) else None
            out["form_q"] = next((lab for lab, r in FORM2 if r.search(qtext)), None)
        else:
            out[k] = sorted(labels[k])
            out[f"{k}_g"] = sorted(lab for lab in labels[k] if rx.get(lab) and rx[lab].search(qtext))
            if k == "skin":
                from rules_v2 import skin_labels
                out["skin_q"] = sorted(skin_labels(qtext))
            else:
                out[f"{k}_q"] = sorted(lab for lab, r in BENEFIT2 if r.search(qtext))
    return out


# ── 실행 ───────────────────────────────────────────────────────
def generate_text(model, proc, prompt, max_new_tokens=250):
    import torch
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                    tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def run(args):
    import time
    from tqdm import tqdm
    out_path = OUT_ENUM if args.structured else OUT
    inputs = product_inputs(args.n_per_seg, args.n_missing)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["asin"] for l in f}
    todo = [(a, x) for a, x in inputs.items() if a not in done]
    print(f"[{'enum 구조화' if args.structured else '자유 JSON'}] 상품 {len(inputs)}개, "
          f"완료 {len(done)}개, 남음 {len(todo)}개")

    model, proc = load_model(args.model_path)
    # 윈도우에는 Triton이 없어 torch.compile 시도가 매번 실패하고 경고만 쏟아냄 → 컴파일 끄기
    import torch._dynamo
    torch._dynamo.config.disable = True
    if args.structured:
        import outlines
        omodel = outlines.from_transformers(model, proc.tokenizer)
        schema = build_schema()
        print("스키마 준비 완료 (처음 한 번은 토큰 제약 계산에 시간이 좀 걸려요)")

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for a, x in tqdm(todo):
            try:
                if args.structured:
                    chat = proc.tokenizer.apply_chat_template(
                        [{"role": "user", "content": build_prompt_enum(x)}],
                        tokenize=False, add_generation_prompt=True)
                    raw = omodel(chat, schema, max_new_tokens=300, do_sample=False)
                else:
                    raw = generate_text(model, proc, build_prompt(x))
            except Exception as e:
                raw = f"ERROR: {e}"
            f.write(json.dumps({"asin": a, "raw": raw, "src": source_text(x)}, ensure_ascii=False) + "\n")
            f.flush()
    if todo:
        print(f"평균 {(time.time() - t0) / len(todo):.1f}초/상품")


# ── 평가 ───────────────────────────────────────────────────────
def load_preds(args):
    preds = {}
    with open(OUT_ENUM if args.structured else OUT, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            preds[r["asin"]] = parse_llm(r["raw"], r["src"])
    return preds


def v2_agg(asins, titles):
    ocr = load_ocr()
    agg = {}
    for a, g in hires_index(asins).groupby("parent_asin"):
        agg[a] = epi.aggregate([label_v2(ocr.get((a, int(k)), "")) for k in g["order"]])
        agg[a]["form"] = title_form(titles.get(a)) or agg[a]["form"]
    return agg


def llm_agg(preds, suf, rule_agg):
    """제형·피부타입·효능은 LLM, 성분은 규칙 v2"""
    return {a: {"form": p[f"form{suf}"], "skin": p[f"skin{suf}"], "benefit": p[f"benefit{suf}"],
                "active": rule_agg.get(a, {}).get("active", [])}
            for a, p in preds.items() if p}


def run_eval(args):
    preds = load_preds(args)
    print(f"[{'enum 구조화' if args.structured else '자유 JSON'}]")
    ok = {a for a, p in preds.items() if p}
    n_trunc = sum(bool(p and p.get("truncated")) for p in preds.values())
    print(f"LLM 결과 {len(preds)}개, 파싱 실패 {len(preds) - len(ok)}개, 잘린 출력 복구 {n_trunc}개")
    ev, ms = load_samples()
    titles = {**dict(zip(ev.parent_asin, ev.title)), **dict(zip(ms.parent_asin, ms.title))}
    v2 = v2_agg(ok, titles)

    for name, agg in [("규칙 v2", v2),
                      ("LLM 원출력 (+성분은 규칙)", llm_agg(preds, "", v2)),
                      ("LLM 근거 검증 (+성분은 규칙)", llm_agg(preds, "_g", v2)),
                      ("LLM 근거 → 규칙 라벨 (+성분은 규칙)", llm_agg(preds, "_q", v2))]:
        e = attach(ev, agg)
        print(f"\n===== {name}  (평가 상품 {len(e)}개, 같은 상품) =====")
        for seg in ["Face", "Body"]:
            print(f"  <{seg}>")
            print_scores(e[e.segment == seg])
        m = attach(ms, agg)
        if len(m):
            print(f"  <빈칸 상품 {len(m)}개> 채움률  피부타입 {fill_rate(m['pred_skin']):5.1%}  "
                  f"효능 {fill_rate(m['pred_benefit']):5.1%}  제형 {m['pred_form'].notna().astype(float).mean():5.1%}  "
                  f"성분 {fill_rate(m['pred_active']):5.1%}")


REVIEW_METHODS = {"v2": ("규칙v2", None), "raw": ("LLM원출력", ""), "g": ("근거검증", "_g"), "q": ("근거→규칙", "_q")}


def review_path(args):
    name = REVIEW_METHODS[args.review_method][0]
    return os.path.join(DATA, f"error_review_{name}{'_enum' if args.structured else ''}.csv")


def method_eval_frame(args):
    """선택한 방식의 예측을 붙인 평가용 DataFrame"""
    preds = load_preds(args)
    ev, _ = load_samples()
    titles = dict(zip(ev.parent_asin, ev.title))
    ok = {a for a, p in preds.items() if p}
    v2 = v2_agg(ok, titles)
    suf = REVIEW_METHODS[args.review_method][1]
    agg = v2 if suf is None else llm_agg(preds, suf, v2)
    return attach(ev, agg), preds


def run_review(args):
    """선택한 방식의 오답(판매자 정답과 다른 예측)을 속성별로 n개씩 뽑아 A/B/C 판정용 CSV로 저장
    A = 판매자 정답이 불완전 (실제로는 맞음)  B = 예측이 틀림  C = 애매"""
    e, preds = method_eval_frame(args)
    random.seed(42)
    rows = []
    for attr in ["form", "skin", "benefit"]:
        cands = []
        for r in e.itertuples():
            if attr == "form":
                if pd.notna(r.gt_form) and pd.notna(r.pred_form) and r.pred_form != r.gt_form:
                    cands.append((r, r.pred_form, r.gt_form))
            else:
                gt = list(getattr(r, f"gt_{attr}"))
                for lab in (sorted(set(getattr(r, f"pred_{attr}")) - set(gt)) if gt else []):
                    cands.append((r, lab, ", ".join(gt)))
        print(f"{NAMES[attr]} 오답 후보 {len(cands)}개 → {min(args.n, len(cands))}개 추출")
        for r, lab, gt in random.sample(cands, min(args.n, len(cands))):
            rows.append({"속성": NAMES[attr], "상품ID": r.parent_asin, "세그먼트": r.segment,
                         "상품명": str(r.title)[:80], "예측": lab, "판매자 정답": gt,
                         "LLM 근거": preds[r.parent_asin]["evidence"].get(attr, "") if preds.get(r.parent_asin) else "",
                         "사진": f"images_all/{r.parent_asin}_*.jpg", "판정(A/B/C)": "", "메모": ""})
    out = review_path(args)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out}  (방식: {REVIEW_METHODS[args.review_method][0]})")
    print("판정: A=판매자 정답이 불완전(실제로는 맞음)  B=예측이 틀림  C=애매")


def run_review_score(args):
    """A/B/C 판정으로 보정 정밀도 계산: 측정 정밀도 + (1 - 측정 정밀도) × A 비율"""
    path = review_path(args)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    df["판정"] = df["판정(A/B/C)"].astype(str).str.strip().str.upper()
    e, _ = method_eval_frame(args)

    print(f"방식: {REVIEW_METHODS[args.review_method][0]}\n")
    print(f"{'속성':6s} {'측정 정밀도':>10s} {'A':>3s} {'B':>3s} {'C':>3s} {'A비율':>7s} {'보정 정밀도':>10s}")
    for attr in ["form", "skin", "benefit"]:
        if attr == "form":
            f = e[e["gt_form"].notna() & e["pred_form"].notna()]
            measured = (f["pred_form"] == f["gt_form"]).mean() if len(f) else float("nan")
        else:
            tp = fp = 0
            for p, t in zip(e[f"pred_{attr}"], e[f"gt_{attr}"]):
                if not len(t):
                    continue
                p, t = set(p), set(t)
                tp, fp = tp + len(p & t), fp + len(p - t)
            measured = tp / (tp + fp) if tp + fp else float("nan")
        j = df[(df["속성"] == NAMES[attr]) & df["판정"].isin(["A", "B", "C"])]
        a, b, c = (j["판정"] == "A").sum(), (j["판정"] == "B").sum(), (j["판정"] == "C").sum()
        share = a / len(j) if len(j) else float("nan")
        corrected = measured + (1 - measured) * share if len(j) else float("nan")
        print(f"{NAMES[attr]:6s} {measured:10.1%} {a:3d} {b:3d} {c:3d} {share:7.1%} {corrected:10.1%}")
    notes = df[df["판정"] == "B"]["메모"].dropna().astype(str)
    if len(notes):
        print("\nB(틀림) 원인 메모 빈도:")
        print(notes.str.strip().value_counts().head(10).to_string())


NAMES = {"form": "제형", "skin": "피부타입", "benefit": "효능"}
METHOD_SUFFIX = [("규칙v2", None), ("LLM원출력", ""), ("근거검증", "_g"), ("근거→규칙", "_q")]


def audit_path(args):
    return os.path.join(DATA, f"audit{'_enum' if args.structured else ''}.csv")


def run_audit(args):
    """상품 n개의 예측 라벨(네 방식 합집합)을 한 줄씩 내보냄 → 사람이 O/X 판정"""
    from extract_qwen import gt_labels
    preds = load_preds(args)
    ev, ms = load_samples()
    allp = pd.concat([ev.assign(구분="정답있음"), ms.assign(구분="빈칸")]).drop_duplicates("parent_asin")
    info = allp.set_index("parent_asin")
    titles = dict(zip(allp.parent_asin, allp.title))

    ok = sorted(a for a, p in preds.items() if p and a in info.index)
    random.seed(42)
    pick = random.sample(ok, min(args.n_audit, len(ok)))
    v2 = v2_agg(set(pick), titles)
    methods = {m: (v2 if suf is None else llm_agg(preds, suf, v2)) for m, suf in METHOD_SUFFIX}

    rows = []
    for a in pick:
        r = info.loc[a]
        for attr in ["form", "skin", "benefit"]:
            if attr == "form":
                gt = [r["form"]] if isinstance(r["form"], str) and r["form"] != "other" else []
            else:
                gt = gt_labels(r[attr])
            got = {}
            for m, agg in methods.items():
                v = agg.get(a, {}).get(attr)
                got[m] = {v} if isinstance(v, str) else set(v or [])
            for lab in sorted(set().union(*got.values())):
                rows.append({"상품ID": a, "구분": r["구분"], "세그먼트": r["segment"],
                             "상품명": str(r["title"])[:80], "속성": NAMES[attr], "라벨": lab,
                             "판매자 정답": ", ".join(gt) or "(없음)",
                             "판매자 정답에 있음": "Y" if lab in gt else "",
                             **{m: ("V" if lab in got[m] else "") for m in methods},
                             "LLM 근거": preds[a]["evidence"].get(attr, ""),
                             "사진": f"images_all/{a}_*.jpg", "판정(O/X)": "", "메모": ""})
    out = audit_path(args)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out}  (상품 {len(pick)}개, 판정할 라벨 {len(rows)}줄)")
    print("판정(O/X): 상품 설명·사진을 보고 이 라벨이 실제로 맞으면 O, 아니면 X. 애매하면 비워두기")


def run_audit_score(args):
    """사람 판정 기준 정밀도 vs 판매자 정답 기준 정밀도"""
    path = audit_path(args)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:   # 엑셀에서 저장하면 cp949로 바뀌는 경우
        df = pd.read_csv(path, encoding="cp949")
    df["판정"] = df["판정(O/X)"].astype(str).str.strip().str.upper()
    judged = df[df["판정"].isin(["O", "X"])]
    print(f"판정된 라벨 {len(judged)} / 전체 {len(df)}줄\n")
    print(f"{'속성':6s} {'방식':10s} {'예측수':>5s} {'사람기준 정밀도':>14s} {'판매자정답기준':>14s}")
    for attr in ["제형", "피부타입", "효능"]:
        for m, _ in METHOD_SUFFIX:
            sub = judged[(judged["속성"] == attr) & (judged[m] == "V")]
            if not len(sub):
                continue
            human = (sub["판정"] == "O").mean()
            s = sub[sub["구분"] == "정답있음"]
            seller = (s["판매자 정답에 있음"] == "Y").mean() if len(s) else float("nan")
            print(f"{attr:6s} {m:10s} {len(sub):5d} {human:14.1%} {seller:14.1%}")
        print()
    blank = judged[judged["구분"] == "빈칸"]
    if len(blank):
        print("빈칸 상품에서 채운 라벨의 사람 기준 정밀도:")
        for m, _ in METHOD_SUFFIX:
            sub = blank[blank[m] == "V"]
            if len(sub):
                print(f"  {m:10s} {len(sub):4d}개 중 {(sub['판정'] == 'O').mean():6.1%} 맞음")


def compare_path(args):
    return os.path.join(DATA, f"compare{'_enum' if args.structured else ''}.csv")


def run_compare(args):
    """상품 × 속성마다 한 줄: 판매자 정답 + 네 방식 예측을 나란히. '사람 정답' 칸은 판매자 정답으로 미리 채움"""
    from extract_qwen import gt_labels
    preds = load_preds(args)
    ev, ms = load_samples()
    allp = pd.concat([ev.assign(구분="정답있음"), ms.assign(구분="빈칸")]).drop_duplicates("parent_asin")
    info = allp.set_index("parent_asin")
    titles = dict(zip(allp.parent_asin, allp.title))
    ok = sorted(a for a, p in preds.items() if p and a in info.index)
    if args.n_audit and args.n_audit < len(ok):
        random.seed(42)
        ok = sorted(random.sample(ok, args.n_audit))
    v2 = v2_agg(set(ok), titles)
    methods = {m: (v2 if suf is None else llm_agg(preds, suf, v2)) for m, suf in METHOD_SUFFIX}

    fmt = lambda v: (v or "") if isinstance(v, str) or v is None else ", ".join(v)
    rows = []
    for a in ok:
        r = info.loc[a]
        for attr in ["form", "skin", "benefit"]:
            if attr == "form":
                gt = [r["form"]] if isinstance(r["form"], str) and r["form"] != "other" else []
            else:
                gt = gt_labels(r[attr])
            row = {"상품ID": a, "구분": r["구분"], "세그먼트": r["segment"], "상품명": str(r["title"])[:80],
                   "속성": NAMES[attr], "판매자 정답": ", ".join(gt)}
            for m, agg in methods.items():
                row[m] = fmt(agg.get(a, {}).get(attr))
            row.update({"LLM 근거": preds[a]["evidence"].get(attr, ""), "사진": f"images_all/{a}_*.jpg",
                        "사람 정답": ", ".join(gt), "확인(Y)": "", "메모": ""})
            rows.append(row)
    out = compare_path(args)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out}  (상품 {len(ok)}개 × 속성 3 = {len(rows)}줄)")
    print("'사람 정답'을 사진·설명 보고 고친 뒤(빠진 건 추가, 틀린 건 삭제, 쉼표로 구분) '확인(Y)'에 Y")
    print(f"허용 값  제형: {', '.join(FORMS)}")
    print(f"         피부타입: {', '.join(SKINS)}")
    print(f"         효능: {', '.join(BENEFITS)}")


def run_compare_score(args):
    """확인(Y)된 줄만: '사람 정답' 기준으로 네 방식의 정밀도·재현율(제형은 정확도)"""
    path = compare_path(args)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949", dtype=str).fillna("")
    df = df[df["확인(Y)"].str.strip().str.upper() == "Y"]
    to_set = lambda x: {t.strip().lower() for t in str(x).split(",") if t.strip()}
    print(f"확인된 줄 {len(df)}개 (상품 {df['상품ID'].nunique()}개)\n")

    for grp_name, g in [("전체", df), ("정답있음", df[df["구분"] == "정답있음"]), ("빈칸", df[df["구분"] == "빈칸"])]:
        if not len(g):
            continue
        print(f"===== {grp_name} =====")
        for attr in ["제형", "피부타입", "효능"]:
            a = g[g["속성"] == attr]
            if not len(a):
                continue
            print(f"  <{attr}>")
            for m, _ in METHOD_SUFFIX + [("판매자 정답", None)]:
                if m == "판매자 정답" and grp_name == "빈칸":
                    continue
                tp = fp = fn = answered = 0
                for pred, truth in zip(a[m], a["사람 정답"]):
                    p, t = to_set(pred), to_set(truth)
                    answered += bool(p)
                    tp, fp, fn = tp + len(p & t), fp + len(p - t), fn + len(t - p)
                prec = tp / (tp + fp) if tp + fp else float("nan")
                rec = tp / (tp + fn) if tp + fn else float("nan")
                print(f"    {m:10s} 답함 {answered / len(a):5.1%}  정밀도 {prec:6.1%}  재현율 {rec:6.1%}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--model_path", default="./Qwen3-VL-2B-Instruct")
    ap.add_argument("--n_per_seg", type=int, default=150)
    ap.add_argument("--n_missing", type=int, default=50)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--structured", action="store_true", help="outlines로 enum 스키마 강제 (결과는 llm_text_enum.jsonl)")
    ap.add_argument("--audit", action="store_true", help="사람 판정용 CSV 만들기")
    ap.add_argument("--audit_score", action="store_true", help="판정 채운 CSV로 사람 기준 정밀도 계산")
    ap.add_argument("--n_audit", type=int, default=30, help="판정할 상품 수")
    ap.add_argument("--review_method", choices=list(REVIEW_METHODS), default="q",
                    help="오답 검수할 방식: v2=규칙v2, raw=LLM원출력, g=근거검증, q=근거→규칙")
    ap.add_argument("--review_score", action="store_true", help="A/B/C 판정으로 보정 정밀도 계산")
    ap.add_argument("--compare", action="store_true", help="상품별로 네 방식 예측을 나란히 붙인 CSV")
    ap.add_argument("--compare_score", action="store_true", help="사람 정답 기준 네 방식 정밀도·재현율")
    args = ap.parse_args()
    if args.run:
        run(args)
    if args.eval:
        run_eval(args)
    if args.review:
        run_review(args)
    if args.review_score:
        run_review_score(args)
    if args.compare:
        run_compare(args)
    if args.compare_score:
        run_compare_score(args)
    if args.audit:
        run_audit(args)
    if args.audit_score:
        run_audit_score(args)
