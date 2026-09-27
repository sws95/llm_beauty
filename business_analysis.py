"""
화장품 구매 데이터 비즈니스 분석 — "왜 속성 기반 추천이 필요한가"

0) 상품 속성 채우기: 판매자 속성 + 상품명·특징·설명 텍스트에 규칙 v2 적용
   → data_skincare/item_attrs.parquet (추천 실험에서도 사용)
1) 판매 쏠림과 콜드스타트 압력
2) 고객은 같은 속성을 반복해서 사는가 (연속 구매 쌍 vs 인기도 맞춘 무작위 기준선)
3) 리뷰에서 드러나는 불만·만족 요인 (별점별, 리뷰어가 밝힌 피부타입별) — 분석에만 사용, 모델 입력 아님
4) 판매자 입력 품질

사용:
  python business_analysis.py --fill                 # 0) 속성 채우기 (최초 1회, 몇 분)
  python business_analysis.py --market --consistency --seller
  python business_analysis.py --consistency --cross_brand    # 브랜드를 통제한 속성 배수
  python business_analysis.py --brand_attr                   # 브랜드 × 속성 조합 효과(시너지)
  python business_analysis.py --reviews --review_path Beauty_and_Personal_Care.jsonl   # 3) 리뷰 (10~20분)
  python business_analysis.py --all --review_path Beauty_and_Personal_Care.jsonl
결과는 화면 출력 + data_skincare/analysis_report.md
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:          # tqdm이 없으면 진행 바 없이
    def tqdm(x, **k):
        return x

DATA = "./data_skincare"
ATTRS = os.path.join(DATA, "item_attrs.parquet")
REPORT = os.path.join(DATA, "analysis_report.md")
BASE4 = {"dry", "oily", "combination", "normal"}
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def as_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        return [v] if v and v != "other" else []
    return [x for x in list(v) if x]


def canon_skin(s):
    s = set(s)
    if BASE4 <= s or "all" in s:
        s = (s - BASE4) | {"all"}
    return sorted(s)


# ─────────────────────────────── 0) 속성 채우기 ───────────────────────────────
def build_text(r):
    """상품명은 큰 글씨처럼 취급(제형 우선), 특징·설명은 줄 단위로"""
    body = f"{r['features'] or ''}\n{r['description'] or ''}"
    body = re.sub(r"(?<=[.!?])\s+", "\n", body)
    return f"[BIG] {r['title'] or ''}\n{body}"


def run_fill(args):
    from rules_v2 import ACTIVE2, BENEFIT2, LONG_ACTIVES, label_v2, strip_negation, title_form, title_skin
    items = pd.read_parquet(os.path.join(DATA, "items_norm.parquet"))
    inter = pd.read_parquet(os.path.join(DATA, "interactions_all.parquet"), columns=["parent_asin"])
    used = set(inter["parent_asin"])
    items = items[items["parent_asin"].isin(used)].copy()
    out(f"속성 채우기: 구매 기록이 있는 스킨케어 상품 {len(items):,}개")

    rows = []
    for r in tqdm(items.to_dict("records")):
        t = label_v2(build_text(r))
        seller = {"form": as_list(r["form"]), "skin": canon_skin(as_list(r["skin"])),
                  "benefit": as_list(r["benefit"]), "active": as_list(r["active"])}
        text = {"form": as_list(title_form(r["title"]) or t["form"]),
                "skin": canon_skin(set(t["skin"]) | set(title_skin(r["title"]))),
                "benefit": t["benefit"], "active": t["active"]}
        row = {"parent_asin": r["parent_asin"], "segment": r["segment"], "store": r["store"],
               "price": r["price"]}
        # 메인 효능: 판매자가 가장 내세우는 효능은 상품명에 들어감 (설명 속 부가 효과와 구분)
        tl = strip_negation((r["title"] or "").lower())
        row["benefit_main"] = sorted(lab for lab, rx in BENEFIT2 if rx.search(tl))
        # 핵심 성분: 상품명에 나온 성분 (전성분에 조금 들어간 성분과 구분)
        tc = re.sub(r"\s+", "", tl)
        row["active_main"] = sorted(lab for lab, rx in ACTIVE2
                                    if rx.search(tl) or (lab in LONG_ACTIVES and rx.search(tc)))
        for k in ["form", "skin", "benefit", "active"]:
            row[f"{k}_seller"] = seller[k]
            row[f"{k}_text"] = text[k]
            if k == "form":   # 제형은 하나: 판매자 값 우선, 없으면 텍스트
                row[k] = seller[k][:1] or text[k][:1]
            else:             # 복수 속성: 합집합
                merged = set(seller[k]) | set(text[k])
                row[k] = canon_skin(merged) if k == "skin" else sorted(merged)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(ATTRS, index=False)
    out(f"저장: {ATTRS}")
    for k in ["form", "skin", "benefit", "active"]:
        s = df[f"{k}_seller"].map(len).gt(0).mean()
        m = df[k].map(len).gt(0).mean()
        out(f"  {k:8s} 판매자 {s:6.1%} → 판매자+텍스트 {m:6.1%}")
    out(f"  메인 효능(상품명) 채움률 {df['benefit_main'].map(len).gt(0).mean():.1%}")
    out(f"  핵심 성분(상품명) 채움률 {df['active_main'].map(len).gt(0).mean():.1%}")


def load_attrs():
    if not os.path.exists(ATTRS):
        raise SystemExit("먼저 --fill 을 실행하세요")
    return pd.read_parquet(ATTRS)


# ─────────────────────────────── 1) 판매 쏠림 ───────────────────────────────
def run_market(args):
    inter = pd.read_parquet(os.path.join(DATA, "interactions_all.parquet"))
    attrs = load_attrs()[["parent_asin", "segment"]]
    inter = inter.merge(attrs, on="parent_asin", how="left")
    out("\n## 1) 판매 쏠림과 콜드스타트 압력\n")
    for seg in ["전체", "Face", "Body"]:
        d = inter if seg == "전체" else inter[inter["segment"] == seg]
        cnt = d["parent_asin"].value_counts().values
        n = len(cnt)
        top1 = cnt[:max(1, n // 100)].sum() / cnt.sum()
        top10 = cnt[:max(1, n // 10)].sum() / cnt.sum()
        lt5 = (cnt < 5).mean()
        c = np.sort(cnt)
        gini = 1 - 2 * np.sum(np.cumsum(c) / c.sum()) / n + 1 / n
        out(f"- **{seg}**: 상품 {n:,}개, 구매 {cnt.sum():,}건 | 상위 1% 상품이 구매의 {top1:.1%}, "
            f"상위 10%가 {top10:.1%} | 구매 5건 미만 상품 {lt5:.1%} | 지니계수 {gini:.2f}")

    # 최근 1년 구매가 '그 전에 이력이 거의 없던 상품'으로 얼마나 가는가
    t = inter["timestamp"]
    cut = t.max() - 365 * 24 * 3600 * 1000
    before = inter[t < cut]["parent_asin"].value_counts()
    recent = inter[t >= cut].copy()
    prev = recent["parent_asin"].map(before).fillna(0)
    out(f"\n최근 1년 구매 {len(recent):,}건 중")
    for k, lab in [(0, "그 전 구매 0건(신상품)"), (5, "그 전 구매 5건 미만"), (20, "그 전 구매 20건 미만")]:
        share = (prev == 0).mean() if k == 0 else (prev < k).mean()
        out(f"- {lab} 상품으로 간 구매: **{share:.1%}**")
    out("\n→ 구매 이력만으로 학습하는 추천은 이 구매들에 대해 쓸 신호가 거의 없음")


# ─────────────────────────────── 2) 속성 일관성 ───────────────────────────────
def pair_stats(pairs_a, pairs_b, attrs_map, key):
    """쌍 목록에서 속성이 겹치는 비율 (둘 다 속성이 있는 쌍만)"""
    hit = tot = 0
    for a, b in zip(pairs_a, pairs_b):
        # parquet에서 읽으면 목록이 numpy 배열이 되므로 list로 통일
        x = list(attrs_map[key].get(a, [])) if attrs_map[key].get(a) is not None else []
        y = list(attrs_map[key].get(b, [])) if attrs_map[key].get(b) is not None else []
        if key == "skin":   # 'all'은 정보가 없어 제외
            x = [v for v in x if v != "all"]
            y = [v for v in y if v != "all"]
        if not x or not y:
            continue
        tot += 1
        hit += bool(set(x) & set(y))
    return hit / tot if tot else float("nan"), tot


def label_lift(first, second, rand, amap, key, min_pairs=300):
    """라벨별: P(다음 상품에 L | 이전 상품에 L) / P(무작위 상품에 L | 이전 상품에 L)"""
    get = lambda a: set(list(amap[key].get(a, []))) if amap[key].get(a) is not None else set()
    hit, base, n = Counter(), Counter(), Counter()
    for a, b, c in zip(first, second, rand):
        x, y, z = get(a), get(b), get(c)
        if not y or not z:
            continue
        for lab in x:
            n[lab] += 1
            hit[lab] += lab in y
            base[lab] += lab in z
    rows = []
    for lab, cnt in n.most_common():
        if cnt >= min_pairs and base[lab]:
            rows.append((lab, hit[lab] / cnt, base[lab] / cnt, (hit[lab] / cnt) / (base[lab] / cnt), cnt))
    return rows


def consistency_tables(pairs, amap, title, overall_keys, label_keys):
    out(f"\n### {title}\n")
    out("| 세그먼트 | 속성 | 실제 겹침 | 무작위 겹침 | 배수 | 비교 쌍 수 |")
    out("|---|---|---|---|---|---|")
    for seg, (first, second, rand) in pairs.items():
        for key, name in overall_keys:
            if key not in amap:
                continue
            obs, n = pair_stats(first, second, amap, key)
            base, _ = pair_stats(first, rand, amap, key)
            out(f"| {seg} | {name} | {obs:.1%} | {base:.1%} | **{obs / base:.2f}배** | {n:,} |")
    for key, name in label_keys:
        if key not in amap:
            continue
        out(f"\n**라벨별 배수 — {name}** ({title})\n")
        out("| 세그먼트 | 라벨 | 다음에도 있음 | 무작위 | 배수 | 쌍 수 |")
        out("|---|---|---|---|---|---|")
        for seg, (first, second, rand) in pairs.items():
            for lab, h, b, lift, cnt in label_lift(first, second, rand, amap, key):
                out(f"| {seg} | {lab} | {h:.1%} | {b:.1%} | {lift:.2f}배 | {cnt:,} |")


def run_consistency(args):
    rng = np.random.default_rng(42)
    inter = pd.read_parquet(os.path.join(DATA, "interactions_u5.parquet"))
    attrs = load_attrs()
    keys = ["form", "skin", "benefit", "benefit_main", "active", "active_main"]
    amap = {k: dict(zip(attrs["parent_asin"], attrs[k])) for k in keys if k in attrs}
    norm = lambda s: s.strip().lower() if isinstance(s, str) and s.strip() else None
    brand = dict(zip(attrs["parent_asin"], attrs["store"].map(norm)))
    amap["store"] = {a: ([b] if b else []) for a, b in brand.items()}
    gap_ms = args.min_gap_days * 24 * 3600 * 1000

    pairs_all, pairs_cross = {}, {}
    for seg in ["Face", "Body"]:
        d = inter[inter["segment"] == seg].sort_values(["user_id", "timestamp"])
        a, u, t = d["parent_asin"].values, d["user_id"].values, d["timestamp"].values
        keep = (u[1:] == u[:-1]) & ((t[1:] - t[:-1]) >= gap_ms)
        first, second = a[:-1][keep], a[1:][keep]
        rand = rng.choice(a, size=len(second), replace=True)   # 구매 분포대로 = 인기도 반영
        pairs_all[seg] = (first, second, rand)
        # 브랜드가 다른 쌍만: 실제 쌍과 무작위 쌍 모두 '이전 상품과 브랜드가 다른 경우'로 같은 조건을 맞춤
        bf = np.array([brand.get(x) for x in first], dtype=object)
        bs = np.array([brand.get(x) for x in second], dtype=object)
        br = np.array([brand.get(x) for x in rand], dtype=object)
        ok_real = (bf != None) & (bs != None) & (bf != bs)
        ok_rand = (bf != None) & (br != None) & (bf != br)
        # 무작위 쪽은 조건을 만족하는 것만 남기고, 길이를 실제 쪽과 맞춰 짝을 유지
        fr, rr = first[ok_rand], rand[ok_rand]
        pairs_cross[seg] = (first[ok_real], second[ok_real], (fr, rr))

    out("\n## 2) 고객은 같은 속성을 반복해서 사는가\n")
    out("유저의 연속 구매 쌍(같은 세그먼트, 시간순)에서 속성이 하나라도 겹치는 비율을,")
    out("두 번째 상품을 **같은 세그먼트의 구매 분포(인기도)대로 무작위로 뽑았을 때**와 비교")
    out(f"(세트 구매를 빼기 위해 구매 간격 {args.min_gap_days}일 이상인 쌍만 사용)")
    overall = [("form", "제형"), ("skin", "피부타입(all 제외)"), ("benefit", "효능(설명 전체)"),
               ("benefit_main", "**메인 효능(상품명)**"), ("active", "성분(전체)"),
               ("active_main", "**핵심 성분(상품명)**"), ("store", "브랜드")]
    labels = [("benefit", "효능(설명 전체)"), ("benefit_main", "메인 효능(상품명)"),
              ("skin", "피부타입"), ("active", "성분(전체)"), ("active_main", "핵심 성분(상품명)")]
    consistency_tables(pairs_all, amap, "전체 쌍", overall, labels)

    if args.cross_brand:
        # 브랜드가 다른 쌍: 실제 쌍의 첫 상품 기준 겹침 vs 무작위 쌍(브랜드 다른 것만)의 겹침
        out("\n---\n\n## 2-b) 브랜드를 통제하면? — 브랜드가 다른 쌍만\n")
        out("속성 신호가 사실은 '같은 브랜드 재구매'였는지 확인. 실제 쌍도 무작위 쌍도 **이전 상품과 브랜드가 다른 경우만** 사용")
        out("→ 배수가 여전히 1보다 크면 속성 자체의 신호, 1 근처로 떨어지면 브랜드가 속성 신호를 대신하고 있던 것\n")
        cross = {}
        for seg, (f1, s1, (fr, rr)) in pairs_cross.items():
            cross[seg] = (f1, s1, fr, rr)
        out("| 세그먼트 | 속성 | 실제 겹침 | 무작위 겹침 | 배수 | 비교 쌍 수 |")
        out("|---|---|---|---|---|---|")
        for seg, (f1, s1, fr, rr) in cross.items():
            for key, name in overall[:-1]:
                if key not in amap:
                    continue
                obs, n = pair_stats(f1, s1, amap, key)
                base, _ = pair_stats(fr, rr, amap, key)
                out(f"| {seg} | {name} | {obs:.1%} | {base:.1%} | **{obs / base:.2f}배** | {n:,} |")
        for key, name in labels:
            if key not in amap:
                continue
            out(f"\n**라벨별 배수 — {name}** (브랜드가 다른 쌍만)\n")
            out("| 세그먼트 | 라벨 | 다음에도 있음 | 무작위 | 배수 | 쌍 수 |")
            out("|---|---|---|---|---|---|")
            for seg, (f1, s1, fr, rr) in cross.items():
                real = {lab: (h, cnt) for lab, h, _, _, cnt in label_lift(f1, s1, s1, amap, key, min_pairs=200)}
                base = {lab: h for lab, h, _, _, _ in label_lift(fr, rr, rr, amap, key, min_pairs=200)}
                for lab, (h, cnt) in real.items():
                    if lab in base and base[lab] > 0:
                        out(f"| {seg} | {lab} | {h:.1%} | {base[lab]:.1%} | {h / base[lab]:.2f}배 | {cnt:,} |")
    out("\n→ 흔한 라벨(보습 등)은 배수가 낮고, 드문 메인 효능일수록 배수가 높으면 "
        "'부가 효과가 메인 효능의 신호를 가린다'는 뜻")



# ─────────────────────────────── 2-c) 브랜드 × 속성 ───────────────────────────────
def run_brand_attr(args):
    """연속 구매 쌍을 (같은/다른 브랜드) × (같은/다른 속성) 네 칸으로 나눠 실제 vs 무작위 비교.
    시너지 = 배수(같은 브랜드 & 같은 속성) / (배수(같은 브랜드) × 배수(같은 속성))
      > 1 이면 둘이 겹칠 때 따로 있을 때보다 더 강함 → 브랜드 × 속성 조합이 별도 신호"""
    rng = np.random.default_rng(42)
    inter = pd.read_parquet(os.path.join(DATA, "interactions_u5.parquet"))
    attrs = load_attrs()
    norm = lambda s: s.strip().lower() if isinstance(s, str) and s.strip() else None
    brand = dict(zip(attrs["parent_asin"], attrs["store"].map(norm)))
    to_set = lambda v: set(list(v)) if v is not None else set()
    keys = [("active_main", "핵심 성분(상품명)"), ("benefit_main", "메인 효능(상품명)"),
            ("active", "성분(전체)"), ("form", "제형")]
    amap = {k: {a: to_set(v) for a, v in zip(attrs["parent_asin"], attrs[k])} for k, _ in keys if k in attrs}
    gap_ms = args.min_gap_days * 24 * 3600 * 1000

    out("\n## 2-c) 브랜드 × 속성 — 둘이 겹치면 더 강해지는가\n")
    out("연속 구매 쌍(구매 간격 1일 이상)을 브랜드 같음/다름 × 속성 겹침/안 겹침 네 칸으로 나눠, "
        "두 번째 상품을 구매 분포대로 무작위로 뽑았을 때와 비교. 두 상품 모두 브랜드와 해당 속성이 있는 쌍만 사용\n")
    out("시너지 = 배수(같은 브랜드·같은 속성) ÷ (배수(같은 브랜드) × 배수(같은 속성)). "
        "1보다 크면 브랜드와 속성이 겹칠 때 따로보다 더 강한 신호\n")

    for seg in ["Face", "Body"]:
        d = inter[inter["segment"] == seg].sort_values(["user_id", "timestamp"])
        a, u, t = d["parent_asin"].values, d["user_id"].values, d["timestamp"].values
        keep = (u[1:] == u[:-1]) & ((t[1:] - t[:-1]) >= gap_ms)
        first, second = a[:-1][keep], a[1:][keep]
        rand = rng.choice(a, size=len(second), replace=True)
        for key, name in keys:
            if key not in amap:
                continue
            m = amap[key]

            def cells(x_list, y_list):
                c = Counter()
                n = 0
                for x, y in zip(x_list, y_list):
                    bx, by = brand.get(x), brand.get(y)
                    ax, ay = m.get(x, set()), m.get(y, set())
                    if not bx or not by or not ax or not ay:
                        continue
                    n += 1
                    c[(bx == by, bool(ax & ay))] += 1
                return c, n

            real, n_real = cells(first, second)
            base, n_base = cells(first, rand)
            if n_real < 500 or n_base < 500:
                continue
            pr = {k: real[k] / n_real for k in [(True, True), (True, False), (False, True), (False, False)]}
            pb = {k: base[k] / n_base for k in pr}
            lift = {k: (pr[k] / pb[k] if pb[k] else float("nan")) for k in pr}
            # 주변 배수: 같은 브랜드, 같은 속성 각각
            lb = (pr[(True, True)] + pr[(True, False)]) / max(1e-12, pb[(True, True)] + pb[(True, False)])
            la = (pr[(True, True)] + pr[(False, True)]) / max(1e-12, pb[(True, True)] + pb[(False, True)])
            syn = lift[(True, True)] / (lb * la) if lb and la else float("nan")
            # 조건부: 같은 브랜드일 때 / 다른 브랜드일 때 속성이 겹칠 확률
            p_attr_sb = pr[(True, True)] / max(1e-12, pr[(True, True)] + pr[(True, False)])
            p_attr_db = pr[(False, True)] / max(1e-12, pr[(False, True)] + pr[(False, False)])
            out(f"**{seg} · {name}** (쌍 {n_real:,}개)\n")
            out("| 칸 | 실제 | 무작위 | 배수 |")
            out("|---|---|---|---|")
            labels = {(True, True): "① 같은 브랜드 · 같은 속성", (True, False): "② 같은 브랜드 · 다른 속성",
                      (False, True): "③ 다른 브랜드 · 같은 속성", (False, False): "④ 둘 다 다름"}
            for k, lab in labels.items():
                out(f"| {lab} | {pr[k]:.1%} | {pb[k]:.2%} | {lift[k]:.2f}배 |")
            out(f"\n- 같은 브랜드 배수 {lb:.2f}배 × 같은 속성 배수 {la:.2f}배 = {lb * la:.2f}배 → "
                f"①의 실제 배수 {lift[(True, True)]:.2f}배, **시너지 {syn:.2f}**")
            out(f"- 같은 브랜드를 다시 샀을 때 속성까지 겹칠 확률 {p_attr_sb:.1%} / "
                f"다른 브랜드로 옮겼을 때 속성이 겹칠 확률 {p_attr_db:.1%}\n")


# ─────────────────────────────── 3) 리뷰 요인 ───────────────────────────────
COMPLAINT = {
    "자극(irritation)": r"\b(irritat\w*|burn(ed|ing|s)?|sting\w*|rash\w*|itch\w*|redness|red bumps|allergic)\b",
    "트러블(breakout)": r"\b(break ?outs?|broke (me )?out|breaking out|pimples?|zits?|clogg?ed pores?|acne)\b",
    "번들거림(greasy)": r"\b(greasy|oily|sticky|heavy)\b",
    "건조(drying)": r"\b(dr(ied|ying) (me|my skin) out|drying|too dry|tight(ness)?|flak\w*)\b",
    "향(fragrance)": r"\b(smell\w*|scent\w*|fragran\w*|perfum\w*|odou?r)\b",
    "효과 없음(no effect)": r"\b(didn'?t (work|help|do)|no (difference|results?|change)|did nothing|waste of money)\b",
}
SELF_SKIN = {
    "민감성": r"\bmy (very |super |extremely )?sensitive skin\b|\bi have (very |super )?sensitive skin\b",
    "지성": r"\bmy (very )?oily skin\b|\bi have (very )?oily skin\b",
    "건성": r"\bmy (very )?dry skin\b|\bi have (very )?dry skin\b",
    "복합성": r"\bmy combination skin\b|\bi have combination skin\b",
    "여드름성": r"\bmy acne[- ]prone skin\b|\bi have acne[- ]prone skin\b",
}


def run_reviews(args):
    if not args.review_path:
        raise SystemExit("--review_path 로 Beauty_and_Personal_Care.jsonl 경로를 주세요")
    items = set(load_attrs()["parent_asin"])
    comp = {k: re.compile(v) for k, v in COMPLAINT.items()}
    selfr = {k: re.compile(v) for k, v in SELF_SKIN.items()}

    by_rating = defaultdict(Counter)            # 별점 구간 → 불만 키워드 수
    n_rating = Counter()
    by_self = defaultdict(Counter)              # (피부타입, 별점 구간) → 불만 키워드 수
    n_self = Counter()
    with open(args.review_path, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="리뷰 읽는 중")):
            if args.max_reviews and i >= args.max_reviews:
                break
            r = json.loads(line)
            if not r.get("verified_purchase") or r.get("parent_asin") not in items:
                continue
            rating = r.get("rating") or 0
            bucket = "low(1-2)" if rating <= 2 else ("high(4-5)" if rating >= 4 else "mid(3)")
            text = f"{r.get('title') or ''} {r.get('text') or ''}".lower()
            hits = [k for k, rx in comp.items() if rx.search(text)]
            n_rating[bucket] += 1
            by_rating[bucket].update(hits)
            for s, rx in selfr.items():
                if rx.search(text):
                    n_self[(s, bucket)] += 1
                    by_self[(s, bucket)].update(hits)

    out("\n## 3) 리뷰에서 드러나는 불만 요인 (분석 전용, 모델 입력 아님)\n")
    out(f"스킨케어 인증구매 리뷰: 낮은 별점 {n_rating['low(1-2)']:,}건 / 높은 별점 {n_rating['high(4-5)']:,}건\n")
    out("| 요인 | 낮은 별점 리뷰 언급률 | 높은 별점 리뷰 언급률 | 배수 |")
    out("|---|---|---|---|")
    for k in COMPLAINT:
        lo = by_rating["low(1-2)"][k] / max(1, n_rating["low(1-2)"])
        hi = by_rating["high(4-5)"][k] / max(1, n_rating["high(4-5)"])
        out(f"| {k} | {lo:.1%} | {hi:.1%} | {lo / hi if hi else float('nan'):.1f}배 |")

    out("\n**리뷰어가 스스로 밝힌 피부타입별, 낮은 별점 리뷰의 불만 요인**\n")
    out("| 피부타입 | 낮은 별점 리뷰 수 | " + " | ".join(COMPLAINT) + " |")
    out("|---|---|" + "---|" * len(COMPLAINT))
    for s in SELF_SKIN:
        n = n_self[(s, "low(1-2)")]
        if n < 30:
            continue
        cells = " | ".join(f"{by_self[(s, 'low(1-2)')][k] / n:.0%}" for k in COMPLAINT)
        out(f"| {s} | {n:,} | {cells} |")
    out("\n→ 피부타입마다 '피해야 할 속성'이 다름 (예: 민감성은 자극, 지성은 번들거림·트러블)")


# ─────────────────────────────── 4) 판매자 입력 품질 ───────────────────────────────
def run_seller(args):
    attrs = load_attrs()
    inter = pd.read_parquet(os.path.join(DATA, "interactions_all.parquet"), columns=["parent_asin"])
    pop = inter["parent_asin"].value_counts()
    attrs["purchases"] = attrs["parent_asin"].map(pop).fillna(0)

    out("\n## 4) 판매자 입력 품질\n")
    out("| 속성 | 판매자 채움률 | 판매자+텍스트 채움률 |")
    out("|---|---|---|")
    for k, name in [("form", "제형"), ("skin", "피부타입"), ("benefit", "효능"), ("active", "성분")]:
        s = attrs[f"{k}_seller"].map(len).gt(0).mean()
        m = attrs[k].map(len).gt(0).mean()
        out(f"| {name} | {s:.1%} | {m:.1%} |")

    sk = attrs["skin_seller"].map(len).gt(0)
    only_all = attrs.loc[sk, "skin_seller"].map(lambda v: list(v) == ["all"]).mean()
    out(f"\n- 판매자가 피부타입을 채운 상품 중 **'all'만 적은 비율: {only_all:.1%}** (구체적 정보 없음)")

    # 인기도 구간별 채움률 (큰 브랜드·인기 상품일수록 성실히 채우는지)
    attrs["pop_bin"] = pd.qcut(attrs["purchases"].rank(method="first"), 5,
                               labels=["하위20%", "20-40%", "40-60%", "60-80%", "상위20%"])
    out("\n**구매 수 구간별 판매자 채움률** (상관관계일 뿐 인과 아님: 큰 브랜드일수록 성실히 채우는 경향이 섞여 있음)\n")
    out("| 구매 수 구간 | 피부타입 | 효능 | 성분 |")
    out("|---|---|---|---|")
    for b, g in attrs.groupby("pop_bin", observed=True):
        f = lambda k: g[f"{k}_seller"].map(len).gt(0).mean()
        out(f"| {b} | {f('skin'):.1%} | {f('benefit'):.1%} | {f('active'):.1%} |")

    # 판매자 값과 텍스트 규칙이 둘 다 있을 때 일치율 (판매자 값의 일관성 점검용)
    both = attrs[attrs["form_seller"].map(len).gt(0) & attrs["form_text"].map(len).gt(0)]
    agree = (both["form_seller"].map(lambda v: v[0]) == both["form_text"].map(lambda v: v[0])).mean()
    out(f"\n- 제형: 판매자 값과 상품명·설명 기반 값이 둘 다 있는 {len(both):,}개 중 일치 {agree:.1%} "
        f"(불일치 중 일부는 판매자 오입력 — 예: 'Foaming Wash'를 oil로 입력)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--consistency", action="store_true")
    ap.add_argument("--reviews", action="store_true")
    ap.add_argument("--seller", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--review_path", default=None)
    ap.add_argument("--max_reviews", type=int, default=0, help="테스트용: 앞에서 N줄만 읽기")
    ap.add_argument("--min_gap_days", type=float, default=1, help="연속 구매 쌍의 최소 간격(일), 세트 구매 제외용")
    ap.add_argument("--cross_brand", action="store_true", help="브랜드가 다른 쌍만으로 속성 배수를 다시 계산")
    ap.add_argument("--brand_attr", action="store_true", help="브랜드 × 속성 네 칸 분석과 시너지")
    args = ap.parse_args()

    out("# 화장품 구매 데이터 분석 리포트\n")
    if args.fill or (args.all and not os.path.exists(ATTRS)):
        run_fill(args)
    if args.market or args.all:
        run_market(args)
    if args.consistency or args.all:
        run_consistency(args)
    if args.brand_attr:
        run_brand_attr(args)
    if args.seller or args.all:
        run_seller(args)
    if args.reviews or (args.all and args.review_path):
        run_reviews(args)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\n리포트 저장: {REPORT}")


if __name__ == "__main__":
    main()