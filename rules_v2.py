"""
OCR 규칙 v2: 오답 30개 검수에서 찾은 패턴을 반영

1) 부정 표현 제거: "oil-free", "soap free", "no mineral oil", "does not contain a sunscreen", "free of ..."
2) 사용법·주의 구역 제외: Directions / How to use / Warning / Caution 이후 줄,
   apply·rinse·massage 등으로 시작하는 줄은 피부타입·효능·제형에서 제외
3) 전성분 구역 분리: "Ingredients:" 이후 줄은 성분에만 사용 (제형이 'Coconut Oil'로 잡히는 문제)
4) 단어 경계·모호어 정리: massaging ≠ aging, 'clean'(=순수)·'combo'·'gentle'·'pores' 제거
5) 피부타입은 'skin' 문맥이 있을 때만 ("dry skin", "skin type: normal, oily ...")
6) 제형은 상품명에서 먼저 찾고, 없으면 OCR 큰 글씨([BIG])에서만. clay를 mask보다 우선

출력 형식은 extract_qwen.parse / label_by_rules와 같음 (+ active_hero: 큰 글씨로 강조된 성분)
"""
import re

from extract_qwen import LONG_ACTIVES
from prepare_eval_sample import ACTIVE_RX, FORM_RX

_c = lambda rules: [(lab, re.compile(rx)) for lab, rx in rules]

# 제형: clay_scrub을 mask_sheet보다 먼저 ("TURMERIC CLAY MASK" → clay_scrub)
FORM2 = _c([r for r in FORM_RX if r[0] == "clay_scrub"] + [r for r in FORM_RX if r[0] != "clay_scrub"])

SKIN2 = _c([
    ("all", r"\b(all|any|every)\s+(skin|types?)"),
    ("dry", r"\bdry\b"),
    ("oily", r"\boily\b"),
    ("combination", r"\bcombination\b"),
    ("normal", r"\bnormal\b"),
    ("sensitive", r"\bsensitive\b"),
    ("acne_prone", r"\b(acne|blemish)[- ]?prone\b|\bacne\b"),
    ("mature", r"\b(mature|aging|ageing)\b"),
])

BENEFIT2 = _c([
    ("hydrating", r"\b(moistur\w*|hydrat\w*)"),
    ("anti_aging", r"\b(anti[- ]?ag\w*|aging|ageing|firm(ing|s|er)?|wrinkl\w*|lift(ing|s)?|plump\w*|tighten\w*|rejuven\w*|fine lines?)\b"),
    ("exfoliating", r"\b(exfoliat\w*|peel\w*|resurfac\w*)"),
    ("soothing", r"\b(sooth\w*|calm\w*|anti[- ]?inflam\w*|reduc\w* redness)"),
    ("smoothing", r"\b(smooth(s|ing|er|es)?|soften\w*|texture)\b"),
    ("cleansing", r"\b(cleans\w*|detox\w*|purif\w*|unclog\w*|decongest\w*)"),
    ("nourishing", r"\b(nourish\w*|replenish\w*|repair\w*|revitaliz\w*|restor\w*|barrier)"),
    ("brightening", r"\b(bright\w*|whiten\w*|radian\w*|glow\w*|lighten\w*|dark spots?|evens? (skin )?tone)"),
    ("antioxidant", r"\banti-?oxid\w*"),
    ("sun_protection", r"\b(spf\s*\d+|broad spectrum|uva|uvb|sun ?protect\w*|sunscreen)"),
    ("acne_care", r"\b(acne|blemish\w*|breakouts?|oil control|mattif\w*)"),
])

ACTIVE2 = _c(ACTIVE_RX)

SECTION_INGR = re.compile(r"\bingr[eé]dients?\s*:|^\s*ingr[eé]dients?\s*$")
SECTION_SKIP = re.compile(r"\b(directions?|how to use|suggested use|warnings?|cautions?)\b")
SKIP_START = re.compile(r"^\s*(apply|rinse|wet|massage|use|avoid|discontinue|keep out|for external|store|smooth)\b")


def strip_negation(line):
    """한 줄 안에서 부정 표현 뒤를 지움 (줄 단위라 다른 줄까지 지우지 않음)"""
    line = re.sub(r"\b[a-z]+[- ]free\b", " ", line)   # oil-free, soap free, paraben-free
    line = re.sub(r"\b(no|non|without|free of|free from|does not|doesn't|do not|not)\b[^.;]*", " ", line)
    return line


def split_sections(tagged, negation=True):
    """[BIG] 태그가 붙은 OCR 줄을 효능 구역 / 전성분 구역으로 나눔. 사용법·주의 구역은 버림
    negation=False면 부정 표현을 지우지 않음 (LLM이 문맥으로 판단하도록)"""
    neg = strip_negation if negation else (lambda s: s)
    claim, ingr = [], []
    section = "claim"
    for raw in (tagged or "").split("\n"):
        big = raw.startswith("[BIG] ")
        low = (raw[6:] if big else raw).lower().replace("_", " ")
        if SECTION_INGR.search(low):
            section = "ingr"
        elif SECTION_SKIP.search(low):
            section = "skip"
        elif big:
            section = "claim"          # 큰 글씨 제목이 나오면 새 패널로 보고 효능 구역으로 복귀
        if section == "ingr":
            ingr.append(neg(low))
        elif section == "claim" and not SKIP_START.search(low):
            claim.append((neg(low), big))
    return claim, ingr


def skin_labels(text):
    labels = set()
    for lab, rx in SKIN2:
        for m in rx.finditer(text):
            if lab == "all":
                labels.add(lab)
                break
            after, before = text[m.end():m.end() + 60], text[max(0, m.start() - 90):m.start()]
            if "skin" in after or "skin type" in before:
                labels.add(lab)
                break
    return labels


def label_v2(tagged):
    claim, ingr = split_sections(tagged)
    claim_text = " ".join(l for l, _ in claim)
    big_text = " ".join(l for l, b in claim if b)
    ingr_text = re.sub(r"-\s+(?=[a-z])", "", " ".join(ingr))   # 줄 끝 하이픈 이어붙이기
    active_src = claim_text + " " + ingr_text
    compact = re.sub(r"\s+", "", active_src)

    labels = {
        "skin": skin_labels(claim_text),
        "benefit": {lab for lab, rx in BENEFIT2 if rx.search(claim_text)},
        "active": {lab for lab, rx in ACTIVE2
                   if rx.search(active_src) or (lab in LONG_ACTIVES and rx.search(compact))},
    }
    forms = [lab for lab, rx in FORM2 if rx.search(big_text)]
    form = forms[0] if forms else None

    out = {"form": form, "form_g": form, "evidence": claim_text[:300], "truncated": False,
           "active_hero": sorted({lab for lab, rx in ACTIVE2 if rx.search(big_text)})}
    for c, v in labels.items():
        out[c] = sorted(v)
        out[f"{c}_g"] = sorted(v)
    return out


def title_form(title):
    """상품명에서 제형 (부정 표현 제거 후, clay 우선 순서)"""
    t = re.sub(r"\b[a-z]+[- ]free\b", " ", (title or "").lower())
    for lab, rx in FORM2:
        if rx.search(t):
            return lab
    return None
