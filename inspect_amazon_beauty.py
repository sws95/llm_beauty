"""
Amazon Reviews 2023 뷰티 데이터 1차 점검 (jsonl 스트리밍, 대용량 대응)

확인하는 것
[메타] 필드별 채움률(description/features/images/details/categories), 카테고리 분포,
       details 키 빈도 (→ LLM 추출의 '정답'으로 쓸 수 있는 키 찾기: Skin Type, Item Form 등)
[리뷰] 인증구매 비율, 유저당 리뷰 수 분포, 필터(인증구매 + 유저 k개 이상) 후 남는 규모

사용법
  python inspect_amazon_beauty.py --meta meta_Beauty_and_Personal_Care.jsonl \
      --review Beauty_and_Personal_Care.jsonl --category "Skin Care"
  (--category 생략 시 전체, --max_meta / --max_review 로 앞부분만 샘플링 가능)
"""
import argparse
import json
from collections import Counter


def stream(path, limit=None):
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def nonempty(v):
    return v not in (None, "", [], {}, "None")


def get_details(m):
    """details가 dict일 수도, JSON 문자열일 수도 있음 (배포 경로에 따라 형식이 다름)"""
    d = m.get("details")
    if isinstance(d, str):
        try:
            d = json.loads(d)
        except Exception:
            d = {}
    return d if isinstance(d, dict) else {}


def count_images(m):
    """images가 리스트[dict]일 수도, dict[리스트] 형식일 수도 있음"""
    imgs = m.get("images")
    if isinstance(imgs, list):
        return len(imgs)
    if isinstance(imgs, dict):
        return len(imgs.get("variant") or imgs.get("large") or [])
    return 0


def inspect_meta(path, category, limit):
    n, fill = 0, Counter()
    cat_top, cat_2nd, details_keys, n_images = Counter(), Counter(), Counter(), Counter()
    keep_items = set()
    examples = []

    for m in stream(path, limit):
        cats = m.get("categories") or []
        if category and category not in cats:
            continue
        n += 1
        keep_items.add(m.get("parent_asin"))
        for k in ["title", "description", "features", "images", "details", "categories", "price", "store"]:
            if nonempty(m.get(k)):
                fill[k] += 1
        if cats:
            cat_top[cats[0]] += 1
            if len(cats) > 1:
                cat_2nd[" > ".join(cats[:3])] += 1
        details_keys.update(get_details(m).keys())
        n_images[min(count_images(m), 10)] += 1
        if len(examples) < 2 and get_details(m) and nonempty(m.get("description")):
            examples.append(m)

    print(f"\n=== 메타 ({category or '전체'}) : 상품 {n:,}개 ===")
    print("\n[필드 채움률]")
    for k in ["title", "description", "features", "images", "details", "categories", "price", "store"]:
        print(f"  {k:12s} {fill[k] / max(n, 1):6.1%}")
    print("\n[상품당 이미지 수 분포 (10=10장 이상)]")
    for k in sorted(n_images):
        print(f"  {k:2d}장: {n_images[k]:,}")
    print("\n[카테고리 상위 3단계 Top 20]")
    for c, v in cat_2nd.most_common(20):
        print(f"  {v:7,}  {c}")
    print("\n[details 키 빈도 Top 40]  ← 정답 후보 키 찾기")
    for k, v in details_keys.most_common(40):
        print(f"  {v / max(n, 1):6.1%}  {k}")
    print("\n[예시 상품 2개]")
    for m in examples:
        print("  title:", m.get("title"))
        print("  features:", str(m.get("features"))[:300])
        print("  description:", str(m.get("description"))[:300])
        print("  details:", m.get("details"))
        print()
    return keep_items


def inspect_reviews(path, keep_items, limit, k_list=(3, 5)):
    total, verified = 0, 0
    per_user_all, per_user_ver = Counter(), Counter()
    ts_min, ts_max = None, None

    for r in stream(path, limit):
        if keep_items is not None and r.get("parent_asin") not in keep_items:
            continue
        total += 1
        u = r.get("user_id")
        per_user_all[u] += 1
        if r.get("verified_purchase"):
            verified += 1
            per_user_ver[u] += 1
        t = r.get("timestamp")
        if t:
            ts_min = t if ts_min is None else min(ts_min, t)
            ts_max = t if ts_max is None else max(ts_max, t)

    print(f"\n=== 리뷰 : {total:,}건 ===")
    print(f"인증구매 비율: {verified / max(total, 1):.1%}")
    if ts_min:
        import datetime as dt
        f = lambda x: dt.datetime.utcfromtimestamp(x / 1000).date()
        print(f"기간: {f(ts_min)} ~ {f(ts_max)}")
    print(f"유저 수(전체): {len(per_user_all):,}, 유저 수(인증구매): {len(per_user_ver):,}")

    dist = Counter(min(c, 10) for c in per_user_ver.values())
    print("\n[인증구매 기준 유저당 리뷰 수 (10=10개 이상)]")
    for k in sorted(dist):
        print(f"  {k:2d}개: {dist[k]:,}명")
    for k in k_list:
        users = sum(1 for c in per_user_ver.values() if c >= k)
        inter = sum(c for c in per_user_ver.values() if c >= k)
        print(f"인증구매 + 유저 {k}개 이상 → 유저 {users:,}명, 상호작용 {inter:,}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--review", default=None)
    ap.add_argument("--category", default=None, help='예: "Skin Care" (categories 리스트 안의 값)')
    ap.add_argument("--max_meta", type=int, default=None)
    ap.add_argument("--max_review", type=int, default=None)
    args = ap.parse_args()

    keep = inspect_meta(args.meta, args.category, args.max_meta)
    if args.review:
        inspect_reviews(args.review, keep if args.category else None, args.max_review)


if __name__ == "__main__":
    main()
