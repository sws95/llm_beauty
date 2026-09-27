"""
화장품 추천 실험 — 자동 추출 속성이 콜드스타트 상품 추천을 개선하는가

모델: 피처 BPR (LightFM과 같은 구조)
  상품 벡터 = ID 벡터(학습 기록 있는 상품만) + Σ 속성 필드별 평균 임베딩
  유저 벡터 = ID 벡터
  점수 = <유저, 상품> + 상품 편향(ID 편향 + 속성 편향)
  학습 기록이 없는 상품(평가 기간 신상품)은 ID 부분이 0 → 속성만으로 점수를 받음

분할: 시간 기준. 마지막 test_frac 기간을 평가로 사용 (학습 기록이 있는 유저만 평가)
      학습 기간의 마지막 val_frac을 검증으로 떼서 최적 에폭을 찾고(조기 종료), 그 에폭 수로 전체 학습 데이터를 다시 학습
평가: ① 전체 순위: Recall@K, NDCG@K + 정답 상품의 '학습 기간 구매 수' 구간별 Recall
      ② 신상품 전용 순위: 학습 기간 판매 5건 미만 상품끼리만 순위를 매긴 Recall@K / NDCG@K
과적합 방지: 유저·상품 벡터 L2 정규화
ID 드롭아웃(+drop): 학습 시 id_dropout 확률로 상품 ID 부분을 꺼서, 속성만으로도 점수를 매기도록 학습 (DropoutNet 아이디어)

비교 (--variants):
  pop        인기순 (학습 기간 구매 수)
  id         ID만
  brand      ID + 브랜드 + 세그먼트
  seller     ID + 브랜드 + 세그먼트 + 판매자가 입력한 속성
  filled     ID + 브랜드 + 세그먼트 + 판매자∪자동 추출 속성
             (효능은 메인(상품명)과 설명 속 효능을 별도 필드로, 피부타입 'all' 제외, 흔한 라벨은 IDF로 가중치 낮춤)
  attr_only  ID 없이 브랜드 + 자동 추출 속성만 (콜드스타트 상한 참고)
  이름 뒤에 +drop 을 붙이면 ID 드롭아웃 적용 (예: seller+drop, filled+drop), +drop0.3 처럼 비율 지정 가능
  --active_mode: 성분을 all(전체) / main(상품명의 핵심 성분만) / split(핵심과 나머지를 별도 필드) 중 선택
  --seeds 42 43 44: 시드마다 반복해 평균 ± 표준편차 보고

사용:
  python rec_experiment.py                                  # 기본 비교 전체
  python rec_experiment.py --variants pop id filled+drop --max_epochs 10  # 빠른 확인
  python rec_experiment.py --ablation                       # filled에서 속성을 하나씩 뺀 버전 추가
결과: 화면 + data_skincare/rec_results.md
"""
import argparse
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

DATA = "./data_skincare"
BASE4 = {"dry", "oily", "combination", "normal"}
POP_BINS = [(0, 0, "신상품(0건)"), (1, 4, "1~4건"), (5, 19, "5~19건"), (20, 10**9, "20건 이상")]


# ─────────────────────────────── 데이터 ───────────────────────────────
def as_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        return [v] if v and v != "other" else []
    return [x for x in list(v) if x]


def load_split(test_frac):
    inter = pd.read_parquet(os.path.join(DATA, "interactions_u5.parquet"))
    cut = inter["timestamp"].quantile(1 - test_frac)
    train = inter[inter["timestamp"] < cut]
    test = inter[inter["timestamp"] >= cut]
    test = test[test["user_id"].isin(set(train["user_id"]))]      # 학습 기록 있는 유저만 평가
    return inter, train, test, cut


def build_index(inter, train, test):
    users = sorted(set(train["user_id"]))
    items = sorted(set(train["parent_asin"]) | set(test["parent_asin"]))
    uid = {u: k for k, u in enumerate(users)}
    iid = {a: k for k, a in enumerate(items)}
    train_cnt = train["parent_asin"].value_counts()
    item_train = np.array([train_cnt.get(a, 0) for a in items], dtype=np.int64)
    return uid, iid, items, item_train


def item_tokens(attrs_row, variant, drop=None, active_mode="all"):
    """상품 하나의 (필드, 토큰) 목록. variant에 따라 쓰는 속성이 달라짐"""
    drop = drop or set()
    fields = {}
    if variant in ("brand", "seller", "filled", "attr_only"):
        store = attrs_row.get("store")
        if isinstance(store, str) and store and store != "None":
            fields["brand"] = [f"brand={store.strip().lower()}"]
        if isinstance(attrs_row.get("segment"), str):
            fields["seg"] = [f"seg={attrs_row['segment']}"]
    if variant == "seller":
        src = {k: as_list(attrs_row.get(f"{k}_seller")) for k in ["form", "skin", "benefit", "active"]}
        sk = set(src["skin"])
        if BASE4 <= sk or "all" in sk:          # 기본 네 가지 나열 = all → 정보 없음으로 제외
            sk = sk - BASE4 - {"all"}
        src["skin"] = sorted(sk)
        for k, v in src.items():
            if v:
                fields[k] = [f"{k}={x}" for x in v]
    if variant in ("filled", "attr_only"):
        skin = [s for s in as_list(attrs_row.get("skin")) if s != "all"]     # 'all'은 정보 없음
        main = as_list(attrs_row.get("benefit_main"))
        desc = [b for b in as_list(attrs_row.get("benefit")) if b not in main]
        src = {"form": as_list(attrs_row.get("form")), "skin": skin, "benefit_main": main,
               "benefit_desc": desc}
        act_all = as_list(attrs_row.get("active"))
        act_main = as_list(attrs_row.get("active_main"))
        if active_mode == "all":
            src["active"] = act_all
        elif active_mode == "main":
            src["active_main"] = act_main
        else:   # split
            src["active_main"] = act_main
            src["active_desc"] = [x for x in act_all if x not in act_main]
        for k, v in src.items():
            if v and k not in drop:
                prefix = k.split("_")[0] if k.startswith(("benefit", "active")) else k
                fields[k] = [f"{k}:{prefix}={x}" for x in v]
    if "brand" in drop:
        fields.pop("brand", None)
    return fields


def build_features(items, variant, drop=None, idf=True, active_mode="all"):
    """상품별 토큰 인덱스와 가중치 (필드마다 평균, 흔한 토큰은 IDF로 가중치 낮춤)"""
    attrs = pd.read_parquet(os.path.join(DATA, "item_attrs.parquet")).set_index("parent_asin")
    per_item = []
    df_cnt = Counter()
    for a in items:
        row = attrs.loc[a].to_dict() if a in attrs.index else {}
        f = item_tokens(row, variant, drop, active_mode)
        per_item.append(f)
        for toks in f.values():
            df_cnt.update(set(toks))
    # 브랜드는 상품 2개 이상인 것만 (1개짜리는 ID와 같아 일반화에 도움 안 됨)
    vocab = {t: k + 1 for k, t in enumerate(sorted(t for t, c in df_cnt.items()
                                                   if not (t.startswith("brand=") and c < 2)))}
    n = len(items)
    # 필드 안에서 가장 드문 토큰의 IDF를 1로 맞춤 → 흔한 라벨(보습 등)만 작아지고 드문 라벨은 유지
    field_max = Counter()
    for f in per_item:
        for field, toks in f.items():
            for t in toks:
                field_max[field] = max(field_max[field], math.log(n / df_cnt[t]))
    max_len = max(1, max((sum(len(v) for v in f.values()) for f in per_item), default=1))
    idx = np.zeros((n, max_len), dtype=np.int64)
    w = np.zeros((n, max_len), dtype=np.float32)
    for r, f in enumerate(per_item):
        c = 0
        for field, toks in f.items():
            toks = [t for t in toks if t in vocab]
            if not toks:
                continue
            for t in toks:
                wt = 1.0 / len(toks)
                if idf and field not in ("brand", "seg") and field_max[field] > 0:
                    wt *= max(0.05, math.log(n / df_cnt[t]) / field_max[field])   # 흔할수록 작게 (0.05~1)
                idx[r, c], w[r, c] = vocab[t], wt
                c += 1
    return idx, w, len(vocab) + 1


# ─────────────────────────────── 평가 ───────────────────────────────
def evaluate(score_fn, train_u, train_i, test_u, test_i, item_train, n_users, n_items, k, device,
             batch=1024, pool=None, max_users=None, seed=0):
    """score_fn(user_idx_tensor) → [B, n_items] 점수. 학습 기간에 산 상품은 제외
    pool: bool 배열이면 그 상품들끼리만 순위를 매김 (신상품 전용 평가)"""
    import torch
    from collections import defaultdict
    seen = defaultdict(list)
    for u, i in zip(train_u, train_i):
        seen[u].append(i)
    truth = defaultdict(set)
    for u, i in zip(test_u, test_i):
        if pool is None or pool[i]:
            truth[u].add(i)
    pool_t = None if pool is None else torch.tensor(~pool, device=device)
    eval_users = np.array(sorted(truth))
    if max_users and len(eval_users) > max_users:
        eval_users = np.random.default_rng(seed).choice(eval_users, max_users, replace=False)
    recalls, ndcgs = [], []
    bin_hit, bin_tot = Counter(), Counter()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for s in range(0, len(eval_users), batch):
        ub = eval_users[s:s + batch]
        with torch.no_grad():
            sc = score_fn(torch.tensor(ub, device=device))
            rows, cols = [], []
            for r, u in enumerate(ub):
                rows += [r] * len(seen[u])
                cols += seen[u]
            if rows:
                sc[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = -1e9
            if pool_t is not None:
                sc[:, pool_t] = -1e9
            top = torch.topk(sc, k, dim=1).indices.cpu().numpy()
        for r, u in enumerate(ub):
            t = truth[u]
            hits = [x in t for x in top[r]]
            recalls.append(sum(hits) / len(t))
            ideal = disc[:min(len(t), k)].sum()
            ndcgs.append(float(np.dot(hits, disc)) / ideal)
            topset = set(top[r])
            for i in t:
                c = item_train[i]
                lab = next(l for lo, hi, l in POP_BINS if lo <= c <= hi)
                bin_tot[lab] += 1
                bin_hit[lab] += i in topset
    res = {"recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs)), "users": len(eval_users)}
    for _, _, lab in POP_BINS:
        res[f"R_{lab}"] = bin_hit[lab] / bin_tot[lab] if bin_tot[lab] else float("nan")
        res[f"N_{lab}"] = bin_tot[lab]
    return res


# ─────────────────────────────── 모델 ───────────────────────────────
def make_model(n_users, n_items, n_tokens, dim, use_id, feat_idx, feat_w, has_train, device,
               id_dropout=0.0):
    import torch
    import torch.nn as nn

    class FeatBPR(nn.Module):
        def __init__(self):
            super().__init__()
            self.U = nn.Embedding(n_users, dim)
            self.I = nn.Embedding(n_items, dim)
            self.Ib = nn.Embedding(n_items, 1)
            self.T = nn.Embedding(n_tokens, dim, padding_idx=0)
            self.Tb = nn.Embedding(n_tokens, 1, padding_idx=0)
            for e in [self.U, self.I, self.T]:
                nn.init.normal_(e.weight, std=0.05)
            nn.init.zeros_(self.Ib.weight)
            nn.init.zeros_(self.Tb.weight)
            with torch.no_grad():
                self.T.weight[0].zero_()
                self.Tb.weight[0].zero_()
            self.fi = torch.tensor(feat_idx, device=device)
            self.fw = torch.tensor(feat_w, device=device).unsqueeze(-1)
            self.mask = torch.tensor(has_train, device=device, dtype=torch.float32).unsqueeze(-1)

        def item_vec(self, i):
            v = (self.T(self.fi[i]) * self.fw[i]).sum(1)
            b = (self.Tb(self.fi[i]) * self.fw[i]).sum(1)
            if use_id:   # 학습 기록 없는 상품은 ID 부분 0
                m = self.mask[i]
                if self.training and id_dropout > 0:   # ID 드롭아웃: 속성만으로 점수를 매기도록
                    m = m * (torch.rand_like(m) >= id_dropout).float()
                v = v + self.I(i) * m
                b = b + self.Ib(i) * m
            return v, b

        def score(self, u, i):
            v, b = self.item_vec(i)
            return (self.U(u) * v).sum(-1) + b.squeeze(-1)

        def all_items(self):
            return self.item_vec(torch.arange(n_items, device=device))

    return FeatBPR().to(device)


def fit(args, name, n_users, n_items, feat, use_id, id_dropout, tr_u, tr_i, epochs, device,
        val=None):
    """epochs만큼 학습. val=(u, i)가 있으면 에폭마다 검증 Recall을 재서 최적 에폭과 점수 반환"""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fidx, fw, n_tok = feat
    has_train = np.zeros(n_items, dtype=bool)
    has_train[np.unique(tr_i)] = True
    item_cnt = np.bincount(tr_i, minlength=n_items)
    model = make_model(n_users, n_items, n_tok, args.dim, use_id, fidx, fw, has_train, device, id_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_items = np.unique(tr_i)
    U, I = torch.tensor(tr_u, device=device), torch.tensor(tr_i, device=device)
    n = len(tr_u)
    best, best_ep, bad = -1.0, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, args.batch):
            b = perm[s:s + args.batch]
            u, i = U[b], I[b]
            j = torch.tensor(rng.choice(train_items, len(b)), device=device)
            vi, bi = model.item_vec(i)
            vj, bj = model.item_vec(j)
            pu = model.U(u)
            diff = (pu * vi).sum(-1) + bi.squeeze(-1) - (pu * vj).sum(-1) - bj.squeeze(-1)
            reg = args.reg * (pu.pow(2).sum(-1) + vi.pow(2).sum(-1) + vj.pow(2).sum(-1)).mean()
            loss = -F.logsigmoid(diff).mean() + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        msg = f"  [{name}] epoch {ep + 1:2d} loss {total / n:.4f}"
        if val is not None:
            model.eval()
            with torch.no_grad():
                V, B = model.all_items()
            r = evaluate(lambda ub: model.U(ub) @ V.T + B.T, tr_u, tr_i, val[0], val[1], item_cnt,
                         n_users, n_items, args.k, device, max_users=args.val_users, seed=args.seed)["recall"]
            msg += f" | 검증 Recall@{args.k} {r:.4f}"
            if r > best:
                best, best_ep, bad = r, ep + 1, 0
            else:
                bad += 1
            print(msg)
            if bad >= args.patience:
                break
        else:
            print(msg)
    return model, best_ep, best


def train_eval(name, args, data, feat=None, use_id=True, id_dropout=0.0):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (train_u, train_i, test_u, test_i, item_train, n_users, n_items, val) = data
    if feat is None:
        feat = (np.zeros((n_items, 1), dtype=np.int64), np.zeros((n_items, 1), dtype=np.float32), 1)
    # 1) 검증으로 최적 에폭 찾기
    _, best_ep, best = fit(args, name, n_users, n_items, feat, use_id, id_dropout,
                           val["fit_u"], val["fit_i"], args.max_epochs, device, val=(val["u"], val["i"]))
    print(f"  → 최적 에폭 {best_ep} (검증 Recall {best:.4f}), 전체 학습 데이터로 다시 학습")
    # 2) 전체 학습 데이터로 최적 에폭만큼 다시 학습
    model, _, _ = fit(args, name, n_users, n_items, feat, use_id, id_dropout,
                      train_u, train_i, best_ep, device)
    model.eval()
    with torch.no_grad():
        V, B = model.all_items()
    score_fn = lambda ub: model.U(ub) @ V.T + B.T
    res = evaluate(score_fn, train_u, train_i, test_u, test_i, item_train, n_users, n_items, args.k, device)
    cold = evaluate(score_fn, train_u, train_i, test_u, test_i, item_train, n_users, n_items, args.k, device,
                    pool=item_train < args.cold_max)
    res.update({"cold_recall": cold["recall"], "cold_ndcg": cold["ndcg"], "cold_users": cold["users"],
                "best_epoch": best_ep})
    return res


def pop_eval(args, data):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (train_u, train_i, test_u, test_i, item_train, n_users, n_items, _) = data
    pop = torch.tensor(item_train, device=device, dtype=torch.float32)
    score_fn = lambda ub: pop.unsqueeze(0).expand(len(ub), -1).clone()
    res = evaluate(score_fn, train_u, train_i, test_u, test_i, item_train, n_users, n_items, args.k, device)
    cold = evaluate(score_fn, train_u, train_i, test_u, test_i, item_train, n_users, n_items, args.k, device,
                    pool=item_train < args.cold_max)
    res.update({"cold_recall": cold["recall"], "cold_ndcg": cold["ndcg"], "cold_users": cold["users"],
                "best_epoch": "-"})
    return res


# ─────────────────────────────── 실행 ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["pop", "id", "brand", "seller", "filled", "seller+drop", "filled+drop", "attr_only"])
    ap.add_argument("--ablation", action="store_true", help="ablation_base에서 속성을 하나씩 뺀 버전도 실행")
    ap.add_argument("--ablation_base", default="filled+drop")
    ap.add_argument("--id_dropout", type=float, default=0.5, help="+drop 모델의 기본 ID 드롭아웃 확률")
    ap.add_argument("--val_frac", type=float, default=0.1, help="학습 기간 중 검증으로 쓸 마지막 비율")
    ap.add_argument("--val_users", type=int, default=8000, help="검증 평가 유저 수 (속도)")
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--cold_max", type=int, default=5, help="신상품 전용 평가: 학습 기간 판매 수 < cold_max")
    ap.add_argument("--no_idf", action="store_true", help="흔한 라벨 가중치 낮추기 끄기")
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--active_mode", choices=["all", "main", "split"], default="split")
    args = ap.parse_args()

    inter, train, test, cut = load_split(args.test_frac)
    uid, iid, items, item_train = build_index(inter, train, test)
    tr = train[train["user_id"].isin(uid)]
    # 검증 분할: 학습 기간의 마지막 val_frac → 최적 에폭 찾기용
    vcut = tr["timestamp"].quantile(1 - args.val_frac)
    fit_df, val_df = tr[tr["timestamp"] < vcut], tr[tr["timestamp"] >= vcut]
    val_df = val_df[val_df["user_id"].isin(set(fit_df["user_id"]))]
    val = {"fit_u": fit_df["user_id"].map(uid).to_numpy(np.int64),
           "fit_i": fit_df["parent_asin"].map(iid).to_numpy(np.int64),
           "u": val_df["user_id"].map(uid).to_numpy(np.int64),
           "i": val_df["parent_asin"].map(iid).to_numpy(np.int64)}
    data = (tr["user_id"].map(uid).to_numpy(np.int64), tr["parent_asin"].map(iid).to_numpy(np.int64),
            test["user_id"].map(uid).to_numpy(np.int64), test["parent_asin"].map(iid).to_numpy(np.int64),
            item_train, len(uid), len(iid), val)
    tp = test["parent_asin"].map(iid).to_numpy()
    new_share = (item_train[tp] == 0).mean()
    print(f"학습 {len(tr):,}건 / 평가 {len(test):,}건 (기준 시점 {pd.to_datetime(cut, unit='ms').date()}), "
          f"유저 {len(uid):,}, 상품 {len(iid):,}")
    print(f"평가 구매 중 학습 기간에 한 번도 안 팔린 상품: {new_share:.1%}")
    print(f"검증 분할: 학습 {len(val['fit_u']):,}건 / 검증 {len(val['u']):,}건")

    runs = [(v, v, None) for v in args.variants]
    if args.ablation:
        act = {"all": ["active"], "main": ["active_main"], "split": ["active_main", "active_desc"]}[args.active_mode]
        for d in ["brand", "form", "skin", "benefit_main", "benefit_desc"] + act:
            runs.append((f"{args.ablation_base} - {d}", args.ablation_base, {d}))

    keys = ["recall", "ndcg", "cold_recall", "cold_ndcg"] + [f"R_{l}" for _, _, l in POP_BINS]
    results = []
    for name, variant, drop in runs:
        base, drop_p = variant, 0.0
        if "+drop" in variant:
            base, tail = variant.split("+drop")
            drop_p = float(tail) if tail else args.id_dropout
        feat = None
        if base not in ("pop", "id"):
            feat = build_features(items, base, drop, idf=not args.no_idf, active_mode=args.active_mode)
        per_seed = []
        for seed in (args.seeds if base != "pop" else args.seeds[:1]):
            args.seed = seed
            print(f"\n=== {name} (seed {seed}) ===")
            if base == "pop":
                r = pop_eval(args, data)
            else:
                r = train_eval(name, args, data, feat, use_id=(base != "attr_only"), id_dropout=drop_p)
            per_seed.append(r)
            print({k: round(r[k], 4) for k in keys})
        agg = {"model": name, "n_seeds": len(per_seed)}
        for k in keys:
            vals = np.array([r[k] for r in per_seed], dtype=float)
            agg[k], agg[k + "_sd"] = float(np.nanmean(vals)), float(np.nanstd(vals))
        eps = [r["best_epoch"] for r in per_seed if r["best_epoch"] != "-"]
        agg["best_epoch"] = f"{np.mean(eps):.0f}" if eps else "-"
        agg["cold_users"] = per_seed[0]["cold_users"]
        for _, _, l in POP_BINS:
            agg[f"N_{l}"] = per_seed[0][f"N_{l}"]
        results.append(agg)

    # 결과 표
    k = args.k
    bins = [l for _, _, l in POP_BINS]
    lines = [f"# 추천 실험 결과 (Recall@{k}, NDCG@{k})\n",
             f"- 시간 분할: 마지막 {args.test_frac:.0%} 기간 평가, 학습 {len(tr):,}건 / 평가 {len(test):,}건",
             f"- 평가 구매 중 학습 기간에 한 번도 안 팔린 상품: {new_share:.1%}",
             f"- 하이퍼파라미터 공통: dim {args.dim}, lr {args.lr}, L2 {args.reg}, 에폭은 검증 조기 종료, "
             f"ID 드롭아웃 기본 {args.id_dropout}(+drop), IDF {'off' if args.no_idf else 'on'}, "
             f"성분 {args.active_mode}",
             f"- 시드 {len(args.seeds)}개({', '.join(map(str, args.seeds))}) 평균 ± 표준편차",
             f"- 신상품 전용: 학습 기간 판매 {args.cold_max}건 미만 상품끼리만 순위\n",
             f"| 모델 | 에폭 | Recall@{k} | NDCG@{k} | **신상품 전용 Recall@{k}** | 신상품 전용 NDCG@{k} | " +
             " | ".join(f"Recall {b}" for b in bins) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(bins)]
    f = lambda r, k: f"{r[k]:.4f}" + (f" ± {r[k + '_sd']:.4f}" if r["n_seeds"] > 1 else "")
    for r in results:
        lines.append(f"| {r['model']} | {r['best_epoch']} | {f(r, 'recall')} | {f(r, 'ndcg')} | "
                     f"**{f(r, 'cold_recall')}** | {f(r, 'cold_ndcg')} | " +
                     " | ".join(f(r, f"R_{b}") for b in bins) + " |")
    lines.append("\n구간별 정답 수: " + ", ".join(f"{b} {results[0][f'N_{b}']:,}" for b in bins) +
                 f" | 신상품 전용 평가 유저 {results[0]['cold_users']:,}명")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(DATA, "rec_results.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n저장: {os.path.join(DATA, 'rec_results.md')}")


if __name__ == "__main__":
    main()