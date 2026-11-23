#!/usr/bin/env python3
"""
Movie recommender: collaborative filtering vs content-based filtering (MovieLens).

Models compared (all produce a full users x items matrix of predicted ratings):
  Baseline      global mean + damped user/item biases
  Popularity    rank by number of training ratings          (ranking only)
  Random        random scores                               (ranking only)
  UserCF        user-based kNN on baseline residuals        (collaborative)
  ItemCF        item-based kNN on baseline residuals        (collaborative)
  MF            biased matrix factorisation, SGD            (collaborative)
  ContentKNN    item-kNN using TF-IDF genre/tag similarity  (content-based)
  ContentRidge  per-user linear (kernel-ridge) profile      (content-based)
  Hybrid        weighted blend of MF and ContentRidge

Metrics: RMSE / MAE (rating prediction); Precision@k, Recall@k, NDCG@k, catalogue
coverage, mean popularity (top-N); RMSE by item-popularity bucket (cold-start view).

Usage:
  python recsys_compare.py                 # downloads ml-latest-small (needs internet)
  python recsys_compare.py --synthetic     # offline smoke test on generated data
  python recsys_compare.py --data-dir path # folder containing ml-latest-small/
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
SYN_GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Horror", "Musical", "Mystery",
              "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_movielens(data_dir):
    path = os.path.join(data_dir, "ml-latest-small")
    if not os.path.isdir(path):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading {ML_URL} ...")
        with urllib.request.urlopen(ML_URL, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(data_dir)
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(path, "movies.csv"))
    tags = pd.read_csv(os.path.join(path, "tags.csv"))
    return ratings, movies, tags


def make_synthetic(seed=0, n_users=600, n_items=3000):
    """Offline stand-in with MovieLens-like schema: long-tailed popularity, genre
    taste + latent taste. For testing the pipeline only - not for drawing conclusions."""
    rng = np.random.default_rng(seed)
    G, D = len(SYN_GENRES), 8
    ig = np.zeros((n_items, G))
    for i in range(n_items):
        ig[i, rng.choice(G, rng.integers(1, 4), replace=False)] = 1
    il, ib = rng.normal(0, 1, (n_items, D)), rng.normal(0, 0.4, n_items)
    up, ul, ub = rng.normal(0, 0.6, (n_users, G)), rng.normal(0, 0.3, (n_users, D)), rng.normal(0, 0.3, n_users)
    pop = rng.pareto(1.2, n_items) + 1
    p = pop / pop.sum()
    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(3.9, 0.8), 20, 500))
        items = rng.choice(n_items, size=n, replace=False, p=p)
        raw = 3.3 + ub[u] + ib[items] + ig[items] @ up[u] + il[items] @ ul[u] + rng.normal(0, 0.5, n)
        r = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
        rows += [(u + 1, int(i) + 1, float(x)) for i, x in zip(items, r)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    years = rng.integers(1960, 2019, n_items)
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_items + 1),
        "title": [f"Movie {i} ({y})" for i, y in zip(range(1, n_items + 1), years)],
        "genres": ["|".join(SYN_GENRES[g] for g in np.flatnonzero(ig[i])) for i in range(n_items)]})
    tag_rows = []
    for i in range(n_items):
        for d in range(D):
            if abs(il[i, d]) > 1 and rng.random() < 0.7:
                tag_rows.append((1, i + 1, f"theme{d}{'p' if il[i, d] > 0 else 'n'}"))
        for _ in range(rng.integers(0, 3)):
            tag_rows.append((1, i + 1, f"noise{rng.integers(0, 60)}"))
    tags = pd.DataFrame(tag_rows, columns=["userId", "movieId", "tag"])
    return ratings, movies, tags


def prepare(ratings, min_user, min_item):
    ic = ratings.movieId.value_counts()
    ratings = ratings[ratings.movieId.map(ic) >= min_item]
    uc = ratings.userId.value_counts()
    ratings = ratings[ratings.userId.map(uc) >= min_user].copy()
    user_ids, item_ids = np.sort(ratings.userId.unique()), np.sort(ratings.movieId.unique())
    ratings["u"] = np.searchsorted(user_ids, ratings.userId)
    ratings["i"] = np.searchsorted(item_ids, ratings.movieId)
    return ratings[["u", "i", "rating"]].reset_index(drop=True), user_ids, item_ids


def per_user_split(df, test_frac, seed):
    df = df.sample(frac=1.0, random_state=seed)
    pos = df.groupby("u").cumcount()
    n = df.groupby("u")["u"].transform("size")
    is_test = pos < np.ceil(n * test_frac)
    return df[~is_test].copy(), df[is_test].copy()


def to_matrix(df, nu, ni):
    R = np.zeros((nu, ni), dtype=np.float32)
    R[df.u.values, df.i.values] = df.rating.values
    return R, (R > 0).astype(np.float32)


# ----------------------------------------------------------------------------
# Content features (genres + tags + decade -> TF-IDF)
# ----------------------------------------------------------------------------
def build_content(movies, tags, item_ids):
    m = movies.set_index("movieId").loc[item_ids]
    tag_map = (tags.dropna(subset=["tag"]).assign(tag=lambda d: d.tag.astype(str).str.lower().str.strip().str.replace(r"\s+", "_", regex=True))
               .groupby("movieId").tag.apply(list).to_dict())
    docs = []
    for mid, row in m.iterrows():
        toks = [f"g_{g.replace(' ', '_').lower()}" for g in str(row.genres).split("|")]
        y = re.search(r"\((\d{4})\)\s*$", str(row.title))
        if y:
            toks.append(f"dec_{int(y.group(1)) // 10 * 10}")
        toks += [f"t_{t}" for t in tag_map.get(mid, [])]
        docs.append(" ".join(toks))
    X = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, sublinear_tf=True).fit_transform(docs)
    return normalize(X)


# ----------------------------------------------------------------------------
# Models - each returns a dense (users x items) matrix of predicted ratings
# ----------------------------------------------------------------------------
def fit_baselines(R, M, reg_u=15.0, reg_i=10.0):
    mu = R.sum() / M.sum()
    bi = (R - mu * M).sum(0) / (reg_i + M.sum(0))
    bu = (R - mu * M - M * bi[None, :]).sum(1) / (reg_u + M.sum(1))
    full = mu + bu[:, None] + bi[None, :]                                  # uses item signal (collaborative)
    bu0 = (R - mu * M).sum(1) / (reg_u + M.sum(1))
    user_only = np.repeat((mu + bu0)[:, None], R.shape[1], axis=1)         # no item signal (content-safe)
    return full.astype(np.float32), user_only.astype(np.float32)


def topk_rows(S, k):
    S = S.copy()
    np.fill_diagonal(S, 0)
    S[S < 0] = 0
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.zeros_like(S)
    r = np.arange(S.shape[0])[:, None]
    out[r, idx] = S[r, idx]
    return out


def cosine_shrunk(A, Mask, shrink):
    """cosine similarity between columns of A, shrunk by co-rating counts."""
    G = A.T @ A
    nrm = np.sqrt((A ** 2).sum(0)) + 1e-9
    n = Mask.T @ Mask
    return G / np.outer(nrm, nrm) * (n / (n + shrink))


def item_knn(R, M, B, S_item, k, damp):
    """pred(u,i) = B + sum_j S_ij * (r_uj - B_uj) / (sum_j S_ij + damp), top-k neighbours per item."""
    Rc = (R - B) * M
    S = topk_rows(S_item, k)
    return B + (Rc @ S.T) / ((M @ S.T) + damp)


def user_knn(R, M, B, k, shrink, damp):
    Rc = (R - B) * M
    S = topk_rows(cosine_shrunk(Rc.T, M.T, shrink), k)
    return B + (S @ Rc) / ((S @ M) + damp)


def content_ridge(R, M, B, S_c, lam):
    """Per-user linear model on TF-IDF features, solved in the dual with the linear kernel
    K = X X^T:  w_u = X_J^T (K_JJ + lam I)^-1 y,  scores = K[:, J] @ alpha."""
    nu, ni = R.shape
    P = B.copy()
    for u in range(nu):
        J = np.flatnonzero(M[u])
        if len(J) == 0:
            continue
        y = (R[u, J] - B[u, J]).astype(np.float64)
        K = S_c[np.ix_(J, J)].astype(np.float64)
        alpha = np.linalg.solve(K + lam * np.eye(len(J)), y)
        P[u] = B[u] + S_c[:, J] @ alpha
    return P


class BiasedMF:
    def __init__(self, k=32, lr=0.02, reg=0.1, epochs=40, seed=0):
        self.k, self.lr, self.reg, self.epochs, self.seed = k, lr, reg, epochs, seed

    def fit_predict(self, df, nu, ni, verbose=True):
        rng = np.random.default_rng(self.seed)
        u, i, r = df.u.values, df.i.values, df.rating.values.astype(np.float32)
        mu = r.mean()
        bu, bi = np.zeros(nu, np.float32), np.zeros(ni, np.float32)
        P = rng.normal(0, 0.1, (nu, self.k)).astype(np.float32)
        Q = rng.normal(0, 0.1, (ni, self.k)).astype(np.float32)
        lr, reg = self.lr, self.reg
        for ep in range(self.epochs):
            se = 0.0
            for idx in rng.permutation(len(r)):
                a, b = u[idx], i[idx]
                pu, qi = P[a].copy(), Q[b].copy()
                e = r[idx] - (mu + bu[a] + bi[b] + pu @ qi)
                se += e * e
                bu[a] += lr * (e - reg * bu[a])
                bi[b] += lr * (e - reg * bi[b])
                P[a] += lr * (e * qi - reg * pu)
                Q[b] += lr * (e * pu - reg * qi)
            lr *= 0.95
            if verbose and (ep % 10 == 9 or ep == self.epochs - 1):
                print(f"    MF epoch {ep + 1:2d}  train RMSE {np.sqrt(se / len(r)):.4f}")
        return (mu + bu[:, None] + bi[None, :] + P @ Q.T).astype(np.float32)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def rating_metrics(P, test):
    pred = np.clip(P[test.u.values, test.i.values], 0.5, 5.0)
    err = pred - test.rating.values
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def ranking_metrics(P, train_M, test_rel, item_pop, ks=(5, 10)):
    S = P.astype(np.float64).copy()
    S[train_M > 0] = -np.inf                       # never recommend what the user already rated
    kmax = max(ks)
    top = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
    top = np.take_along_axis(top, order, 1)
    hits = np.take_along_axis(test_rel, top, 1)
    n_rel = test_rel.sum(1)
    ok = n_rel > 0                                  # users with at least one relevant test item
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    cum = np.cumsum(disc)
    out = {}
    for k in ks:
        h = hits[ok, :k]
        out[f"P@{k}"] = float((h.sum(1) / k).mean())
        out[f"R@{k}"] = float((h.sum(1) / n_rel[ok]).mean())
        dcg = (h * disc[:k]).sum(1)
        idcg = cum[np.minimum(n_rel[ok], k).astype(int) - 1]
        out[f"NDCG@{k}"] = float((dcg / idcg).mean())
    top10 = top[:, :10]
    out["Coverage@10"] = float(len(np.unique(top10)) / P.shape[1])
    out["AvgLogPop@10"] = float(np.log1p(item_pop[top10]).mean())
    return out


def bucket_rmse(P, test, item_pop):
    bins = [(0, 4, "0-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 10 ** 9, "100+")]
    pred = np.clip(P[test.u.values, test.i.values], 0.5, 5.0)
    err2 = (pred - test.rating.values) ** 2
    cnt = item_pop[test.i.values]
    return {name: (float(np.sqrt(err2[(cnt >= lo) & (cnt <= hi)].mean())) if ((cnt >= lo) & (cnt <= hi)).any() else np.nan)
            for lo, hi, name in bins}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=40, help="neighbours for kNN models")
    ap.add_argument("--min-user", type=int, default=20)
    ap.add_argument("--min-item", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--rel-thresh", type=float, default=4.0, help="rating >= this counts as relevant")
    ap.add_argument("--mf-epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.75, help="hybrid weight on MF")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    if args.synthetic:
        ratings, movies, tags = make_synthetic(args.seed)
        print("*** USING SYNTHETIC DATA - pipeline test only, numbers are not MovieLens results ***")
    else:
        try:
            ratings, movies, tags = load_movielens(args.data_dir)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not load MovieLens ({e}).\nDownload {ML_URL}, unzip into {args.data_dir}/ "
                     f"(so that {args.data_dir}/ml-latest-small/ratings.csv exists) or run with --synthetic.")

    df, user_ids, item_ids = prepare(ratings, args.min_user, args.min_item)
    nu, ni = len(user_ids), len(item_ids)
    train, test = per_user_split(df, args.test_frac, args.seed)
    R, M = to_matrix(train, nu, ni)
    item_pop = M.sum(0)
    test_rel = np.zeros((nu, ni), np.float32)
    sel = test.rating >= args.rel_thresh
    test_rel[test.u[sel].values, test.i[sel].values] = 1
    print(f"users={nu}  items={ni}  train={len(train)}  test={len(test)}  "
          f"density={len(df) / (nu * ni):.3%}  relevant test pairs={int(test_rel.sum())}")

    B_full, B_user = fit_baselines(R, M)
    X = build_content(movies, tags, item_ids)
    S_c = (X @ X.T).toarray().astype(np.float32)
    print(f"content features: {X.shape[1]} TF-IDF terms")

    preds = {}
    print("Fitting models ...")
    preds["Baseline (bias)"] = B_full
    preds["UserCF (kNN)"] = user_knn(R, M, B_full, args.k, shrink=25, damp=0.2)
    S_item = cosine_shrunk((R - B_full) * M, M, shrink=25)
    preds["ItemCF (kNN)"] = item_knn(R, M, B_full, S_item, args.k, damp=0.2)
    print("  MF ...")
    preds["MF (biased SGD)"] = BiasedMF(epochs=args.mf_epochs, seed=args.seed).fit_predict(train, nu, ni)
    preds["ContentKNN"] = item_knn(R, M, B_user, S_c, args.k, damp=0.2)
    print("  ContentRidge ...")
    preds["ContentRidge"] = content_ridge(R, M, B_user, S_c, lam=1.0)
    preds[f"Hybrid ({args.alpha:.2f} MF + {1 - args.alpha:.2f} Content)"] = (
        args.alpha * preds["MF (biased SGD)"] + (1 - args.alpha) * preds["ContentRidge"])

    rng = np.random.default_rng(args.seed)
    rank_only = {"Popularity": np.tile(item_pop, (nu, 1)), "Random": rng.random((nu, ni))}

    rows, cold = [], []
    for name, P in preds.items():
        rmse, mae = rating_metrics(P, test)
        rows.append({"Model": name, "RMSE": rmse, "MAE": mae, **ranking_metrics(P, M, test_rel, item_pop)})
        cold.append({"Model": name, **bucket_rmse(P, test, item_pop)})
    for name, P in rank_only.items():
        rows.append({"Model": name, "RMSE": np.nan, "MAE": np.nan, **ranking_metrics(P, M, test_rel, item_pop)})
    res, cold = pd.DataFrame(rows), pd.DataFrame(cold)

    # hybrid sensitivity (descriptive; the headline hybrid uses the fixed --alpha)
    sens = []
    for a in [0, 0.25, 0.5, 0.75, 1.0]:
        P = a * preds["MF (biased SGD)"] + (1 - a) * preds["ContentRidge"]
        rm, _ = rating_metrics(P, test)
        rk = ranking_metrics(P, M, test_rel, item_pop)
        sens.append({"alpha(MF weight)": a, "RMSE": rm, "P@10": rk["P@10"], "NDCG@10": rk["NDCG@10"]})
    sens = pd.DataFrame(sens)

    pd.set_option("display.width", 200)
    fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    print("\n=== Main results (higher is better except RMSE/MAE/AvgLogPop) ===")
    print(fmt(res))
    print("\n=== RMSE by item popularity (number of training ratings) ===")
    print(fmt(cold))
    print("\n=== Hybrid weight sensitivity ===")
    print(fmt(sens))
    res.to_csv(os.path.join(args.out, "results.csv"), index=False)
    cold.to_csv(os.path.join(args.out, "cold_start_rmse.csv"), index=False)
    sens.to_csv(os.path.join(args.out, "hybrid_sensitivity.csv"), index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        r1 = res.dropna(subset=["RMSE"])
        ax[0].barh(r1.Model, r1.RMSE); ax[0].set_title("RMSE (lower is better)"); ax[0].invert_yaxis()
        ax[0].set_xlim(r1.RMSE.min() * 0.9, r1.RMSE.max() * 1.03)
        ax[1].barh(res.Model, res["P@10"]); ax[1].set_title("Precision@10"); ax[1].invert_yaxis()
        ax[2].barh(res.Model, res["Coverage@10"]); ax[2].set_title("Catalogue coverage@10"); ax[2].invert_yaxis()
        for a_ in ax[1:]:
            a_.set_yticklabels([])
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "comparison.png"), dpi=130)
    except ImportError:
        pass
    print(f"\nSaved results to {args.out}/  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
