import joblib
import numpy as np
import recsys_compare as rc

def build_and_save(data_dir="data", out_path="model_bundle.joblib"):
    ratings, movies, tags = rc.load_movielens(data_dir)
    df, user_ids, item_ids = rc.prepare(ratings, min_user=20, min_item=5)
    nu, ni = len(user_ids), len(item_ids)
    R, M = rc.to_matrix(df, nu, ni)

    B_full, B_user = rc.fit_baselines(R, M)
    X = rc.build_content(movies, tags, item_ids)
    S_c = (X @ X.T).toarray().astype(np.float32)

    pred_mf = rc.BiasedMF(epochs=40, seed=42).fit_predict(df, nu, ni)
    pred_content = rc.content_ridge(R, M, B_user, S_c, lam=1.0)
    pred_hybrid = 0.75 * pred_mf + 0.25 * pred_content

    movie_meta = (
        movies.set_index("movieId")
        .loc[item_ids][["title", "genres"]]
        .reset_index()
        .to_dict(orient="records")
    )

    bundle = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "seen_mask": M > 0,
        "pred_mf": pred_mf,
        "pred_content": pred_content,
        "pred_hybrid": pred_hybrid,
        "S_content": S_c,
        "movies": movie_meta,
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"Saved bundle to {out_path} ({nu} users, {ni} movies)")

if __name__ == "__main__":
    build_and_save()