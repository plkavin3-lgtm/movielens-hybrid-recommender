import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="MovieLens Hybrid Recommender API")
bundle = joblib.load("model_bundle.joblib")

USER_TO_IDX = {int(uid): idx for idx, uid in enumerate(bundle["user_ids"])}
ITEM_TO_IDX = {int(mid): idx for idx, mid in enumerate(bundle["item_ids"])}
MOVIES = bundle["movies"]

@app.get("/")
def health():
    return {
        "status": "online",
        "users": len(USER_TO_IDX),
        "movies": len(ITEM_TO_IDX),
    }

@app.get("/recommend/{user_id}")
def recommend_for_user(
    user_id: int,
    model: str = Query("hybrid", enum=["hybrid", "collaborative", "content"]),
    k: int = Query(10, ge=1, le=50),
):
    if user_id not in USER_TO_IDX:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in training set")

    u_idx = USER_TO_IDX[user_id]
    matrix_map = {
        "hybrid": bundle["pred_hybrid"],
        "collaborative": bundle["pred_mf"],
        "content": bundle["pred_content"],
    }
    scores = matrix_map[model][u_idx].copy()
    scores[bundle["seen_mask"][u_idx]] = -np.inf  # Exclude already watched movies

    top_idx = np.argsort(-scores)[:k]
    recs = [
        {
            **MOVIES[int(i)],
            "predicted_rating": round(float(np.clip(scores[i], 0.5, 5.0)), 2),
        }
        for i in top_idx
    ]
    return {"user_id": user_id, "model": model, "recommendations": recs}

@app.get("/similar/{movie_id}")
def similar_movies(movie_id: int, k: int = Query(10, ge=1, le=50)):
    """Content-based item similarity (handles cold-start / unauthenticated users)."""
    if movie_id not in ITEM_TO_IDX:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    i_idx = ITEM_TO_IDX[movie_id]
    sims = bundle["S_content"][i_idx].copy()
    sims[i_idx] = -1.0
    top_idx = np.argsort(-sims)[:k]

    return {
        "source_movie": MOVIES[i_idx],
        "similar_movies": [
            {**MOVIES[int(i)], "similarity": round(float(sims[i]), 3)}
            for i in top_idx
        ],
    }