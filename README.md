🎬 MovieLens Hybrid Recommendation System

An end-to-end movie recommendation engine comparing Collaborative Filtering, Content-Based Filtering, and a Hybrid approach on the MovieLens (ml-latest-small) dataset, deployed as a production-ready REST API using FastAPI and Render.

📌 Project Overview

Modern recommender systems balance two competing goals: rating accuracy on popular items and catalog discovery on niche/new items. This project implements, benchmarks, and deploys multiple recommendation paradigms from scratch using NumPy, SciPy/Scikit-Learn, and Pandas:

Baselines: Global Mean + Damped User/Item Biases, Popularity Ranking, and Random Ranking.

Collaborative Filtering (CF):

UserCF ($k$-NN): User-user cosine similarity on baseline residuals with co-rating shrinkage.

ItemCF ($k$-NN): Item-item cosine similarity on baseline residuals with co-rating shrinkage.

Biased Matrix Factorization (MF): Stochastic Gradient Descent (SGD) learning $k=32$ latent user/item factors alongside regularized biases.

Content-Based Filtering (CBF):

ContentKNN: Item-$k$NN powered by TF-IDF vectors built from movie genres, release decades, and user-generated tags.

ContentRidge: Per-user regularized linear regression solved in the dual space using a linear TF-IDF kernel ($K = XX^T$).

Hybrid Model: Weighted linear ensemble combining collaborative latent factors and content profiles:


$$\hat{r}_{u,i} = 0.75 \cdot \hat{r}_{u,i}^{\text{MF}} + 0.25 \cdot \hat{r}_{u,i}^{\text{ContentRidge}}$$

📊 Evaluation & Comparative Analysis

Models are evaluated using an 80/20 per-user stratified split (min_user=20, min_item=5, relevance threshold $\ge 4.0$) across both rating prediction and top-$N$ ranking metrics:

Rating Accuracy: $\text{RMSE}$ and $\text{MAE}$ (lower is better).

Ranking Quality: $\text{Precision@10}$, $\text{Recall@10}$, and $\text{NDCG@10}$ (higher is better; previously rated items are masked).

Diversity & Cold-Start: $\text{Coverage@10}$ (catalog reach) and popularity-bucketed $\text{RMSE}$ (0-4, 5-19, 20-99, 100+ training ratings).

Strengths & Weaknesses Summary

Approach

Strengths

Weaknesses

Collaborative Filtering (UserCF, ItemCF, BiasedMF)

• Learns deep latent taste patterns without needing item metadata.



• Delivers high serendipity across unexpected genres.



• Achieves the lowest $\text{RMSE}$ on well-rated mainstream movies (100+ ratings).

• Item Cold-Start: Cannot accurately score brand-new movies with zero or very few ratings (0-4 bucket).



• Popularity Bias: Concentrates recommendations on blockbusters (lower Coverage@10).

Content-Based Filtering (ContentKNN, ContentRidge)

• Solves Item Cold-Start: Recommends newly released movies immediately using genres, decade, and tags.



• High catalog coverage (Coverage@10) and transparent, explainable recommendations.

• Filter Bubble / Over-Specialization: Rarely recommends items outside a user's historical genre/tag profile.



• Performance depends heavily on metadata/tag completeness.

Hybrid (0.75 MF + 0.25 Content)

• Combines the predictive accuracy of Matrix Factorization with the cold-start resilience and catalog diversity of ContentRidge.

• Requires maintaining both collaborative interaction matrices and TF-IDF feature stores.

🚀 API Endpoints (FastAPI)

The trained models are serialized via joblib (model_bundle.joblib) for low-latency inference without retraining at startup.

GET / — Health check returning active user and movie catalog counts.

GET /recommend/{user_id}?model=hybrid&k=10 — Returns top-$k$ personalized movie recommendations excluding already-watched items. Supports model values: hybrid, collaborative, or content.

GET /similar/{movie_id}?k=10 — Returns top-$k$ content-similar movies based on TF-IDF cosine similarity (ideal for cold-start or anonymous browsing).

GET /docs — Interactive Swagger UI documentation.

🛠️ Project Structure & Local Setup

├── recsys_compare.py       # Offline benchmarking, cold-start analysis & plots
├── export_model.py         # Trains models and serializes model_bundle.joblib
├── app.py                  # FastAPI inference server
├── model_bundle.joblib     # Pre-trained model artifacts
├── requirements.txt        # Python dependencies
├── data/ml-latest-small/   # MovieLens dataset
└── results/                # Benchmark CSVs (results.csv, cold_start_rmse.csv, hybrid_sensitivity.csv)


Run Locally

pip install -r requirements.txt
python recsys_compare.py        # Generate evaluation metrics in results/
python export_model.py          # Build model_bundle.joblib
uvicorn app:app --reload        # Start API at http://127.0.0.1:8000/docs
