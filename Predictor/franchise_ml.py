import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

# =========================
# MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# BUILD TEXT
# =========================
def build_text(row):
    return " ".join([
        str(row.get("originalTitle", "")),
        str(row.get("keywords", "")),
        str(row.get("genres", "")),
        str(row.get("topFiveActors", ""))[:100],  # limit noise
    ]).lower()

# =========================
# EMBEDDINGS
# =========================
def add_embeddings(df):
    texts = df.apply(build_text, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)

    embeddings = np.array(embeddings, dtype=np.float32)

    # 🚨 remove any bad rows BEFORE anything else
    mask = np.isfinite(embeddings).all(axis=1)
    embeddings = embeddings[mask]
    df = df.iloc[mask].copy()

    # normalize safely
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10

    embeddings = embeddings / norms

    return df, embeddings

# =========================
# CLUSTERING
# =========================
def cluster_movies(df, eps=0.35, min_samples=2):
    df = df.copy()

    df, embeddings = add_embeddings(df)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    )

    labels = clustering.fit_predict(embeddings)

    df["franchise_id"] = labels

    return df