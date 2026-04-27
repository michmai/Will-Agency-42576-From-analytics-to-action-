import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_text(row):
    return " ".join([
        str(row.get("originalTitle", "")),
        str(row.get("plotMedium", "")),
        str(row.get("keywords", "")),
        str(row.get("genres", "")),
        str(row.get("topFiveActors", ""))[:200],
    ]).lower()

def add_embeddings(df):
    texts = df.apply(build_text, axis=1).tolist()
    
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # remove bad rows
    mask = np.isfinite(embeddings).all(axis=1)
    df = df.iloc[mask].copy()
    embeddings = embeddings[mask]

    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    return df, embeddings

def cluster_movies(df, eps=0.25, min_samples=2):
    print("Clustering movies with semantic embeddings...")

    df, embeddings = add_embeddings(df)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    )

    labels = clustering.fit_predict(embeddings)

    df["franchise_id"] = labels

    # remove noise (-1 clusters)
    df = df[df["franchise_id"] != -1]

    return df