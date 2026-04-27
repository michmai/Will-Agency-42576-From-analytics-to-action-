import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# MODEL
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# TEXT BUILDING
# ----------------------------
def build_text(row):
    title = str(row.get("originalTitle", "")).lower()
    boosted_title = (title + " ") * 3

    return " ".join([
        boosted_title,
        str(row.get("plotMedium", "")),
        str(row.get("keywords", "")),
        str(row.get("genres", "")),
        str(row.get("topFiveActors", ""))[:200],
    ])


# ----------------------------
# EMBEDDINGS
# ----------------------------
def add_embeddings(df):
    print("Generating embeddings...")

    texts = df.apply(build_text, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    mask = np.isfinite(embeddings).all(axis=1)
    df = df.iloc[mask].copy()
    embeddings = embeddings[mask]

    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    # 🔥 IMPORTANT FIX
    df = df.reset_index(drop=True)

    return df, embeddings


# ----------------------------
# BLOCKING (IMPROVED)
# ----------------------------
def create_blocks(df):
    blocks = {}

    for idx, row in df.iterrows():
        title = str(row["originalTitle"]).lower().split()
        actors = str(row.get("topFiveActors", "")).lower().split(",")

        keys = set()

        # first 2 words of title
        if len(title) >= 1:
            keys.add("t1:" + title[0])
        if len(title) >= 2:
            keys.add("t2:" + title[1])

        # first actor
        if actors:
            keys.add("a:" + actors[0].strip())

        for k in keys:
            blocks.setdefault(k, []).append(idx)

    return blocks


# ----------------------------
# CLUSTERING (GRAPH)
# ----------------------------
def cluster_movies(df, similarity_threshold=0.65, debug=True):
    print("🔍 Clustering movies (graph-based)...")

    df, embeddings = add_embeddings(df)
    blocks = create_blocks(df)

    n = len(df)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # compare only within blocks
    for block, indices in blocks.items():
        if len(indices) < 2:
            continue

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]

                sim = np.dot(embeddings[a], embeddings[b])

                if sim > similarity_threshold:
                    union(a, b)

    df["franchise_id"] = [find(i) for i in range(n)]

    # remove small clusters
    sizes = df.groupby("franchise_id").size()
    df = df[df["franchise_id"].isin(sizes[sizes >= 2].index)]

    # ----------------------------
    # DEBUG
    # ----------------------------
    if debug:
        sizes = df.groupby("franchise_id").size().sort_values(ascending=False)

        print("\n Top clusters:")
        print(sizes.head(10))

        for fid in sizes.head(3).index:
            print("\n--- Cluster", fid, "---")
            print(df[df["franchise_id"] == fid][["originalTitle", "releaseYear"]].head(10))

    return df