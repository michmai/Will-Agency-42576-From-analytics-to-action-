import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("Data/European_data_2000.csv")

titles = df["originalTitle"].dropna().tolist()

print("Total movies:", len(titles))



model = SentenceTransformer("all-MiniLM-L6-v2")

title_embeddings = model.encode(titles)

from sklearn.cluster import DBSCAN

clustering = DBSCAN(
    eps=0.35,        # similarity threshold
    min_samples=2,
    metric="cosine"
)

labels = clustering.fit_predict(title_embeddings)

df["franchise_cluster"] = labels

franchises = df[df["franchise_cluster"] != -1]

print("Movies in franchises:", len(franchises))
print("Number of franchises:", franchises["franchise_cluster"].nunique())

for cluster_id in sorted(df["franchise_cluster"].unique())[:10]:

    if cluster_id == -1:
        continue

    print("\nFranchise", cluster_id)

    movies = df[df["franchise_cluster"] == cluster_id]["originalTitle"]

    for movie in movies:
        print(movie)

sequel_counts = df[df["franchise_cluster"] != -1]

print("Movies that belong to franchises:", len(sequel_counts))

# ---------------------------------------------------
# 1. Prepare franchise data
# ---------------------------------------------------

franchise_df = df[df["franchise_cluster"] != -1].copy()

franchise_df = franchise_df.dropna(subset=["imdbRating", "releaseYear"])

# sort movies within each franchise
franchise_df = franchise_df.sort_values(by=["franchise_cluster", "releaseYear"])


# ---------------------------------------------------
# 2. Compute per-franchise stats
# ---------------------------------------------------

franchise_results = []

for cluster_id, group in franchise_df.groupby("franchise_cluster"):
    
    group = group.sort_values("releaseYear")
    
    # skip very small groups
    if len(group) < 2:
        continue
    
    original = group.iloc[0]
    sequels = group.iloc[1:]
    
    original_rating = original["imdbRating"]
    sequel_avg = sequels["imdbRating"].mean()
    
    diff = sequel_avg - original_rating
    
    franchise_results.append({
        "franchise_id": cluster_id,
        "original_title": original["originalTitle"],
        "original_rating": original_rating,
        "sequel_avg_rating": sequel_avg,
        "difference": diff,
        "num_movies": len(group)
    })


# convert to dataframe
results_df = pd.DataFrame(franchise_results)

print(results_df.head())

# ---------------------------------------------------
# Plot: Original vs Sequel per Franchise
# ---------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# limit to top N franchises (otherwise too crowded)
top_n = 15

plot_df = results_df.sort_values("num_movies", ascending=False).head(top_n)

x = np.arange(len(plot_df))

originals = plot_df["original_rating"]
sequels = plot_df["sequel_avg_rating"]

width = 0.35

plt.figure()

plt.bar(x - width/2, originals, width, label="Original")
plt.bar(x + width/2, sequels, width, label="Sequels")

plt.xticks(x, plot_df["original_title"], rotation=45, ha="right")

plt.ylabel("IMDb Rating")
plt.title("Original vs Sequel Ratings per Franchise")
plt.legend()

plt.tight_layout()
plt.show()

for i in range(len(plot_df)):
    diff = sequels.iloc[i] - originals.iloc[i]
    
    plt.text(
        x[i],
        max(originals.iloc[i], sequels.iloc[i]) + 0.1,
        f"{diff:+.2f}",
        ha="center"
    )