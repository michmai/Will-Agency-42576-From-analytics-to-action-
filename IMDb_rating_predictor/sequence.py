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

for cluster_id in sorted(df["franchise_cluster"].unique()):

    if cluster_id == -1:
        continue

    print("\nFranchise", cluster_id)

    movies = df[df["franchise_cluster"] == cluster_id]["originalTitle"]

    for movie in movies:
        print(movie)

sequel_counts = df[df["franchise_cluster"] != -1]

print("Movies that belong to franchises:", len(sequel_counts))