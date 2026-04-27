from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_loader import load_tmdb_data
from features import build_features
from model import train_model
from predictor import predict_next_movie
from visualization import plot_franchise_sequence
from visualization import plot_with_trend
from visualization import plot_feature_importance
from visualization import plot_top_franchises

# ----------------------------
# DOWNLOAD DATA
# ----------------------------
api = KaggleApi()
api.authenticate()
path = "tmdb-movie-metadata"
api.dataset_download_files("tmdb/tmdb-movie-metadata", path=path, unzip=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = load_tmdb_data(path)

# keep stable franchises
sizes = df.groupby("franchise_id").size()
df = df[df["franchise_id"].isin(sizes[sizes >= 3].index)]

print("DATASET STATS")
print("Total franchises:", df["franchise_id"].nunique())

# ----------------------------
# BUILD FEATURES
# ----------------------------
feat_df = build_features(df)
feat_df = feat_df.dropna()

os.makedirs("outputs", exist_ok=True)
feat_df.to_csv("outputs/features.csv", index=False)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model, mae = train_model(feat_df)
print("\n Model MAE:", mae)

# ----------------------------
# PREDICT
# ----------------------------
results = []

for fid, group in df.groupby("franchise_id"):
    pred = predict_next_movie(df, model, fid)
    
    if pred is None:
        continue
    
    group_sorted = group.sort_values("releaseYear")
    
    results.append({
        "franchise_id": fid,
        "last_movie": group_sorted["originalTitle"].iloc[-1],
        "predicted_rating": pred
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/predictions.csv", index=False)

# ----------------------------
# VISUALIZATION 1: TOP FRANCHISES
# ----------------------------
plot_top_franchises(df, n=10)


# ----------------------------
# VISUALIZATION 3: PROFIT VS RATING
# ----------------------------
grouped = df.groupby("franchise_id").agg({
    "imdbRating": "mean",
    "revenue": "mean",
    "budget": "mean"
})

grouped["profit"] = grouped["revenue"] - grouped["budget"]

plt.figure(figsize=(9,6))

plt.scatter(
    grouped["profit"],
    grouped["imdbRating"],
    alpha=0.6
)

plt.xscale("log") 

plt.title("Profit vs Rating", fontsize=18, weight="bold")
plt.xlabel("Average Profit (log scale)", fontsize=13)
plt.ylabel("Average Rating", fontsize=13)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

fid = df.groupby("franchise_id").size().sort_values(ascending=False).index[1]
plot_franchise_sequence(df, fid)

X = feat_df.drop(columns=["target"])
plot_feature_importance(model, X.columns)

plot_with_trend(df, fid)

grouped = feat_df.copy()

plt.scatter(grouped["profitability"], grouped["rating_trend"], alpha=0.5)
plt.xlabel("Profitability")
plt.ylabel("Rating Trend")
plt.title("Profit vs Quality Trend")
plt.show()

plt.hist(feat_df["rating_trend"], bins=30)
plt.title("Distribution of Franchise Trends")
plt.xlabel("Trend (slope)")
plt.ylabel("Count")
plt.show()