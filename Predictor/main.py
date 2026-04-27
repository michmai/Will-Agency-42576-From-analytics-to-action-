from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_loader import load_tmdb_data
from features import build_features
from model import train_model
from predictor import predict_next_movie
from visualization import plot_franchise_sequence

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
print("\n📉 Model MAE:", mae)

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
top = results_df.sort_values("predicted_rating", ascending=False).head(10)

plt.figure()
plt.barh(top["last_movie"], top["predicted_rating"])
plt.xlabel("Predicted Rating")
plt.title("Top 10 Franchises (TMDB)")
plt.gca().invert_yaxis()
plt.show()

# ----------------------------
# VISUALIZATION 2: PREDICTED VS ACTUAL
# ----------------------------
X = feat_df.drop(columns=["target"])
y = feat_df["target"]
preds = model.predict(X)

plt.figure()
plt.scatter(y, preds, alpha=0.5)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Predicted vs Actual Ratings")
plt.show()

# ----------------------------
# VISUALIZATION 3: PROFIT VS RATING
# ----------------------------
grouped = df.groupby("franchise_id").agg({
    "imdbRating": "mean",
    "revenue": "mean",
    "budget": "mean"
})

grouped["profit"] = grouped["revenue"] - grouped["budget"]

plt.figure()
plt.scatter(grouped["profit"], grouped["imdbRating"], alpha=0.5)
plt.xlabel("Average Profit")
plt.ylabel("Average Rating")
plt.title("Profit vs Rating")
plt.show()

# pick a good franchise (not too small, not too big)
fid = df.groupby("franchise_id").size().sort_values(ascending=False).index[0]

plot_franchise_sequence(df, fid)