from data_loader import load_data
from features import build_features
from franchise_ml import cluster_movies
from model import train_model
from predictor import predict_next_movie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = load_data("/Users/michellemai/Documents/GitHub/Will-Agency-42576-From-analytics-to-action-/Data/Movie_50k.csv")

# ----------------------------
# CLEAN + DEDUP
# ----------------------------
df["norm"] = (
    df["originalTitle"]
    .str.lower()
    .str.replace(r"\(.*?\)", "", regex=True)
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.strip()
)

df = df.drop_duplicates(subset=["norm", "releaseYear"])

# ----------------------------
# CLUSTER INTO FRANCHISES
# ----------------------------
df = cluster_movies(df)

# Keep only franchises with >= 3 movies (stability)
sizes = df.groupby("franchise_id").size()
df = df[df["franchise_id"].isin(sizes[sizes >= 3].index)]

# Recompute sizes AFTER filtering (fix bug)
sizes = df.groupby("franchise_id").size()

print("\n📊 DATASET STATS")
print("Total franchises:", df["franchise_id"].nunique())
print("Franchises with >=2 movies:", (sizes >= 2).sum())
print("Franchises with >=3 movies:", (sizes >= 3).sum())

# ----------------------------
# BUILD FEATURES
# ----------------------------
feat_df = build_features(df)
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
feat_df = feat_df.dropna()

feat_df.to_csv("outputs/features.csv", index=False)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model, mae = train_model(feat_df)
print("\n📉 Model MAE:", mae)

# ----------------------------
# PREDICT NEXT MOVIE
# ----------------------------
results = []

# dynamic threshold (better than fixed 6.5)
threshold = df["imdbRating"].mean()

for fid, group in df.groupby("franchise_id"):
    if len(group) < 2:
        continue

    group_sorted = group.sort_values("releaseYear")
    last_movie = group_sorted["originalTitle"].iloc[-1]

    pred = predict_next_movie(df, model, fid)

    results.append({
        "franchise_id": fid,
        "last_movie": last_movie,
        "num_movies": len(group),
        "predicted_rating": pred["predicted_rating"],
        "should_make_movie": bool(pred["predicted_rating"] > threshold)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/franchise_predictions.csv", index=False)

# ----------------------------
# SAVE FRANCHISE SEQUENCES
# ----------------------------
franchise_strings = []

for fid, group in df.groupby("franchise_id"):
    if len(group) < 2:
        continue

    group = group.sort_values("releaseYear")
    movies = " → ".join(group["originalTitle"].tolist())

    franchise_strings.append({
        "franchise_id": fid,
        "movies": movies
    })

pd.DataFrame(franchise_strings).to_csv("outputs/franchise_sequences.csv", index=False)

# ----------------------------
# VISUALIZATION
# ----------------------------
top = results_df.sort_values("predicted_rating", ascending=False).head(10)

plt.figure()
plt.barh(top["last_movie"], top["predicted_rating"])
plt.xlabel("Predicted Rating")
plt.title("Top 10 Franchises (Next Movie Prediction)")
plt.gca().invert_yaxis()
plt.show()

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
X = feat_df.drop(columns=["target"])

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure()
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()