from data_loader import load_data
from franchise import create_franchises
from features import build_features
from model import train_model
from predictor import predict_next_movie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

# Load data
df = load_data("/Users/michellemai/Documents/GitHub/Will-Agency-42576-From-analytics-to-action-/Data/Movie_50k.csv")
df, edges = create_franchises(df)

# Stats (moved outside loop)
print("Total franchises:", df["franchise_id"].nunique())

sizes = df.groupby("franchise_id").size()
print("Franchises with >=2 movies:", (sizes >= 2).sum())
print("Franchises with >=3 movies:", (sizes >= 3).sum())

# Build features
feat_df = build_features(df)
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
feat_df = feat_df.dropna()
feat_df.to_csv("outputs/features.csv", index=False)

# Train
model, score = train_model(feat_df)
print("Model score:", score)

# Predict
results = []

for fid, group in df.groupby("franchise_id"):
    if len(group) < 2:
        continue
    
    group_sorted = group.sort_values("releaseYear")
    last_movie = group_sorted["originalTitle"].iloc[-1]  # ✅ last movie
    
    pred = predict_next_movie(df, model, fid)
    
    results.append({
        "franchise_id": fid,
        "last_movie": last_movie,  # ✅ NEW
        "num_movies": len(group),
        "predicted_rating": pred["predicted_rating"],
        "should_make_movie": pred["should_make_movie"]
    })
    
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/franchise_predictions.csv", index=False)

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
# ------------------------
# Visualization: Top franchises
# ------------------------
def get_franchise_name(fid):
    group = df[df["franchise_id"] == fid].sort_values("releaseYear")
    return group["originalTitle"].iloc[0] if not group.empty else "Unknown"

top = results_df.sort_values("predicted_rating", ascending=False).head(10)

plt.figure()
plt.barh(top["last_movie"], top["predicted_rating"])
plt.xlabel("Predicted Rating")
plt.title("Top 100 Franchises (Next Movie After Last Entry)")
plt.gca().invert_yaxis()
plt.show()

# ------------------------
# Feature importance
# ------------------------
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

# ------------------------
# Graph visualization (cleaned)
# ------------------------
import networkx as nx

G = nx.Graph()

# Add nodes
for i in df.index:
    G.add_node(i)

# Use only subset of edges (avoid hairball)
G.add_edges_from(edges[:200])

largest_id = df["franchise_id"].value_counts().idxmax()
largest_group = df[df["franchise_id"] == largest_id].index

G_sub = G.subgraph(largest_group)

plt.figure(figsize=(8, 6))
nx.draw(G_sub, node_size=50, with_labels=False)
plt.title("Franchise Graph (Sampled)")

plt.savefig("outputs/franchise_graph.png", dpi=300, bbox_inches="tight")
plt.show()