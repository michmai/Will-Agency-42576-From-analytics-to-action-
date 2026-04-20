from data_loader import load_data
from franchise import create_franchises
from features import build_features
from model import train_model
from predictor import predict_next_movie
import numpy as np

# Load data
df = load_data("/Users/michellemai/Documents/GitHub/Will-Agency-42576-From-analytics-to-action-/Data/Movie_50k.csv")


# Process
df = create_franchises(df)
print(df)
feat_df = build_features(df)
feat_df = feat_df.dropna(subset=["target"])
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
feat_df = feat_df.dropna()

print("Feature dataset size:", feat_df.shape)
print(feat_df.head())

# Train
model, score = train_model(feat_df)
print("Model score:", score)

# Predict
best_id = df["franchise_id"].value_counts().idxmax()
print("Using franchise:", best_id)

print(predict_next_movie(df, model, franchise_id=best_id))

import matplotlib.pyplot as plt

importances = model.feature_importances_
features = feat_df.columns

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()