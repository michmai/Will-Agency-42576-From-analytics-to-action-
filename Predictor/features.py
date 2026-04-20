import numpy as np
import pandas as pd

def build_features(df):
    features = []
    
    for fid, group in df.groupby("franchise_id"):
        group = group.sort_values("releaseYear")
        
        if len(group) < 2:  # ✅ FIXED
            continue
        
        for i in range(1, len(group)):
            prev = group.iloc[:i]
            next_movie = group.iloc[i]
            
            feat = compute_features(prev)
            feat["target"] = next_movie["imdbRating"]
            
            features.append(feat)
    
    return pd.DataFrame(features)


def compute_features(group):
    ratings = group["imdbRating"].values
    years = group["releaseYear"].values
    
    return {
        "num_movies": len(group),
        "avg_rating": ratings.mean(),
        "rating_std": ratings.std() if len(ratings) > 1 else 0,
        "rating_trend": np.polyfit(range(len(ratings)), ratings, 1)[0] if len(ratings) > 1 else 0,
        "last_rating": ratings[-1],
        "year_gap": years[-1] - years[-2] if len(years) > 1 else 0,
        "franchise_age": years[-1] - years[0],
        "rating_range": ratings.max() - ratings.min(),
        "recent_trend": ratings[-1] - ratings[-2] if len(ratings) > 1 else 0,
    }