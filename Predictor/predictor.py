import pandas as pd
from features import compute_features

def predict_next_movie(df, model, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")
    
    if len(group) < 2:
        return {
            "predicted_rating": None,
            "should_make_movie": None
        }
    
    features = pd.DataFrame([compute_features(group)])
    features = features.fillna(0)
    
    pred = model.predict(features)[0]
    
    return {
        "predicted_rating": float(pred),
        "should_make_movie": bool(pred > 6.5)
    }