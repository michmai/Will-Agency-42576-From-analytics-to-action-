import pandas as pd
import json
import os

def load_tmdb_data(path):
    df = pd.read_csv(os.path.join(path, "movies_metadata.csv"), low_memory=False)

    # ----------------------------
    # COLLECTION ID
    # ----------------------------
    def extract_collection_id(x):
        if pd.isna(x):
            return None
        try:
            data = json.loads(x.replace("'", '"'))
            return data.get("id")
        except:
            return None

    df["franchise_id"] = df["belongs_to_collection"].apply(extract_collection_id)

    # keep only movies in collections
    df = df[df["franchise_id"].notna()]

    # ----------------------------
    # CLEAN NUMERIC FIELDS
    # ----------------------------
    df["releaseYear"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["imdbRating"] = pd.to_numeric(df["vote_average"], errors="coerce")

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    # remove invalid rows
    df = df.dropna(subset=["releaseYear", "imdbRating", "budget", "revenue"])
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

    df = df.rename(columns={"title": "originalTitle"})

    return df