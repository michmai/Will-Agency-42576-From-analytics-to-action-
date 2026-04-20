import re
import pandas as pd
from difflib import SequenceMatcher
from itertools import combinations

TITLE_SIMILARITY_THRESHOLD = 0.6
MAX_YEAR_GAP = 15
FRANCHISE_SCORE_THRESHOLD = 3

STOPWORDS = {"the", "a", "an", "and", "of", "in", "on", "to"}

WEIGHTS = {
    "prefix_match": 2,
    "title_similarity": 1,
    "same_director": 3,
    "shared_actors": 3,
    "shared_keywords": 2,
    "shared_genres": 1,
}

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def split_to_set(x):
    x = normalize_text(x)
    if not x:
        return set()
    return {p.strip() for p in re.split(r"[;,|/]+", x) if p.strip()}

def clean_title(title):
    title = normalize_text(title)
    title = re.sub(r"\b(part|chapter|episode|vol|volume)\s*\d+\b", "", title)
    title = re.sub(r"\b\d+\b", "", title)
    title = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", "", title)
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()

def get_prefix(x, n=2):
    words = [w for w in normalize_text(x).split() if w not in STOPWORDS]
    return " ".join(words[:n]) if words else ""

def similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def overlap(a, b):
    return len(a & b)

def safe_year(x):
    try:
        return int(x)
    except:
        return None

def is_duplicate(row1, row2):
    if row1["year"] != row2["year"]:
        return False
    if row1["cleanTitle"] == row2["cleanTitle"]:
        return True
    return similarity(row1["mainTitle"], row2["mainTitle"]) > 0.95

def create_blocks(row):
    blocks = set()

    prefix = get_prefix(row["mainTitle"], 2)
    if prefix:
        blocks.add(f"title:{prefix}")

    for actor in list(row["actors_set"])[:2]:
        blocks.add(f"actor:{actor}")

    for d in list(row["directors_set"])[:1]:
        blocks.add(f"director:{d}")

    for k in list(row["keywords_set"])[:2]:
        blocks.add(f"kw:{k}")

    return blocks

def create_franchises(df):
    df = df.copy().reset_index(drop=True)

    df["mainTitle"] = df["originalTitle"].fillna("").str.lower()
    df["cleanTitle"] = df["mainTitle"].apply(clean_title)
    df["year"] = df["releaseYear"].apply(safe_year)

    df["directors_set"] = df["directors"].apply(split_to_set) if "directors" in df.columns else [set()] * len(df)
    df["actors_set"] = df["topFiveActors"].apply(split_to_set) if "topFiveActors" in df.columns else [set()] * len(df)
    df["keywords_set"] = df["keywords"].apply(split_to_set) if "keywords" in df.columns else [set()] * len(df)
    df["genres_set"] = df["genres"].apply(split_to_set) if "genres" in df.columns else [set()] * len(df)

    df["blocks"] = df.apply(create_blocks, axis=1)

    block_dict = {}
    for idx, row in df.iterrows():
        for b in row["blocks"]:
            block_dict.setdefault(b, []).append(idx)

    parent = list(range(len(df)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for indices in block_dict.values():
        if len(indices) < 2:
            continue

        for i, j in combinations(indices, 2):
            row1, row2 = df.loc[i], df.loc[j]

            if is_duplicate(row1, row2):
                continue

            y1, y2 = row1["year"], row2["year"]
            if y1 and y2 and abs(y1 - y2) > MAX_YEAR_GAP:
                continue

            score = 0

            if get_prefix(row1["mainTitle"], 2) == get_prefix(row2["mainTitle"], 2):
                score += WEIGHTS["prefix_match"]

            if similarity(row1["mainTitle"], row2["mainTitle"]) >= TITLE_SIMILARITY_THRESHOLD:
                score += WEIGHTS["title_similarity"]

            if overlap(row1["actors_set"], row2["actors_set"]) >= 1:
                score += WEIGHTS["shared_actors"]

            if overlap(row1["directors_set"], row2["directors_set"]) >= 1:
                score += WEIGHTS["same_director"]

            if overlap(row1["keywords_set"], row2["keywords_set"]) >= 1:
                score += WEIGHTS["shared_keywords"]

            if overlap(row1["genres_set"], row2["genres_set"]) >= 1:
                score += WEIGHTS["shared_genres"]

            if score >= FRANCHISE_SCORE_THRESHOLD:
                union(i, j)

    df["franchise_id"] = [find(i) for i in df.index]
    return df