import re
import pandas as pd
from difflib import SequenceMatcher
from itertools import combinations

# =========================
# SETTINGS
# =========================
TITLE_SIMILARITY_THRESHOLD = 0.72
MAX_YEAR_GAP = 15
FRANCHISE_SCORE_THRESHOLD = 5

WEIGHTS = {
    "same_clean_title": 4,
    "numbered_title_pattern": 3,
    "title_similarity": 2,
    "same_director": 2,
    "shared_actors": 2,
    "shared_keywords": 1,
    "shared_genres": 1,
}

# =========================
# HELPERS
# =========================
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


def has_number_pattern(title):
    title = normalize_text(title)
    return bool(re.search(r"\b(\d+|ii|iii|iv|v)\b", title))


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def overlap(a, b):
    return len(a & b)


def safe_year(x):
    try:
        return int(x)
    except:
        return None


# =========================
# MAIN FUNCTION
# =========================
def create_franchises(df):
    df = df.copy()

    # -------------------------
    # BASIC FEATURES
    # -------------------------
    df["mainTitle"] = df["originalTitle"].fillna("").str.lower()
    df["cleanTitle"] = df["mainTitle"].apply(clean_title)
    df["hasNumber"] = df["mainTitle"].apply(has_number_pattern)
    df["year"] = df["releaseYear"].apply(safe_year)

    df["directors_set"] = df["directors"].apply(split_to_set) if "directors" in df else [set()] * len(df)
    df["actors_set"] = df["topFiveActors"].apply(split_to_set) if "topFiveActors" in df else [set()] * len(df)
    df["keywords_set"] = df["keywords"].apply(split_to_set) if "keywords" in df else [set()] * len(df)
    df["genres_set"] = df["genres"].apply(split_to_set) if "genres" in df else [set()] * len(df)

    # -------------------------
    # BLOCKING (CRITICAL FOR SPEED)
    # -------------------------
    df["block"] = df["cleanTitle"].apply(lambda x: x.split()[0] if x else "")

    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # -------------------------
    # PAIRWISE COMPARISON (BLOCKED)
    # -------------------------
    for _, indices in df.groupby("block").groups.items():

        if len(indices) < 2:
            continue

        for i, j in combinations(indices, 2):
            row1 = df.loc[i]
            row2 = df.loc[j]

            y1, y2 = row1["year"], row2["year"]

            # year filter
            if y1 and y2 and abs(y1 - y2) > MAX_YEAR_GAP:
                continue

            score = 0

            # same cleaned title
            if row1["cleanTitle"] == row2["cleanTitle"]:
                score += WEIGHTS["same_clean_title"]

            # numbered sequel
            if row1["cleanTitle"] == row2["cleanTitle"] and (row1["hasNumber"] or row2["hasNumber"]):
                score += WEIGHTS["numbered_title_pattern"]

            # similarity
            if similarity(row1["mainTitle"], row2["mainTitle"]) >= TITLE_SIMILARITY_THRESHOLD:
                score += WEIGHTS["title_similarity"]

            # director
            if overlap(row1["directors_set"], row2["directors_set"]) >= 1:
                score += WEIGHTS["same_director"]

            # actors
            if overlap(row1["actors_set"], row2["actors_set"]) >= 1:
                score += WEIGHTS["shared_actors"]

            # keywords
            if overlap(row1["keywords_set"], row2["keywords_set"]) >= 1:
                score += WEIGHTS["shared_keywords"]

            # genres
            if overlap(row1["genres_set"], row2["genres_set"]) >= 1:
                score += WEIGHTS["shared_genres"]

            if score >= FRANCHISE_SCORE_THRESHOLD:
                union(i, j)

    # -------------------------
    # ASSIGN GROUPS
    # -------------------------
    df["franchise_id"] = [find(i) for i in df.index]

    return df