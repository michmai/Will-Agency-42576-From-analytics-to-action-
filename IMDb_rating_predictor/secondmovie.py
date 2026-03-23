import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from difflib import SequenceMatcher
from itertools import combinations

# =========================
# LOAD DATA
# =========================
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "Data" / "European_data_2000.csv")

# Optional: keep only movies if titleType exists
if "titleType" in df.columns:
    df = df[df["titleType"].astype(str).str.lower() == "movie"].copy()

df = df.reset_index(drop=True)

# =========================
# SETTINGS
# =========================
TITLE_SIMILARITY_THRESHOLD = 0.72
ACTOR_OVERLAP_MIN = 1
KEYWORD_OVERLAP_MIN = 1
GENRE_OVERLAP_MIN = 1
MAX_YEAR_GAP = 15

# Final score needed to say two movies are related
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
# HELPER FUNCTIONS
# =========================
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def split_to_set(x):
    """
    Splits text fields like:
    'actor1, actor2, actor3'
    into a cleaned set.
    """
    x = normalize_text(x)
    if not x:
        return set()

    parts = re.split(r"[;,|/]+", x)
    return {p.strip() for p in parts if p.strip()}


def get_main_title(row):
    """
    Prefer englishTitle, fall back to originalTitle.
    """
    english = normalize_text(row.get("englishTitle", ""))
    original = normalize_text(row.get("originalTitle", ""))
    return english if english else original


def clean_title_for_matching(title):
    """
    Removes sequel indicators and punctuation to get a base title.
    """
    title = normalize_text(title)

    # Remove sequel words with numbers
    title = re.sub(r"\b(part|chapter|episode|vol|volume)\s*\d+\b", "", title)
    title = re.sub(r"\bpart\s+(ii|iii|iv|v|vi|vii|viii|ix|x)\b", "", title)

    # Remove standalone numbers and roman numerals
    title = re.sub(r"\b\d+\b", "", title)
    title = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b", "", title)

    # Remove punctuation
    title = re.sub(r"[^\w\s]", " ", title)

    # Collapse spaces
    title = re.sub(r"\s+", " ", title).strip()
    return title


def has_numbered_sequel_pattern(title):
    """
    Checks if title contains sequel-like numbering.
    """
    title = normalize_text(title)

    patterns = [
        r"\b\d+\b",
        r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b",
        r"\b(part|chapter|episode|vol|volume)\s+\d+\b",
        r"\bpart\s+(ii|iii|iv|v|vi|vii|viii|ix|x)\b",
    ]

    return any(re.search(pattern, title) for pattern in patterns)


def title_similarity(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def overlap_size(set1, set2):
    return len(set1 & set2)


def safe_year(x):
    try:
        return int(x)
    except Exception:
        return None


# =========================
# FEATURE PREPARATION
# =========================
df["mainTitle"] = df.apply(get_main_title, axis=1)
df["cleanTitle"] = df["mainTitle"].apply(clean_title_for_matching)
df["hasNumberPattern"] = df["mainTitle"].apply(has_numbered_sequel_pattern)

# Ensure releaseYear_int exists even if releaseYear is missing
if "releaseYear" in df.columns:
    df["releaseYear_int"] = df["releaseYear"].apply(safe_year)
else:
    df["releaseYear"] = None
    df["releaseYear_int"] = None

df["directors_set"] = df["directors"].apply(split_to_set) if "directors" in df.columns else [set()] * len(df)
df["actors_set"] = df["topFiveActors"].apply(split_to_set) if "topFiveActors" in df.columns else [set()] * len(df)
df["keywords_set"] = df["keywords"].apply(split_to_set) if "keywords" in df.columns else [set()] * len(df)
df["genres_set"] = df["genres"].apply(split_to_set) if "genres" in df.columns else [set()] * len(df)

# =========================
# PAIRWISE COMPARISON
# =========================
related_pairs = []

for i, j in combinations(df.index, 2):
    row1 = df.loc[i]
    row2 = df.loc[j]

    year1 = row1["releaseYear_int"]
    year2 = row2["releaseYear_int"]

    # Skip likely duplicate record of the same movie:
    # same normalized title + same release year
    if (
        row1["cleanTitle"]
        and row1["cleanTitle"] == row2["cleanTitle"]
        and year1 is not None
        and year2 is not None
        and year1 == year2
    ):
        continue

    # Skip if time gap is too large and both years are known
    if year1 is not None and year2 is not None:
        if abs(year1 - year2) > MAX_YEAR_GAP:
            continue

    score = 0
    reasons = []

    # Same cleaned title
    if row1["cleanTitle"] and row1["cleanTitle"] == row2["cleanTitle"]:
        score += WEIGHTS["same_clean_title"]
        reasons.append("same_clean_title")

    # Numbered sequel pattern
    if (
        row1["cleanTitle"]
        and row1["cleanTitle"] == row2["cleanTitle"]
        and (row1["hasNumberPattern"] or row2["hasNumberPattern"])
    ):
        score += WEIGHTS["numbered_title_pattern"]
        reasons.append("numbered_title_pattern")

    # Similar titles
    sim = title_similarity(row1["mainTitle"], row2["mainTitle"])
    if sim >= TITLE_SIMILARITY_THRESHOLD:
        score += WEIGHTS["title_similarity"]
        reasons.append(f"title_similarity={sim:.2f}")

    # Same director
    director_overlap = overlap_size(row1["directors_set"], row2["directors_set"])
    if director_overlap >= 1:
        score += WEIGHTS["same_director"]
        reasons.append(f"same_director={director_overlap}")

    # Shared actors: 1 or more is enough
    actor_overlap = overlap_size(row1["actors_set"], row2["actors_set"])
    if actor_overlap >= ACTOR_OVERLAP_MIN:
        score += WEIGHTS["shared_actors"]
        reasons.append(f"shared_actors={actor_overlap}")

    # Shared keywords
    keyword_overlap = overlap_size(row1["keywords_set"], row2["keywords_set"])
    if keyword_overlap >= KEYWORD_OVERLAP_MIN:
        score += WEIGHTS["shared_keywords"]
        reasons.append(f"shared_keywords={keyword_overlap}")

    # Shared genres
    genre_overlap = overlap_size(row1["genres_set"], row2["genres_set"])
    if genre_overlap >= GENRE_OVERLAP_MIN:
        score += WEIGHTS["shared_genres"]
        reasons.append(f"shared_genres={genre_overlap}")

    if score >= FRANCHISE_SCORE_THRESHOLD:
        related_pairs.append({
            "idx1": i,
            "idx2": j,
            "title1": row1["mainTitle"],
            "title2": row2["mainTitle"],
            "cleanTitle1": row1["cleanTitle"],
            "cleanTitle2": row2["cleanTitle"],
            "year1": year1,
            "year2": year2,
            "score": score,
            "reasons": "; ".join(reasons),
        })

pairs_df = pd.DataFrame(related_pairs)

# Safety filter: remove exact same title + same year pairs
if not pairs_df.empty:
    pairs_df = pairs_df[
        ~(
            (pairs_df["cleanTitle1"] == pairs_df["cleanTitle2"])
            & (pairs_df["year1"] == pairs_df["year2"])
            & pairs_df["year1"].notna()
        )
    ].copy()

# =========================
# BUILD GROUPS
# =========================
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


if not pairs_df.empty:
    for _, row in pairs_df.iterrows():
        union(int(row["idx1"]), int(row["idx2"]))

df["franchiseGroup"] = [find(i) for i in df.index]

group_sizes = df.groupby("franchiseGroup").size().to_dict()
df["franchiseSize"] = df["franchiseGroup"].map(group_sizes)

# True if movie appears in a group with 2 or more movies
df["hasSecondMovie"] = df["franchiseSize"] >= 2

# True if movie is not the earliest release in its group
group_min_year = df.groupby("franchiseGroup")["releaseYear_int"].min().to_dict()
df["isLikelyLaterMovie"] = df.apply(
    lambda r: (
        r["hasSecondMovie"]
        and r["releaseYear_int"] is not None
        and group_min_year.get(r["franchiseGroup"]) is not None
        and r["releaseYear_int"] > group_min_year[r["franchiseGroup"]]
    ),
    axis=1
)

# =========================
# SUMMARY
# =========================
total_movies = len(df)
movies_in_multi_movie_groups = int(df["hasSecondMovie"].sum())
later_movies = int(df["isLikelyLaterMovie"].sum())

pct_multi_movie = 100 * movies_in_multi_movie_groups / total_movies if total_movies else 0
pct_later_movies = 100 * later_movies / total_movies if total_movies else 0

print("========== SUMMARY ==========")
print(f"Total movies: {total_movies}")
print(f"Movies with a detected second movie or more: {movies_in_multi_movie_groups}")
print(f"Percentage with a detected second movie or more: {pct_multi_movie:.2f}%")
print(f"Movies that look like later entries in a franchise: {later_movies}")
print(f"Percentage that look like later entries: {pct_later_movies:.2f}%")

if not pairs_df.empty:
    print("\n========== EXAMPLE RELATED PAIRS ==========")
    show_cols = ["title1", "year1", "title2", "year2", "score", "reasons"]
    print(pairs_df.sort_values("score", ascending=False)[show_cols].head(20).to_string(index=False))
else:
    print("\nNo related pairs found with current thresholds.")

# =========================
# READABLE REVIEW TABLE
# =========================
if not pairs_df.empty:
    review = pairs_df.copy()

    # Sort older -> newer for readability
    review["year_from"] = review[["year1", "year2"]].min(axis=1)
    review["year_to"] = review[["year1", "year2"]].max(axis=1)

    review["movie_from"] = review.apply(
        lambda r: r["title1"] if (r["year1"] <= r["year2"]) else r["title2"], axis=1
    )
    review["movie_to"] = review.apply(
        lambda r: r["title2"] if (r["year1"] <= r["year2"]) else r["title1"], axis=1
    )

    review["confidence"] = pd.cut(
        review["score"],
        bins=[-1, 6, 9, 100],
        labels=["Low", "Medium", "High"]
    )

    review_table = review[
        ["movie_from", "year_from", "movie_to", "year_to", "score", "confidence", "reasons"]
    ].sort_values(["confidence", "score", "year_from"], ascending=[False, False, True])

    review_table = review_table.drop_duplicates(
        subset=["movie_from", "year_from", "movie_to", "year_to"]
    )

    print("\n========== READABLE REVIEW (TOP 40) ==========")
    print(review_table.head(40).to_string(index=False))
else:
    review_table = pd.DataFrame()

# =========================
# DETECTED GROUPS (TOP ONLY)
# =========================
print("\n========== DETECTED GROUPS (2+ MOVIES) ==========")
multi_groups = df[df["franchiseSize"] >= 2].sort_values(
    ["franchiseSize", "franchiseGroup", "releaseYear_int"],
    ascending=[False, True, True]
)

if multi_groups.empty:
    print("No groups with 2+ movies.")
else:
    print(f"Total groups: {multi_groups['franchiseGroup'].nunique()}")
    print("Showing top 10 largest groups only:")

    top_group_ids = (
        multi_groups.groupby("franchiseGroup").size().sort_values(ascending=False).head(10).index
    )

    for group_id in top_group_ids:
        group = multi_groups[multi_groups["franchiseGroup"] == group_id].sort_values("releaseYear_int")
        print(f"\nGroup {group_id} | size = {len(group)}")
        print(group[["mainTitle", "releaseYear"]].to_string(index=False))

# =========================
# SAVE OUTPUTS
# =========================
output_dir = root / "Data"
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / "movies_with_sequel_flags.csv", index=False)

if not pairs_df.empty:
    pairs_df.to_csv(output_dir / "detected_related_pairs.csv", index=False)

if not review_table.empty:
    review_table.to_csv(output_dir / "sequel_review_table.csv", index=False)

# =========================
# PLOT
# =========================
summary_df = pd.DataFrame({
    "Category": ["All movies", "Movies in franchise groups", "Likely later movies"],
    "Count": [total_movies, movies_in_multi_movie_groups, later_movies]
})

plt.figure(figsize=(8, 5))
plt.bar(summary_df["Category"], summary_df["Count"])
plt.title("Detected Franchise / Sequel Counts")
plt.ylabel("Number of Movies")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(output_dir / "sequel_summary_chart.png", dpi=300)
plt.show()

print(f"\nSaved movie-level output to: {output_dir / 'movies_with_sequel_flags.csv'}")
if not pairs_df.empty:
    print(f"Saved pair-level output to: {output_dir / 'detected_related_pairs.csv'}")
if not review_table.empty:
    print(f"Saved readable review table to: {output_dir / 'sequel_review_table.csv'}")
print(f"Saved chart to: {output_dir / 'sequel_summary_chart.png'}")

# =========================
# FIRST vs SECOND MOVIE COMPARISON PLOTS
# =========================
comp_dir = output_dir / "first_vs_second_plots"
comp_dir.mkdir(parents=True, exist_ok=True)

# Build pair table with first/second ordering
if not pairs_df.empty:
    p = pairs_df.copy()

    p["first_title"] = p.apply(lambda r: r["title1"] if r["year1"] <= r["year2"] else r["title2"], axis=1)
    p["second_title"] = p.apply(lambda r: r["title2"] if r["year1"] <= r["year2"] else r["title1"], axis=1)
    p["first_year"] = p[["year1", "year2"]].min(axis=1)
    p["second_year"] = p[["year1", "year2"]].max(axis=1)
    p["year_gap"] = p["second_year"] - p["first_year"]

    # Keep one best link per first->second title pair
    p = p.sort_values("score", ascending=False).drop_duplicates(
        subset=["first_title", "second_title", "first_year", "second_year"]
    )

    # If ratings exist in original df, map them in for direct comparison
    # (uses first matching row by normalized title + year)
    if "mainTitle" in df.columns and "releaseYear_int" in df.columns:
        rating_col = None
        for c in ["averageRating", "rating", "imdbRating"]:
            if c in df.columns:
                rating_col = c
                break

        votes_col = None
        for c in ["numVotes", "votes"]:
            if c in df.columns:
                votes_col = c
                break

        if rating_col is not None or votes_col is not None:
            lookup_cols = ["mainTitle", "releaseYear_int"]
            add_cols = []
            if rating_col is not None:
                add_cols.append(rating_col)
            if votes_col is not None:
                add_cols.append(votes_col)

            movie_lookup = (
                df[lookup_cols + add_cols]
                .dropna(subset=["mainTitle", "releaseYear_int"])
                .copy()
            )
            movie_lookup["mainTitle_norm"] = movie_lookup["mainTitle"].astype(str).str.strip().str.lower()

            # first movie merge
            p["first_title_norm"] = p["first_title"].astype(str).str.strip().str.lower()
            p = p.merge(
                movie_lookup.rename(columns={
                    "mainTitle_norm": "first_title_norm",
                    "releaseYear_int": "first_year",
                    **({rating_col: "first_rating"} if rating_col else {}),
                    **({votes_col: "first_votes"} if votes_col else {}),
                })[
                    ["first_title_norm", "first_year"]
                    + (["first_rating"] if rating_col else [])
                    + (["first_votes"] if votes_col else [])
                ],
                on=["first_title_norm", "first_year"],
                how="left"
            )

            # second movie merge
            p["second_title_norm"] = p["second_title"].astype(str).str.strip().str.lower()
            p = p.merge(
                movie_lookup.rename(columns={
                    "mainTitle_norm": "second_title_norm",
                    "releaseYear_int": "second_year",
                    **({rating_col: "second_rating"} if rating_col else {}),
                    **({votes_col: "second_votes"} if votes_col else {}),
                })[
                    ["second_title_norm", "second_year"]
                    + (["second_rating"] if rating_col else [])
                    + (["second_votes"] if votes_col else [])
                ],
                on=["second_title_norm", "second_year"],
                how="left"
            )

    # Save clean comparison table
    keep = ["first_title", "first_year", "second_title", "second_year", "year_gap", "score", "reasons"]
    for c in ["first_rating", "second_rating", "first_votes", "second_votes"]:
        if c in p.columns:
            keep.append(c)

    compare_df = p[keep].copy()
    compare_df.to_csv(output_dir / "first_vs_second_comparison.csv", index=False)

    # -------- Plot 1: Year gap distribution --------
    plt.figure(figsize=(8, 4))
    compare_df["year_gap"].dropna().astype(int).value_counts().sort_index().plot(kind="bar")
    plt.title("First vs Second Movie: Release Year Gap")
    plt.xlabel("Years between first and second movie")
    plt.ylabel("Number of pairs")
    plt.tight_layout()
    plt.savefig(comp_dir / "year_gap_first_vs_second.png", dpi=300)
    plt.close()

    # -------- Plot 2: Rating delta distribution --------
    if {"first_rating", "second_rating"}.issubset(compare_df.columns):
        tmp = compare_df.dropna(subset=["first_rating", "second_rating"]).copy()
        if not tmp.empty:
            tmp["rating_delta"] = tmp["second_rating"] - tmp["first_rating"]

            plt.figure(figsize=(8, 4))
            plt.hist(tmp["rating_delta"], bins=20, edgecolor="black")
            plt.title("First vs Second Movie: Rating Change (Second - First)")
            plt.xlabel("Rating change")
            plt.ylabel("Number of pairs")
            plt.tight_layout()
            plt.savefig(comp_dir / "rating_delta_hist.png", dpi=300)
            plt.close()

            # -------- Plot 3: First vs second rating scatter --------
            plt.figure(figsize=(6, 6))
            plt.scatter(tmp["first_rating"], tmp["second_rating"], alpha=0.6)
            min_r = min(tmp["first_rating"].min(), tmp["second_rating"].min())
            max_r = max(tmp["first_rating"].max(), tmp["second_rating"].max())
            plt.plot([min_r, max_r], [min_r, max_r], linestyle="--")  # equal line
            plt.title("First vs Second Movie Ratings")
            plt.xlabel("First movie rating")
            plt.ylabel("Second movie rating")
            plt.tight_layout()
            plt.savefig(comp_dir / "first_vs_second_rating_scatter.png", dpi=300)
            plt.close()

            # -------- Plot 4: Share improved vs declined --------
            improved = (tmp["rating_delta"] > 0).sum()
            same = (tmp["rating_delta"] == 0).sum()
            declined = (tmp["rating_delta"] < 0).sum()

            plt.figure(figsize=(6, 4))
            plt.bar(["Improved", "Same", "Declined"], [improved, same, declined])
            plt.title("Did the Second Movie Improve?")
            plt.ylabel("Number of pairs")
            plt.tight_layout()
            plt.savefig(comp_dir / "improved_same_declined.png", dpi=300)
            plt.close()

    # -------- Plot 5: Votes change (if available) --------
    if {"first_votes", "second_votes"}.issubset(compare_df.columns):
        tmpv = compare_df.dropna(subset=["first_votes", "second_votes"]).copy()
        if not tmpv.empty:
            tmpv["votes_delta"] = tmpv["second_votes"] - tmpv["first_votes"]
            plt.figure(figsize=(8, 4))
            plt.hist(tmpv["votes_delta"], bins=30, edgecolor="black")
            plt.title("First vs Second Movie: Vote Count Change (Second - First)")
            plt.xlabel("Vote count change")
            plt.ylabel("Number of pairs")
            plt.tight_layout()
            plt.savefig(comp_dir / "votes_delta_hist.png", dpi=300)
            plt.close()

    print(f"\nSaved first-vs-second comparison table to: {output_dir / 'first_vs_second_comparison.csv'}")
    print(f"Saved comparison plots to: {comp_dir}")