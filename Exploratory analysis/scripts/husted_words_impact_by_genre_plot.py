data_path = "/home/husted42/Downloads/European_data_2000.csv"

# load the data
import pandas as pd
df = pd.read_csv(
    data_path,
    sep=",",
    engine="python",
    quotechar='"',
    on_bad_lines="skip"   
)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------------
# 1. Keep relevant columns
# -----------------------------------
df_text = df[['genres', 'plotMedium', 'imdbRating']].copy()
df_text = df_text.dropna(subset=['genres', 'plotMedium', 'imdbRating']).copy()
df_text["plot_clean"] = df_text["plotMedium"].astype(str)

# -----------------------------------
# 2. Process genres
# -----------------------------------
genres_list = df_text["genres"].fillna("").apply(
    lambda s: [g.strip() for g in s.split(",") if g.strip()]
)

mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(genres_list)

genre_names = list(mlb.classes_)
genre_df = pd.DataFrame(
    genre_dummies,
    columns=genre_names,
    index=df_text.index
)

df_genre = pd.concat([df_text, genre_df], axis=1)

# -----------------------------------
# 3. Vectorize text as binary presence
# -----------------------------------
vectorizer = CountVectorizer(
    stop_words="english",
    lowercase=True,
    min_df=10,
    max_df=0.8,
    ngram_range=(1, 2),
    binary=True
)

X = vectorizer.fit_transform(df_genre["plot_clean"])
terms = vectorizer.get_feature_names_out()

term_df = pd.DataFrame.sparse.from_spmatrix(
    X,
    index=df_genre.index,
    columns=terms
)

# -----------------------------------
# 4. Compute rating difference per term within each genre
# -----------------------------------
top_n = 20
min_movies_genre = 30
min_term_count_in_genre = 5

genre_term_results = {}

for genre in genre_names:
    genre_mask = df_genre[genre] == 1
    n_genre_movies = int(genre_mask.sum())

    if n_genre_movies < min_movies_genre:
        continue

    genre_subset = df_genre.loc[genre_mask]
    genre_term_subset = term_df.loc[genre_mask]

    results = []

    for term in terms:
        mask = genre_term_subset[term] > 0
        n_with_term = int(mask.sum())
        n_without_term = int((~mask).sum())

        # skip rare terms inside this genre
        if n_with_term < min_term_count_in_genre or n_without_term < min_term_count_in_genre:
            continue

        avg_rating_with_term = genre_subset.loc[mask, "imdbRating"].mean()
        avg_rating_without_term = genre_subset.loc[~mask, "imdbRating"].mean()
        diff = avg_rating_with_term - avg_rating_without_term

        results.append({
            "genre": genre,
            "term": term,
            "n_genre_movies": n_genre_movies,
            "n_with_term": n_with_term,
            "n_without_term": n_without_term,
            "avg_rating_with_term": avg_rating_with_term,
            "avg_rating_without_term": avg_rating_without_term,
            "rating_difference": diff
        })

    if results:
        result_df = pd.DataFrame(results).sort_values(
            "rating_difference", ascending=False
        )
        genre_term_results[genre] = result_df.head(top_n)

# -----------------------------------
# 5. Print results
# -----------------------------------
for genre, result in genre_term_results.items():import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------------
# 1. Keep relevant columns
# -----------------------------------
df_text = df[['genres', 'plotMedium', 'imdbRating']].copy()
df_text = df_text.dropna(subset=['genres', 'plotMedium', 'imdbRating']).copy()
df_text["plot_clean"] = df_text["plotMedium"].astype(str)

# -----------------------------------
# 2. Process genres
# -----------------------------------
genres_list = df_text["genres"].fillna("").apply(
    lambda s: [g.strip() for g in s.split(",") if g.strip()]
)

mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(genres_list)

genre_names = list(mlb.classes_)
genre_df = pd.DataFrame(
    genre_dummies,
    columns=genre_names,
    index=df_text.index
)

df_genre = pd.concat([df_text, genre_df], axis=1)

# -----------------------------------
# 3. Vectorize text as binary presence
# -----------------------------------
vectorizer = CountVectorizer(
    stop_words="english",
    lowercase=True,
    min_df=10,
    max_df=0.8,
    ngram_range=(1, 2),
    binary=True
)

X = vectorizer.fit_transform(df_genre["plot_clean"])
terms = vectorizer.get_feature_names_out()

term_df = pd.DataFrame.sparse.from_spmatrix(
    X,
    index=df_genre.index,
    columns=terms
)

# -----------------------------------
# 4. Compute rating difference per term within each genre
# -----------------------------------
top_n = 20
min_movies_genre = 30
min_term_count_in_genre = 5

genre_term_results = {}

for genre in genre_names:
    genre_mask = df_genre[genre] == 1
    n_genre_movies = int(genre_mask.sum())

    if n_genre_movies < min_movies_genre:
        continue

    genre_subset = df_genre.loc[genre_mask]
    genre_term_subset = term_df.loc[genre_mask]

    results = []

    for term in terms:
        mask = genre_term_subset[term] > 0
        n_with_term = int(mask.sum())
        n_without_term = int((~mask).sum())

        # skip rare terms inside this genre
        if n_with_term < min_term_count_in_genre or n_without_term < min_term_count_in_genre:
            continue

        avg_rating_with_term = genre_subset.loc[mask, "imdbRating"].mean()
        avg_rating_without_term = genre_subset.loc[~mask, "imdbRating"].mean()
        diff = avg_rating_with_term - avg_rating_without_term

        results.append({
            "genre": genre,
            "term": term,
            "n_genre_movies": n_genre_movies,
            "n_with_term": n_with_term,
            "n_without_term": n_without_term,
            "avg_rating_with_term": avg_rating_with_term,
            "avg_rating_without_term": avg_rating_without_term,
            "rating_difference": diff
        })

    if results:
        result_df = pd.DataFrame(results).sort_values(
            "rating_difference", ascending=False
        )
        genre_term_results[genre] = result_df.head(top_n)

# -----------------------------------
# 5. Print results
# -----------------------------------
for genre, result in genre_term_results.items():
    print(f"\nTop terms linked to higher IMDb rating within genre: {genre}")
    print(result.to_string(index=False))
    print(f"\nTop terms linked to higher IMDb rating within genre: {genre}")
    print(result.to_string(index=False))

all_results = pd.concat(genre_term_results.values(), ignore_index=True)
all_results = all_results.sort_values(
    ["genre", "rating_difference"], ascending=[True, False]
)
print(all_results.to_string(index=False))

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# -----------------------------
# 1. Take top terms per genre
# -----------------------------
top_per_genre = 10
plot_df = (
    all_results.sort_values(["genre", "rating_difference"], ascending=[True, False])
    .groupby("genre")
    .head(top_per_genre)
    .copy()
)

# -----------------------------
# 2. Convert genre names to numeric y positions
# -----------------------------
genres = sorted(plot_df["genre"].unique())
genre_to_y = {genre: i for i, genre in enumerate(genres)}
plot_df["y"] = plot_df["genre"].map(genre_to_y)

# Add a little vertical jitter so points/labels do not start at exactly same place
np.random.seed(42)
plot_df["y_jitter"] = plot_df["y"] + np.random.uniform(-0.15, 0.15, size=len(plot_df))

# -----------------------------
# 3. Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(16, 10))

ax.scatter(
    plot_df["rating_difference"],
    plot_df["y_jitter"],
    s=plot_df["n_with_term"] * 8,
    alpha=0.7
)

# Horizontal guide lines
for genre, y in genre_to_y.items():
    ax.axhline(y=y, linestyle="--", linewidth=0.6, alpha=0.4)

# Vertical reference line
ax.axvline(0, linestyle="--", linewidth=1)

# -----------------------------
# 4. Add labels
# -----------------------------
texts = []
for _, row in plot_df.iterrows():
    texts.append(
        ax.text(
            row["rating_difference"],
            row["y_jitter"],
            row["term"],
            fontsize=9
        )
    )

adjust_text(
    texts,
    ax=ax,
    expand=(1.2, 1.4),
    force_text=(0.8, 1.2),
    force_points=(0.5, 0.8),
    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6)
)

# -----------------------------
# 5. Format axes
# -----------------------------
ax.set_yticks(list(genre_to_y.values()))
ax.set_yticklabels(list(genre_to_y.keys()))

ax.set_xlabel("Rating difference (movies with term − without term)")
ax.set_ylabel("Genre")
ax.set_title("Words associated with higher IMDb ratings within each genre")

plt.tight_layout()
plt.show()

# save the plot as png
fig.savefig("../assets/word_impact_by_genre.png", dpi=300)