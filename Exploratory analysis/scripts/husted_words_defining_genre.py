import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

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

# Keep only relevant columns
df_word = df[['genres', 'keywords']].copy()

# ---------- 1. Process genres ----------
genres_list = df_word["genres"].fillna("").apply(
    lambda s: [g.strip() for g in s.split(",") if g.strip()]
)

mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(genres_list)

genre_cols = [f"genre_{g}" for g in mlb.classes_]
genre_df = pd.DataFrame(genre_dummies, columns=genre_cols, index=df_word.index)

df_genre = pd.concat([df_word, genre_df], axis=1)

# ---------- 2. Process keywords ----------
# Assumes keywords are comma-separated, like: "space, future, robot"
df_genre["keywords_clean"] = df_genre["keywords"].fillna("").apply(
    lambda s: ",".join([k.strip().lower() for k in s.split(",") if k.strip()])
)

# CountVectorizer using comma as separator
vectorizer = CountVectorizer(
    tokenizer=lambda x: x.split(","),
    preprocessor=lambda x: x,
    token_pattern=None,
    binary=True,        # presence/absence of keyword
    min_df=5            # ignore very rare keywords
)

X_keywords = vectorizer.fit_transform(df_genre["keywords_clean"])
keyword_names = vectorizer.get_feature_names_out()

# ---------- 3. Find most defining keywords per genre ----------
top_n = 15
genre_keyword_results = {}

for genre_col in genre_cols:
    y = df_genre[genre_col]

    # Skip genres with too few positive examples
    if y.sum() < 5:
        continue

    chi_scores, p_values = chi2(X_keywords, y)

    result_df = pd.DataFrame({
        "keyword": keyword_names,
        "chi2_score": chi_scores,
        "p_value": p_values
    }).sort_values("chi2_score", ascending=False)

    genre_keyword_results[genre_col] = result_df.head(top_n)

# ---------- 4. Print results ----------
for genre, result in genre_keyword_results.items():
    print(f"\nTop keywords for {genre}:")
    print(result.to_string(index=False))