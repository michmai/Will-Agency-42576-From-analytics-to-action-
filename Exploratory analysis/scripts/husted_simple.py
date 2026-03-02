
########## ---------- import ---------- ##########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer


########## ---------- Load data ---------- ##########
data_path = "../../data/European_data_2000.csv"

df = pd.read_csv(
    data_path,
    sep=",",
    engine="python",
    quotechar='"',
    on_bad_lines="skip"   
)

########## ---------- Average rating by country ---------- ##########
plt.figure(figsize=(12, 8))
df_genre = df.groupby('mainCountry')['imdbRating'].mean().sort_values(ascending=False)
df_genre.plot(kind='bar', color='orange', edgecolor='black')

# overall average rating
overall_avg = df['imdbRating'].mean()
plt.axhline(overall_avg, color='red', linestyle='--',
            label=f'Overall avg ({overall_avg:.2f})')

plt.title('Average Rating by country')
plt.xlabel('Country')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.ylim(5, 8)
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.savefig("../assets/average_rating_by_country.png", dpi=300)


########## ---------- English vs Non-English Movies ---------- ##########
'''
    Hypothesis: The average number of votes are higher for movies in english than for movies in other languages.
'''

df['isEnglish'] = df['firstLanguage'] == 'en'

# plot of votes
english_movies = df.groupby('isEnglish')['numberOfVotes'].mean()
english_movies = english_movies.reindex([True, False])

plt.figure(figsize=(6, 6))
english_movies.plot(kind='bar', color=['orange', 'blue'], edgecolor='black')
plt.title('English vs Non-English Movies')
plt.xlabel('Is English')
plt.ylabel('Average number of votes')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.savefig("../assets/english_vs_non_english_movies_votes.png", dpi=300)


# plot of ratings
english_movies = df.groupby('isEnglish')['imdbRating'].mean()
english_movies = english_movies.reindex([True, False])

plt.figure(figsize=(6, 6))
english_movies.plot(kind='bar', color=['orange', 'blue'], edgecolor='black')
plt.title('Average Rating: English vs Non-English Movies')
plt.xlabel('Is English')
plt.ylabel('Average Rating')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.savefig("../assets/english_vs_non_english_movies_ratings.png", dpi=300)


########## ---------- votes by genre ---------- ##########
# one-hot encoding for each genre
genres_list = df["genres"].fillna("").apply(
    lambda s: [g.strip() for g in s.split(",") if g.strip()]
)
mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(genres_list)

# Turn encoding into a DataFrame and concatenate with original DataFrame
genre_cols = [f"genre_{g}" for g in mlb.classes_]
genre_df = pd.DataFrame(genre_dummies, columns=genre_cols, index=df.index)
df_genre = pd.concat([df, genre_df], axis=1)

# Calculate average rating for each genre and overall average
column_value = "numberOfVotes"
overall_avg = df_genre[column_value].mean()
genre_avg = df_genre[genre_cols].multiply(df_genre[column_value], axis=0).sum() / df_genre[genre_cols].sum()
genre_avg = genre_avg.sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 5))
plt.bar(genre_avg.index.str.replace("genre_", ""), genre_avg.values)
plt.axhline(overall_avg, linestyle="--")
plt.xticks(rotation=90)
plt.xlabel("Genre")
plt.ylabel("Average number of votes")
plt.title("Average Movie Votes by Genre")
plt.tight_layout()
plt.savefig("../assets/average_votes_by_genre.png", dpi=300)

print("Overall Average Number of Votes:", overall_avg)

########## ---------- rating by genre ---------- ##########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# one-hot encoding for each genre
genres_list = df["genres"].fillna("").apply(
    lambda s: [g.strip() for g in s.split(",") if g.strip()]
)
mlb = MultiLabelBinarizer()
genre_dummies = mlb.fit_transform(genres_list)

# Turn encoding into a DataFrame and concatenate with original DataFrame
genre_cols = [f"genre_{g}" for g in mlb.classes_]
genre_df = pd.DataFrame(genre_dummies, columns=genre_cols, index=df.index)
df_genre = pd.concat([df, genre_df], axis=1)

# Calculate average rating for each genre and overall average
column_value = "imdbRating"
overall_avg = df_genre[column_value].mean()
genre_avg = df_genre[genre_cols].multiply(df_genre[column_value], axis=0).sum() / df_genre[genre_cols].sum()
genre_avg = genre_avg.sort_values(ascending=False)

# Plot
plt.figure(figsize=(12, 5))
plt.bar(genre_avg.index.str.replace("genre_", ""), genre_avg.values)
plt.axhline(overall_avg, linestyle="--")
plt.xticks(rotation=90)
plt.xlabel("Genre")
plt.ylabel("Average IMDb rating")
plt.title("Average Movie Rating by Genre")
plt.tight_layout()
plt.savefig("../assets/average_rating_by_genre.png", dpi=300)

print("Overall Average Rating:", overall_avg)