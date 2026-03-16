import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack


# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------

df = pd.read_csv("/Users/michellemai/Documents/GitHub/Will-Agency-42576-From-analytics-to-action-/Data/European_data_2000.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)


# ---------------------------------------------------
# 2. Select Features
# ---------------------------------------------------

features = [
    "releaseYear",
    "runtimeMinutes",
    "genres",
    "mainCountry",
    "plotMedium"
]

target = "imdbRating"

df = df[["originalTitle"] + features + [target]].dropna()

titles = df["originalTitle"]


# ---------------------------------------------------
# 3. Separate Text and Structured Data
# ---------------------------------------------------

text_data = df["plotMedium"]
structured_df = df.drop(["plotMedium", "originalTitle"], axis=1)


# ---------------------------------------------------
# 4. Encode Categorical Variables
# ---------------------------------------------------

structured_encoded = pd.get_dummies(
    structured_df,
    columns=["genres", "mainCountry"]
)


# ---------------------------------------------------
# 5. Convert Plot Text to TF-IDF
# ---------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

plot_features = tfidf.fit_transform(text_data)


# ---------------------------------------------------
# 6. Combine Structured + Text Features
# ---------------------------------------------------

X_structured = structured_encoded.drop("imdbRating", axis=1)

# convert all columns to numeric
X_structured = X_structured.astype(float)

X = hstack([X_structured, plot_features])

y = structured_encoded["imdbRating"]


# ---------------------------------------------------
# 7. Train/Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test, title_train, title_test = train_test_split(
    X,
    y,
    titles,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# 8. Train Model
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# 9. Evaluate Model
# ---------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)


# ---------------------------------------------------
# 10. Predict Rating for One Movie
# ---------------------------------------------------

index = 0

sample = X_test[index]
movie_title = title_test.iloc[index]

predicted_rating = model.predict(sample.reshape(1, -1))
actual_rating = y_test.iloc[index]

print("Movie:", movie_title)
print("Actual rating:", actual_rating)
print("Predicted rating:", predicted_rating[0])

# for i in range(10):
#     print("Movie:", title_test.iloc[i])
#     print("Actual:", y_test.iloc[i])
#     print("Predicted:", predictions[i])
#     print("-----")

# ---------------------------------------------------
# 11. Feature Importance (Structured Features Only)
# ---------------------------------------------------

importances = model.feature_importances_[:len(X_structured.columns)]

feature_importance = pd.Series(
    importances,
    index=X_structured.columns
)

top_features = feature_importance.nlargest(10)

plt.figure()
top_features.plot(kind="barh")

plt.title("Top Features Influencing IMDb Rating")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ---------------------------------------------------
# 12. Predicted vs Actual Ratings Plot
# ---------------------------------------------------

plt.figure()

plt.scatter(y_test, predictions, alpha=0.5)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.title("Predicted vs Actual IMDb Ratings")

plt.show()