########## ---------- 
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

print("Columns in the dataset: ")
print(df.columns)
print("\nShape of the dataset: ", df.shape)
print("Unique titles ", df['titleId'].nunique())
print("Duplicate titles ", df['titleId'].duplicated().sum())
print("Missing values: ", df.isnull().sum().sum())


# Plot the average rating by genre
plt.figure(figsize=(12, 8))
df_genre = df.groupby('mainCountry')['imdbRating'].mean().sort_values(ascending=False)
df_genre.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Average Rating by country')
plt.xlabel('Country')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.ylim(5, 8)
plt.grid(axis='y', alpha=0.75)
plt.show()