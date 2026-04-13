# pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# ---------------------------------------------------------
print("Processing data...")
file_path = r'C:\Users\andre\OneDrive - Danmarks Tekniske Universitet\DTU\Industriel økonomi og Teknologiledelse\3. Semester\42576 Analytics to Action\Codebase GitHub\Will-Agency-42576-From-analytics-to-action--main\Data\Movie_50k.csv'
df = pd.read_csv(file_path)

# Keep our core metrics and the plot description
columns_to_keep = [
    'originalTitle', 'imdbRating', 'numberOfVotes', 
    'plotLong', 'topFiveActors', 'keywords', 
    'production', 'genres', 'firstLanguage'
]
# Drop rows that are missing any of this critical information
df = df[columns_to_keep].dropna().reset_index(drop=True)

# ---------------------------------------------------------
# 2. EXTRACT WORDS FROM PLOTS (Anthropological themes)
# ---------------------------------------------------------
print("Processing plot words...")
# Extract the top 2000 most meaningful words across all plots
vectorizer = CountVectorizer(stop_words='english', max_features=2000)
word_matrix = vectorizer.fit_transform(df['plotLong'])
extracted_words = vectorizer.get_feature_names_out()

# Create a DataFrame for words (prefix 'word_')
word_df = pd.DataFrame(word_matrix.toarray(), columns=["word_" + word for word in extracted_words])

# ---------------------------------------------------------
# 3. PROCESS CATEGORIES (Actors, Genres, Keywords, etc.)
# ---------------------------------------------------------
print("Processing categories...")

# Helper function to find the most common items in a comma-separated column 
# and create True/False (1/0) columns for them.
def get_top_category_columns(dataframe, column_name, top_n=10):
    # Split by comma, clean spaces, and find the top N most frequent items
    all_items = dataframe[column_name].astype(str).str.split(',').explode().str.strip()
    top_items = all_items.value_counts().head(top_n).index.tolist()
    
    # Create a temporary dataframe to hold our new 1/0 columns
    temp_df = pd.DataFrame()
    for item in top_items:
        # Create a clean column name (e.g., 'genre_Drama')
        col_name = f"{column_name}_{item}".replace(" ", "_").replace("'", "")
        # Mark 1 if the movie has this item, 0 if it doesn't
        temp_df[col_name] = dataframe[column_name].astype(str).str.contains(item, na=False, regex=False).astype(int)
    
    return temp_df

# Extract the top 10 from each category
genre_df = get_top_category_columns(df, 'genres', top_n=10)
language_df = get_top_category_columns(df, 'firstLanguage', top_n=10)
production_df = get_top_category_columns(df, 'production', top_n=10)
keywords_df = get_top_category_columns(df, 'keywords', top_n=10)
actors_df = get_top_category_columns(df, 'topFiveActors', top_n=10)

# ---------------------------------------------------------
# 4. COMBINE EVERYTHING AND CALCULATE CORRELATIONS
# ---------------------------------------------------------
print("Calculating correlations...\n")
# Stitch the original numerical data, the word data, and all the category data together
df_master = pd.concat([df[['imdbRating', 'numberOfVotes']], word_df, genre_df, language_df, production_df, keywords_df, actors_df], axis=1)

# Run the massive correlation matrix
correlation_matrix = df_master.corr()

# ---------------------------------------------------------
# 5. DISPLAY RESULTS
# ---------------------------------------------------------
def print_top_correlations(target_column, title):
    print(f"=== WHAT DRIVES {title} ===")
    
    # Get correlations for the target (exclude the target itself and the other metric)
    corrs = correlation_matrix[target_column].drop(['imdbRating', 'numberOfVotes'])
    
    print("POSITIVE DRIVERS (Presence increases score):")
    print(corrs.sort_values(ascending=False).head(10)) # Top 10 positive
    
    print("\nNEGATIVE DRIVERS (Presence decreases score):")
    print(corrs.sort_values(ascending=True).head(10))  # Top 10 negative
    print("-" * 50 + "\n")

# Print results for both of our key acquisition metrics
print_top_correlations('imdbRating', 'EXPECTED SATISFACTION (Ratings)')
print_top_correlations('numberOfVotes', 'BROAD APPEAL (Votes)')

# ---------------------------------------------------------
# 6. PLOTTING SETUP
# ---------------------------------------------------------
print("Generating visualizations...")
# Set the visual style for our plots
sns.set_theme(style="whitegrid")

# Helper function to plot Diverging Bar Charts
def plot_diverging_bars(target_col, title, filename):
    # Get correlations, drop the core metrics, and sort them
    corrs = correlation_matrix[target_col].drop(['imdbRating', 'numberOfVotes'])
    
    # Get the top 10 positive and top 10 negative correlations
    top_positive = corrs.sort_values(ascending=False).head(10)
    top_negative = corrs.sort_values(ascending=True).head(10)
    
    # Combine them into one series
    combined_corrs = pd.concat([top_positive, top_negative]).sort_values()
    
    # Create the plot
    plt.subplots(figsize=(10, 8))
    
    # Create colors: Green for positive, Red for negative
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in combined_corrs.values]
    
    # Draw the barplot
    sns.barplot(x=combined_corrs.values, y=combined_corrs.index, palette=colors)
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Movie Features (Words, Genres, etc.)', fontsize=12)
    
    # Add a vertical line at 0 for clarity
    plt.axvline(0, color='black', linewidth=1)
    
    # Adjust layout so labels aren't cut off, and save
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # Close to prevent overlapping with the next plot

# ---------------------------------------------------------
# 7. GENERATE THE PLOTS
# ---------------------------------------------------------

# Plot 1: What drives Expected Satisfaction?
plot_diverging_bars('imdbRating', 
                    'Top Drivers of Expected Satisfaction (Ratings)', 
                    'satisfaction_drivers.png')

# Plot 2: What drives Broad Appeal?
plot_diverging_bars('numberOfVotes', 
                    'Top Drivers of Broad Appeal (Votes)', 
                    'appeal_drivers.png')

# Plot 3: The "Golden Goose" Heatmap
# Let's look at how the absolute top 15 features interact with BOTH metrics
plt.subplots(figsize=(10, 10))

# Find features that have the strongest absolute correlation with either metric
all_corrs = correlation_matrix[['imdbRating', 'numberOfVotes']].drop(['imdbRating', 'numberOfVotes'])
all_corrs['max_impact'] = all_corrs.abs().max(axis=1)
top_features_for_heatmap = all_corrs.sort_values(by='max_impact', ascending=False).head(15)

# Drop the helper column and plot the heatmap
heatmap_data = top_features_for_heatmap[['imdbRating', 'numberOfVotes']]
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)

plt.title('Feature Impact on Both Acquisition Goals', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('acquisition_heatmap.png')
plt.close()

print("Success! Check your project folder for the 3 generated PNG images.")