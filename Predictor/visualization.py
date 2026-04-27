import matplotlib.pyplot as plt

def plot_franchise_sequence(df, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")

    if len(group) < 2:
        print("Not enough movies.")
        return

    print("\n🎬 Franchise sequence:\n")
    for i, row in group.iterrows():
        print(f"{int(row['releaseYear'])} - {row['originalTitle']} ({row['imdbRating']})")

    plt.figure(figsize=(10, 5))

    plt.plot(group["releaseYear"], group["imdbRating"], marker="o")

    # label each movie
    for _, row in group.iterrows():
        plt.text(
            row["releaseYear"],
            row["imdbRating"],
            row["originalTitle"],
            fontsize=8,
            rotation=30
        )

    plt.xlabel("Release Year")
    plt.ylabel("IMDb Rating")
    plt.title(f"Franchise Evolution (ID {franchise_id})")

    plt.grid()
    plt.tight_layout()
    plt.show()
    
def plot_top_franchises(df, n=3):
    top_ids = df.groupby("franchise_id").size().sort_values(ascending=False).head(n).index

    for fid in top_ids:
        plot_franchise_sequence(df, fid)
        
def plot_franchise_clean(df, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")

    plt.figure(figsize=(10, 5))

    plt.plot(group["releaseYear"], group["imdbRating"], marker="o")

    plt.xlabel("Year")
    plt.ylabel("Rating")
    plt.title(f"Franchise Evolution (ID {franchise_id})")

    plt.grid()
    plt.show()