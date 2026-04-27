import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8")

def plot_franchise_sequence(df, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")

    plt.figure(figsize=(12, 6))

    plt.plot(
        group["releaseYear"],
        group["imdbRating"],
        marker="o",
        linewidth=3
    )

    plt.scatter(
        group["releaseYear"],
        group["imdbRating"],
        s=80,
        zorder=3
    )

    for _, row in group.iterrows():
        plt.annotate(
            row["originalTitle"],
            (row["releaseYear"], row["imdbRating"]),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=9
        )

    plt.title("Franchise Rating Over Time", fontsize=18, weight="bold")
    plt.xlabel("Year", fontsize=13)
    plt.ylabel("IMDb Rating", fontsize=13)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    movie = group["originalTitle"].iloc[-1].replace(" ", "_").replace("/", "_")
    plt.savefig(f"outputs/franchise_{movie}.png", dpi=300)
    plt.show()
    
def plot_top_franchises(df, n=10):
    top_ids = df.groupby("franchise_id").size().sort_values(ascending=False).head(n).index

    for fid in top_ids:
        plot_franchise_sequence(df, fid)
        
def plot_franchise_clean(df, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")

    plt.figure(figsize=(10, 5))

    plt.plot(group["releaseYear"], group["imdbRating"], marker="o")

    plt.xlabel("Year")
    plt.ylabel("Rating")
    movie = group["originalTitle"].iloc[-1].replace(" ", "_").replace("/", "_")
    plt.title(f"Franchise Evolution ({movie})")

    plt.grid()
    plt.show()

def plot_feature_importance(model, feature_names):
    import pandas as pd

    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))

    plt.barh(feat_imp["feature"], feat_imp["importance"])

    plt.title("Feature Importance", fontsize=18, weight="bold")
    plt.xlabel("Importance", fontsize=12)

    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=300) 
    plt.show()

import numpy as np

def plot_with_trend(df, franchise_id):
    group = df[df["franchise_id"] == franchise_id].sort_values("releaseYear")

    x = np.arange(len(group))
    y = group["imdbRating"]

    z = np.polyfit(x, y, 1)
    trend = np.poly1d(z)

    plt.figure(figsize=(10,5))
    plt.plot(group["releaseYear"], y, marker="o", label="Actual")
    plt.plot(group["releaseYear"], trend(x), linestyle="--", label="Trend")

    plt.title("Franchise Trend (Improving or Declining)")
    plt.legend()
    #fetch the name of the last movie in the franchise for naming the file
    movie = group["originalTitle"].iloc[-1].replace(" ", "_").replace("/", "_")
    plt.savefig(f"outputs/franchise_{movie}_trend.png", dpi=300)
    plt.show()

    
