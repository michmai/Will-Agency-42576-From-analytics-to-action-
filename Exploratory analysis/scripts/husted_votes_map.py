########## ---------- import ---------- ##########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import geopandas as gpd
import matplotlib.pyplot as plt
import pycountry


########## ---------- Load data ---------- ##########
data_path = "../../data/European_data_2000.csv"

df = pd.read_csv(
    data_path,
    sep=",",
    engine="python",
    quotechar='"',
    on_bad_lines="skip"   
)


# -------------------------
# Prepare rating data
# -------------------------
df["numberOfVotes"] = pd.to_numeric(df["numberOfVotes"], errors="coerce")

country_votes = (
    df.groupby("mainCountry", as_index=False)["numberOfVotes"]
      .mean()
      .rename(columns={"mainCountry": "iso2", "numberOfVotes": "avgVotes"})
)

def iso2_to_iso3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None

country_votes["iso3"] = country_votes["iso2"].apply(iso2_to_iso3)

# -------------------------
# Load shapefile (CHANGE PATH)
# -------------------------
world = gpd.read_file(
    r"C:\Users\huste\Downloads\110m_cultural\ne_110m_admin_0_countries.shp"
)

# Keep Europe only
europe = world[world["CONTINENT"].str.contains("Europe", na=False)].copy()

# Merge ratings
europe = europe.merge(country_votes, left_on="ADM0_A3", right_on="iso3", how="left")
europe = europe.dropna(subset=["avgVotes"])
europe = europe.explode(index_parts=False)

# Keep largest geometry per country
europe["area"] = europe.geometry.area
europe = europe.sort_values("area", ascending=False)
europe = europe.drop_duplicates(subset="ADM0_A3")

europe = europe.drop(columns="area")


# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 10))

europe["avgVotesK"] = europe["avgVotes"] / 1000.0

# compute vmin/vmax for the color scale (in thousands)
vmin = europe["avgVotesK"].min()
vmax = europe["avgVotesK"].max()

europe.plot(
    column="avgVotesK",
    cmap="RdYlBu",
    linewidth=0.5,
    edgecolor="black",
    legend=True,
    vmin=vmin,
    vmax=vmax,
    missing_kwds={
        "color": "lightgrey",
        "label": "No data"
    },
    ax=ax
)

# Create label positions (centroids)
europe_proj = europe.to_crs(epsg=3035)  # Europe projection
europe["centroid"] = europe_proj.geometry.centroid.to_crs(europe.crs)


for idx, row in europe.iterrows():
    if pd.notnull(row["avgVotesK"]):
        ax.text(
            row["centroid"].x,
            row["centroid"].y,
            f"{row['NAME']}\n{row['avgVotesK']:.2f}k",
            fontsize=10,
            color="black",
            ha="center",
            va="center"
        )

ax.set_title("Average IMDb Votes (thousands) by Country (Europe)", fontsize=14)
ax.axis("off")

plt.tight_layout()

# save the plot as png
fig.savefig("../assets/map_average_votes_by_country.png", dpi=300)