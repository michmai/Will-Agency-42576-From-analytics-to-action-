########## ---------- import ---------- ##########
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import geopandas as gpd
import matplotlib.pyplot as plt
import pycountry


########## ---------- Load data ---------- ##########
data_path = "../../Data/European_data_2000.csv"
county_path = r"/home/husted42/Downloads/110m_cultural/ne_110m_admin_0_countries.shp"

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

# Calculate total votes per country
country_votes = (
    df.groupby("mainCountry", as_index=False)["numberOfVotes"]
      .sum()
      .rename(columns={"mainCountry": "iso2", "numberOfVotes": "totalVotes"})
)

# European country populations (2023 estimates in millions)
population_data = {
    "AT": 9.1,   # Austria
    "BE": 11.6,  # Belgium
    "BG": 6.9,   # Bulgaria
    "HR": 3.9,   # Croatia
    "CY": 1.2,   # Cyprus
    "CZ": 10.5,  # Czech Republic
    "DK": 5.9,   # Denmark
    "EE": 1.4,   # Estonia
    "FI": 5.6,   # Finland
    "FR": 68.0,  # France
    "DE": 83.4,  # Germany
    "GR": 10.4,  # Greece
    "HU": 9.7,   # Hungary
    "IE": 5.2,   # Ireland
    "IT": 57.5,  # Italy
    "LV": 1.9,   # Latvia
    "LT": 2.8,   # Lithuania
    "LU": 0.7,   # Luxembourg
    "MT": 0.5,   # Malta
    "NL": 17.8,  # Netherlands
    "PL": 36.7,  # Poland
    "PT": 10.5,  # Portugal
    "RO": 19.1,  # Romania
    "SK": 5.5,   # Slovakia
    "SI": 2.1,   # Slovenia
    "ES": 47.6,  # Spain
    "SE": 10.6,  # Sweden
    "CH": 8.8,   # Switzerland
    "GB": 68.5,  # United Kingdom
    "NO": 5.5,   # Norway
    "IS": 0.4,   # Iceland
    "UA": 38.0,  # Ukraine
    "RS": 6.6,   # Serbia
    "ME": 0.6,   # Montenegro
    "MK": 2.1,   # North Macedonia
    "AL": 2.9,   # Albania
    "BA": 3.2,   # Bosnia and Herzegovina
    "MD": 2.6,   # Moldova
}

country_votes["population"] = country_votes["iso2"].map(population_data)

# Calculate votes per capita (votes per person in millions for better scale)
country_votes["votes_per_capita"] = country_votes["totalVotes"] / (country_votes["population"] * 1000000)
country_votes = country_votes.dropna(subset=["votes_per_capita"])

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
    county_path
)

# Keep Europe only
europe = world[world["CONTINENT"].str.contains("Europe", na=False)].copy()

# Merge ratings
europe = europe.merge(country_votes, left_on="ADM0_A3", right_on="iso3", how="left")
europe = europe.dropna(subset=["votes_per_capita"])
europe = europe.explode(index_parts=False)

# Keep largest geometry per country
europe_proj = europe.to_crs(epsg=3035)  # Reproject to projected CRS for accurate area calculation
europe_proj["area"] = europe_proj.geometry.area
europe["area"] = europe_proj["area"]
europe = europe.sort_values("area", ascending=False)
europe = europe.drop_duplicates(subset="ADM0_A3")

europe = europe.drop(columns="area")

# print total votes per country for verification
print(country_votes[["iso2", "totalVotes"]])



# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 10))

# Convert to votes per capita per 1000 people for better readability
europe["votes_per_capita_k"] = europe["votes_per_capita"] * 1000

# compute vmin/vmax for the color scale
vmin = europe["votes_per_capita_k"].min()
vmax = europe["votes_per_capita_k"].max()

europe.plot(
    column="votes_per_capita_k",
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
    if pd.notnull(row["votes_per_capita_k"]):
        ax.text(
            row["centroid"].x,
            row["centroid"].y,
            f"{row['NAME']}\n{row['votes_per_capita_k']:.1f}",
            fontsize=10,
            color="black",
            ha="center",
            va="center"
        )

ax.set_title("IMDb Votes per Capita (per 1000 people) by Country (Europe)", fontsize=14)
ax.axis("off")

plt.tight_layout()

# save the plot as png
fig.savefig("../assets/map_votes_per_capita_by_country.png", dpi=300)