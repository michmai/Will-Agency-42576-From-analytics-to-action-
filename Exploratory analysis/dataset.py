import pandas as pd
import matplotlib.pyplot as plt

########## ---------- Load data ---------- ##########
# Specify path
data_path = "../data/European_data_2000.csv"

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

print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\nDescriptive statistics (numeric columns):")
print(df.describe().round(2))
