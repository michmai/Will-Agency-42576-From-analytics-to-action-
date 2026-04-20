import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["originalTitle"])
    return df