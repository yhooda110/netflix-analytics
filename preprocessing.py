import pandas as pd

def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # Fill missing values
    df.fillna("Unknown", inplace=True)

    return df
