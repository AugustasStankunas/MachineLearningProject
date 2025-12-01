import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    df = load_data("data/properties.csv")