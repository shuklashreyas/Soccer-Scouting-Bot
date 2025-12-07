import pandas as pd

# Simple script to normalize and persist cleaned league stats
def preprocess_stats(path):
    """Read a CSV, normalize column names, coerce numeric stats, and save cleaned CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=['player'])

    # Coerce remaining columns (stat columns) to numeric where possible
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0, inplace=True)
    df.to_csv("../data/processed/premier_league_clean.csv", index=False)
    print("✅ Cleaned data saved.")


if __name__ == "__main__":
    preprocess_stats("../data/raw/premier_league_stats.csv")
