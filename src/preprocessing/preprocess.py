import pandas as pd

def preprocess_stats(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=['player'])
    # convert stats to numeric where possible
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True)
    df.to_csv("../data/processed/premier_league_clean.csv", index=False)
    print("✅ Cleaned data saved.")

if __name__ == "__main__":
    preprocess_stats("../data/raw/premier_league_stats.csv")
