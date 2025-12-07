import pandas as pd
import os

# Script: clean raw FBref CSV into a compact, typed `all_leagues_clean.csv`
input_path = "data/all_leagues_stats.csv"
output_path = "data/processed/all_leagues_clean.csv"

df = pd.read_csv(input_path)

# Normalize column names
df.columns = (
    df.columns
    .str.replace("Unnamed: \\d+_level_0_", "", regex=True)
    .str.strip()
    .str.replace(" ", "_")
)
drop_cols = [col for col in df.columns if col.lower().startswith("matches")]
df = df.drop(columns=drop_cols, errors="ignore")
for col in df.columns:
    if col not in ["Player", "Nation", "Pos", "Squad", "League"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Player"]).reset_index(drop=True)
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to {output_path}")
print(f"📊 Columns: {len(df.columns)} | Rows: {len(df)}")
