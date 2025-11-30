# src/modeling/debug_profiles.py

import pandas as pd
from src.modeling.similarity import PlayerEmbeddingModel

DATA_PATH = "data/processed/all_leagues_clean.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Build model
    model = PlayerEmbeddingModel(df)

    target = "erling haaland"

    print("\n=== SIMILAR PLAYERS ===")
    sims = model.get_similar_players(target, top_k=10)
    print(sims)

    print("\n=== EXPLANATION OF TOP MATCH ===")
    top_match = sims.iloc[0]["Player"]
    explanation = model.explain_pair(target, top_match)
    print(explanation)

if __name__ == "__main__":
    main()
