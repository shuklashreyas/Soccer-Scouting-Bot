"""Small debug script to inspect similarity results and explanations."""

import pandas as pd
from src.modeling.similarity import PlayerEmbeddingModel

DATA_PATH = "data/processed/all_leagues_clean.csv"


def main():
    """Build an embedding model and print similar players + explanation for a target."""
    df = pd.read_csv(DATA_PATH)

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
