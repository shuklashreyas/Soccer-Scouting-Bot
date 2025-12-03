"""Train and persist a PlayerEmbeddingModel for faster app startup.

Usage (from project root):
    python3 -m src.modeling.train_player_embedding

This script loads `data/processed/all_leagues_clean.csv`, builds the
`PlayerEmbeddingModel`, and saves it to `data/models/player_embedding_model.pkl`.
"""
from pathlib import Path
import pandas as pd
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_player_embedding")

    data_path = Path("data/processed/all_leagues_clean.csv")
    out_path = Path("data/models/player_embedding_model.pkl")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    # Create fallback score columns if the model's expected features are missing.
    # This mirrors the quick ad-hoc mapping used during interactive debugging
    # so training will succeed against your existing CSV.
    def first_existing(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    # Candidate source columns (prefer expected xG/xAG/per90 names when present)
    shooting_src = first_existing(["Expected_xG", "Expected_npxG", "Per_90_Minutes_xG", "Performance_Gls"])
    dribbling_src = first_existing(["Progression_PrgC", "Carries_C 3rd", "Carries_CPA", "Progression_PrgR"])
    passing_src = first_existing(["Per_90_Minutes_xAG", "Per_90_Minutes_Ast", "Passing_PrgDist", "Passing_TotDist"])
    creation_src = first_existing(["Passing_KP", "SCA_SCA", "GCA_GCA", "Per_90_Minutes_xG+xAG"]) 
    carrying_src = first_existing(["Progression_PrgC", "Carries_C 3rd", "Carries_CPA"])
    defending_src = first_existing(["Tackles_Tkl", "Int", "Clr", "Aerial Duels_Won"])

    logger.info(f"Chosen sources for score cols: shooting={shooting_src}, dribbling={dribbling_src}, passing={passing_src}, creation={creation_src}, carrying={carrying_src}, defending={defending_src}")

    # Work on a copy so we don't modify original dataset externally
    df2 = df.copy()

    # Helper to create numeric fallback column
    def _mk(col_name, src):
        if src and src in df2.columns:
            df2[col_name] = pd.to_numeric(df2[src], errors="coerce").fillna(0.0)
        else:
            df2[col_name] = 0.0

    _mk("Shooting_Score", shooting_src)
    _mk("Dribbling_Score", dribbling_src)
    _mk("Passing_Score", passing_src)
    _mk("Creation_Score", creation_src)
    _mk("Carrying_Score", carrying_src)
    _mk("Defending_Score", defending_src)

    # Lazy import to avoid requiring heavy deps unless running training
    from src.modeling.similarity import PlayerEmbeddingModel

    logger.info("Building PlayerEmbeddingModel (this may take a few seconds)")
    model = PlayerEmbeddingModel(df2)

    logger.info(f"Saving model to {out_path}")
    model.save(str(out_path))
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
