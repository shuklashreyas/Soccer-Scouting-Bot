import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================
#   COSINE SIMILARITY MODEL
# ============================

class SimilarityModel:
    """Clean cosine-similarity player finder."""

    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df = df.copy()
        self.feature_cols = feature_cols

        # Filter only rows with needed features
        self.df = self.df.dropna(subset=feature_cols)

        # Ensure Player column exists
        if "Player" not in self.df.columns:
            raise ValueError("Dataset must contain a 'Player' column.")

        # Fit the scaler
        self.scaler = StandardScaler()
        self.matrix = self.scaler.fit_transform(self.df[self.feature_cols])

    def find_similar_players(self, player_name: str, top_k: int = 5):
        """Return a DataFrame of most similar players."""

        # Make sure the player exists
        if player_name not in self.df["Player"].values:
            raise ValueError(f"Player '{player_name}' not found in dataset.")

        # Locate index
        idx = self.df.index[self.df["Player"] == player_name][0]

        # Cosine similarity
        target_vec = self.matrix[idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.matrix)[0]

        # Ranking
        similar_idx = np.argsort(sims)[::-1][1:top_k + 1]

        results = self.df.iloc[similar_idx][["Player", "Squad", "Pos"]].copy()
        results["similarity"] = sims[similar_idx]

        return results.reset_index(drop=True)


# ============================
#   STREAMLIT COMPAT WRAPPER
# ============================

def find_similar_players(target_player: str, df: pd.DataFrame, top_k: int = 5):
    """
    Wrapper function used by Streamlit.
    Picks the best available feature set and returns ONLY the list of names.
    """

    preferred_feature_sets = [
        # Most detailed — if your FBref cleaned dataset contains them
        ["Per 90 Minutes_xG", "Per 90 Minutes_xAG", "Progression_PrgP", "Playing Time_90s"],

        # Basic attacking profile
        ["Performance_Gls", "Performance_Ast", "Expected_xG", "Expected_xAG", "Progression_PrgP"],

        # High-level
        ["goals", "assists", "xg", "xa", "progressive_passes", "ppa", "g+a"]
    ]

    # Pick the first valid feature set
    for feature_set in preferred_feature_sets:
        usable = [col for col in feature_set if col in df.columns]
        if len(usable) >= 2:
            model = SimilarityModel(df, usable)
            results = model.find_similar_players(target_player, top_k=top_k)
            return results["Player"].tolist()

    # No valid numeric columns
    raise ValueError("❌ No suitable numeric features found in dataframe for similarity matching.")
