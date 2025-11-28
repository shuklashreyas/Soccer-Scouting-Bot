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

def find_similar_players(target_player: str, df: pd.DataFrame, scaler=None, knn=None, feature_cols=None, top_k: int = 5):
    """Compatibility wrapper for Streamlit.

    Behavior:
    - If `scaler`, `knn`, and `feature_cols` are provided (training pickle), use them
      to compute nearest neighbors on the subset df.dropna(subset=feature_cols).
    - Otherwise, fall back to building a `SimilarityModel` on the best available
      in-df feature set.

    Returns a list of player names (strings).
    """

    # If we have a trained kNN model and feature columns, use it directly
    if scaler is not None and knn is not None and feature_cols:
        # Ensure columns exist in df
        cols = [c for c in feature_cols if c in df.columns]
        if len(cols) < 2:
            raise ValueError("Not enough trained feature columns present in dataframe.")

        df_sub = df.dropna(subset=cols).reset_index(drop=True)

        # Find target in df_sub
        matches = df_sub.index[df_sub["Player"].str.lower() == target_player.lower()].tolist()
        if not matches:
            # try substring match
            matches = df_sub.index[df_sub["Player"].str.contains(target_player, case=False, na=False)].tolist()
        if not matches:
            raise ValueError(f"Player '{target_player}' not found in dataset (after filtering for trained features).")

        target_idx = matches[0]

        # Build scaled matrix (assume scaler was fitted on same ordering)
        X = df_sub[cols].fillna(0).astype(float).values
        X_scaled = scaler.transform(X)

        # Query neighbors (request top_k+1 to skip self if included)
        try:
            distances, indices = knn.kneighbors(X_scaled[target_idx].reshape(1, -1), n_neighbors=min(len(df_sub), top_k + 1))
            inds = indices[0].tolist()
        except Exception:
            # Fallback: compute euclidean distances manually
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(X_scaled[target_idx].reshape(1, -1), X_scaled)[0]
            inds = list(np.argsort(dists))[: min(len(dists), top_k + 1)]

        # Remove self if present, then take top_k
        inds = [i for i in inds if i != target_idx]
        inds = inds[:top_k]

        return df_sub.iloc[inds]["Player"].tolist()

    # Otherwise fall back to building an internal model from available columns
    preferred_feature_sets = [
        ["Per_90_Minutes_xG", "Per_90_Minutes_xAG", "Progression_PrgP", "Playing_Time_90s"],
        ["Performance_Gls", "Performance_Ast", "Expected_xG", "Expected_xAG", "Progression_PrgP"],
        ["goals", "assists", "xg", "xa", "progressive_passes", "ppa", "g+a"]
    ]

    for feature_set in preferred_feature_sets:
        usable = [col for col in feature_set if col in df.columns]
        if len(usable) >= 2:
            model = SimilarityModel(df, usable)
            results = model.find_similar_players(target_player, top_k=top_k)
            return results["Player"].tolist()

    raise ValueError("No suitable numeric features found in dataframe for similarity matching.")
