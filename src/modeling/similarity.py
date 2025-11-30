# src/modeling/similarity.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
#  CONFIG: FEATURE GROUPS
# -------------------------------

# These are "candidate" features. We only keep the ones present in the DF.
CANDIDATE_FEATURE_GROUPS = {
    "attacking": [
        "Performance_Gls",
        "Performance_Ast",
        "Per 90 Minutes_Gls",
        "Per 90 Minutes_Ast",
        "Per 90 Minutes_G+A",
        "Per 90 Minutes_G-PK",
        "Per 90 Minutes_xG",
        "Per 90 Minutes_xAG",
        "Per 90 Minutes_xG+xAG",
        "Per 90 Minutes_npxG",
        "Per 90 Minutes_npxG+xAG",
        "Expected_xG",
        "Expected_npxG",
        "Expected_xAG",
        "Expected_npxG+xAG",
        "SCA_SCA",                  # shot-creating actions (if present)
        "GCA_GCA",                  # goal-creating actions (if present)
    ],
    "possession_progression": [
        "Progression_PrgC",
        "Progression_PrgP",
        "Progression_PrgR",
        "Touches_Att 3rd",          # attacking third touches (if present)
        "Touches_Att Pen",          # penalty box touches
        "Carries_CPA",              # carries into penalty area
        "Carries_C 3rd",            # carries into final third
    ],
    "passing": [
        "Passing_Cmp%",             # pass completion %
        "Passing_TotDist",
        "Passing_PrgDist",
        "Passing_1/3",              # passes into final third
        "Passing_PPA",              # passes into penalty area
        "Passing_KP",               # key passes
    ],
    "defending": [
        "Tackles_Tkl",
        "Tackles_Def 3rd",
        "Tackles_Mid 3rd",
        "Tackles_Att 3rd",
        "Blocks_Blocks",
        "Blocks_Sh",
        "Blocks_ShSv",
        "Int",                      # interceptions
        "Clr",                      # clearances
        "Aerial Duels_Won",
        "Aerial Duels_Won%",        # aerial %
    ],
    "game_time": [
        "Playing Time_MP",
        "Playing Time_Starts",
        "Playing Time_Min",
        "Playing Time_90s",
    ],
}

# Optional: human readable names for explanation
FRIENDLY_NAMES = {
    "Performance_Gls": "Goals",
    "Performance_Ast": "Assists",
    "Per 90 Minutes_Gls": "Goals per 90",
    "Per 90 Minutes_Ast": "Assists per 90",
    "Per 90 Minutes_G+A": "Goals + Assists per 90",
    "Per 90 Minutes_G-PK": "Non-penalty goals per 90",
    "Per 90 Minutes_xG": "xG per 90",
    "Per 90 Minutes_xAG": "xAG per 90",
    "Per 90 Minutes_xG+xAG": "xG+xAG per 90",
    "Per 90 Minutes_npxG": "Non-penalty xG per 90",
    "Per 90 Minutes_npxG+xAG": "Non-penalty xG + xAG per 90",
    "Expected_xG": "Total xG",
    "Expected_xAG": "Total xAG",
    "Progression_PrgC": "Progressive carries",
    "Progression_PrgP": "Progressive passes",
    "Progression_PrgR": "Progressive receptions",
    "Touches_Att 3rd": "Attacking third touches",
    "Touches_Att Pen": "Penalty box touches",
    "Carries_CPA": "Carries into penalty area",
    "Carries_C 3rd": "Carries into final third",
    "Passing_KP": "Key passes",
    "Passing_PPA": "Passes into penalty area",
    "Aerial Duels_Won": "Aerial duels won",
    "Aerial Duels_Won%": "Aerial win %"
}


# -------------------------------
#  MAIN MODEL
# -------------------------------

class PlayerEmbeddingModel:
    """
    Full pipeline:
      - choose numeric features from DF
      - StandardScaler
      - PCA (embeddings)
      - KMeans (roles)
      - cosine similarity for neighbors
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_components: int = 20,
        n_clusters: int = 12,
        min_features: int = 5,
    ):
        if "Player" not in df.columns:
            raise ValueError("DataFrame must contain a 'Player' column.")

        self.raw_df = df.copy()
        self.n_components = n_components
        self.n_clusters = n_clusters

        # 1) choose feature columns that actually exist
        feature_cols = []
        for group_cols in CANDIDATE_FEATURE_GROUPS.values():
            feature_cols.extend([c for c in group_cols if c in df.columns])

        feature_cols = sorted(list(set(feature_cols)))
        if len(feature_cols) < min_features:
            raise ValueError(
                f"Not enough numeric features. Found {len(feature_cols)}, "
                f"need at least {min_features}."
            )

        self.feature_cols = feature_cols

        # 2) keep only rows with no NA in those features + player metadata
        numeric = df[self.feature_cols].copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        mask = numeric.notna().all(axis=1)
        self.df = df.loc[mask].reset_index(drop=True)
        numeric = numeric.loc[mask].reset_index(drop=True)

        # optional position info
        self.pos_col = None
        for candidate in ["Pos", "position", "Position"]:
            if candidate in self.df.columns:
                self.pos_col = candidate
                break

        # 3) fit scaler
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(numeric.values)

        # 4) fit PCA for embeddings
        n_features = self.X_scaled.shape[1]
        n_comp = min(self.n_components, n_features)
        if n_comp <= 2:
            # just keep scaled features; PCA not worth it
            self.embeddings = self.X_scaled
            self.pca = None
        else:
            self.pca = PCA(n_components=n_comp, random_state=42)
            self.embeddings = self.pca.fit_transform(self.X_scaled)

        # 5) fit KMeans for role clusters
        n_clusters = min(self.n_clusters, self.embeddings.shape[0])
        if n_clusters >= 2:
            self.kmeans = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            )
            self.role_labels = self.kmeans.fit_predict(self.embeddings)
        else:
            self.kmeans = None
            self.role_labels = np.zeros(self.embeddings.shape[0], dtype=int)

        self.df["role_cluster"] = self.role_labels

    # -------------------------------
    #  HELPERS
    # -------------------------------

    def _match_player_index(self, player_name: str):
        """case-insensitive substring match; returns index in self.df or None."""
        name = player_name.lower().strip()

        # exact match first
        exact = self.df.index[self.df["Player"].str.lower() == name]
        if len(exact) > 0:
            return int(exact[0])

        # substring match
        mask = self.df["Player"].str.lower().str.contains(name, na=False)
        candidates = self.df.index[mask]
        if len(candidates) == 0:
            return None

        return int(candidates[0])

    def _cluster_name(self, cluster_id: int) -> str:
        """
        Placeholder: you can manually map cluster IDs to human labels later,
        after inspecting cluster stats.
        """
        return f"Role Cluster {cluster_id}"

    # -------------------------------
    #  PUBLIC INTERFACE
    # -------------------------------

    def get_similar_players(self, player_name: str, top_k: int = 5) -> pd.DataFrame:
        """
        Returns dataframe with columns:
        [Player, Squad (if available), Pos (if available),
         similarity, role_cluster, role_label]
        """
        idx = self._match_player_index(player_name)
        if idx is None:
            raise ValueError(f"Player '{player_name}' not found in dataset.")

        target_vec = self.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.embeddings)[0]

        # sort by similarity (descending) and skip self
        order = np.argsort(sims)[::-1]
        order = [i for i in order if i != idx][:top_k]

        out = self.df.iloc[order].copy()
        out["similarity"] = sims[order]
        out["role_cluster"] = self.role_labels[order]
        out["role_label"] = out["role_cluster"].apply(self._cluster_name)

        cols = ["Player"]
        if "Squad" in out.columns:
            cols.append("Squad")
        if self.pos_col and self.pos_col in out.columns:
            cols.append(self.pos_col)
        cols += ["similarity", "role_label", "role_cluster"]

        return out[cols].reset_index(drop=True)

    def explain_pair(
        self,
        player_a: str,
        player_b: str,
        top_shared: int = 5,
    ) -> dict:
        """
        Returns dict with:
          {
            "player_a": str,
            "player_b": str,
            "role_info": str,
            "shared_strengths": [ ... ],
            "style_summary": str
          }
        """

        idx_a = self._match_player_index(player_a)
        idx_b = self._match_player_index(player_b)
        if idx_a is None:
            raise ValueError(f"Player '{player_a}' not found.")
        if idx_b is None:
            raise ValueError(f"Player '{player_b}' not found.")

        row_a = self.df.iloc[idx_a]
        row_b = self.df.iloc[idx_b]

        # role info
        cluster_a = int(self.role_labels[idx_a])
        cluster_b = int(self.role_labels[idx_b])
        role_name_a = self._cluster_name(cluster_a)
        role_name_b = self._cluster_name(cluster_b)

        # work in z-score space for features
        z_a = self.X_scaled[idx_a]
        z_b = self.X_scaled[idx_b]

        # high positive overlap: both high on same feature
        overlap = z_a * z_b  # large positive => both above-average, aligned
        order = np.argsort(overlap)[::-1]

        strengths = []
        for i in order:
            if overlap[i] <= 0:
                continue
            col = self.feature_cols[i]
            friendly = FRIENDLY_NAMES.get(col, col)
            strengths.append(friendly)
            if len(strengths) >= top_shared:
                break

        # style summary sentence
        if len(strengths) == 0:
            style = (
                "Statistically they are not strongly aligned on any major above-average traits, "
                "but they share a similar overall role profile."
            )
        else:
            if len(strengths) == 1:
                traits = strengths[0]
            elif len(strengths) == 2:
                traits = f"{strengths[0]} and {strengths[1]}"
            else:
                traits = ", ".join(strengths[:-1]) + f", and {strengths[-1]}"

            style = (
                f"They are both above average in {traits}, "
                f"which drives the similarity in their playing profiles."
            )

        # build role sentence
        if cluster_a == cluster_b:
            role_text = (
                f"Both players fall into the same role group: **{role_name_a}**, "
                "meaning they tend to occupy similar zones and play similar tactical roles."
            )
        else:
            role_text = (
                f"{player_a} is in role group **{role_name_a}**, while {player_b} is in "
                f"**{role_name_b}**. They are similar statistically, but with slightly "
                "different role emphases."
            )

        return {
            "player_a": row_a["Player"],
            "player_b": row_b["Player"],
            "role_info": role_text,
            "shared_strengths": strengths,
            "style_summary": style,
        }


# -------------------------------
#  SIMPLE WRAPPERS FOR STREAMLIT
# -------------------------------

def find_similar_players(
    target_player: str,
    df: pd.DataFrame,
    top_k: int = 5,
) -> list[str]:
    """
    Minimal wrapper used by Streamlit.
    Returns list of similar player names only.
    """
    model = PlayerEmbeddingModel(df)
    sims_df = model.get_similar_players(target_player, top_k=top_k)
    return sims_df["Player"].tolist()


def explain_similarity(
    target_player: str,
    df: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """
    Convenience helper: returns explanations for the top similar player.
    """
    model = PlayerEmbeddingModel(df)
    sims_df = model.get_similar_players(target_player, top_k=top_k)
    if sims_df.empty:
        raise ValueError("No similar players found.")
    top_match = sims_df.iloc[0]["Player"]
    return model.explain_pair(target_player, top_match)
