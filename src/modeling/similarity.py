CANDIDATE_FEATURE_GROUPS = {
    "shooting": ["Shooting_Score"],
    "dribbling": ["Dribbling_Score"],
    "passing": ["Passing_Score"],
    "creation": ["Creation_Score"],
    "carrying": ["Carrying_Score"],
    "defending": ["Defending_Score"],
}

# Optional: human readable names for explanation
FRIENDLY_NAMES = {
    "Shooting_Score": "Shooting",
    "Dribbling_Score": "Dribbling",
    "Passing_Score": "Passing",
    "Creation_Score": "Chance Creation",
    "Carrying_Score": "Ball Carrying",
    "Defending_Score": "Defending",
}

# Human-readable cluster labels (simple mapping: cluster id -> label)
ROLE_LABELS = {
    0: "Attacking Fullbacks / Offensive Wide Progressors",
    1: "Superstar Inverted Forwards / Elite Finishers & Creators",
    2: "Defensive Stoppers / Deep Defenders",
    3: "High-Workrate Wide Forwards / Transitional Threats",
    4: "Target 9 / Ball-Retaining Strikers",
    5: "Creative Dribbling Wingers / Ball-Carrying Engines",
    6: "Deep-Lying Playmakers / Box-to-Box Controllers",
    7: "Elite Central Finisher",
    8: "High-Volume Shooters / Secondary Finishers",
    9: "Defensive Anchors / Ball-Winning Midfielders",
    10: "Hybrid 8/10s & Second Strikers",
    11: "Advanced Playmaking Wingers / Creative Threats",
}


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

"""Player embedding pipeline: select numeric features, scale, PCA, cluster, and explain similarities."""

# Z-score features from z_scores.csv
CANDIDATE_FEATURE_GROUPS = {
    "shooting": ["Shooting_Score"],
    "dribbling": ["Dribbling_Score"],
    "passing": ["Passing_Score"],
    "creation": ["Creation_Score"],
    "carrying": ["Carrying_Score"],
    "defending": ["Defending_Score"],
}

# Optional: human readable names for explanation
FRIENDLY_NAMES = {
    "Shooting_Score": "Shooting",
    "Dribbling_Score": "Dribbling",
    "Passing_Score": "Passing",
    "Creation_Score": "Chance Creation",
    "Carrying_Score": "Ball Carrying",
    "Defending_Score": "Defending",
}

# Human-readable cluster labels (simple mapping: cluster id -> label)
ROLE_LABELS = {
    0: "Attacking Fullbacks / Offensive Wide Progressors",
    1: "Superstar Inverted Forwards / Elite Finishers & Creators",
    2: "Defensive Stoppers / Deep Defenders",
    3: "High-Workrate Wide Forwards / Transitional Threats",
    4: "Target 9 / Ball-Retaining Strikers",
    5: "Creative Dribbling Wingers / Ball-Carrying Engines",
    6: "Deep-Lying Playmakers / Box-to-Box Controllers",
    7: "Elite Central Finisher",
    8: "High-Volume Shooters / Secondary Finishers",
    9: "Defensive Anchors / Ball-Winning Midfielders",
    10: "Hybrid 8/10s & Second Strikers",
    11: "Advanced Playmaking Wingers / Creative Threats",
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
        # Optional persistent mapping: cluster_id -> human label (frozen)
        # This can be set by an admin script to avoid label drift across retrains.
        self.cluster_label_map = {}

    # -------------------------------
    #  Persistence helpers
    # -------------------------------
    def save(self, path: str):
        """Serialize the full model (including fitted sklearn objects and
        the filtered dataframe) to `path` using pickle.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load a pickled PlayerEmbeddingModel instance from `path`.
        Returns the deserialized object.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(p, "rb") as f:
            obj = pickle.load(f)
        return obj

    def set_cluster_label_map(self, mapping: dict):
        """Set a persistent cluster label mapping (cluster_id -> label string).

        This will be serialized when `save()` is called.
        """
        # normalize keys to int
        norm = {int(k): v for k, v in mapping.items()}
        self.cluster_label_map = norm

    def update_and_save_labels(self, mapping: dict, path: str):
        """Convenience: set mapping and save the model to `path`."""
        self.set_cluster_label_map(mapping)
        self.save(path)

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
        # Prefer an explicit frozen mapping set on the model instance first.
        try:
            if getattr(self, 'cluster_label_map', None) and int(cluster_id) in self.cluster_label_map:
                return self.cluster_label_map[int(cluster_id)]
        except Exception:
            pass

        # Prefer human-friendly labels when available. ROLE_LABELS entries may
        # be either a string (legacy) or a dict with a 'label' key (preferred).
        try:
            entry = ROLE_LABELS.get(int(cluster_id))
            if isinstance(entry, dict):
                return entry.get("label", f"Role Cluster {cluster_id}")
            if isinstance(entry, str):
                return entry
            return f"Role Cluster {cluster_id}"
        except Exception:
            return f"Role Cluster {cluster_id}"

    # -------------------------------
    #  PUBLIC INTERFACE
    # -------------------------------

    def get_similar_players(self, player_name: str, top_k: int = 5) -> pd.DataFrame:
        """
        Returns dataframe with columns:
        [Player, Squad (if available), Pos (if available),
         similarity, role_cluster, role_label]
        Only returns players who share at least one position with the target player.
        """
        idx = self._match_player_index(player_name)
        if idx is None:
            raise ValueError(f"Player '{player_name}' not found in dataset.")

        target_vec = self.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.embeddings)[0]

        # Get target player's position(s)
        target_pos = None
        if self.pos_col and self.pos_col in self.df.columns:
            target_pos = str(self.df.iloc[idx][self.pos_col])

        # Helper function to check if positions overlap
        def positions_match(pos1, pos2):
            """Check if two position strings share at least one position."""
            if pd.isna(pos1) or pd.isna(pos2):
                return True  # If position data missing, include player

            # Split positions (e.g., "FW,MF" or "DF" or "MF,FW,DF")
            pos1_set = set(str(pos1).upper().replace(' ', '').split(','))
            pos2_set = set(str(pos2).upper().replace(' ', '').split(','))

            return len(pos1_set & pos2_set) > 0  # True if any position matches

        # Sort by similarity (descending) and filter by position
        order = np.argsort(sims)[::-1]

        # Filter candidates: skip self and filter by position
        candidates = []
        for i in order:
            if i == idx:  # Skip the target player
                continue

            # Check position match
            if target_pos is not None:
                candidate_pos = self.df.iloc[i][self.pos_col] if self.pos_col in self.df.columns else None
                if not positions_match(target_pos, candidate_pos):
                    continue  # Skip if positions don't match

            candidates.append(i)

            if len(candidates) >= top_k:
                break

        # If we don't have enough candidates after filtering, relax the constraint
        if len(candidates) < top_k:
            # Add more candidates without position filter
            for i in order:
                if i == idx or i in candidates:
                    continue
                candidates.append(i)
                if len(candidates) >= top_k:
                    break

        out = self.df.iloc[candidates].copy()
        out["similarity"] = sims[candidates]
        out["role_cluster"] = self.role_labels[candidates]
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
