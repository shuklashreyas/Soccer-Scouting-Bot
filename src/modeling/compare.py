import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.modeling.similarity import PlayerEmbeddingModel, FRIENDLY_NAMES


class PlayerComparisonEngine:
    """
    True comparison engine (NOT similarity search).
    
    Produces:
      - statistical overlap
      - statistical contrast
      - cluster/role comparison
      - NLP-style justification
      - final scouting report
    """

    def __init__(self, df: pd.DataFrame):
        self.model = PlayerEmbeddingModel(df)
        self.df = self.model.df

    def _idx(self, name):
        return self.model._match_player_index(name)

    def _safe_name(self, raw):
        return raw["Player"]

    def compare(self, player_a, player_b, top_k=6):
        """
        Main public method.
        Returns a dictionary with:
            overlap, differences, role_info, style, summary
        """
        idx_a = self._idx(player_a)
        idx_b = self._idx(player_b)
        if idx_a is None:
            raise ValueError(f"{player_a} not found.")
        if idx_b is None:
            raise ValueError(f"{player_b} not found.")

        row_a = self.df.iloc[idx_a]
        row_b = self.df.iloc[idx_b]

        # --- ROLE CLUSTER COMPARISON ---
        cluster_a = int(self.model.role_labels[idx_a])
        cluster_b = int(self.model.role_labels[idx_b])
        role_name_a = f"Role Cluster {cluster_a}"
        role_name_b = f"Role Cluster {cluster_b}"

        if cluster_a == cluster_b:
            role_info = (
                f"Both players belong to **{role_name_a}**, meaning they operate in "
                "similar zones and assume similar tactical responsibilities."
            )
        else:
            role_info = (
                f"{row_a['Player']} is in **{role_name_a}**, while "
                f"{row_b['Player']} is in **{role_name_b}** — indicating similar end output but "
                "different tactical applications and pitch behaviors."
            )

        # --- FEATURE DIFFERENCES / OVERLAP ---
        z_a = self.model.X_scaled[idx_a]
        z_b = self.model.X_scaled[idx_b]

        diff = z_a - z_b
        overlap = z_a * z_b

        feature_cols = self.model.feature_cols

        # Top shared strengths
        shared_idx = np.argsort(overlap)[::-1]
        shared_feats = []
        for i in shared_idx:
            if overlap[i] <= 0:
                continue
            feat = FRIENDLY_NAMES.get(feature_cols[i], feature_cols[i])
            shared_feats.append(feat)
            if len(shared_feats) == top_k:
                break

        # Top areas where A > B
        advantage_a_idx = np.argsort(diff)[::-1]
        adv_a = []
        for i in advantage_a_idx:
            if diff[i] <= 0:
                continue
            feat = FRIENDLY_NAMES.get(feature_cols[i], feature_cols[i])
            adv_a.append(feat)
            if len(adv_a) == top_k:
                break

        # Top areas where B > A
        advantage_b_idx = np.argsort(diff)
        adv_b = []
        for i in advantage_b_idx:
            if diff[i] >= 0:
                continue
            feat = FRIENDLY_NAMES.get(feature_cols[i], feature_cols[i])
            adv_b.append(feat)
            if len(adv_b) == top_k:
                break

        # --- STYLE SUMMARY (NLP LOGIC) ---
        if shared_feats:
            if len(shared_feats) > 1:
                shared_text = ", ".join(shared_feats[:-1]) + f", and {shared_feats[-1]}"
            else:
                shared_text = shared_feats[0]
            style_text = (
                f"Both players show above-average performance in **{shared_text}**, "
                "which explains why their statistical output often looks similar."
            )
        else:
            style_text = (
                "The players share few above-average traits, indicating different styles "
                "with minimal statistical overlap."
            )

        # --- FINAL COMPARISON PARAGRAPH ---
        final_paragraph = self._build_summary(
            row_a["Player"], row_b["Player"], role_info, shared_feats, adv_a, adv_b
        )

        return {
            "player_a": row_a["Player"],
            "player_b": row_b["Player"],
            "role_info": role_info,
            "shared_strengths": shared_feats,
            "advantages_player_a": adv_a,
            "advantages_player_b": adv_b,
            "style_commentary": style_text,
            "final_summary": final_paragraph,
        }

    def _build_summary(self, name_a, name_b, role_text, shared, adv_a, adv_b):
        """
        Turns the numeric stats into a scouting-style paragraph.
        """

        # shared traits
        if shared:
            shared_str = ", ".join(shared)
            shared_sentence = f"They overlap in **{shared_str}**, showing clear similarities in output."
        else:
            shared_sentence = "They share minimal statistical overlap."

        # A strengths
        if adv_a:
            adv_a_str = ", ".join(adv_a)
            adv_a_sentence = f"**{name_a}** stands out in {adv_a_str}."
        else:
            adv_a_sentence = f"{name_a} has no major statistical advantages."

        # B strengths
        if adv_b:
            adv_b_str = ", ".join(adv_b)
            adv_b_sentence = f"**{name_b}** leads in {adv_b_str}."
        else:
            adv_b_sentence = f"{name_b} has no major statistical advantages."

        return (
            f"{role_text} {shared_sentence} "
            f"{adv_a_sentence} {adv_b_sentence}"
        )
