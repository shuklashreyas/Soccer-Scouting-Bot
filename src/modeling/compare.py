import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.modeling.similarity import PlayerEmbeddingModel, FRIENDLY_NAMES
from src.modeling.role_check import determine_comparison_style
import pickle
from pathlib import Path


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

    def __init__(self, df: pd.DataFrame = None, model: PlayerEmbeddingModel = None, model_path: str = None):
        """
        Initialize the comparison engine.

        Provide either:
         - `model`: an instantiated `PlayerEmbeddingModel`, or
         - `model_path`: path to a pickled `PlayerEmbeddingModel`, or
         - `df`: a DataFrame (will build an in-memory model).

        Building from `df` is the slowest option; prefer passing a pre-built
        `model` or a `model_path` to a persisted model.
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            # load persisted model
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with open(p, "rb") as f:
                self.model = pickle.load(f)
        elif df is not None:
            self.model = PlayerEmbeddingModel(df)
        else:
            raise ValueError("Provide either df, model, or model_path to initialize PlayerComparisonEngine")

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

        # Use the model's friendly cluster label when available
        role_label_a = self.model._cluster_name(cluster_a)
        role_label_b = self.model._cluster_name(cluster_b)

        # Determine comparison style using a role sanity-check
        cmp_style = determine_comparison_style(self.model, idx_a, idx_b)

        if cmp_style['style'] == 'stat':
            if cluster_a == cluster_b:
                role_info = (
                    f"Both players belong to **{role_label_a}**, meaning they operate in "
                    "similar zones and assume similar tactical responsibilities."
                )
            else:
                role_info = (
                    f"{row_a['Player']} is in **{role_label_a}**, while "
                    f"{row_b['Player']} is in **{role_label_b}** — they are different clusters but share a coarse role group, so a stat-based comparison is appropriate."
                )
        else:
            # Role-based comparison: be explicit that stat comparisons are misleading
            role_info = (
                f"{row_a['Player']} is in **{role_label_a}** ({cmp_style.get('group_a','unknown')}), "
                f"while {row_b['Player']} is in **{role_label_b}** ({cmp_style.get('group_b','unknown')}). "
                "These players perform fundamentally different tactical functions — direct comparison on attacking output would be misleading."
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

        # --- STRUCTURED PER-FEATURE OUTPUT ---
        feature_details = {}
        for i, feat in enumerate(feature_cols):
            z_a_i = float(z_a[i])
            z_b_i = float(z_b[i])
            d = float(diff[i])
            ov = float(overlap[i])
            feature_details[feat] = {
                "z_a": z_a_i,
                "z_b": z_b_i,
                "diff": d,
                "overlap": ov,
                "both_above": bool(z_a_i > 0 and z_b_i > 0),
            }

        # --- STYLE SUMMARY (NLP LOGIC) ---
        if cmp_style['style'] == 'stat':
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
        else:
            # When roles differ, explicitly state comparison style and provide directional insights
            style_text = (
                f"Players occupy different tactical roles ({cmp_style.get('group_a')} vs {cmp_style.get('group_b')}). "
                "Focus on role-based contributions rather than raw attacking/defensive metrics."
            )

        # --- FINAL COMPARISON PARAGRAPH ---
        # If role comparison is requested, adjust the advantages lists to role-relevant features
        if cmp_style['style'] == 'role':
            # coarse preferred metric groups
            preferred = {
                'attacker': ['Shooting_Score', 'Creation_Score', 'Dribbling_Score', 'Carrying_Score', 'Passing_Score'],
                'midfielder': ['Passing_Score', 'Creation_Score', 'Carrying_Score', 'Dribbling_Score'],
                'defender': ['Defending_Score', 'Passing_Score', 'Carrying_Score'],
                'goalkeeper': [],
                'unknown': []
            }

            def _top_role_feats(z, group, top_n=6):
                cols = feature_cols
                cand = preferred.get(group, [])
                # restrict to candidate features that exist
                idxs = [i for i, c in enumerate(cols) if c in cand]
                # fall back to global ordering if none matched
                if not idxs:
                    idxs = list(range(len(cols)))
                # sort by z descending
                idxs_sorted = sorted(idxs, key=lambda i: z[i], reverse=True)
                feats = [FRIENDLY_NAMES.get(cols[i], cols[i]) for i in idxs_sorted[:top_n]]
                return feats

            adv_a = _top_role_feats(z_a, cmp_style.get('group_a', 'unknown'), top_n=6)
            adv_b = _top_role_feats(z_b, cmp_style.get('group_b', 'unknown'), top_n=6)

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
            "feature_details": feature_details,
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
