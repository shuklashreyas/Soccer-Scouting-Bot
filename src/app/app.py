import streamlit as st
import pandas as pd
import pickle
import sys
from pathlib import Path
import uuid
import io
import matplotlib.pyplot as plt


# Fix module import paths so `src` can be imported when running the app
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local module imports
from src.nlp.intent_classifier import predict_intent
from src.nlp.entity_extraction import extract_entities
from src.modeling.similarity import find_similar_players, ROLE_LABELS, FRIENDLY_NAMES
from src.player.lookup import lookup_player
from src.player.extract import extract_player_profile
from src.player.roles import classify_role
from src.player.insights import generate_strengths_weaknesses, ordinal
from src.visualization.radar_chart import plot_radar
from src.visualization.cluster_plots import (
    plot_pca_scatter,
    cluster_size_bar,
    cluster_centroid_radar,
    compare_z_bar,
    similarity_heatmap,
    cluster_feature_violin,
)


@st.cache_data
def load_data():
    # Load core datasets and return commonly used lists (players, leagues, stats)
    z_scores_df = pd.read_csv("data/processed/z_scores.csv")
    all_leagues_df = pd.read_csv("data/processed/all_leagues_clean.csv")
    outfield_df = pd.read_csv("data/processed/outfield_clean.csv")

    # Merge scoring z-scores into the main leagues dataframe
    df = all_leagues_df.merge(
        z_scores_df[['Player', 'Squad', 'Shooting_Score', 'Dribbling_Score',
                     'Passing_Score', 'Creation_Score', 'Carrying_Score', 'Defending_Score']],
        on=['Player', 'Squad'],
        how='left'
    )

    players = df["Player"].dropna().unique().tolist()
    leagues = ["premier league", "la liga", "bundesliga", "serie a", "ligue 1"]

    # Minimal stat keywords used by the entity extractor
    stats = ["goals", "assists", "xg", "xa", "progressive passes", "ppa", "g+a"]

    return df, outfield_df, players, leagues, stats


df, outfield_df, players_list, league_list, stat_list = load_data()

# Cache a single embedding model instance so we don't rebuild it per query
@st.cache_resource
def get_embedding_model():
    try:
        # Lazy import to avoid heavy deps at module import time; prefer loading a persisted model
        from src.modeling.similarity import PlayerEmbeddingModel

        model_path = "data/models/player_embedding_model.pkl"
        try:
            return PlayerEmbeddingModel.load(model_path)
        except Exception:
            # Fall back to building from dataframe if persisted load fails
            try:
                return PlayerEmbeddingModel(df)
            except Exception:
                return None
    except Exception:
        # Modeling dependencies may be unavailable in some environments
        return None


# Load optional similarity model artifacts (kept for backward compatibility)
try:
    with open("data/models/similarity_model.pkl", "rb") as f:
        obj = pickle.load(f)

    # Support both older (scaler, knn) and newer (scaler, knn, feature_cols) formats
    if isinstance(obj, tuple):
        if len(obj) == 2:
            scaler, knn = obj
            feature_cols = None
        elif len(obj) == 3:
            scaler, knn, feature_cols = obj
        else:
            # Unexpected tuple shape
            scaler = knn = None
            feature_cols = None
    else:
        scaler = knn = None
        feature_cols = None

    similarity_model_missing = scaler is None or knn is None
except Exception:
    scaler = knn = None
    feature_cols = None
    similarity_model_missing = True


# ======================================
# STREAMLIT UI
# ======================================
st.set_page_config(page_title="Soccer Scouting Chatbot", layout="wide")

st.title("⚽ Soccer Scouting Chatbot — Working Demo")
st.markdown("Ask about **players, comparisons, or find similar profiles.**")

# Cluster Explorer: show cluster labels and sample members before assigning
with st.expander("Cluster Explorer (show role clusters and sample members)"):
    try:
        model = get_embedding_model()
        if model is None:
            st.info("Embedding model not available — run training script or check model file to view clusters.")
        else:
            # show summary: number of players, number of clusters
            n_players = len(model.df)
            n_clusters = len(set(model.role_labels))
            st.markdown(f"**Model contains {n_players} players across {n_clusters} clusters.**")

            # iterate clusters in numeric order and display label, profile, examples, size, sample players
            clusters = sorted(list(set(model.role_labels)))
            for cid in clusters:
                label = model._cluster_name(int(cid))
                # try to get rich metadata from ROLE_LABELS if present
                try:
                    from src.modeling.similarity import ROLE_LABELS
                    meta = ROLE_LABELS.get(int(cid))
                except Exception:
                    meta = None

                st.markdown(f"---\n**Cluster {cid}: {label}**")
                if isinstance(meta, dict):
                    profile = meta.get('profile')
                    examples = meta.get('examples')
                    summary = meta.get('summary')
                    if profile:
                        st.markdown(f"**Profile:** {profile}")
                    if examples:
                        st.markdown(f"**Examples:** {', '.join(examples[:8])}")
                    if summary:
                        st.markdown(f"*{summary}*")

                members = model.df[model.df['role_cluster'] == int(cid)]
                st.markdown(f"**Cluster size:** {len(members)}")
                # show small sample table
                sample = members[['Player']].head(12)
                st.table(sample)

            # --- Cluster visuals ---
            try:
                st.markdown("---")
                st.subheader("Model Visualizations")
                # PCA scatter + cluster size side-by-side
                col1, col2 = st.columns([3, 1])
                with col1:
                    try:
                        fig = plot_pca_scatter(model)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render PCA scatter: {e}")
                with col2:
                    try:
                        fig2 = cluster_size_bar(model)
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render cluster size bar: {e}")

                # Cluster centroid radar selector
                st.markdown("---")
                st.subheader("Cluster Centroid Radar")
                try:
                    sel = st.selectbox("Select cluster to view centroid radar", clusters, format_func=lambda x: f"{x} — {model._cluster_name(int(x))}")
                    fig3 = cluster_centroid_radar(model, int(sel))
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render centroid radar: {e}")

                # Pairwise z-score compare
                st.markdown("---")
                st.subheader("Pairwise Z-score Comparison")
                try:
                    players = model.df['Player'].dropna().unique().tolist()
                    p1 = st.selectbox("Player A", players, index=0)
                    p2 = st.selectbox("Player B", players, index=1)
                    if st.button("Render comparison chart"):
                        try:
                            fig4 = compare_z_bar(model, p1, p2)
                            st.plotly_chart(fig4, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render pairwise chart: {e}")
                except Exception as e:
                    st.warning(f"Could not render pairwise comparison UI: {e}")

                # Similarity heatmap
                st.markdown("---")
                st.subheader("Similarity Heatmap")
                try:
                    target_for_heat = st.selectbox("Choose player for top-K similarity heatmap", players, index=0, key='heat_player')
                    topk = st.slider("Top K", min_value=3, max_value=20, value=8)
                    if st.button("Render similarity heatmap"):
                        sims_df = model.get_similar_players(target_for_heat, top_k=topk)
                        names = [target_for_heat] + sims_df['Player'].tolist()
                        fig5 = similarity_heatmap(model, names)
                        st.plotly_chart(fig5, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render similarity heatmap: {e}")

                # Feature violin selector
                st.markdown("---")
                st.subheader("Feature Distribution by Cluster")
                try:
                    feat = st.selectbox("Feature", model.feature_cols, format_func=lambda x: FRIENDLY_NAMES.get(x, x))
                    fig6 = cluster_feature_violin(model, feat)
                    st.plotly_chart(fig6, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render feature violin: {e}")

            except Exception:
                # keep UI stable if visual modules fail
                pass
    except Exception as e:
        st.warning(f"Could not display clusters: {e}")

# If similarity model missing, show a single in-app warning (avoid spamming console)
try:
    if similarity_model_missing:
        if not st.session_state.get("_similarity_model_missing_shown"):
            st.session_state["_similarity_model_missing_shown"] = True
            st.warning(
                "Similarity model not found — run `python src/modeling/train_similarity_model.py` to create `data/models/similarity_model.pkl`."
            )
except Exception:
    # If session_state isn't available yet (rare), skip the UI warning silently
    pass


# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"sender": "bot", "text": "Hi — ask about any player, comparison, or similarity search!"}
    ]


# Display chat message nicely
def render_message(sender, text):
    # ChatGPT-like high-contrast bubbles so text is always readable
    if sender == "bot":
        bubble_bg = "#0f766e"     # teal (bot)
        text_color = "#ffffff"
        align = "left"
    else:
        bubble_bg = "#1f2937"     # dark slate (user)
        text_color = "#ffffff"
        align = "right"

    st.markdown(
        f"""
        <div style='text-align:{align}; margin:6px;'>
            <div style='display:inline-block; background:{bubble_bg}; color:{text_color}; padding:12px 16px; border-radius:12px; max-width:75%; font-size:14px; line-height:1.4; white-space:pre-wrap;'>
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================
# SHOW CHAT HISTORY
# ===========================
for m in st.session_state.messages:
    render_message(m["sender"], m["text"])

# Render any deferred charts (e.g., radar charts) after messages
if st.session_state.get("_deferred_charts"):
    # Use unique keys to avoid StreamlitDuplicateElementId when multiple charts
    for fig in st.session_state.pop("_deferred_charts"):
        unique_key = f"deferred_chart_{uuid.uuid4().hex}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)

# Render any deferred matplotlib images (from plot_radar)
if st.session_state.get("_deferred_images"):
    for img_buf in st.session_state.pop("_deferred_images"):
        st.image(img_buf, width=600)  # Fixed width in pixels (adjust as needed)


# If a scouting report was generated for the last query, render it as its own
# expandable panel so it appears with a clear "Scouting Report" identifier.
if st.session_state.get("_last_scout"):
    try:
        with st.expander("Scouting Report", expanded=False):
            st.markdown(st.session_state.pop("_last_scout"), unsafe_allow_html=True)
    except Exception:
        # If rendering fails for any reason, remove it to avoid persistent state issues
        _ = st.session_state.pop("_last_scout", None)


# ===========================
# INPUT FORM
# ===========================
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Type your question here...")
    submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():

        st.session_state.messages.append({"sender": "user", "text": user_query})

        # ===== STEP 1: Entity Extraction =====
        entities = extract_entities(
            user_query,
            players_list,
            league_list,
            stat_list
        )

        # ===== STEP 2: Intent Recognition (rule-based with entity-aware rules) =====
        intent = predict_intent(user_query, entities)

        # ===== GENERATE RESPONSE =====
        response = ""

        # ---------------------------------------------------
        # PLAYER STATS (polished)
        # Prefer exact player match, provide position-aware radar chart
        # ---------------------------------------------------
        if intent == "player_stats":
            if len(entities["players"]) == 0:
                response = "I couldn't detect which player you mean."
            else:
                query_name = entities["players"][0]

                # Use fuzzy lookup to resolve to one or more candidates
                candidates = lookup_player(query_name, df, top_n=6)
                if not candidates:
                    response = f"Couldn't find stats for {query_name}."
                else:
                    # If multiple and top score is low, ask for clarification
                    top_score = candidates[0]["score"]
                    if len(candidates) > 1 and top_score < 90:
                        # present top candidates for clarification
                        choices = [f"{c['name']} — {c.get('squad','')} ({c.get('pos','')})" for c in candidates]
                        response = "I found multiple players matching that name — please be more specific.\n"
                        for ch in choices:
                            response += f"- {ch}\n"
                    else:
                        # pick best candidate
                        chosen = candidates[0]["name"]
                        idxs = candidates[0]["indices"]
                        if not idxs:
                            response = f"Couldn't find stats for {chosen}."
                        else:
                            r = df.loc[idxs[0]]

                            # Build structured profile
                            profile = extract_player_profile(r, df)

                            # Classify role and generate insights
                            role_label, role_reasons, _ = classify_role(profile, df)
                            strengths, weaknesses, pct_map = generate_strengths_weaknesses(profile, df)

                            # Build HTML summary (compact, no empty sections)
                            def _fmt_num(v, ndigits=1):
                                try:
                                    return f"{float(v):.{ndigits}f}"
                                except Exception:
                                    return v if v is not None else "N/A"


                            def get_stat(row, *possible_names):
                                """Try multiple column names and return first available value"""
                                for name in possible_names:
                                    val = row.get(name)
                                    if val is not None and pd.notna(val):
                                        return val
                                return 'N/A'


                            outfield_match = outfield_df[
                                (outfield_df['Player'] == chosen) &
                                (outfield_df['Squad'] == profile.squad)
                                ]

                            if outfield_match.empty:
                                outfield_match = outfield_df[outfield_df['Player'] == chosen]

                            if not outfield_match.empty:
                                r_outfield = outfield_match.iloc[0]

                                # Get stats from outfield_clean.csv instead of all_leagues_clean.csv
                                goals = get_stat(r_outfield, 'Gls')
                                assists = get_stat(r_outfield, 'Ast')
                                xg = get_stat(r_outfield, 'xG', 'npxG')
                                xa = get_stat(r_outfield, 'xAG')
                                prg = get_stat(r_outfield, 'PrgP')
                            else:
                                # Fallback to all_leagues if not found in outfield
                                goals = get_stat(r, 'Performance_Gls')
                                assists = get_stat(r, 'Performance_Ast')
                                xg = get_stat(r, 'Expected_xG')
                                xa = get_stat(r, 'Expected_xAG')
                                prg = get_stat(r, 'Progression_PrgP')


                            response_html = []
                            response_html.append(f"<div style='font-size:14px; line-height:1.4;'>")
                            response_html.append(
                                f"<strong style='font-size:16px;'>{chosen} — Key Stats & Profile</strong><br>")
                            response_html.append(f"<strong>Position:</strong> {profile.pos}  <br>")
                            response_html.append(f"<strong>Squad:</strong> {profile.squad}  <br>")
                            response_html.append(f"<strong>Role:</strong> {role_label}  <br>")
                            response_html.append(
                                "<div style='display:flex; gap:12px; flex-wrap:wrap; margin-top:6px;'>"
                                + f"<div style='min-width:84px;'><strong>Goals:</strong> {goals}</div>"
                                + f"<div style='min-width:100px;'><strong>Assists:</strong> {assists}</div>"
                                + f"<div style='min-width:84px;'><strong>xG:</strong> {_fmt_num(xg, 1)}</div>"
                                + f"<div style='min-width:84px;'><strong>xA:</strong> {_fmt_num(xa, 1)}</div>"
                                + f"<div style='min-width:140px;'><strong>Progressive Passes:</strong> {prg}</div>"
                                + "</div>"
                            )

                            # Strengths
                            if strengths:
                                response_html.append("<div style='margin-top:8px;'><strong>Strengths:</strong><ul style='margin:6px 0 8px 18px;'>")
                                for s in strengths[:8]:
                                    item = s.lstrip("✓ ")
                                    # keep text readable (inherit color from bubble), color only the icon
                                    response_html.append(
                                        f"<li style='color:inherit; margin-bottom:4px;'><span style='color:#16a34a; margin-right:8px;'>✓</span>{item}</li>"
                                    )
                                response_html.append("</ul></div>")

                            # Weaknesses
                            if weaknesses:
                                response_html.append("<div style='margin-top:6px;'><strong>Weaknesses:</strong><ul style='margin:6px 0 0 18px;'>")
                                for w in weaknesses[:8]:
                                    item = w.lstrip("✗ ")
                                    response_html.append(
                                        f"<li style='color:inherit; margin-bottom:4px;'><span style='color:#ef4444; margin-right:8px;'>✗</span>{item}</li>"
                                    )
                                response_html.append("</ul></div>")

                            response_html.append("</div>")
                            response = "".join(response_html)

                            # ===== BUILD RADAR CHART USING plot_radar from radar_chart.py =====
                            # Define position-specific feature sets from outfield_clean.csv with display names
                            defense_features = {
                                'Tackles_Tkl': 'Tackles',
                                'Tackles_TklW': 'Tackles Won',
                                'Challenges_Att': 'Challenges Attempted',
                                'Challenges_Tkl%': 'Challenges Won %',
                                'Blocks_Blocks': 'Blocks',
                                'Int': 'Interceptions',
                                'Clr': 'Clearances',
                                'Att': 'Passes Attempted',
                                'Cmp%': 'Pass Completion %',
                                'PrgP': 'Progressive Passes',
                                'PrgC': 'Progressive Carries',
                            }

                            midfield_features = {
                                'Gls': 'Goals',
                                'xG': 'xG',
                                'Ast': 'Assists',
                                'xAG': 'xA',
                                'Att': 'Passes Attempted',
                                'Cmp%': 'Pass Completion %',
                                'PrgP': 'Progressive Passes',
                                'PrgC': 'Progressive Carries',
                                'PPA': 'PPA',
                                'Carries_CPA': 'CPA',
                                'KP': 'Key Passes',
                                'Challenges_Att': 'Challenges Attempted',
                                'Challenges_Tkl%': 'Challenges Won %',
                                'Take-Ons_Att': 'Dribbles Attempted',
                                'Take-Ons_Succ%': 'Dribble Success %'
                            }

                            forward_features = {
                                'Gls': 'Goals',
                                'xG': 'xG',
                                'Ast': 'Assists',
                                'xAG': 'xA',
                                'Att': 'Passes Attempted',
                                'Cmp%': 'Pass Completion %',
                                'PrgP': 'Progressive Passes',
                                'PrgC': 'Progressive Carries',
                                'PPA': 'PPA',
                                'Carries_CPA': 'CPA',
                                'KP': 'Key Passes',
                                'Take-Ons_Att': 'Dribbles Attempted',
                                'Take-Ons_Succ%': 'Dribble Success %'
                            }

                            # Determine position string from profile
                            pos = str(profile.pos).upper() if getattr(profile, 'pos', None) else ""
                            if "GK" in pos or pos.startswith("GK"):
                                preferred = {'Playing_Time_90s': '90s Played', 'Performance_CrdY': 'Yellow Cards'}
                            elif any(p in pos for p in ["DF", "D", "CB", "LB", "RB", "WB"]):
                                preferred = defense_features
                            elif any(p in pos for p in ["MF", "M", "CM", "AM", "DM", "CDM", "CAM"]):
                                preferred = midfield_features
                            elif any(p in pos for p in ["FW", "F", "ST", "CF", "W", "LW", "RW", "ATT"]):
                                preferred = forward_features
                            else:
                                preferred = midfield_features

                            # Get available features and their display names
                            available_features = {}
                            for col_name, display_name in preferred.items():
                                if col_name in outfield_df.columns:
                                    available_features[col_name] = display_name

                            # Need at least 3 features for radar
                            if len(available_features) < 3:
                                fallback = {
                                    'Performance_Gls': 'Goals',
                                    'Performance_Ast': 'Assists',
                                    'Expected_xG': 'xG',
                                    'Expected_xAG': 'xA',
                                    'Progression_PrgP': 'Progressive Passes'
                                }
                                available_features = {k: v for k, v in fallback.items() if k in outfield_df.columns}

                            # Render radar chart using plot_radar
                            if len(available_features) >= 3:
                                try:
                                    feature_cols = list(available_features.keys())
                                    display_names = list(available_features.values())

                                    df_feats = outfield_df[feature_cols].fillna(0).astype(float)

                                    # Find matching player in outfield_df
                                    outfield_match = outfield_df[
                                        (outfield_df['Player'] == chosen) &
                                        (outfield_df['Squad'] == profile.squad)
                                        ]

                                    if outfield_match.empty:
                                        outfield_match = outfield_df[outfield_df['Player'] == chosen]

                                    if not outfield_match.empty:
                                        r_outfield = outfield_match.iloc[0]

                                        player_vals = []
                                        for c in feature_cols:
                                            val = r_outfield.get(c, 0)
                                            try:
                                                player_vals.append(float(val) if pd.notna(val) else 0.0)
                                            except Exception:
                                                player_vals.append(0.0)

                                        # Calculate percentile ranks (0-100 scale)
                                        norm_player = []
                                        for i, c in enumerate(feature_cols):
                                            # Get all values for this feature
                                            all_values = df_feats[c]
                                            player_value = player_vals[i]

                                            # Calculate percentile rank
                                            percentile = (all_values < player_value).sum() / len(all_values) * 100

                                            norm_player.append(int(percentile))

                                        # Create dataframe for plot_radar
                                        radar_dict = {}
                                        for i, display_name in enumerate(display_names):
                                            radar_dict[display_name] = norm_player[i]
                                        radar_data = pd.DataFrame([radar_dict])
                                        radar_data['player'] = chosen
                                        radar_data['team'] = profile.squad if hasattr(profile, 'squad') else 'Unknown'

                                        # Call plot_radar with display_names as the cols parameter
                                        plot_radar(radar_data, display_names, df2=None)

                                        # Get the current figure
                                        fig = plt.gcf()

                                        # Convert to image
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
                                        buf.seek(0)
                                        plt.close(fig)

                                        # Queue for display
                                        st.session_state.setdefault("_deferred_images", []).append(buf)

                                except Exception as e:
                                    import traceback

                                    response += f"\n\n(Note: could not render radar chart: {e})\n{traceback.format_exc()}"

                                # Attempt to generate a scouting report and store it
                                try:
                                    model = get_embedding_model()
                                    if model is not None:
                                        try:
                                            from src.modeling.scouting import generate_scouting_report

                                            scout_html = generate_scouting_report(model, chosen, top_k=6)
                                            # Store scouting HTML as its own identifier in session state
                                            st.session_state["_last_scout"] = scout_html
                                        except Exception:
                                            # non-fatal: skip scouting if it errors
                                            pass
                                except Exception:
                                    # ignore errors getting the model
                                    pass

        # ---------------------------------------------------
        # COMPARE PLAYERS
        # ---------------------------------------------------
        elif intent == "compare_players":
            if len(entities["players"]) < 2:
                response = "Provide two players to compare."
            else:
                p1, p2 = entities["players"][:2]

                r1 = df[df["Player"].str.contains(p1, case=False, na=False)]
                r2 = df[df["Player"].str.contains(p2, case=False, na=False)]

                if r1.empty or r2.empty:
                    response = "One of the players couldn't be found."
                else:
                    a, b = r1.iloc[0], r2.iloc[0]
                    # Build a readable HTML comparison table (larger names, clear stat columns)
                    try:
                        left_g = a.get('Performance_Gls', 'N/A')
                        right_g = b.get('Performance_Gls', 'N/A')
                        left_ast = a.get('Performance_Ast', 'N/A')
                        right_ast = b.get('Performance_Ast', 'N/A')
                        left_xg = a.get('Expected_xG', 'N/A')
                        right_xg = b.get('Expected_xG', 'N/A')
                        left_prg = a.get('Progression_PrgP', 'N/A')
                        right_prg = b.get('Progression_PrgP', 'N/A')

                        response_html = []
                        response_html.append("<div style='font-size:15px; line-height:1.45;'>")
                        response_html.append(f"<div style='display:flex; justify-content:space-between; align-items:center;'>")
                        response_html.append(f"<div style='font-weight:700; color:inherit; font-size:18px;'>🔵 {p1}</div>")
                        response_html.append(f"<div style='font-weight:700; font-size:16px; text-align:center;'>Comparison</div>")
                        response_html.append(f"<div style='font-weight:700; color:inherit; font-size:18px; text-align:right;'>🔴 {p2}</div>")
                        response_html.append("</div>")

                        response_html.append("<table style='width:100%; margin-top:8px; border-collapse:collapse;'>")
                        # row helper
                        def _row(l, label, r):
                            return (
                                "<tr>"
                                f"<td style='width:40%; padding:6px; font-size:15px; font-weight:600; color:inherit;'>{l}</td>"
                                f"<td style='width:20%; padding:6px; text-align:center; color:inherit; font-size:14px;'>{label}</td>"
                                f"<td style='width:40%; padding:6px; font-size:15px; font-weight:600; color:inherit; text-align:right;'>{r}</td>"
                                "</tr>"
                            )

                        response_html.append(_row(left_g, 'Goals', right_g))
                        response_html.append(_row(left_ast, 'Assists', right_ast))
                        response_html.append(_row(left_xg, 'xG', right_xg))
                        response_html.append(_row(left_prg, 'Progressive Passes', right_prg))

                        response_html.append("</table>")
                        response_html.append("</div>")

                        # Attempt to run the richer comparison engine and render its
                        # NLP-style outputs below the numeric table.
                        try:
                            from src.modeling.compare import PlayerComparisonEngine

                            try:
                                engine = PlayerComparisonEngine(model_path="data/models/player_embedding_model.pkl")
                            except Exception:
                                engine = PlayerComparisonEngine(df=df)

                            comp = engine.compare(p1, p2, top_k=5)

                            response_html.append("<hr style='opacity:0.12;'/>")
                            response_html.append(f"<div style='margin-top:8px;'><strong>Comparison role summary:</strong> {comp.get('role_info','')}</div>")
                            if comp.get('shared_strengths'):
                                response_html.append(f"<div style='margin-top:6px;'><strong>Shared strengths:</strong> {', '.join(comp['shared_strengths'])}</div>")
                            if comp.get('advantages_player_a'):
                                response_html.append(f"<div style='margin-top:6px;'><strong>Advantages — {p1}:</strong> {', '.join(comp['advantages_player_a'])}</div>")
                            if comp.get('advantages_player_b'):
                                response_html.append(f"<div style='margin-top:6px;'><strong>Advantages — {p2}:</strong> {', '.join(comp['advantages_player_b'])}</div>")
                            if comp.get('final_summary'):
                                response_html.append(f"<div style='margin-top:8px;'><em>{comp['final_summary']}</em></div>")
                        except Exception as e:
                            response_html.append(f"<div style='margin-top:8px; color:inherit;'>(Could not compute detailed comparison: {e})</div>")

                        response = "".join(response_html)
                    except Exception as e:
                        # Fallback to previous markdown if anything goes wrong
                        response = (
                            f"### 🔵 {p1} vs 🔴 {p2}\n"
                            f"**Goals:** {a['Performance_Gls']} vs {b['Performance_Gls']}\n"
                            f"**Assists:** {a['Performance_Ast']} vs {b['Performance_Ast']}\n"
                            f"**xG:** {a['Expected_xG']} vs {b['Expected_xG']}\n"
                            f"**Progressive Passes:** {a['Progression_PrgP']} vs {b['Progression_PrgP']}"
                        )

        # ---------------------------------------------------
        # SIMILAR PLAYERS
        # ---------------------------------------------------
        elif intent == "similar_players":
            if len(entities["players"]) == 0:
                response = "Tell me which player you want similar profiles for."
            else:
                target = entities["players"][0]

                try:
                    # Prefer the embedding model (cached) which provides explanations
                    model = get_embedding_model()
                    if model is not None:
                        sims_df = model.get_similar_players(target, top_k=6)

                        response_html = []
                        response_html.append(f"<div style='font-size:15px; line-height:1.45;'><strong>Players similar to <em>{target}</em>:</strong>")
                        response_html.append("<ul style='margin:6px 0 6px 18px;'>")

                        for _, row in sims_df.iterrows():
                            name = row["Player"]
                            squad = row.get("Squad", "") if "Squad" in row.index else ""
                            pos = row.get(model.pos_col, "") if model.pos_col and model.pos_col in row.index else ""
                            score = row.get("similarity", None)
                            # Show one decimal place to avoid rounding near-100% misleading outputs
                            score_label = f"{score*100:.1f}%" if isinstance(score, (float, int)) else ""
                            meta = " • ".join([p for p in [pos, squad] if p])
                            meta_label = f"<span style='font-size:13px; color:inherit; opacity:0.9;'>{score_label}{(' • ' + meta) if meta else ''}</span>"
                            response_html.append(f"<li style='margin-bottom:8px; color:inherit;'><span style='font-size:15px; font-weight:600;'>{name}</span> {meta_label}</li>")

                        response_html.append("</ul>")

                        # Also show the three least similar players (global lowest cosine similarity)
                        try:
                            from sklearn.metrics.pairwise import cosine_similarity
                            import numpy as _np

                            tidx = model._match_player_index(target)
                            sims_all = cosine_similarity(model.embeddings[tidx].reshape(1, -1), model.embeddings)[0]
                            asc = _np.argsort(sims_all)  # ascending (least similar first)

                            response_html.append("<div style='margin-top:8px;'><strong>Least similar players:</strong></div>")
                            response_html.append("<ul style='margin:6px 0 6px 18px;'>")
                            count = 0
                            for ii in asc:
                                if ii == tidx:
                                    continue
                                name_l = model.df.iloc[ii]["Player"]
                                squad_l = model.df.iloc[ii]["Squad"] if "Squad" in model.df.columns else ""
                                pos_l = model.df.iloc[ii][model.pos_col] if model.pos_col and model.pos_col in model.df.columns else ""
                                score_l = sims_all[ii]
                                score_label_l = f"{score_l*100:.1f}%"
                                meta_l = " • ".join([p for p in [pos_l, squad_l] if p])
                                response_html.append(f"<li style='margin-bottom:8px; color:inherit;'><span style='font-size:15px; font-weight:600;'>{name_l}</span> <span style='font-size:13px; color:inherit; opacity:0.9;'>{score_label_l}{(' • ' + meta_l) if meta_l else ''}</span></li>")
                                count += 1
                                if count >= 3:
                                    break
                            response_html.append("</ul>")
                        except Exception:
                            # If anything goes wrong computing global similarities, skip silently
                            pass

                        # Add explanation vs top match
                        if not sims_df.empty:
                            try:
                                top_match = sims_df.iloc[0]["Player"]
                                expl = model.explain_pair(target, top_match)
                                response_html.append("<hr style='opacity:0.12;'/>")
                                response_html.append(f"<div style='margin-top:8px;'><strong>Role summary:</strong> {expl.get('role_info','')}</div>")
                                # Attempt to include richer role metadata from ROLE_LABELS
                                try:
                                    top_cluster = int(sims_df.iloc[0]["role_cluster"])
                                    role_meta = ROLE_LABELS.get(top_cluster)
                                    if isinstance(role_meta, dict):
                                        profile = role_meta.get('profile')
                                        examples = role_meta.get('examples')
                                        summary = role_meta.get('summary')
                                        if profile:
                                            response_html.append(f"<div style='margin-top:6px;'><strong>Role profile:</strong> {profile}</div>")
                                        if examples:
                                            ex_list = ', '.join(examples[:6])
                                            response_html.append(f"<div style='margin-top:6px;'><strong>Examples:</strong> {ex_list}</div>")
                                        if summary:
                                            response_html.append(f"<div style='margin-top:6px;'><em>{summary}</em></div>")
                                except Exception:
                                    # ignore failures to keep UI stable
                                    pass

                                if expl.get("shared_strengths"):
                                    response_html.append(f"<div style='margin-top:6px;'><strong>Shared strengths:</strong> {', '.join(expl['shared_strengths'])}</div>")
                                if expl.get("style_summary"):
                                    response_html.append(f"<div style='margin-top:6px;'><em>{expl['style_summary']}</em></div>")
                            except Exception as e:
                                response_html.append(f"<div style='margin-top:8px; font-size:13px; color:inherit;'>(Could not compute explanation: {e})</div>")

                        response_html.append("</div>")
                        response = "".join(response_html)
                    else:
                        # Fallback: use the lightweight wrapper that returns names only
                        sims = find_similar_players(target, df, top_k=6)
                        response_html = []
                        response_html.append(f"<div style='font-size:15px; line-height:1.45;'><strong>Players similar to <em>{target}</em>:</strong>")
                        response_html.append("<ul style='margin:6px 0 6px 18px;'>")
                        for name in sims:
                            response_html.append(f"<li style='margin-bottom:6px; color:inherit;'>{name}</li>")
                        response_html.append("</ul>")
                        response_html.append("</div>")
                        response = "".join(response_html)
                except Exception as e:
                    response = f"Could not compute similarity: {e}"

        # ---------------------------------------------------
        # LEAGUE FIT
        # ---------------------------------------------------
        elif intent == "league_fit":
            player = entities["players"][0] if entities["players"] else None
            league = entities["league"]

            if not player or not league:
                response = "Please specify both a player and a league."
            else:
                response = (
                    f"### {player} → {league.title()}\n"
                    f"This is a placeholder league-fit heuristic. "
                    f"League fit modeling launching in Milestone 3."
                )

        # ---------------------------------------------------
        # UNKNOWN INTENT
        # ---------------------------------------------------
        else:
            response = "I'm not sure what you mean — try asking about stats, comparisons, or similar players."

        # Add bot message
        st.session_state.messages.append({"sender": "bot", "text": response})
        st.rerun()