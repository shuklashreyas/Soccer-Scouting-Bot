import streamlit as st
import pandas as pd
import pickle
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import uuid

# ======================================
# FIX MODULE PATHS
# ======================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Your modules
from src.nlp.intent_classifier import predict_intent
from src.nlp.entity_extraction import extract_entities
from src.modeling.similarity import find_similar_players
from src.player.lookup import lookup_player
from src.player.extract import extract_player_profile
from src.player.roles import classify_role
from src.player.insights import generate_strengths_weaknesses


# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    df_path = "data/processed/all_leagues_clean.csv"

    df = pd.read_csv(df_path)

    players = df["Player"].dropna().unique().tolist()
    leagues = ["premier league", "la liga", "bundesliga", "serie a", "ligue 1"]

    # These stats names are used only for entity extraction
    stats = ["goals", "assists", "xg", "xa", "progressive passes", "ppa", "g+a"]

    return df, players, leagues, stats


df, players_list, league_list, stat_list = load_data()


# ======================================
# LOAD SIMILARITY MODEL (OPTIONAL)
# Your new wrapper doesn't need scaler/knn but we keep it for compatibility
# ======================================
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


# ===========================
# INPUT FORM
# ===========================
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Type your question here...")
    submitted = st.form_submit_button("Send")

    if submitted and user_query.strip():

        st.session_state.messages.append({"sender": "user", "text": user_query})

        # ===== STEP 1: Intent Recognition =====
        intent = predict_intent(user_query)

        # ===== STEP 2: Entity Extraction =====
        entities = extract_entities(
            user_query,
            players_list,
            league_list,
            stat_list
        )

        # ===== GENERATE RESPONSE =====
        response = ""

        # ---------------------------------------------------
        # PLAYER STATS (polished)
        # Prefer exact player match, provide position-aware radar chart
        # ---------------------------------------------------
        if intent == "player_stats":
            if len(entities["players"]) == 0:
                response = "I couldn’t detect which player you mean."
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

                            # Textual summary (key stats + role + insights)
                            response = (
                                f"### {chosen} — Key Stats & Profile\n"
                                f"**Position:** {profile.pos}  \n"
                                f"**Squad:** {profile.squad}  \n"
                                f"**Role:** {role_label}  \n"
                                f"- Goals: {profile.stats['shooting'].get('goals', 'N/A')}  \n"
                                f"- Assists: {profile.stats['assisting'].get('assists', 'N/A')}  \n"
                                f"- xG: {profile.stats['shooting'].get('xg', 'N/A')}  \n"
                                f"- Progressive Passes: {profile.stats['progression'].get('prg_passes', 'N/A')}\n\n"
                            )

                            # Add strengths/weaknesses section
                            if strengths:
                                response += "**Strengths:**\n"
                                for s in strengths:
                                    response += f"- {s}\n"
                                response += "\n"
                            if weaknesses:
                                response += "**Weaknesses:**\n"
                                for w in weaknesses:
                                    response += f"- {w}\n"
                                response += "\n"

                            # Build radar chart using previous logic: pick feature_cols_to_use
                            # Prefer trained feature_cols if available
                            if feature_cols is not None:
                                trained_cols = [c for c in feature_cols if c in df.columns]
                                if len(trained_cols) >= 2:
                                    feature_cols_to_use = trained_cols
                                else:
                                    feature_cols_to_use = [c for c in [
                                        "Per_90_Minutes_xG", "Per_90_Minutes_xAG", "Progression_PrgP", "Playing_Time_90s",
                                    ] if c in df.columns]
                            else:
                                feature_cols_to_use = [c for c in [
                                    "Per_90_Minutes_xG", "Per_90_Minutes_xAG", "Progression_PrgP", "Playing_Time_90s",
                                ] if c in df.columns]

                            # Fallback
                            if len(feature_cols_to_use) < 3:
                                feature_cols_to_use = [c for c in ["Per_90_Minutes_xG", "Per_90_Minutes_xAG", "Per_90_Minutes_Gls"] if c in df.columns]

                            if feature_cols_to_use:
                                try:
                                    df_feats = df[feature_cols_to_use].fillna(0).astype(float)
                                    player_vals = []
                                    for c in feature_cols_to_use:
                                        v = r.get(c, 0)
                                        try:
                                            player_vals.append(float(v) if pd.notna(v) else 0.0)
                                        except Exception:
                                            player_vals.append(0.0)

                                    mins = df_feats.min()
                                    maxs = df_feats.max()

                                    norm_player = []
                                    norm_avg = []
                                    for i, c in enumerate(feature_cols_to_use):
                                        lo = mins[c]
                                        hi = maxs[c]
                                        if hi == lo:
                                            p_norm = 50.0
                                            a_norm = 50.0
                                        else:
                                            p_norm = 100.0 * (player_vals[i] - lo) / (hi - lo)
                                            a_norm = 100.0 * (df_feats[c].mean() - lo) / (hi - lo)
                                        norm_player.append(max(0, min(100, p_norm)))
                                        norm_avg.append(max(0, min(100, a_norm)))

                                    labels = [c.replace("_", " ") for c in feature_cols_to_use]
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatterpolar(r=norm_avg + [norm_avg[0]], theta=labels + [labels[0]], fill='toself', name='League avg', line=dict(color='rgba(200,200,200,0.6)')))
                                    fig.add_trace(go.Scatterpolar(r=norm_player + [norm_player[0]], theta=labels + [labels[0]], fill='toself', name=chosen, line=dict(color='#0f766e')))
                                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True, margin=dict(l=20,r=20,t=30,b=20))
                                    st.session_state.setdefault("_deferred_charts", []).append(fig)
                                except Exception as e:
                                    response += f"\n\n(Note: could not render radar chart: {e})"

                    attack_features = [
                        "Per_90_Minutes_xG",
                        "Per_90_Minutes_xAG",
                        "Per_90_Minutes_Gls",
                        "Per_90_Minutes_Ast",
                        "Progression_PrgP",
                    ]
                    mid_features = [
                        "Per_90_Minutes_xG",
                        "Per_90_Minutes_xAG",
                        "Per_90_Minutes_Ast",
                        "Per_90_Minutes_G+A",
                        "Progression_PrgP",
                    ]
                    def_features = [
                        "Progression_PrgP",
                        "Progression_PrgC",
                        "Per_90_Minutes_xG",
                        "Per_90_Minutes_xAG",
                        "Per_90_Minutes_G+A",
                    ]
                    gk_features = [
                        "Playing_Time_90s",
                        "Performance_CrdY",
                    ]

                    # determine position string from profile
                    pos = str(profile.pos).upper() if getattr(profile, 'pos', None) else ""
                    if "GK" in pos or pos.startswith("GK"):
                        preferred = gk_features
                    elif any(p in pos for p in ["F", "FW", "ST", "CF", "W"]):
                        preferred = attack_features
                    elif any(p in pos for p in ["M", "CM", "AM", "DM"]):
                        preferred = mid_features
                    else:
                        preferred = def_features

                    # Prefer trained feature columns from pickle when available
                    if feature_cols is not None:
                        trained_cols = [c for c in feature_cols if c in df.columns]
                        if len(trained_cols) >= 2:
                            feature_cols_to_use = trained_cols
                        else:
                            feature_cols_to_use = [c for c in preferred if c in df.columns]
                    else:
                        feature_cols_to_use = [c for c in preferred if c in df.columns]

                    # Fallback generic set
                    if len(feature_cols_to_use) < 3:
                        fallback = [
                            "Per_90_Minutes_xG",
                            "Per_90_Minutes_xAG",
                            "Progression_PrgP",
                            "Per_90_Minutes_Gls",
                        ]
                        feature_cols_to_use = [c for c in fallback if c in df.columns]

                    # Render radar chart comparing player to dataset average
                    if feature_cols_to_use:
                        try:
                            df_feats = df[feature_cols_to_use].fillna(0).astype(float)

                            player_vals = []
                            for c in feature_cols_to_use:
                                val = r.get(c, 0)
                                try:
                                    player_vals.append(float(val) if pd.notna(val) else 0.0)
                                except Exception:
                                    player_vals.append(0.0)

                            mins = df_feats.min()
                            maxs = df_feats.max()

                            norm_player = []
                            norm_avg = []
                            for i, c in enumerate(feature_cols_to_use):
                                lo = mins[c]
                                hi = maxs[c]
                                if hi == lo:
                                    p_norm = 50.0
                                    a_norm = 50.0
                                else:
                                    p_norm = 100.0 * (player_vals[i] - lo) / (hi - lo)
                                    a_norm = 100.0 * (df_feats[c].mean() - lo) / (hi - lo)
                                norm_player.append(max(0, min(100, p_norm)))
                                norm_avg.append(max(0, min(100, a_norm)))

                            labels = [c.replace("_", " ") for c in feature_cols_to_use]

                            fig = go.Figure()
                            fig.add_trace(go.Scatterpolar(r=norm_avg + [norm_avg[0]], theta=labels + [labels[0]], fill='toself', name='League avg', line=dict(color='rgba(200,200,200,0.6)')))
                            fig.add_trace(go.Scatterpolar(r=norm_player + [norm_player[0]], theta=labels + [labels[0]], fill='toself', name=chosen, line=dict(color='#0f766e')))
                            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=True, margin=dict(l=20,r=20,t=30,b=20))

                            # queue chart for rendering after messages
                            st.session_state.setdefault("_deferred_charts", []).append(fig)
                        except Exception as e:
                            response += f"\n\n(Note: could not render radar chart: {e})"

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
                    sims = find_similar_players(target, df, scaler=scaler, knn=knn, feature_cols=feature_cols)
                    response = f"### Players similar to **{target}**:\n"
                    for s in sims:
                        response += f"- {s}\n"
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
