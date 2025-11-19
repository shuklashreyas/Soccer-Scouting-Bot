import streamlit as st
import pandas as pd
import pickle
import sys
from pathlib import Path

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
        # PLAYER STATS
        # ---------------------------------------------------
        if intent == "player_stats":
            if len(entities["players"]) == 0:
                response = "I couldn’t detect which player you mean."
            else:
                player = entities["players"][0]
                row = df[df["Player"].str.contains(player, case=False, na=False)]
                if row.empty:
                    response = f"Couldn't find stats for {player}."
                else:
                    r = row.iloc[0]
                    response = (
                        f"### {player} — Key Stats\n"
                        f"- Goals: {r.get('Performance_Gls', 'N/A')}\n"
                        f"- Assists: {r.get('Performance_Ast', 'N/A')}\n"
                        f"- xG: {r.get('Expected_xG', 'N/A')}\n"
                        f"- Progressive Passes: {r.get('Progression_PrgP', 'N/A')}"
                    )

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
                    sims = find_similar_players(target, df)
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
