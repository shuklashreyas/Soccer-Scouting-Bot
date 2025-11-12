import streamlit as st
import pandas as pd
from typing import List, Dict


st.set_page_config(page_title="Soccer Scouting Chatbot", layout="wide")


CSS = """
<style>
body { background: linear-gradient(180deg, #0f1723 0%, #071026 100%); }
.chat-container { max-width: 900px; margin: 0 auto; }
.msg { padding: 12px 16px; border-radius: 12px; margin: 8px 0; display: inline-block; }
.msg.user { background: #dbeafe; color: #012a4a; float: right; }
.msg.bot { background: #d1fae5; color: #064e3b; float: left; }
.clear { clear: both; }
.player-card { background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); padding: 12px; border-radius: 10px; }
.muted { color: #9aa4b2; font-size: 0.9em }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.title("⚽ Soccer Scouting Chatbot — UI Prototype")
st.markdown("""
Small interactive UI for scouting flows. This is a front-end only prototype — buttons and inputs show placeholders and sample outputs.
""")


def sample_players() -> List[Dict]:
	return [
		{"name": "Joao Cancelo", "pos": "RB/LB", "team": "Top Club", "img": "https://via.placeholder.com/120"},
		{"name": "Alphonso Davies", "pos": "LB", "team": "Top Club", "img": "https://via.placeholder.com/120"},
		{"name": "Trent Alexander-Arnold", "pos": "RB", "team": "Top Club", "img": "https://via.placeholder.com/120"},
		{"name": "Kieran Tierney", "pos": "LB", "team": "Mid Club", "img": "https://via.placeholder.com/120"}
	]


if "messages" not in st.session_state:
	st.session_state.messages = [
		{"sender": "bot", "text": "Hi — ask me about players, comparisons, or find similar profiles."}
	]


def render_message(m: Dict):
	sender = m.get("sender", "bot")
	text = m.get("text", "")
	cls = "bot" if sender == "bot" else "user"
	st.markdown(f"<div class='msg {cls}'>{text}</div><div class='clear'></div>", unsafe_allow_html=True)


with st.sidebar:
	st.header("Navigation")
	page = st.radio("Go to", ["Home", "Player Profile", "Compare Players", "Find Similar", "League Fit"]) 
	st.markdown("---")
	st.markdown("Authors: Shreyas, Ansh, Komdean, Ethan, Youssof")
	st.markdown("""
	<div class='muted'>This UI is a prototype and does not call any backend models. It's ready for wiring.</div>
	""", unsafe_allow_html=True)


def add_user_message(text: str):
	st.session_state.messages.append({"sender": "user", "text": text})
	st.session_state.messages.append({"sender": "bot", "text": "(placeholder response) — this would show a concise scouting rationale and stats visualizations."})


## Home
if page == "Home":
	st.subheader("Chat — prototype")
	chat_col, viz_col = st.columns((2, 1))

	with chat_col:
		for m in st.session_state.messages:
			render_message(m)

		with st.form(key="chat_form", clear_on_submit=True):
			user_input = st.text_input("Ask me about a player or request a flow (e.g. 'Compare X vs Y')")
			submit = st.form_submit_button("Send")
			if submit and user_input:
				add_user_message(user_input)

		st.markdown("\n")

	with viz_col:
		st.markdown("### Quick Actions")
		if st.button("Show sample Player Card"):
			st.session_state.messages.append({"sender": "bot", "text": "Showing a sample player card (open Player Profile page)."})
		st.button("Export shortlist (disabled)", disabled=True)


## Player Profile
elif page == "Player Profile":
	st.subheader("Player Profile — sample output")
	players = sample_players()
	names = [p["name"] for p in players]
	sel = st.selectbox("Select player", names)
	p = next((x for x in players if x["name"] == sel), players[0])

	left, right = st.columns((1, 2))
	with left:
		st.image(p["img"], width=140)
		st.markdown(f"### {p['name']}")
		st.markdown(f"**Position:** {p['pos']}")
		st.markdown(f"**Team:** {p['team']}")
		st.markdown("\n")
		st.button("Add to shortlist (no-op)")

	with right:
		st.markdown("**Concise scouting summary**")
		st.info("Pacey fullback with strong progressive passing. Aerials and long duels below average.")
		st.markdown("**Strengths**")
		st.write("- Progressive carries\n- Passes into final third\n- Versatile wide positioning")
		st.markdown("**Key metrics (sample)**")
		df = pd.DataFrame({
			"Metric": ["Progressive Passes", "Progressive Carries", "Aerials Won %", "Shot-creating Actions"],
			"Value": [12.4, 7.5, 38.0, 2.1]
		})
		st.table(df)


## Compare Players
elif page == "Compare Players":
	st.subheader("Compare two players (sample)")
	players = sample_players()
	names = [p["name"] for p in players]
	a, b = st.columns(2)
	with a:
		p1 = st.selectbox("Player A", names, index=0)
	with b:
		p2 = st.selectbox("Player B", names, index=1)

	if p1 == p2:
		st.warning("Pick two different players to compare.")
	else:
		left, right = st.columns(2)
		with left:
			st.markdown(f"### {p1}")
			st.markdown("**Profile highlights**")
			st.write("- Progressive passing\n- Good crossing\n- Creative outlet")
		with right:
			st.markdown(f"### {p2}")
			st.markdown("**Profile highlights**")
			st.write("- Rapid recoveries\n- Defensive positioning\n- Less creative in final third")


## Find Similar
elif page == "Find Similar":
	st.subheader("Find similar players")
	players = [p["name"] for p in sample_players()]
	sel = st.selectbox("Player to find similars for", players)
	if st.button("Show top-5 similar"):
		st.markdown("**Top-5 similar players (sample)**")
		for i, s in enumerate(players[::-1], 1):
			st.markdown(f"{i}. **{s}** — similar in progressive passing and width use.")


## League Fit
elif page == "League Fit":
	st.subheader("League-fit preview (heuristic)")
	league = st.selectbox("Choose league", ["Premier League", "LaLiga", "Bundesliga", "MLS"])
	uncertainty = st.slider("Uncertainty (display only)", 0.0, 1.0, 0.35)
	st.markdown(f"**Heuristic fit for** {league}")
	st.progress(int((1 - uncertainty) * 100))
	st.markdown("Rationale: This player profile emphasizes progressive passing and high-possession involvement, which suits ball-dominant leagues.")


## Footer
st.markdown("---")
st.markdown("Prototype UI only — no models invoked. Run `streamlit run src/app.py` to view locally.")
