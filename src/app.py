import streamlit as st
from intent_model import detect_intent
from entity_extraction import extract_entities
from similarity import build_similarity, find_similar_players
import pandas as pd

st.title("⚽ NLP Soccer Scouting Chatbot")

df = pd.read_csv("data/processed/premier_league_clean.csv")
players = df['player'].tolist()
leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A"]
stats = ["goals", "assists", "xg", "xA", "ppa"]

query = st.text_input("Ask me something like: Find players like Pedri")

if query:
    intent = detect_intent(query)
    entities = extract_entities(query, players, leagues, stats)

    if intent == "similarity" and entities['player']:
        model, scaler = build_similarity(df, stats[0:3])  # example features
        result = find_similar_players(entities['player'][0], df, model, scaler, stats[0:3])
        st.success(f"Players similar to {entities['player'][0]}: {', '.join(result)}")
