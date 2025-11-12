import streamlit as st
from intent_model import detect_intent
from entity_extraction import extract_entities
from similarity import build_similarity, find_similar_players
import pandas as pd

st.title("⚽ NLP Soccer Scouting Chatbot")

#