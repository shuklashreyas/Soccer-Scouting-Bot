import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# ===== Training Data =====
data = {
    "text": [
        "Show me Haaland's goals",
        "Find Messi's assists",
        "Compare Haaland and Mbappé",
        "Players like Pedri",
        "Who is similar to Bellingham?",
        "How would Mbappé fit in La Liga?",
        "Stats for Bruno Fernandes",
        "Show me top scorers in Premier League",
    ],
    "intent": [
        "player_stats",
        "player_stats",
        "compare_players",
        "similar_players",
        "similar_players",
        "league_fit",
        "player_stats",
        "player_stats",
    ]
}

df = pd.DataFrame(data)

# TF–IDF Vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])
y = df["intent"]

# Logistic Regression
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Save models
os.makedirs("data/models", exist_ok=True)

with open("data/models/intent_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/models/intent_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Intent model trained and saved!")
