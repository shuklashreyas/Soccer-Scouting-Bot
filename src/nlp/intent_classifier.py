import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os


os.makedirs("models", exist_ok=True)  # ✅ Add this line

    
# ===== Training Data =====
data = {
    "text": [
        "Show me Haaland’s goals",
        "Find Messi’s assists",
        "Compare Haaland and Mbappé",
        "Players like Pedri",
        "Who is similar to Bellingham",
        "How would Mbappé fit in La Liga",
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

# ===== Vectorize =====
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["intent"]

# ===== Train Model =====
model = LogisticRegression()
model.fit(X, y)

# ===== Save Model & Vectorizer =====
with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Intent Recognition Model trained and saved!")
