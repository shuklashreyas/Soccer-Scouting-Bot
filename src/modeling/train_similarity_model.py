"""Train a lightweight kNN similarity model (scaler + kNN) and persist it.

This script fits a scaler and a NearestNeighbors model on z-score features
and writes `data/models/similarity_model.pkl` for legacy compatibility.
"""

import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

print("📘 Loading cleaned dataset...")
df = pd.read_csv("../../data/processed/z_scores.csv")

# Important features
feature_cols = [
    # Use column names that match `all_leagues_clean.csv` (underscores)
    "Shooting_Score",
    "Passing_Score",
    "Defending_Score",
    "Carrying_Score",
    "Dribbling_Score",
    "Creation_Score"
]

# Filter valid rows
df = df.dropna(subset=feature_cols)

X = df[feature_cols].values

print("📊 Fitting scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🤝 Training kNN similarity model...")
knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
knn.fit(X_scaled)

# Save model
os.makedirs("data/models", exist_ok=True)
with open("data/models/similarity_model.pkl", "wb") as f:
    pickle.dump((scaler, knn, feature_cols), f)

print("✅ similarity_model.pkl has been saved!")
