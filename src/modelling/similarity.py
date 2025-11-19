from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def build_similarity(df, features):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    model.fit(X_scaled)
    return model, scaler

def find_similar_players(player_name, df, model, scaler, features):
    X_scaled = scaler.transform(df[features].values)
    idx = df.index[df['player'].str.lower() == player_name.lower()].tolist()
    if not idx:
        return []
    distances, indices = model.kneighbors([X_scaled[idx[0]]])
    similar_players = df.iloc[indices[0]]['player'].tolist()[1:]
    return similar_players
