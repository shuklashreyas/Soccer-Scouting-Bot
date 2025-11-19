import re

# Example lists — can be loaded from your scraped data
players = ["Haaland", "Mbappé", "Pedri", "Messi", "Bellingham", "Bruno Fernandes"]
stats = ["goals", "assists", "xG", "xA"]
leagues = ["Premier League", "La Liga", "Serie A", "Bundesliga"]

def extract_entities(query):
    found_players = [p for p in players if p.lower() in query.lower()]
    found_stats = [s for s in stats if s.lower() in query.lower()]
    found_leagues = [l for l in leagues if l.lower() in query.lower()]

    return {
        "players": found_players,
        "stats": found_stats,
        "leagues": found_leagues
    }

# Example usage
example = "Compare Haaland and Mbappé in La Liga"
print(extract_entities(example))
