import re


def extract_entities(query, players_list, league_list, stat_list):
    """Lightweight entity extractor: finds players, leagues, and stat keywords in text."""

    query_lower = query.lower()

    # Players
    players_found = [p for p in players_list if p.lower() in query_lower]

    # Leagues
    league_found = next((l for l in league_list if l.lower() in query_lower), None)

    # Stats
    stats_found = [s for s in stat_list if s.lower() in query_lower]

    return {
        "players": players_found,
        "league": league_found,
        "stats": stats_found
    }
