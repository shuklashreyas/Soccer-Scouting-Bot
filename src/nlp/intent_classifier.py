import re
import pickle
from typing import Dict, Optional

# Try to load an ML classifier if present (optional). If not available, we'll fall back to rule-based.
try:
    with open("data/models/intent_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("data/models/intent_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
except Exception:
    vectorizer = None
    clf = None


# --- Expanded keyword banks with weights (Layer 1 + Layer 3) ---
# Weights reflect signal strength for each keyword/phrase (higher => stronger signal)
STATS_KEYWORDS_WEIGHTS = {
    # strong stat indicators
    "stats": 3, "statistics": 3, "profile": 2, "overview": 2, "key stats": 2, "performance": 2,
    # specific metrics
    "goals": 2, "assists": 2, "xg": 2, "xa": 2, "g+a": 2, "per 90": 1, "per90": 1,
    "passing": 1, "defensive": 1, "attacking": 1, "metrics": 1, "details": 1, "breakdown": 1,
    "how good": 1, "how good is": 1, "show me": 1, "tell me about": 1,
}

COMPARE_KEYWORDS_WEIGHTS = {
    "compare": 3, " vs ": 3, "versus": 3, "difference": 2, "better than": 2, "who is better": 2,
    "head to head": 2, "side by side": 2, "matchup": 1,
}

SIMILAR_KEYWORDS_WEIGHTS = {
    "similar": 3, "like": 1, "players like": 3, "who plays like": 3, "replacement": 2, "alternative": 2,
    "same style": 2, "closest match": 2, "same archetype": 2,
}

LEAGUE_KEYWORDS_WEIGHTS = {
    "fit": 3, "good for": 3, "suit": 2, "suited for": 2, "match": 1, "would .* fit": 3, "fit in": 2,
    "prem ready": 2, "best league for": 2, "would .* succeed": 2,
}

# --- Precise phrase patterns (Layer 4) ---
# Patterns return strong intent when matched; these are applied before weighted scoring.
PATTERN_INTENTS = [
    (re.compile(r"(.+?)\s+vs\.??\s+(.+?)", re.I), "compare_players"),
    (re.compile(r"compare\s+(.+?)\s+(and|vs|versus)\s+(.+?)", re.I), "compare_players"),
    (re.compile(r"who plays like\s+(.+)", re.I), "similar_players"),
    (re.compile(r"players like\s+(.+)", re.I), "similar_players"),
    (re.compile(r"similar to\s+(.+)", re.I), "similar_players"),
    (re.compile(r"is\s+(.+?)\s+(good|fit)\s+for\s+(.+)", re.I), "league_fit"),
    (re.compile(r"would\s+(.+?)\s+(fit|succeed|thrive)\s+in\s+(.+)", re.I), "league_fit"),
    (re.compile(r"show me\s+(.+?)'s\s+stats", re.I), "player_stats"),
    (re.compile(r"tell me about\s+(.+)", re.I), "player_stats"),
]


def _count_weighted_matches(text: str, weights: Dict[str, int]) -> int:
    """Sum weights for any keyword occurrences in text."""
    total = 0
    for kw, w in weights.items():
        try:
            if re.search(re.escape(kw), text):
                total += w
        except Exception:
            if kw in text:
                total += w
    return total


def _match_pattern(text: str):
    """Return intent name if any precise pattern matches, else None."""
    for pat, intent in PATTERN_INTENTS:
        if pat.search(text):
            return intent
    return None


def predict_intent(query: str, entities: Optional[Dict] = None) -> str:
    """Predict intent using rule-based logic (Layer 1 + Layer 2).

    Parameters:
    - query: raw user text
    - entities: optional dict from entity extractor (should contain 'players' and 'league')

    Priority (applied):
    1) compare_players if >=2 players
    2) similar_players if similar-like keywords present (and 1 player)
    3) league_fit if league-fit keywords present (and 1 player or league entity)
    4) player_stats if 1 player and stats keywords present
    5) fallback to keyword-scoring
    6) finally, if ML classifier is available, use it as a fallback
    """
    text = query.lower()

    players = 0
    leagues = 0
    if entities:
        players = len(entities.get("players", [])) if entities.get("players") is not None else 0
        leagues = len(entities.get("league", [])) if entities.get("league") is not None else 0

    # (1) compare_players: explicit player-count rule
    if players >= 2:
        return "compare_players"

    # Pattern-based detection (Layer 4) — applies early for precise constructs
    pat_intent = _match_pattern(text)
    if pat_intent:
        # if pattern indicates compare but we have only 1 player entity, still trust pattern
        return pat_intent

    # Quick checks when exactly 1 player is present
    if players == 1:
        # (2) similar players
        if _count_weighted_matches(text, SIMILAR_KEYWORDS_WEIGHTS) > 0:
            return "similar_players"

        # (3) league fit (require league keyword or league entity)
        if _count_weighted_matches(text, LEAGUE_KEYWORDS_WEIGHTS) > 0 or leagues >= 1:
            return "league_fit"

        # (4) player stats
        if _count_weighted_matches(text, STATS_KEYWORDS_WEIGHTS) > 0:
            return "player_stats"

    # No players detected or ambiguous: use broader keyword scoring
    # Weighted scoring across categories (Layer 3)
    score_compare = _count_weighted_matches(text, COMPARE_KEYWORDS_WEIGHTS)
    score_similar = _count_weighted_matches(text, SIMILAR_KEYWORDS_WEIGHTS)
    score_league = _count_weighted_matches(text, LEAGUE_KEYWORDS_WEIGHTS)
    score_stats = _count_weighted_matches(text, STATS_KEYWORDS_WEIGHTS)

    # Priority order to break ties: compare > similar > league_fit > player_stats
    best_score = max(score_compare, score_similar, score_league, score_stats)
    if best_score == 0:
        # fallback to ML classifier if available
        if clf is not None and vectorizer is not None:
            try:
                X = vectorizer.transform([query])
                pred = clf.predict(X)[0]
                return pred
            except Exception:
                return "unknown"
        return "unknown"

    if score_compare == best_score:
        return "compare_players"
    if score_similar == best_score:
        return "similar_players"
    if score_league == best_score:
        return "league_fit"
    if score_stats == best_score:
        return "player_stats"

    return "unknown"


__all__ = ["predict_intent"]
