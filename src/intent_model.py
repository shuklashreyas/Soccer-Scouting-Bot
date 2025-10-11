def detect_intent(query: str) -> str:
    query = query.lower()
    if "compare" in query:
        return "compare"
    elif "like" in query or "similar" in query:
        return "similarity"
    elif "show" in query or "find" in query:
        return "profile"
    return "unknown"
