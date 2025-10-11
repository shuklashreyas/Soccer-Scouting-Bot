import re

def extract_entities(query, players, leagues, stats):
    entities = {'player': [], 'league': [], 'stat': []}
    for p in players:
        if p.lower() in query.lower():
            entities['player'].append(p)
    for l in leagues:
        if l.lower() in query.lower():
            entities['league'].append(l)
    for s in stats:
        if re.search(rf"\b{s}\b", query.lower()):
            entities['stat'].append(s)
    return entities
