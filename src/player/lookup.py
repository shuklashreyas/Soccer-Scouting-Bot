import re
import unicodedata
from typing import List, Dict

import pandas as pd
from rapidfuzz import fuzz, process


def _normalize_name(name: str) -> str:
    """Normalize player names: remove diacritics, lowercase, strip punctuation, collapse spaces."""
    if not isinstance(name, str):
        return ""
    # remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join([c for c in nfkd if not unicodedata.combining(c)])
    # lowercase
    s = ascii_only.lower()
    # replace punctuation with spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def lookup_player(query: str, df: pd.DataFrame, top_n: int = 5, score_cutoff: int = 55) -> List[Dict]:
    """Return a list of candidate players matching the query.

    Each candidate is a dict with keys: `name`, `score`, `indices` (list of df indices),
    `squad` and `pos` (examples from the first matching row).
    """
    # Gather unique player names from dataframe
    names = df["Player"].dropna().astype(str).unique().tolist()
    if len(names) == 0:
        return []

    # fast exact normalized match
    normalized_map = {n: _normalize_name(n) for n in names}
    inv_map = {}
    for n, norm in normalized_map.items():
        inv_map.setdefault(norm, []).append(n)

    qnorm = _normalize_name(query)
    # exact normalized match
    if qnorm in inv_map:
        results = []
        for real_name in inv_map[qnorm]:
            idxs = df.index[df["Player"] == real_name].tolist()
            row0 = df.loc[idxs[0]] if idxs else None
            results.append({
                "name": real_name,
                "score": 100,
                "indices": idxs,
                "squad": row0.get("Squad") if row0 is not None else None,
                "pos": row0.get("Pos") if row0 is not None else None,
            })
        return results

    # fuzzy match using RapidFuzz token_sort_ratio
    choices = names
    scorer = fuzz.token_sort_ratio
    extracted = process.extract(query, choices, scorer=scorer, limit=top_n)

    candidates = []
    for name, score, _ in extracted:
        if score < score_cutoff:
            continue
        idxs = df.index[df["Player"] == name].tolist()
        row0 = df.loc[idxs[0]] if idxs else None
        candidates.append({
            "name": name,
            "score": int(score),
            "indices": idxs,
            "squad": row0.get("Squad") if row0 is not None else None,
            "pos": row0.get("Pos") if row0 is not None else None,
        })

    return candidates


__all__ = ["lookup_player"]
