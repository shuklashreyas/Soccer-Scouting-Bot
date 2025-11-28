from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.player.extract import PlayerProfile


def _percentile_of(series: pd.Series, value) -> Optional[float]:
    s = series.dropna()
    if s.empty or value is None:
        return None
    try:
        pct = (s < value).sum() / len(s) * 100.0
        return float(pct)
    except Exception:
        return None


def _column_or_none(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    return df[col] if col in df.columns else None


def ordinal(n: float) -> str:
    """Return ordinal string for integer-like percentiles, e.g. 1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11->'11th'."""
    try:
        i = int(round(float(n)))
    except Exception:
        return f"{n:.0f}th"
    if 10 <= (i % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suffix}"


def generate_strengths_weaknesses(profile: PlayerProfile, df: pd.DataFrame, compare_group: str = "position",
                                   strong_thresh: float = 75.0, weak_thresh: float = 25.0) -> Tuple[List[str], List[str], Dict[str, Optional[float]]]:
    """Compute percentiles for a set of candidate metrics and return strengths and weaknesses.

    - compare_group: currently supports 'position' to compare within same-position players when possible.
    - thresholds: percentiles for strengths and weaknesses.

    Returns (strengths, weaknesses, percentiles_map)
    """
    pos = (profile.pos or "").upper()
    df_group = df
    if compare_group == "position" and pos:
        token = pos.split(",")[0].strip()
        if token:
            mask = df["Pos"].fillna("").str.contains(token, case=False, na=False)
            if mask.sum() >= max(50, int(0.05 * len(df))):
                df_group = df[mask]

    # Candidate metrics mapping to columns (preferred col names)
    metric_cols = {
        "goals_per90": ["Per_90_Minutes_Gls"],
        "xg_per90": ["Per_90_Minutes_xG", "Expected_xG"],
        "xa_per90": ["Per_90_Minutes_xAG", "Expected_xAG"],
        "assists_per90": ["Per_90_Minutes_Ast"],
        "progressive_passes": ["Progression_PrgP"],
        "progressive_carries": ["Progression_PrgC"],
        "pressures": ["Pressures"],
        "tackles": ["Tackles"],
    }

    percentiles: Dict[str, Optional[float]] = {}

    # Helper to get player's value for a metric and series
    for metric, candidates in metric_cols.items():
        series = None
        for c in candidates:
            if c in df_group.columns:
                series = df_group[c]
                break

        # get player's value: try profile.stats/per90 then raw
        val = None
        # common per90 keys in profile.stats['per90']
        per90_map = {
            "goals_per90": "g90",
            "xg_per90": "xg90",
            "xa_per90": "xa90",
            "assists_per90": "a90",
            "progressive_passes": None,
            "progressive_carries": None,
            "pressures": None,
            "tackles": None,
        }

        key = per90_map.get(metric)
        if key and profile.stats.get("per90"):
            val = profile.stats["per90"].get(key)

        if val is None:
            # try raw columns
            for c in candidates:
                if c in profile.raw.index:
                    try:
                        v = profile.raw.get(c)
                        if pd.notna(v):
                            val = float(v)
                            break
                    except Exception:
                        continue

        if series is not None and val is not None:
            pct = _percentile_of(series, val)
        else:
            pct = None

        percentiles[metric] = pct

    strengths: List[str] = []
    weaknesses: List[str] = []

    # Build human-readable messages
    label_map = {
        "goals_per90": "Goals per90",
        "xg_per90": "xG per90",
        "xa_per90": "xA per90",
        "assists_per90": "Assists per90",
        "progressive_passes": "Progressive passes",
        "progressive_carries": "Progressive carries",
        "pressures": "Pressures",
        "tackles": "Tackles",
    }

    for metric, pct in percentiles.items():
        if pct is None:
            continue
        label = label_map.get(metric, metric)
        if pct >= strong_thresh:
            strengths.append(f"✓ {label} — {ordinal(pct)} percentile")
        elif pct <= weak_thresh:
            weaknesses.append(f"✗ {label} — {ordinal(pct)} percentile")

    # Update profile
    profile.percentiles = percentiles
    profile.strengths = strengths
    profile.weaknesses = weaknesses

    return strengths, weaknesses, percentiles


__all__ = ["generate_strengths_weaknesses"]
