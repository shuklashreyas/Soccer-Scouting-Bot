from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from src.player.extract import PlayerProfile


def _percentile_of(series: pd.Series, value) -> float:
    """Return percentile (0-100) of value within series, ignoring NaNs."""
    s = series.dropna()
    if s.empty:
        return np.nan
    try:
        # proportion of values strictly less than value
        pct = (s < value).sum() / len(s) * 100.0
        return float(pct)
    except Exception:
        return np.nan


def _get_column_value(row: pd.Series, col: str):
    try:
        v = row.get(col, None)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def classify_role(profile: PlayerProfile, df: pd.DataFrame, compare_group: str = "position") -> Tuple[str, List[str], Dict[str, float]]:
    """Classify a player's role using heuristics and percentiles.

    Returns: (role_label, reasons, percentiles)
    """
    # Determine group for percentile comparisons
    pos = (profile.pos or "").upper()
    df_group = df
    if compare_group == "position" and pos:
        # try to narrow by first token of position (e.g., 'FW', 'MF', 'DF', 'GK')
        token = pos.split(",")[0].strip()
        if token:
            mask = df["Pos"].fillna("").str.contains(token, case=False, na=False)
            if mask.sum() >= max(50, int(0.05 * len(df))):
                df_group = df[mask]

    # Candidate metrics and where to pull from row
    candidate_cols = [
        "Per_90_Minutes_xG",
        "Per_90_Minutes_xAG",
        "Per_90_Minutes_Gls",
        "Per_90_Minutes_Ast",
        "Progression_PrgP",
        "Progression_PrgC",
        "Playing_Time_90s",
        "Performance_Gls",
        "Performance_Ast",
        "Pressures",
        "Tackles",
    ]

    percentiles: Dict[str, float] = {}
    # compute percentiles for available columns
    for col in candidate_cols:
        if col in df_group.columns:
            val = _get_column_value(profile.raw, col)
            if val is None:
                # try per90 nested stats from profile.stats
                if col.startswith("Per_90_Minutes_"):
                    val = _get_column_value(profile.raw, col)
            pct = _percentile_of(df_group[col], val) if val is not None else np.nan
            percentiles[col] = float(pct) if not np.isnan(pct) else None

    reasons: List[str] = []
    role = "Unclassified"

    # GK rule
    if "GK" in pos:
        role = "Goalkeeper"
        reasons.append("Position indicates goalkeeper")
        return role, reasons, percentiles

    # Forward rules
    xg_pct = percentiles.get("Per_90_Minutes_xG") or 0
    xa_pct = percentiles.get("Per_90_Minutes_xAG") or 0
    g_pct = percentiles.get("Per_90_Minutes_Gls") or 0
    prgP_pct = percentiles.get("Progression_PrgP") or 0
    prgC_pct = percentiles.get("Progression_PrgC") or 0
    pressures_pct = percentiles.get("Pressures") or 0

    if any(tok in pos for tok in ["FW", "ST", "CF"]):
        # Poacher / Goal scorer
        if xg_pct >= 75 and g_pct >= 60:
            role = "Goal-scoring forward / Poacher"
            reasons.append(f"High xG per90 (pct {xg_pct:.0f}) and goals per90 (pct {g_pct:.0f})")
            return role, reasons, percentiles

        # Creative forward
        if xa_pct >= 70 or prgP_pct >= 65:
            role = "Creative forward / False 9"
            reasons.append(f"High chance-creation (xA pct {xa_pct:.0f}) or progression (pct {prgP_pct:.0f})")
            return role, reasons, percentiles

        # Default forward
        role = "Forward / Winger"
        reasons.append("Position suggests attacking role")
        return role, reasons, percentiles

    # Midfielder rules
    if any(tok in pos for tok in ["M", "CM", "AM", "DM"]):
        # Progressive carrier / creator
        if prgC_pct >= 75 or prgP_pct >= 75:
            role = "Progressive midfielder / Ball-carrier"
            reasons.append(f"High progression percentiles (carries {prgC_pct:.0f}, passes {prgP_pct:.0f})")
            return role, reasons, percentiles

        # Box-to-box
        if pressures_pct >= 70:
            role = "Box-to-box / Ball-winner"
            reasons.append(f"High pressing activity (pressures pct {pressures_pct:.0f})")
            return role, reasons, percentiles

        role = "Central midfielder"
        reasons.append("Position suggests midfield role; no extreme progression or defensive metrics detected")
        return role, reasons, percentiles

    # Defender rules
    if any(tok in pos for tok in ["DF", "CB", "LB", "RB"]):
        if prgP_pct >= 70 or prgC_pct >= 70:
            role = "Progressive fullback / Ball-playing defender"
            reasons.append(f"High progression metrics (passes {prgP_pct:.0f}, carries {prgC_pct:.0f})")
            return role, reasons, percentiles

        # Ball-winning CB heuristic: if pressures/tackles are high
        t_pct = percentiles.get("Tackles") or 0
        if t_pct >= 70 or pressures_pct >= 70:
            role = "Ball-winning center-back / Defensive anchor"
            reasons.append(f"High defensive engagement (tackles pct {t_pct:.0f}, pressures pct {pressures_pct:.0f})")
            return role, reasons, percentiles

        role = "Defender"
        reasons.append("Position indicates defender; no standout progressive or defensive specialization")
        return role, reasons, percentiles

    # Fallback
    role = "Player"
    reasons.append("Could not match detailed role heuristics; returned generic label")
    return role, reasons, percentiles


__all__ = ["classify_role"]
