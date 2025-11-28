from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class PlayerProfile:
    name: str
    age: Optional[int]
    pos: Optional[str]
    squad: Optional[str]
    raw: pd.Series
    stats: Dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None
    percentiles: Dict[str, Any] = field(default_factory=dict)
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        # raw is a Series; convert to dict for JSON-friendly output
        d["raw"] = self.raw.to_dict() if isinstance(self.raw, pd.Series) else None
        return d


def _get(row: pd.Series, col: str, default=None):
    try:
        return row.get(col, default)
    except Exception:
        return default


def extract_player_profile(row: pd.Series, df: pd.DataFrame) -> PlayerProfile:
    """Build a PlayerProfile from a dataframe row.

    The function groups stats into meaningful buckets (playing_time, shooting,
    assisting, progression, passing, defensive, per90). It is robust to missing
    columns and will compute per-90 rates when possible.
    """
    name = _get(row, "Player")
    try:
        age = int(_get(row, "Age")) if pd.notna(_get(row, "Age")) else None
    except Exception:
        age = None
    pos = _get(row, "Pos")
    squad = _get(row, "Squad")

    stats = {}

    # Playing time
    stats["playing_time"] = {
        "minutes": _get(row, "Playing_Time_Min"),
        "starts": _get(row, "Playing_Time_Starts"),
        "90s": _get(row, "Playing_Time_90s"),
    }

    # Shooting / finishing
    stats["shooting"] = {
        "goals": _get(row, "Performance_Gls"),
        "per90_goals": _get(row, "Per_90_Minutes_Gls"),
        "xg": _get(row, "Expected_xG"),
        "per90_xg": _get(row, "Per_90_Minutes_xG"),
    }

    # Assisting / chance creation
    stats["assisting"] = {
        "assists": _get(row, "Performance_Ast"),
        "per90_assists": _get(row, "Per_90_Minutes_Ast"),
        "xa": _get(row, "Expected_xAG"),
        "per90_xa": _get(row, "Per_90_Minutes_xAG"),
    }

    # Progressions
    stats["progression"] = {
        "prg_passes": _get(row, "Progression_PrgP"),
        "prg_carries": _get(row, "Progression_PrgC"),
        "prg_receipts": _get(row, "Progression_PrgR"),
    }

    # Passing / involvement
    stats["passing"] = {
        "passes": _get(row, "Passing_Passes") if "Passing_Passes" in row.index else None,
        "progressive_passes": _get(row, "Progression_PrgP"),
    }

    # Defensive
    stats["defensive"] = {
        "yellow_cards": _get(row, "Performance_CrdY"),
        "red_cards": _get(row, "Performance_CrdR"),
        # pressures and tackles may not exist in this dataset; include if present
        "pressures": _get(row, "Pressures") if "Pressures" in row.index else None,
        "tackles": _get(row, "Tackles") if "Tackles" in row.index else None,
    }

    # Per-90 aggregate
    per90 = {
        "xg90": _get(row, "Per_90_Minutes_xG"),
        "xa90": _get(row, "Per_90_Minutes_xAG"),
        "g+a90": _get(row, "Per_90_Minutes_G+A"),
        "prgp90": _get(row, "Progression_PrgP") if _get(row, "Playing_Time_90s") else None,
    }

    # If explicit per90 columns not present, try to compute from counts
    try:
        ninety = float(_get(row, "Playing_Time_90s") or 0)
    except Exception:
        ninety = 0.0

    def _compute_rate(count_col, per90_col_name):
        val = _get(row, count_col)
        if per90_col_name in row.index and pd.notna(_get(row, per90_col_name)):
            return _get(row, per90_col_name)
        try:
            if val is None or ninety == 0:
                return None
            return float(val) / ninety
        except Exception:
            return None

    per90["g90"] = _compute_rate("Performance_Gls", "Per_90_Minutes_Gls")
    per90["a90"] = _compute_rate("Performance_Ast", "Per_90_Minutes_Ast")
    per90["xg90"] = per90.get("xg90") or _compute_rate("Expected_xG", "Per_90_Minutes_xG")
    per90["xa90"] = per90.get("xa90") or _compute_rate("Expected_xAG", "Per_90_Minutes_xAG")

    stats["per90"] = per90

    profile = PlayerProfile(
        name=name,
        age=age,
        pos=pos,
        squad=squad,
        raw=row,
        stats=stats,
    )

    return profile


__all__ = ["PlayerProfile", "extract_player_profile"]
