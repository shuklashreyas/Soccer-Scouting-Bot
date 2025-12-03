import math
from typing import List, Tuple
import pandas as pd
from src.modeling.similarity import PlayerEmbeddingModel, FRIENDLY_NAMES


def compute_strengths_weaknesses(model: PlayerEmbeddingModel, idx: int, top_n: int = 5) -> Tuple[List[str], List[str]]:
    """Return human-friendly strengths and weaknesses for a player index.

    Strengths are the top positive z-scores; weaknesses are the most negative.
    """
    z = model.X_scaled[idx]
    cols = model.feature_cols

    # Sort indices by z value
    order_desc = list(reversed(sorted(range(len(z)), key=lambda i: z[i])))
    order_asc = sorted(range(len(z)), key=lambda i: z[i])

    strengths = []
    for i in order_desc:
        if z[i] <= 0:
            break
        strengths.append(FRIENDLY_NAMES.get(cols[i], cols[i]))
        if len(strengths) >= top_n:
            break

    weaknesses = []
    for i in order_asc:
        if z[i] >= 0:
            break
        weaknesses.append(FRIENDLY_NAMES.get(cols[i], cols[i]))
        if len(weaknesses) >= top_n:
            break

    return strengths, weaknesses


def format_list(items: List[str]) -> str:
    if not items:
        return "None"
    return "\n" + "\n".join([f"- {i}" for i in items])


def generate_scouting_report(model: PlayerEmbeddingModel, player_name: str, top_k: int = 5, target_club: str = None) -> str:
    """Generate a markdown-formatted scouting report for `player_name`.

    Returns a markdown string that can be embedded in the UI.
    """
    idx = model._match_player_index(player_name)
    if idx is None:
        raise ValueError(f"Player '{player_name}' not found in model dataset.")

    row = model.df.iloc[idx]

    # Basic metadata
    age = row.get('Age', 'N/A')
    squad = row.get('Squad', 'N/A')
    pos = row.get(model.pos_col, row.get('Pos', 'N/A')) if model.pos_col else row.get('Pos', 'N/A')
    minutes = row.get('Playing Time_Min', row.get('Playing Time_90s', 'N/A'))

    # Role / cluster
    cluster_id = int(model.role_labels[idx])
    cluster_label = model._cluster_name(cluster_id)

    # Sample players from same cluster (top 5)
    members = model.df[model.df['role_cluster'] == cluster_id]
    sample_players = members['Player'].dropna().unique().tolist()[:6]

    # Strengths & weaknesses
    strengths, weaknesses = compute_strengths_weaknesses(model, idx, top_n=6)

    # Similar/comparable players
    try:
        sims_df = model.get_similar_players(player_name, top_k=top_k)
        comparables = sims_df['Player'].tolist()
    except Exception:
        comparables = []

    top_match = comparables[0] if comparables else None
    explanation = None
    if top_match:
        try:
            explanation = model.explain_pair(player_name, top_match)
        except Exception:
            explanation = None

    # Build textual report (markdown)
    report_lines = []
    report_lines.append(f"<div style='font-size:14px; line-height:1.45;'>")
    report_lines.append(f"<strong>📌 SCOUTING REPORT: {row['Player']}</strong><br>")
    report_lines.append(f"<strong>Club:</strong> {squad}  —  <strong>Age:</strong> {age}  —  <strong>Position:</strong> {pos}<br>")
    report_lines.append(f"<strong>Primary Role:</strong> {cluster_label}  (Cluster {cluster_id})")
    if sample_players:
        report_lines.append(f" — <em>Sample cluster members:</em> {', '.join(sample_players[:6])}")
    report_lines.append("<br><br>")

    # Strengths / Weaknesses
    report_lines.append("<strong>🔥 Strengths</strong>")
    if strengths:
        report_lines.append("<ul style='margin:6px 0 6px 18px;'>")
        for s in strengths:
            report_lines.append(f"<li style='margin-bottom:4px;'>{s}</li>")
        report_lines.append("</ul>")
    else:
        report_lines.append("<div>None identified</div>")

    report_lines.append("<strong>❗ Weaknesses</strong>")
    if weaknesses:
        report_lines.append("<ul style='margin:6px 0 6px 18px;'>")
        for w in weaknesses:
            report_lines.append(f"<li style='margin-bottom:4px;'>{w}</li>")
        report_lines.append("</ul>")
    else:
        report_lines.append("<div>None identified</div>")

    # Similar players
    report_lines.append("<br><strong>👥 Comparable Players</strong>")
    if comparables:
        report_lines.append("<ul style='margin:6px 0 6px 18px;'>")
        for c in comparables:
            report_lines.append(f"<li style='margin-bottom:4px;'>{c}</li>")
        report_lines.append("</ul>")
    else:
        report_lines.append("<div>None available</div>")

    # Explanation / style summary
    if explanation:
        role_info = explanation.get('role_info')
        shared = explanation.get('shared_strengths', [])
        style_summary = explanation.get('style_summary')

        if role_info:
            report_lines.append(f"<br><strong>🎭 Playing Style Summary</strong>")
            report_lines.append(f"<div style='margin-top:6px;'>{style_summary}</div>")

        report_lines.append(f"<br><strong>🔍 Role Fit</strong>")
        report_lines.append(f"<div style='margin-top:6px;'>{role_info}</div>")

    # Fit assessment (optional, coarse heuristic)
    if target_club:
        report_lines.append("<br><strong>⚙️ Fit Assessment</strong>")
        # very simple heuristic: if player shares many strengths with cluster, call fit
        fit_comment = f"Assessing fit for {target_club} requires club-style model; placeholder recommendation: likely a reasonable fit." 
        report_lines.append(f"<div style='margin-top:6px;'>{fit_comment}</div>")

    # Final summary
    report_lines.append("<br><strong>📚 Final Summary</strong>")
    # Compose a short final sentence using available info
    s0 = strengths[0] if strengths else 'their primary strengths'
    s1 = strengths[1] if len(strengths) > 1 else (strengths[0] if strengths else '')
    comparables_list = ', '.join(comparables[:3]) if comparables else 'comparable players not identified'
    final = f"{row['Player']} profiles as a **{cluster_label}**, combining {s0.lower()} and {s1.lower()}. Compared to players like {comparables_list}, they display { (explanation.get('style_summary') if explanation else 'a style consistent with their role').lower() }"
    report_lines.append(f"<div style='margin-top:6px;'>{final}</div>")

    report_lines.append("</div>")
    return '\n'.join(report_lines)
