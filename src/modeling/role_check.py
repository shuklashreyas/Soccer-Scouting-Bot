from typing import Tuple, Dict


def _infer_group_from_label(label: str) -> str:
    """Infer a coarse role group from a human-friendly cluster label.

    Returns one of: 'defender', 'midfielder', 'attacker', 'goalkeeper', 'unknown'
    """
    if not label or not isinstance(label, str):
        return 'unknown'
    s = label.lower()
    # goalkeeper
    if 'goal' in s or 'keeper' in s or 'gk' in s:
        return 'goalkeeper'
    # defender / stopper / back
    if any(k in s for k in ['defend', 'defensive', 'stopper', 'centre-back', 'center-back', 'cb', 'back', 'fullback', 'full-back', 'wing-back', 'wingback']):
        return 'defender'
    # midfielder
    if any(k in s for k in ['midfield', 'midfielder', 'playmaker', 'creator', 'deep-lying', 'holding', 'ball-winning', 'anchor', 'regista', 'mezzala']):
        return 'midfielder'
    # attacker / winger / forward / finisher
    if any(k in s for k in ['forward', 'attacker', 'winger', 'finisher', 'striker', 'pressing forward', 'false nine', 'advanced']):
        return 'attacker'
    # fallback: look for obvious words
    if 'attack' in s or 'goal' in s or 'score' in s:
        return 'attacker'
    return 'unknown'


def determine_comparison_style(model, idx_a: int, idx_b: int) -> Dict:
    """Decide whether two players should be compared stat-wise or role-wise.

    Returns a dict with keys:
      - 'style': 'stat' or 'role'
      - 'group_a', 'group_b': coarse groups
      - 'label_a', 'label_b': human cluster labels
      - 'reason': short explanation
    """
    # Get cluster ids and labels via model
    try:
        cluster_a = int(model.role_labels[idx_a])
        cluster_b = int(model.role_labels[idx_b])
    except Exception:
        # If role labels not available, default to stat
        return {'style': 'stat', 'group_a': 'unknown', 'group_b': 'unknown', 'label_a': None, 'label_b': None, 'reason': 'No cluster info available.'}

    label_a = model._cluster_name(cluster_a)
    label_b = model._cluster_name(cluster_b)

    group_a = _infer_group_from_label(label_a)
    group_b = _infer_group_from_label(label_b)

    # If both are the same coarse group, allow stat comparisons
    if group_a == group_b and group_a != 'unknown':
        return {'style': 'stat', 'group_a': group_a, 'group_b': group_b, 'label_a': label_a, 'label_b': label_b, 'reason': 'Same coarse role group.'}

    # If either is goalkeeper, prefer role comparison
    if group_a == 'goalkeeper' or group_b == 'goalkeeper':
        return {'style': 'role', 'group_a': group_a, 'group_b': group_b, 'label_a': label_a, 'label_b': label_b, 'reason': 'At least one player is a goalkeeper.'}

    # If groups differ (defender vs attacker, etc.), prefer role-based comparison
    if group_a != group_b:
        return {'style': 'role', 'group_a': group_a, 'group_b': group_b, 'label_a': label_a, 'label_b': label_b, 'reason': 'Players belong to different coarse role groups.'}

    # Fallback
    return {'style': 'stat', 'group_a': group_a, 'group_b': group_b, 'label_a': label_a, 'label_b': label_b, 'reason': 'Fallback to stat comparison.'}
