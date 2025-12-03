import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from src.modeling.similarity import FRIENDLY_NAMES


def plot_pca_scatter(model):
    """Return a Plotly scatter of first two embedding dims colored by cluster label."""
    emb = getattr(model, 'embeddings', None)
    if emb is None:
        # try PCA on X_scaled
        X = getattr(model, 'X_scaled', None)
        if X is None:
            raise ValueError('No embeddings or X_scaled available in model')
        pca = PCA(n_components=2, random_state=42)
        emb2 = pca.fit_transform(X)
    else:
        if emb.shape[1] < 2:
            # fallback to PCA
            pca = PCA(n_components=2, random_state=42)
            emb2 = pca.fit_transform(emb)
        else:
            emb2 = emb[:, :2]

    df = pd.DataFrame(emb2, columns=['PC1', 'PC2'])
    df['Player'] = model.df['Player'].values
    df['cluster'] = model.role_labels
    df['label'] = df['cluster'].apply(lambda c: model._cluster_name(int(c)))

    fig = px.scatter(df, x='PC1', y='PC2', color='label', hover_data=['Player'], width=900, height=600)
    fig.update_layout(legend_title_text='Cluster')
    return fig


def cluster_size_bar(model):
    import pandas as pd
    sizes = pd.Series(model.role_labels).value_counts().sort_index()
    labels = [model._cluster_name(int(i)) for i in sizes.index]
    fig = go.Figure(go.Bar(x=sizes.values, y=labels, orientation='h', marker_color='steelblue'))
    fig.update_layout(height=500, margin=dict(l=220, r=20, t=30, b=30), xaxis_title='Players')
    return fig


def cluster_centroid_radar(model, cluster_id: int):
    """Return a radar (polar) chart for the centroid z-scores of a cluster.

    Percentiles are computed per-feature against the whole dataset to render 0-100 scale.
    """
    if not hasattr(model, 'X_scaled'):
        raise ValueError('Model missing X_scaled')

    cols = model.feature_cols
    # compute centroid z (mean z across members)
    mask = model.role_labels == int(cluster_id)
    if mask.sum() == 0:
        raise ValueError(f'No members for cluster {cluster_id}')

    centroid = model.X_scaled[mask].mean(axis=0)

    # compute percentile rank of centroid per feature
    percentiles = []
    for i, val in enumerate(centroid):
        all_vals = model.X_scaled[:, i]
        pct = float((all_vals < val).sum()) / len(all_vals) * 100.0
        percentiles.append(pct)

    labels = [FRIENDLY_NAMES.get(c, c) for c in cols]

    # close the loop for radar
    r = percentiles + [percentiles[0]]
    theta = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name=f'Cluster {cluster_id}'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=500)
    return fig


def compare_z_bar(model, player_a: str, player_b: str):
    i = model._match_player_index(player_a)
    j = model._match_player_index(player_b)
    if i is None or j is None:
        raise ValueError('Player not found')

    cols = model.feature_cols
    za = model.X_scaled[i]
    zb = model.X_scaled[j]
    labels = [FRIENDLY_NAMES.get(c, c) for c in cols]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=za, name=player_a, marker_color='royalblue'))
    fig.add_trace(go.Bar(x=labels, y=zb, name=player_b, marker_color='indianred'))
    fig.update_layout(barmode='group', xaxis_tickangle=40, height=450, yaxis_title='Z-score')
    return fig


def similarity_heatmap(model, players: List[str]):
    # resolve indices
    idxs = [model._match_player_index(p) for p in players]
    if any(i is None for i in idxs):
        raise ValueError('One or more players not found')
    emb = model.embeddings
    sims = cosine_similarity(emb[idxs], emb[idxs])
    df = pd.DataFrame(sims, index=players, columns=players)
    fig = px.imshow(df, text_auto='.2f', aspect='auto', color_continuous_scale='viridis')
    fig.update_layout(height=500)
    return fig


def cluster_feature_violin(model, feature_col: str):
    # compute z per player for the named feature_col
    if feature_col not in model.feature_cols:
        raise ValueError('Feature not found')
    idx = model.feature_cols.index(feature_col)
    df = pd.DataFrame({
        'z': model.X_scaled[:, idx],
        'cluster': model.role_labels,
        'label': [model._cluster_name(int(c)) for c in model.role_labels],
    })
    fig = px.violin(df, x='label', y='z', box=True, points='outliers', height=450)
    fig.update_layout(xaxis_tickangle=45, yaxis_title='Z-score')
    return fig
