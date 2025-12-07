from mplsoccer import Radar
from mplsoccer import FontManager
import matplotlib.pyplot as plt


def plot_radar(df, cols, df2=None):
    """Render a percentiles-style radar chart for one (or two) players.

    `df` should contain a single row with columns matching `cols` and metadata
    fields `player` and `team` used for labels.
    """
    team = df['team'].iloc[0]
    p1 = df['player'].iloc[0]
    df = df[cols]

    radar = Radar(
        params=cols,
        min_range=[0 for _ in cols],
        max_range=[100 for _ in cols]
    )

    fig, ax = radar.setup_axis()

    rings_inner = radar.draw_circles(
        ax=ax, facecolor='#f7f7f7', edgecolor='#a6a4a1', linestyle='-', lw=1
    )
    if df2 is not None:
        team2 = df2['team'].iloc[0]
        p2 = df2['player'].iloc[0]
        df2 = df2[cols]

        rings_output = radar.draw_radar_compare(
            list(df.values.flatten()),
            list(df2.values.flatten()),
            ax=ax,
            kwargs_radar={'facecolor': '#cfc2b2', 'alpha': .6, 'edgecolor': '#a6a4a1', 'linewidth': 1},
            kwargs_compare={'facecolor': '#62745A', 'alpha': .6, 'edgecolor': '#a6a4a1', 'linewidth': 1},

        )


    else:
        rings_output = radar.draw_radar(
            ax=ax,
            values=list(df.values.flatten()),
            kwargs_radar={'facecolor': '#4A90E2', 'alpha': .6}
        )

    radar_labels = radar.draw_range_labels(
        ax=ax, fontsize=13, alpha=.4
    )

    param_labels = radar.draw_param_labels(
        ax=ax, fontsize=13, offset=.6, alpha=.4,
    )

    lines = radar.spoke(
        ax=ax, color='#a6a4a1', linestyle='-', zorder=2, alpha=.4
    )

    if df2 is not None:
        # Top-left corner inside the radar chart
        title1_text = fig.text(0.15, 0.87, p1, fontsize=18, color='#cfc2b2', ha='left', va='top')
        title2_text = fig.text(0.15, 0.85, team, fontsize=15,
                               ha='left', va='top', color='#cfc2b2')
        # Top-right corner inside the radar chart
        title3_text = fig.text(0.87, 0.87, p2, fontsize=18,
                               ha='right', va='top', color='#62745A')
        title4_text = fig.text(0.87, 0.85, team2, fontsize=15,
                               ha='right', va='top', color='#62745A')
    else:
        # Top-left corner inside the radar chart
        title1_text = fig.text(0.15, 0.85, p1, fontsize=25, color='#4A90E2', ha='left', va='top')
        title2_text = fig.text(0.15, 0.80, team, fontsize=20,
                               ha='left', va='top', color='#4A90E2')


    plt.show()

