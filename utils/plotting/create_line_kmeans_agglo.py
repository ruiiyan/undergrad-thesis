import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ─────────────────────────────────────────────────
kmeans = pd.read_csv('data/clusters/kmeans_comparison.csv')
agg    = pd.read_csv('data/clusters/agglomerative_comparison.csv')

# HDBSCAN final metrics (seed=42, mcs=10, min_samples=1)
HDBSCAN_PURITY = 0.3283
HDBSCAN_NMI    = 0.4758

# Thesis colours
COL_KMEANS  = '#5EAAA8'
COL_AGG     = '#7C6BAE'
COL_HDBSCAN = '#D4826A'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('#F8FAFC')

metrics = [
    ('purity', 'Purity Score',                       axes[0]),
    ('nmi',    'Normalised Mutual Information (NMI)', axes[1]),
]

hdbscan_vals = {'purity': HDBSCAN_PURITY, 'nmi': HDBSCAN_NMI}

for metric, ylabel, ax in metrics:
    ax.set_facecolor('#F8FAFC')

    # K-Means line — labels ABOVE
    ax.plot(kmeans['k'], kmeans[metric],
            color=COL_KMEANS, linewidth=2.2, marker='o',
            markersize=6, label='K-Means', zorder=3)
    for _, row in kmeans.iterrows():
        ax.text(row['k'], row[metric] + 0.006,
                f'{row[metric]:.4f}',
                ha='center', va='bottom',
                fontsize=7.5, color=COL_KMEANS, fontweight='600')

    # Agglomerative line — labels BELOW
    ax.plot(agg['k'], agg[metric],
            color=COL_AGG, linewidth=2.2, marker='s',
            markersize=6, label='Agglomerative', zorder=3)
    for _, row in agg.iterrows():
        ax.text(row['k'], row[metric] - 0.009,
                f'{row[metric]:.4f}',
                ha='center', va='top',
                fontsize=7.5, color=COL_AGG, fontweight='600')

    # HDBSCAN reference line
    ax.axhline(hdbscan_vals[metric],
               color=COL_HDBSCAN, linewidth=2.2,
               linestyle='--', label='HDBSCAN', zorder=2)

    # HDBSCAN annotation on right edge
    ax.text(kmeans['k'].max() + 0.4,
            hdbscan_vals[metric] + 0.006,
            f'HDBSCAN\n{hdbscan_vals[metric]:.4f}',
            color=COL_HDBSCAN, fontsize=8.5,
            fontweight='700', va='bottom')

    ax.set_xlabel('Number of Clusters (k)', fontsize=11,
                  color='#475569', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11,
                  color='#475569', labelpad=8)
    ax.set_xticks(kmeans['k'])
    ax.set_xlim(kmeans['k'].min() - 1, kmeans['k'].max() + 2.5)
    ax.set_ylim(
        min(agg[metric].min(), kmeans[metric].min()) - 0.04,
        max(hdbscan_vals[metric],
            kmeans[metric].max(),
            agg[metric].max()) + 0.06
    )
    ax.yaxis.grid(True, color='#E2E8F0',
                  linewidth=0.8, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CBD5E1')
    ax.tick_params(labelsize=10, colors='#64748B')
    ax.legend(fontsize=10, framealpha=0.9,
              edgecolor='#E2E8F0', loc='upper left')

fig.suptitle(
    'Clustering Performance — K-Means vs Agglomerative vs HDBSCAN',
    fontsize=13, fontweight='700', color='#0F172A', y=1.02
)

plt.tight_layout()
plt.savefig('clustering_comparison_lines.png',
            dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved → clustering_comparison_lines.png")