import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from umap import UMAP

# ── Paths — update these to match your directory ──────────────
EMBEDDINGS_PATH = 'data/embeddings/all/all-MiniLM-L6-v2/all.npy'
UMAP_REDUCER_PATH = 'data/clusters/umap_reducer.pkl'
CLUSTERED_PATH = 'data/clusters/clustered_reflections.csv'

# ── Load data ─────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
df = pd.read_csv(CLUSTERED_PATH)

print("Loading UMAP reducer and transforming to 10D...")
with open(UMAP_REDUCER_PATH, 'rb') as f:
    reducer = pickle.load(f)
reduced_10d = reducer.transform(embeddings)
print(f"Reduced shape: {reduced_10d.shape}")

# ── Run 2D UMAP for visualisation ─────────────────────────────
print("Running 2D UMAP for visualisation...")
reducer_2d = UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
coords_2d = reducer_2d.fit_transform(reduced_10d)

# ── K-Means (k=20 — best from comparison) ────────────────────
print("Running K-Means (k=20)...")
km = KMeans(n_clusters=20, random_state=42, n_init=10)
km_labels = km.fit_predict(reduced_10d)

# ── Agglomerative (k=20 — best from comparison) ───────────────
print("Running Agglomerative (k=20)...")
agg = AgglomerativeClustering(n_clusters=20, linkage='ward')
agg_labels = agg.fit_predict(reduced_10d)

# ── HDBSCAN labels from saved CSV ─────────────────────────────
hdbscan_labels = df['cluster'].values
noise_mask = hdbscan_labels == -1

# ── Colour palettes ───────────────────────────────────────────
base_colours = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

def get_colour_map(labels):
    unique = sorted([l for l in np.unique(labels) if l != -1])
    return {c: base_colours[i % len(base_colours)] for i, c in enumerate(unique)}

km_cmap      = get_colour_map(km_labels)
agg_cmap     = get_colour_map(agg_labels)
hdbscan_cmap = get_colour_map(hdbscan_labels)

# ── Three panel plot ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor('#F8FAFC')

configs = [
    ('K-Means (k=20)',            km_labels,      km_cmap,      False),
    ('Agglomerative (k=20)',      agg_labels,     agg_cmap,     False),
    (f'HDBSCAN — 45 Clusters',   hdbscan_labels, hdbscan_cmap, True),
]

for ax, (title, labels, cmap, has_noise) in zip(axes, configs):
    ax.set_facecolor('#F8FAFC')

    if has_noise:
        nmask = labels == -1
        ax.scatter(
            coords_2d[nmask, 0], coords_2d[nmask, 1],
            c='#CBD5E1', s=8, alpha=0.4, zorder=1,
            label=f'Noise ({nmask.sum()}, {nmask.mean()*100:.1f}%)'
        )

    unique = sorted([l for l in np.unique(labels) if l != -1])
    for cluster_id in unique:
        mask = labels == cluster_id
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[cmap[cluster_id]], s=12, alpha=0.75, zorder=2
        )

    # Cluster centroid labels
    for cluster_id in unique:
        mask = labels == cluster_id
        cx = coords_2d[mask, 0].mean()
        cy = coords_2d[mask, 1].mean()
        ax.text(cx, cy, str(cluster_id),
                fontsize=6, fontweight='700',
                ha='center', va='center', color='white',
                bbox=dict(boxstyle='round,pad=0.12',
                          fc=cmap[cluster_id], ec='none', alpha=0.85),
                zorder=3)

    ax.set_title(title, fontsize=12, fontweight='700',
                 color='#0F172A', pad=10)
    ax.set_xlabel('UMAP Dimension 1', fontsize=9, color='#475569')
    ax.set_ylabel('UMAP Dimension 2', fontsize=9, color='#475569')
    ax.tick_params(colors='#94A3B8', labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CBD5E1')

    if has_noise:
        import matplotlib.patches as mpatches
        noise_patch = mpatches.Patch(
            color='#CBD5E1',
            label=f'Noise — {nmask.sum()} ({nmask.mean()*100:.1f}%)'
        )
        ax.legend(handles=[noise_patch], fontsize=8,
                  framealpha=0.9, edgecolor='#E2E8F0', loc='lower right')

fig.suptitle(
    'Clustering Comparison — K-Means vs Agglomerative vs HDBSCAN',
    fontsize=14, fontweight='700', color='#0F172A', y=1.02
)

plt.tight_layout()
plt.savefig('umap_clustering_comparison.png',
            dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved → umap_clustering_comparison.png")