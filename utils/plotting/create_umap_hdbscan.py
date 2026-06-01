import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

# ── Paths — update to match your directory ────────────────────
EMBEDDINGS_PATH  = 'data/embeddings/all/all-MiniLM-L6-v2/all.npy'
CLUSTERED_PATH   = 'data/clusters/clustered_reflections.csv'
KEYWORDS_PATH    = 'data/clusters/cluster_keywords.csv'

# ── Noise colour ──────────────────────────────────────────────
NOISE_FACE = '#A7A7A7'
NOISE_EDGE = '#828282'
NOISE_RGB  = np.array([0xA7/255, 0xA7/255, 0xA7/255])

def colour_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1[:3]) - np.array(c2[:3]))**2))

def is_greyscale(c, threshold=0.08):
    r, g, b = c[0], c[1], c[2]
    return max(abs(r-g), abs(r-b), abs(g-b)) < threshold

def is_usable(c):
    dist_to_noise = colour_distance(c, NOISE_RGB)
    luminance = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
    return (dist_to_noise > 0.30
            and luminance < 0.93
            and not is_greyscale(c))

def get_label_colour(bg_colour):
    r, g, b = bg_colour[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#0F172A' if luminance > 0.55 else 'white'

# ── Load data ─────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
df         = pd.read_csv(CLUSTERED_PATH)
keywords   = pd.read_csv(KEYWORDS_PATH)

cluster_labels  = df['cluster'].values
noise_mask      = cluster_labels == -1
unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
n_clusters      = len(unique_clusters)

# ── Colour palette ────────────────────────────────────────────
np.random.seed(42)

base_colours = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)
extra_colours = [
    (0.894, 0.102, 0.110), (0.216, 0.494, 0.722), (0.302, 0.686, 0.290),
    (0.596, 0.306, 0.639), (1.000, 0.498, 0.000), (0.651, 0.337, 0.157),
    (0.969, 0.506, 0.749), (0.600, 0.600, 0.100), (0.100, 0.700, 0.600),
    (0.800, 0.200, 0.400), (0.200, 0.400, 0.800), (0.700, 0.500, 0.100),
]

filtered_colours = [c for c in (base_colours + extra_colours) if is_usable(c)]
print(f"Available colours: {len(filtered_colours)} for {n_clusters} clusters")

cluster_colour_map = {
    c: filtered_colours[i % len(filtered_colours)]
    for i, c in enumerate(unique_clusters)
}

# ── 2D UMAP projection ────────────────────────────────────────
print("Running 2D UMAP projection...")
reducer_2d = UMAP(
    n_components=2,
    n_neighbors=20,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
coords_2d = reducer_2d.fit_transform(embeddings)
print("Done.")

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor('#F8FAFC')
ax.set_facecolor('#F8FAFC')

# Noise points
ax.scatter(
    coords_2d[noise_mask, 0],
    coords_2d[noise_mask, 1],
    c=NOISE_FACE,
    edgecolors=NOISE_EDGE,
    linewidths=0.6,
    s=12, alpha=0.8,
    zorder=1
)

# Each cluster
for cluster_id in unique_clusters:
    mask   = cluster_labels == cluster_id
    colour = cluster_colour_map[cluster_id]
    ax.scatter(
        coords_2d[mask, 0],
        coords_2d[mask, 1],
        c=[colour],
        s=14, alpha=0.75,
        zorder=2
    )

# Cluster centroid labels
for cluster_id in unique_clusters:
    mask = cluster_labels == cluster_id
    cx   = coords_2d[mask, 0].mean()
    cy   = coords_2d[mask, 1].mean()
    bg   = cluster_colour_map[cluster_id]
    fg   = get_label_colour(bg)
    ax.text(cx, cy, str(cluster_id),
            fontsize=6.5, fontweight='700',
            ha='center', va='center',
            color=fg,
            bbox=dict(boxstyle='round,pad=0.15',
                      fc=bg, ec='none', alpha=0.88),
            zorder=3)

# Formatting
ax.set_title(
    f'UMAP Projection of Reference Reflections — {n_clusters} Clusters',
    fontsize=13, fontweight='700', color='#0F172A', pad=14, loc='left'
)
ax.set_xlabel('UMAP Dimension 1', fontsize=10, color='#475569', labelpad=8)
ax.set_ylabel('UMAP Dimension 2', fontsize=10, color='#475569', labelpad=8)
ax.tick_params(colors='#94A3B8', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#CBD5E1')

# Legend
noise_patch = mpatches.Patch(
    facecolor=NOISE_FACE,
    edgecolor=NOISE_EDGE,
    linewidth=1.2,
    label=f'Noise — {noise_mask.sum()} reflections ({noise_mask.mean()*100:.1f}%)'
)
ax.legend(handles=[noise_patch], fontsize=9,
          framealpha=0.9, edgecolor='#E2E8F0', loc='lower right')

plt.tight_layout()
plt.savefig('umap_clusters.png', dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved → umap_clusters.png")