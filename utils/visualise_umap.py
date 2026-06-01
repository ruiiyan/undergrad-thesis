"""
visualise_umap.py
-----------------
Projects the cached sentence embeddings into 2D via UMAP and plots
each reflection coloured by its final cluster assignment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

EMBEDDINGS_FILE = "reflection_embeddings.npy"
CLUSTERED_FILE  = "reflections_clustered_final.json"
OUTPUT_PNG      = "umap_projection.png"

UMAP_PARAMS = dict(n_neighbors=10   , min_dist=0.1, n_components=2, metric="cosine", random_state=42)

CLUSTER_LABELS = {
    -1: "unclustered",
     0: "Coding problem-solving",
     1: "Technical fabrication",
     2: "Technical drawing",
     3: "RGM design and ideation",
     4: "Team communication",
     5: "Project planning & time mgmt",
     6: "Group presentation",
     7: "Critical thinking",
     8: "Collaborative design non-RGM",
     9: "Conceptual solution development",
}

# Colour palette — noise is grey, clusters get distinct colours
PALETTE = {
    -1: "#aaaaaa",
     0: "#e6194b",
     1: "#f58231",
     2: "#ffe119",
     3: "#3cb44b",
     4: "#42d4f4",
     5: "#4363d8",
     6: "#911eb4",
     7: "#f032e6",
     8: "#a9a9ff",
     9: "#9a6324",
}

# ---------------------------------------------------------------------------
print("[1] Loading embeddings and cluster assignments...")

embeddings = np.load(EMBEDDINGS_FILE)
with open(CLUSTERED_FILE, "r", encoding="utf-8") as f:
    reflections = json.load(f)

assert len(embeddings) == len(reflections), (
    f"Mismatch: {len(embeddings)} embeddings vs {len(reflections)} reflections"
)

final_ids = np.array([r["final_cluster_id"] for r in reflections])
print(f"  {len(embeddings)} embeddings, {len(set(final_ids))} final clusters")

# ---------------------------------------------------------------------------
print("[2] Reducing to 2D with UMAP...")

reducer   = umap.UMAP(**UMAP_PARAMS)
projected = reducer.fit_transform(embeddings)
print(f"  Projection shape: {projected.shape}")

# ---------------------------------------------------------------------------
print("[3] Plotting...")

fig, ax = plt.subplots(figsize=(13, 9))

# Plot noise first (bottom layer)
for fid in sorted(CLUSTER_LABELS.keys()):
    mask   = final_ids == fid
    colour = PALETTE[fid]
    alpha  = 0.25 if fid == -1 else 0.65
    size   = 6    if fid == -1 else 10
    zorder = 1    if fid == -1 else 2
    ax.scatter(
        projected[mask, 0],
        projected[mask, 1],
        c=colour,
        s=size,
        alpha=alpha,
        linewidths=0,
        zorder=zorder,
    )

# Cluster centroids with labels
for fid in sorted(k for k in CLUSTER_LABELS if k != -1):
    mask = final_ids == fid
    if not mask.any():
        continue
    cx, cy = projected[mask, 0].mean(), projected[mask, 1].mean()
    ax.annotate(
        CLUSTER_LABELS[fid],
        xy=(cx, cy),
        fontsize=7.5,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
        zorder=3,
    )

# Legend
patches = [
    mpatches.Patch(color=PALETTE[fid], label=f"{fid}: {CLUSTER_LABELS[fid]}")
    for fid in sorted(CLUSTER_LABELS.keys())
]
ax.legend(
    handles=patches,
    loc="upper left",
    fontsize=7,
    framealpha=0.85,
    title="Final cluster",
    title_fontsize=8,
)

ax.set_title("UMAP projection of engineering student reflections\n(coloured by final cluster)", fontsize=12)
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=180)
print(f"  Saved → '{OUTPUT_PNG}'")
plt.show()
print("Done.")
