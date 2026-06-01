"""
compare_clustering.py
---------------------
Validates HDBSCAN clustering by comparing against K-Means and
Agglomerative (Ward) clustering on the same embedding space.
"""

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import umap

warnings.filterwarnings("ignore")

EMBEDDINGS_FILE = "reflection_embeddings.npy"
CLUSTERED_FILE  = "reflections_clustered_final.json"
OUTPUT_DIR      = "."

UMAP_10D = dict(n_neighbors=15, min_dist=0.1, n_components=10, metric="cosine", random_state=42)
UMAP_2D  = dict(n_neighbors=15, min_dist=0.1, n_components=2,  metric="cosine", random_state=42)

K_RANGE = range(5, 21)
FIXED_K = 10

PALETTE = [
    "#e6194b","#f58231","#b8a800","#3cb44b","#00b4d8",
    "#4363d8","#911eb4","#f032e6","#7b7bff","#9a6324",
    "#469990","#800000","#aaffc3","#000075","#e6beff","#808000",
]

FINAL_THEMES = {
    -1: "Unclustered",
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def intra_cosine(embeddings_384, labels):
    """Mean pairwise cosine similarity within each cluster, then grand mean."""
    per_cluster = {}
    for lbl in sorted(set(labels)):
        sub = embeddings_384[labels == lbl]
        if len(sub) < 2:
            per_cluster[int(lbl)] = float("nan")
            continue
        sim = cosine_similarity(sub)
        n   = len(sub)
        per_cluster[int(lbl)] = float((sim.sum() - n) / (n * (n - 1)))
    valid = [v for v in per_cluster.values() if not np.isnan(v)]
    return np.mean(valid), per_cluster


def plot_umap_2d(emb2d, labels, title, path, theme_map=None):
    unique = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, lbl in enumerate(unique):
        mask  = labels == lbl
        color = PALETTE[i % len(PALETTE)]
        name  = (theme_map or {}).get(lbl, str(lbl))
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=color, s=8, alpha=0.7, linewidths=0, label=f"{lbl}: {name}")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

print("\n[Step 1] Loading data...")
emb384 = np.load(EMBEDDINGS_FILE)
with open(CLUSTERED_FILE) as f:
    reflections = json.load(f)

assert len(emb384) == len(reflections), \
    f"Mismatch: {len(emb384)} embeddings vs {len(reflections)} reflections"
print(f"  {len(reflections)} reflections  ·  embeddings {emb384.shape}")

# ---------------------------------------------------------------------------
# Step 2 — UMAP reductions
# ---------------------------------------------------------------------------

print("\n[Step 2] Computing UMAP reductions (10D for clustering, 2D for plots)...")
print("  Running 10D UMAP...")
emb10d = umap.UMAP(**UMAP_10D).fit_transform(emb384)
print(f"  10D shape: {emb10d.shape}")

print("  Running 2D UMAP...")
emb2d = umap.UMAP(**UMAP_2D).fit_transform(emb384)
print(f"  2D shape:  {emb2d.shape}")

# ---------------------------------------------------------------------------
# Step 3a — K-Means optimal K
# ---------------------------------------------------------------------------

print(f"\n[Step 3a] K-Means sweep K={min(K_RANGE)}..{max(K_RANGE)}...")
inertias, silhouettes = [], []

for k in K_RANGE:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    lbl = km.fit_predict(emb10d)
    inertias.append(km.inertia_)
    sil = silhouette_score(emb384, lbl)
    silhouettes.append(sil)
    print(f"  K={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sil:.4f}")

optimal_k = list(K_RANGE)[int(np.argmax(silhouettes))]
print(f"\n  → Optimal K = {optimal_k}  (silhouette = {max(silhouettes):.4f})")

# Elbow + silhouette plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(list(K_RANGE), inertias, "o-", color="steelblue", linewidth=2)
ax1.axvline(optimal_k, color="crimson", linestyle="--", alpha=0.8,
            label=f"Optimal K={optimal_k}")
ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")
ax1.set_title("K-Means — Elbow Curve")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(list(K_RANGE), silhouettes, "o-", color="darkorange", linewidth=2)
ax2.axvline(optimal_k, color="crimson", linestyle="--", alpha=0.8,
            label=f"Optimal K={optimal_k}")
ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("K-Means — Silhouette Scores")
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("K-Means Hyperparameter Selection", fontsize=13, y=1.01)
plt.tight_layout()
out = f"{OUTPUT_DIR}/kmeans_elbow_silhouette.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")

# ---------------------------------------------------------------------------
# Step 3b — Final K-Means runs
# ---------------------------------------------------------------------------

print(f"\n[Step 3b] Final K-Means: K={optimal_k} and K={FIXED_K}...")

km_opt     = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(emb10d)
lbl_km_opt = km_opt.labels_
sil_km_opt = silhouette_score(emb384, lbl_km_opt)
cos_km_opt, per_cos_km_opt = intra_cosine(emb384, lbl_km_opt)
print(f"  K={optimal_k}: silhouette={sil_km_opt:.4f}  intra-cosine={cos_km_opt:.4f}")

km_10      = KMeans(n_clusters=FIXED_K, random_state=42, n_init=10).fit(emb10d)
lbl_km_10  = km_10.labels_
sil_km_10  = silhouette_score(emb384, lbl_km_10)
cos_km_10, per_cos_km_10 = intra_cosine(emb384, lbl_km_10)
print(f"  K={FIXED_K}:  silhouette={sil_km_10:.4f}  intra-cosine={cos_km_10:.4f}")

print(f"\n  K={FIXED_K} per-cluster intra-cosine:")
for lbl, cs in per_cos_km_10.items():
    print(f"    Cluster {lbl}: {cs:.4f}")

# ---------------------------------------------------------------------------
# Step 3c — UMAP plots for K-Means
# ---------------------------------------------------------------------------

print(f"\n[Step 3c] Plotting K-Means UMAP visualisations...")
plot_umap_2d(emb2d, lbl_km_opt,
             f"K-Means Clustering (K={optimal_k})",
             f"{OUTPUT_DIR}/umap_kmeans_optimal.png")
plot_umap_2d(emb2d, lbl_km_10,
             f"K-Means Clustering (K={FIXED_K})",
             f"{OUTPUT_DIR}/umap_kmeans_k10.png")

# ---------------------------------------------------------------------------
# Step 4a — Agglomerative clustering
# ---------------------------------------------------------------------------

print(f"\n[Step 4a] Agglomerative clustering (Ward, K={FIXED_K})...")
agg     = AgglomerativeClustering(n_clusters=FIXED_K, linkage="ward")
lbl_agg = agg.fit_predict(emb10d)
sil_agg = silhouette_score(emb384, lbl_agg)
cos_agg, per_cos_agg = intra_cosine(emb384, lbl_agg)
print(f"  Silhouette:         {sil_agg:.4f}")
print(f"  Intra-cluster cosine: {cos_agg:.4f}")

print("\n  Per-cluster intra-cosine similarity:")
for lbl, cs in per_cos_agg.items():
    print(f"    Cluster {lbl}: {cs:.4f}")

# ---------------------------------------------------------------------------
# Step 4b — UMAP plot for Agglomerative
# ---------------------------------------------------------------------------

print(f"\n[Step 4b] Plotting Agglomerative UMAP visualisation...")
plot_umap_2d(emb2d, lbl_agg,
             f"Agglomerative Clustering (Ward, K={FIXED_K})",
             f"{OUTPUT_DIR}/umap_agglomerative.png")

# ---------------------------------------------------------------------------
# Step 4c — Dendrogram
# ---------------------------------------------------------------------------

print(f"\n[Step 4c] Computing Ward linkage and dendrogram...")
Z = linkage(emb10d, method="ward")

# Cut height that yields exactly FIXED_K clusters
n         = len(emb10d)
heights   = Z[:, 2]
cut_height = (heights[n - FIXED_K - 1] + heights[n - FIXED_K]) / 2
print(f"  Cut height for K={FIXED_K}: {cut_height:.4f}")

# Verify the cut produces FIXED_K clusters
cut_labels = fcluster(Z, t=cut_height, criterion="distance")
n_cut = len(set(cut_labels))
print(f"  Clusters at cut: {n_cut}")

fig, ax = plt.subplots(figsize=(22, 11))

ddata = dendrogram(
    Z,
    truncate_mode="lastp",
    p=50,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=8,
    show_leaf_counts=True,
    color_threshold=cut_height,
    above_threshold_color="#888888",
)

ax.axhline(y=cut_height, color="crimson", linestyle="--", linewidth=2,
           label=f"K={FIXED_K} cut  (height = {cut_height:.1f})")

# Annotate the cut line
ax.text(ax.get_xlim()[1] * 0.98, cut_height + heights.max() * 0.01,
        f"← K={FIXED_K} cut", color="crimson", fontsize=10,
        ha="right", va="bottom")

ax.set_title("Ward Linkage Dendrogram — Engineering Student Reflections\n"
             "(truncated to last 50 merges; leaf labels show cluster sizes)",
             fontsize=14, pad=12)
ax.set_xlabel("Cluster (sample count in parentheses)", fontsize=11)
ax.set_ylabel("Ward Linkage Distance", fontsize=11)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
out = f"{OUTPUT_DIR}/dendrogram_ward.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"  Saved → {out}")

# ---------------------------------------------------------------------------
# Step 5 — HDBSCAN silhouette
# ---------------------------------------------------------------------------

print(f"\n[Step 5] HDBSCAN silhouette score (excluding noise)...")
hdb_ids      = np.array([r["final_cluster_id"] for r in reflections])
non_noise    = hdb_ids != -1
emb_nn       = emb384[non_noise]
lbl_nn       = hdb_ids[non_noise]
n_noise      = int((hdb_ids == -1).sum())

sil_hdb  = silhouette_score(emb_nn, lbl_nn)
cos_hdb, per_cos_hdb = intra_cosine(emb_nn, lbl_nn)
print(f"  Silhouette (excl. {n_noise} noise): {sil_hdb:.4f}")
print(f"  Intra-cluster cosine:              {cos_hdb:.4f}")

# ---------------------------------------------------------------------------
# Step 6 — Comparison summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("  CLUSTERING METHOD COMPARISON")
print("=" * 65)
print(f"  {'Method':<28} {'Silhouette':>10} {'Intra-Cosine':>13} {'Noise':>6}")
print("-" * 65)
rows = [
    ("HDBSCAN (current)",              sil_hdb,     cos_hdb,     n_noise),
    (f"K-Means (optimal K={optimal_k})", sil_km_opt, cos_km_opt, 0),
    (f"K-Means (K={FIXED_K})",           sil_km_10,  cos_km_10,  0),
    (f"Agglomerative (K={FIXED_K})",     sil_agg,    cos_agg,    0),
]
for name, sil, cos, noise in rows:
    print(f"  {name:<28} {sil:>10.3f} {cos:>13.3f} {noise:>6}")
print("=" * 65)

# ---------------------------------------------------------------------------
# Step 7 — Save JSON
# ---------------------------------------------------------------------------

print(f"\n[Step 7] Saving clustering_comparison.json...")

def per_sil(embeddings, labels):
    s = silhouette_samples(embeddings, labels)
    return {int(l): round(float(s[labels == l].mean()), 4)
            for l in sorted(set(labels))}

def to_json_safe(obj):
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

comparison = {
    "optimal_k": int(optimal_k),
    "methods": {
        "hdbscan": {
            "silhouette":           round(sil_hdb, 4),
            "intra_cluster_cosine": round(cos_hdb, 4),
            "noise_points":         n_noise,
            "per_cluster_silhouette": per_sil(emb_nn, lbl_nn),
            "per_cluster_intra_cosine": per_cos_hdb,
        },
        f"kmeans_k{optimal_k}": {
            "silhouette":           round(sil_km_opt, 4),
            "intra_cluster_cosine": round(cos_km_opt, 4),
            "noise_points":         0,
            "per_cluster_silhouette":   per_sil(emb384, lbl_km_opt),
            "per_cluster_intra_cosine": per_cos_km_opt,
        },
        "kmeans_k10": {
            "silhouette":           round(sil_km_10, 4),
            "intra_cluster_cosine": round(cos_km_10, 4),
            "noise_points":         0,
            "per_cluster_silhouette":   per_sil(emb384, lbl_km_10),
            "per_cluster_intra_cosine": per_cos_km_10,
        },
        "agglomerative_k10": {
            "silhouette":           round(sil_agg, 4),
            "intra_cluster_cosine": round(cos_agg, 4),
            "noise_points":         0,
            "per_cluster_silhouette":   per_sil(emb384, lbl_agg),
            "per_cluster_intra_cosine": per_cos_agg,
        },
    },
}

with open(f"{OUTPUT_DIR}/clustering_comparison.json", "w") as f:
    json.dump(to_json_safe(comparison), f, indent=2)
print(f"  Saved → {OUTPUT_DIR}/clustering_comparison.json")

print("\nDone.")
