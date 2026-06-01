import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter

# ── Paths — update to match your directory ────────────────────
EMBEDDINGS_PATH  = 'data/embeddings/all/all-MiniLM-L6-v2/all.npy'
CLUSTERED_PATH   = 'data/clusters/clustered_reflections.csv'
UMAP_OUTPUT_PATH = 'data/clusters/umap_tuning_results.csv'
HDBSCAN_OUTPUT_PATH = 'data/clusters/hdbscan_tuning_results.csv'

# ── Tuning ranges ─────────────────────────────────────────────
UMAP_N_COMPONENTS = [5, 7, 10]
UMAP_N_NEIGHBORS  = [10, 15, 20, 30]
UMAP_MIN_DIST     = 0.0

# Fixed HDBSCAN defaults for UMAP tuning phase
HDBSCAN_DEFAULT_MCS = 10
HDBSCAN_DEFAULT_MS  = 3

# HDBSCAN tuning ranges (run after UMAP is fixed)
HDBSCAN_MIN_CLUSTER_SIZES = [5, 7, 10, 15, 20]
HDBSCAN_MIN_SAMPLES       = [2, 3, 5, 7]

# Target cluster range
TARGET_MIN = 10
TARGET_MAX = 25

# Best UMAP config (locked in after tuning)
BEST_N_COMPONENTS = 10
BEST_N_NEIGHBORS  = 20

# ── Helper functions ──────────────────────────────────────────
def purity_score(labels, topics):
    total, correct = 0, 0
    for label in set(labels) - {-1}:
        idx = np.where(labels == label)[0]
        cluster_topics = [topics[i] for i in idx]
        correct += Counter(cluster_topics).most_common(1)[0][1]
        total += len(idx)
    return correct / total if total > 0 else 0.0

def compute_nmi(labels, topics):
    valid = labels != -1
    if valid.sum() < 2:
        return 0.0
    return normalized_mutual_info_score(
        [t for t, v in zip(topics, valid) if v],
        [l for l, v in zip(labels, valid) if v]
    )

def in_target(n_clusters):
    return TARGET_MIN <= n_clusters <= TARGET_MAX

def distance_to_target(n_clusters):
    if in_target(n_clusters):
        return 0
    return min(abs(n_clusters - TARGET_MIN), abs(n_clusters - TARGET_MAX))

# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
embeddings = np.load(EMBEDDINGS_PATH)
df         = pd.read_csv(CLUSTERED_PATH)
topics     = df['topic'].fillna('unknown').tolist()
print(f"  Embeddings: {embeddings.shape}")
print(f"  Reflections: {len(topics)}")

# ─────────────────────────────────────────────────────────────
# STAGE 1 — UMAP TUNING
# Fixed HDBSCAN at mcs=10, ms=3
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 1 — UMAP TUNING")
print(f"Fixed HDBSCAN: mcs={HDBSCAN_DEFAULT_MCS}, ms={HDBSCAN_DEFAULT_MS}")
print("=" * 60)

umap_results = []

for nc in UMAP_N_COMPONENTS:
    for nn in UMAP_N_NEIGHBORS:
        print(f"\n  nc={nc}, nn={nn}...")

        # Reduce dimensions
        reducer = umap.UMAP(
            n_components=nc,
            n_neighbors=nn,
            min_dist=UMAP_MIN_DIST,
            metric='cosine',
            random_state=42
        )
        reduced = reducer.fit_transform(embeddings)

        # Cluster
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_DEFAULT_MCS,
            min_samples=HDBSCAN_DEFAULT_MS,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = (labels == -1).mean() * 100
        purity     = purity_score(labels, topics)
        nmi        = compute_nmi(labels, topics)
        target     = in_target(n_clusters)
        dist       = distance_to_target(n_clusters)

        print(f"    clusters={n_clusters}, noise={noise_pct:.1f}%, "
              f"purity={purity:.4f}, nmi={nmi:.4f}, "
              f"in_target={target}")

        umap_results.append({
            'n_components':        nc,
            'n_neighbors':         nn,
            'n_clusters':          n_clusters,
            'noise_pct':           round(noise_pct, 1),
            'purity':              round(purity, 4),
            'nmi':                 round(nmi, 4),
            'in_target_range':     target,
            'distance_to_target':  dist
        })

umap_df = pd.DataFrame(umap_results)
umap_df.to_csv(UMAP_OUTPUT_PATH, index=False)
print(f"\nSaved → {UMAP_OUTPUT_PATH}")

# Best UMAP config
best_umap = umap_df.sort_values(
    ['distance_to_target', 'purity'], ascending=[True, False]
).iloc[0]
print(f"\nBest UMAP config: n_components={int(best_umap['n_components'])}, "
      f"n_neighbors={int(best_umap['n_neighbors'])}")
print(f"  clusters={int(best_umap['n_clusters'])}, "
      f"purity={best_umap['purity']:.4f}, nmi={best_umap['nmi']:.4f}")

# ─────────────────────────────────────────────────────────────
# STAGE 2 — HDBSCAN TUNING
# Fixed UMAP at best config from Stage 1
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2 — HDBSCAN TUNING")
print(f"Fixed UMAP: n_components={BEST_N_COMPONENTS}, "
      f"n_neighbors={BEST_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}")
print("=" * 60)

# Reduce with best UMAP config
print("\nReducing dimensions with best UMAP config...")
reducer = umap.UMAP(
    n_components=BEST_N_COMPONENTS,
    n_neighbors=BEST_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    metric='cosine',
    random_state=42
)
reduced = reducer.fit_transform(embeddings)
print(f"  Reduced shape: {reduced.shape}")

hdbscan_results = []

for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
    for ms in HDBSCAN_MIN_SAMPLES:
        print(f"\n  mcs={mcs}, ms={ms}...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = (labels == -1).mean() * 100
        purity     = purity_score(labels, topics)
        nmi        = compute_nmi(labels, topics)
        target     = in_target(n_clusters)

        print(f"    clusters={n_clusters}, noise={noise_pct:.1f}%, "
              f"purity={purity:.4f}, nmi={nmi:.4f}, "
              f"in_target={target}")

        hdbscan_results.append({
            'min_cluster_size': mcs,
            'min_samples':      ms,
            'n_clusters':       n_clusters,
            'noise_pct':        round(noise_pct, 1),
            'purity':           round(purity, 4),
            'nmi':              round(nmi, 4),
            'in_target_range':  target
        })

hdbscan_df = pd.DataFrame(hdbscan_results)
hdbscan_df.to_csv(HDBSCAN_OUTPUT_PATH, index=False)
print(f"\nSaved → {HDBSCAN_OUTPUT_PATH}")

# Best HDBSCAN config
best_in_range = hdbscan_df[hdbscan_df['in_target_range']].sort_values(
    'purity', ascending=False
)
if len(best_in_range) > 0:
    best = best_in_range.iloc[0]
    print(f"\nBest in-range HDBSCAN config: "
          f"mcs={int(best['min_cluster_size'])}, "
          f"ms={int(best['min_samples'])}")
    print(f"  clusters={int(best['n_clusters'])}, "
          f"purity={best['purity']:.4f}, nmi={best['nmi']:.4f}")
else:
    print("\nNo configs in target range — check sweep CSV for closest.")

print("\n" + "=" * 60)
print("TUNING COMPLETE")
print(f"  UMAP results    → {UMAP_OUTPUT_PATH}")
print(f"  HDBSCAN results → {HDBSCAN_OUTPUT_PATH}")
print("=" * 60)