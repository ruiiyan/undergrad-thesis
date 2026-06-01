"""
extract_cluster_metadata.py
----------------------------
Offline one-time computation of cluster-level metadata for the similarity
scoring pipeline.

Membership probabilities are NOT stored in the existing JSON — this script
re-runs UMAP + HDBSCAN with the same parameters + prediction_data=True to
recover them, then verifies the recovered labels match the stored cluster_id
field before aggregating.

Run from utils/ directory:
    python extract_cluster_metadata.py

Output: ../data/cluster_metadata.json
"""

import json
import os
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import umap
import hdbscan as hdbscan_lib

# ── Config ───────────────────────────────────────────────────────────────────

REFLECTIONS_FILE  = "./data/reflections_reparsed.json"
EMBEDDINGS_FILE   = "./data/reflection_embeddings.npy"
OUTPUT_FILE       = "./data/cluster_metadata.json"
EMBED_MODEL       = "all-MiniLM-L6-v2"

# Cached intermediate outputs — skip recomputation on subsequent runs
UMAP_CACHE_FILE   = "umap_reduced_10d.npy"      # (1879, 10)  UMAP output
PROBS_CACHE_FILE  = "hdbscan_probs.npy"          # (1879,)     HDBSCAN membership probs

UMAP_PARAMS    = dict(n_neighbors=15, min_dist=0.1, n_components=10,
                      metric="cosine", random_state=42)
HDBSCAN_PARAMS = dict(min_cluster_size=15, min_samples=5, metric="euclidean",
                      prediction_data=True)

# Merge map from consolidate_clusters.py — source cluster_id → final_cluster_id
MERGE_MAP = {
     0: 0,   2: 0,
     1: 1,
     4: 2,
     6: 3,   9: 3,
    12: 4,  13: 4,
     7: 5,  11: 5,  14: 5,
     8: 6,
     3: 7,
    10: 8,
     5: 9,
    -1: -1,
}

FINAL_THEMES = {
    0: "Coding problem-solving",
    1: "Technical fabrication",
    2: "Technical drawing",
    3: "RGM design and ideation",
    4: "Team communication",
    5: "Project planning and time management",
    6: "Group presentation",
    7: "Critical thinking",
    8: "Collaborative design non-RGM",
    9: "Conceptual solution development",
}

# ── Step 1: Load data ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  CLUSTER METADATA EXTRACTION")
print("=" * 60)

print("\n[Step 1] Loading data...")

with open(REFLECTIONS_FILE, "r", encoding="utf-8") as f:
    reflections = json.load(f)

n = len(reflections)

embeddings = np.load(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else None

if embeddings is not None:
    print(f"  Reflections : {n}")
    print(f"  Embeddings  : {embeddings.shape}")

if embeddings is None or embeddings.shape[0] != n:
    if embeddings is not None:
        print(f"  ⚠ Count mismatch ({embeddings.shape[0]} embeddings vs {n} reflections) — regenerating...")
    else:
        print(f"  ⚠ No embeddings file found — generating...")
    from sentence_transformers import SentenceTransformer
    texts = [r.get("full_text") or "" for r in reflections]
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"  Saved regenerated embeddings → '{EMBEDDINGS_FILE}'")
    print(f"  Embeddings  : {embeddings.shape}")

print("  ✓ Counts match")

# ── Step 2: UMAP + HDBSCAN (load from cache if available) ────────────────────

print("\n[Step 2] Recovering UMAP reduction + HDBSCAN membership probabilities...")
print("  NOTE: membership probabilities were not stored in the original run.")

# ── 2a: UMAP reduction ────────────────────────────────────────────────────────
if os.path.exists(UMAP_CACHE_FILE):
    print(f"  Loading cached UMAP reduction from '{UMAP_CACHE_FILE}'...")
    reduced = np.load(UMAP_CACHE_FILE)
    print(f"  Loaded shape: {reduced.shape}")
else:
    print(f"  No cache found — running UMAP (params: {UMAP_PARAMS}) ...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(embeddings)
    np.save(UMAP_CACHE_FILE, reduced)
    print(f"  Reduced shape: {reduced.shape}  →  saved to '{UMAP_CACHE_FILE}'")

# ── 2b: HDBSCAN membership probabilities ─────────────────────────────────────
if os.path.exists(PROBS_CACHE_FILE):
    print(f"  Loading cached HDBSCAN probabilities from '{PROBS_CACHE_FILE}'...")
    membership_probs = np.load(PROBS_CACHE_FILE)
    # Recover labels from a lightweight refit (no prediction_data needed, fast)
    _clusterer_check = hdbscan_lib.HDBSCAN(
        min_cluster_size=50, min_samples=5, metric="euclidean"
    )
    recovered_labels = _clusterer_check.fit_predict(reduced)
    print(f"  Loaded {len(membership_probs)} probability values")
else:
    print(f"  No cache found — running HDBSCAN (params: {HDBSCAN_PARAMS}) ...")
    clusterer = hdbscan_lib.HDBSCAN(**HDBSCAN_PARAMS)
    recovered_labels = clusterer.fit_predict(reduced)
    membership_probs = clusterer.probabilities_
    np.save(PROBS_CACHE_FILE, membership_probs)
    print(f"  Saved probabilities to '{PROBS_CACHE_FILE}'")

n_clusters_recovered = len(set(recovered_labels)) - (1 if -1 in recovered_labels else 0)
n_noise_recovered    = int((recovered_labels == -1).sum())
print(f"  Recovered: {n_clusters_recovered} clusters, {n_noise_recovered} noise points")

# ── Step 3: Verify alignment with stored cluster_id ───────────────────────────

print("\n[Step 3] Verifying label alignment with stored cluster_id...")

stored_labels = np.array([r["cluster_id"] for r in reflections])
mismatches = int((recovered_labels != stored_labels).sum())

if mismatches == 0:
    print("  ✓ Perfect match — all 1879 cluster_id labels align")
else:
    mismatch_pct = mismatches / n * 100
    print(f"  ⚠ {mismatches} mismatches ({mismatch_pct:.1f}%) between recovered "
          f"and stored labels.")
    print("    This can happen if HDBSCAN produced > 20 clusters on the first run")
    print("    and was re-run with min_cluster_size=50.  Attempting rerun...")

    clusterer2 = hdbscan_lib.HDBSCAN(**{**HDBSCAN_PARAMS, "min_cluster_size": 50})
    recovered_labels2 = clusterer2.fit_predict(reduced)
    mismatches2 = int((recovered_labels2 != stored_labels).sum())

    if mismatches2 < mismatches:
        print(f"  Rerun with min_cluster_size=50 reduced mismatches to {mismatches2}")
        clusterer = clusterer2
        recovered_labels = recovered_labels2
        membership_probs = clusterer.probabilities_
        mismatches = mismatches2

    if mismatches > 0:
        print(f"  ⚠ {mismatches} mismatches remain — using recovered probabilities "
              "as-is (labels may differ due to HDBSCAN's label ordering).")
        print("    Membership probabilities are still valid per the recovered assignment.")

# Build per-reflection probability lookup keyed by list position
# (order is the same as embeddings / reflections list)
probs_array = membership_probs  # shape (n,), index-aligned with reflections

# ── Step 4: Aggregate membership probabilities per final_cluster_id ───────────

print("\n[Step 4] Aggregating membership probabilities per final cluster...")

# Group reflections by final_cluster_id; store (list_index, prob, reflection)
cluster_groups = defaultdict(list)
for i, r in enumerate(reflections):
    fid   = r["final_cluster_id"]
    prob  = float(probs_array[i])
    cluster_groups[fid].append((i, prob, r))

membership_stats = {}
for fid, items in cluster_groups.items():
    if fid == -1:
        continue
    probs = np.array([p for _, p, _ in items])
    membership_stats[fid] = {
        "mean": float(np.mean(probs)),
        "min":  float(np.min(probs)),
        "max":  float(np.max(probs)),
        "std":  float(np.std(probs)),
    }

print("  Done.")

# ── Step 5: Compute intra-cluster cosine similarity (tightness) ───────────────

print("\n[Step 5] Computing intra-cluster cosine similarity (384D embeddings)...")

intra_cosine_stats = {}
cluster_ids_sorted = sorted(k for k in cluster_groups if k != -1)

for fid in cluster_ids_sorted:
    items   = cluster_groups[fid]
    indices = [i for i, _, _ in items]
    emb_sub = embeddings[indices]           # shape: (k, 384)

    # Pairwise cosine similarity; embeddings are already L2-normalised
    # (from SentenceTransformer with normalize_embeddings=True)
    sim_matrix = sk_cosine(emb_sub)        # shape: (k, k)

    k = len(indices)
    if k < 2:
        intra_cosine_stats[fid] = {"mean": 1.0, "std": 0.0}
        print(f"  Cluster {fid:2d}: only 1 reflection — skipping pairwise")
        continue

    # Upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(k, k=1)
    upper_vals = sim_matrix[upper_idx]

    intra_cosine_stats[fid] = {
        "mean": float(np.mean(upper_vals)),
        "std":  float(np.std(upper_vals)),
    }
    theme = FINAL_THEMES.get(fid, f"Cluster {fid}")
    print(f"  Cluster {fid:2d} ({k:4d} reflections)  "
          f"intra-cosine mean={intra_cosine_stats[fid]['mean']:.4f}  "
          f"std={intra_cosine_stats[fid]['std']:.4f}  — {theme}")

# ── Step 6: Compute cluster centroids ─────────────────────────────────────────

print("\n[Step 6] Computing cluster centroids (384D)...")

centroids = {}
for fid in cluster_ids_sorted:
    items   = cluster_groups[fid]
    indices = [i for i, _, _ in items]
    emb_sub = embeddings[indices]
    centroid = np.mean(emb_sub, axis=0)    # shape: (384,)
    centroids[fid] = centroid
    print(f"  Cluster {fid:2d}: centroid computed from {len(indices)} reflections")

# ── Step 7: Compute tightness weights via min-max normalisation ───────────────

print("\n[Step 7] Computing tightness weights...")

cosine_means = np.array([intra_cosine_stats[fid]["mean"] for fid in cluster_ids_sorted])
c_min = cosine_means.min()
c_max = cosine_means.max()

tightness_weights = {}
if c_max == c_min:
    for fid in cluster_ids_sorted:
        tightness_weights[fid] = 1.0
else:
    for fid in cluster_ids_sorted:
        raw = intra_cosine_stats[fid]["mean"]
        tightness_weights[fid] = float((raw - c_min) / (c_max - c_min))

print(f"  Cosine mean range: [{c_min:.4f}, {c_max:.4f}]")
for fid in cluster_ids_sorted:
    print(f"  Cluster {fid:2d}: tightness_weight = {tightness_weights[fid]:.4f}")

# ── Step 8: Print summary table ───────────────────────────────────────────────

print()
print("=" * 100)
print("  CLUSTER METADATA SUMMARY")
print("=" * 100)
hdr = (f"{'ID':>4}  {'Theme':<38}  {'N':>5}    "
       f"{'mem_mean':>8} {'mem_min':>7} {'mem_max':>7} {'mem_std':>7}    "
       f"{'cos_mean':>8} {'cos_std':>7}    {'tight':>6}")
print(hdr)
print("-" * 100)

for fid in cluster_ids_sorted:
    items  = cluster_groups[fid]
    n_c    = len(items)
    theme  = FINAL_THEMES.get(fid, f"Cluster {fid}")
    ms     = membership_stats[fid]
    cs     = intra_cosine_stats[fid]
    tw     = tightness_weights[fid]
    print(f"  {fid:>2}  {theme:<38}  {n_c:>5}    "
          f"{ms['mean']:>8.4f} {ms['min']:>7.4f} {ms['max']:>7.4f} {ms['std']:>7.4f}    "
          f"{cs['mean']:>8.4f} {cs['std']:>7.4f}    {tw:>6.4f}")

# Noise row
noise_items = cluster_groups.get(-1, [])
print(f"  {-1:>2}  {'unclustered':<38}  {len(noise_items):>5}    "
      f"{'N/A':>8} {'N/A':>7} {'N/A':>7} {'N/A':>7}    "
      f"{'N/A':>8} {'N/A':>7}    {'N/A':>6}")
print("=" * 100)

# ── Step 9: Save cluster_metadata.json ───────────────────────────────────────

print(f"\n[Step 9] Saving cluster metadata to '{OUTPUT_FILE}'...")

output = {
    "generated_at":    datetime.now().isoformat(),
    "embedding_model": EMBED_MODEL,
    "n_reflections":   n,
    "n_clusters":      len(cluster_ids_sorted),
    "clusters": {},
}

for fid in cluster_ids_sorted:
    items = cluster_groups[fid]
    ms    = membership_stats[fid]
    cs    = intra_cosine_stats[fid]

    output["clusters"][str(fid)] = {
        "theme":         FINAL_THEMES.get(fid, f"Cluster {fid}"),
        "n_reflections": len(items),
        "membership_probability": {
            "mean": round(ms["mean"], 4),
            "min":  round(ms["min"],  4),
            "max":  round(ms["max"],  4),
            "std":  round(ms["std"],  4),
        },
        "intra_cluster_cosine": {
            "mean": round(cs["mean"], 4),
            "std":  round(cs["std"],  4),
        },
        "tightness_weight": round(tightness_weights[fid], 4),
        "centroid": [round(float(v), 4) for v in centroids[fid]],
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"  Saved {len(output['clusters'])} clusters → '{OUTPUT_FILE}'")

# ── Step 10: Validation — high / low membership samples per cluster ───────────

print("\n[Step 10] Validation — 2 highest and 2 lowest membership probability samples per cluster")
print("=" * 80)

for fid in cluster_ids_sorted:
    items  = cluster_groups[fid]
    theme  = FINAL_THEMES.get(fid, f"Cluster {fid}")
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

    print(f"\n  Cluster {fid} — \"{theme}\" (n={len(items)})")
    print(f"  {'─'*76}")

    top2    = sorted_items[:2]
    bottom2 = sorted_items[-2:]

    for rank, (idx, prob, r) in enumerate(top2, 1):
        preview = (r.get("full_text") or "")[:200].replace("\n", " ")
        print(f"  [HIGH #{rank}] prob={prob:.4f}  id={r['id']}")
        print(f"    {preview}...")

    print(f"  ···")

    for rank, (idx, prob, r) in enumerate(bottom2, 1):
        preview = (r.get("full_text") or "")[:200].replace("\n", " ")
        print(f"  [LOW  #{rank}] prob={prob:.4f}  id={r['id']}")
        print(f"    {preview}...")

print("\n" + "=" * 80)
print("Done. Cluster metadata saved to:", OUTPUT_FILE)
