"""
phase2.5_cluster.py
Experiments to improve clustering metrics and reduce noise.
Covers:
  - Soft cluster assignment (HDBSCAN probabilities)
  - cluster_selection_epsilon tuning
  - Threshold-based nearest centroid assignment for noise points
Run after clustering_final.py has been executed.
"""
import numpy as np
import pandas as pd
import pickle
import hdbscan as hdb
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def purity_score(labels, topics):
    total, correct = 0, 0
    for label in set(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        cluster_topics = [topics[i] for i in idx]
        most_common = Counter(cluster_topics).most_common(1)[0][1]
        correct += most_common
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


def print_metrics(name, labels, topics):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    noise_pct = n_noise / len(labels) * 100
    purity = purity_score(labels, topics)
    nmi = compute_nmi(labels, topics)
    print(f"  {name:35s} clusters={n_clusters:3d} "
          f"noise={noise_pct:5.1f}% "
          f"purity={purity:.4f} nmi={nmi:.4f}")
    return n_clusters, noise_pct, purity, nmi


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv('data/clustered_reflections.csv')
topics = df['topic'].fillna('unknown').tolist()

# Restore original hard labels if previous assignment exists
if 'cluster_assigned' in df.columns:
    df.loc[df['cluster_assigned'].isin(
        ['nearest_centroid', 'unassigned']), 'cluster'] = -1

labels_hard = df['cluster'].values.copy()

print(f"Loaded {len(df)} reflections")
print(f"Baseline — noise: {np.sum(labels_hard == -1)} "
      f"({np.sum(labels_hard == -1)/len(labels_hard)*100:.1f}%)")

# Load embeddings
emb_dir = 'data/embeddings/all-MiniLM-L6-v2'
thematic = np.load(os.path.join(emb_dir, 'thematic.npy'))

# Load UMAP reducer and transform
with open('data/umap_reducer.pkl', 'rb') as f:
    reducer = pickle.load(f)
reduced = reducer.transform(thematic)
print(f"UMAP reduced shape: {reduced.shape}")

# Load centroids
centroids_dict = np.load('data/cluster_centroids.npy',
                         allow_pickle=True).item()
cluster_ids = sorted(centroids_dict.keys())
centroid_matrix = np.stack([centroids_dict[c] for c in cluster_ids])


# ─────────────────────────────────────────────
# EXPERIMENT A — SOFT CLUSTER ASSIGNMENT
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT A — Soft Cluster Assignment (HDBSCAN probabilities)")
print("=" * 65)

# Refit with prediction_data=True
print("Refitting HDBSCAN for soft clustering...")
clusterer_soft = hdb.HDBSCAN(
    min_cluster_size=15,
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
labels_refit = clusterer_soft.fit_predict(reduced)
soft_clusters = hdb.all_points_membership_vectors(clusterer_soft)
noise_idx_soft = np.where(labels_refit == -1)[0]

print(f"\n{'Threshold':>12} {'Assigned':>10} {'Noise%':>8} "
      f"{'Purity':>8} {'NMI':>8}")
print("─" * 55)

for threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    new_labels = labels_refit.copy()
    assigned = 0
    for idx in noise_idx_soft:
        probs = soft_clusters[idx]
        best_cluster = np.argmax(probs)
        best_prob = probs[best_cluster]
        if best_prob >= threshold:
            new_labels[idx] = best_cluster
            assigned += 1
    remaining = np.sum(new_labels == -1)
    noise_pct = remaining / len(new_labels) * 100
    purity = purity_score(new_labels, topics)
    nmi = compute_nmi(new_labels, topics)
    print(f"  {threshold:>10.2f} {assigned:>10} {noise_pct:>7.1f}% "
          f"{purity:>8.4f} {nmi:>8.4f}")

print(f"\n  {'Baseline':>10} {'—':>10} "
      f"{np.sum(labels_refit == -1)/len(labels_refit)*100:>7.1f}% "
      f"{purity_score(labels_refit, topics):>8.4f} "
      f"{compute_nmi(labels_refit, topics):>8.4f}  ← baseline")

print("\nConclusion: Soft assignment reduces noise but at significant "
      "purity cost. Rejected.")


# ─────────────────────────────────────────────
# EXPERIMENT B — CLUSTER SELECTION EPSILON
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT B — cluster_selection_epsilon Tuning")
print("=" * 65)

print(f"\n{'Epsilon':>10} {'Clusters':>10} {'Noise%':>8} "
      f"{'Purity':>8} {'NMI':>8}")
print("─" * 55)

for eps in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    clusterer_eps = hdb.HDBSCAN(
        min_cluster_size=15,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=eps
    )
    labels_eps = clusterer_eps.fit_predict(reduced)
    n_clusters = len(set(labels_eps)) - (1 if -1 in labels_eps else 0)
    n_noise = np.sum(labels_eps == -1)
    noise_pct = n_noise / len(labels_eps) * 100
    purity = purity_score(labels_eps, topics)
    nmi = compute_nmi(labels_eps, topics)
    baseline = '← baseline' if eps == 0.0 else ''
    print(f"  {eps:>8.2f} {n_clusters:>10} {noise_pct:>7.1f}% "
          f"{purity:>8.4f} {nmi:>8.4f}  {baseline}")

print("\nConclusion: Epsilon only reduces noise at 0.3+ where it "
      "aggressively merges clusters, dropping purity. Rejected.")


# ─────────────────────────────────────────────
# EXPERIMENT C — NEAREST CENTROID ASSIGNMENT
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT C — Threshold-based Nearest Centroid Assignment")
print("=" * 65)

# Compute similarities for all noise points
noise_idx = np.where(labels_hard == -1)[0]
noise_embeddings = thematic[noise_idx]
sims = cosine_similarity(noise_embeddings, centroid_matrix)
nearest_cluster_idx = np.argmax(sims, axis=1)
nearest_cluster_labels = [cluster_ids[i] for i in nearest_cluster_idx]
nearest_sims = np.max(sims, axis=1)

print(f"\nSimilarity of noise points to nearest centroid:")
print(f"  Min:    {nearest_sims.min():.4f}")
print(f"  Mean:   {nearest_sims.mean():.4f}")
print(f"  Median: {np.median(nearest_sims):.4f}")
print(f"  Max:    {nearest_sims.max():.4f}")

print(f"\n{'Threshold':>12} {'Assigned':>10} {'Remaining':>11} "
      f"{'Noise%':>8} {'Purity':>8} {'NMI':>8}")
print("─" * 65)

best_threshold = None
best_score = -1

for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]:
    new_labels = labels_hard.copy()
    assigned = 0
    for idx, cluster_label, sim in zip(
            noise_idx, nearest_cluster_labels, nearest_sims):
        if sim >= threshold:
            new_labels[idx] = cluster_label
            assigned += 1
    remaining = np.sum(new_labels == -1)
    noise_pct = remaining / len(new_labels) * 100
    purity = purity_score(new_labels, topics)
    nmi = compute_nmi(new_labels, topics)
    coverage = 1 - noise_pct / 100
    score = (2 * purity * coverage / (purity + coverage)
             if (purity + coverage) > 0 else 0)
    label = '← all' if threshold == 1.0 else ''
    print(f"  {threshold:>10.2f} {assigned:>10} {remaining:>11} "
          f"{noise_pct:>7.1f}% {purity:>8.4f} {nmi:>8.4f}  {label}")
    if score > best_score and threshold < 1.0:
        best_score = score
        best_threshold = threshold

print(f"\n  {'Baseline':>10} {'—':>10} "
      f"{len(noise_idx):>11} "
      f"{len(noise_idx)/len(labels_hard)*100:>7.1f}% "
      f"{purity_score(labels_hard, topics):>8.4f} "
      f"{compute_nmi(labels_hard, topics):>8.4f}  ← baseline")

print(f"\nConclusion: Nearest centroid assignment reduces noise but "
      f"at purity cost. No threshold provides a clean tradeoff.")
print(f"Recommended threshold if used: {best_threshold}")


# ─────────────────────────────────────────────
# FINAL DECISION — KEEP HARD LABELS
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("FINAL DECISION")
print("=" * 65)
print("""
All three noise reduction experiments showed that meaningful noise
reduction consistently degrades purity. The 23.7% noise rate reflects
genuine ambiguity in open-ended reflective writing — these reflections
do not fit cleanly into any thematic cluster.

Decision: Retain HDBSCAN hard labels unchanged.
  - Noise points flagged as 'unassigned' in pipeline
  - In online phase: unassigned target reflections compared
    against full corpus (no cluster constraint) as fallback
  - Low confidence flagged → human review recommended

Final configuration:
  UMAP:    384d → 10d (nn=20, metric=cosine, min_dist=0.0)
  HDBSCAN: mcs=15, ms=3, csm=eom
  Clusters: 24
  Noise:    396 (23.7%)
  Purity:   0.2933
  NMI:      0.4148
""")

# Restore clean hard labels in CSV
df['cluster'] = labels_hard
df['cluster_assigned'] = 'hdbscan'
df.loc[labels_hard == -1, 'cluster_assigned'] = 'unassigned'
df.to_csv('data/clustered_reflections.csv', index=False)
print("Restored clean hard labels to data/clustered_reflections.csv")
print(df['cluster_assigned'].value_counts().to_string())

# ─────────────────────────────────────────────
# EXPERIMENT D — UMAP RANDOM SEED SENSITIVITY
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT D — UMAP Random Seed Sensitivity")
print("=" * 65)
print("Testing stability of clustering across different UMAP seeds...")

seeds = [42, 0, 7, 123, 999]
seed_results = []

print(f"\n{'Seed':>8} {'Clusters':>10} {'Noise%':>8} "
      f"{'Purity':>8} {'NMI':>8}")
print("─" * 50)

for seed in seeds:
    reducer_seed = __import__('umap').UMAP(
        n_components=10,
        n_neighbors=20,
        min_dist=0.0,
        metric='cosine',
        random_state=seed
    )
    reduced_seed = reducer_seed.fit_transform(thematic)

    clusterer_seed = hdb.HDBSCAN(
        min_cluster_size=15,
        min_samples=3,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels_seed = clusterer_seed.fit_predict(reduced_seed)

    n_clusters = len(set(labels_seed)) - (1 if -1 in labels_seed else 0)
    n_noise = np.sum(labels_seed == -1)
    noise_pct = n_noise / len(labels_seed) * 100
    purity = purity_score(labels_seed, topics)
    nmi = compute_nmi(labels_seed, topics)

    marker = '← baseline seed' if seed == 42 else ''
    print(f"  {seed:>6} {n_clusters:>10} {noise_pct:>7.1f}% "
          f"{purity:>8.4f} {nmi:>8.4f}  {marker}")

    seed_results.append({
        'seed': seed, 'n_clusters': n_clusters,
        'noise_pct': round(noise_pct, 1),
        'purity': round(purity, 4), 'nmi': round(nmi, 4)
    })

df_seeds = pd.DataFrame(seed_results)
print(f"\nPurity range:   {df_seeds['purity'].min():.4f} – "
      f"{df_seeds['purity'].max():.4f} "
      f"(std={df_seeds['purity'].std():.4f})")
print(f"NMI range:      {df_seeds['nmi'].min():.4f} – "
      f"{df_seeds['nmi'].max():.4f} "
      f"(std={df_seeds['nmi'].std():.4f})")
print(f"Cluster range:  {df_seeds['n_clusters'].min()} – "
      f"{df_seeds['n_clusters'].max()}")

if df_seeds['purity'].std() < 0.01:
    print("\nConclusion: Results are stable across seeds — "
          "pipeline is robust to UMAP non-determinism.")
else:
    print("\nConclusion: Results vary across seeds — "
          "seed choice affects clustering quality. "
          "seed=42 retained as fixed random state.")

df_seeds.to_csv('data/seed_sensitivity.csv', index=False)


# ─────────────────────────────────────────────
# EXPERIMENT E — TOPIC COVERAGE ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT E — Topic Coverage Analysis")
print("=" * 65)

# Reload clean hard labels
labels_final = df['cluster'].values
unique_topics = sorted(set(topics))
n_topics = len(unique_topics)
n_clusters_final = len(set(labels_final)) - (1 if -1 in labels_final else 0)

print(f"\nDataset has {n_topics} unique topics")
print(f"Pipeline produced {n_clusters_final} clusters")

# Which topics appear in which clusters
topic_cluster_map = {}
for topic in unique_topics:
    topic_idx = [i for i, t in enumerate(topics) if t == topic]
    cluster_counts = Counter([labels_final[i] for i in topic_idx])
    dominant_cluster = cluster_counts.most_common(1)[0][0]
    topic_cluster_map[topic] = {
        'n_reflections': len(topic_idx),
        'dominant_cluster': dominant_cluster,
        'n_clusters_present': len(cluster_counts),
        'dominant_pct': round(
            cluster_counts.most_common(1)[0][1] / len(topic_idx) * 100, 1)
    }

df_topics = pd.DataFrame(topic_cluster_map).T
df_topics.index.name = 'topic'
df_topics = df_topics.reset_index()

# Topics with good cluster alignment (>50% in one cluster)
well_aligned = df_topics[df_topics['dominant_pct'] >= 50]
fragmented = df_topics[df_topics['dominant_pct'] < 50]

print(f"\nTopics well-aligned to a cluster (≥50% in one cluster): "
      f"{len(well_aligned)}/{n_topics}")
print(f"Fragmented topics (<50% in one cluster): "
      f"{len(fragmented)}/{n_topics}")

# Topics in noise
noise_topics = Counter(
    [topics[i] for i in range(len(topics)) if labels_final[i] == -1]
)
print(f"\nTop 10 topics contributing to noise:")
for topic, count in noise_topics.most_common(10):
    total = topics.count(topic)
    print(f"  {topic:40s} {count:4d}/{total:4d} "
          f"({count/total*100:.1f}% noise)")

df_topics.to_csv('data/topic_coverage.csv', index=False)


# ─────────────────────────────────────────────
# EXPERIMENT F — CLUSTER SIZE DISTRIBUTION
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT F — Cluster Size Distribution")
print("=" * 65)

cluster_sizes = []
for label in sorted(set(labels_final)):
    if label == -1:
        continue
    size = np.sum(labels_final == label)
    cluster_sizes.append({'cluster': label, 'size': size})

df_sizes = pd.DataFrame(cluster_sizes).sort_values('size', ascending=False)

print(f"\nCluster size statistics:")
print(f"  Min:    {df_sizes['size'].min()}")
print(f"  Max:    {df_sizes['size'].max()}")
print(f"  Mean:   {df_sizes['size'].mean():.1f}")
print(f"  Median: {df_sizes['size'].median():.1f}")
print(f"  Std:    {df_sizes['size'].std():.1f}")

print(f"\nTop 5 largest clusters:")
print(df_sizes.head(5).to_string(index=False))

print(f"\nBottom 5 smallest clusters:")
print(df_sizes.tail(5).to_string(index=False))

# Flag highly imbalanced clusters
imbalanced = df_sizes[df_sizes['size'] < 10]
print(f"\nClusters with fewer than 10 reflections: {len(imbalanced)}")
if len(imbalanced) > 0:
    print("  These may produce unreliable Signal 1 estimates "
          "due to few available neighbours")
    print(imbalanced.to_string(index=False))

df_sizes.to_csv('data/cluster_sizes.csv', index=False)


# ─────────────────────────────────────────────
# EXPERIMENT G — NOISE POINT TOPIC DISTRIBUTION
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXPERIMENT G — Noise Point Topic Distribution")
print("=" * 65)

noise_idx_final = np.where(labels_final == -1)[0]
noise_topics_list = [topics[i] for i in noise_idx_final]
noise_topic_counts = Counter(noise_topics_list)

total_noise = len(noise_idx_final)
print(f"\nTotal noise points: {total_noise}")
print(f"Unique topics in noise: {len(noise_topic_counts)}")

print(f"\nFull noise topic distribution:")
print(f"{'Topic':45s} {'Noise':>6} {'Total':>6} {'Noise%':>8}")
print("─" * 70)

for topic, noise_count in sorted(
        noise_topic_counts.items(),
        key=lambda x: -x[1]):
    total_for_topic = topics.count(topic)
    noise_pct = noise_count / total_for_topic * 100
    print(f"  {topic:43s} {noise_count:>6} {total_for_topic:>6} "
          f"{noise_pct:>7.1f}%")

# Topics where majority of reflections are noise
high_noise_topics = {
    t: c for t, c in noise_topic_counts.items()
    if c / topics.count(t) > 0.5
}
print(f"\nTopics where >50% of reflections are noise: "
      f"{len(high_noise_topics)}")
for t, c in sorted(high_noise_topics.items(), key=lambda x: -x[1]):
    total = topics.count(t)
    print(f"  {t:43s} {c}/{total} ({c/total*100:.1f}%)")

pd.DataFrame([
    {'topic': t, 'noise_count': c,
     'total': topics.count(t),
     'noise_pct': round(c/topics.count(t)*100, 1)}
    for t, c in noise_topic_counts.items()
]).sort_values('noise_pct', ascending=False).to_csv(
    'data/noise_topic_distribution.csv', index=False)


# ─────────────────────────────────────────────
# SUMMARY OF ALL EXPERIMENTS
# ─────────────────────────────────────────────

print("\n" + "=" * 65)
print("PHASE 2.5 SUMMARY")
print("=" * 65)
print("""
Experiment A — Soft assignment:     REJECTED (purity drops significantly)
Experiment B — Epsilon tuning:      REJECTED (merges clusters aggressively)
Experiment C — Nearest centroid:    REJECTED (no clean noise/purity tradeoff)
Experiment D — Seed sensitivity:    see data/seed_sensitivity.csv
Experiment E — Topic coverage:      see data/topic_coverage.csv
Experiment F — Cluster sizes:       see data/cluster_sizes.csv
Experiment G — Noise topics:        see data/noise_topic_distribution.csv

Final configuration locked in:
  UMAP:    384d → 10d (nn=20, metric=cosine, min_dist=0.0)
  HDBSCAN: mcs=15, ms=3, csm=eom
  Clusters: 24 | Noise: 23.7% | Purity: 0.2933 | NMI: 0.4148
""")

print("Saved outputs:")
print("  data/seed_sensitivity.csv")
print("  data/topic_coverage.csv")
print("  data/cluster_sizes.csv")
print("  data/noise_topic_distribution.csv")