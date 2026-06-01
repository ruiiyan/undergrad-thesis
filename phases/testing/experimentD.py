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
# emb_dir = 'data/embeddings/all-MiniLM-L12-v2'
# thematic = np.load(os.path.join(emb_dir, 'thematic.npy'))

# # Load UMAP reducer and transform
# with open('data/umap_reducer.pkl', 'rb') as f:
#     reducer = pickle.load(f)
# reduced = reducer.transform(thematic)
# print(f"UMAP reduced shape: {reduced.shape}")

# # Load centroids
# centroids_dict = np.load('data/cluster_centroids.npy',
#                          allow_pickle=True).item()
# cluster_ids = sorted(centroids_dict.keys())
# centroid_matrix = np.stack([centroids_dict[c] for c in cluster_ids])


print("\n" + "=" * 65)
print("EXPERIMENT D — UMAP Random Seed Sensitivity")
print("=" * 65)
print("Testing stability of clustering across different UMAP seeds...")

seeds = [42, 0, 7, 123, 999]
min_dist_values = [0.0, 0.05, 0.10, 0.15, 0.17, 0.20, 0.25]
seed_results = []

print(f"\n{'Seed':>8} {'Clusters':>10} {'Noise%':>8} "
      f"{'Purity':>8} {'NMI':>8}")
print("─" * 50)

models = ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2']
scopes = ['thematic', 'all']
min_cluster_sizes = [10, 15, 20, 25]
min_samples_values = [1, 3, 5]

all_results = []

# for model in models:
#     for scope in scopes:
scope = 'all'
model = 'all-MiniLM-L12-v2'
emb_dir = f'data/embeddings/{scope}/{model}'
emb_file = 'all.npy' if scope == 'all' else 'thematic.npy'
embeddings = np.load(os.path.join(emb_dir, emb_file))

print(f"\n{'='*65}")
print(f"Model: {model} | Sections: {scope}")
print(f"{'='*65}")


for seed in seeds:
    # for min_dist in min_dist_values:

    reducer_seed = __import__('umap').UMAP(
        n_components=10,
        n_neighbors=20,
        min_dist=0.0,
        metric="cosine",
        random_state=seed
    )
    reduced_seed = reducer_seed.fit_transform(embeddings)

    for mcs in min_cluster_sizes:
        for min_samples in min_samples_values:
            clusterer_seed = hdb.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=min_samples,
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
            # print(f" {min_dist} {seed:>6} {n_clusters:>10} {noise_pct:>7.1f}% "
            #     f"{purity:>8.4f} {nmi:>8.4f}  {marker}")

            print(f" {mcs} {min_samples} {seed:>6} {n_clusters:>10} {noise_pct:>7.1f}% "
                f"{purity:>8.4f} {nmi:>8.4f}  {marker}")


            seed_results.append({
                'seed': seed,
                'mcs': mcs,          # add this
                'min_samples': min_samples,  # add this
                'n_clusters': n_clusters,
                'noise_pct': round(noise_pct, 1),
                'purity': round(purity, 4),
                'nmi': round(nmi, 4),
            })

            all_results.append({
                'model': model,
                'scope': scope,
                'seed': seed,
                # 'min_dist': min_dist,
                'mcs': mcs,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_pct': noise_pct,
                'purity': purity,
                'nmi': nmi
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

df_seeds.to_csv('data/hdbscan_sweep.csv', index=False)