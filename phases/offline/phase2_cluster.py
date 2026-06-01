import sys
sys.path.append('.')


import numpy as np
import pandas as pd
import os
import hdbscan
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from keybert import KeyBERT
from collections import Counter
from constants import SECTIONS, SBERT_MODEL, UMAP_CONFIG, HDBSCAN_CONFIG, KMEANS_K_RANGE

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def purity_score(labels: np.ndarray, topics: list) -> float:
    total = 0
    correct = 0
    unique_labels = set(labels)
    unique_labels.discard(-1)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cluster_topics = [topics[i] for i in idx]
        most_common = Counter(cluster_topics).most_common(1)[0][1]
        correct += most_common
        total += len(idx)
    return correct / total if total > 0 else 0.0


def compute_nmi(labels: np.ndarray, topics: list) -> float:
    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return 0.0
    return normalized_mutual_info_score(
        [t for t, v in zip(topics, valid_mask) if v],
        [l for l, v in zip(labels, valid_mask) if v]
    )

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(sections: np.ndarray, topics: list) -> tuple:
    """Run UMAP + HDBSCAN with locked-in configuration."""
    # UMAP reduction
    print(f"\n=== UMAP: {sections.shape[1]}d → {UMAP_CONFIG['n_components']}d ===")
    reducer = umap.UMAP(**UMAP_CONFIG)
    reduced = reducer.fit_transform(sections)
    print(f"  Output: {reduced.shape}")

    # HDBSCAN clustering
    print(f"\n=== HDBSCAN: mcs={HDBSCAN_CONFIG['min_cluster_size']}, ms={HDBSCAN_CONFIG['min_samples']} ===")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
    labels = clusterer.fit_predict(reduced)

    return labels, clusterer, reduced, reducer


def run_comparative(reduced: np.ndarray, topics: list) -> tuple:
    """K-Means and Agglomerative for comparison."""
    print("\n=== K-MEANS COMPARISON ===")
    km_results = []
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels, metric='cosine')
        purity = purity_score(labels, topics)
        nmi = normalized_mutual_info_score(topics, labels)
        km_results.append({'k': k, 'silhouette': round(sil, 4),
                           'purity': round(purity, 4),
                           'nmi': round(nmi, 4)})
        print(f"  k={k:3d} → silhouette={sil:.4f} "
              f"purity={purity:.4f} nmi={nmi:.4f}")

    print("\n=== AGGLOMERATIVE COMPARISON ===")
    agg_results = []
    for k in KMEANS_K_RANGE:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(reduced)
        sil = silhouette_score(reduced, labels, metric='cosine')
        purity = purity_score(labels, topics)
        nmi = normalized_mutual_info_score(topics, labels)
        agg_results.append({'k': k, 'silhouette': round(sil, 4),
                            'purity': round(purity, 4),
                            'nmi': round(nmi, 4)})
        print(f"  k={k:3d} → silhouette={sil:.4f} "
              f"purity={purity:.4f} nmi={nmi:.4f}")

    return pd.DataFrame(km_results), pd.DataFrame(agg_results)


def extract_cluster_keywords(df: pd.DataFrame,
                              labels: np.ndarray,
                              n_keywords: int = 5) -> dict:
    print("\n=== KEYBERT KEYWORD EXTRACTION ===")
    kw_model = KeyBERT(model=SBERT_MODEL)
    unique_labels = sorted([l for l in set(labels) if l != -1])
    cluster_keywords = {}

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        cluster_text = ' '.join([
            ' '.join([df.iloc[i][f'preprocessed_{s}']
                      for s in SECTIONS])
            for i in idx
        ])
        keywords = kw_model.extract_keywords(
            cluster_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=n_keywords
        )
        cluster_keywords[label] = [kw for kw, score in keywords]
        print(f"  Cluster {label:3d} ({len(idx):4d} reflections): "
              f"{', '.join(cluster_keywords[label])}")

    return cluster_keywords


def compute_centroids(sections: np.ndarray,
                      labels: np.ndarray) -> dict:
    """Centroids in original 384d space for online phase."""
    centroids = {}
    for label in set(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        centroids[label] = np.mean(sections[idx], axis=0)
    return centroids

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('data/preprocessed_sections.csv')
    topics = df['topic'].fillna('unknown').tolist()
    print(f"Loaded {len(df)} reflections")

    # Load embeddings - using all sections
    emb_dir = f'data/embeddings/all/{SBERT_MODEL}'
    sections = np.load(os.path.join(emb_dir, 'all.npy'))
    print(f"sections embeddings: {sections.shape}")

    # ── Run pipeline ──
    labels, clusterer, reduced, reducer = run_pipeline(sections, topics)

    # Add to dataframe
    df['cluster'] = labels
    df.to_csv('data/clusters/clustered_reflections.csv', index=False)

    # ── Comparative methods ──
    km_results, agg_results = run_comparative(reduced, topics)
    km_results.to_csv('data/clusters/kmeans_comparison.csv', index=False)
    agg_results.to_csv('data/clusters/agglomerative_comparison.csv', index=False)

    # ── KeyBERT ──
    cluster_keywords = extract_cluster_keywords(df, labels)

    # Save keywords
    kw_df = pd.DataFrame([
        {'cluster': k, 'keywords': ', '.join(v)}
        for k, v in cluster_keywords.items()
    ])
    kw_df.to_csv('data/clusters/cluster_keywords.csv', index=False)

    # ── Centroids in original space ──
    centroids = compute_centroids(sections, labels)
    np.save('data/clusters/cluster_centroids.npy', centroids)

    # ── Save UMAP reducer for online phase ──
    import pickle
    with open('data/clusters/umap_reducer.pkl', 'wb') as f:
        pickle.dump(reducer, f)

    # ── Final summary ──
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    final_purity = purity_score(labels, topics)
    final_nmi = compute_nmi(labels, topics)

    print("\n" + "=" * 60)
    print("FINAL CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"All sections:  {'+'.join([s.upper()[:3] for s in SECTIONS])}")
    print(f"\n=== UMAP: {sections.shape[1]}d → {UMAP_CONFIG['n_components']}d ===")
    print(f"\n=== HDBSCAN: mcs={HDBSCAN_CONFIG['min_cluster_size']}, ms={HDBSCAN_CONFIG['min_samples']} ===")
    print(f"HDBSCAN:            mcs={HDBSCAN_CONFIG['min_cluster_size']}, ms={HDBSCAN_CONFIG['min_samples']}, ")
    print(f"csm={HDBSCAN_CONFIG['cluster_selection_method']}")
    print(f"Clusters found:     {n_clusters}")
    print(f"Noise points:       {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"Purity score:       {final_purity:.4f}")
    print(f"NMI score:          {final_nmi:.4f}")
    print(f"\nSaved outputs:")
    print(f"  data/clustered_reflections.csv")
    print(f"  data/cluster_keywords.csv")
    print(f"  data/cluster_centroids.npy")
    print(f"  data/umap_reducer.pkl")
    print(f"  data/kmeans_comparison.csv")
    print(f"  data/agglomerative_comparison.csv")