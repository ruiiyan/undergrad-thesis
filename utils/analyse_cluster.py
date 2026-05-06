"""
AI GEN
Util file for analysing 'strength' of the benchmark cluster.
- we need to analyse the 'quality'/'strength' of the benchmark cluster as it is important to ensure that this cluster (or clusters) are semantically
    similar. 
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew, kurtosis
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


def centroid_similarity_stats(embeddings):
    """
    Compute similarity of each embedding to the centroid.
    """
    
    centroid = embeddings.mean(axis=0)
    
    sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    stats = {
        "n": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std(ddof=1)),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "range": float(sims.max() - sims.min()),
        "cv": float(sims.std(ddof=1) / sims.mean()),
        "skew": float(skew(sims)),
        "kurtosis": float(kurtosis(sims))
    }
    
    return sims, stats

def pairwise_similarity_stats(embeddings):
    """
    Compute pairwise cosine similarity between all reflections.
    """
    
    sims = []
    
    for i, j in combinations(range(len(embeddings)), 2):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]
        sims.append(sim)
    
    sims = np.array(sims)
    
    stats = {
        "n_pairs": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std(ddof=1)),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "range": float(sims.max() - sims.min()),
        "cv": float(sims.std(ddof=1) / sims.mean()),
        "skew": float(skew(sims)),
        "kurtosis": float(kurtosis(sims))
    }
    
    return sims, stats

def plot_similarity_distribution(values, title):
    
    plt.figure(figsize=(6,4))
    
    plt.hist(values, bins=30)
    
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    
    plt.show()



def plot_similarity_heatmap(embeddings):
    
    sim_matrix = cosine_similarity(embeddings)

    plt.figure(figsize=(8,6))
    
    sns.heatmap(
        sim_matrix,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Cosine Similarity"}
    )

    plt.title("Pairwise Similarity Heatmap (Benchmark Reflections)")
    
    plt.xlabel("Reflection Index")
    plt.ylabel("Reflection Index")

    plt.show()

import matplotlib.pyplot as plt
import umap.umap_ as umap

def plot_umap_projection(embeddings, title="UMAP Projection of Benchmark Reflections"):
    """
    Project embeddings into 2D using UMAP and display a scatter plot.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    projection = reducer.fit_transform(embeddings)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(projection[:, 0], projection[:, 1], alpha=0.8)
    # plt.title(title)
    # plt.xlabel("UMAP Dimension 1")
    # plt.ylabel("UMAP Dimension 2")
    # plt.show()

    return projection

def plot_similarity_subplot(values, ax, title):
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")