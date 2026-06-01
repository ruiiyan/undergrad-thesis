import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from utils import (
    centroid_similarity_stats,
    pairwise_similarity_stats,
    plot_similarity_distribution,
    plot_similarity_heatmap,
    plot_umap_projection,
    plot_similarity_subplot
)

def cluster_metrics(situation_embeddings: list, task_action_embeddings: list, result_embeddings: list ):

# 4. printing some metrics to verify the embeddings (centroid and pairwise)
    situation_centroid_sims, situation_centroid_stats = centroid_similarity_stats(situation_embeddings)
    task_action_centroid_sims, task_action_centroid_stats = centroid_similarity_stats(task_action_embeddings)
    result_centroid_sims, result_centroid_stats = centroid_similarity_stats(result_embeddings)

    print("\nSituation Centroid stats:")
    print(situation_centroid_stats)

    print("\nTask/Action Centroid stats:")
    print(task_action_centroid_stats)

    print("\nResult Centroid stats:")
    print(result_centroid_stats)

    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    plot_similarity_subplot(situation_centroid_sims, axes[0], "Situation Centroid Similarity")
    plot_similarity_subplot(task_action_centroid_sims, axes[1], "Task/Action Centroid Similarity")
    plot_similarity_subplot(result_centroid_sims, axes[2], "Result Centroid Similarity")

    xmin = min(
        situation_centroid_sims.min(),
        task_action_centroid_sims.min(),
        result_centroid_sims.min()
    )

    xmax = max(
        situation_centroid_sims.max(),
        task_action_centroid_sims.max(),
        result_centroid_sims.max()
    )

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.show()
