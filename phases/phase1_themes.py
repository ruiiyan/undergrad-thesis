import pandas as pd
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

# assume df_all contains ALL reflections (good + new) with a 'combined' column
#  → e.g. df_all["combined"] = df_all["situation"] + " " + df_all["task_action"] + " " + df_all["result"]

# 1. Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df_all["combined"].tolist(), normalize_embeddings=True, show_progress_bar=True)

# 2. Reduce (optional but helpful)
reducer = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0, random_state=42)
reduced = reducer.fit_transform(embeddings)

# 3. Cluster with HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
cluster_labels = clusterer.fit_predict(reduced)
probabilities = clusterer.probabilities_

df_all["cluster_id"] = cluster_labels
df_all["cluster_confidence"] = probabilities

print(df_all["cluster_id"].value_counts())
print(f"Noise (−1) count: {(df_all['cluster_id']==-1).sum()}")
