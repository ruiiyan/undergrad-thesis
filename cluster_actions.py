# modules/packages
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#util files
from utils.lemmatizer import preprocess
from utils.label_clusters import label_clusters


# 1. load in the reflections
actions_path = Path(__file__).parent / "fixtures" / "actions.js"
with open(actions_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [row["action"] for row in data]

# 2. preprocess each text
processed = [preprocess(text) for text in texts]

# 3. load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(processed, normalize_embeddings=True)
print("Embedding Shape: ", embeddings.shape)

# 4. Cluster each embedding 
# One of the challenges of K-means, is finding the best 'K' Value]
# here we employ the 'Silhouette method', where for each 'k', we
# calculate the average silhouette score, and pick the K with the highest
# score.

best_k_metrics = {
    "K":None, 
    "score": -1, 
    "labels":None, 
    "KM":None
}

for k in range(3, 8):
    KM = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = KM.fit_predict(embeddings) #here, we give the embeddings, and assign them to their cluster

    if len(set(labels)) < 2: continue # sometimes, all embeddings are in a single cluster, not really useful for us.
    s = silhouette_score(embeddings, labels)
    if s > best_k_metrics["score"]:
        best_k_metrics = {
            "K":k, 
            "score": s, 
            "labels":labels, 
            "KM":KM
        }

# after going through the range of Ks, we print out the best one we found
print(f"Best K = {best_k_metrics['K']} | Silhouette = {best_k_metrics['score']:.3f}")

# 5. cluster each embeddings, and label each one

labels = best_k_metrics["labels"]
kmeans = best_k_metrics["KM"]
centers = kmeans.cluster_centers_

clusters = label_clusters(
    texts=texts, 
    embeddings=embeddings, 
    labels=labels, 
    centers=centers, 
    top_n=5,   # using 5 phrases per cluster -> can vary
    ngram=(1,2)  # unigrams + bigrams
)

for c in clusters:
    print(f"\n--- Cluster {c['cluster']} (n={c['size']}) ---")
    print("Keywords:", ", ".join(c["keywords"]))
    print("Representative:", c["representative"])