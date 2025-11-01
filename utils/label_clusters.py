from keybert import KeyBERT
import numpy as np

# using keybert for now

def label_clusters(texts, embeddings, labels, centers, ngram=(1,2), top_n=5):
    """
    texts: original texts for readibility 
    embeddings: embedded texts
    labels: the assigned cluster, for each text
    centers : each cluster centroids, from KMeans()
    """

    keyBert = KeyBERT(model="all-MiniLM-L6-v2")
    clusters = []

    for c in sorted(set(labels)):
        ids = [i for i, lab in enumerate(labels) if lab == c]
        cluster_members = [texts[i] for i in ids]
        cluster_doc = "\n".join(cluster_members)

        # extract label keywords/phrases for a specific cluster
        kw = keyBert.extract_keywords(
            cluster_doc,
            keyphrase_ngram_range=ngram, 
            stop_words="english", # this removes any generic stop_words -> maybe we can have more specific stop_words, specific to our context
            use_mmr=True,
            top_n=top_n
        )
        keywords = [k for k, _ in kw]

        # choose a representative example, defined by being closest to the centroid
        distances = np.linalg.norm(embeddings[ids] - centers[c], axis=1)
        representative = cluster_members[int(np.argmin(distances))]

        clusters.append({
            "cluster": c, 
            "size": len(ids),
            "keywords": keywords, 
            "representative": representative, 
            "members_ids": ids
        })
    return clusters