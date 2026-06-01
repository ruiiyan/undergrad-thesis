# ---- Offline - Phase 1 - Embedding ----
# This phase involves embedding the reference corpus
# at first, we will compare three base SBERT models
# 1. all-MiniLM-L6-v2
# 2. all-mpnet-base-v2
# 3. paraphrase-mpnet-base-v2 -> This is chosen as an interesting hypothesis further explained in Chapter 4


import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
from constants import SECTIONS, THEMATIC_SECTIONS, MODELS_TO_COMPARE, SBERT_MODEL


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

def load_model(model_name: str) -> SentenceTransformer:
    """Load SBERT model."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")
    return model


def embed_sections(df: pd.DataFrame, 
                   model: SentenceTransformer,
                   model_name: str,
                   chosen_sections: list) -> dict:
    """
    Embed all 5 preprocessed sections for every reflection.
    Returns dict of {section_name: np.array of shape (n_reflections, dim)}
    """
    print(f"\n  Embedding sections with {model_name}...")
    embeddings = {}

    for section in chosen_sections:
        col = f'preprocessed_{section}'
        texts = df[col].tolist()
        embeddings[section] = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    print(f"  Done — shape per section: {embeddings['situation'].shape}")
    return embeddings


def compute_embedding(embeddings: dict, chosen_sections: list) -> np.ndarray:
    """
    Mean aggregate THEMATIC_SECTIONS into a single representative vector per reflection.
    """
    return np.mean(
        [embeddings[s] for s in chosen_sections],
        axis=0
    )

# ─────────────────────────────────────────────
# INTRINSIC EVALUATION
# ─────────────────────────────────────────────

def evaluate_embedding_quality(thematic_embeddings: np.ndarray,
                                df: pd.DataFrame,
                                model_name: str) -> dict:
    """
    Evaluate embedding quality using topic labels as ground truth.

    Metrics:
    - Intra-topic cosine similarity: avg similarity between reflections on same topic
    - Inter-topic cosine similarity: avg similarity between reflections on different topics
    - Separability ratio: intra / inter — higher = better separation
    - Topic coverage: how many unique topics are represented
    """
    print(f"\n  Evaluating embedding quality for {model_name}...")
    topics = df['topic'].fillna('unknown').tolist()
    unique_topics = sorted(list(set(topics)))
    sample_topics = unique_topics

    intra_sims = []
    inter_sims = []

    for topic in sample_topics:
        topic_idx = [i for i, t in enumerate(topics) if t == topic]
        other_idx = [i for i, t in enumerate(topics) if t != topic]

        if len(topic_idx) < 2 or len(other_idx) < 2:
            continue

        # Intra-topic similarity — full pairwise, upper triangle only
        topic_embs = thematic_embeddings[topic_idx]
        sim_matrix = cosine_similarity(topic_embs)
        upper = sim_matrix[np.triu_indices(len(topic_idx), k=1)]
        if len(upper) > 0:
            intra_sims.extend(upper.tolist())

        # Inter-topic similarity — full pairwise (no sampling)
        other_embs = thematic_embeddings[other_idx]
        inter_sim = cosine_similarity(topic_embs, other_embs)
        inter_sims.extend(inter_sim.flatten().tolist())

    avg_intra = np.mean(intra_sims) if intra_sims else 0
    avg_inter = np.mean(inter_sims) if inter_sims else 0
    separability = avg_intra / avg_inter if avg_inter > 0 else 0

    results = {
        'model': model_name,
        'avg_intra_topic_similarity': round(avg_intra, 4),
        'avg_inter_topic_similarity': round(avg_inter, 4),
        'separability_ratio': round(separability, 4),
        'unique_topics': len(unique_topics),
        'n_reflections': len(thematic_embeddings),
        'embedding_dim': thematic_embeddings.shape[1]
    }

    print(f"    Intra-topic similarity:  {avg_intra:.4f}")
    print(f"    Inter-topic similarity:  {avg_inter:.4f}")
    print(f"    Separability ratio:      {separability:.4f}  ← higher is better")

    return results


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_embeddings(embeddings: dict, 
                    thematic: np.ndarray,
                    model_name: str,
                    section_type: str,
                    output_dir: str = 'data/embeddings') -> None:
    """Save embeddings for a given model."""
    # Sanitise model name for folder
    safe_name = model_name.replace('/', '_')
    model_dir = os.path.join(f"{output_dir}/{section_type}", safe_name)
    os.makedirs(model_dir, exist_ok=True)

    for section, emb in embeddings.items():
        np.save(os.path.join(model_dir, f'{section}.npy'), emb)

    np.save(os.path.join(model_dir, 'thematic.npy'), thematic)
    print(f"  Saved to {model_dir}/")


# def load_embeddings(model_name: str,
#                     output_dir: str = 'data/embeddings') -> tuple:
#     """Load saved embeddings for a given model."""
#     safe_name = model_name.replace('/', '_')
#     model_dir = os.path.join(output_dir, safe_name)

#     embeddings = {}
#     for section in THEMATIC_SECTIONS:
#         embeddings[section] = np.load(os.path.join(model_dir, f'{section}.npy'))

#     thematic = np.load(os.path.join(model_dir, 'thematic.npy'))
#     return embeddings, thematic


# ─────────────────────────────────────────────
# COMPARISON RUNNER
# ─────────────────────────────────────────────

def compare_models(df: pd.DataFrame,
                   models: list = MODELS_TO_COMPARE,
                   output_dir: str = 'data/embeddings') -> pd.DataFrame:
    """
    Run embedding and evaluation for all models.
    Returns a summary DataFrame for comparison.
    """
    print("=" * 60)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 60)
    all_results = []

    for model_name in models:
        print(f"\n{'─' * 40}")
        print(f"Model: {model_name}")
        print(f"{'─' * 40}")

        # Load and embed
        # model = load_model(model_name)
        # embeddings = embed_sections(df, model, model_name, THEMATIC_SECTIONS)
        # thematic = compute_embedding(embeddings, THEMATIC_SECTIONS)

        thematic = np.load(f'data/embeddings/all/{model_name}/all.npy')

        # Evaluate
        results = evaluate_embedding_quality(thematic, df, model_name)
        all_results.append(results)

        # # Save
        # save_embeddings(embeddings, thematic, model_name, 'thematic')

        # Free memory
        # del model

    # Summary table
    summary = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("SUMMARY — Embedding Model Comparison")
    print("=" * 60)
    print(summary[[
        'model', 
        'embedding_dim',
        'avg_intra_topic_similarity', 
        'avg_inter_topic_similarity',
        'separability_ratio'
    ]].to_string(index=False))
    print("\nBest separability ratio → recommended model for clustering")

    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    df = pd.read_csv('data/preprocessed_sections.csv')
    print(f"Loaded {len(df)} preprocessed reflections")

    model = load_model(SBERT_MODEL)
    embeddings = embed_sections(df, model, SBERT_MODEL, SECTIONS)
    aggregate = compute_embedding(embeddings, SECTIONS)
    embeddings['all'] = aggregate
    save_embeddings(embeddings, aggregate, SBERT_MODEL, 'all')
    print(f"\nEmbeddings saved to data/embeddings/all/{SBERT_MODEL}/")
    print("Next step: run phase2_cluster.py")