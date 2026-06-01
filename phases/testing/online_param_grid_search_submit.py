"""
phase5a_grid_submit.py
----------------------
Precomputes Signal 1 for all test reflections and submits
Signal 2 annotations as a batch job for grid search evaluation.

Run phase5b_grid_collect.py after batch completes.

Usage:
    python3 phases/testing/phase5a_grid_submit.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from phases.online.phase0_grade import (
    load_resources,
    preprocess_target,
    embed_target,
    assign_cluster,
)
from utils.bloom.annotate_bloom import load_system_prompt, build_user_message
from constants import EXAMPLE_REFLECTIONS

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OUTPUT_DIR        = 'data/evaluation'
BATCH_ID_FILE     = 'data/evaluation/grid_batch_id.txt'
CACHE_FILE        = 'data/evaluation/precomputed_signals.json'
BLOOM_MODEL       = 'claude-sonnet-4-6'
MAX_K             = 20  # precompute all neighbours up to this k

TEST_SET          = EXAMPLE_REFLECTIONS

# ─────────────────────────────────────────────
# PRECOMPUTE SIGNAL 1
# ─────────────────────────────────────────────

def precompute_signal1(embedding: np.ndarray,
                       cluster: int,
                       df_ref: pd.DataFrame,
                       ref_embeddings: np.ndarray,
                       max_k: int = MAX_K) -> dict:
    """
    Precompute all neighbour grades and similarities up to max_k.
    During grid search, slice to current k without recomputing.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cluster_mask    = df_ref['cluster'] == cluster
    cluster_indices = df_ref[cluster_mask].index.tolist()

    if len(cluster_indices) == 0:
        return {'flag': True, 'neighbours': [], 'cluster_size': 0}

    cluster_embeddings = ref_embeddings[cluster_indices]
    sims = cosine_similarity(embedding.reshape(1, -1), cluster_embeddings)[0]

    top_k       = min(max_k, len(cluster_indices))
    top_k_idx   = np.argsort(sims)[::-1][:top_k]
    top_sims    = sims[top_k_idx]
    top_grade_indices = [cluster_indices[i] for i in top_k_idx]
    top_grades  = df_ref.iloc[top_grade_indices]['grade'].values

    neighbours = [
        {
            'grade':      float(top_grades[j]),
            'similarity': float(top_sims[j]),
        }
        for j in range(top_k)
    ]

    return {
        'flag':         False,
        'neighbours':   neighbours,
        'cluster_size': len(cluster_indices),
        'avg_similarity': float(np.mean(top_sims)),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('PHASE 5a — GRID SEARCH SIGNAL PRECOMPUTATION')
    print('=' * 60)

    resources = load_resources()
    system_prompt = load_system_prompt()

    precomputed = []
    batch_requests = []

    for i, ex in enumerate(TEST_SET):
        print(f'  [{i+1}/{len(TEST_SET)}] Precomputing signals for example {ex["id"]}...')

        raw_sections = {
            'situation':   ex['situation'],
            'task_action': ex['task_action'],
            'result':      ex['result'],
            'pfr':         ex['pfr'],
            'learning':    ex['learning'],
        }

        # Preprocess + embed
        preprocessed = preprocess_target(raw_sections, resources['lemmatizer'])
        embedding    = embed_target(preprocessed, resources['sbert'])

        # Cluster assignment
        cluster_result = assign_cluster(
            embedding,
            resources['reducer'],
            resources['cluster_ids'],
            resources['centroid_matrix'],
            resources['kw_lookup'],
        )

        # Signal 1 precompute
        s1_precomputed = precompute_signal1(
            embedding,
            cluster_result['cluster'],
            resources['df_ref'],
            resources['ref_embeddings'],
            max_k=MAX_K,
        )

        # Store precomputed data
        entry = {
            'id':                    int(ex['id']),
            'expected_grade':        float(ex['grade']),
            'topic':                 ex['topic'],
            'cluster':               int(cluster_result['cluster']),
            'cluster_similarity':    float(cluster_result['cluster_similarity']),
            'cluster_confidence':    cluster_result['cluster_confidence'],
            'flag_low_confidence':   bool(cluster_result['flag_low_confidence']),
            'keywords':              cluster_result['keywords'],
            's1_neighbours':         [
                {'grade': float(n['grade']), 'similarity': float(n['similarity'])}
                for n in s1_precomputed['neighbours']
            ],
            's1_cluster_size':       int(s1_precomputed['cluster_size']),
            's1_avg_similarity':     float(s1_precomputed.get('avg_similarity', 0.0)),
            's1_flag':               bool(s1_precomputed['flag']),
        }
        precomputed.append(entry)

        # Build S2 batch request
        user_message = build_user_message(
            topic=ex['topic'],
            keywords=cluster_result['keywords'],
            result=ex['result'],
            pfr=ex['pfr'],
            learning=ex['learning'],
        )
        batch_requests.append({
            'custom_id': str(ex['id']),
            'params': {
                'model':      BLOOM_MODEL,
                'max_tokens': 300,
                'temperature': 0,
                'system':     system_prompt,
                'messages':   [{'role': 'user', 'content': user_message}],
            }
        })

    # Save precomputed S1 cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(precomputed, f, indent=2)
    print(f'\nS1 cache saved to: {CACHE_FILE}')

    # Submit S2 batch
    print(f'\nSubmitting S2 batch ({len(batch_requests)} requests)...')
    batch = client.messages.batches.create(requests=batch_requests)

    with open(BATCH_ID_FILE, 'w') as f:
        f.write(batch.id)

    metadata = {
        'batch_id':   batch.id,
        'n_requests': len(batch_requests),
        'test_set':   [ex['id'] for ex in TEST_SET],
    }
    with open('data/evaluation/grid_batch_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Batch submitted: {batch.id}')
    print(f'Run phase5b_grid_collect.py when complete.')


if __name__ == '__main__':
    main()