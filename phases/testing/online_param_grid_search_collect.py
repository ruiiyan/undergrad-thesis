"""
phase5b_grid_collect.py
-----------------------
Collects S2 batch results, runs full grid search using cached signals,
finds optimal parameter combination, saves results to CSV.

Usage:
    python3 phases/testing/phase5b_grid_collect.py
"""

import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from phases.online.phase0_grade import (
    normalise_bloom_to_grade,
    compute_bloom_penalty,
    get_band_s1,
    S2_RESULT_WEIGHT, S2_PFR_WEIGHT, S2_LEARNING_WEIGHT,
    S2_NORM_CONFIG, BLOOM_TO_GRADE,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BATCH_ID_FILE  = 'data/evaluation/grid_batch_id.txt'
CACHE_FILE     = 'data/evaluation/precomputed_signals.json'
OUTPUT_DIR     = 'data/evaluation'
TOLERANCE      = 1.5
POLL_INTERVAL  = 60  # seconds

# Parameter grid
PARAM_GRID = {
    'k':                     [3, 5, 7, 10, 15],
    'alpha':                 [0.10, 0.15, 0.20, 0.25],
    'min_multiplier':        [0.3, 0.4, 0.5, 0.6],
    's1_weight':             [0.4, 0.5, 0.6, 0.7],
    'disparity_threshold':   [2.0, 2.5, 3.0, 3.5, 4.0],
    'similarity_threshold':  [0.4, 0.5, 0.6, 0.7],
    'cluster_size_threshold':[10, 15, 20],
}

# ─────────────────────────────────────────────
# COLLECT BATCH
# ─────────────────────────────────────────────

def collect_batch(client) -> dict:
    """Poll and collect S2 batch results."""
    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    print(f'Polling batch {batch_id}...')
    while True:
        batch  = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f'  [{time.strftime("%H:%M:%S")}] {batch.processing_status} | '
              f'succeeded={counts.succeeded} | errored={counts.errored}')

        if batch.processing_status == 'ended':
            break
        time.sleep(POLL_INTERVAL)

    # Parse results
    s2_annotations = {}
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type != 'succeeded':
            continue
        raw = result.result.message.content[0].text.strip()
        try:
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
                raw = raw.strip()
            annotation = json.loads(raw)
            s2_annotations[idx] = annotation
        except Exception as e:
            print(f'  Parse error for id {idx}: {e}')

    print(f'S2 annotations collected: {len(s2_annotations)}')
    return s2_annotations


# ─────────────────────────────────────────────
# EVALUATE ONE COMBINATION
# ─────────────────────────────────────────────

def evaluate_params(precomputed: list,
                    s2_annotations: dict,
                    params: dict,
                    tolerance: float = TOLERANCE) -> dict:
    """
    Evaluate one parameter combination using cached signals.
    No API calls needed.
    """
    k                    = params['k']
    alpha                = params['alpha']
    min_multiplier       = params['min_multiplier']
    s1_weight            = params['s1_weight']
    s2_weight            = 1 - s1_weight
    disparity_threshold  = params['disparity_threshold']
    similarity_threshold = params['similarity_threshold']
    cluster_size_threshold = params['cluster_size_threshold']

    correct = 0
    errors  = []
    flagged = 0

    for entry in precomputed:
        ex_id          = entry['id']
        expected_grade = entry['expected_grade']
        neighbours     = entry['s1_neighbours']
        avg_similarity = entry['s1_avg_similarity']
        cluster_size   = entry['s1_cluster_size']
        flag_low_conf  = entry['flag_low_confidence']

        # Signal 1 — slice to k
        top_k = min(k, len(neighbours))
        if top_k == 0:
            errors.append(10.0)
            flagged += 1
            continue

        top_neighbours = neighbours[:top_k]
        top_sims   = np.array([n['similarity'] for n in top_neighbours])
        top_grades = np.array([n['grade']      for n in top_neighbours])
        weights    = top_sims / top_sims.sum()
        signal1_grade = float(np.dot(weights, top_grades))
        avg_sim_k  = float(np.mean(top_sims))

        # Signal 2 — from cached annotation
        if ex_id not in s2_annotations:
            errors.append(10.0)
            flagged += 1
            continue

        ann     = s2_annotations[ex_id]
        r_level = ann['result']['level']
        p_level = ann['pfr']['level']
        l_level = ann['learning']['level']

        bloom_weighted = (
            S2_RESULT_WEIGHT * r_level +
            S2_PFR_WEIGHT    * p_level +
            S2_LEARNING_WEIGHT * l_level
        )
        signal2_normalised = normalise_bloom_to_grade(bloom_weighted, S2_NORM_CONFIG)

        # Bloom penalty
        s1_band      = get_band_s1(signal1_grade, avg_sim_k)
        bloom_penalty = compute_bloom_penalty(bloom_weighted, s1_band)

        # Disparity penalty
        disparity        = abs(signal1_grade - signal2_normalised)
        disparity_penalty = 1 / (1 + alpha * disparity)

        # Final grade
        adjusted_s1  = signal1_grade * bloom_penalty * disparity_penalty
        final_grade  = s1_weight * adjusted_s1 + s2_weight * signal2_normalised
        final_grade  = float(max(0.0, min(10.0, final_grade)))

        # Confidence gate flags
        flags = []
        if disparity > disparity_threshold:
            flags.append('high_disparity')
        if avg_sim_k < similarity_threshold:
            flags.append('low_similarity')
        if cluster_size < cluster_size_threshold:
            flags.append('small_cluster')
        if flag_low_conf:
            flags.append('low_cluster_confidence')

        n_flags = len(flags)
        flag_for_review = n_flags >= 2

        # Metrics
        diff = abs(final_grade - expected_grade)
        errors.append(diff)
        if diff <= tolerance:
            correct += 1
        if flag_for_review:
            flagged += 1

    n        = len(precomputed)
    mae      = sum(errors) / n
    rmse     = (sum(e**2 for e in errors) / n) ** 0.5
    accuracy = correct / n * 100
    flag_rate = flagged / n * 100

    return {
        **params,
        'accuracy':  round(accuracy, 1),
        'mae':       round(mae, 3),
        'rmse':      round(rmse, 3),
        'flag_rate': round(flag_rate, 1),
        'n_correct': correct,
        'n_flagged': flagged,
        'n_total':   n,
    }


# ─────────────────────────────────────────────
# FIND OPTIMAL
# ─────────────────────────────────────────────

def find_optimal(df: pd.DataFrame) -> pd.Series:
    # Filter to max accuracy first
    max_acc = df['accuracy'].max()
    top_acc = df[df['accuracy'] == max_acc]
    
    # Among max accuracy, find lowest MAE with flag_rate between 10-30%
    reasonable_flag = top_acc[
        (top_acc['flag_rate'] >= 10) & 
        (top_acc['flag_rate'] <= 30)
    ]
    
    if len(reasonable_flag) > 0:
        return reasonable_flag.sort_values('mae').iloc[0]
    
    # Fallback to lowest MAE if no reasonable flag rate found
    return top_acc.sort_values('mae').iloc[0]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('PHASE 5b — GRID SEARCH COLLECTION & EVALUATION')
    print('=' * 60)

    # Load precomputed S1 cache
    with open(CACHE_FILE) as f:
        precomputed = json.load(f)
    print(f'Loaded S1 cache: {len(precomputed)} reflections')

    # Collect S2 batch
    s2_annotations = collect_batch(client)

    # Build grid
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f'\nRunning grid search: {total} combinations...')

    rows = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        result = evaluate_params(precomputed, s2_annotations, params)
        rows.append(result)

        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{total} combinations evaluated...')

    df = pd.DataFrame(rows)

    # Save full grid results
    full_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    df.to_csv(full_path, index=False)
    print(f'\nFull grid results saved to: {full_path}')

    # Find optimal
    optimal = find_optimal(df)
    optimal_path = os.path.join(OUTPUT_DIR, 'optimal_params.csv')
    pd.DataFrame([optimal]).to_csv(optimal_path, index=False)

    print('\n' + '=' * 60)
    print('OPTIMAL PARAMETERS')
    print('=' * 60)
    for key in keys:
        print(f'  {key:25s} = {optimal[key]}')
    print()
    print(f'  accuracy  = {optimal["accuracy"]}%')
    print(f'  mae       = {optimal["mae"]}')
    print(f'  rmse      = {optimal["rmse"]}')
    print(f'  flag_rate = {optimal["flag_rate"]}%')

    # Top 10 combinations
    print('\nTop 10 combinations:')
    top10 = df.sort_values(['accuracy', 'mae'], ascending=[False, True]).head(10)
    print(top10[keys + ['accuracy', 'mae', 'rmse', 'flag_rate']].to_string(index=False))

    print(f'\nOptimal params saved to: {optimal_path}')


if __name__ == '__main__':
    main()