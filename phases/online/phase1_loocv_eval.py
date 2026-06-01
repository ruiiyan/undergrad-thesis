"""
phase6_loocv_evaluation.py
--------------------------
Leave-One-Out Cross-Validation (LOOCV) evaluation on 200 real corpus reflections.

Samples 200 reflections stratified across grade bands from annotated_reflections.csv.
For each reflection:
  - Masks it out of Signal 1 neighbour lookup (leave-one-out)
  - Reads existing Bloom annotations from CSV (no API call)
  - Computes final grade using same pipeline logic as phase4_online.py
  - Compares against human-assigned grade

Usage:
    python3 phase6_loocv_evaluation.py

Outputs:
    data/evaluation/loocv_results.csv
    data/evaluation/loocv_summary.txt
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from constants import SBERT_MODEL, ONLINE_PARAMETERS, S2_LEARNING_WEIGHT, S2_PFR_WEIGHT, S2_RESULT_WEIGHT, S1_WEIGHT, S2_WEIGHT, BLOOM_TO_GRADE, EXPECTED_BLOOM, GRADE_TOLERANCE

# ─────────────────────────────────────────────
# PATHS — adjust to match your project structure
# ─────────────────────────────────────────────
ANNOTATED_PATH    = 'data/annotated_reflections.csv'
EMBEDDINGS_PATH   = f'data/embeddings/all/{SBERT_MODEL}/all.npy'
CENTROIDS_PATH    = 'data/clusters/cluster_centroids.npy'
KEYWORDS_PATH     = 'data/clusters/cluster_keywords.csv'
OUTPUT_DIR        = 'data/evaluation'
OUTPUT_CSV        = f'{OUTPUT_DIR}/loocv_results_03_tol_1_scaled.csv'
OUTPUT_SUMMARY    = f'{OUTPUT_DIR}/loocv_summary_03_tol_1_scaleds.txt'

K                        = ONLINE_PARAMETERS['k']
ALPHA                    = ONLINE_PARAMETERS['alpha']
MIN_MULTIPLIER           = ONLINE_PARAMETERS['min_multiplier']
S1_WEIGHT                = 0.3
S2_WEIGHT                = 1 - S1_WEIGHT
DISPARITY_THRESHOLD      = ONLINE_PARAMETERS['disparity_threshold']
SIMILARITY_THRESHOLD     = ONLINE_PARAMETERS['similarity_threshold']
CLUSTER_SIZE_THRESHOLD   = ONLINE_PARAMETERS['cluster_size_threshold']
TOLERANCE = 1

EXPECTED_BLOOM = {
    'low':  (1.0, 2.0),
    'mid':  (2.0, 3.5),
    'high': (3.5, 5.0),
}

# Confidence gate thresholds
S1_LOW_MAX  = 6.5
S1_HIGH_MIN = 7.5

# Stratified sample plan — 40 per band
# Adjust ranges to match your grade scale (0-10)
SAMPLE_PLAN = {
    'Fail':        (0.0,  5.0,  40),
    'Pass':        (5.0,  6.5,  40),
    'Credit':      (6.5,  7.5,  40),
    'Distinction': (7.5,  8.5,  40),
    'HD':          (8.5,  10.1, 40),
}

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def normalise_bloom_to_grade(bloom_weighted: float) -> float:
    import math
    lower    = max(1, min(5, math.floor(bloom_weighted)))
    upper    = max(1, min(5, math.ceil(bloom_weighted)))
    fraction = bloom_weighted - lower
    if lower == upper:
        return BLOOM_TO_GRADE[lower]
    return round(BLOOM_TO_GRADE[lower] + fraction * (BLOOM_TO_GRADE[upper] - BLOOM_TO_GRADE[lower]), 2)


def get_s1_band(grade: float, avg_similarity: float) -> str:
    sim_scalar     = 0.5 + 0.5 * avg_similarity
    adjusted_grade = grade * sim_scalar
    if adjusted_grade <= S1_LOW_MAX:
        return 'low'
    elif adjusted_grade >= S1_HIGH_MIN:
        return 'high'
    return 'mid'


def compute_bloom_penalty(bloom_weighted: float, s1_band: str) -> float:
    low, high = EXPECTED_BLOOM[s1_band]
    if bloom_weighted >= low:
        return 1.0
    deficit    = low - bloom_weighted
    bloom_range = 4.0
    penalty    = 1.0 - (deficit / bloom_range)
    return max(0.1, penalty)


def compute_signal1_loocv(target_idx: int,
                           target_embedding: np.ndarray,
                           cluster: int,
                           df_ref: pd.DataFrame,
                           ref_embeddings: np.ndarray,
                           k: int = K) -> dict:
    """
    Signal 1 with leave-one-out masking.
    Excludes the target reflection from its own neighbour lookup.
    """
    # Filter to cluster, excluding target
    cluster_mask    = (df_ref['cluster'] == cluster) & (df_ref.index != target_idx)
    cluster_indices = df_ref[cluster_mask].index.tolist()

    if len(cluster_indices) == 0:
        return {'signal1_grade': None, 'flag': True,
                'avg_similarity': 0.0, 'cluster_size': 0}

    cluster_embeddings = ref_embeddings[cluster_indices]
    sims               = cosine_similarity(
        target_embedding.reshape(1, -1), cluster_embeddings
    )[0]

    top_k     = min(k, len(cluster_indices))
    top_k_idx = np.argsort(sims)[::-1][:top_k]

    top_sims           = sims[top_k_idx]
    top_grade_indices  = [cluster_indices[i] for i in top_k_idx]
    top_grades         = df_ref.iloc[top_grade_indices]['grade'].values
    weights            = top_sims / top_sims.sum()
    signal1_grade      = float(np.dot(weights, top_grades))

    # Cluster grade normalisation — correct for grade ceiling bias
    cluster_mean  = float(np.mean(df_ref.iloc[top_grade_indices]['grade'].values))
    corpus_mean   = float(df_ref['grade'].mean())
    grade_bias    = cluster_mean - corpus_mean
    signal1_grade = float(np.clip(signal1_grade - grade_bias, 0.0, 10.0))

    return {
        'signal1_grade':    signal1_grade,
        'avg_similarity':   float(np.mean(top_sims)),
        'cluster_size':     len(sims),
        'flag':             False,
    }


def compute_signal2_from_csv(row: pd.Series) -> dict:
    """
    Read existing Bloom annotations from offline phase CSV.
    No API call needed.
    """
    r_level = float(row.get('result_bloom_level', 2.0))
    p_level = float(row.get('pfr_bloom_level', 2.0))
    l_level = float(row.get('learning_bloom_level', 2.0))

    bloom_weighted     = (S2_RESULT_WEIGHT * r_level +
                          S2_PFR_WEIGHT    * p_level +
                          S2_LEARNING_WEIGHT * l_level)
    signal2_normalised = normalise_bloom_to_grade(bloom_weighted)

    return {
        'bloom_weighted':     round(bloom_weighted, 2),
        'signal2_normalised': round(signal2_normalised, 2),
        'result_level':       r_level,
        'pfr_level':          p_level,
        'learning_level':     l_level,
    }


def confidence_gate(disparity: float,
                    avg_similarity: float,
                    cluster_size: int) -> dict:
    flags = []
    if disparity > DISPARITY_THRESHOLD:
        flags.append('high_disparity')
    if avg_similarity < SIMILARITY_THRESHOLD:
        flags.append('low_similarity')
    if cluster_size < CLUSTER_SIZE_THRESHOLD:
        flags.append('small_cluster')

    n_flags = len(flags)
    if n_flags == 0:
        confidence, flag_for_review = 'high', False
    elif n_flags == 1:
        confidence, flag_for_review = 'medium', False
    else:
        confidence, flag_for_review = 'low', True

    return {'confidence': confidence, 'flag_for_review': flag_for_review, 'flags': flags}


def compute_final_grade(s1_grade: float,
                        s2_normalised: float,
                        bloom_penalty: float,
                        disparity: float) -> float:
    adjusted_s1 = s1_grade * bloom_penalty
    final       = S1_WEIGHT * adjusted_s1 + S2_WEIGHT * s2_normalised
    return round(float(max(0.0, min(10.0, final))), 2)


def grade_to_band(grade: float) -> str:
    if grade < 5.0:
        return 'Fail'
    elif grade < 6.5:
        return 'Pass'
    elif grade < 7.5:
        return 'Credit'
    elif grade < 8.5:
        return 'Distinction'
    else:
        return 'HD'


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading offline artifacts...")
    df_ref        = pd.read_csv(ANNOTATED_PATH)
    ref_embeddings = np.load(EMBEDDINGS_PATH)

    # Filter to non-noise only
    df_valid = df_ref[df_ref['cluster'] != -1].reset_index(drop=True)
    print(f"Loaded {len(df_ref)} reflections, {len(df_valid)} non-noise")

    # Check required Bloom columns exist
    bloom_cols = ['result_bloom_level', 'pfr_bloom_level', 'learning_bloom_level']
    for col in bloom_cols:
        if col not in df_valid.columns:
            raise ValueError(f"Missing column: {col} — ensure annotated_reflections.csv includes Bloom levels")

    # ─── Stratified sampling ───
    print("\nStratified sampling...")
    sampled_indices = []
    np.random.seed(RANDOM_SEED)

    for band, (low, high, n) in SAMPLE_PLAN.items():
        band_df = df_valid[
            (df_valid['grade'] >= low) &
            (df_valid['grade'] < high)
        ]
        available = len(band_df)
        n_sample  = min(n, available)
        picked    = band_df.sample(n_sample, random_state=RANDOM_SEED)
        sampled_indices.extend(picked.index.tolist())
        print(f"  {band:12s}: {n_sample:3d} sampled from {available:4d} available "
              f"(grade {low}–{high})")

    df_sample = df_valid.loc[sampled_indices].reset_index()
    df_sample = df_sample.rename(columns={'index': 'orig_idx'})
    print(f"\nTotal sampled: {len(df_sample)} reflections")

    # ─── LOOCV evaluation ───
    print("\nRunning LOOCV evaluation...")
    rows = []

    for i, row in df_sample.iterrows():
        orig_idx   = int(row['orig_idx'])
        cluster    = int(row['cluster'])
        grade      = float(row['grade'])
        target_emb = ref_embeddings[orig_idx]

        # Signal 1 — masked
        s1 = compute_signal1_loocv(
            target_idx=orig_idx,
            target_embedding=target_emb,
            cluster=cluster,
            df_ref=df_valid,
            ref_embeddings=ref_embeddings,
            k=K
        )

        if s1['flag']:
            print(f"  [{i+1}/{len(df_sample)}] SKIP — no neighbours after masking")
            continue

        # Signal 2 — from CSV
        s2 = compute_signal2_from_csv(row)

        # Compute final grade
        s1_band       = get_s1_band(s1['signal1_grade'], s1['avg_similarity'])
        bloom_penalty = compute_bloom_penalty(s2['bloom_weighted'], s1_band)
        disparity     = abs(s1['signal1_grade'] - s2['signal2_normalised'])
        final_grade   = compute_final_grade(
            s1['signal1_grade'], s2['signal2_normalised'],
            bloom_penalty, disparity
        )

        # Confidence gate
        gate      = confidence_gate(disparity, s1['avg_similarity'], s1['cluster_size'])
        confirmed = abs(final_grade - grade) <= TOLERANCE
        band      = grade_to_band(grade)

        rows.append({
            'orig_idx':           orig_idx,
            'grade':              grade,
            'band':               band,
            'cluster':            cluster,
            'signal1_grade':      round(s1['signal1_grade'], 2),
            'signal2_normalised': s2['signal2_normalised'],
            'bloom_weighted':     s2['bloom_weighted'],
            'result_bloom':       s2['result_level'],
            'pfr_bloom':          s2['pfr_level'],
            'learning_bloom':     s2['learning_level'],
            'bloom_penalty':      round(bloom_penalty, 3),
            'disparity':          round(disparity, 2),
            'final_grade':        final_grade,
            'difference':         round(abs(final_grade - grade), 2),
            'confirmed':          confirmed,
            'confidence':         gate['confidence'],
            'flag_for_review':    gate['flag_for_review'],
            'flags':              '|'.join(gate['flags']),
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(df_sample)}...")

    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # ─── Summary ───
    total     = len(df_results)
    correct   = df_results['confirmed'].sum()
    flagged   = df_results['flag_for_review'].sum()
    accuracy  = correct / total * 100
    mae       = df_results['difference'].mean()
    rmse      = (df_results['difference'] ** 2).mean() ** 0.5

    summary_lines = [
        "=" * 60,
        "LOOCV EVALUATION SUMMARY",
        "=" * 60,
        f"Total evaluated:        {total}",
        f"Confirmed (±{TOLERANCE}):     {correct}/{total} ({accuracy:.1f}%)",
        f"Flagged for review:     {flagged}/{total} ({flagged/total*100:.1f}%)",
        f"MAE:                    {mae:.3f}",
        f"RMSE:                   {rmse:.3f}",
        "",
        "─" * 60,
        "ACCURACY BY GRADE BAND",
        "─" * 60,
    ]

    band_order = ['Fail', 'Pass', 'Credit', 'Distinction', 'HD']
    for band in band_order:
        band_df  = df_results[df_results['band'] == band]
        n        = len(band_df)
        if n == 0:
            continue
        n_correct = band_df['confirmed'].sum()
        band_acc  = n_correct / n * 100
        band_mae  = band_df['difference'].mean()
        summary_lines.append(
            f"  {band:12s}: {n_correct:3d}/{n:3d} ({band_acc:5.1f}%)  MAE={band_mae:.3f}"
        )

    summary_lines += [
        "",
        "─" * 60,
        "CONFIDENCE BREAKDOWN",
        "─" * 60,
    ]
    for conf in ['high', 'medium', 'low']:
        conf_df  = df_results[df_results['confidence'] == conf]
        n        = len(conf_df)
        if n == 0:
            continue
        n_correct = conf_df['confirmed'].sum()
        summary_lines.append(
            f"  {conf:8s}: {n_correct:3d}/{n:3d} ({n_correct/n*100:.1f}% confirmed)"
        )

    summary_lines += ["", "=" * 60]
    summary = "\n".join(summary_lines)

    print("\n" + summary)
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to {OUTPUT_SUMMARY}")


if __name__ == '__main__':
    # # import pandas as pd
    # # df = pd.read_csv('data/evaluation/loocv_results.csv')
    # # print(df[['band', 'signal1_grade', 'signal2_normalised', 'bloom_penalty', 'disparity', 'final_grade']].groupby('band').mean().round(3))
    # main()
    import pandas as pd
    df = pd.read_csv('data/annotated_reflections.csv')
print(df[df['grade'] == 10.0]['unit'].value_counts())
print()
print("Total grade 10.0:", len(df[df['grade'] == 10.0]))
print("From ENGG10xx:", len(df[(df['grade'] == 10.0) & (df['unit'].str.contains('ENGG1'))]))