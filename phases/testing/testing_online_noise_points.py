"""
evaluate_noise_points.py
------------------------
Evaluates pipeline on noise points — reflections excluded from clustering.
These are genuine student reflections unseen by Signal 1.

Two evaluations:
1. Signal 2 only — Spearman correlation between Bloom levels and grades
2. Full pipeline — assess how pipeline handles genuinely ambiguous reflections

Usage:
    python3 phases/testing/evaluate_noise_points.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.bloom.annotate_bloom import load_system_prompt, build_user_message
from phases.online.phase0_grade import (
    load_resources,
    assess_reflection,
    normalise_bloom_to_grade,
    S2_RESULT_WEIGHT, S2_PFR_WEIGHT, S2_LEARNING_WEIGHT,
    S2_NORM_CONFIG,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLUSTERED_CSV   = 'data/clusters/clustered_reflections.csv'
KEYWORDS_CSV    = 'data/clusters/cluster_keywords.csv'
OUTPUT_DIR      = 'data/evaluation'
BATCH_ID_FILE   = 'data/evaluation/noise_batch_id.txt'
CACHE_FILE      = 'data/evaluation/noise_annotations.json'
BLOOM_MODEL     = 'claude-sonnet-4-6'
POLL_INTERVAL   = 60
N_SAMPLE        = 50   # sample from 274 noise points — stratified by grade band

# ─────────────────────────────────────────────
# STEP 1 — SAMPLE NOISE POINTS
# ─────────────────────────────────────────────

def sample_noise_points(df_clustered: pd.DataFrame,
                        n: int = N_SAMPLE) -> pd.DataFrame:
    """
    Stratified sample of noise points across grade bands.
    """
    noise = df_clustered[df_clustered['cluster'] == -1].copy()
    print(f'Total noise points: {len(noise)}')
    print(f'Grade distribution:')
    print(f'  mean={noise["grade"].mean():.2f} '
          f'min={noise["grade"].min()} '
          f'max={noise["grade"].max()}')

    # Stratified sample
    bands = [
        ('fail',        0,    5.0),
        ('pass',        5.0,  6.5),
        ('credit',      6.5,  7.5),
        ('distinction', 7.5,  8.5),
        ('hd',          8.5,  10.1),
    ]

    sampled = []
    per_band = n // len(bands)

    for band, low, high in bands:
        subset = noise[(noise['grade'] >= low) & (noise['grade'] < high)]
        k = min(per_band, len(subset))
        if k > 0:
            sampled.append(subset.sample(k, random_state=42))
            print(f'  {band}: {k} sampled from {len(subset)} available')

    return pd.concat(sampled).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 2 — SUBMIT BATCH
# ─────────────────────────────────────────────

def submit_batch(df_noise: pd.DataFrame,
                 kw_lookup: dict,
                 client,
                 system_prompt: str) -> str:
    """Submit S2 annotations for noise points as batch."""

    requests = []
    for idx, row in df_noise.iterrows():
        # Use nearest cluster keywords if available, else generic
        keywords = 'No keywords available'

        user_message = build_user_message(
            topic=str(row.get('topic', 'Unknown')),
            keywords=keywords,
            result=str(row.get('result', '')),
            pfr=str(row.get('pfr', '')),
            learning=str(row.get('learning', '')),
        )
        requests.append({
            'custom_id': str(idx),
            'params': {
                'model':       BLOOM_MODEL,
                'max_tokens':  300,
                'temperature': 0,
                'system':      system_prompt,
                'messages':    [{'role': 'user', 'content': user_message}],
            }
        })

    batch = client.messages.batches.create(requests=requests)
    print(f'Batch submitted: {batch.id} ({len(requests)} requests)')

    with open(BATCH_ID_FILE, 'w') as f:
        f.write(batch.id)

    return batch.id


# ─────────────────────────────────────────────
# STEP 3 — COLLECT BATCH
# ─────────────────────────────────────────────

def collect_batch(client) -> dict:
    """Poll and collect batch results."""
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

    annotations = {}
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
            annotations[idx] = json.loads(raw)
        except Exception as e:
            print(f'  Parse error for idx {idx}: {e}')

    print(f'Annotations collected: {len(annotations)}')
    return annotations


# ─────────────────────────────────────────────
# STEP 4 — SIGNAL 2 SPEARMAN
# ─────────────────────────────────────────────

def evaluate_signal2(df_noise: pd.DataFrame,
                     annotations: dict) -> pd.DataFrame:
    """
    Compute Bloom levels from annotations and run Spearman vs grade.
    """
    rows = []
    for idx, row in df_noise.iterrows():
        if idx not in annotations:
            continue
        ann = annotations[idx]
        try:
            r = ann['result']['level']
            p = ann['pfr']['level']
            l = ann['learning']['level']
            bloom_weighted = (
                S2_RESULT_WEIGHT * r +
                S2_PFR_WEIGHT    * p +
                S2_LEARNING_WEIGHT * l
            )
            signal2_normalised = normalise_bloom_to_grade(bloom_weighted, S2_NORM_CONFIG)
            rows.append({
                'idx':                idx,
                'grade':              float(row['grade']),
                'result_bloom':       r,
                'pfr_bloom':          p,
                'learning_bloom':     l,
                'bloom_weighted':     round(bloom_weighted, 2),
                'signal2_normalised': round(signal2_normalised, 2),
                'topic':              str(row.get('topic', '')),
            })
        except Exception as e:
            print(f'  Error processing idx {idx}: {e}')

    df_results = pd.DataFrame(rows)

    # Spearman correlations
    print('\n' + '=' * 60)
    print('SIGNAL 2 SPEARMAN — NOISE POINTS (real student writing)')
    print('=' * 60)
    print(f'N = {len(df_results)}')
    print()

    for section in ['result_bloom', 'pfr_bloom', 'learning_bloom',
                    'bloom_weighted', 'signal2_normalised']:
        r, p = stats.spearmanr(df_results['grade'], df_results[section])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
        print(f'  {section:20s} — rho={r:.4f}, p={p:.4f} {sig}')

    # Grade band breakdown
    print('\nBloom levels by grade band:')
    bands = [
        ('Fail',        0,    5.0),
        ('Pass',        5.0,  6.5),
        ('Credit',      6.5,  7.5),
        ('Distinction', 7.5,  8.5),
        ('HD',          8.5,  10.1),
    ]
    print(f'  {"Band":<14} {"N":>4} {"Result":>8} '
          f'{"PFR":>8} {"Learning":>10} {"Weighted":>10}')
    print('  ' + '-' * 55)
    for band, low, high in bands:
        subset = df_results[
            (df_results['grade'] >= low) &
            (df_results['grade'] < high)
        ]
        if len(subset) == 0:
            continue
        print(f'  {band:<14} {len(subset):>4} '
              f'{subset["result_bloom"].mean():>8.2f} '
              f'{subset["pfr_bloom"].mean():>8.2f} '
              f'{subset["learning_bloom"].mean():>10.2f} '
              f'{subset["bloom_weighted"].mean():>10.2f}')

    return df_results


# ─────────────────────────────────────────────
# STEP 5 — FULL PIPELINE EVALUATION
# ─────────────────────────────────────────────

def evaluate_full_pipeline(df_noise: pd.DataFrame,
                           resources: dict,
                           n_sample: int = 20) -> pd.DataFrame:
    """
    Run full pipeline on a subsample of noise points.
    Expects high flag rate — documents how pipeline handles ambiguous cases.
    """
    print('\n' + '=' * 60)
    print('FULL PIPELINE — NOISE POINTS')
    print('=' * 60)

    sample = df_noise.sample(min(n_sample, len(df_noise)), random_state=42)
    rows   = []

    for i, (_, row) in enumerate(sample.iterrows()):
        print(f'  [{i+1}/{len(sample)}] grade={row["grade"]}...')
        try:
            result = assess_reflection(
                situation=str(row.get('situation', '')),
                task_action=str(row.get('task_action', '')),
                result=str(row.get('result', '')),
                pfr=str(row.get('pfr', '')),
                learning=str(row.get('learning', '')),
                topic=str(row.get('topic', 'Unknown')),
                resources=resources,
            )
            rows.append({
                'expected_grade':  float(row['grade']),
                'predicted_grade': result.get('final_grade'),
                'flag_for_review': result.get('flag_for_review'),
                'confidence':      result.get('confidence'),
                'cluster':         result.get('assigned_cluster'),
                'cluster_sim':     result.get('cluster_similarity'),
                's1_grade':        result.get('signal1_grade'),
                's2_bloom':        result.get('signal2_bloom_weighted'),
            })
        except Exception as e:
            print(f'    ERROR: {e}')

    df_pipeline = pd.DataFrame(rows)

    # Metrics
    valid = df_pipeline[df_pipeline['predicted_grade'].notna()]
    if len(valid) > 0:
        errors   = abs(valid['predicted_grade'] - valid['expected_grade'])
        mae      = errors.mean()
        accuracy = (errors <= 1.5).mean() * 100
        flag_rate = df_pipeline['flag_for_review'].mean() * 100

        print(f'\nResults (N={len(valid)}):')
        print(f'  Accuracy (±1.5): {accuracy:.1f}%')
        print(f'  MAE:             {mae:.3f}')
        print(f'  Flag rate:       {flag_rate:.1f}%')
        print(f'  (Expected high flag rate — noise points are ambiguous)')

    return df_pipeline


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('NOISE POINT EVALUATION')
    print('=' * 60)

    # Load data
    df_clustered = pd.read_csv(CLUSTERED_CSV)
    df_keywords  = pd.read_csv(KEYWORDS_CSV)
    kw_lookup    = dict(zip(df_keywords['cluster'], df_keywords['keywords']))

    # Sample noise points
    df_noise = sample_noise_points(df_clustered, n=N_SAMPLE)

    # Check if batch already run
    if os.path.exists(CACHE_FILE):
        print(f'\nLoading cached annotations from {CACHE_FILE}...')
        with open(CACHE_FILE) as f:
            annotations = json.load(f)
        annotations = {int(k): v for k, v in annotations.items()}
    else:
        # Submit batch
        system_prompt = load_system_prompt()
        submit_batch(df_noise, kw_lookup, client, system_prompt)

        # Collect
        annotations = collect_batch(client)

        # Cache
        with open(CACHE_FILE, 'w') as f:
            json.dump({str(k): v for k, v in annotations.items()}, f, indent=2)
        print(f'Annotations cached to {CACHE_FILE}')

    # Evaluate Signal 2
    df_s2 = evaluate_signal2(df_noise, annotations)
    df_s2.to_csv(os.path.join(OUTPUT_DIR, 'noise_signal2_results.csv'), index=False)

    # Evaluate full pipeline
    resources   = load_resources()
    df_pipeline = evaluate_full_pipeline(df_noise, resources, n_sample=20)
    df_pipeline.to_csv(
        os.path.join(OUTPUT_DIR, 'noise_pipeline_results.csv'), index=False
    )

    print('\n' + '=' * 60)
    print('OUTPUTS')
    print('=' * 60)
    print(f'  Signal 2 results:  data/evaluation/noise_signal2_results.csv')
    print(f'  Pipeline results:  data/evaluation/noise_pipeline_results.csv')


if __name__ == '__main__':
    main()