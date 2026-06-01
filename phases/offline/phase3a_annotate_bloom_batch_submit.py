"""
phase3a_bloom_submit.py
-----------------------
Offline Phase 3a — Submit Bloom annotation batch to Anthropic API.

Builds all annotation requests and submits as a single batch job.
Saves the batch ID to data/batch_id.txt for phase3b to collect.

Run phase3b_bloom_collect.py after this completes to retrieve results.

Usage:
    python3 phases/offline/phase3a_bloom_submit.py
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.bloom.annotate_bloom import load_system_prompt, build_user_message

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXTRACTED_CSV  = 'data/extracted_sections.csv'
CLUSTERED_CSV  = 'data/clusters/clustered_reflections.csv'
KEYWORDS_CSV   = 'data/clusters/cluster_keywords.csv'
BATCH_ID_FILE  = 'data/batch_id.txt'

MODEL          = 'claude-sonnet-4-6'
PROMPT_VERSION = 'v4'

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    print('=' * 60)
    print('PHASE 3a — BLOOM BATCH SUBMISSION')
    print('=' * 60)

    # ── Load data ──
    df_extracted = pd.read_csv(EXTRACTED_CSV)
    df_clustered = pd.read_csv(CLUSTERED_CSV)
    df_keywords  = pd.read_csv(KEYWORDS_CSV)

    print(f'Extracted sections:  {len(df_extracted)} reflections')
    print(f'Clustered:           {len(df_clustered)} reflections')

    # ── Assign cluster directly by position (same pipeline run) ──
    df_extracted = df_extracted.copy()
    df_extracted['cluster'] = df_clustered['cluster'].values

    # ── Build keyword lookup ──
    kw_lookup = dict(zip(df_keywords['cluster'], df_keywords['keywords']))

    # ── Filter non-noise ──
    df_to_annotate = df_extracted[df_extracted['cluster'] != -1].reset_index(drop=True)
    print(f'Non-noise (to annotate): {len(df_to_annotate)} reflections')
    print(f'Noise (skipped):         {len(df_extracted) - len(df_to_annotate)} reflections')

    # ── Load prompt ──
    system_prompt = load_system_prompt()

    # ── Build batch requests ──
    print(f'\nBuilding {len(df_to_annotate)} batch requests...')
    requests = []

    for idx, row in df_to_annotate.iterrows():
        cluster  = int(row['cluster'])
        keywords = kw_lookup.get(cluster, 'No keywords available')

        user_message = build_user_message(
            topic=str(row.get('topic', 'Unknown')),
            keywords=keywords,
            result=str(row.get('result', '')),
            pfr=str(row.get('pfr', '')),
            learning=str(row.get('learning', ''))
        )

        requests.append({
            "custom_id": str(idx),
            "params": {
                "model": MODEL,
                "max_tokens": 300,
                "temperature": 0,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}]
            }
        })

    print(f'Requests built: {len(requests)}')

    # ── Submit batch ──
    print('\nSubmitting batch to Anthropic API...')
    batch = client.messages.batches.create(requests=requests)

    print(f'\nBatch submitted successfully!')
    print(f'  Batch ID:  {batch.id}')
    print(f'  Status:    {batch.processing_status}')
    print(f'  Requests:  {batch.request_counts.processing}')

    # ── Save batch ID and metadata ──
    os.makedirs('data', exist_ok=True)
    with open(BATCH_ID_FILE, 'w') as f:
        f.write(batch.id)

    # Save metadata for phase3b
    metadata = {
        'batch_id':      batch.id,
        'prompt_version': PROMPT_VERSION,
        'model':         MODEL,
        'n_requests':    len(requests),
        'row_indices':   df_to_annotate.index.tolist(),
    }
    with open('data/batch_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nBatch ID saved to: {BATCH_ID_FILE}')
    print(f'Metadata saved to: data/batch_metadata.json')
    print(f'\nRun phase3b_bloom_collect.py to poll and retrieve results.')
    print(f'Typical completion time: 15-60 minutes.')

if __name__ == '__main__':
    main()