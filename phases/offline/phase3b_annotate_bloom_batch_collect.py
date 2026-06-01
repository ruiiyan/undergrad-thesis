"""
phase3b_bloom_collect.py
------------------------
Offline Phase 3b — Poll and collect Bloom annotation batch results.

Polls the Anthropic batch API every N minutes until complete,
then downloads results and outputs annotated_reflections.csv.

Must be run after phase3a_bloom_submit.py.

Usage:
    python3 phases/offline/phase3b_bloom_collect.py
"""

import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXTRACTED_CSV   = 'data/extracted_sections.csv'
CLUSTERED_CSV   = 'data/clusters/clustered_reflections.csv'
KEYWORDS_CSV    = 'data/clusters/cluster_keywords.csv'
BATCH_ID_FILE   = 'data/batch_id.txt'
BATCH_META_FILE = 'data/batch_metadata.json'
OUTPUT_CSV      = 'data/annotated_reflections.csv'

POLL_INTERVAL   = 120   # seconds between polls (2 minutes)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_result(result) -> dict:
    """Parse a single batch result into annotation dict."""
    idx = int(result.custom_id)

    if result.result.type != 'succeeded':
        return {
            'idx': idx,
            'success': False,
            'error': result.result.error.type
        }

    raw = result.result.message.content[0].text.strip()

    try:
        # Strip markdown fences if present
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
            raw = raw.strip()

        annotation = json.loads(raw)
        return {
            'idx':     idx,
            'success': True,
            'result_bloom_level':          annotation['result']['level'],
            'result_bloom_justification':  annotation['result']['justification'],
            'pfr_bloom_level':             annotation['pfr']['level'],
            'pfr_bloom_justification':     annotation['pfr']['justification'],
            'learning_bloom_level':        annotation['learning']['level'],
            'learning_bloom_justification':annotation['learning']['justification'],
        }
    except Exception as e:
        return {'idx': idx, 'success': False, 'error': f'Parse error: {e}'}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    print('=' * 60)
    print('PHASE 3b — BLOOM BATCH COLLECTION')
    print('=' * 60)

    # ── Load batch ID ──
    if not os.path.exists(BATCH_ID_FILE):
        print(f'ERROR: {BATCH_ID_FILE} not found.')
        print('Run phase3a_bloom_submit.py first.')
        sys.exit(1)

    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    with open(BATCH_META_FILE) as f:
        metadata = json.load(f)

    prompt_version = metadata.get('prompt_version', 'unknown')
    n_requests     = metadata.get('n_requests', '?')

    print(f'Batch ID:       {batch_id}')
    print(f'Prompt version: {prompt_version}')
    print(f'Requests:       {n_requests}')
    print()

    # ── Poll until complete ──
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts

        print(f'[{time.strftime("%H:%M:%S")}] Status: {batch.processing_status} | '
              f'Processing: {counts.processing} | '
              f'Succeeded: {counts.succeeded} | '
              f'Errored: {counts.errored}')

        if batch.processing_status == 'ended':
            print('\nBatch complete!')
            break

        print(f'  Waiting {POLL_INTERVAL//60} minutes before next poll...')
        time.sleep(POLL_INTERVAL)

    # ── Download results ──
    print('\nDownloading results...')
    results = list(client.messages.batches.results(batch_id))
    print(f'Results received: {len(results)}')

    # ── Parse results ──
    print('Parsing annotations...')
    parsed = {}
    failed = []

    for result in results:
        parsed_result = parse_result(result)
        idx = parsed_result['idx']
        if parsed_result['success']:
            parsed[idx] = parsed_result
        else:
            failed.append(idx)
            print(f'  Row {idx} failed: {parsed_result.get("error")}')

    print(f'Successfully parsed: {len(parsed)}')
    print(f'Failed:             {len(failed)}')

    # ── Reconstruct dataframe ──
    print('\nReconstructing annotated dataframe...')
    df_extracted = pd.read_csv(EXTRACTED_CSV)
    df_clustered = pd.read_csv(CLUSTERED_CSV)

    df_extracted = df_extracted.copy()
    df_extracted['cluster'] = df_clustered['cluster'].values

    df_to_annotate = df_extracted[df_extracted['cluster'] != -1].reset_index(drop=True)

    # ── Add Bloom columns ──
    bloom_cols = [
        'result_bloom_level', 'result_bloom_justification',
        'pfr_bloom_level',    'pfr_bloom_justification',
        'learning_bloom_level', 'learning_bloom_justification',
        'prompt_version'
    ]
    for col in bloom_cols:
        df_to_annotate[col] = None

    # ── Apply parsed results ──
    for idx, result in parsed.items():
        df_to_annotate.at[idx, 'result_bloom_level']          = result['result_bloom_level']
        df_to_annotate.at[idx, 'result_bloom_justification']  = result['result_bloom_justification']
        df_to_annotate.at[idx, 'pfr_bloom_level']             = result['pfr_bloom_level']
        df_to_annotate.at[idx, 'pfr_bloom_justification']     = result['pfr_bloom_justification']
        df_to_annotate.at[idx, 'learning_bloom_level']        = result['learning_bloom_level']
        df_to_annotate.at[idx, 'learning_bloom_justification']= result['learning_bloom_justification']
        df_to_annotate.at[idx, 'prompt_version']              = prompt_version

    # ── Save output ──
    df_to_annotate.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──
    print()
    print('=' * 60)
    print('ANNOTATION COMPLETE')
    print('=' * 60)
    print(f'Total annotated:  {len(parsed)}')
    print(f'Failed:           {len(failed)}')
    if failed:
        print(f'Failed indices:   {failed}')
    print(f'Output saved to:  {OUTPUT_CSV}')
    print(f'Prompt version:   {prompt_version}')
    print()

    # ── Bloom level distribution ──
    print('Bloom level distribution:')
    for section in ['result', 'pfr', 'learning']:
        col = f'{section}_bloom_level'
        dist = df_to_annotate[col].value_counts().sort_index()
        print(f'  {section}: {dict(dist)}')


if __name__ == '__main__':
    main()