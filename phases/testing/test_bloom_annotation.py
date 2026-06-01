"""
test_bloom_annotation.py
------------------------
Tests Bloom's annotation on 3 random reflections from clustered_reflections.csv.
Prints results for inspection before bulk annotation run.

Usage:
    python3 test_bloom_annotation.py
"""

import os
import sys
import random
import pandas as pd
from dotenv import load_dotenv
import anthropic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.bloom.annotate_bloom import annotate_bloom

CLUSTERED_CSV   = "data/clusters/clustered_reflections.csv"
KEYWORDS_CSV    = "data/clusters/cluster_keywords.csv"
N_SAMPLES       = 3
RANDOM_SEED     = 42
MODEL           = "claude-sonnet-4-6"

# This is the breakdown of the test reflections that is annotated before the bulk annotation. 
# Trying to get a even representation of the corpus, to determine whether the prompt is working or not

# grade band: (min_grade, max_grade, num_of)
SAMPLE_PLAN = {
    'fail':        (0,   5.0,  3),
    'pass':        (5.0, 6.5,  4),
    'credit':      (6.5, 7.5,  4),
    'distinction': (7.5, 8.5,  4),
    'high_dist':   (8.5, 10.1, 4),
}

def sample_reflections(df: pd.DataFrame, kw_lookup: dict) -> list:
    """
    Sample 25 reflections:
    - 3-4 per grade band (fail/pass/credit/distinction/HD)
    - 3 paired W7+W13 from same student
    - 3 random
    """
    samples = []
    random.seed(RANDOM_SEED)
    df_valid = df[df['cluster'] != -1].reset_index(drop=True)

    # 1. Grade band sampling
    for band, (low, high, n) in SAMPLE_PLAN.items():
        band_df = df_valid[
            (df_valid['grade'] >= low) & 
            (df_valid['grade'] < high)
        ]
        if len(band_df) == 0:
            print(f"WARNING: No reflections found for band {band}")
            continue
        picked = band_df.sample(min(n, len(band_df)), random_state=RANDOM_SEED)
        for _, row in picked.iterrows():
            samples.append((band, row))
        print(f"  {band}: {len(picked)} sampled from {len(band_df)} available")

    # 2. Paired W7 + W13
    paired = df_valid.groupby('student_id').filter(
        lambda x: {'Ref1_w7', 'Ref2_w13'}.issubset(set(x['submission_ref']))
    )
    if len(paired) > 0:
        student_ids = list(paired['student_id'].unique())
        chosen_ids = random.sample(student_ids, min(3, len(student_ids)))
        for chosen_id in chosen_ids:
            student_rows = paired[paired['student_id'] == chosen_id]
            w7_row  = student_rows[student_rows['submission_ref'] == 'Ref1_w7'].iloc[0]
            w13_row = student_rows[student_rows['submission_ref'] == 'Ref2_w13'].iloc[0]
            samples.append((f'W7_{chosen_id[:8]}',  w7_row))
            samples.append((f'W13_{chosen_id[:8]}', w13_row))
    else:
        print("WARNING: No paired W7/W13 found")

    # 3. Random
    random_rows = df_valid.sample(3, random_state=RANDOM_SEED + 1)
    for _, row in random_rows.iterrows():
        samples.append(('random', row))

    print(f"\nTotal samples: {len(samples)}")
    return samples

def main():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    df = pd.read_csv(CLUSTERED_CSV)
    if 'submission_ref' not in df.columns:
        print("WARNING: submission_ref column not found — rerun phase0 first")
        df['submission_ref'] = 'unknown'

    kw_df = pd.read_csv(KEYWORDS_CSV)
    kw_lookup = dict(zip(kw_df['cluster'], kw_df['keywords']))
    df_valid = df[df['cluster'] != -1].reset_index(drop=True)
    print(f"Loaded {len(df)} reflections, {len(df_valid)} non-noise")

    samples = sample_reflections(df, kw_lookup)
    print(f"\nTesting Bloom annotation on {len(samples)} reflections")

    rows = []

    for i, (label, row) in enumerate(samples):
        topic    = str(row.get('topic', 'Unknown'))
        cluster  = int(row['cluster'])
        keywords = kw_lookup.get(cluster, 'No keywords available')
        star_result = str(row.get('result', ''))
        pfr      = str(row.get('pfr', ''))
        learning = str(row.get('learning', ''))
        grade    = row.get('grade', 'Unknown')

        print(f"  [{i+1}/{len(samples)}] [{label}] grade={grade} annotating...")

        result = annotate_bloom(
            client=client,
            topic=topic,
            keywords=keywords,
            result=star_result,
            pfr=pfr,
            learning=learning,
            model=MODEL
        )

        if result['success']:
            rows.append({
                'sample_label':          label,
                'student_id':            row.get('student_id', '?'),
                'submission_ref':        row.get('submission_ref', '?'),
                'grade':                 grade,
                'cluster':               cluster,
                'topic':                 topic[:80],
                'keywords':              keywords,
                'result_text':           star_result,
                'pfr_text':              pfr,
                'learning_text':         learning,
                'result_bloom_level':    result['result']['level'],
                'result_justification':  result['result']['justification'],
                'pfr_bloom_level':       result['pfr']['level'],
                'pfr_justification':     result['pfr']['justification'],
                'learning_bloom_level':  result['learning']['level'],
                'learning_justification':result['learning']['justification'],
                'prompt_version':        'v4',
            })
        else:
            print(f"    ERROR: {result['error']}")
            rows.append({
                'sample_label': label,
                'student_id':   row.get('student_id', '?'),
                'grade':        grade,
                'error':        result['error'],
            })

    # Save to CSV
    output_path = 'data/bloom_pilot_results.csv'
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved {len(rows)} annotations to {output_path}")
    print("Open the CSV to review annotations before bulk run.")


if __name__ == "__main__":
    main()