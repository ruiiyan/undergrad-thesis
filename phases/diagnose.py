"""
diagnose.py
-----------
Inspect raw values in the score and reflection columns before running
the full extraction pipeline. Run this first to understand the data shape.
"""

import pandas as pd
import re

INPUT_FILE     = "../data/ENGG_2407_STAR_REFLECTIONS.xlsx"
SCORE_COL_IDX  = 9   # Column J
REFLECT_COL_IDX = 11  # Column K
SHEET_NAME     = 0
SAMPLE_ROWS    = 5   # Number of sample values to print per check

# ---------------------------------------------------------------------------

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None, dtype=str)

print(f"\n{'='*55}")
print(f"  File shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"{'='*55}")

# --- 1. Show first few rows of score column ---
print(f"\n--- Score column (index {SCORE_COL_IDX}) — first {SAMPLE_ROWS} raw values ---")
for i, val in enumerate(df[SCORE_COL_IDX].head(SAMPLE_ROWS)):
    print(f"  Row {i+1}: {repr(val)}")

# --- 2. Score column value distribution ---
print(f"\n--- Score column unique value counts (top 20) ---")
print(df[SCORE_COL_IDX].value_counts(dropna=False).head(20).to_string())

# --- 3. Show first few rows of reflection column ---
print(f"\n--- Reflection column (index {REFLECT_COL_IDX}) — first {SAMPLE_ROWS} raw values (truncated) ---")
for i, val in enumerate(df[REFLECT_COL_IDX].head(SAMPLE_ROWS)):
    preview = str(val)[:300].replace("\n", " ↵ ") if pd.notna(val) else "NaN"
    print(f"  Row {i+1}: {repr(preview)}")

# --- 4. Check for STAR headers in reflection column ---
print(f"\n--- STAR header detection in reflection column ---")
patterns = {
    "SITUATION:":   re.compile(r"SITUATION\s*:",   re.IGNORECASE),
    "TASK/ACTION:": re.compile(r"TASK/ACTION\s*:", re.IGNORECASE),
    "RESULT:":      re.compile(r"RESULT\s*:",       re.IGNORECASE),
}

for label, pattern in patterns.items():
    count = df[REFLECT_COL_IDX].dropna().apply(
        lambda x: bool(pattern.search(str(x)))
    ).sum()
    print(f"  '{label}' found in {count} / {df[REFLECT_COL_IDX].notna().sum()} non-empty rows")

# --- 5. Show a sample reflection that has at least one header ---
print(f"\n--- Sample reflection text (first row containing 'SITUATION:') ---")
for i, val in enumerate(df[REFLECT_COL_IDX]):
    if pd.notna(val) and re.search(r"SITUATION\s*:", str(val), re.IGNORECASE):
        print(f"  Row {i+1} (first 600 chars):")
        print(str(val)[:600])
        break
else:
    print("  No rows found containing 'SITUATION:' — header format may differ")
    print("\n  First non-empty reflection (first 600 chars):")
    for i, val in enumerate(df[REFLECT_COL_IDX]):
        if pd.notna(val) and str(val).strip():
            print(f"  Row {i+1}:")
            print(str(val)[:600])
            break

# --- 6. Total column count check ---
print(f"\n--- Column index check ---")
print(f"  File has {df.shape[1]} columns (indices 0–{df.shape[1]-1})")
print(f"  Score col index {SCORE_COL_IDX} → {'OK' if SCORE_COL_IDX < df.shape[1] else 'OUT OF RANGE'}")
print(f"  Reflect col index {REFLECT_COL_IDX} → {'OK' if REFLECT_COL_IDX < df.shape[1] else 'OUT OF RANGE'}")