import pandas as pd
import json
import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Score band boundaries (inclusive)
SCORE_BANDS = {
    "high": (8, 10),
    "mid":  (4, 7),
    "low":  (0, 3),
}

# Minimum number of STAR sections required to keep a reflection
MIN_SECTIONS_REQUIRED = 2

# STAR section header patterns
HEADERS = {
    "situation":   re.compile(r"SITUATION\s*:", re.IGNORECASE),
    "task_action": re.compile(r"TASK/ACTION\s*:", re.IGNORECASE),
    "result":      re.compile(r"RESULT\s*:",     re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_numeric_score(value) -> bool:
    """Return True if value is a real numeric score (not 'Not yet graded' etc.)."""
    if pd.isna(value):
        return False
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def parse_score(value) -> Optional[float]:
    """Parse a cell value to float score, or None if not possible."""
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def assign_band(score: float) -> str:
    """Assign a score band label based on numeric score."""
    if 8 <= score <= 10:
        return "high"
    elif 4 <= score < 8:
        return "mid"
    elif 0 <= score < 4:
        return "low"
    else:
        return "unknown"


def parse_star_reflection(
    text: str,
    ref_id: str,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Extract SITUATION, TASK/ACTION, RESULT sections from a STAR reflection.

    Returns (parsed_dict, None) on success or (None, reason_string) on failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty cell"

    # Find where each section's content starts (after the header colon)
    matches = {}
    for key, pattern in HEADERS.items():
        m = pattern.search(text)
        if m:
            matches[key] = m.end()

    if len(matches) < MIN_SECTIONS_REQUIRED:
        found = list(matches.keys()) if matches else []
        return None, (
            f"too few sections (found: {found if found else 'none'}, "
            f"need at least {MIN_SECTIONS_REQUIRED})"
        )

    # Sort found headers by position and slice content between them
    ordered = sorted(matches.items(), key=lambda x: x[1])
    sections = {}
    for i, (key, start) in enumerate(ordered):
        if i + 1 < len(ordered):
            next_key   = ordered[i + 1][0]
            next_match = HEADERS[next_key].search(text)
            end        = next_match.start() if next_match else len(text)
        else:
            end = len(text)
        sections[key] = text[start:end].strip()

    return {
        "id":          ref_id,
        "situation":   sections.get("situation",   ""),
        "task_action": sections.get("task_action", ""),
        "result":      sections.get("result",      ""),
    }, None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_reflections(
    input_file:      str,
    score_col_idx:   int = 9,    # Column J (0-based)
    reflect_col_idx: int = 11,   # Column L (0-based)
    sheet_name:      int = 0,
    output_file:     str = "reflections.json",
) -> Tuple[dict, list, list, list]:
    """
    Load a headerless Excel file, filter and extract STAR reflections,
    bucket by score.

    Parameters
    ----------
    input_file      : path to the .xlsx file
    score_col_idx   : zero-based column index for the numeric score (default 9 = col J)
    reflect_col_idx : zero-based column index for the STAR reflection text (default 10 = col K)
    sheet_name      : sheet index or name (default 0)
    output_file     : path for the output JSON file

    Returns
    -------
    bucketed         : dict with keys "high", "mid", "low" → list of reflection dicts
    skipped_ungraded : list of IDs filtered out because score was not numeric
    skipped_empty    : list of IDs skipped because reflection cell was empty
    skipped_invalid  : list of dicts with ID + reason for structural failures
    """
    # Read without header — all columns come in as integers (0, 1, 2 ...)
    df = pd.read_excel(input_file, sheet_name=sheet_name, header=None, dtype=str)

    # Validate indices are within bounds
    max_col = df.shape[1] - 1
    for idx, label in [(score_col_idx, "score"), (reflect_col_idx, "reflection")]:
        if idx > max_col:
            raise ValueError(
                f"{label} column index {idx} is out of range "
                f"(file has {max_col + 1} columns, indices 0–{max_col})"
            )

    # Counters / accumulators
    bucketed         = {"high": [], "mid": [], "low": [], "unknown": []}
    skipped_ungraded = []
    skipped_empty    = []
    skipped_invalid  = []

    for row_num, row in df.iterrows():
        # Use 1-based row number as ID (matches Excel row numbers)
        ref_id = f"Row_{row_num + 1:04d}"

        # --- Filter: must have a numeric score ---
        raw_score = row[score_col_idx]
        if not is_numeric_score(raw_score):
            skipped_ungraded.append({
                "id":    ref_id,
                "value": str(raw_score).strip(),
            })
            continue

        score = parse_score(raw_score)
        band  = assign_band(score)

        # --- Filter: reflection cell must not be empty ---
        cell_value = row[reflect_col_idx]
        if pd.isna(cell_value) or str(cell_value).strip() == "":
            skipped_empty.append(ref_id)
            continue

        # --- Parse STAR sections ---
        parsed, reason = parse_star_reflection(str(cell_value), ref_id)

        if parsed is None:
            skipped_invalid.append({"id": ref_id, "reason": reason})
            continue

        # --- Attach score metadata and bucket ---
        parsed["score"] = score
        parsed["band"]  = band
        bucketed[band].append(parsed)

    # --- Save output ---
    output = {
        "bands":  SCORE_BANDS,
        "counts": {
            band: len(reflections)
            for band, reflections in bucketed.items()
        },
        "reflections": bucketed,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    total = (
        sum(len(v) for v in bucketed.values())
        + len(skipped_ungraded)
        + len(skipped_empty)
        + len(skipped_invalid)
    )

    print(f"\n{'='*55}")
    print(f"  Extraction Summary  ({total} rows processed)")
    print(f"{'='*55}")
    print(f"  ✅  Successfully extracted  : {sum(len(v) for v in bucketed.values())}")
    print(f"      📊  High  (8–10)        : {len(bucketed['high'])}")
    print(f"      📊  Mid   (4–7)         : {len(bucketed['mid'])}")
    print(f"      📊  Low   (0–3)         : {len(bucketed['low'])}")
    print(f"      📊  Unknown band        : {len(bucketed['unknown'])}")
    print(f"  ⏭️   Skipped (not graded)   : {len(skipped_ungraded)}")
    print(f"  ⚠️   Skipped (empty cells)  : {len(skipped_empty)}")
    print(f"  ❌  Skipped (ill-formatted) : {len(skipped_invalid)}")
    print(f"  📄  Output saved to        : {output_file}")
    print(f"{'='*55}\n")

    if skipped_ungraded:
        print("Not-yet-graded rows filtered out:")
        for s in skipped_ungraded:
            print(f"  - {s['id']} (value: '{s['value']}')")

    if skipped_empty:
        print("\nEmpty reflection cells skipped:")
        for ref_id in skipped_empty:
            print(f"  - {ref_id}")

    if skipped_invalid:
        print("\nIll-formatted STAR reflections skipped:")
        for s in skipped_invalid:
            print(f"  - {s['id']}: {s['reason']}")

    return bucketed, skipped_ungraded, skipped_empty, skipped_invalid


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bucketed, skipped_ungraded, skipped_empty, skipped_invalid = extract_reflections(
        input_file      = "../data/ENGG_2407_STAR_REFLECTIONS.xlsx",
        score_col_idx   = 9,    # Column J — numeric score 0–10
        reflect_col_idx = 11,   # Column L — STAR reflection text
        sheet_name      = 0,
        output_file     = "STAR_REFLECTIONS.json",
    )