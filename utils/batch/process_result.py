"""
process_results.py
------------------
Parses batch_results.jsonl and merges Bloom's taxonomy labels back into
STAR_REFLECTIONS.json using the shared reflection ID (custom_id).

Output: ANNOTATED_REFLECTIONS.json — same structure as STAR_REFLECTIONS.json
but each reflection now has a "bloom" field with section-level labels
and a weighted composite score.

Weighted composite:
    bloom_weighted_score = (0.1 * situation) + (0.3 * task_action) + (0.6 * result)
"""

import json
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REFLECTIONS_FILE = "../phases/STAR_REFLECTIONS.json"
RESULTS_FILE     = "batch_results.jsonl"
OUTPUT_FILE      = "ANNOTATED_REFLECTIONS.json"

# Section weights for weighted composite Bloom score
WEIGHTS = {
    "situation":   0.1,
    "task_action": 0.3,
    "result":      0.6,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bloom_response(raw_text: str):
    """
    Parse Claude's JSON response into a bloom dict.
    Returns None if parsing fails.
    """
    try:
        # Strip any accidental markdown fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, IndexError):
        return None


def compute_weighted_score(bloom: dict) -> float:
    """
    Compute weighted composite Bloom score from section levels.
    Falls back gracefully if a section is missing.
    """
    score = 0.0
    for section, weight in WEIGHTS.items():
        level = bloom.get(section, {}).get("level")
        if level is not None:
            score += weight * level
    return round(score, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_results():

    # --- Load original reflections ---
    with open(REFLECTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build a flat lookup: id → reflection object (with band reference)
    id_to_reflection = {}
    for band in ["high", "mid", "low"]:
        for reflection in data["reflections"][band]:
            id_to_reflection[reflection["id"]] = reflection

    print(f"Loaded {len(id_to_reflection)} reflections from {REFLECTIONS_FILE}")

    # --- Load and parse batch results ---
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"'{RESULTS_FILE}' not found. Run poll_batch.py first."
        )

    matched   = 0
    failed    = 0
    not_found = 0

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id")

            # Skip if we can't match back to a reflection
            if custom_id not in id_to_reflection:
                not_found += 1
                print(f"  [WARN] custom_id '{custom_id}' not found in reflections")
                continue

            # Check for API-level errors
            if result.get("result", {}).get("type") == "error":
                error = result["result"].get("error", {})
                print(f"  [ERROR] {custom_id}: {error.get('type')} — {error.get('message')}")
                failed += 1
                continue

            # Extract text response
            try:
                raw_text = result["result"]["message"]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                print(f"  [ERROR] {custom_id}: unexpected response structure")
                failed += 1
                continue

            # Parse Bloom JSON from response
            bloom = parse_bloom_response(raw_text)

            if bloom is None:
                print(f"  [ERROR] {custom_id}: failed to parse JSON response")
                print(f"          Raw: {raw_text[:200]}")
                failed += 1
                continue

            # Attach bloom labels + weighted score to the reflection
            bloom["weighted_score"] = compute_weighted_score(bloom)
            id_to_reflection[custom_id]["bloom"] = bloom
            matched += 1

    # --- Rebuild bucketed structure with bloom labels ---
    annotated = {
        "bands":    data["bands"],
        "counts":   data["counts"],
        "bloom_weights": WEIGHTS,
        "reflections": {
            band: data["reflections"][band]
            for band in ["high", "mid", "low"]
        }
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  Processing Summary")
    print(f"{'='*55}")
    print(f"  ✅  Successfully annotated : {matched}")
    print(f"  ❌  Failed / parse errors  : {failed}")
    print(f"  ⚠️   ID not matched         : {not_found}")
    print(f"  📄  Output saved to        : {OUTPUT_FILE}")
    print(f"{'='*55}")

    # Show a sample annotated reflection
    sample = next(
        (r for r in data["reflections"]["high"] if "bloom" in r),
        None
    )
    if sample:
        print(f"\n--- Sample annotated reflection ({sample['id']}) ---")
        print(json.dumps(sample["bloom"], indent=2))


if __name__ == "__main__":
    process_results()