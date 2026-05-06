"""
build_batch.py
--------------
Reads STAR_REFLECTIONS.json, builds one batch request per reflection,
submits to Anthropic Batch API, and saves the batch_id for polling.

Usage:
    python3 build_batch.py            # submits the full batch
    python3 build_batch.py --dry-run  # previews requests without submitting
"""

import json
import os
import sys
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE      = "../phases/STAR_REFLECTIONS.json"
BATCH_ID_FILE   = "batch_id.txt"
MODEL           = "claude-sonnet-4-6"
MAX_TOKENS      = 1024
DRY_RUN_PREVIEW = 3

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in assessing reflective writing by undergraduate engineering students.

You will be given a student reflection written in STAR format (Situation, Task/Action, Result).
Your job is to assign a Bloom's Taxonomy level (1–6) to each section independently.

Important context:
- These are engineering undergraduates reflecting on real experiences: team projects, labs, placements, or design tasks
- Each STAR section has a different expected ceiling — do not apply the same standard across all three
- Bloom 4–5 in the Result section is the benchmark for a strong undergraduate reflection

SECTION-SPECIFIC GUIDANCE:

[SITUATION]
The situation section is scene-setting by nature. Bloom 1–2 is expected and appropriate here.
- Bloom 1: Lists what happened, where, who was involved
- Bloom 2: Explains the context or significance of the situation in their own words
- Bloom 3+: Only if the student analyses why the situation arose or contextualises it within broader engineering/professional frameworks

[TASK/ACTION]
This is where students should demonstrate applied and analytical engineering thinking.
- Bloom 2: Describes what they did without explaining why
- Bloom 3: Connects actions to specific skills, methods, or engineering principles (e.g. "I used X technique because...")
- Bloom 4: Analyses trade-offs, constraints, or decision-making process (e.g. "I chose X over Y because of Z constraint")
- Bloom 5: Critically evaluates their approach against alternatives or criteria
- Bloom 6: Proposes a novel framework, method, or generalised insight from the experience

[RESULT]
This is the most important section for reflective depth. This is where insight, learning, and professional growth should appear.

Scoring must be based on the REASONING PRESENT, not the vocabulary used. A result that uses words like
"evaluate", "assess", or "consider" without performing those cognitive acts scores no higher than Bloom 2-3.

WHAT REASONING LOOKS LIKE AT EACH LEVEL:

- Bloom 1-2: Describes the outcome only
    ("we succeeded", "the project was submitted", "we met the brief")

- Bloom 3: Describes what was done and that it worked, without explaining why
    ("we used X approach and it was effective")

- THE FOLLOWING PATTERNS SCORE NO HIGHER THAN BLOOM 3, regardless of vocabulary:
    * Listing criteria without applying them
      ("we considered time, feasibility and creativity" — criteria are named but not weighed or reasoned through)
    * Naming a skill gained without analysing how or why
      ("I improved my communication skills")
    * Describing a fix that worked without explaining the trade-off or causal mechanism
      ("I changed X and it worked better")
    * Hedged qualifications without causal reasoning
      ("it wasn't perfect but was acceptable given the circumstances")
    * Outcome attribution to vague group effort
      ("our meticulous approach led to success")

- Bloom 4: Demonstrates genuine causal analysis — the student identifies WHY the outcome occurred,
  what specifically caused success or failure, or distinguishes between multiple contributing factors.
  The reasoning must be explicit, not implied.
    ("the design failed because our load assumption didn't account for dynamic stress,
      which we had not considered during the static analysis phase")

- Bloom 5: Makes explicit criteria-based judgements, weighs competing outcomes, or reasons through
  what they would do differently and why. Must contain a genuine trade-off, counterfactual, or
  criteria-applied judgement — not just an acknowledgement that improvement is possible.
    ("I reduced the domino chain to one loop which cut failure points but made the machine
      less impressive — I judged reliability as the higher priority given our time constraints")

- Bloom 6: Synthesises a transferable insight or personal framework that generalises beyond
  this specific experience and could be applied to future engineering contexts.

SCORING PRIORITY:
Weigh the actual reasoning above all else. Casual or simple language that contains genuine
trade-off reasoning, causal analysis, or criteria-based judgement should score 4-5.
Sophisticated or academic-sounding language that contains none of these should score 1-3.
When in doubt between two levels, ask: "Is there an actual argument here, or just a description
dressed up as one?" If the latter, score down.

GLOBAL SCORING RULES:
- Assign the HIGHEST level the section CLEARLY demonstrates
- Base judgement on the reasoning present, not the sophistication of vocabulary
- Hedged or qualified descriptions are Bloom 2–3 unless accompanied by explicit causal reasoning or criteria-based judgement
- First-person casual language can still demonstrate Bloom 4–5 if the underlying reasoning is analytical
- A section that sounds evaluative but contains no actual judgement, criteria, or causal analysis should score no higher than Bloom 3

Return ONLY valid JSON in this exact format, no preamble or explanation:
{
  "situation":   { "level": <int 1-6>, "justification": "<one sentence>" },
  "task_action": { "level": <int 1-6>, "justification": "<one sentence>" },
  "result":      { "level": <int 1-6>, "justification": "<one sentence>" }
}"""

def build_user_message(reflection: dict) -> str:
    return f"""Please assign Bloom's Taxonomy levels to each section of this STAR reflection:

SITUATION:
{reflection.get('situation', '').strip()}

TASK/ACTION:
{reflection.get('task_action', '').strip()}

RESULT:
{reflection.get('result', '').strip()}"""


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(all_reflections: list, requests: list):
    """
    Preview requests without submitting. Validates IDs, sections, distribution.
    requests is a list of dicts with keys: custom_id, params
    """
    print(f"\n{'='*60}")
    print(f"  DRY RUN — no requests will be submitted")
    print(f"  Previewing {DRY_RUN_PREVIEW} of {len(requests)} requests")
    print(f"{'='*60}")

    # --- ID uniqueness check ---
    all_ids    = [r["custom_id"] for r in requests]
    unique_ids = set(all_ids)
    if len(all_ids) == len(unique_ids):
        print(f"\n✅ IDs: all {len(all_ids)} custom_ids are unique")
    else:
        dupes = [id for id in all_ids if all_ids.count(id) > 1]
        print(f"\n❌ IDs: found duplicate custom_ids: {set(dupes)}")

    # --- Empty section check ---
    empty_situation   = sum(1 for r in all_reflections if not r.get("situation",   "").strip())
    empty_task_action = sum(1 for r in all_reflections if not r.get("task_action", "").strip())
    empty_result      = sum(1 for r in all_reflections if not r.get("result",      "").strip())

    print(f"\n✅ Section completeness:")
    print(f"   situation    empty : {empty_situation}")
    print(f"   task_action  empty : {empty_task_action}")
    print(f"   result       empty : {empty_result}")
    if empty_situation + empty_task_action + empty_result == 0:
        print(f"   All sections present across all reflections")
    else:
        print(f"   ⚠️  Some reflections have empty sections — check extraction")

    # --- Band distribution ---
    band_counts = {}
    for r in all_reflections:
        band = r.get("band", "unknown")
        band_counts[band] = band_counts.get(band, 0) + 1

    print(f"\n✅ Band distribution:")
    for band, count in sorted(band_counts.items()):
        print(f"   {band:<8} : {count}")

    # --- Preview first N requests ---
    print(f"\n{'─'*60}")
    print(f"  PREVIEWING FIRST {DRY_RUN_PREVIEW} REQUESTS")
    print(f"{'─'*60}")

    for i, (reflection, request) in enumerate(
        zip(all_reflections[:DRY_RUN_PREVIEW], requests[:DRY_RUN_PREVIEW])
    ):
        user_message = request["params"]["messages"][0]["content"]

        print(f"\n[Request {i + 1}]")
        print(f"  custom_id  : {request['custom_id']}")
        print(f"  model      : {request['params']['model']}")
        print(f"  max_tokens : {request['params']['max_tokens']}")
        print(f"  band       : {reflection.get('band',  '?')}")
        print(f"  score      : {reflection.get('score', '?')}")
        print(f"\n  --- User message ---")
        for line in user_message[:600].splitlines():
            print(f"  {line}")
        if len(user_message) > 600:
            print(f"  ... [{len(user_message) - 600} more chars]")

    print(f"\n{'='*60}")
    print(f"  Dry run complete.")
    print(f"  If everything looks correct, run:")
    print(f"    python3 build_batch.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_and_submit_batch(dry_run_mode: bool = False):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all bands
    all_reflections = []
    for band in ["high", "mid", "low"]:
        all_reflections.extend(data["reflections"][band])

    print(f"Loaded {len(all_reflections)} reflections across all bands")

    # Build requests as plain dicts for dry run inspection,
    # then convert to SDK objects only when actually submitting
    requests_raw = []
    for reflection in all_reflections:
        requests_raw.append({
            "custom_id": reflection["id"],
            "params": {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "system":     SYSTEM_PROMPT,
                "messages": [
                    {
                        "role":    "user",
                        "content": build_user_message(reflection),
                    }
                ],
            }
        })

    print(f"Built {len(requests_raw)} batch requests")

    # --- Dry run: preview and exit ---
    if dry_run_mode:
        dry_run(all_reflections, requests_raw)
        return

    # --- Live run: convert to SDK format and submit ---
    print("Submitting batch to Anthropic API...")

    sdk_requests = [
        anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            custom_id=r["custom_id"],
            params=r["params"],
        )
        for r in requests_raw
    ]

    batch = client.messages.batches.create(requests=sdk_requests)

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

    print(f"\n{'='*50}")
    print(f"  Batch submitted successfully")
    print(f"  Batch ID      : {batch.id}")
    print(f"  Total requests: {len(sdk_requests)}")
    print(f"  Status        : {batch.processing_status}")
    print(f"  Saved to      : {BATCH_ID_FILE}")
    print(f"{'='*50}")
    print(f"\nRun poll_batch.py to check status and download results.")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    build_and_submit_batch(dry_run_mode=is_dry_run)