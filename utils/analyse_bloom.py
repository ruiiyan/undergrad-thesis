
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE      = "../phases/STAR_REFLECTIONS.json"
BATCH_ID_FILE   = "batch_id.txt"
MODEL           = "claude-sonnet-4-6"
MAX_TOKENS      = 400
DRY_RUN_PREVIEW = 3

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
OG_SYSTEM_PROMPT = """You are an expert in assessing reflective writing in professional education contexts.

You will be given a student reflection structured in STAR format (Situation, Task/Action, Result).
Your job is to assign a Bloom's Taxonomy level (1–6) to each section independently.

Bloom's Taxonomy levels as they apply to reflective writing:

1 - Remember: The writer recalls or lists what happened. Pure description of facts, events, or actions with no interpretation.
2 - Understand: The writer explains or summarises their experience in their own words. They show they grasp what happened but do not yet analyse it.
3 - Apply: The writer describes using specific skills, techniques, or prior knowledge in the situation. They connect theory to practice.
4 - Analyse: The writer breaks down the experience, examines causes and effects, identifies patterns, and explores why things happened the way they did.
5 - Evaluate: The writer makes critical judgements about their actions and outcomes. They weigh alternatives, justify decisions, and assess what worked and what did not.
6 - Create: The writer synthesises insights and proposes new strategies, frameworks, or approaches. They generate something new from the experience.

Rules:
- Assign the HIGHEST level that the section clearly demonstrates
- Base your judgement on the actual reasoning present, not just the vocabulary used
- A section that uses the word "evaluate" but contains no actual evaluation should score lower
- Keep justifications to one concise sentence

Return ONLY valid JSON in this exact format, no preamble or explanation:
{
  "situation":   { "level": <int 1-6>, "justification": "<one sentence>" },
  "task_action": { "level": <int 1-6>, "justification": "<one sentence>" },
  "result":      { "level": <int 1-6>, "justification": "<one sentence>" }
}"""


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


"""
analyse_bloom.py
----------------
Analyses ANNOTATED_REFLECTIONS.json to understand Bloom's score
distribution across bands and sections.
"""

import json
from collections import Counter
import anthropic


def score_bloom(reflection: dict) -> dict:
    """
    Hit the Claude API with a single reflection and return Bloom's scores.
    reflection: dict with 'situation', 'task_action', 'result' keys
    """
    client = anthropic.Anthropic()

    user_message = (
        f"[SITUATION]\n{reflection['situation']}\n\n"
        f"[TASK/ACTION]\n{reflection['task_action']}\n\n"
        f"[RESULT]\n{reflection['result']}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]          # drop opening fence + lang tag
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip() # drop closing fence
    return json.loads(raw)

INPUT_FILE = "ANNOTATED_REFLECTIONS.json"

# ---------------------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

SECTIONS = ["situation", "task_action", "result"]
BANDS    = ["high", "mid", "low"]

print(f"\n{'='*55}")
print(f"  AVERAGE WORD COUNT PER SECTION BY BAND")
print(f"{'='*55}")

for band in BANDS:
    reflections = data["reflections"].get(band, [])
    print(f"\n  [{band.upper()}]  (n={len(reflections)})")
    for section in SECTIONS:
        word_counts = [
            len(r[section].split())
            for r in reflections
            if section in r and r[section]
        ]
        avg = round(sum(word_counts) / len(word_counts), 1) if word_counts else 0
        print(f"    {section:<12}  avg={avg} words")

# high_reflections = data["reflections"].get('high', [])
# annotated   = [r for r in high_reflections if "bloom" in r]
# lower_bloom = [] #scoring around 1-2 
# higher_bloom = [] #scoring around 4-6

# for obj in annotated: 
#     if obj['bloom'].get('result').get('level') in [1, 2]:
#         lower_bloom.append(obj)
#     elif obj['bloom'].get('result').get('level') in [4, 5, 6]:
#         higher_bloom.append(obj)

# with open("higher_bloom.json", "w", encoding="utf-8") as f:
#     json.dump(higher_bloom, f, indent=2, ensure_ascii=False)
# print(f"Wrote {len(higher_bloom)} reflections to higher_bloom.json")

# low_bloom = higher_bloom[80]
# high_bloom = higher_bloom[8]

# print(low_bloom['result'], low_bloom['bloom'].get('result').get('level'))
# print()
# print(high_bloom['result'], high_bloom['bloom'].get('result').get('level'))

# low_result = score_bloom(low_bloom)
# # {'situation': {'level': 1, 'justification': '...'}, 'task_action': {...}, 'result': {...}}
# high_result = score_bloom(high_bloom)

# print("REASSIGNED")
# print(json.dumps(low_result, indent=2))
# print(json.dumps(high_result, indent=2))

# ---------------------------------------------------------------------------

def avg(values):
    return round(sum(values) / len(values), 3) if values else None

def stats(values):
    if not values:
        return {}
    return {
        "count":  len(values),
        "mean":   avg(values),
        "min":    min(values),
        "max":    max(values),
    }

# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  BLOOM'S SCORE ANALYSIS BY BAND")
print(f"{'='*60}")

for band in BANDS:
    reflections = data["reflections"].get(band, [])
    annotated   = [r for r in reflections if "bloom" in r]

    if not annotated:
        print(f"\n[{band.upper()}] — no annotated reflections found")
        continue

    weighted_scores = [r["bloom"]["weighted_score"] for r in annotated]
    section_levels  = {s: [] for s in SECTIONS}

    for r in annotated:
        for section in SECTIONS:
            level = r["bloom"].get(section, {}).get("level")
            if level is not None:
                section_levels[section].append(level)

    print(f"\n[{band.upper()} BAND]  ({len(annotated)} annotated / {len(reflections)} total)")
    print(f"  Weighted Bloom score:")
    s = stats(weighted_scores)
    print(f"    mean={s['mean']}  min={s['min']}  max={s['max']}")

    print(f"  Section-level breakdown:")
    for section in SECTIONS:
        levels = section_levels[section]
        s = stats(levels)
        dist = dict(sorted(Counter(levels).items()))
        print(f"    {section:<12} mean={s['mean']}  dist={dist}")

# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  BLOOM LEVEL DISTRIBUTION ACROSS ALL BANDS")
print(f"{'='*60}")

for section in SECTIONS:
    print(f"\n  {section}:")
    for band in BANDS:
        reflections = data["reflections"].get(band, [])
        levels = [
            r["bloom"][section]["level"]
            for r in reflections
            if "bloom" in r and section in r["bloom"]
        ]
        dist = dict(sorted(Counter(levels).items()))
        print(f"    {band:<6} : {dist}  (mean={avg(levels)})")

# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  WEIGHTED SCORE DISTRIBUTION SUMMARY")
print(f"{'='*60}")

all_scores = {band: [] for band in BANDS}
for band in BANDS:
    for r in data["reflections"].get(band, []):
        if "bloom" in r:
            all_scores[band].append(r["bloom"]["weighted_score"])

for band in BANDS:
    scores = all_scores[band]
    if scores:
        buckets = {"1.0-2.0": 0, "2.0-3.0": 0, "3.0-4.0": 0, "4.0-5.0": 0, "5.0-6.0": 0}
        for s in scores:
            if s < 2:    buckets["1.0-2.0"] += 1
            elif s < 3:  buckets["2.0-3.0"] += 1
            elif s < 4:  buckets["3.0-4.0"] += 1
            elif s < 5:  buckets["4.0-5.0"] += 1
            else:        buckets["5.0-6.0"] += 1
        print(f"\n  {band.upper()} (n={len(scores)}):")
        for bucket, count in buckets.items():
            bar = "█" * int(count / len(scores) * 40)
            pct = round(count / len(scores) * 100, 1)
            print(f"    {bucket} : {bar} {count} ({pct}%)")