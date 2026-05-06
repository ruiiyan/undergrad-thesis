"""
consolidate_clusters.py
-----------------------
Merges the raw HDBSCAN clusters into ~10 final reference clusters
based on a hand-specified merge map.  No re-embedding or re-clustering.

Input  : reflections_clustered.json
Output : reflections_clustered_final.json
         cluster_summary.csv
"""

import json
import csv
from collections import Counter, defaultdict

INPUT_FILE  = "reflections_clustered.json"
OUTPUT_FILE = "reflections_clustered_final.json"
SUMMARY_CSV = "cluster_summary.csv"

# ---------------------------------------------------------------------------
# Merge map  {source cluster_id -> (final_cluster_id, final_cluster_theme)}
# ---------------------------------------------------------------------------

MERGE_MAP_RAW = {
    0:  (0,  "Coding problem-solving"),
    2:  (0,  "Coding problem-solving"),
    1:  (1,  "Technical fabrication"),
    4:  (2,  "Technical drawing"),
    6:  (3,  "RGM design and ideation"),
    9:  (3,  "RGM design and ideation"),
    12: (4,  "Team communication"),
    13: (4,  "Team communication"),
    7:  (5,  "Project planning and time management"),
    11: (5,  "Project planning and time management"),
    14: (5,  "Project planning and time management"),
    8:  (6,  "Group presentation"),
    3:  (7,  "Critical thinking"),
    10: (8,  "Collaborative design non-RGM"),
    5:  (9,  "Conceptual solution development"),
    -1: (-1, "unclustered"),
}

FINAL_ORDER = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

print("\n[Step 1] Loading data...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reflections = json.load(f)
print(f"  Loaded {len(reflections)} reflections")

# ---------------------------------------------------------------------------
# Step 2 — Apply merge map
# ---------------------------------------------------------------------------

print("\n[Step 2] Applying merge map...")

unmapped = []
for r in reflections:
    src = r["cluster_id"]
    if src in MERGE_MAP_RAW:
        fid, ftheme = MERGE_MAP_RAW[src]
    else:
        fid, ftheme = -1, "unclustered"
        unmapped.append(src)

    r["final_cluster_id"]    = fid
    r["final_cluster_theme"] = ftheme

if unmapped:
    print(f"  [warn] {len(unmapped)} reflections had unmapped cluster_ids: {sorted(set(unmapped))}")
    print(f"         → assigned to final_cluster_id=-1 (unclustered)")
else:
    print(f"  All cluster_ids mapped successfully")

# Quick count check
from collections import Counter
dist = Counter(r["final_cluster_id"] for r in reflections)
for fid in FINAL_ORDER:
    theme = MERGE_MAP_RAW.get(
        next((k for k, v in MERGE_MAP_RAW.items() if v[0] == fid), None),
        (fid, "?")
    )[1]
    print(f"  final_cluster {fid:>3}: {dist.get(fid, 0):>5} reflections")

# ---------------------------------------------------------------------------
# Step 3 — Summary table
# ---------------------------------------------------------------------------

print("\n[Step 3] Building summary table...")

def bloom_weighted(r):
    return r.get("bloom", {}).get("weighted_score") or 0.0

def avg(vals):
    return round(sum(vals) / len(vals), 3) if vals else 0.0

# Build one canonical theme name per final_cluster_id
final_theme_map = {}
for src, (fid, ftheme) in MERGE_MAP_RAW.items():
    final_theme_map[fid] = ftheme

rows = []
for fid in FINAL_ORDER:
    bucket = [r for r in reflections if r["final_cluster_id"] == fid]
    if not bucket:
        continue
    theme     = final_theme_map.get(fid, "unclustered")
    n         = len(bucket)
    high      = sum(1 for r in bucket if r.get("band") == "high")
    mid       = sum(1 for r in bucket if r.get("band") == "mid")
    low       = sum(1 for r in bucket if r.get("band") == "low")
    mean_bloom = avg([bloom_weighted(r) for r in bucket])
    mean_score = avg([r.get("score") or 0 for r in bucket])
    rows.append({
        "final_cluster_id":    fid,
        "final_cluster_theme": theme,
        "N":                   n,
        "high":                high,
        "mid":                 mid,
        "low":                 low,
        "mean_bloom":          mean_bloom,
        "mean_score":          mean_score,
    })

col_w = 38
print(f"\n  {'ID':>4}  {'Theme':<{col_w}}  {'N':>5}  {'high':>5} {'mid':>5} {'low':>5}  {'Bloom':>6}  {'Score':>6}")
print(f"  {'-'*90}")
for row in rows:
    fid_str = str(row["final_cluster_id"]).rjust(4)
    print(
        f"  {fid_str}  {row['final_cluster_theme']:<{col_w}}  "
        f"{row['N']:>5}  {row['high']:>5} {row['mid']:>5} {row['low']:>5}  "
        f"{row['mean_bloom']:>6.3f}  {row['mean_score']:>6.2f}"
    )
print(f"  {'-'*90}")
print(f"  {'':>4}  {'TOTAL':<{col_w}}  {len(reflections):>5}")

# ---------------------------------------------------------------------------
# Step 4 — Outlier flagging
# ---------------------------------------------------------------------------

print("\n[Step 4] Flagging potential outliers per final cluster...")

for fid in FINAL_ORDER:
    bucket = [r for r in reflections if r["final_cluster_id"] == fid]
    if not bucket or fid == -1:
        continue

    # Find the majority source cluster within this final cluster
    src_counts = Counter(r["cluster_id"] for r in bucket)
    majority_src = src_counts.most_common(1)[0][0]

    # Reflections whose source cluster differs from the majority
    minority = [r for r in bucket if r["cluster_id"] != majority_src]

    theme = final_theme_map.get(fid, "?")
    print(f"\n  final_cluster {fid} — \"{theme}\"")
    print(f"    Source distribution: {dict(src_counts)}")

    if not minority:
        print(f"    No minority-source reflections (all from cluster {majority_src})")
        continue

    samples = minority[:3]
    print(f"    Minority-source samples ({len(minority)} total, showing up to 3):")
    for r in samples:
        preview = (r.get("result") or "")[:220].replace("\n", " ")
        print(f"      [{r['id']}] (src cluster {r['cluster_id']}) {preview}...")

# ---------------------------------------------------------------------------
# Step 5 — Save outputs
# ---------------------------------------------------------------------------

print(f"\n[Step 5] Saving outputs...")

# JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(reflections, f, indent=2, ensure_ascii=False)
print(f"  Saved {len(reflections)} reflections → '{OUTPUT_FILE}'")

# CSV
csv_fields = ["final_cluster_id", "final_cluster_theme", "N", "high", "mid", "low", "mean_bloom", "mean_score"]
with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"  Saved {len(rows)} cluster rows → '{SUMMARY_CSV}'")

print("\nDone.")
