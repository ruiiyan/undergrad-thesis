"""
create_reflection_clusters.py
------------------------------
Full clustering pipeline for engineering student reflections.
Steps: load → embed → UMAP → HDBSCAN → Claude labels → export
"""

import json
import os
import numpy as np
import anthropic
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE       = "ANNOTATED_REFLECTIONS.json"
EMBEDDINGS_FILE  = "reflection_embeddings.npy"
OUTPUT_FILE      = "reflections_clustered.json"
EMBED_MODEL      = "all-MiniLM-L6-v2"
LABEL_MODEL      = "claude-sonnet-4-6"

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=10, metric="cosine", random_state=42)
HDBSCAN_PARAMS = dict(min_cluster_size=15, min_samples=5, metric="euclidean")
MAX_CLUSTERS_BEFORE_RERUN = 20
HDBSCAN_RERUN_MIN_CLUSTER_SIZE = 50
LABEL_SAMPLE_SIZE = 10

SUBCLUSTER_SIZE_THRESHOLD = 500          # clusters larger than this get sub-clustered
SUBCLUSTER_UMAP_PARAMS    = dict(n_neighbors=10, min_dist=0.05, n_components=5, metric="cosine", random_state=42)
SUBCLUSTER_HDBSCAN_PARAMS = dict(min_cluster_size=30, min_samples=5, metric="euclidean")

# ---------------------------------------------------------------------------
# Step 2 — Load and prepare data
# ---------------------------------------------------------------------------

print("\n[Step 2] Loading and preparing data...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

reflections = []
for band, items in raw["reflections"].items():
    for r in items:
        parts = [
            r.get("situation") or "",
            r.get("task_action") or "",
            r.get("result") or "",
        ]
        r["full_text"] = " ".join(p.strip() for p in parts if p.strip())
        reflections.append(r)

print(f"  Loaded {len(reflections)} reflections across bands: "
      + ", ".join(f"{b}={sum(1 for r in reflections if r['band']==b)}" for b in ["high","mid","low"]))

texts = [r["full_text"] for r in reflections]

# ---------------------------------------------------------------------------
# Step 3 — Embeddings
# ---------------------------------------------------------------------------

print(f"\n[Step 3] Generating embeddings with '{EMBED_MODEL}'...")

if os.path.exists(EMBEDDINGS_FILE):
    print(f"  Found cached embeddings at '{EMBEDDINGS_FILE}', loading...")
    embeddings = np.load(EMBEDDINGS_FILE)
    if embeddings.shape[0] != len(reflections):
        print(f"  Cache size mismatch ({embeddings.shape[0]} vs {len(reflections)}), regenerating...")
        embeddings = None
else:
    embeddings = None

if embeddings is None:
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"  Saved embeddings to '{EMBEDDINGS_FILE}'")

print(f"  Embeddings shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# Step 4 — UMAP dimensionality reduction
# ---------------------------------------------------------------------------

print(f"\n[Step 4] Reducing with UMAP → {UMAP_PARAMS['n_components']}D...")

reducer = umap.UMAP(**UMAP_PARAMS)
reduced = reducer.fit_transform(embeddings)
print(f"  Reduced shape: {reduced.shape}")

# ---------------------------------------------------------------------------
# Step 5 — HDBSCAN clustering
# ---------------------------------------------------------------------------

print(f"\n[Step 5] Clustering with HDBSCAN (min_cluster_size={HDBSCAN_PARAMS['min_cluster_size']})...")

clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
labels = clusterer.fit_predict(reduced)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())

print(f"  Found {n_clusters} clusters, {n_noise} noise points")

if n_clusters > MAX_CLUSTERS_BEFORE_RERUN:
    print(f"  >{MAX_CLUSTERS_BEFORE_RERUN} clusters — rerunning with min_cluster_size={HDBSCAN_RERUN_MIN_CLUSTER_SIZE}")
    clusterer = hdbscan.HDBSCAN(**{**HDBSCAN_PARAMS, "min_cluster_size": HDBSCAN_RERUN_MIN_CLUSTER_SIZE})
    labels = clusterer.fit_predict(reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  After rerun: {n_clusters} clusters, {n_noise} noise points")

print("\n  Cluster distribution:")
from collections import Counter
for cid, count in sorted(Counter(labels).items()):
    label = "noise" if cid == -1 else f"cluster {cid}"
    print(f"    {label:12}: {count}")

# ---------------------------------------------------------------------------
# Step 6 — Auto-label clusters with Claude
# ---------------------------------------------------------------------------

print(f"\n[Step 6] Labelling clusters with Claude ({LABEL_MODEL})...")

client = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are an expert in engineering education. Given a sample of student reflections, "
    "identify the single most prominent thematic topic they share in 3-5 words. "
    'Return ONLY a JSON object: {"theme": "<theme label>", "description": "<one sentence>"}'
)

cluster_ids = sorted(c for c in set(labels) if c != -1)
cluster_meta = {}

for cid in cluster_ids:
    indices = [i for i, l in enumerate(labels) if l == cid]
    sample_indices = indices[:LABEL_SAMPLE_SIZE]
    samples = "\n\n---\n\n".join(
        f"Situation: {reflections[i].get('situation','')}\n"
        f"Task/Action: {reflections[i].get('task_action','')}\n"
        f"Result: {reflections[i].get('result','')}"
        for i in sample_indices
    )
    user_msg = f"Here are sample reflections from the same cluster:\n\n{samples}"

    try:
        response = client.messages.create(
            model=LABEL_MODEL,
            max_tokens=200,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_resp = response.content[0].text.strip()
        if raw_resp.startswith("```"):
            raw_resp = raw_resp.split("```", 2)[1]
            if raw_resp.startswith("json"):
                raw_resp = raw_resp[4:]
            raw_resp = raw_resp.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw_resp)
        cluster_meta[cid] = {
            "theme": parsed.get("theme", f"Cluster {cid}"),
            "description": parsed.get("description", ""),
            "size": len(indices),
        }
    except Exception as e:
        print(f"  [warn] Cluster {cid} labelling failed: {e}")
        cluster_meta[cid] = {"theme": f"Cluster {cid}", "description": "", "size": len(indices)}

    print(f"  Cluster {cid:2d} ({len(indices):4d} reflections) → \"{cluster_meta[cid]['theme']}\"")

# ---------------------------------------------------------------------------
# Step 6b — Sub-cluster any oversized clusters
# ---------------------------------------------------------------------------

oversized = [cid for cid in cluster_ids if cluster_meta[cid]["size"] > SUBCLUSTER_SIZE_THRESHOLD]

if oversized:
    print(f"\n[Step 6b] Sub-clustering {len(oversized)} oversized cluster(s): {oversized}")

sub_labels  = labels.copy()          # will be mutated with new virtual cluster IDs
next_cid    = max(cluster_ids) + 1   # start sub-cluster IDs after existing ones
sub_cluster_meta = {}                # new entries, keyed by virtual cid

for parent_cid in oversized:
    parent_indices = [i for i, l in enumerate(labels) if l == parent_cid]
    parent_embeddings = embeddings[parent_indices]

    print(f"\n  Parent cluster {parent_cid} — \"{cluster_meta[parent_cid]['theme']}\" ({len(parent_indices)} reflections)")
    print(f"    UMAP {SUBCLUSTER_UMAP_PARAMS}  →  HDBSCAN {SUBCLUSTER_HDBSCAN_PARAMS}")

    sub_reducer  = umap.UMAP(**SUBCLUSTER_UMAP_PARAMS)
    sub_reduced  = sub_reducer.fit_transform(parent_embeddings)

    sub_clusterer = hdbscan.HDBSCAN(**SUBCLUSTER_HDBSCAN_PARAMS)
    raw_sub_labels = sub_clusterer.fit_predict(sub_reduced)

    n_sub      = len(set(raw_sub_labels)) - (1 if -1 in raw_sub_labels else 0)
    n_sub_noise = int((raw_sub_labels == -1).sum())
    print(f"    Found {n_sub} sub-clusters, {n_sub_noise} noise points")

    if n_sub < 2:
        print(f"    Only {n_sub} sub-cluster found — keeping parent cluster as-is")
        continue

    # Map sub-cluster IDs to new global IDs; noise stays with parent label
    sub_id_map = {}
    for raw_sid in sorted(set(raw_sub_labels)):
        if raw_sid == -1:
            sub_id_map[raw_sid] = parent_cid   # noise stays in parent
        else:
            sub_id_map[raw_sid] = next_cid
            next_cid += 1

    for local_pos, global_i in enumerate(parent_indices):
        sub_labels[global_i] = sub_id_map[raw_sub_labels[local_pos]]

    # Label each new sub-cluster with Claude
    new_cids = [v for v in sub_id_map.values() if v != parent_cid]
    for new_cid in sorted(new_cids):
        sub_indices = [i for i, l in enumerate(sub_labels) if l == new_cid]
        sample_indices = sub_indices[:LABEL_SAMPLE_SIZE]
        samples = "\n\n---\n\n".join(
            f"Situation: {reflections[i].get('situation','')}\n"
            f"Task/Action: {reflections[i].get('task_action','')}\n"
            f"Result: {reflections[i].get('result','')}"
            for i in sample_indices
        )
        user_msg = f"Here are sample reflections from the same cluster:\n\n{samples}"
        try:
            response = client.messages.create(
                model=LABEL_MODEL,
                max_tokens=200,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_resp = response.content[0].text.strip()
            if raw_resp.startswith("```"):
                raw_resp = raw_resp.split("```", 2)[1]
                if raw_resp.startswith("json"):
                    raw_resp = raw_resp[4:]
                raw_resp = raw_resp.rsplit("```", 1)[0].strip()
            parsed = json.loads(raw_resp)
            theme = parsed.get("theme", f"Sub-cluster {new_cid}")
            desc  = parsed.get("description", "")
        except Exception as e:
            print(f"    [warn] Sub-cluster {new_cid} labelling failed: {e}")
            theme = f"Sub-cluster {new_cid}"
            desc  = ""

        sub_cluster_meta[new_cid] = {
            "theme":       theme,
            "description": desc,
            "size":        len(sub_indices),
            "parent_cid":  parent_cid,
        }
        print(f"    Sub-cluster {new_cid:3d} ({len(sub_indices):4d} reflections) → \"{theme}\"")

    # Update parent cluster size to reflect only the noise members still assigned to it
    parent_remaining = int((sub_labels == parent_cid).sum())
    cluster_meta[parent_cid]["size"] = parent_remaining
    print(f"    Parent cluster {parent_cid} now has {parent_remaining} noise reflections")

# Merge sub_cluster_meta into cluster_meta and update cluster_ids for downstream steps
cluster_meta.update(sub_cluster_meta)
labels = sub_labels
cluster_ids = sorted(c for c in set(labels) if c != -1)

# ---------------------------------------------------------------------------
# Step 7 — Enrich reflections and export
# ---------------------------------------------------------------------------

print(f"\n[Step 7] Enriching reflections and saving to '{OUTPUT_FILE}'...")

for i, r in enumerate(reflections):
    cid = int(labels[i])
    r["cluster_id"]    = cid
    r["cluster_theme"] = "unclustered" if cid == -1 else cluster_meta[cid]["theme"]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(reflections, f, indent=2, ensure_ascii=False)

print(f"  Saved {len(reflections)} reflections")

# Summary table
print(f"\n{'─'*75}")
print(f"  {'ID':>3}  {'Theme':<32}  {'N':>5}  {'high':>5} {'mid':>4} {'low':>4}  {'Bloom':>5}")
print(f"{'─'*75}")

def bloom_weighted(r):
    return r.get("bloom", {}).get("weighted_score") or 0.0

for cid in cluster_ids:
    meta   = cluster_meta[cid]
    bucket = [r for r in reflections if r["cluster_id"] == cid]
    high   = sum(1 for r in bucket if r["band"] == "high")
    mid    = sum(1 for r in bucket if r["band"] == "mid")
    low    = sum(1 for r in bucket if r["band"] == "low")
    avg_bloom = round(sum(bloom_weighted(r) for r in bucket) / len(bucket), 2) if bucket else 0
    prefix = f"  {cid:>3}" if "parent_cid" not in meta else f"  {cid:>3}*"
    print(f"{prefix}  {meta['theme']:<32}  {meta['size']:>5}  {high:>5} {mid:>4} {low:>4}  {avg_bloom:>5.2f}")

noise_count = int((labels == -1).sum())
print(f"  {'–1':>3}  {'unclustered':<32}  {noise_count:>5}")
print(f"{'─'*75}")

# ---------------------------------------------------------------------------
# Step 8 — Validation: 2 example results per cluster
# ---------------------------------------------------------------------------

print(f"\n[Step 8] Cluster validation — 2 example results per cluster\n")

for cid in cluster_ids:
    theme = cluster_meta[cid]["theme"]
    indices = [i for i, l in enumerate(labels) if l == cid]
    examples = indices[:2]
    print(f"  Cluster {cid} — \"{theme}\"")
    for idx in examples:
        r = reflections[idx]
        preview = (r.get("result") or "")[:200].replace("\n", " ")
        print(f"    [{r['id']}] {preview}...")
    print()

print("Done.")

# ---------------------------------------------------------------------------
# Step 9 — Write summary report to txt
# ---------------------------------------------------------------------------



from datetime import datetime

# Get current date and time
now = datetime.now()

# Format as: YYYY-MM-DD HH:MM:SS
date_string = now.strftime("%Y-%m-%d %H:%M:%S")

SUMMARY_FILE = f"cluster_summary_{date_string}.txt"

lines = []
lines.append("=" * 65)
lines.append("  REFLECTION CLUSTERING SUMMARY")
lines.append(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 65)

lines.append("\n--- Parameters ---")
lines.append(f"  UMAP:    n_neighbors={UMAP_PARAMS['n_neighbors']}, min_dist={UMAP_PARAMS['min_dist']}, "
             f"n_components={UMAP_PARAMS['n_components']}, metric={UMAP_PARAMS['metric']!r}")
lines.append(f"  HDBSCAN: min_cluster_size={HDBSCAN_PARAMS['min_cluster_size']}, "
             f"min_samples={HDBSCAN_PARAMS['min_samples']}, metric={HDBSCAN_PARAMS['metric']!r}")
lines.append(f"  Label sample size: {LABEL_SAMPLE_SIZE}")
if oversized:
    lines.append(f"  Sub-cluster threshold: >{SUBCLUSTER_SIZE_THRESHOLD} reflections")
    lines.append(f"  Sub-cluster UMAP:    n_neighbors={SUBCLUSTER_UMAP_PARAMS['n_neighbors']}, min_dist={SUBCLUSTER_UMAP_PARAMS['min_dist']}, "
                 f"n_components={SUBCLUSTER_UMAP_PARAMS['n_components']}")
    lines.append(f"  Sub-cluster HDBSCAN: min_cluster_size={SUBCLUSTER_HDBSCAN_PARAMS['min_cluster_size']}, "
                 f"min_samples={SUBCLUSTER_HDBSCAN_PARAMS['min_samples']}")

lines.append(f"\n--- Overview ---")
lines.append(f"  Total reflections : {len(reflections)}")
lines.append(f"  Clusters found    : {n_clusters}")
lines.append(f"  Noise (unclustered): {noise_count} ({round(noise_count/len(reflections)*100, 1)}%)")

lines.append(f"\n--- Cluster Table ---")
lines.append(f"  {'ID':>3}  {'Theme':<35}  {'N':>5}  {'high':>5} {'mid':>4} {'low':>4}  {'Bloom':>5}")
lines.append(f"  {'-'*70}")

for cid in cluster_ids:
    meta   = cluster_meta[cid]
    bucket = [r for r in reflections if r["cluster_id"] == cid]
    high   = sum(1 for r in bucket if r["band"] == "high")
    mid    = sum(1 for r in bucket if r["band"] == "mid")
    low    = sum(1 for r in bucket if r["band"] == "low")
    avg_bloom = round(sum(bloom_weighted(r) for r in bucket) / len(bucket), 2) if bucket else 0
    tag    = " *" if "parent_cid" in meta else "  "
    lines.append(f"  {cid:>3}{tag}  {meta['theme']:<35}  {meta['size']:>5}  {high:>5} {mid:>4} {low:>4}  {avg_bloom:>5.2f}")

lines.append(f"  {'–1':>3}    {'unclustered':<35}  {noise_count:>5}")
if oversized:
    lines.append("  (* = sub-cluster derived from an oversized parent cluster)")

lines.append(f"\n--- Cluster Descriptions ---")
for cid in cluster_ids:
    meta = cluster_meta[cid]
    parent_note = f"  [sub-cluster of {meta['parent_cid']}]" if "parent_cid" in meta else ""
    lines.append(f"  [{cid}]{parent_note} {meta['theme']}")
    lines.append(f"      {meta['description']}")

lines.append(f"\n--- Validation Samples (2 results per cluster) ---")
for cid in cluster_ids:
    theme = cluster_meta[cid]["theme"]
    indices = [i for i, l in enumerate(labels) if l == cid]
    lines.append(f"\n  Cluster {cid} — \"{theme}\"")
    for idx in indices[:2]:
        r = reflections[idx]
        preview = (r.get("result") or "")[:300].replace("\n", " ")
        lines.append(f"    [{r['id']}] {preview}...")

lines.append("\n" + "=" * 65)

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n[Step 9] Summary written to '{SUMMARY_FILE}'")
