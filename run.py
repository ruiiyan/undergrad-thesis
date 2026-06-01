import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    ("Phase 0 — Data Preparation",  "phases/__preprocess/phase0_data_prep.py"),
    ("Phase 1 — Embedding",          "phases/offline/phase1_embed.py"),
    ("Phase 2 — Clustering",         "phases/offline/phase2_cluster.py"),
    ("Phase 3 — Online Grading",     "phases/online/phase0_main_grade.py"),
]

def run_phase(label: str, script: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\nERROR: {label} failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)

if __name__ == '__main__':
    for label, script in PHASES:
        run_phase(label, script)

    print(f"\n{'=' * 60}")
    print("  All phases complete.")
    print(f"{'=' * 60}")
