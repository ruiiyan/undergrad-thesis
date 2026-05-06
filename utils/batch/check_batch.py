"""
poll_batch.py
-------------
Checks the status of a submitted Anthropic batch job.
When complete, downloads results and saves to batch_results.jsonl.

Run this after build_batch.py. Re-run as many times as needed until
the batch is complete.
"""

import json
import os
import time
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_ID_FILE   = "batch_id.txt"
RESULTS_FILE    = "batch_results.jsonl"
POLL_INTERVAL   = 30       # seconds between status checks when auto-polling
AUTO_POLL       = True     # set to False to just check once and exit

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def poll_batch():
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load batch_id
    if not os.path.exists(BATCH_ID_FILE):
        raise FileNotFoundError(
            f"'{BATCH_ID_FILE}' not found. Run build_batch.py first."
        )

    with open(BATCH_ID_FILE, "r") as f:
        batch_id = f.read().strip()

    print(f"Checking batch: {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        counts = batch.request_counts
        total  = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired

        print(f"\n{'='*50}")
        print(f"  Status      : {batch.processing_status}")
        print(f"  Total       : {total}")
        print(f"  Succeeded   : {counts.succeeded}")
        print(f"  Processing  : {counts.processing}")
        print(f"  Errored     : {counts.errored}")
        print(f"  Expired     : {counts.expired}")
        print(f"{'='*50}")

        if batch.processing_status == "ended":
            print("\nBatch complete — downloading results...")
            download_results(client, batch_id)
            break

        if not AUTO_POLL:
            print(f"\nBatch not yet complete. Re-run this script to check again.")
            break

        print(f"\nNot yet complete. Checking again in {POLL_INTERVAL} seconds... (Ctrl+C to stop)")
        time.sleep(POLL_INTERVAL)


def download_results(client, batch_id: str):
    """Stream batch results to a .jsonl file."""
    results = []

    for result in client.messages.batches.results(batch_id):
        results.append(result.model_dump())

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'='*50}")
    print(f"  Results saved : {RESULTS_FILE}")
    print(f"  Total results : {len(results)}")
    print(f"{'='*50}")
    print(f"\nRun process_results.py to merge Bloom labels into your reflections.")


if __name__ == "__main__":
    poll_batch()