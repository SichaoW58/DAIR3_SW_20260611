"""
submit_finetune.py
Upload train.jsonl, kick off the OpenAI fine-tune job, wait, print the
resulting model id.

Run order:
    python assemble_dataset.py     # writes train.jsonl + eval.jsonl
    python submit_finetune.py      # this script
    python evaluate.py             # compares base vs fine-tuned

Costs (sample dataset, gpt-4o-mini-2024-07-18):
    Training: ~$0.50 - $1.00.
    Inference: same as the base model.

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import os
import sys
import time

import openai


BASE_MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX = "bibreformat"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(here, "train.jsonl")
    if not os.path.isfile(train_path):
        sys.exit("train.jsonl not found. Run assemble_dataset.py first.")

    client = openai.OpenAI()

    print(f"Uploading {train_path}...")
    with open(train_path, "rb") as f:
        upload = client.files.create(file=f, purpose="fine-tune")
    print(f"  file_id = {upload.id}")

    print(f"Submitting fine-tune job (base={BASE_MODEL}, suffix={SUFFIX})...")
    job = client.fine_tuning.jobs.create(
        training_file=upload.id,
        model=BASE_MODEL,
        suffix=SUFFIX,
    )
    print(f"  job_id  = {job.id}")
    print(f"  status  = {job.status}")

    print("Waiting for fine-tune to finish (polling every 30s)...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"  [{time.strftime('%H:%M:%S')}] status = {job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)

    if job.status != "succeeded":
        sys.exit(f"Fine-tune did not succeed: {job.status}")

    print("\n=== SUCCESS ===")
    print(f"Fine-tuned model id: {job.fine_tuned_model}")
    print("Add this id to providers.json (see fine_tune_demos/README.md).")


if __name__ == "__main__":
    main()
