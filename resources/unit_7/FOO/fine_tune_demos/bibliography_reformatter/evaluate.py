"""
evaluate.py
Side-by-side comparison: base model vs fine-tuned model on the held-out set.

Metric: exact-match accuracy after whitespace normalization. Bibliography
formatting is deterministic enough that exact match is a reasonable bar; a
softer metric (BLEU / Levenshtein ratio) is reported alongside for nuance.

Usage:
    python evaluate.py ft:gpt-4o-mini-2024-07-18:your-org:bibreformat:abc123

If no model id is passed, only the base model is scored (lets you see the
'before' picture without paying for a fine-tune).

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

import openai


BASE_MODEL = "gpt-4o-mini-2024-07-18"


def normalize(s):
    """Collapse whitespace; lowercase; strip surrounding quotes/punct."""
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.strip('"\'`')
    return s


def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def load_eval(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sys_msg = obj["messages"][0]["content"]
            user_msg = obj["messages"][1]["content"]
            gold = obj["messages"][2]["content"]
            items.append((sys_msg, user_msg, gold))
    return items


def predict(client, model, system, user):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def score(client, model, items):
    exact = 0
    sims = []
    rows = []
    for sys_msg, user_msg, gold in items:
        pred = predict(client, model, sys_msg, user_msg)
        is_exact = normalize(pred) == normalize(gold)
        sim = similarity(pred, gold)
        if is_exact:
            exact += 1
        sims.append(sim)
        rows.append((user_msg, gold, pred, is_exact, sim))
    return {
        "exact_acc": exact / len(items) if items else 0.0,
        "mean_similarity": sum(sims) / len(sims) if sims else 0.0,
        "rows": rows,
    }


def print_report(label, result):
    print(f"\n=== {label} ===")
    print(f"Exact-match accuracy : {result['exact_acc']:.2%}")
    print(f"Mean similarity      : {result['mean_similarity']:.3f}")
    print("--- per-example ---")
    for i, (user, gold, pred, ok, sim) in enumerate(result["rows"]):
        marker = "OK" if ok else "..."
        print(f"[{i+1}] {marker} sim={sim:.2f}")
        print(f"    in:    {user[:100]}{'...' if len(user) > 100 else ''}")
        print(f"    gold:  {gold}")
        print(f"    pred:  {pred}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ft_model", nargs="?", default=None,
                        help="Fine-tuned model id (ft:...). If omitted, only base is scored.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(here, "eval.jsonl")
    if not os.path.isfile(eval_path):
        sys.exit("eval.jsonl not found. Run assemble_dataset.py first.")

    items = load_eval(eval_path)
    print(f"Loaded {len(items)} eval examples from {eval_path}")

    client = openai.OpenAI()

    base_result = score(client, BASE_MODEL, items)
    print_report(f"Base model: {BASE_MODEL}", base_result)

    if args.ft_model:
        ft_result = score(client, args.ft_model, items)
        print_report(f"Fine-tuned model: {args.ft_model}", ft_result)

        print("\n=== HEAD-TO-HEAD ===")
        print(f"Base exact-match     : {base_result['exact_acc']:.2%}")
        print(f"Fine-tune exact-match: {ft_result['exact_acc']:.2%}")
        print(f"Improvement          : {(ft_result['exact_acc'] - base_result['exact_acc'])*100:+.1f} pp")


if __name__ == "__main__":
    main()
