"""
evaluate.py
Side-by-side comparison: base model vs fine-tuned model on the held-out set.

Voice transfer is harder to score deterministically than bibliography
reformatting. We report three numbers:

  - Surface diversity     : type-token ratio of the prediction (higher = richer vocabulary).
  - Length ratio          : len(pred) / len(gold). Far from 1.0 suggests structural mismatch.
  - Hedge density         : fraction of hedge markers ('appears', 'not clear', 'would benefit', etc.)
                            present in the prediction. NIH voice is hedge-heavy.

These are not gold-standard metrics; they are *operational indicators*. For
real evaluation, have an experienced reviewer rate predictions blind on a
rubric (clarity, fidelity to NIH conventions, technical accuracy).

Usage:
    python evaluate.py ft:gpt-4o-mini-2024-07-18:your-org:nihvoice:abc123

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import argparse
import json
import os
import re
import sys

import openai


BASE_MODEL = "gpt-4o-mini-2024-07-18"

HEDGE_MARKERS = [
    "appears", "would benefit", "is not clear", "may be", "raises concerns",
    "modest concern", "underspecified", "would strengthen", "notable weakness",
    "the pi should", "the application would", "is not adequately", "remains",
    "anticipates", "suggests", "consistent with", "consider", "warrants",
]


def load_eval(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append((
                obj["messages"][0]["content"],
                obj["messages"][1]["content"],
                obj["messages"][2]["content"],
            ))
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


def type_token_ratio(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hedge_density(text):
    lower = text.lower()
    hits = sum(1 for m in HEDGE_MARKERS if m in lower)
    return hits / max(1, len(HEDGE_MARKERS))


def score(client, model, items):
    rows = []
    ttrs, length_ratios, hedges = [], [], []
    for sys_msg, user, gold in items:
        pred = predict(client, model, sys_msg, user)
        ttr = type_token_ratio(pred)
        lr = len(pred) / max(1, len(gold))
        hd = hedge_density(pred)
        ttrs.append(ttr); length_ratios.append(lr); hedges.append(hd)
        rows.append((user, gold, pred, ttr, lr, hd))
    return {
        "mean_ttr": sum(ttrs) / len(ttrs) if ttrs else 0.0,
        "mean_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else 0.0,
        "mean_hedge_density": sum(hedges) / len(hedges) if hedges else 0.0,
        "rows": rows,
    }


def print_report(label, result):
    print(f"\n=== {label} ===")
    print(f"Mean type-token ratio : {result['mean_ttr']:.3f}")
    print(f"Mean length ratio     : {result['mean_length_ratio']:.2f}  (1.00 = same length as gold)")
    print(f"Mean hedge density    : {result['mean_hedge_density']:.3f}")
    print("--- per-example ---")
    for i, (user, gold, pred, ttr, lr, hd) in enumerate(result["rows"]):
        print(f"[{i+1}] ttr={ttr:.2f} len_ratio={lr:.2f} hedge={hd:.2f}")
        print(f"    in:   {user}")
        print(f"    gold: {gold[:200]}{'...' if len(gold) > 200 else ''}")
        print(f"    pred: {pred[:200]}{'...' if len(pred) > 200 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ft_model", nargs="?", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(here, "eval.jsonl")
    if not os.path.isfile(eval_path):
        sys.exit("eval.jsonl not found. Run assemble_dataset.py first.")

    items = load_eval(eval_path)
    print(f"Loaded {len(items)} eval examples from {eval_path}")

    client = openai.OpenAI()
    base = score(client, BASE_MODEL, items)
    print_report(f"Base model: {BASE_MODEL}", base)

    if args.ft_model:
        ft = score(client, args.ft_model, items)
        print_report(f"Fine-tuned: {args.ft_model}", ft)
        print("\n=== HEAD-TO-HEAD ===")
        print(f"hedge density   base={base['mean_hedge_density']:.3f}  ft={ft['mean_hedge_density']:.3f}")
        print(f"length ratio    base={base['mean_length_ratio']:.2f}   ft={ft['mean_length_ratio']:.2f}")
        print(f"TTR             base={base['mean_ttr']:.3f}            ft={ft['mean_ttr']:.3f}")


if __name__ == "__main__":
    main()
