"""Forensic inspection of specific contamination-audit matches. For each flagged
item, pull the exact matching training row and surrounding context so we can
judge whether the match is benign (common idiom) or concerning (derivative
contamination)."""

import json
import re
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent

# These are the HumanEval items that my deep audit flagged as matching in
# r1_clean / r1_clean_v2. Pull the match from each file and show the
# surrounding training context so we can judge benign vs concerning.
HE_FLAGGED = {
    "HumanEval/4":   "mean = sum(numbers) / len(numbers)",           # MAD impl
    "HumanEval/13":  "def greatest_common_divisor(a:",                # GCD sig
    "HumanEval/18":  "if string[i:i+len(substring)] == substring:",   # substring match
    "HumanEval/55":  "return fib(n - 1) + fib(n - 2)",                # fibonacci
    "HumanEval/90":  "You are given a list of integers",               # docstring
    "HumanEval/94":  "You are given a list of integers",
    "HumanEval/97":  "Assume the input is always valid",
    "HumanEval/105": "sorted_arr = sorted(arr, reverse=True)",
    "HumanEval/133": "You are given a list of integers",
    "HumanEval/147": "You are given a positive integer n",
    "HumanEval/155": "return (even_count, odd_count)",
    "HumanEval/158": "Write a function that accepts a list of strings",
}


def find_rows_with_pattern(path: Path, pattern: str, max_rows: int = 3):
    hits = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            full = "\n".join(m.get("content", "") for m in row.get("messages", []))
            if pattern in full:
                hits.append((i, row, full))
                if len(hits) >= max_rows:
                    break
    return hits


def main():
    print("=" * 80)
    print("HUMANEVAL MATCH FORENSICS")
    print("=" * 80)

    # Load the canonical HumanEval items once for cross-reference
    he_ds = load_dataset("openai/openai_humaneval", split="test")
    he_by_id = {r["task_id"]: r for r in he_ds}

    for tid, pat in HE_FLAGGED.items():
        canon = he_by_id.get(tid, {})
        print(f"\n--- {tid} matched on: {pat!r} ---")
        print(f"HumanEval canonical prompt excerpt: {canon.get('prompt', '')[:200]!r}")

        # Find a sample training row
        path = REPO / "training" / "data" / "r1_clean.jsonl"
        rows = find_rows_with_pattern(path, pat, max_rows=2)
        if not rows:
            print("  (no training row found — may only be in an RL source)")
            continue
        for i, row, full in rows:
            idx = full.find(pat)
            start = max(0, idx - 200)
            end = min(len(full), idx + len(pat) + 200)
            print(f"  Training row {i}:")
            print(f"  ...{full[start:end]}...")
            print()


if __name__ == "__main__":
    main()
