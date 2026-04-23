"""
Download and filter candidate datasets for deep contamination audit.
Each candidate is written as a JSONL of {"messages": [...]} rows matching the
format our existing audit expects.

Candidates (2026-04-23):
  - KodCode/KodCode-V1 : filter to benchmark_similarity < 0.5 + HE func-name blocklist
  - TIGER-Lab/AceCode-87K : raw + our own HE deny-set filter
  - google-research-datasets/mbpp 'full' (not sanitized) : for RSFT prompt expansion

Writes:
  training/data/kodcode_candidate.jsonl
  training/data/acecode_candidate.jsonl
  training/data/mbpp_full_candidate.jsonl (just prompts for RSFT, assistant blank)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

from training.data_sources import build_deny_ngrams, is_contaminated

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("prep_candidates")

REPO = Path(__file__).resolve().parent.parent


# HumanEval function names to blocklist. If a training row contains any of these
# as a def'd function name, we drop it — catches "Evol-Instruct paraphrase" leaks
# that share function names with HumanEval canonicals.
def _load_humaneval_func_names() -> set[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    names = set()
    for r in ds:
        for m in re.finditer(r"^[ \t]*def\s+(\w+)\s*\(", r["prompt"], re.MULTILINE):
            names.add(m.group(1))
    log.info("Loaded %d HumanEval function names", len(names))
    return names


def _row_has_he_funcname(text: str, names: set[str]) -> bool:
    """Return True if any HE function name appears as a def'd function in text."""
    for m in re.finditer(r"^[ \t]*def\s+(\w+)\s*\(", text, re.MULTILINE):
        if m.group(1) in names:
            return True
    return False


# ----------------------------------------------------------------------------
# KodCode
# ----------------------------------------------------------------------------


def build_kodcode(
    max_rows: int = 30000,
    similarity_threshold: float = 0.5,
    deny: set | None = None,
    he_names: set[str] | None = None,
):
    """Stream KodCode-V1, filter, write up to max_rows to JSONL."""
    out_path = REPO / "training" / "data" / "kodcode_candidate.jsonl"
    log.info("Streaming KodCode/KodCode-V1 ...")
    # KodCode has multiple subsets; pick the main solutions subset. Shard to limit pull.
    try:
        ds = load_dataset(
            "KodCode/KodCode-V1",
            split="train",
            streaming=True,
        )
    except Exception as e:
        log.error("Couldn't load KodCode: %s", e)
        return

    if deny is None:
        deny = build_deny_ngrams()
    if he_names is None:
        he_names = _load_humaneval_func_names()

    kept = 0
    seen = 0
    dropped_sim = 0
    dropped_funcname = 0
    dropped_contam = 0
    dropped_missing = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            if seen % 5000 == 0:
                log.info(
                    "  KodCode seen=%d kept=%d dropped_sim=%d funcname=%d contam=%d missing=%d",
                    seen, kept, dropped_sim, dropped_funcname, dropped_contam, dropped_missing,
                )
            if kept >= max_rows:
                break
            # Check similarity if available
            sim_fields = ["benchmark_similarity", "max_similarity", "similarity"]
            sim = None
            for k in sim_fields:
                if k in row and row[k] is not None:
                    try:
                        sim = float(row[k])
                        break
                    except (TypeError, ValueError):
                        pass
            if sim is not None and sim >= similarity_threshold:
                dropped_sim += 1
                continue

            # Get question + solution
            question = row.get("question") or row.get("prompt") or row.get("problem")
            solution = row.get("solution") or row.get("response") or row.get("code")
            if not question or not solution:
                dropped_missing += 1
                continue

            combined = f"{question}\n{solution}"
            if _row_has_he_funcname(combined, he_names):
                dropped_funcname += 1
                continue
            if is_contaminated(question, deny) or is_contaminated(solution, deny):
                dropped_contam += 1
                continue

            row_out = {
                "messages": [
                    {"role": "user", "content": str(question)},
                    {"role": "assistant", "content": str(solution)},
                ]
            }
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            kept += 1

    log.info(
        "KodCode: kept=%d / seen=%d  (dropped_sim=%d funcname=%d contam=%d missing=%d)",
        kept, seen, dropped_sim, dropped_funcname, dropped_contam, dropped_missing,
    )
    return out_path


# ----------------------------------------------------------------------------
# AceCode-87K
# ----------------------------------------------------------------------------


def build_acecode(
    max_rows: int = 30000,
    deny: set | None = None,
    he_names: set[str] | None = None,
):
    out_path = REPO / "training" / "data" / "acecode_candidate.jsonl"
    log.info("Streaming TIGER-Lab/AceCode-87K ...")
    try:
        ds = load_dataset("TIGER-Lab/AceCode-87K", split="train", streaming=True)
    except Exception as e:
        log.error("Couldn't load AceCode: %s", e)
        return

    if deny is None:
        deny = build_deny_ngrams()
    if he_names is None:
        he_names = _load_humaneval_func_names()

    kept = 0
    seen = 0
    dropped_funcname = 0
    dropped_contam = 0
    dropped_missing = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            seen += 1
            if seen % 5000 == 0:
                log.info(
                    "  AceCode seen=%d kept=%d funcname=%d contam=%d missing=%d",
                    seen, kept, dropped_funcname, dropped_contam, dropped_missing,
                )
            if kept >= max_rows:
                break
            # AceCode fields: id, source, question, test_cases, inferences (list of
            # {completion, ...}), context_messages. We use the first completion
            # as the assistant response for SFT.
            question = row.get("question")
            inferences = row.get("inferences") or []
            solution = None
            if isinstance(inferences, list) and inferences:
                first = inferences[0] or {}
                if isinstance(first, dict):
                    solution = first.get("completion")
            if not solution:
                solution = row.get("inferred_code") or row.get("response") or row.get("code")
            if not question or not solution:
                dropped_missing += 1
                continue

            combined = f"{question}\n{solution}"
            if _row_has_he_funcname(combined, he_names):
                dropped_funcname += 1
                continue
            if is_contaminated(question, deny) or is_contaminated(solution, deny):
                dropped_contam += 1
                continue

            row_out = {
                "messages": [
                    {"role": "user", "content": str(question)},
                    {"role": "assistant", "content": str(solution)},
                ]
            }
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            kept += 1

    log.info(
        "AceCode: kept=%d / seen=%d  (funcname=%d contam=%d missing=%d)",
        kept, seen, dropped_funcname, dropped_contam, dropped_missing,
    )
    return out_path


# ----------------------------------------------------------------------------
# MBPP 'full' (for RSFT prompt expansion)
# ----------------------------------------------------------------------------


def build_mbpp_full_prompts(
    deny: set | None = None,
):
    """Extract MBPP full split prompts (minus sanitized-train overlap, decontam'd vs HumanEval).
    Used for rejection-sampling SFT expansion — we sample rollouts from R13 on these prompts,
    not train on the canonical solutions."""
    out_path = REPO / "training" / "data" / "mbpp_full_prompts.jsonl"
    log.info("Loading MBPP full ...")

    if deny is None:
        deny = build_deny_ngrams()

    full = load_dataset("google-research-datasets/mbpp", "full", split="train")
    sanitized = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    sanitized_ids = {r.get("task_id") for r in sanitized}
    log.info("MBPP full=%d, sanitized-train=%d already-known-clean", len(full), len(sanitized))

    kept = 0
    dropped_contam = 0
    dropped_overlap_sanitized = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in full:
            text = r.get("text") or r.get("prompt")
            code = r.get("code")
            tests = r.get("test_list") or []
            if not text or not tests:
                continue
            # Skip rows already in sanitized-train (we have those)
            if r.get("task_id") in sanitized_ids:
                dropped_overlap_sanitized += 1
                continue
            if is_contaminated(text, deny) or is_contaminated(code or "", deny):
                dropped_contam += 1
                continue
            # For each test, verify too
            if any(is_contaminated(t, deny) for t in tests):
                dropped_contam += 1
                continue
            row_out = {
                "text": text,
                "code": code or "",
                "tests": list(tests),
            }
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            kept += 1

    log.info(
        "MBPP-full-prompts: kept=%d (dropped_contam=%d overlap_sanitized=%d)",
        kept, dropped_contam, dropped_overlap_sanitized,
    )
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["kodcode", "acecode", "mbpp_full", "all"], default="all")
    p.add_argument("--kodcode_max", type=int, default=30000)
    p.add_argument("--acecode_max", type=int, default=30000)
    p.add_argument("--sim_threshold", type=float, default=0.5)
    args = p.parse_args()

    log.info("Building HE deny-set and function-name list ...")
    deny = build_deny_ngrams()
    he_names = _load_humaneval_func_names()

    if args.which in ("kodcode", "all"):
        build_kodcode(args.kodcode_max, args.sim_threshold, deny, he_names)
    if args.which in ("acecode", "all"):
        build_acecode(args.acecode_max, deny, he_names)
    if args.which in ("mbpp_full", "all"):
        build_mbpp_full_prompts(deny)


if __name__ == "__main__":
    main()
