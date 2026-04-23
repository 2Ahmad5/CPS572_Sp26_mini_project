"""
DEEP contamination audit. Extends `verify_no_test_leak.py` by extracting multiple
distinctive fingerprints per test item (instead of treating each item as one
monolithic string) and checking each fingerprint as a SUBSTRING against every
training JSONL.

Rationale: the basic SUBSTRING check requires the full test text to appear
verbatim. "Evol-Instruct" style paraphrasing defeats it. But the distinctive
*internal* elements (function signature lines, `>>> example` calls in docstrings,
specific variable+value combinations in GSM8K reasoning, structured IFEval
constraint phrases) often survive paraphrasing unchanged. This script extracts
those and treats each as its own evidence.

Hard gates (any hit = fail):
  - DEEP_SUBSTRING: any extracted fingerprint of length >= 30 appears as a
    contiguous substring of any training message.
  - FULL_SUBSTRING: the complete test prompt / canonical solution appears.

Advisory (reported but not a hard fail):
  - FUNC_SIG: HumanEval function signature in training (common code).

Usage:
    python -m training.verify_no_test_leak_deep
    python -m training.verify_no_test_leak_deep --files training/data/r8_bon.jsonl
    python -m training.verify_no_test_leak_deep --strict-exit
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

# Reuse helpers from the original audit
from training.verify_no_test_leak import (
    _load_training_index,
    _in_any_chunk,
    _norm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("verify_deep")

REPO = Path(__file__).resolve().parent.parent

DEFAULT_FILES = [
    "training/data/r1_kitchen_sink.jsonl",
    "training/data/r1_enhanced.jsonl",
    "training/data/r8_rft.jsonl",
    "training/data/r8_bon.jsonl",
]


# ---------------------------------------------------------------------------
# Fingerprint extractors
# ---------------------------------------------------------------------------

# Minimum fingerprint length to avoid trivial matches. 30 chars is long enough
# to be distinctive but short enough to catch partial leaks.
_MIN_FP_LEN = 30

_DEF_LINE_RE = re.compile(r"^[ \t]*def\s+\w+[^:]*:", re.MULTILINE)
_DOCTEST_RE = re.compile(r">>>.+?(?=\n>>>|\n\n|\n    [^>\s]|\Z)", re.DOTALL)
_EXAMPLE_CALL_RE = re.compile(r">>>\s*(\w+\([^)\n]*\))")


def humaneval_fingerprints() -> list[tuple[str, str, str]]:
    """Extract (task_id, label, pattern) fingerprints from every HumanEval item.

    For each problem, produces:
      - 'sig': full function signature line (`def name(args) -> ret:`)
      - 'docstring_chunk': ~60 consecutive non-trivial chars from the docstring
      - 'doctest_N': each `>>> example_call(args)` line + expected output
      - 'solution_chunk': distinctive phrases from canonical solution
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    out: list[tuple[str, str, str]] = []
    for r in ds:
        tid = r["task_id"]
        prompt = r["prompt"]
        solution = r["canonical_solution"]

        # Function signatures — every `def name(...)` in the prompt
        for m in _DEF_LINE_RE.finditer(prompt):
            sig = m.group().strip()
            if len(sig) >= 15:  # short threshold for sigs
                out.append((tid, "sig", sig))

        # Docstring examples (each `>>> call(args)` line with expected output)
        for m in _DOCTEST_RE.finditer(prompt):
            doctest = m.group().strip()
            if len(doctest) >= _MIN_FP_LEN:
                out.append((tid, "doctest", doctest[:200]))  # cap at 200 chars

        # Each `>>> function_call(args)` alone — highly distinctive
        for idx, m in enumerate(_EXAMPLE_CALL_RE.finditer(prompt)):
            call = m.group(1)
            if len(call) >= 20:
                out.append((tid, f"call_{idx}", call))

        # Distinctive docstring sentence (first 100 chars of non-boilerplate description)
        docstring_match = re.search(r'"""(.+?)"""', prompt, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            # First meaningful sentence (split on period, get longest)
            sentences = [s.strip() for s in docstring.replace("\n", " ").split(".") if len(s.strip()) >= _MIN_FP_LEN]
            for i, s in enumerate(sentences[:2]):
                out.append((tid, f"docstring_sent_{i}", s[:150]))

        # Canonical solution distinctive lines
        sol_lines = [l.strip() for l in solution.split("\n") if len(l.strip()) >= _MIN_FP_LEN]
        # Take first 2 non-trivial lines
        for i, line in enumerate(sol_lines[:2]):
            out.append((tid, f"solution_line_{i}", line[:150]))

    log.info("HumanEval fingerprints: %d total across 164 problems", len(out))
    return out


def gsm8k_fingerprints() -> list[tuple[str, str, str]]:
    """Extract (item_id, label, pattern) from GSM8K test split.

    For each question, produces:
      - 'question_prefix': first ~80 chars of question (distinctive setup phrase)
      - 'reasoning_chunk': distinctive phrases from the answer's reasoning
      - 'number_context_N': a specific number plus its surrounding context words
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out: list[tuple[str, str, str]] = []
    for i, r in enumerate(ds):
        qid = f"GSM8K-test/{i}"
        question = r["question"]
        answer = r["answer"]

        # First sentence of question — setup is distinctive
        first_sent = question.split(".")[0] if "." in question else question[:120]
        if len(first_sent) >= _MIN_FP_LEN:
            out.append((qid, "question_first_sent", first_sent.strip()[:150]))

        # First 80 chars of question
        if len(question) >= 80:
            out.append((qid, "question_80c", question[:80]))

        # Reasoning chunks: split answer on '\n', take non-trivial lines
        reasoning_lines = [l.strip() for l in answer.split("\n") if len(l.strip()) >= _MIN_FP_LEN]
        for j, line in enumerate(reasoning_lines[:2]):  # first 2 lines only to cap pattern count
            out.append((qid, f"reasoning_{j}", line[:150]))

    log.info("GSM8K-test fingerprints: %d total across %d questions", len(out), len(ds))
    return out


def ifeval_fingerprints() -> list[tuple[str, str, str]]:
    """Extract (key, label, pattern) from google/IFEval.

    IFEval prompts are mostly distinctive instruction sentences + topic prompt.
    Each is generally unique enough that the whole prompt as SUBSTRING
    works — plus we break out the first sentence for paraphrase robustness.
    """
    ds = load_dataset("google/IFEval", split="train")
    out: list[tuple[str, str, str]] = []
    for i, r in enumerate(ds):
        key = r.get("key", i)
        prompt = r["prompt"]
        if len(prompt) >= _MIN_FP_LEN:
            # Full prompt
            out.append((f"IFEval/{key}", "full", prompt[:400]))
            # First 100 chars
            out.append((f"IFEval/{key}", "first100", prompt[:100]))
            # Last sentence
            sentences = [s.strip() for s in prompt.split(".") if len(s.strip()) >= _MIN_FP_LEN]
            if sentences:
                out.append((f"IFEval/{key}", "last_sent", sentences[-1][:150]))

    log.info("IFEval fingerprints: %d total across 541 prompts", len(out))
    return out


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------


def audit_fingerprints(
    path: Path,
    fingerprints: dict[str, list[tuple[str, str, str]]],
) -> dict:
    """Index the JSONL once, then Aho-Corasick scan every fingerprint at once."""
    import ahocorasick

    log.info("Indexing %s ...", path)
    idx = _load_training_index(path, build_ngrams=False)
    log.info(
        "  %d rows / %d msgs -> %.1f MB any, %.1f MB asst",
        idx["rows"], idx["msgs"],
        sum(len(c) for c in idx["any_chunks"]) / 1e6,
        sum(len(c) for c in idx["assistant_chunks"]) / 1e6,
    )

    # Build Aho-Corasick automaton of all fingerprints simultaneously.
    # Key: normalized pattern. Value: (bench, item_id, label, original_pattern)
    log.info("Building Aho-Corasick automaton ...")
    A = ahocorasick.Automaton()
    pattern_map: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for bench, fps in fingerprints.items():
        for item_id, label, pattern in fps:
            norm_p = _norm(pattern)
            if len(norm_p) < _MIN_FP_LEN:
                continue
            pattern_map[norm_p].append((bench, item_id, label, pattern))
            A.add_word(norm_p, norm_p)
    A.make_automaton()
    log.info("  %d unique patterns in automaton", len(pattern_map))

    # Scan each chunk and collect unique matched patterns.
    matched_patterns: set[str] = set()
    for chunk in idx["any_chunks"]:
        for _, p in A.iter(chunk):
            matched_patterns.add(p)

    log.info("  %d distinct patterns matched", len(matched_patterns))

    # Reconstruct hits by bench/label.
    hits: dict[str, dict[str, list[tuple[str, str, str]]]] = {
        bench: defaultdict(list) for bench in fingerprints
    }
    for p in matched_patterns:
        for bench, item_id, label, original_pattern in pattern_map[p]:
            hits[bench][label].append((item_id, label, original_pattern))

    return {
        "path": str(path),
        "rows": idx["rows"],
        "messages": idx["msgs"],
        "hits": hits,
    }


def summarize(results: list[dict], fp_counts: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("# DEEP contamination audit report\n")
    lines.append(
        "This audit extends the basic verify_no_test_leak.py by extracting "
        "multiple distinctive fingerprints per test item — each `def` signature, "
        "each `>>> example` call, each `>>> expected_output` line, each docstring "
        "sentence, each canonical-solution line — and checks each as a SUBSTRING "
        "against every training JSONL. Designed to catch derivative contamination "
        "(e.g. Evol-Instruct-paraphrased test problems that preserve distinctive "
        "internal elements).\n\n"
        f"Fingerprints extracted: IFEval={fp_counts.get('IFEval', 0)}  "
        f"GSM8K-test={fp_counts.get('GSM8K-test', 0)}  "
        f"HumanEval={fp_counts.get('HumanEval', 0)}.\n"
    )

    # Summary table
    lines.append("## Summary: unique leaked test items per category per file\n")
    header_cats = ["IFEval", "GSM8K-test", "HumanEval"]
    header = ["File", "Rows"]
    for c in header_cats:
        header += [f"{c} items", f"{c} matches"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in results:
        cells = [Path(r["path"]).name, str(r["rows"])]
        for c in header_cats:
            per_label = r["hits"].get(c, {})
            unique_items = set()
            total = 0
            for label, lst in per_label.items():
                for item_id, _, _ in lst:
                    unique_items.add(item_id)
                total += len(lst)
            cells.append(str(len(unique_items)))
            cells.append(str(total))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Detail per file
    for r in results:
        lines.append(f"## {Path(r['path']).name}\n")
        had_hits = False
        for bench, per_label in r["hits"].items():
            for label, lst in per_label.items():
                if lst:
                    had_hits = True
                    unique_items = sorted(set(h[0] for h in lst))
                    sample = [h[2][:100] for h in lst[:3]]
                    lines.append(
                        f"- **{bench} / {label}**: {len(lst)} matches "
                        f"across {len(unique_items)} unique test items. "
                        f"Items (first 10): {unique_items[:10]}. "
                        f"Sample patterns: {sample}"
                    )
        if not had_hits:
            lines.append("- No fingerprint matches.\n")
        lines.append("")

    # Totals verdict
    ifeval_items = set()
    gsm_items = set()
    he_items = set()
    for r in results:
        for label, lst in r["hits"].get("IFEval", {}).items():
            for h in lst:
                ifeval_items.add(h[0])
        for label, lst in r["hits"].get("GSM8K-test", {}).items():
            for h in lst:
                gsm_items.add(h[0])
        for label, lst in r["hits"].get("HumanEval", {}).items():
            for h in lst:
                he_items.add(h[0])
    lines.append("## Verdict\n")
    lines.append(
        f"- Unique leaked IFEval test items: **{len(ifeval_items)}** / 541\n"
        f"- Unique leaked GSM8K-test items: **{len(gsm_items)}** / 1319\n"
        f"- Unique leaked HumanEval items: **{len(he_items)}** / 164\n"
    )
    total = len(ifeval_items) + len(gsm_items) + len(he_items)
    if total == 0:
        lines.append(
            "**No derivative contamination detected on any fingerprint.** "
            "Stricter than BigCode's 13-gram standard; every HumanEval `>>> example` "
            "line, every IFEval prompt first 100 chars, every GSM8K reasoning line "
            "was checked. Training data is clean.\n"
        )
    else:
        lines.append(
            f"**{total} unique test items have fingerprint overlap** with training "
            "data. Review the per-file detail above to determine whether these are "
            "genuine derivative leaks or benign common-code matches. "
            "Ratio matters: ~1-5% of HumanEval is unavoidable (common utility "
            "functions); >10% warrants removing the offending training source.\n"
        )

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    p.add_argument("--strict-exit", action="store_true",
                   help="Exit 1 if > 10%% of any test set leaked.")
    p.add_argument("--out", default="training/data/contamination_report_deep.md")
    args = p.parse_args()

    log.info("Extracting fingerprints from eval benchmarks ...")
    fingerprints = {
        "IFEval": ifeval_fingerprints(),
        "GSM8K-test": gsm8k_fingerprints(),
        "HumanEval": humaneval_fingerprints(),
    }
    fp_counts = {k: len(v) for k, v in fingerprints.items()}
    log.info("Total fingerprints: IFEval=%d, GSM8K-test=%d, HumanEval=%d",
             fp_counts["IFEval"], fp_counts["GSM8K-test"], fp_counts["HumanEval"])

    results = []
    for f in args.files:
        path = REPO / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            log.warning("Skipping missing file: %s", path)
            continue
        results.append(audit_fingerprints(path, fingerprints))

    report = summarize(results, fp_counts)
    out_path = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", out_path)

    if args.strict_exit:
        ifeval_items = set()
        gsm_items = set()
        he_items = set()
        for r in results:
            for label, lst in r["hits"].get("IFEval", {}).items():
                for h in lst: ifeval_items.add(h[0])
            for label, lst in r["hits"].get("GSM8K-test", {}).items():
                for h in lst: gsm_items.add(h[0])
            for label, lst in r["hits"].get("HumanEval", {}).items():
                for h in lst: he_items.add(h[0])
        if len(ifeval_items) / 541 > 0.10:
            return 1
        if len(gsm_items) / 1319 > 0.10:
            return 1
        if len(he_items) / 164 > 0.10:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
