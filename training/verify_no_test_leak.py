"""
Independent contamination audit of every materialized training JSONL against the
three held-out eval benchmarks (IFEval, GSM8K test, HumanEval).

This is deliberately stricter than training/data_sources.py's 13-gram filter and
is run *after* training data has already been materialized — it's the paper-trail
equivalent of the other team's "we ran a script, got 0 overlaps" statement.

For every (training file × test set) pair, three checks are run *independently*:

  1. EXACT: normalized-whitespace, lowercased equality between test prompt and
     any user-or-assistant message in training data.
  2. SUBSTRING: test prompt appears (normalized) as a contiguous substring of a
     training message. This is the tightest practical contamination indicator —
     if a test question appears verbatim, even surrounded by other text, we flag.
  3. NGRAM: 8-gram word overlap (stricter than our training-time 13-gram filter).
     Any shared 8-gram between test-set text and training text is flagged. Much
     noisier — the headline number should be EXACT + SUBSTRING; NGRAM is a
     belt-and-suspenders check for near-duplicates.

For HumanEval we additionally check:
  4. FUNC_SIG: the `def <name>(` line of each canonical HumanEval problem must
     not appear in training assistant messages. Captures "Magicoder-style" leaks
     where someone baked the reference solution into an instruction/response.

Usage:
    python -m training.verify_no_test_leak
    python -m training.verify_no_test_leak --files training/data/r8_bon.jsonl
    python -m training.verify_no_test_leak --strict-exit   # exit(1) if any leak

Report is written to `training/data/contamination_report.md`.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("verify_no_test_leak")


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "training" / "data" / "contamination_report.md"


# JSONLs we've actually trained on (or published a checkpoint from).
DEFAULT_FILES = [
    "training/data/r1_kitchen_sink.jsonl",
    "training/data/r1_enhanced.jsonl",
    "training/data/r8_rft.jsonl",
    "training/data/r8_bon.jsonl",
]


_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace. Tight enough for exact/substring comparison,
    loose enough to survive trivial reformatting."""
    return _WS_RE.sub(" ", s.lower()).strip()


def _words(s: str) -> list[str]:
    return s.lower().split()


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


# ----------------------------------------------------------------------------
# Test-set loaders. Each returns a list of (label, text) — label identifies the
# specific test item (e.g. "HumanEval/0") for reporting.
# ----------------------------------------------------------------------------


def load_ifeval_texts() -> list[tuple[str, str]]:
    ds = load_dataset("google/IFEval", split="train")
    return [(f"IFEval/{r.get('key', i)}", r["prompt"]) for i, r in enumerate(ds)]


def load_gsm8k_test_texts() -> list[tuple[str, str]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        out.append((f"GSM8K-test/q{i}", r["question"]))
        out.append((f"GSM8K-test/a{i}", r["answer"]))
    return out


def load_humaneval_texts() -> list[tuple[str, str]]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for r in ds:
        tid = r["task_id"]
        out.append((f"{tid}/prompt", r["prompt"]))
        out.append((f"{tid}/solution", r["canonical_solution"]))
    return out


# Use [ \t]* (not \s*) so leading whitespace doesn't consume newlines and
# displace m.start() onto a \n byte.
_DEF_RE = re.compile(r"^[ \t]*def\s+(\w+)\s*\(", re.MULTILINE)


def humaneval_func_signatures() -> list[tuple[str, str]]:
    """Return (task_id, canonical-signature-line) for each HumanEval problem.
    Signature line = 'def <name>(...)' extracted from the canonical prompt."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    sigs = []
    for r in ds:
        prompt = r["prompt"]
        m = _DEF_RE.search(prompt)
        if not m:
            continue
        # Walk forward past any remaining leading whitespace to find 'def'.
        start = m.start()
        while start < len(prompt) and prompt[start] in " \t":
            start += 1
        line_end = prompt.find("\n", start)
        sig = prompt[start:line_end] if line_end > start else prompt[start:]
        sig = sig.strip()
        # Skip any pathological empty/short sigs.
        if len(sig) < 10:
            continue
        sigs.append((r["task_id"], sig))
    return sigs


# ----------------------------------------------------------------------------
# Per-file scanner
# ----------------------------------------------------------------------------


# Chunked buffers bound memory. Each chunk is roughly this size, after which
# we start a new buffer. `\x00` separator ensures test substrings can't straddle
# two concatenated messages.
_CHUNK_BYTES = 40_000_000  # 40 MB per chunk


def _load_training_index(path: Path):
    """Walk a training JSONL once, build:
       - exact_set: set of normalized message strings (any role)
       - any_chunks: list of '\\x00'-joined normalized ANY-role chunks, each ~40MB
       - assistant_chunks: list of '\\x00'-joined normalized ASSISTANT chunks
       - ngram_set: union of all 8-grams across any-role messages
       - rows, msgs counts
    """
    exact_set: set[str] = set()
    any_chunks: list[str] = []
    assistant_chunks: list[str] = []
    ngram_set: set[tuple[str, ...]] = set()
    rows = 0
    msgs = 0

    any_buf: list[str] = []
    any_size = 0
    asst_buf: list[str] = []
    asst_size = 0
    sep = "\x00"

    def _flush_any():
        nonlocal any_buf, any_size
        if any_buf:
            any_chunks.append(sep.join(any_buf))
            any_buf = []
            any_size = 0

    def _flush_asst():
        nonlocal asst_buf, asst_size
        if asst_buf:
            assistant_chunks.append(sep.join(asst_buf))
            asst_buf = []
            asst_size = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            for m in row.get("messages", []):
                content = m.get("content", "")
                if not content:
                    continue
                role = m.get("role", "?")
                norm_c = _norm(content)
                exact_set.add(norm_c)
                any_buf.append(norm_c)
                any_size += len(norm_c) + 1
                if any_size >= _CHUNK_BYTES:
                    _flush_any()
                if role == "assistant":
                    asst_buf.append(norm_c)
                    asst_size += len(norm_c) + 1
                    if asst_size >= _CHUNK_BYTES:
                        _flush_asst()
                ngram_set.update(_ngrams(_words(content), 8))
                msgs += 1
    _flush_any()
    _flush_asst()

    return {
        "exact_set": exact_set,
        "any_chunks": any_chunks,
        "assistant_chunks": assistant_chunks,
        "ngram_set": ngram_set,
        "rows": rows,
        "msgs": msgs,
    }


def _in_any_chunk(pat: str, chunks: list[str]) -> bool:
    for c in chunks:
        if pat in c:
            return True
    return False


def audit_file(
    path: Path,
    test_texts: dict[str, list[tuple[str, str]]],
    he_sigs: list[tuple[str, str]],
    ngram_n: int = 8,
) -> dict:
    """Test-centric scan: one pass over training to build indices, then each
    test text checked against the indices. O(|training|) + O(|tests|) instead
    of O(|training| × |tests|)."""
    log.info("Indexing %s ...", path)
    idx = _load_training_index(path)
    any_bytes = sum(len(c) for c in idx["any_chunks"])
    asst_bytes = sum(len(c) for c in idx["assistant_chunks"])
    log.info(
        "  %d rows / %d msgs -> exact_set=%d, any=%.1f MB in %d chunks, asst=%.1f MB in %d chunks, 8-grams=%d",
        idx["rows"], idx["msgs"], len(idx["exact_set"]),
        any_bytes / 1e6, len(idx["any_chunks"]),
        asst_bytes / 1e6, len(idx["assistant_chunks"]),
        len(idx["ngram_set"]),
    )

    exact_hits = defaultdict(list)
    substring_hits = defaultdict(list)
    ngram_hits = defaultdict(list)
    he_sig_hits = []

    for bench, items in test_texts.items():
        for lbl, txt in items:
            norm_t = _norm(txt)
            if not norm_t:
                continue
            # EXACT
            if norm_t in idx["exact_set"]:
                exact_hits[bench].append((None, lbl))
            # SUBSTRING (test text >= 40 chars to avoid trivial matches)
            if len(norm_t) >= 40 and _in_any_chunk(norm_t, idx["any_chunks"]):
                substring_hits[bench].append((None, lbl))
            # NGRAM
            grams_t = _ngrams(_words(txt), ngram_n)
            if grams_t and (grams_t & idx["ngram_set"]):
                ngram_hits[bench].append((None, lbl))

    for tid, sig in he_sigs:
        if _in_any_chunk(sig, idx["assistant_chunks"]):
            he_sig_hits.append((None, tid, sig))

    return {
        "path": str(path),
        "rows": idx["rows"],
        "messages": idx["msgs"],
        "exact": {b: hits for b, hits in exact_hits.items()},
        "substring": {b: hits for b, hits in substring_hits.items()},
        "ngram": {b: hits for b, hits in ngram_hits.items()},
        "humaneval_func_sig": he_sig_hits,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def summarize(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Contamination verification report\n")
    lines.append(
        "Independent audit of materialized training JSONLs against IFEval (541), "
        "GSM8K test (1319), and HumanEval (164). Three checks - EXACT (normalized "
        "equality), SUBSTRING (normalized containment, test text >= 40 chars), and "
        "NGRAM (8-gram overlap, noisier near-duplicate detector) - plus a "
        "HumanEval function-signature check on assistant messages.\n"
    )

    # Headline table
    lines.append("## Summary\n")
    lines.append("| File | Rows | Msgs | IFEval exact | IFEval substring | IFEval 8-gram | GSM8K-test exact | GSM8K-test substring | GSM8K-test 8-gram | HumanEval exact | HumanEval substring | HumanEval 8-gram | HumanEval func-sig |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        row = [
            Path(r["path"]).name,
            str(r["rows"]),
            str(r["messages"]),
            str(len(r["exact"].get("IFEval", []))),
            str(len(r["substring"].get("IFEval", []))),
            str(len(r["ngram"].get("IFEval", []))),
            str(len(r["exact"].get("GSM8K-test", []))),
            str(len(r["substring"].get("GSM8K-test", []))),
            str(len(r["ngram"].get("GSM8K-test", []))),
            str(len(r["exact"].get("HumanEval", []))),
            str(len(r["substring"].get("HumanEval", []))),
            str(len(r["ngram"].get("HumanEval", []))),
            str(len(r["humaneval_func_sig"])),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-file detail
    for r in results:
        lines.append(f"## {Path(r['path']).name}\n")
        lines.append(f"Rows: {r['rows']}  |  messages scanned: {r['messages']}\n")
        for check in ("exact", "substring", "ngram"):
            for bench, hits in r[check].items():
                if hits:
                    sample = [h[1] for h in hits[:5]]
                    lines.append(
                        f"- **{check.upper()} / {bench}**: {len(hits)} leaked test items "
                        f"(e.g. {sample})"
                    )
        if r["humaneval_func_sig"]:
            sample = [(h[1], h[2]) for h in r["humaneval_func_sig"][:5]]
            lines.append(
                f"- **HUMANEVAL-FUNC-SIG**: {len(r['humaneval_func_sig'])} hits "
                f"(e.g. {sample})"
            )
        if not any(r[check] for check in ("exact", "substring", "ngram")) and not r["humaneval_func_sig"]:
            lines.append("- No leaks detected.\n")
        lines.append("")

    # Verdict
    exact_total = sum(
        len(hits) for r in results for b, hits in r["exact"].items()
    )
    substring_total = sum(
        len(hits) for r in results for b, hits in r["substring"].items()
    )
    func_sig_total = sum(len(r["humaneval_func_sig"]) for r in results)
    ngram_total = sum(
        len(hits) for r in results for b, hits in r["ngram"].items()
    )
    lines.append("## Verdict\n")
    lines.append(
        f"- EXACT hits across all files: **{exact_total}**\n"
        f"- SUBSTRING hits across all files: **{substring_total}**\n"
        f"- HUMANEVAL-FUNC-SIG hits: **{func_sig_total}**\n"
        f"- 8-GRAM overlap hits (noisy near-duplicates): {ngram_total}\n"
    )
    if exact_total == 0 and substring_total == 0 and func_sig_total == 0:
        lines.append(
            "**No verbatim test-set leakage detected** in any materialized training "
            "JSONL. 8-gram overlap may still be nonzero (common phrases like "
            "'the answer is', 'how many', function names), but no EXACT, SUBSTRING, "
            "or function-signature match was found.\n"
        )
    else:
        lines.append(
            "**⚠ LEAKS DETECTED**. See per-file detail above. These rows must be "
            "removed from the affected JSONL and any downstream checkpoints "
            "retrained before submission.\n"
        )

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                   help="Training JSONLs to audit.")
    p.add_argument("--ngram", type=int, default=8)
    p.add_argument("--strict-exit", action="store_true",
                   help="Exit non-zero if ANY exact/substring/func-sig hit found.")
    p.add_argument("--out", default=str(REPORT_PATH))
    args = p.parse_args()

    log.info("Loading test-set texts ...")
    test_texts = {
        "IFEval": load_ifeval_texts(),
        "GSM8K-test": load_gsm8k_test_texts(),
        "HumanEval": load_humaneval_texts(),
    }
    he_sigs = humaneval_func_signatures()
    log.info("  IFEval: %d texts", len(test_texts["IFEval"]))
    log.info("  GSM8K-test: %d texts (questions + answers)", len(test_texts["GSM8K-test"]))
    log.info("  HumanEval: %d texts (prompts + canonical solutions)", len(test_texts["HumanEval"]))
    log.info("  HumanEval func-sigs: %d signatures", len(he_sigs))

    results = []
    for f in args.files:
        path = REPO_ROOT / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            log.warning("Skipping missing file: %s", path)
            continue
        results.append(audit_file(path, test_texts, he_sigs, ngram_n=args.ngram))

    report = summarize(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    log.info("Report written to %s", args.out)

    print(report)

    if args.strict_exit:
        exact_total = sum(len(h) for r in results for b, h in r["exact"].items())
        substring_total = sum(len(h) for r in results for b, h in r["substring"].items())
        func_sig_total = sum(len(r["humaneval_func_sig"]) for r in results)
        if exact_total + substring_total + func_sig_total > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
