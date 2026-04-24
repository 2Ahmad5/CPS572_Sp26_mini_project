"""Inspect specific 8-gram contamination matches to categorize them as
BENIGN (common phrase) vs CONCERNING (actual test-content leak).

For each flagged 8-gram, we find:
  - Which SPECIFIC test item(s) the 8-gram came from
  - Which SPECIFIC training row(s) matched
  - The surrounding context on both sides

If the surrounding context differs substantially, the 8-gram is coincidental
(benign). If surrounding context also matches, it's real contamination.

This is the ground-truth way to resolve the 74K-flagged-rows concern: open up
each flag and look at it.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).resolve().parent.parent


def _word_ngrams_positions(text: str, n: int):
    """Return list of (ngram, start_char_idx) so we can show context."""
    words_with_pos = []
    for m in re.finditer(r"[\w]+", text.lower()):
        words_with_pos.append((m.group(), m.start()))
    out = []
    for i in range(len(words_with_pos) - n + 1):
        ngram = tuple(w for w, _ in words_with_pos[i : i + n])
        start_char = words_with_pos[i][1]
        out.append((ngram, start_char))
    return out


def _iter_training_rows(path: Path, max_rows=None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in row:
                content = "\n".join(m.get("content", "") for m in row["messages"])
            else:
                content = " ".join(
                    str(row.get(k, "")) for k in ("text", "code", "prompt", "question")
                )
                tests = row.get("tests") or row.get("test_list") or []
                if tests:
                    content = content + "\n" + "\n".join(str(t) for t in tests)
            yield i, content


def _norm_for_ngram(text: str) -> str:
    """Same normalization as the TA-approved n-gram filter: lowercase,
    punctuation stripped, whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def show_context(text: str, pat: str, width: int = 200) -> str:
    """Return a substring of text around the first position where pat appears
    (pat is an 8-gram as space-joined string; match anywhere in lowercased text)."""
    lower = text.lower()
    # Find pat as a near-substring — allow any whitespace between words.
    words = pat.split()
    pattern = re.compile(r"\W+".join(re.escape(w) for w in words))
    m = pattern.search(lower)
    if not m:
        return "(no match found in context)"
    start = max(0, m.start() - width)
    end = min(len(text), m.end() + width)
    return "..." + text[start:end].replace("\n", " / ") + "..."


def investigate_suspicious_gsm_match():
    """The 'fewer rose stamps than truck stamps how many' match flagged in
    gsm8k-train → gsm8k-test. Find it in both places and compare contexts."""
    target = "fewer rose stamps than truck stamps how many"
    print(f"\n{'='*80}\nINVESTIGATING: {target!r}\n{'='*80}")

    # Find in GSM8K-test
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_hits = []
    for i, row in enumerate(test_ds):
        q_lower = row["question"].lower()
        a_lower = row["answer"].lower()
        if target in _norm_for_ngram(row["question"]):
            test_hits.append((i, "question", row["question"][:300]))
        if target in _norm_for_ngram(row["answer"]):
            test_hits.append((i, "answer", row["answer"][:300]))
    print(f"\nGSM8K-TEST occurrences of this 8-gram:")
    for tid, field, excerpt in test_hits[:3]:
        print(f"  test/{tid} [{field}]: {excerpt!r}")

    # Find in GSM8K-train
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    train_hits = []
    for i, row in enumerate(train_ds):
        q_lower = row["question"].lower()
        a_lower = row["answer"].lower()
        if target in _norm_for_ngram(row["question"]):
            train_hits.append((i, "question", row["question"][:300]))
        if target in _norm_for_ngram(row["answer"]):
            train_hits.append((i, "answer", row["answer"][:300]))
    print(f"\nGSM8K-TRAIN occurrences of this 8-gram:")
    for tid, field, excerpt in train_hits[:3]:
        print(f"  train/{tid} [{field}]: {excerpt!r}")


def inspect_argilla_matches(max_samples=5):
    """Show several argilla IFEval 8-gram matches to confirm they're all
    shared constraint-library phrases."""
    target_phrases = [
        "highlighted section your entire response should be in",
        "response finish your response with this exact phrase",
        "your answer must contain a title wrapped in",
    ]
    print(f"\n{'='*80}\nARGILLA IFEval 8-GRAM MATCHES\n{'='*80}")

    # Find in IFEval
    ife = load_dataset("google/IFEval", split="train")
    argilla_path = REPO / "training" / "data" / "_audit_argilla_filtered.jsonl"

    for pat in target_phrases:
        print(f"\n--- 8-gram: {pat!r} ---")
        # IFEval sources
        ife_hits = []
        for i, row in enumerate(ife):
            if pat in _norm_for_ngram(row["prompt"]):
                ife_hits.append((i, row["prompt"][:200]))
        print(f"  Found in {len(ife_hits)} IFEval prompts. Sample: {ife_hits[0][1] if ife_hits else '(none)'!r}")
        # Argilla
        arg_hits_count = 0
        arg_sample = None
        for i, content in _iter_training_rows(argilla_path):
            if pat in _norm_for_ngram(content):
                arg_hits_count += 1
                if arg_sample is None:
                    arg_sample = content[:250]
        print(f"  Found in {arg_hits_count} argilla rows. Sample: {arg_sample!r}")


def inspect_humaneval_matches():
    """HE 8-gram matches in r1_clean: are they generic or specific?"""
    target_phrases = [
        "your task is to write a function that",
        "for i in range n for j in",
        "1 2 3 4 5 6 7 8",
    ]
    print(f"\n{'='*80}\nHUMANEVAL 8-GRAM MATCHES IN r1_clean\n{'='*80}")

    he = load_dataset("openai/openai_humaneval", split="test")
    r1_path = REPO / "training" / "data" / "r1_clean.jsonl"

    for pat in target_phrases:
        print(f"\n--- 8-gram: {pat!r} ---")
        he_hits = []
        for i, row in enumerate(he):
            combined = _norm_for_ngram(row["prompt"] + " " + row["canonical_solution"])
            if pat in combined:
                field = "prompt" if pat in _norm_for_ngram(row["prompt"]) else "solution"
                # show which HumanEval item
                he_hits.append((row["task_id"], field))
        print(f"  In {len(he_hits)} HumanEval item(s): {he_hits[:5]}")

        r1_hits_count = 0
        r1_sample = None
        for i, content in _iter_training_rows(r1_path):
            if pat in _norm_for_ngram(content):
                r1_hits_count += 1
                if r1_sample is None:
                    r1_sample = content[:300]
                if r1_hits_count >= 100:
                    break
        print(f"  In >={r1_hits_count} r1_clean rows. Sample: {r1_sample!r}")


def main():
    investigate_suspicious_gsm_match()
    inspect_argilla_matches()
    inspect_humaneval_matches()


if __name__ == "__main__":
    main()
