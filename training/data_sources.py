"""
Per-source adapters, contamination filter, and length filter for the R1 mix.

Every adapter returns either a standardized dict ``{"messages": [{role, content}, ...]}``
or ``None`` to drop the row. No silent fallbacks — if an adapter can't extract the
numeric answer for a math row, it drops the row and the caller counts the drop.

Hard eval-data blacklist (never train on any of these):
  - google/IFEval (541 prompts)
  - openai/gsm8k split="test" (1319 items)
  - openai/openai_humaneval (164 items — prompts + canonical solutions)

A 13-gram word-level filter against a combined deny-set catches verbatim leakage.
Any training-source row with a matching 13-gram in its user-or-assistant content is
dropped. This mirrors BigCode's decontamination pipeline.
"""

import logging
import re
from typing import Callable

from datasets import load_dataset

logger = logging.getLogger(__name__)

N_GRAM = 13


def build_deny_ngrams(n: int = N_GRAM) -> set[tuple[str, ...]]:
    """Build the n-gram deny set from the three held-out eval benchmarks."""
    deny: set[tuple[str, ...]] = set()
    texts: list[str] = []

    ifeval = load_dataset("google/IFEval", split="train")
    texts.extend(r["prompt"] for r in ifeval)

    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    texts.extend(r["question"] for r in gsm_test)
    texts.extend(r["answer"] for r in gsm_test)

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    texts.extend(r["prompt"] for r in humaneval)
    texts.extend(r["canonical_solution"] for r in humaneval)

    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            deny.add(tuple(words[i : i + n]))

    logger.info(
        "Built deny-set: %d eval texts → %d unique %d-grams",
        len(texts),
        len(deny),
        n,
    )
    return deny


def is_contaminated(text: str, deny: set[tuple[str, ...]], n: int = N_GRAM) -> bool:
    """Return True if any word-level n-gram of ``text`` is in the deny set."""
    words = text.lower().split()
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) in deny:
            return True
    return False


def row_contaminated(messages: list[dict], deny: set[tuple[str, ...]]) -> bool:
    """Check all message contents in a standardized row."""
    for m in messages:
        if is_contaminated(m["content"], deny):
            return True
    return False


# -----------------------------------------------------------------------------
# Per-source adapters. Each returns standardized ``{"messages": [...]}`` or None.
# -----------------------------------------------------------------------------


def adapt_gsm8k_train(row: dict) -> dict | None:
    """openai/gsm8k (train split only). Converts "#### N" ending to "ANSWER: N"."""
    question = row["question"]
    answer = row["answer"]
    if "####" not in answer:
        return None
    reasoning, final = answer.rsplit("####", 1)
    final = final.strip()
    reasoning = reasoning.strip()
    assistant = f"{reasoning}\nANSWER: {final}"
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
    }


def adapt_openmathinstruct2(row: dict) -> dict | None:
    """nvidia/OpenMathInstruct-2. Uses the ``expected_answer`` field directly."""
    problem = row["problem"]
    solution = row["generated_solution"]
    answer = row.get("expected_answer")
    if not answer:
        return None
    assistant = f"{solution}\nANSWER: {answer}"
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant},
        ]
    }


_METAMATH_ANSWER_RE = re.compile(r"The answer is:\s*([^\n]+)")


def adapt_metamathqa(row: dict) -> dict | None:
    """meta-math/MetaMathQA. Extracts final answer from 'The answer is: N'."""
    query = row["query"]
    response = row["response"]
    m = _METAMATH_ANSWER_RE.search(response)
    if m is None:
        return None
    ans_raw = m.group(1).strip().rstrip(".")
    num_m = re.match(r"(-?[\d,]+\.?\d*|-?\.\d+)", ans_raw)
    ans = num_m.group(1).replace(",", "") if num_m else ans_raw
    assistant = response if "ANSWER:" in response else f"{response}\nANSWER: {ans}"
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant},
        ]
    }


def _normalize_messages(msgs: list) -> list[dict] | None:
    """Normalize Tulu-3 messages. Handles both [{role,content}] and [{content}] shapes."""
    if not msgs or len(msgs) < 2:
        return None
    out = []
    roles = ["user", "assistant"]
    for i, m in enumerate(msgs[:2]):
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant", "system"):
            role = roles[i]
        content = m["content"] if isinstance(m, dict) else str(m)
        out.append({"role": role, "content": content})
    return out


_NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def adapt_persona_math_grade(row: dict) -> dict | None:
    """allenai/tulu-3-sft-personas-math-grade. Appends 'ANSWER: N' using last numeric."""
    msgs = _normalize_messages(row["messages"])
    if msgs is None:
        return None
    assistant_content = msgs[1]["content"]
    nums = _NUM_RE.findall(assistant_content)
    if not nums:
        return None
    final = nums[-1].replace(",", "")
    if "ANSWER:" not in assistant_content:
        msgs[1]["content"] = f"{assistant_content}\nANSWER: {final}"
    return {"messages": msgs}


def adapt_magicoder_evol(row: dict) -> dict | None:
    """ise-uiuc/Magicoder-Evol-Instruct-110K. Filters to Python-containing responses."""
    instruction = row["instruction"]
    response = row["response"]
    has_py_fence = "```python" in response.lower()
    has_def = bool(re.search(r"(?m)^\s*def\s+\w+\s*\(", response))
    if not (has_py_fence or has_def):
        return None
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def adapt_magicoder_oss(row: dict) -> dict | None:
    """ise-uiuc/Magicoder-OSS-Instruct-75K. Uses the ``lang`` field to filter Python."""
    if row.get("lang") != "python":
        return None
    return {
        "messages": [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ]
    }


def adapt_persona_code(row: dict) -> dict | None:
    """allenai/tulu-3-sft-personas-code."""
    msgs = _normalize_messages(row["messages"])
    if msgs is None:
        return None
    return {"messages": msgs}


def adapt_bigcode_self_oss(row: dict) -> dict | None:
    """bigcode/self-oss-instruct-sc2-exec-filter-50k. Python-only, execution-validated."""
    instruction = row["instruction"]
    response = row["response"]
    if not re.search(r"(?m)^\s*def\s+\w+\s*\(", response):
        return None
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def adapt_persona_if(row: dict) -> dict | None:
    """allenai/tulu-3-sft-personas-instruction-following. Pass-through."""
    msgs = _normalize_messages(row["messages"])
    if msgs is None:
        return None
    return {"messages": msgs}


def adapt_no_robots(row: dict) -> dict | None:
    """HuggingFaceH4/no_robots. Keeps system+user+assistant shape."""
    msgs = row["messages"]
    if not msgs or len(msgs) < 2:
        return None
    out = []
    for m in msgs:
        role = m["role"] if m.get("role") in ("system", "user", "assistant") else "user"
        out.append({"role": role, "content": m["content"]})
    return {"messages": out}


# -----------------------------------------------------------------------------
# Source registry: (hf_path, hf_name_or_None, split, adapter, bucket)
# -----------------------------------------------------------------------------

SourceSpec = tuple[str, str | None, str, Callable[[dict], dict | None], str]

SOURCES: list[SourceSpec] = [
    # (path, name, split, adapter, bucket)
    # Dropped on cost-audit 2026-04-19: Magicoder-OSS (redundant with Magicoder-Evol,
    # same distillation lineage) and no_robots (1% slice, unmeasurable signal).
    ("openai/gsm8k", "main", "train", adapt_gsm8k_train, "math"),
    ("nvidia/OpenMathInstruct-2", None, "train_1M", adapt_openmathinstruct2, "math"),
    ("meta-math/MetaMathQA", None, "train", adapt_metamathqa, "math"),
    ("allenai/tulu-3-sft-personas-math-grade", None, "train", adapt_persona_math_grade, "math"),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", None, "train", adapt_magicoder_evol, "code"),
    ("allenai/tulu-3-sft-personas-code", None, "train", adapt_persona_code, "code"),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, "train", adapt_bigcode_self_oss, "code"),
    ("allenai/tulu-3-sft-personas-instruction-following", None, "train", adapt_persona_if, "if"),
]
