"""
Best-of-N distillation for code (HumanEval-focused).

Samples N completions per MBPP prompt from a checkpoint, executes each against
MBPP's public test cases in a timeout-guarded subprocess, keeps the top-1 per
prompt, and writes a standard SFT JSONL that `training/sft_train.py --config
r8_bon_8b` consumes.

Why MBPP (not HumanEval): HumanEval is our eval set — training on it would be
contamination. MBPP is the standard Python SFT companion dataset, and we also
13-gram decontaminate against HumanEval as a belt-and-suspenders check.

Why subprocess (not sandbox service): we control inputs end-to-end and the
surface area is ~80 lines. No Modal / SandboxFusion needed.

Usage:
    python -m training.bon_run \\
        --checkpoint "tinker://..../sampler_weights/final" \\
        --base_model meta-llama/Llama-3.1-8B \\
        --n_prompts 500 --n_samples 16 --max_tokens 1024 \\
        --output training/data/r8_bon.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import tinker
from datasets import load_dataset
from tinker import SamplingParams

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from training.data_sources import build_deny_ngrams, is_contaminated

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("bon_run")


_PY_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(response: str) -> str:
    """Pull Python from a ```python fenced block. Fall back to raw response."""
    m = _PY_FENCE_RE.search(response)
    if m:
        return m.group(1)
    return response


def _run_tests(code: str, tests: list[str], timeout: float = 5.0) -> int:
    """Execute `code` followed by each test in a subprocess. Return # passed."""
    passed = 0
    for test in tests:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code + "\n\n" + test + "\n")
            path = f.name
        try:
            result = subprocess.run(
                [sys.executable, path],
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        finally:
            try:
                Path(path).unlink()
            except OSError:
                pass
    return passed


def _load_mbpp_decontaminated(deny: set) -> list[dict]:
    """Load MBPP train, drop any row whose text/code collides with HumanEval."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    kept = []
    dropped_contam = 0
    for r in ds:
        text = r["prompt"]  # MBPP sanitized uses "prompt" field
        code = r["code"]
        tests = list(r["test_list"])
        if is_contaminated(text, deny) or is_contaminated(code, deny):
            dropped_contam += 1
            continue
        kept.append({"text": text, "code": code, "tests": tests})
    log.info("MBPP sanitized: %d rows kept, %d dropped as HumanEval-contaminated",
             len(kept), dropped_contam)
    return kept


async def _sample_and_score(
    sampling_client,
    renderer,
    prompt_row: dict,
    n_samples: int,
    max_tokens: int,
) -> tuple[str, str, int] | None:
    """Sample N completions for a prompt. Return (question, best_response, best_score)."""
    # MBPP's first test is traditionally included in the prompt to disambiguate function signature.
    first_test = prompt_row["tests"][0] if prompt_row["tests"] else ""
    question = f"{prompt_row['text']}\nYour code should satisfy: {first_test}"
    convo = [{"role": "user", "content": question}]
    model_input = renderer.build_generation_prompt(convo)
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )
    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=n_samples,
        sampling_params=params,
    )

    best_resp = None
    best_score = -1
    for seq in resp.sequences:
        msg, _ = renderer.parse_response(seq.tokens)
        content = msg["content"]
        code = _extract_code(content)
        score = _run_tests(code, prompt_row["tests"])
        if score > best_score:
            best_score = score
            best_resp = content

    if best_score <= 0:
        return None  # no rollout passed any test — drop
    return (question, best_resp, best_score)


def _load_extra_prompts(path: Path) -> list[dict]:
    """Load {text, code, tests} rows from a prep_candidate_datasets-style JSONL."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("text") and r.get("tests"):
                rows.append({"text": r["text"], "code": r.get("code", ""), "tests": list(r["tests"])})
    return rows


async def run(args):
    sc = tinker.ServiceClient()
    sampling_client = await sc.create_sampling_client_async(model_path=args.checkpoint)

    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    log.info("Building HumanEval deny-set for MBPP decontamination ...")
    deny = build_deny_ngrams()
    rows = _load_mbpp_decontaminated(deny)

    # If an --extra_prompts file is given, append those (already pre-decontam'd
    # by prep_candidate_datasets.py). Useful for R14 RSFT where we want more
    # prompt coverage than the 120-row MBPP sanitized split gives.
    if args.extra_prompts:
        extra = _load_extra_prompts(Path(args.extra_prompts))
        log.info("Loaded %d extra prompts from %s", len(extra), args.extra_prompts)
        rows.extend(extra)

    rows = rows[: args.n_prompts]
    log.info("Using %d prompts total for BoN rollout", len(rows))

    sem = asyncio.Semaphore(args.concurrency)

    async def _with_sem(row):
        async with sem:
            return await _sample_and_score(sampling_client, renderer, row,
                                           args.n_samples, args.max_tokens)

    tasks = [asyncio.create_task(_with_sem(r)) for r in rows]

    kept: list[dict] = []
    all_tests = 0
    passed_total = 0
    completed = 0
    for fut in asyncio.as_completed(tasks):
        result = await fut
        completed += 1
        if result is None:
            continue
        question, response, score = result
        kept.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ],
        })
        passed_total += score
        all_tests += 3  # MBPP has 3 tests per row
        if completed % 25 == 0:
            log.info("BoN progress: %d/%d prompts, %d kept (%.1f%% test-pass rate on winners)",
                     completed, len(tasks), len(kept),
                     100.0 * passed_total / max(1, all_tests))

    log.info("BoN done: %d kept from %d prompts, winner pass-rate %.1f%%",
             len(kept), len(rows), 100.0 * passed_total / max(1, all_tests))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows → %s", len(kept), out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--renderer_name", default="role_colon")
    p.add_argument("--n_prompts", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--extra_prompts", default=None,
                   help="JSONL of extra prompts ({text,code,tests}) to append to "
                        "MBPP sanitized. Already-decontaminated at build time.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 25 prompts × 4 samples")
    args = p.parse_args()
    if args.smoke:
        args.n_prompts = 25
        args.n_samples = 4
        log.info("SMOKE MODE: 25 prompts × 4 samples")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
