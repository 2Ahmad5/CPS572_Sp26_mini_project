"""
GRPO environment + dataset builder for code generation (MBPP-driven, HumanEval-aligned).

Reward = 1.0 if the sampled code passes ALL public tests for the prompt, 0.0 else.
Tests are executed in a timeout-guarded subprocess — same pattern as
`training/bon_run.py:_run_tests`, which is already trusted in-project. No Modal
or SandboxFusion dependency.

Prompts come from `google-research-datasets/mbpp` sanitized train split. We hard
decontaminate against `openai/openai_humaneval` using the project's existing
13-gram deny set to guarantee no eval leakage.

The first MBPP test is appended to the prompt to disambiguate function signature
(matches bon_run.py's convention and MBPP's standard usage).

Pattern mirrors `training/rl_math_env.py`.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import cast

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from training.data_sources import build_deny_ngrams, is_contaminated

logger = logging.getLogger(__name__)


_PY_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(response: str) -> str:
    """Prefer fenced ```python block; fall back to the raw response."""
    m = _PY_FENCE_RE.search(response)
    if m:
        return m.group(1)
    return response


def _run_tests(code: str, tests: list[str], timeout: float = 5.0) -> int:
    """Execute `code` followed by each test in a subprocess. Return number passed."""
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


class CodeRLEnv(ProblemEnv):
    """Single-turn env where reward = 1.0 iff generated code passes every MBPP test."""

    def __init__(
        self,
        question: str,
        tests: list[str],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        test_timeout: float = 5.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.question = question
        self.tests = tests
        self.test_timeout = test_timeout

    def get_question(self) -> str:
        return self.question

    def check_format(self, sample_str: str) -> bool:
        return bool(sample_str.strip())

    def check_answer(self, sample_str: str) -> bool:
        code = _extract_code(sample_str)
        passed = _run_tests(code, self.tests, timeout=self.test_timeout)
        return passed == len(self.tests) and len(self.tests) > 0

    def get_reference_answer(self) -> str:
        return f"{len(self.tests)} tests"


def _load_mbpp_decontaminated(max_rows: int | None = None) -> list[dict]:
    """Load MBPP sanitized train and drop any row that leaks HumanEval n-grams."""
    logger.info("Building HumanEval deny-set for MBPP decontamination ...")
    deny = build_deny_ngrams()
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    kept: list[dict] = []
    dropped_contam = 0
    for r in ds:
        text = r["prompt"]
        code = r["code"]
        tests = list(r["test_list"])
        if is_contaminated(text, deny) or is_contaminated(code, deny):
            dropped_contam += 1
            continue
        kept.append({"text": text, "code": code, "tests": tests})
        if max_rows is not None and len(kept) >= max_rows:
            break
    logger.info(
        "MBPP sanitized: %d rows kept, %d dropped as HumanEval-contaminated",
        len(kept),
        dropped_contam,
    )
    return kept


class CodeRLDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        seed: int = 0,
        convo_prefix: list[renderers.Message] | None = None,
        max_rows: int | None = None,
    ):
        rows = _load_mbpp_decontaminated(max_rows=max_rows)
        # Prepend the first test to the user prompt, matching bon_run.py's
        # MBPP convention so the model knows the target function signature.
        prompts = []
        for r in rows:
            first_test = r["tests"][0] if r["tests"] else ""
            question = f"{r['text']}\nYour code should satisfy: {first_test}"
            prompts.append({"question": question, "tests": r["tests"]})
        self.ds = Dataset.from_list(prompts).shuffle(seed=seed)
        logger.info("CodeRLDataset: %d MBPP train prompts", len(self.ds))
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.ds))
        assert start < end, "Incorrect batch size"
        out: list[EnvGroupBuilder] = []
        for row in self.ds.select(range(start, end)):
            out.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        CodeRLEnv,
                        row["question"],
                        list(row["tests"]),
                        self.renderer,
                        convo_prefix=self.convo_prefix,
                    ),
                    num_envs=self.group_size,
                    dataset_name="mbpp_rl",
                )
            )
        return out

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)


@chz.chz
class CodeRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    max_rows: int | None = None

    async def __call__(self) -> tuple[CodeRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            CodeRLDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                seed=self.seed,
                max_rows=self.max_rows,
            ),
            None,
        )
