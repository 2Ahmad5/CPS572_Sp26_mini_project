"""
GRPO environment + dataset builder for GSM8K-style math problems.

Reward = 1.0 if the extracted numeric answer matches ground truth, 0.0 otherwise.
Answer extraction matches our SFT format (`ANSWER: N`) first, falls back to
last number in the response. This is intentional: we must reward the SAME output
format we trained the SFT model to produce, or RL would un-learn GSM8K.

Prompts come from `openai/gsm8k` train split (7,473 problems), dedupped against
the GSM8K *test* split via the project's existing 13-gram deny set. The train
set has no overlap with test by construction (different HuggingFace splits),
but dedup is kept as a belt-and-suspenders sanity check.

Pattern mirrors `training/rl_ifeval.py`.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from functools import partial
from typing import cast

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# Prefers our SFT-trained "ANSWER: N" sentinel over stray numbers earlier in the CoT.
_ANSWER_TAG_RE = re.compile(r"ANSWER:\s*(-?[\d.,]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    m = _ANSWER_TAG_RE.search(text)
    candidate = m.group(1) if m else None
    if candidate is None:
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
        candidate = nums[-1]
    try:
        return float(candidate.replace(",", "").rstrip("."))
    except ValueError:
        return None


def _parse_gsm8k_answer(answer_field: str) -> float | None:
    """GSM8K's answer field ends with '#### N'. Extract N as float."""
    if "####" not in answer_field:
        return None
    tail = answer_field.rsplit("####", 1)[1].strip()
    try:
        return float(tail.replace(",", ""))
    except ValueError:
        return None


class MathRLEnv(ProblemEnv):
    """Single-turn env where reward = numeric answer matches ground truth."""

    def __init__(
        self,
        question: str,
        ground_truth: float,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.question = question
        self.ground_truth = ground_truth

    def get_question(self) -> str:
        return self.question

    def check_format(self, sample_str: str) -> bool:
        return bool(sample_str.strip())

    def check_answer(self, sample_str: str) -> bool:
        extracted = _extract_number(sample_str)
        if extracted is None:
            return False
        return math.isclose(extracted, self.ground_truth, rel_tol=1e-6, abs_tol=1e-6)

    def get_reference_answer(self) -> str:
        return str(self.ground_truth)


class MathRLDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        seed: int = 0,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = cast(Dataset, ds)
        rows = [
            {"question": r["question"], "answer": _parse_gsm8k_answer(r["answer"])}
            for r in ds
        ]
        rows = [r for r in rows if r["answer"] is not None]
        logger.info("MathRLDataset: %d gsm8k train rows with parseable answers", len(rows))
        self.rows = rows
        self.ds = Dataset.from_list(rows).shuffle(seed=seed)
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
                        MathRLEnv,
                        row["question"],
                        float(row["answer"]),
                        self.renderer,
                        convo_prefix=self.convo_prefix,
                    ),
                    num_envs=self.group_size,
                    dataset_name="gsm8k_rl",
                )
            )
        return out

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)


@chz.chz
class MathRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[MathRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            MathRLDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                seed=self.seed,
            ),
            None,
        )
