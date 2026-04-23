"""
GRPO environment + dataset builder for IFEval-style instruction following.

Reward = IFEval's verifier. Binary per-prompt (all constraints pass → 1.0 else 0.0).
Prompt source = argilla/ifeval-like-data filtered subset. The filter guarantees the
dataset's own constraints are machine-verifiable. We ALSO dedup against
google/IFEval prompts so none of the eval's 541 prompts leak into RL training.

Pattern mirrors tinker_cookbook/recipes/math_rl/math_env.py's MathEnv / MathDataset.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from functools import partial
from typing import Literal, cast

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks._common import parse_kwargs
from tinker_cookbook.eval.benchmarks._ifeval_verify import verify_all_instructions
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class IFEvalEnv(ProblemEnv):
    """Single-turn env where reward comes from IFEval's verify_all_instructions."""

    def __init__(
        self,
        prompt: str,
        instruction_ids: list[str],
        kwargs_list: list[dict],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.prompt = prompt
        self.instruction_ids = instruction_ids
        self.kwargs_list = kwargs_list

    def get_question(self) -> str:
        return self.prompt

    def check_format(self, sample_str: str) -> bool:
        return bool(sample_str.strip())

    def check_answer(self, sample_str: str) -> bool:
        try:
            fraction, _ = verify_all_instructions(
                sample_str, self.instruction_ids, self.kwargs_list
            )
        except Exception as e:
            logger.warning("IFEval verifier failed on prompt (treating as wrong): %s", e)
            return False
        return fraction == 1.0

    def get_reference_answer(self) -> str:
        return "|".join(self.instruction_ids)


def _load_ifeval_deny_prompts() -> set[str]:
    """Exact-match deny set of the 541 google/IFEval prompts."""
    ds = load_dataset("google/IFEval", split="train")
    return {r["prompt"].strip() for r in ds}


def _load_argilla_filtered() -> Dataset:
    """Load argilla/ifeval-like-data's filtered subset (56K verifier-guaranteed prompts)."""
    # "filtered" is a config (subset), not a split. The split within is "train".
    ds = load_dataset("argilla/ifeval-like-data", name="filtered", split="train")
    return cast(Dataset, ds)


class IFEvalRLDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        seed: int = 0,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        ds = _load_argilla_filtered()
        deny = _load_ifeval_deny_prompts()
        ds = ds.filter(lambda row: row["prompt"].strip() not in deny)
        logger.info("argilla filtered: %d rows after dedup against google/IFEval", len(ds))
        self.ds = ds.shuffle(seed=seed)
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
            builder = self._make_group_builder(row)
            if builder is not None:
                out.append(builder)
        return out

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_group_builder(self, row: dict) -> ProblemGroupBuilder | None:
        prompt = row["prompt"]
        instruction_ids = list(row["instruction_id_list"])
        try:
            kwargs_list = parse_kwargs(row["kwargs"], instruction_ids)
        except Exception as e:
            logger.warning("parse_kwargs failed (dropping row): %s", e)
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                IFEvalEnv,
                prompt,
                instruction_ids,
                kwargs_list,
                self.renderer,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=self.group_size,
            dataset_name="ifeval_rl",
        )


@chz.chz
class IFEvalRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[IFEvalRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            IFEvalRLDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                seed=self.seed,
            ),
            None,
        )
