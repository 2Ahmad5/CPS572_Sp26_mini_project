"""
Rejection Fine-Tuning (RFT) pipeline.

Samples N rollouts per prompt from a checkpoint, verifies each with the same
verifier GRPO would use (IFEval's verify_all_instructions for IF / numeric
match for math), keeps only verifier-correct rollouts, and writes them as
a standard SFT JSONL that `training/sft_train.py --config r8_rft_8b` consumes.

This is the cheapest of the three RL additions — no policy gradients, no KL
penalty, no group advantage. Just sample-and-filter. Per Reinforce-Rej (arxiv
2504.11343), RFT is competitive with GRPO on math when the base is strong.

Usage:
    python -m training.rft_run \\
        --checkpoint "tinker://588e8de1-.../sampler_weights/final" \\
        --base_model meta-llama/Llama-3.1-8B \\
        --n_prompts_per_source 500 --n_samples 8 --max_tokens 1024 \\
        --output training/data/r8_rft.jsonl

    # Smoke test first:
    python -m training.rft_run --smoke --checkpoint "tinker://..." \\
        --base_model meta-llama/Llama-3.1-8B --output training/data/r8_rft_smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import tinker
from tinker import SamplingParams

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from training.rl_ifeval import IFEvalRLDatasetBuilder
from training.rl_math_env import MathRLDatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("rft_run")


async def _collect_for_source(
    sampling_client,
    renderer,
    dataset,
    source_tag: str,
    n_prompts: int,
    n_samples: int,
    max_tokens: int,
    concurrency: int,
) -> list[dict]:
    """Iterate `n_prompts` group builders from `dataset`, sample N, keep correct."""
    sem = asyncio.Semaphore(concurrency)
    stop_seqs = renderer.get_stop_sequences()

    async def _sample_one(group_builder: ProblemGroupBuilder) -> list[dict]:
        env = group_builder.env_thunk()
        question = env.get_question()
        convo = [{"role": "user", "content": question}]
        model_input = renderer.build_generation_prompt(convo)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
            stop=stop_seqs,
        )
        async with sem:
            resp = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=n_samples,
                sampling_params=params,
            )
        kept = []
        for seq in resp.sequences:
            msg, _ = renderer.parse_response(seq.tokens)
            content = msg["content"]
            if env.check_answer(content):
                kept.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": content},
                    ],
                })
        return kept

    # Walk dataset in order, collecting the first n_prompts group builders.
    prompts_seen = 0
    tasks = []
    for batch_idx in range(len(dataset)):
        batch = dataset.get_batch(batch_idx)
        for group_builder in batch:
            if prompts_seen >= n_prompts:
                break
            tasks.append(asyncio.create_task(_sample_one(group_builder)))
            prompts_seen += 1
        if prompts_seen >= n_prompts:
            break
    log.info("[%s] scheduled %d prompts × %d samples", source_tag, prompts_seen, n_samples)

    all_kept: list[dict] = []
    completed = 0
    for fut in asyncio.as_completed(tasks):
        batch_kept = await fut
        all_kept.extend(batch_kept)
        completed += 1
        if completed % 25 == 0:
            log.info("[%s] %d/%d prompts processed, %d verified-correct so far",
                     source_tag, completed, len(tasks), len(all_kept))

    log.info("[%s] done: %d verified-correct rollouts from %d prompts (%.1f%% keep)",
             source_tag, len(all_kept), prompts_seen,
             100.0 * len(all_kept) / max(1, prompts_seen * n_samples))
    return all_kept


async def run(args):
    sc = tinker.ServiceClient()
    sampling_client = await sc.create_sampling_client_async(model_path=args.checkpoint)

    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    # Group size 1 — for RFT we want distinct prompts, not replicated groups.
    if_builder = IFEvalRLDatasetBuilder(
        batch_size=64,
        model_name_for_tokenizer=args.base_model,
        renderer_name=args.renderer_name,
        group_size=1,
        seed=0,
    )
    math_builder = MathRLDatasetBuilder(
        batch_size=64,
        model_name_for_tokenizer=args.base_model,
        renderer_name=args.renderer_name,
        group_size=1,
        seed=0,
    )
    if_ds, _ = await if_builder()
    math_ds, _ = await math_builder()

    all_kept: list[dict] = []
    all_kept.extend(await _collect_for_source(
        sampling_client, renderer, if_ds, "if",
        n_prompts=args.n_prompts_per_source,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    ))
    all_kept.extend(await _collect_for_source(
        sampling_client, renderer, math_ds, "math",
        n_prompts=args.n_prompts_per_source,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in all_kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows → %s", len(all_kept), out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="sampler_weights tinker:// URI")
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--renderer_name", default="role_colon")
    p.add_argument("--n_prompts_per_source", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 50 prompts × 4 samples per source")
    args = p.parse_args()
    if args.smoke:
        args.n_prompts_per_source = 50
        args.n_samples = 4
        log.info("SMOKE MODE: 50 prompts/source × 4 samples, max_tokens=%d", args.max_tokens)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
