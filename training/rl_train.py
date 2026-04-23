"""
Entry point for GRPO runs on IFEval-style prompts.

Usage:
    python -m training.rl_train --config r3_grpo_ifeval_3b
    python -m training.rl_train --config r7_grpo_ifeval_8b

Mirrors tinker_cookbook/recipes/math_rl/train.py's structure but uses our
IFEvalRLDatasetBuilder. Hyperparameters derived from Tulu-3 RLVR (paper's
LR 3e-7 for full-FT → LoRA uses ~10x per "LoRA Without Regret") with KL β=0.05
from Tulu-3's middle sweep value. Group size and batch kept moderate for
3B first; matches the cookbook's math_rl defaults.
"""

import argparse
import asyncio
import logging
import sys

# Force UTF-8 stdout/stderr on Windows so rollouts containing non-cp1252 unicode
# (bullets, smart quotes, emoji in Magicoder code comments, etc.) don't crash logging.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main

from training.rl_ifeval import IFEvalRLDatasetBuilder
from training.rl_math_env import MathRLDatasetBuilder
from training.rl_code_env import CodeRLDatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("rl_train")


# Tinker's LoRA-LR formula: lr = 5e-5 * 10 * (2000/hidden)^0.781
#   3B (hidden 3072): 3.56e-4 for SFT LR. For RL we use ~10x smaller still
#   (Tulu-3 paper had full-FT RL LR 3e-7; LoRA = ~3e-6).
# We pick 1e-5 as a moderate value (cookbook's math_rl default) — conservative,
# safe against catastrophic forgetting of the SFT model.

RUN_CONFIGS: dict[str, dict] = {
    "r3_grpo_ifeval_3b": dict(
        model_name="meta-llama/Llama-3.2-3B",
        renderer_name="role_colon",
        load_checkpoint_path=None,  # set to R1 3B best checkpoint at invocation
        log_path="logs/r3_grpo_ifeval_3b",
        lora_rank=32,
        learning_rate=1e-5,
        kl_penalty_coef=0.05,
        group_size=16,
        groups_per_batch=32,
        max_tokens=1024,
        temperature=1.0,
        save_every=50,
        eval_every=50,
        max_steps=200,
        wandb_name="r3_grpo_ifeval_3b",
    ),
    "r7_grpo_ifeval_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="role_colon",
        load_checkpoint_path=None,  # set to R6 8B best checkpoint at invocation
        log_path="logs/r7_grpo_ifeval_8b",
        lora_rank=32,
        learning_rate=1e-5,
        kl_penalty_coef=0.05,
        group_size=16,
        groups_per_batch=32,
        max_tokens=1024,
        temperature=1.0,
        save_every=50,
        eval_every=50,
        max_steps=200,
        wandb_name="r7_grpo_ifeval_8b",
    ),
    # R8: mixed-reward GRPO on 8B starting from R6. Tulu-3 §6 / IFBench: single-task
    # RL regresses other tasks; interleaving IF + math prevents the R3 failure mode.
    # Hyperparameters from Tulu-3 tulu3.md and cookbook math_rl README:
    #   group_size 8, groups_per_batch 64 (vs cookbook default 16/32) = same-quality
    #   GRPO signal at ~1/4 wall-clock cost. LR 1.5e-5 (10x Tulu-3's 3e-7 full-FT per
    #   "LoRA Without Regret"'s 10x rule). KL beta 0.05 (Tulu-3 8B sweep winner, which
    #   R3 already used correctly — the problem was single-task, not KL).
    "r8_grpo_mixed_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="role_colon",
        # set to R6 8B final at invocation: tinker://588e8de1-.../weights/final
        load_checkpoint_path=None,
        log_path="logs/r8_grpo_mixed_8b",
        lora_rank=32,
        learning_rate=1.5e-5,
        kl_penalty_coef=0.05,
        group_size=8,
        groups_per_batch=64,
        max_tokens=1024,
        temperature=1.0,
        save_every=25,
        eval_every=25,
        max_steps=100,  # early-stop on plateau; realistic stop ~50 steps
        wandb_name="r8_grpo_mixed_8b",
        # Mixed: 50% IF, 50% math. Groups never cross domains within a batch
        # (InterleavedRLDatasetBuilder preserves GRPO's per-group advantage centering).
        mixed_weights=[0.5, 0.5],
    ),
    # R9: 3-way mixed GRPO (IF + math + code) starting from R8-BoN final. The gap
    # identified in the audit: R8-GRPO only optimized IF+math rewards; HumanEval
    # improvements in R8-BoN came from SFT distillation, not RL. Adding MBPP as
    # a code verifier closes the loop — same pattern as IF's verify_all_instructions
    # and math's numeric match, but for code via subprocess test exec.
    #
    # Weights 0.34/0.33/0.33: equal-ish split. Prior R8-GRPO used 50/50 IF/math;
    # we back off IF because it's already saturated (reward > 0.95 at step 99) and
    # redirect that signal toward code, where HumanEval at 54% has headroom.
    #
    # Groups still never cross domains within a batch — the interleaver scheduler
    # produces single-domain groups so GRPO's per-group advantage centering stays
    # intact per domain.
    "r9_grpo_3way_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="role_colon",
        # set to R8-BoN final at invocation: tinker://e53abe63-.../weights/final
        load_checkpoint_path=None,
        log_path="logs/r9_grpo_3way_8b",
        lora_rank=32,
        learning_rate=1.5e-5,
        kl_penalty_coef=0.05,
        group_size=8,
        groups_per_batch=48,  # 16 groups/domain × 3 domains. Down from 64 to keep
                              # wall-clock similar despite 3-domain reward compute.
        max_tokens=1024,
        temperature=1.0,
        save_every=15,
        eval_every=15,
        max_steps=60,  # Shorter than R8's 100 — R8 plateaued around step 75 on
                      # training reward; cost cap prefers early stop.
        wandb_name="r9_grpo_3way_8b",
        mixed_weights=[0.34, 0.33, 0.33],  # [IF, math, code]
        # NOTE: without total_batches, MBPP (~120 decontam'd rows / 16 groups/batch
        # = 7 batches) exhausts and training stops early. R9 actual step count
        # was 7. To force longer training, set total_batches below.
    ),
    # R10: same mix as R9 but explicitly cycles the MBPP source by setting
    # total_batches=60. Each MBPP prompt is visited ~8x but with different GRPO
    # rollouts each time — reward signal is not stale because the policy is
    # changing between visits.
    "r10_grpo_3way_cycle_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="role_colon",
        load_checkpoint_path=None,
        log_path="logs/r10_grpo_3way_cycle_8b",
        lora_rank=32,
        learning_rate=1.5e-5,
        kl_penalty_coef=0.05,
        group_size=8,
        groups_per_batch=48,
        max_tokens=1024,
        temperature=1.0,
        save_every=15,
        eval_every=15,
        max_steps=60,
        total_batches=60,  # forces cycling on the short source (MBPP)
        wandb_name="r10_grpo_3way_cycle_8b",
        mixed_weights=[0.34, 0.33, 0.33],
    ),
}


def build_config(name: str, load_checkpoint_path: str | None) -> Config:
    cfg = RUN_CONFIGS[name]
    if "mixed_weights" in cfg:
        # Mixed IF+math (+ optional code) via InterleavedRLDatasetBuilder. Each
        # source produces single-domain groups; the interleaver schedules them
        # by weight.
        if_builder = IFEvalRLDatasetBuilder(
            batch_size=cfg["groups_per_batch"],
            model_name_for_tokenizer=cfg["model_name"],
            renderer_name=cfg["renderer_name"],
            group_size=cfg["group_size"],
            seed=0,
        )
        math_builder = MathRLDatasetBuilder(
            batch_size=cfg["groups_per_batch"],
            model_name_for_tokenizer=cfg["model_name"],
            renderer_name=cfg["renderer_name"],
            group_size=cfg["group_size"],
            seed=0,
        )
        sources = [if_builder, math_builder]
        if len(cfg["mixed_weights"]) == 3:
            code_builder = CodeRLDatasetBuilder(
                batch_size=cfg["groups_per_batch"],
                model_name_for_tokenizer=cfg["model_name"],
                renderer_name=cfg["renderer_name"],
                group_size=cfg["group_size"],
                seed=0,
            )
            sources.append(code_builder)
        kwargs = dict(
            sources=sources,
            weights=cfg["mixed_weights"],
            groups_per_batch=cfg["groups_per_batch"],
        )
        # If `total_batches` is in the config, pass it through so the interleaver
        # cycles short sources (e.g. MBPP sanitized at ~120 rows) rather than
        # terminating at the smallest-source exhaustion point. Without this,
        # max_steps=60 is silently truncated to 7 when MBPP runs out.
        if cfg.get("total_batches"):
            kwargs["total_batches"] = cfg["total_batches"]
        dataset_builder = InterleavedRLDatasetBuilder(**kwargs)
    else:
        dataset_builder = IFEvalRLDatasetBuilder(
            batch_size=cfg["groups_per_batch"],
            model_name_for_tokenizer=cfg["model_name"],
            renderer_name=cfg["renderer_name"],
            group_size=cfg["group_size"],
            seed=0,
        )
    ckpt_path = load_checkpoint_path or cfg["load_checkpoint_path"]
    # KLReferenceConfig.load_checkpoint_path expects sampler_weights (inference-only path);
    # Config.load_checkpoint_path expects weights (training-state path). Derive sampler from state.
    ref_ckpt_path = ckpt_path.replace("/weights/", "/sampler_weights/")
    return Config(
        learning_rate=cfg["learning_rate"],
        dataset_builder=dataset_builder,
        model_name=cfg["model_name"],
        renderer_name=cfg["renderer_name"],
        lora_rank=cfg["lora_rank"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        log_path=cfg["log_path"],
        load_checkpoint_path=ckpt_path,
        kl_penalty_coef=cfg["kl_penalty_coef"],
        # KL reference = the same SFT checkpoint → penalize drift from the starting policy.
        kl_reference_config=KLReferenceConfig(
            base_model=cfg["model_name"],
            load_checkpoint_path=ref_ckpt_path,
        ),
        eval_every=cfg["eval_every"],
        save_every=cfg["save_every"],
        max_steps=cfg["max_steps"],
        wandb_name=cfg.get("wandb_name"),
        loss_fn="importance_sampling",
    )


def main_entry():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=list(RUN_CONFIGS))
    parser.add_argument(
        "--load_checkpoint_path",
        required=True,
        help="tinker:// URI of the SFT checkpoint to start RL from",
    )
    args = parser.parse_args()

    cfg = RUN_CONFIGS[args.config]
    log.info("Launching RL run: %s from %s", args.config, args.load_checkpoint_path)
    log.info("  LR=%s  KL=%s  group=%d  batch=%d  max_tokens=%d  steps=%d",
             cfg["learning_rate"], cfg["kl_penalty_coef"],
             cfg["group_size"], cfg["groups_per_batch"],
             cfg["max_tokens"], cfg["max_steps"])

    config = build_config(args.config, args.load_checkpoint_path)
    asyncio.run(main(config))


if __name__ == "__main__":
    main_entry()
