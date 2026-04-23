"""
Run SFT via the tinker-cookbook supervised training loop.

Usage:
    python -m training.sft_train --config r0_smoke
    python -m training.sft_train --config r1_kitchen_sink_3b
    python -m training.sft_train --config r6_kitchen_sink_8b

Each config is a small dict below. The actual heavy lifting (pipelining, LR
schedule, checkpointing, NLL eval, resume) is delegated to
``tinker_cookbook.supervised.train.main``. We don't wrap or catch exceptions
beyond what the cookbook already does — if something breaks, the stack trace
points at the real problem.

After training completes, saved checkpoints are listed with their
``sampler_path`` URIs, which can be passed to ``evaluation/eval_all.py``.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads TINKER_API_KEY (and optionally HF_TOKEN) from .env if present

from tinker_cookbook import model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sft_train")


# Each run config carries everything needed to construct a train.Config.
# LR values come from Tinker's formula:
#   lr = 5e-5 * 10 * (2000/hidden_size) ** 0.781
# Llama-3.2-3B hidden=3072 → 3.56e-4; Llama-3.1-8B hidden=4096 → 2.83e-4.

RUN_CONFIGS: dict[str, dict] = {
    "r0_smoke": dict(
        model_name="meta-llama/Llama-3.2-3B",
        data_file="training/data/r0_smoke.jsonl",
        log_path="logs/r0_smoke",
        learning_rate=3.6e-4,
        lora_rank=16,
        batch_size=32,
        max_length=2048,
        num_epochs=1,
        save_every=50,
        eval_every=25,
        infrequent_eval_every=0,
        test_size=150,
        wandb_name="r0_smoke",
    ),
    "r1_kitchen_sink_3b": dict(
        model_name="meta-llama/Llama-3.2-3B",
        data_file="training/data/r1_kitchen_sink.jsonl",
        log_path="logs/r1_kitchen_sink_3b",
        learning_rate=3.6e-4,
        lora_rank=32,
        batch_size=128,
        max_length=3072,
        num_epochs=1,
        save_every=1000,
        eval_every=500,
        infrequent_eval_every=0,
        test_size=500,
        wandb_name="r1_kitchen_sink_3b",
    ),
    "r6_kitchen_sink_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        data_file="training/data/r1_kitchen_sink.jsonl",  # same data as R1
        log_path="logs/r6_kitchen_sink_8b",
        learning_rate=2.83e-4,
        lora_rank=32,
        batch_size=128,
        max_length=3072,
        num_epochs=1,
        save_every=1000,
        eval_every=500,
        infrequent_eval_every=0,
        test_size=500,
        wandb_name="r6_kitchen_sink_8b",
    ),
    "r6_enhanced_8b": dict(
        # Uses r1_enhanced data (bigcode added, 2048-token filter)
        model_name="meta-llama/Llama-3.1-8B",
        data_file="training/data/r1_enhanced.jsonl",
        log_path="logs/r6_enhanced_8b",
        learning_rate=2.83e-4,
        lora_rank=32,
        batch_size=128,
        max_length=2048,  # tighter than R1's 3072 → shorter CoT, fits eval's 1024 max_tokens
        num_epochs=1,
        save_every=1000,
        eval_every=500,
        infrequent_eval_every=0,
        test_size=500,
        wandb_name="r6_enhanced_8b",
    ),
    "r8_rft_8b": dict(
        # RFT phase: second SFT pass on verifier-filtered rollouts from R6.
        # Lower LR (5e-5) prevents drift from R6's established capabilities.
        # Dataset is ~5.9K rows → ~46 steps at batch 128, so save/eval every 25.
        model_name="meta-llama/Llama-3.1-8B",
        data_file="training/data/r8_rft.jsonl",
        log_path="logs/r8_rft_8b",
        learning_rate=5e-5,
        lora_rank=32,
        batch_size=128,
        max_length=2048,
        num_epochs=1,
        save_every=25,
        eval_every=25,
        infrequent_eval_every=0,
        test_size=200,
        wandb_name="r8_rft_8b",
        load_checkpoint_path="tinker://588e8de1-f856-5f47-bf64-c5be7e8bf807:train:0/weights/final",
    ),
    "r8_bon_8b": dict(
        # Best-of-N code distillation: SFT on subprocess-verified top-1 rollouts from MBPP.
        # Starts from R8-GRPO (best so far at 1.645 avg_norm) to compound gains.
        # Dataset is ~110 rows, so 3 epochs × batch 16 ≈ 20 steps of concentrated code training.
        model_name="meta-llama/Llama-3.1-8B",
        data_file="training/data/r8_bon.jsonl",
        log_path="logs/r8_bon_8b",
        learning_rate=5e-5,
        lora_rank=32,
        batch_size=16,
        max_length=2048,
        num_epochs=3,
        save_every=10,
        eval_every=10,
        infrequent_eval_every=0,
        test_size=10,
        wandb_name="r8_bon_8b",
        load_checkpoint_path="tinker://d8ff2c4d-59b6-589a-b8ce-b258ce3a2d97:train:0/weights/final",
    ),
    # R9-BoN: same recipe as R8-BoN but seeded from R9 (3-way GRPO) best. Kept
    # small (3 epochs, batch 16) because the dataset is tiny (~100-150 MBPP rows
    # that passed the verifier). If R9 ate into HumanEval via RL, BoN recovers
    # it by SFT'ing on known-correct code outputs.
    # data_file + load_checkpoint_path overridable via CLI at launch time.
    "r9_bon_8b": dict(
        model_name="meta-llama/Llama-3.1-8B",
        data_file="training/data/r9_bon.jsonl",
        log_path="logs/r9_bon_8b",
        learning_rate=5e-5,
        lora_rank=32,
        batch_size=16,
        max_length=2048,
        num_epochs=3,
        save_every=10,
        eval_every=10,
        infrequent_eval_every=0,
        test_size=10,
        wandb_name="r9_bon_8b",
        load_checkpoint_path=None,  # set at launch to R9 best
    ),
}


def build_config(name: str) -> train.Config:
    cfg = RUN_CONFIGS[name]
    renderer_name = model_info.get_recommended_renderer_name(cfg["model_name"])

    dataset_builder = FromConversationFileBuilder(
        file_path=cfg["data_file"],
        test_size=cfg["test_size"],
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cfg["model_name"],
            renderer_name=renderer_name,
            max_length=cfg["max_length"],
            batch_size=cfg["batch_size"],
        ),
    )

    return train.Config(
        log_path=cfg["log_path"],
        model_name=cfg["model_name"],
        dataset_builder=dataset_builder,
        learning_rate=cfg["learning_rate"],
        lr_schedule="linear",
        num_epochs=cfg["num_epochs"],
        lora_rank=cfg["lora_rank"],
        save_every=cfg["save_every"],
        eval_every=cfg["eval_every"],
        infrequent_eval_every=cfg["infrequent_eval_every"],
        ttl_seconds=7 * 24 * 3600,  # 7 days — submission is ~1 week out, no need for 30d
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        load_checkpoint_path=cfg.get("load_checkpoint_path"),
        wandb_project=cfg.get("wandb_project"),
        wandb_name=cfg.get("wandb_name"),
    )


def print_checkpoint_list(log_path: str, model_name: str) -> None:
    p = Path(log_path) / "checkpoints.jsonl"
    if not p.exists():
        log.warning("No checkpoints.jsonl at %s — nothing to publish.", p)
        return
    records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    log.info("=" * 60)
    log.info("Saved %d checkpoint(s):", len(records))
    for r in records:
        log.info(
            "  step=%s  final=%s  sampler_path=%s",
            r.get("batch"),
            r.get("final"),
            r.get("sampler_path"),
        )
    log.info("=" * 60)
    log.info("To evaluate the final checkpoint:")
    final = next((r for r in records if r.get("final")), records[-1])
    if final.get("sampler_path"):
        log.info(
            '  python evaluation/eval_all.py --checkpoint_path "%s" --base_model %s',
            final["sampler_path"],
            model_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=list(RUN_CONFIGS))
    parser.add_argument("--load_checkpoint_path", default=None,
                        help="Override load_checkpoint_path for configs like r9_bon_8b.")
    parser.add_argument("--data_file", default=None,
                        help="Override data_file for configs that build data at launch time.")
    args = parser.parse_args()

    cfg = RUN_CONFIGS[args.config]
    if args.load_checkpoint_path is not None:
        cfg["load_checkpoint_path"] = args.load_checkpoint_path
    if args.data_file is not None:
        cfg["data_file"] = args.data_file
    log.info("Launching run: %s", args.config)
    log.info("  model=%s  data=%s  log=%s", cfg["model_name"], cfg["data_file"], cfg["log_path"])
    log.info("  lr=%s  rank=%d  batch=%d  max_length=%d  epochs=%d",
             cfg["learning_rate"], cfg["lora_rank"], cfg["batch_size"],
             cfg["max_length"], cfg["num_epochs"])

    config = build_config(args.config)
    asyncio.run(train.main(config))

    print_checkpoint_list(cfg["log_path"], cfg["model_name"])


if __name__ == "__main__":
    main()
