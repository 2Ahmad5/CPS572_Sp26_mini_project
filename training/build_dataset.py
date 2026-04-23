"""
Build the training JSONL for a given run config.

Usage:
    python -m training.build_dataset --config r0_smoke --output training/data/r0_smoke.jsonl
    python -m training.build_dataset --config r1_kitchen_sink --output training/data/r1_kitchen_sink.jsonl

Pipeline:
    1. Build 13-gram deny set from held-out eval benchmarks.
    2. For each source in data_sources.SOURCES:
       - load_dataset(...).shuffle(seed).select(first oversample*target rows)
       - run per-source adapter (None → drop)
       - contamination filter
       - length filter (char pre-filter → exact tokenization in ambiguous zone)
    3. Cap each source at its configured target count.
    4. Concat + shuffle globally.
    5. Write JSONL with one {"messages": [...]} per line.
    6. Print source-by-source stats and 3 sample rows for eyeballing.
"""

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from training.data_sources import SOURCES, build_deny_ngrams, row_contaminated

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("build_dataset")


CONFIGS: dict[str, dict[str, int]] = {
    "r0_smoke": {
        "openai/gsm8k": 300,
        "nvidia/OpenMathInstruct-2": 300,
        "meta-math/MetaMathQA": 300,
        "allenai/tulu-3-sft-personas-math-grade": 300,
        "ise-uiuc/Magicoder-Evol-Instruct-110K": 300,
        "allenai/tulu-3-sft-personas-code": 200,
        "allenai/tulu-3-sft-personas-instruction-following": 500,
    },
    "r1_kitchen_sink": {
        # Math bucket
        "openai/gsm8k": 7_473,
        "nvidia/OpenMathInstruct-2": 100_000,
        "meta-math/MetaMathQA": 50_000,
        "allenai/tulu-3-sft-personas-math-grade": 50_000,
        # Code bucket
        "ise-uiuc/Magicoder-Evol-Instruct-110K": 45_000,
        "allenai/tulu-3-sft-personas-code": 30_000,
        # Instruction-following bucket
        "allenai/tulu-3-sft-personas-instruction-following": 30_000,
    },
    # Enhancements vs r1_kitchen_sink (run this with --max_len_tokens 2048):
    #   - Add bigcode/self-oss-instruct-sc2-exec-filter-50k (+30K execution-validated Python for HumanEval lift)
    #   - The 2048-token cap forces shorter CoT rationales that fit the eval's max_tokens=1024 budget
    "r1_enhanced": {
        # Math bucket
        "openai/gsm8k": 7_473,
        "nvidia/OpenMathInstruct-2": 100_000,
        "meta-math/MetaMathQA": 50_000,
        "allenai/tulu-3-sft-personas-math-grade": 50_000,
        # Code bucket (with bigcode added)
        "ise-uiuc/Magicoder-Evol-Instruct-110K": 45_000,
        "allenai/tulu-3-sft-personas-code": 30_000,
        "bigcode/self-oss-instruct-sc2-exec-filter-50k": 30_000,
        # IF bucket
        "allenai/tulu-3-sft-personas-instruction-following": 30_000,
    },
    # Same mix as r1_enhanced but with Magicoder-Evol-Instruct replaced by
    # more bigcode + tulu-code. Use this for strict-decontam rebuild; see
    # training/data/contamination_report_r1.md for rationale.
    "r1_clean": {
        # Math bucket
        "openai/gsm8k": 7_473,
        "nvidia/OpenMathInstruct-2": 100_000,
        "meta-math/MetaMathQA": 50_000,
        "allenai/tulu-3-sft-personas-math-grade": 50_000,
        # Code bucket (Magicoder-Evol-Instruct removed; compensating with more
        # bigcode + tulu-code to keep code share roughly comparable).
        "allenai/tulu-3-sft-personas-code": 45_000,  # was 30K
        "bigcode/self-oss-instruct-sc2-exec-filter-50k": 45_000,  # was 30K
        # IF bucket
        "allenai/tulu-3-sft-personas-instruction-following": 30_000,
    },
}

MAX_LEN_TOKENS = 3072
MODEL_FOR_TOKENIZER = "meta-llama/Llama-3.2-3B"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len_tokens", type=int, default=MAX_LEN_TOKENS)
    parser.add_argument(
        "--max_seen_factor",
        type=float,
        default=5.0,
        help="Safety cap: stop streaming a source after max_seen_factor*target rows even if target not hit",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=10_000,
        help="Streaming shuffle buffer size. Bigger = more random but more memory.",
    )
    args = parser.parse_args()

    targets = CONFIGS[args.config]

    log.info("Building deny-set from eval benchmarks ...")
    deny = build_deny_ngrams()

    log.info("Loading tokenizer + renderer (%s) ...", MODEL_FOR_TOKENIZER)
    tokenizer = get_tokenizer(MODEL_FOR_TOKENIZER)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_FOR_TOKENIZER)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    train_on_what = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    max_tokens = args.max_len_tokens

    def is_too_long(messages: list[dict]) -> bool:
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars < max_tokens * 2:
            return False
        if total_chars > max_tokens * 8:
            return True
        model_input, _ = renderer.build_supervised_example(messages, train_on_what=train_on_what)
        return len(model_input.to_ints()) > max_tokens

    all_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for path, name, split, adapter, bucket in SOURCES:
        target = targets.get(path, 0)
        if target == 0:
            continue
        log.info("→ %s  [target %d, streaming]", path, target)
        ds = load_dataset(path, name=name, split=split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

        max_seen = int(target * args.max_seen_factor)
        kept, n_seen, n_adapt_none, n_contam, n_long = [], 0, 0, 0, 0
        for row in ds:
            n_seen += 1
            adapted = adapter(row)
            if adapted is None:
                n_adapt_none += 1
            elif row_contaminated(adapted["messages"], deny):
                n_contam += 1
            elif is_too_long(adapted["messages"]):
                n_long += 1
            else:
                kept.append(adapted)
            if len(kept) >= target or n_seen >= max_seen:
                break

        stats[path] = {
            "bucket": bucket,
            "seen": n_seen,
            "dropped_adapter": n_adapt_none,
            "dropped_contam": n_contam,
            "dropped_too_long": n_long,
            "kept": len(kept),
            "target": target,
        }
        log.info(
            "   seen=%d  kept=%d/%d   (adapter-None=%d  contam=%d  too_long=%d)",
            n_seen,
            len(kept),
            target,
            n_adapt_none,
            n_contam,
            n_long,
        )
        all_rows.extend(kept)

    random.Random(args.seed).shuffle(all_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    bucket_totals: dict[str, int] = {}
    for s in stats.values():
        bucket_totals[s["bucket"]] = bucket_totals.get(s["bucket"], 0) + s["kept"]
    total = sum(bucket_totals.values())

    log.info("=" * 60)
    log.info("Wrote %d rows → %s", len(all_rows), out)
    log.info("Per-source stats:")
    for path, s in stats.items():
        log.info("  %s: %s", path, s)
    log.info("Bucket totals (share of final mix):")
    for b, n in bucket_totals.items():
        log.info("  %s: %d  (%.1f%%)", b, n, 100.0 * n / total if total else 0.0)

    # Eyeball-check: print 3 random samples
    log.info("=" * 60)
    log.info("Sample rows (random 3):")
    for row in random.Random(args.seed + 1).sample(all_rows, min(3, len(all_rows))):
        log.info("---")
        for m in row["messages"]:
            content = m["content"]
            preview = content if len(content) < 400 else content[:300] + " ... " + content[-80:]
            log.info("  [%s] %s", m["role"], preview)


if __name__ == "__main__":
    main()
