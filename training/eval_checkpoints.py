"""
Evaluate every saved checkpoint from an SFT run and pick the best one.

Reads ``<log_path>/checkpoints.jsonl`` (written by the cookbook's train.main),
invokes ``evaluation/eval_all.py`` for each ``sampler_path``, and prints a
comparison table of IFEval / GSM8K / HumanEval across all checkpoints.

Usage:
    python -m training.eval_checkpoints --log_path logs/r0_smoke --base_model meta-llama/Llama-3.2-3B --limit 40
    python -m training.eval_checkpoints --log_path logs/r1_kitchen_sink_3b --base_model meta-llama/Llama-3.2-3B

The ``--limit`` arg forwards to eval_all.py; set it low (e.g. 40-200) for quick
intermediate evaluation, then run at full scale on the winning checkpoint.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("eval_checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task (None=full)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="Output JSON with per-checkpoint scores")
    args = parser.parse_args()

    ckpt_file = Path(args.log_path) / "checkpoints.jsonl"
    if not ckpt_file.exists():
        log.error("No checkpoints.jsonl at %s", ckpt_file)
        sys.exit(1)

    records = [json.loads(line) for line in ckpt_file.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("sampler_path")]
    log.info("Found %d checkpoint(s) with sampler_path in %s", len(records), ckpt_file)

    results = []
    for i, r in enumerate(records):
        sp = r["sampler_path"]
        step = r.get("batch")
        log.info("[%d/%d] evaluating step=%s  sampler_path=%s", i + 1, len(records), step, sp)

        cmd = [
            sys.executable,
            "-m", "evaluation.eval_all",
            "--checkpoint_path", sp,
            "--base_model", args.base_model,
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--output_path", f"evaluation/submission_step{step or i}.json",
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        rc = subprocess.call(cmd)
        if rc != 0:
            log.warning("eval_all.py exited with code %d for step=%s", rc, step)
            continue

        # Read back the per-step submission
        sub_path = Path(f"evaluation/submission_step{step or i}.json")
        if not sub_path.exists():
            continue
        sub = json.loads(sub_path.read_text())

        # Extract summary metrics
        metrics = {}
        for task in ("ifeval", "gsm8k", "humaneval"):
            m = sub.get(task, {}).get("metrics", {})
            metrics[task] = m
        results.append({
            "step": step,
            "sampler_path": sp,
            "metrics": metrics,
        })

    # Print comparison
    log.info("=" * 80)
    log.info("CHECKPOINT COMPARISON")
    log.info("=" * 80)
    for r in results:
        # Pull headline numbers
        ifeval = _best_ifeval_score(r["metrics"].get("ifeval", {}))
        gsm = _pick(r["metrics"].get("gsm8k", {}), "accuracy")
        he = _pick(r["metrics"].get("humaneval", {}), "pass@1") or _pick(r["metrics"].get("humaneval", {}), "mean")
        log.info(
            "step=%4s  IFEval=%s  GSM8K=%s  HumanEval=%s  avg_norm=%s",
            r["step"],
            _fmt(ifeval), _fmt(gsm), _fmt(he),
            _fmt(_avg_norm(ifeval, gsm, he)),
        )

    # Winner
    def avg_norm_of(r):
        m = r["metrics"]
        ifeval = _best_ifeval_score(m.get("ifeval", {}))
        gsm = _pick(m.get("gsm8k", {}), "accuracy")
        he = _pick(m.get("humaneval", {}), "pass@1") or _pick(m.get("humaneval", {}), "mean")
        return _avg_norm(ifeval, gsm, he) or -1

    if results:
        winner = max(results, key=avg_norm_of)
        log.info("=" * 80)
        log.info("WINNER: step=%s  sampler_path=%s  avg_norm=%s",
                 winner["step"], winner["sampler_path"], _fmt(avg_norm_of(winner)))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log.info("Per-checkpoint results written to %s", args.output)


def _pick(d: dict, key_substr: str):
    for k, v in d.items():
        if key_substr in k and isinstance(v, (int, float)):
            return v
    return None


def _best_ifeval_score(d: dict):
    """Average IFEval's 4 sub-metrics (strict×loose, prompt×inst) to match PROJECT.md baseline definition."""
    keys = ["prompt_level_strict_acc", "prompt_level_loose_acc",
            "inst_level_strict_acc", "inst_level_loose_acc"]
    vals = []
    for kwant in keys:
        for k, v in d.items():
            if kwant in k and isinstance(v, (int, float)):
                vals.append(v)
                break
    if len(vals) == 4:
        return sum(vals) / 4
    return _pick(d, "accuracy")


def _avg_norm(ifeval, gsm, he):
    """Normalize each metric by its baseline and average."""
    if None in (ifeval, gsm, he):
        return None
    return (ifeval / 0.45 + gsm / 0.50 + he / 0.30) / 3


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    main()
