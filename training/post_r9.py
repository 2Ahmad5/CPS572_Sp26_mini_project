"""
Post-R9 orchestrator. Runs AFTER `training.rl_train --config r9_grpo_3way_8b`
finishes. Picks the best R9 checkpoint by training-reward heuristic and
evaluates it end-to-end.

Heuristic: rank saved checkpoints (steps in checkpoints.jsonl) by a simple
weighted sum that tracks what we care about (IF + math + code in the same
relative weights as the scoring function):

    score = ifeval_reward / 0.45 + gsm8k_reward / 0.50 + mbpp_reward / 0.30

Training reward is NOT equal to eval accuracy, but it's the cheapest signal to
pick an initial candidate before paying for a full eval run.

Also runs an eval on the final checkpoint for comparison.

Usage:
    python -m training.post_r9
    python -m training.post_r9 --skip-final
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("post_r9")

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "logs" / "r9_grpo_3way_8b" / "metrics.jsonl"
CHECKPOINTS = REPO / "logs" / "r9_grpo_3way_8b" / "checkpoints.jsonl"
SUBMISSION_DIR = REPO / "evaluation"


def load_metrics() -> list[dict]:
    if not METRICS.exists():
        raise SystemExit(f"Missing {METRICS} — R9 has not produced metrics yet.")
    out = []
    with open(METRICS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_checkpoints() -> list[dict]:
    if not CHECKPOINTS.exists():
        raise SystemExit(f"Missing {CHECKPOINTS} — R9 saved no checkpoints yet.")
    out = []
    with open(CHECKPOINTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def best_by_reward(metrics: list[dict], ckpt_steps: list[int]) -> tuple[int, float]:
    """Return (step, score) for the saved checkpoint with highest weighted reward."""
    by_step = {m["step"]: m for m in metrics if m.get("step") is not None}
    best = None
    for step in ckpt_steps:
        m = by_step.get(step) or by_step.get(step - 1) or by_step.get(step + 1)
        if m is None:
            continue
        ifval = m.get("env/ifeval_rl/reward/total") or 0.0
        gsm = m.get("env/gsm8k_rl/reward/total") or 0.0
        code = m.get("env/mbpp_rl/reward/total") or 0.0
        score = ifval / 0.45 + gsm / 0.50 + code / 0.30
        log.info("step=%d  IF=%.3f  GSM=%.3f  CODE=%.3f  weighted=%.3f", step, ifval, gsm, code, score)
        if best is None or score > best[1]:
            best = (step, score)
    return best if best else (-1, 0.0)


def eval_checkpoint(sampler_path: str, out_path: Path) -> dict | None:
    log.info("Evaluating %s -> %s", sampler_path, out_path)
    cmd = [
        sys.executable, "-m", "evaluation.eval_all",
        "--checkpoint_path", sampler_path,
        "--base_model", "meta-llama/Llama-3.1-8B",
        "--output_path", str(out_path),
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Eval failed: %s", result.stderr[-500:])
        return None
    if not out_path.exists():
        log.error("Eval produced no output file")
        return None
    return json.load(open(out_path))


def avg_norm(sub: dict) -> float | None:
    try:
        ifval = sub["ifeval"]["metrics"]["google/IFEval/final_acc"]
        gsm = sub["gsm8k"]["metrics"]["openai/gsm8k/accuracy"]
        hev = sub["humaneval"]["metrics"]["openai/openai_humaneval/accuracy"]
    except (KeyError, TypeError):
        return None
    return (ifval / 0.45 + gsm / 0.50 + hev / 0.30) / 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-final", action="store_true",
                   help="Only eval the best-by-reward checkpoint, not the final step.")
    args = p.parse_args()

    metrics = load_metrics()
    ckpts = load_checkpoints()
    if not ckpts:
        raise SystemExit("No checkpoints saved — nothing to eval.")

    log.info("Loaded %d metrics rows, %d checkpoints", len(metrics), len(ckpts))

    by_name = {c["name"]: c for c in ckpts}
    non_final = [c for c in ckpts if c.get("name") != "final"]
    steps = [c["batch"] for c in non_final]
    best_step, best_score = best_by_reward(metrics, steps)

    candidates: list[tuple[str, str]] = []  # (label, sampler_path)
    if best_step > 0:
        matched = next((c for c in non_final if c["batch"] == best_step), None)
        if matched:
            candidates.append((f"step{best_step:03d}_bestreward", matched["sampler_path"]))
    if not args.skip_final and "final" in by_name:
        candidates.append(("final", by_name["final"]["sampler_path"]))

    results = {}
    for label, sampler_path in candidates:
        out_path = SUBMISSION_DIR / f"submission_r9_{label}.json"
        sub = eval_checkpoint(sampler_path, out_path)
        if sub is None:
            continue
        norm = avg_norm(sub)
        results[label] = (norm, sampler_path, out_path)
        log.info("  %s -> avg_norm=%.4f", label, norm or 0.0)

    # Compare to prior best
    r8_bon_path = SUBMISSION_DIR / "submission_r8_bon.json"
    r8_bon_norm = None
    if r8_bon_path.exists():
        r8_bon_norm = avg_norm(json.load(open(r8_bon_path)))
        log.info("R8-BoN (prior best): avg_norm=%.4f", r8_bon_norm or 0.0)

    if not results:
        log.error("No candidates evaluated successfully")
        return

    best = max(results.items(), key=lambda kv: kv[1][0] or 0.0)
    best_label, (best_norm, best_path, best_sub_path) = best
    log.info("R9 best candidate: %s avg_norm=%.4f", best_label, best_norm or 0.0)

    if r8_bon_norm is None or (best_norm or 0.0) > r8_bon_norm:
        # Update submission.json
        target = SUBMISSION_DIR / "submission.json"
        data = json.load(open(best_sub_path))
        json.dump(data, open(target, "w"), indent=2)
        log.info("Updated %s with R9 best checkpoint (%s)", target, best_label)
    else:
        log.info("R9 did not beat R8-BoN; keeping existing submission.json")


if __name__ == "__main__":
    main()
