"""
Post-RL orchestrator. Runs AFTER an RL config
(e.g. `training.rl_train --config r9_grpo_3way_8b` or r10) finishes.
Picks the best checkpoint by training-reward heuristic and evaluates it end-to-end.

Heuristic: rank saved checkpoints (steps in checkpoints.jsonl) by a simple
weighted sum that tracks what we care about (IF + math + code in the same
relative weights as the scoring function):

    score = ifeval_reward / 0.45 + gsm8k_reward / 0.50 + mbpp_reward / 0.30

Training reward is NOT equal to eval accuracy, but it's the cheapest signal to
pick an initial candidate before paying for a full eval run.

Also runs an eval on the final checkpoint for comparison.

Usage:
    python -m training.post_r9
    python -m training.post_r9 --run r10_grpo_3way_cycle_8b --label r10
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
SUBMISSION_DIR = REPO / "evaluation"


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path} — run has not produced metrics yet.")
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_checkpoints(run_dir: Path) -> list[dict]:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path} — run saved no checkpoints yet.")
    out = []
    with open(path) as f:
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
    p.add_argument("--run", default="r9_grpo_3way_8b",
                   help="Run directory name (default: r9_grpo_3way_8b)")
    p.add_argument("--label", default="r9",
                   help="Label for submission files (default: r9)")
    p.add_argument("--prior-best-file", default="submission_r9.json",
                   help="Prior-best submission JSON to compare against")
    p.add_argument("--skip-final", action="store_true",
                   help="Only eval the best-by-reward checkpoint, not the final step.")
    args = p.parse_args()

    run_dir = REPO / "logs" / args.run
    metrics = load_metrics(run_dir)
    ckpts = load_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit("No checkpoints saved — nothing to eval.")

    log.info("Loaded %d metrics rows, %d checkpoints from %s", len(metrics), len(ckpts), run_dir)

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
        out_path = SUBMISSION_DIR / f"submission_{args.label}_{label}.json"
        sub = eval_checkpoint(sampler_path, out_path)
        if sub is None:
            continue
        norm = avg_norm(sub)
        results[label] = (norm, sampler_path, out_path)
        log.info("  %s -> avg_norm=%.4f", label, norm or 0.0)

    # Compare to prior best
    prior_path = SUBMISSION_DIR / args.prior_best_file
    prior_norm = None
    if prior_path.exists():
        prior_norm = avg_norm(json.load(open(prior_path)))
        log.info("Prior best (%s): avg_norm=%.4f", prior_path.name, prior_norm or 0.0)

    if not results:
        log.error("No candidates evaluated successfully")
        return

    best = max(results.items(), key=lambda kv: kv[1][0] or 0.0)
    best_label, (best_norm, best_path, best_sub_path) = best
    log.info("%s best candidate: %s avg_norm=%.4f", args.label.upper(), best_label, best_norm or 0.0)

    if prior_norm is None or (best_norm or 0.0) > prior_norm:
        target = SUBMISSION_DIR / "submission.json"
        data = json.load(open(best_sub_path))
        json.dump(data, open(target, "w"), indent=2)
        log.info("Updated %s with %s best checkpoint (%s)", target, args.label.upper(), best_label)
    else:
        log.info("%s did not beat prior best; keeping existing submission.json", args.label.upper())


if __name__ == "__main__":
    main()
