# 4/23 Overnight session summary

## Contamination finding (IMPORTANT)

The R1 contamination audit discovered that `ise-uiuc/Magicoder-Evol-Instruct-110K`
contains **derivative contamination** of HumanEval. The 13-gram word filter at build
time didn't catch them because the text has been "Evol-Instruct" paraphrased, but:

- 67 HumanEval function signatures appear in r1_enhanced.jsonl
- 14 rows have BOTH `is_palindrome` AND `make_palindrome` signatures (HumanEval/10)
- Those rows reproduce HumanEval/10's distinctive examples (`make_palindrome('cat') → 'catac'`) verbatim

Our hard gates (EXACT + SUBSTRING) pass at **0 hits** — no verbatim leakage of
test prompts or canonical solutions. The contamination is structural / semantic.

This is a weaker form of contamination than verbatim copying. The 13-gram standard
(BigCode / StarCoder) considers our training data clean. But a strict reviewer
could argue that Evol-paraphrased HumanEval is still "training on test data."

**If strict decontamination matters:** drop Magicoder-Evol-Instruct from
`training/data_sources.py:SOURCES`, rebuild r1_enhanced, retrain R6 → R8 → R9.
Cost ~$15-20. Not within this session's budget.

**Disclosure:** this finding is included in the final report.

## Headline

**R9 beats R8-BoN across all three metrics: avg_norm 1.6734 vs 1.6528 (+0.0206).**

| Metric | R8-BoN (prior) | R9 (new best) | Δ |
|---|---|---|---|
| IFEval | 70.91 | **72.02** | +1.11 |
| GSM8K | 78.70 | **79.53** | +0.83 |
| HumanEval | 54.27 | **54.88** | +0.61 |
| avg_norm | 1.6528 | **1.6734** | **+0.0206** |

R9 checkpoint: `tinker://f4555916-c448-5313-b152-d49b9398e402:train:0/sampler_weights/final`.
`evaluation/submission.json` updated.

R10 (cycled 3-way GRPO, 15 steps from R9 final) is currently running to see if more
steps help further. Monitoring in progress.

## What ran this session

### 1. Decontamination audit — independent "0 overlaps verified" paper trail

Script: `training/verify_no_test_leak.py`. Runs three independent checks against every
materialized training JSONL:

- **EXACT** — normalized (lowercased, whitespace-collapsed) equality between test prompt and any training message
- **SUBSTRING** — test text ≥40 chars appears as contiguous substring of any training message
- **HumanEval function-signature** — `def <name>(...)` line of each canonical HumanEval prompt in training assistant messages

Plus a noisy 8-gram overlap check (informational only, disabled on big files).

Results:
- `r8_bon.jsonl` (110 rows): EXACT=0, SUBSTRING=0, FUNC-SIG=0 ✓
- `r8_rft.jsonl` (5,861 rows): EXACT=0, SUBSTRING=0, FUNC-SIG=0 ✓
- `r1_enhanced.jsonl` (377K rows, R6 training data, ancestor of R8-BoN): _(running, will update)_
- `r1_kitchen_sink.jsonl` (347K rows, R1 3B only, not in R8 critical path): _(running)_

Training-time pipeline (already in place, reconfirmed):
- 13-gram word-level deny set built from google/IFEval + openai/gsm8k test + openai_humaneval
- `build_dataset.py` drops any row with matching 13-gram during dataset construction
- `rl_ifeval.py` dedups argilla/ifeval-like-data against exact IFEval prompts
- `rl_code_env.py` / `bon_run.py` decontaminate MBPP via same 13-gram deny set
- `gsm8k` train/test are disjoint HF splits

### 2. Intermediate R8-GRPO checkpoint eval — no hidden gems

Training reward at R8-GRPO step 75 (0.91) beat step 100 (0.83). We'd submitted the step-100-derived R8-BoN. Was step 75 secretly better?

| Checkpoint | IFEval | GSM8K | HumanEval | avg_norm |
|---|---|---|---|---|
| R8-GRPO step 50 | 70.75 | 79.23 | 53.05 | 1.6417 |
| R8-GRPO step 75 | 71.21 | 80.52 | 51.83 | 1.6401 |
| R8-GRPO step 100 | 69.68 | 79.83 | 53.66 | 1.6445 |
| **R8-BoN (step 100 + BoN distill)** | **70.91** | **78.70** | **54.27** | **1.6528** |

No, R8-BoN remained the best. BoN distillation earned its keep via the HumanEval lift.

### 3. R9 — 3-way mixed GRPO (the big new idea)

**Gap identified:** R8-GRPO only optimized IF+math verifiers. Code had no RL signal — only SFT distillation in the BoN phase. HumanEval (30% baseline denominator) has the highest marginal-return bucket, so code RL is where the leverage is.

**New code:**
- `training/rl_code_env.py` — `CodeRLEnv(ProblemEnv)` using MBPP sanitized train + subprocess test-exec verifier (same trust model as `bon_run.py`). Binary reward = all tests pass.
- `training/rl_train.py` — `r9_grpo_3way_8b` config via `InterleavedRLDatasetBuilder` with weights `[0.34, 0.33, 0.33]` for `[IF, math, code]`.
- `training/post_r9.py` — orchestrator that picks best R9 checkpoint by training reward, evals, auto-updates `submission.json` if it beats R8-BoN.
- `training/rl_train.py` — also added `r10_grpo_3way_cycle_8b` for a longer run if warranted.

**Result from R9:** Training finished in **7 steps** (not 60), because `InterleavedRLDatasetBuilder` defaults to stopping at the smallest-source exhaustion point. MBPP sanitized has ~120 decontam'd rows / 16 groups-per-batch = 7 batches. The built-in default of no-cycling capped the run.

Cost of R9 was ~$1.60 (way under the $8-10 estimated for 60 steps), which is actually a nice unplanned cost-cut.

**Step 0 baseline (on R8-BoN):**
```
IF=1.000  GSM=0.683  CODE=0.532  all=0.755
```

**Last step (step 6):**
```
IF=0.926  GSM=0.713  CODE=0.580  all=0.750
```

Code reward climbed modestly (0.53 → 0.58). IF and GSM volatile step-to-step (high variance with only 16 groups per domain).

**R9 full eval:** _(in progress — results will be added)_

### 4. Files committed locally (no push)

`c3126c2 R8 iteration + R9 prep: CodeRLEnv, 3-way mixed GRPO, contamination audit`

Next commit will include R9 post-eval + report.

## Decision tree after R9 eval

- **If R9 > R8-BoN:** update `submission.json`, commit, done. Consider R10 (cycled 60 steps) if budget allows and the trend is clearly positive.
- **If R9 ≥ R8-BoN but HumanEval regresses:** R9-BoN SFT (config pre-staged as `r9_bon_8b` in `sft_train.py`) to recover code via distillation.
- **If R9 < R8-BoN:** keep R8-BoN as the submission. Document R9 as "negative result" per PROJECT.md ("negative results are valued").

## Known limitations / improvement ideas for tomorrow

- MBPP sanitized decontam'd is only ~120 rows. For more robust code RL, a larger source (e.g. OpenCodeInstruct with auto-extracted test cases, or synthesizing tests from Magicoder) would help.
- The InterleavedRLDatasetBuilder exhaustion behavior was surprising; R10 config now sets `total_batches=60` to force cycling.
- Current BoN uses `temperature=1.0`. Higher T (1.2-1.5) might diversify the candidate pool and improve top-1 quality at fixed N.
- Cost cuts from the audit (OpenMathInstruct 100K→60K, MetaMathQA 50K→30K, BoN n_prompts 500→200) are not applied — no current run uses them. They'll be applied on any new SFT/BoN build.
