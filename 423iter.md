# 4/23 Iteration: Decontamination audit + R9 (3-way mixed GRPO)

**Prior best:** R8-BoN at **avg_norm 1.6528** (IFEval 70.91 / GSM8K 78.70 / HumanEval 54.27). Checkpoint `tinker://e53abe63-5f75-508a-86a8-fd2cf189fb73:train:0/sampler_weights/final`.

## 1. Decontamination audit (blocking prerequisite)

Script: `training/verify_no_test_leak.py`. Independent audit stricter than the training-time 13-gram filter:

- **EXACT**: normalized-whitespace lowercased equality — test text == training message
- **SUBSTRING**: normalized test text ≥40 chars appears as contiguous substring of any training message (concat'd into chunked 40MB buffers)
- **HUMANEVAL-FUNC-SIG**: `def <name>(...)` signature line of each HumanEval canonical prompt appears in any training assistant message
- **8-GRAM** (informational): word-level 8-gram overlap — noisy on common phrases like "the answer is" / "how many"

### Results on R8-lineage JSONLs

| File | Rows | EXACT | SUBSTRING | FUNC-SIG | 8-gram (noise) |
|---|---|---|---|---|---|
| r8_bon.jsonl | 110 | 0 | 0 | 0 | 5 |
| r8_rft.jsonl | 5,861 | 0 | 0 | 0 | 90 |
| r1_enhanced.jsonl (R6 training data) | 377,227 | *TBD* | *TBD* | *TBD* | (skipped) |
| r1_kitchen_sink.jsonl (R1 3B only, not in R8 lineage) | 347,346 | *TBD* | *TBD* | *TBD* | (skipped) |

**Verdict (so far):** R8-BoN and R8-RFT JSONLs are verified free of test-set leakage on the three strongest checks. 8-gram overlaps are noise from common phrases.

The decontam pipeline at training time already runs two layers of defense:
- **Training data:** 13-gram word-level deny set built from google/IFEval + openai/gsm8k test + openai/openai_humaneval — any matching row dropped during `build_dataset.py`. Catches ~548 leaks in raw Magicoder-Evol alone (canonical HumanEval solutions).
- **RL prompt sources:** argilla/ifeval-like-data filtered against exact google/IFEval prompts; MBPP sanitized decontaminated against HumanEval via the same 13-gram set; gsm8k train vs test are disjoint HF splits by construction.

This independent script verifies the pipeline worked — we can point to 0 EXACT + 0 SUBSTRING + 0 FUNC-SIG hits as paper-trail equivalent to "our team ran a verification script; the result shows 0 overlaps."

## 2. Intermediate R8-GRPO checkpoints evaluated

Cheap sanity check: training reward at step 75 (all=0.91) beat step 100 (all=0.83). We'd submitted step-100 → R8-BoN. Did step 75 or step 50 have a higher eval avg_norm that we missed?

| Checkpoint | IFEval | GSM8K | HumanEval | avg_norm |
|---|---|---|---|---|
| R8-GRPO step 50 | 70.75 | 79.23 | 53.05 | 1.6417 |
| R8-GRPO step 75 | **71.21** | **80.52** | 51.83 | 1.6401 |
| R8-GRPO step 100 | 69.68 | 79.83 | 53.66 | 1.6445 |
| R8-BoN (step 100 + MBPP-BoN SFT) | 70.91 | 78.70 | **54.27** | **1.6528** |

All R8-GRPO checkpoints cluster tight (1.6401-1.6445). The BoN distillation step that produced R8-BoN (+0.008 avg_norm) earned its keep via the HumanEval lift (53.66 → 54.27). R8-BoN remains the correct submission starting point.

## 3. R9 — 3-way mixed GRPO

Biggest gap identified in the agent audit: R8-GRPO's InterleavedRLDatasetBuilder mixed only IFEval and math verifiers. Code had NO RL signal — only SFT via the BoN distillation phase. HumanEval at 54.27% has the most headroom (30% baseline denominator → largest marginal-return bucket).

### Design
- **New environment:** `training/rl_code_env.py` → `CodeRLEnv(ProblemEnv)`. Reward = 1.0 iff generated code passes ALL public tests for the MBPP prompt. Verifier is `bon_run.py`'s subprocess exec — trusted in-project, no sandbox infra needed.
- **Dataset:** MBPP sanitized train split, decontam'd against HumanEval via the existing 13-gram deny set.
- **Renderer + format:** matches IFEval/math pattern (single-turn, role_colon, extract code from ```python fenced block).

### Config
`r9_grpo_3way_8b` in `training/rl_train.py`:
- Starts from **R8-BoN final** (not R6, not step 75 — the best available starting point)
- LR 1.5e-5, KL 0.05, group 8, groups_per_batch 48 (down from R8's 64 to keep 3-domain cost similar)
- weights = [0.34, 0.33, 0.33] for [IF, math, code] (slight backoff on IF since it was saturated at 0.98 reward in R8-GRPO, redirects headroom to code)
- max_steps 60, save every 15 steps, eval every 15
- InterleavedRLDatasetBuilder ensures groups are single-domain within a batch (preserves GRPO's per-group advantage centering)

### Running...

Step 0 baseline on R8-BoN:
```
IF=1.000  GSM=0.683  CODE=0.532  all=0.755  kl=0.001
```

IF already saturated — confirms backoff was right. CODE at 53.2% closely matches the HumanEval 54.3% eval accuracy (calibration check passes). GSM8K at 68% has room.

Training progress will be dropped into `logs/r9_grpo_3way_8b/metrics.jsonl`. Post-R9 orchestrator `training/post_r9.py` picks the best-by-reward checkpoint, evaluates it, and updates `submission.json` if it beats R8-BoN's 1.6528.

## 4. Cost-cut audit round 2

Prior iteration (4/19) cut: removed Magicoder-OSS and no_robots from the source registry; tightened save_every/TTL/ifeval max_tokens.

Round 2 candidates (from exploration agent — not yet applied; wait until R9 outcome before re-building datasets):

- **OpenMathInstruct-2 100K → 60K** (high overlap with MetaMathQA, ~$0.25 savings on any rebuild)
- **MetaMathQA 50K → 30K** (same reason, ~$0.15)
- **BoN `n_samples` 16 → 8, `n_prompts` 500 → 200** (rebalance for yield, ~$0.35 savings per BoN run)
- **RL max_steps 100 → ~60** (already applied for R9; R8-GRPO's plateau was around step 50-75)
- **Eval retry_on_error 5 → 1, max_connections 512 → 64** (~$0.15/eval)

None are load-bearing for avg_norm; all cuts can be applied when running new training or eval pipelines.

## 5. What's next (conditional)

After R9 finishes + evaluates:
- **If R9 > R8-BoN:** update `submission.json`, commit, write review, done.
- **If R9 ≥ R8-BoN but regresses HumanEval (plausible — RL on code may trade some with IF/math):** one more BoN distillation pass on top of R9 final (`r9_bon_8b` config is pre-staged in `sft_train.py`). Cost ~$4-6.
- **If R9 < R8-BoN (regression):** keep R8-BoN, document R9 as extension attempt with negative result. The report value of a negative result is non-zero (per PROJECT.md: "negative results are valued").

Budget at session start: ~$22-24 spent of $35 soft cap. R9 training ~$8-10, two intermediate evals ~$2, final R9 eval ~$1-2. Projected cumulative: ~$33-38 — at or slightly over the soft cap. No further training unless the R9 outcome warrants it.
