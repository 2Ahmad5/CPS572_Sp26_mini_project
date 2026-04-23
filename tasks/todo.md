# Implementation Plan — Multi-Task LoRA SFT for IFEval / GSM8K / HumanEval

**Target:** clear IFEval 45 / GSM8K 50 / HumanEval 30 on Llama-3.1-8B via Tinker LoRA; add a cheap RL extension for leaderboard credit.
**Budget:** $250 Tinker. Plan currently estimates ~$90 of spend. Generous headroom for retries.
**Strategy:** 3B kitchen-sink-first. Stack every likely-positive improvement into one 3B SFT run (R1) as a ceiling probe. Gate further 3B ablations on R1 results. Move to 8B once 3B is characterized.

---

## HARD CONSTRAINTS (read first)

1. **Never train on eval test data.** Explicit deny list, enforced by a single filter function before any dataset enters training:
   - `google/IFEval` (541 prompts, full dataset)
   - `openai/gsm8k` split=`test` (1,319 items)
   - `openai/openai_humaneval` (164 problems — prompts, docstrings, and canonical solutions)
2. **Do not modify anything under `evaluation/`** (eval_all.py, eval_ifeval.py, eval_gsm8k.py, eval_code.py, run_eval.sh). These are graded. The only file in that directory we touch is `train_and_publish.py`, which PROJECT.md explicitly tells us to replace.
3. **All data transforms must be deterministic and logged.** Any row we drop or rewrite gets counted; a summary is printed at the start of every training run.

---

## Attribution scheme (how we know what moved the needle)

| Comparison | What it measures | Cost |
|---|---|---|
| Base Llama-3.1-8B → Run 1 | Total effect of mixed SFT | 1 run |
| Run 1 intermediate checkpoints (500/1000/…/final) | Over-/undertraining sweet spot | Free (same run) |
| Run 1 final → Run 2 (GRPO on IFEval) | Effect of cheap post-SFT RL | 1 run |
| Temperature × top_p sweep on best checkpoint | Effect of decoding settings | Free (eval-only) |
| Run 1 with vs. without anti-forgetting general data | Deferred to report-writing analysis if any single-task degrades | — |

We explicitly do **not** separately ablate LoRA rank, LR, or per-dataset contribution. Those are cheap decisions backed by published results (see `tasks/exploration.md` §1–§4). Ablating each would cost one run apiece and isn't worth it.

---

## Experiment budget (kitchen-sink-first)

| Run | Purpose | Model | Approx cost |
|---|---|---|---|
| R0 | Smoke test (data load, tokenization, save/publish/eval round-trip) | Llama-3.2-3B | ~$2 |
| **R1** | **Kitchen-sink 3B SFT (ceiling probe)** | Llama-3.2-3B | ~$12 |
| R2+ | Conditional 3B ablations (only if R1 needs diagnosis) | Llama-3.2-3B | $0 or ~$8 |
| R3 | GRPO on IFEval at 3B from R1 best checkpoint (RL sanity + report material) | Llama-3.2-3B | ~$8 |
| R6 | Kitchen-sink 8B SFT (final recipe from 3B learnings) | Llama-3.1-8B | ~$45 |
| R7 | GRPO on IFEval at 8B from R6 best (only if R3 worked) | Llama-3.1-8B | ~$25 |
| Eval sweeps | Temperature/top_p combos + model soup | inference only | ~$5 |

Total expected: **~$95**. ~$155 headroom.

Not doing (explicitly):
- **GRPO on HumanEval** — leaky eval, expensive rollouts, thin published support. Skip.
- **GRPO on GSM8K** — cookbook recipe exists but SFT alone should clear 50%; RL on GSM8K after SFT is low marginal value in our context. Skip unless Run 1 lands below 40% on GSM8K.
- **On-policy distillation** — needs a stronger teacher model (more spend), and the published results are on Qwen3 not Llama-3.1. Re-consider only if we have >$150 left.
- **DPO** — weak for verifiable-reward tasks per Tulu-3 ablations. Skip.

---

## Progress log

- [x] Phase 0 verification complete (see below — all assumptions held)
- [x] `training/data_sources.py` written — adapters + deny-set + contamination filter
- [x] `training/build_dataset.py` written — preprocessing pipeline with char pre-filter
- [x] `training/sft_train.py` written — wraps cookbook's train.main with R0/R1/R6 configs
- [x] Syntax + imports verified
- [x] Deny-set canary test PASSED (gsm8k-test question flagged `True`, benign sentence `False`)
- [x] GSM8K-train adapter verified: outputs end with `ANSWER: N`
- [x] R0 build complete: 2,600 rows, clean format (1200 math rows all end with `ANSWER: N`, 804 rows have Python code)
- [x] R0 SFT complete: 76 steps on 3B, NLL 0.87 → 0.63, 2 checkpoints saved
- [x] R0 end-to-end eval works. **Preliminary results (40 samples, high variance):**
  - IFEval 41.3% (base: 22.6)
  - GSM8K 32.5% (base: 9.2) — proves `ANSWER:` format transfer works
  - HumanEval 52.5% (base: 0.6) — note only 40 samples; stderr 0.08 → CI [37%, 68%]
- [x] Fixed env loading (`.env` via python-dotenv)
- [x] Fixed eval invocation (`-m` module form; also updated `eval_checkpoints.py`)
- [x] Fixed HF streaming (replaced bulk-download with on-the-fly streaming; no local cache explosion)
- [x] Fixed `no_robots` split (`train`, not `train_sft` — streaming API disagrees with dataset card)
- [x] Adjusted R1 Magicoder targets (observed Python rate ~46-50%, so 60K/40K were infeasible; lowered to 45K/30K)
- [x] R1 kitchen-sink built: 347,346 rows (clean stats, 501 Magicoder contam drops = healthy)
- [x] R1 SFT on 3B complete: 2,709 steps, 6 checkpoints saved, final NLL 0.443
- [x] Evaluated all R1 checkpoints (limit 200) — all 6 beat every baseline
- [x] Full eval on step 2500 + final: **Final wins** (0.590 IF / 0.625 GSM8K / 0.378 HE, avg_norm 1.274)
- [x] Added bigcode/self-oss-instruct adapter; r1_enhanced config ready
- [x] R1_enhanced data build running (streaming, ~30% at last check)
- [x] Wrote R3 GRPO-IFEval code (`training/rl_ifeval.py` + `training/rl_train.py`) — uses argilla/ifeval-like-data with strict google/IFEval dedup
- [ ] **Running:** temperature × top_p sweep on R1 final (limit 200)
- [ ] R3 GRPO-IFEval on R1 final (~200 steps, Tulu-3-style KL β=0.05, LR 1e-5)
- [ ] R6 8B kitchen sink with r1_enhanced data
- [ ] R7 GRPO-IFEval on R6 best
- [ ] Optional RFT on GSM8K

## Phase 0 — Pre-implementation verification (no training, no spend)

**Status: complete.** Findings:
- Tinker `create_lora_training_client(base_model, rank=32, train_mlp=True)` default. MLP + attention are trained by default per SDK source (`service_client.py:182`). Matches "LoRA Without Regret" recommendation.
- LR formula confirmed in `tinker_cookbook/hyperparam_utils.py`: `lr = 5e-5 × 10 × (2000/hidden_size) ** 0.781`. Llama-3.2-3B → **3.56e-4**; Llama-3.1-8B → **2.83e-4**.
- All dataset schemas captured (field names, formats, sizes).
- Inspect AI GSM8K expects `ANSWER: N` at the end — implemented in every math adapter.
- Inspect AI HumanEval expects Python code (markdown-fenced or bare `def`) — code adapters enforce this filter.

Goals: confirm a handful of assumptions before writing pipeline code. Should take <30 minutes.

- [ ] Read `tinker_cookbook/renderers/llama3.py` end-to-end. Confirm loss mask targets only tokens between `<|start_header_id|>assistant<|end_header_id|>` and `<|eot_id|>`.
- [ ] Verify `tinker.ServiceClient().create_lora_training_client(base_model, rank)` applies LoRA to all linear layers by default (or find the kwarg to make it so). "LoRA Without Regret" says MLP-LoRA is mandatory; the cookbook's sl_basic recipe doesn't specify target modules, so default must be correct — but verify.
- [ ] Fetch Tinker pricing page; record per-token cost for Llama-3.1-8B LoRA training and inference. Update cost estimates above.
- [ ] Confirm `get_recommended_renderer_name("meta-llama/Llama-3.1-8B")` returns `"llama3"`.
- [ ] Inspect 2 sample rows from each source dataset on HuggingFace to confirm field names (so map_fn adapters match reality):
  - `nvidia/OpenMathInstruct-2` → fields? (expected: `problem`, `generated_solution`)
  - `meta-math/MetaMathQA` → (expected: `query`, `response`)
  - `microsoft/orca-math-word-problems-200k` → (expected: `question`, `answer`)
  - `openai/gsm8k` split=train → (expected: `question`, `answer`)
  - `ise-uiuc/Magicoder-Evol-Instruct-110K` → (expected: `instruction`, `response`)
  - `ise-uiuc/Magicoder-OSS-Instruct-75K` → (expected: `problem`, `solution`)
  - `bigcode/self-oss-instruct-sc2-exec-filter-50k` → (expected: `instruction`, `response`)
  - `allenai/tulu-3-sft-personas-instruction-following` → `messages`
  - `allenai/tulu-3-sft-personas-math-grade` → `messages`
  - `allenai/tulu-3-sft-personas-code` → `messages`
  - `HuggingFaceH4/no_robots` → `messages`

If any field name is different from assumed, the map_fn fails loudly (no silent skip); we fix and re-run.

---

## Phase 1 — Data pipeline (the high-risk phase)

### 1a. Deny-list enforcement

Before any dataset reaches training, every row's **user message** is passed through a single contamination filter:

- Build a 13-gram set from:
  - All 541 IFEval prompts (load from `google/IFEval`)
  - All 1,319 GSM8K test questions (load from `openai/gsm8k` split=`test`)
  - All 164 HumanEval prompts + docstrings (load from `openai/openai_humaneval`)
- Filter function: if any 13-gram of the user message appears in the deny set, drop the row.
- Log: rows-in, rows-out, rows-dropped, per source. Fail loudly if drop rate exceeds 5% on any source we trust — that means the source is more contaminated than we thought.

13-gram choice mirrors BigCode's decontamination pipeline (standard across Magicoder, StarCoder, OLMo). Overkill for small overlap, fast enough to run on 500K rows in a few minutes.

### 1b. Per-source adapters (map to `{messages: [{role, content}]}`)

Each source has its own field names and assistant-format quirks. Adapters live in one file (`train/data_sources.py`). Each adapter returns a `{messages: ...}` dict that the cookbook's SFT pipeline consumes.

**Math adapters** — critical: Inspect AI's GSM8K scorer extracts the numeric answer after `ANSWER:`. Training data must end assistant response with `ANSWER: {n}` to transfer at eval time.

- `openai/gsm8k` train split:
  - `answer` field already has `####` delimiter before the final answer. Extract N. Reformat assistant = full rationale + `\nANSWER: {N}`. Prefix user prompt with the Inspect AI GSM8K instruction verbatim.
- `nvidia/OpenMathInstruct-2`:
  - Responses end with `\boxed{N}`. Extract N. Append `\nANSWER: {N}`. Drop rows where we can't extract a numeric answer.
- `meta-math/MetaMathQA`:
  - Responses end with `The answer is: N`. Extract N. Append `\nANSWER: {N}`.
- `allenai/tulu-3-sft-personas-math-grade`:
  - Already in messages format. Extract final numeric answer from the last assistant turn (regex: last number in the message). Append `\nANSWER: {N}` if not already present.
- `microsoft/orca-math-word-problems-200k`:
  - Defer to Phase 1c depending on how much math we need. Skip if not required — it has no explicit GSM8K-test decontamination in its card.

**Code adapters** — Inspect AI's HumanEval scorer accepts either markdown code blocks or a bare function. Training data should provide Python code in markdown fences; models naturally learn to output markdown when the instruction asks for code.

- `ise-uiuc/Magicoder-Evol-Instruct-110K`: `instruction` → user, `response` → assistant. Responses already contain markdown code blocks. Pass through.
- `ise-uiuc/Magicoder-OSS-Instruct-75K`: same schema. Filter to rows whose assistant response contains `def ` and `python` (heuristic for Python-only). Pass through.
- `bigcode/self-oss-instruct-sc2-exec-filter-50k`: `instruction`/`response`. Pass through.
- `allenai/tulu-3-sft-personas-code`: already `messages`. Pass through.

**Instruction-following adapters** — no format transform needed:

- `allenai/tulu-3-sft-personas-instruction-following`: already `messages`, already targets IFEval constraint classes.
- `argilla/ifeval-like-data`: use `filtered` subset only (the full one has contradictory constraints). **Extra decontamination step**: exact-match dedup user prompts against IFEval prompts even before the 13-gram filter, since this dataset is MagPie-generated from the same constraint taxonomy. If >1% overlap, drop the source entirely.

**General-quality anti-forgetting** (small):
- `HuggingFaceH4/no_robots`: already `messages`, pass through. 10K rows. License is CC-BY-NC-4.0 — fine for academic use; record in report.

### 1c. Mix assembly

Use cookbook's `InterleavedChatDatasetBuilder` (verified in `tinker_cookbook/supervised/data.py:352`). Example sizes after filtering (subject to Phase 0 sanity-check):

| Bucket | Sources | Rows | Share |
|---|---|---|---|
| Math | OpenMathInstruct-2 (100K sample) + MetaMathQA (50K) + personas-math-grade (50K) | 200K | 53% |
| Code | Magicoder-Evol (60K sample) + Magicoder-OSS Python (40K) + personas-code (20K) | 120K | 32% |
| IF | personas-instruction-following (30K) + argilla/ifeval-like filtered (20K, post-dedup) | 50K | 13% |
| General | no_robots (5K sample) | 5K | 1% |
| **Total** | | **~375K** | 100% |

Interleave weights: proportional to bucket share (`InterleavedChatDatasetBuilder` handles this).

Sequence length: filter rows that tokenize beyond **3072 tokens** (post-renderer). This protects us from OpenMathInstruct-2's occasional very long rationales blowing up training cost and from eval truncation issues (`max_tokens=1024` at eval time means rationales should be learnable at a similar scale).

### 1d. Sanity smoke tests (in `train/data_smoke.py`)

Before Run 0:
- Load 10 examples per source after all transforms. Print: user, assistant, loss-weight mask (assistant-only expected).
- Tokenize and round-trip through the Llama-3 renderer. Assert no empty weight vectors, no all-zero weight rows.
- Assert math rows end with `ANSWER: \d+`.
- Assert code rows contain ```` ```python```` or a Python `def ` block.

---

## Phase 2 — Run 0: pipeline smoke test (Llama-3.2-3B)

**Intent:** every moving part works end-to-end before we spend on 8B.

- Model: `meta-llama/Llama-3.2-3B`
- LoRA rank 16, LR computed via cookbook's `get_lr` for 3B+LoRA (~3.5e-4), batch 32, seq 2048, 1 epoch
- Data: 5,000-row stratified sample of the full mix (same ratios)
- `save_every=100`, `eval_every=50` (NLL only; benchmark eval is too expensive for smoke)
- After training: run `python evaluation/eval_all.py --checkpoint_path ... --base_model meta-llama/Llama-3.2-3B --limit 40`
- **Pass bar:** all three tasks produce a non-zero score and the number is above the 3B base (22.6 / 9.2 / 0.6 per README.md). Absolute numbers don't matter; we're just proving the pipeline.

If Run 0 fails: fix, rerun; do not proceed to Run 1.

---

## Phase 3 — Run 1: main SFT (Llama-3.1-8B)

- Model: `meta-llama/Llama-3.1-8B`
- LoRA rank **32**, LoRA on **all linear layers** (Tinker default; verified in Phase 0)
- LR **2e-4**, linear decay, 3% warmup — conservative vs. cookbook's formula (~2.9e-4); the formula is aggressive and we prefer not to need a retry
- Batch **128**, max_length **3072**, **1 epoch** (Raschka shows multi-epoch degrades instruction tuning)
- Adam betas (0.9, 0.95), eps 1e-8
- `save_every=500`, `eval_every=500` (NLL); `infrequent_eval_every=1000` for benchmark eval on `--limit=200` slice (gives us a learning curve without burning the full 3024-item eval at every checkpoint)
- ~375K rows / batch 128 ≈ **2,930 steps**
- `wandb_project`/`wandb_name` set for logging (report uses these plots)

**After Run 1 finishes:**
- Publish final checkpoint and every saved intermediate (every 500 steps).
- Run full eval (`eval_all.py` no `--limit`) on each: checkpoints at steps ~500, 1000, 1500, 2000, 2500, final. That's ~6 full evals × ~30 min = ~3 hours of eval time.
- Pick the best checkpoint by averaged normalized score = mean of (IFEval/45, GSM8K/50, HumanEval/30). The best is often not the final — this is explicitly called out in PROJECT.md.

**Go/no-go decision after Run 1:**
- If best checkpoint clears all 3 baselines → proceed to Run 2.
- If one task is below baseline → adjust mix (e.g. raise IF share if IFEval below), rerun. Budget tolerates one retry.
- If two+ tasks below baseline → deeper debug; probably a format mismatch in Phase 1 that Run 0 missed.

---

## Phase 4 — Run 2: GRPO on IFEval (from Run 1 best checkpoint)

Template: `tinker_cookbook/recipes/math_rl/train.py` (verified LR 8e-5 config achieves 90.9% GSM8K in 220 steps per cookbook README).

- Starting checkpoint: Run 1 best (`load_checkpoint_path`)
- Environment: custom `IFEvalEnv` that reuses `tinker_cookbook/eval/benchmarks/ifeval.py`'s verifier for reward
- Reward: `prompt_level_strict` (binary, 1.0 if all constraints satisfied, 0.0 else). We optimize the hardest sub-metric; the other three tend to rise with it.
- Prompt source: `allenai/tulu-3-sft-personas-instruction-following` (plus `argilla/ifeval-like-data` filtered) — same sources as Phase 1, which share the IFEval constraint taxonomy. We are NOT rolling out on `google/IFEval` itself.
- Hyperparameters (conservative, derived from Tulu-3 RLVR + LoRA 10× adjustment):
  - LR **3e-6** (Tulu-3 used 3e-7 full-FT; ×10 for LoRA per "LoRA Without Regret")
  - KL β **0.05** (Tulu-3 middle of their sweep)
  - Group size **16**, groups per batch **32**, max tokens **1024**, temperature 1.0 during rollout
  - 200 steps max, save every 50
- After each chunk of 50: full eval on all 3 tasks. Stop early if any non-IF task degrades by >3 points.

**Attribution gate:** Run 2 succeeds only if IFEval increases AND GSM8K/HumanEval don't drop >2 points. If the other tasks degrade more, we report the negative result and submit Run 1 best.

---

## Phase 5 — Submission sweep

- Candidates: {Run 1 best, Run 2 final} if Run 2 passed; else {Run 1 best}.
- Generation-setting sweep on candidate(s): `temperature ∈ {0.0, 0.2}`, `top_p ∈ {1.0, 0.9}` → 4 settings per candidate.
- Full `eval_all.py` for each combo. Pick winner by averaged normalized score.
- Write winning `submission.json`. Leave `evaluation/eval_*.py` untouched.

---

## Bug audit — things that would silently fail

1. **Math format mismatch at eval time.** If `ANSWER: N` isn't appended to math training data, the Inspect AI scorer won't extract the answer even when the model is right. → Phase 1b transform is the single most important line of code in this project. Adapter returns an exception if it can't extract N; row is dropped and counted (no silent pass-through).
2. **HumanEval prompt mismatch.** Inspect AI prepends "Your response should only contain the code for this function." If training data has assistant responses that explain the code first, the model may produce prose at eval time and fail the code-extraction regex. → Phase 1b filter: for code sources, keep only rows whose assistant response starts with ```` ```python```` or a `def `.
3. **IFEval overfit from training on its constraint taxonomy without diversity.** personas-IF was deliberately designed to match IFEval constraint types. If we train heavily on just this 30K, we may overfit to the phrasing style. → Mitigation: mix with argilla/ifeval-like filtered, and cap IF share at ~13% of the total mix.
4. **Contamination slip-through.** 13-gram is great for exact-prompt leakage but misses paraphrase. → We accept this risk but print drop counts; if any source drops >0 rows, inspect them manually.
5. **Renderer loss-mask mismatch.** If `train_on_what` is wrong, we could train on user prompts (wrong gradient signal) or nothing at all (silent NaN loss). → Phase 1d smoke test prints mask for 10 examples; we confirm assistant-only tokens carry weight.
6. **Sequence truncation silently drops critical answer tokens.** If `max_length=3072` truncates mid-rationale, the model learns to produce unterminated chains. → Pre-filter by tokenized length; don't truncate.
7. **Checkpoint TTL.** Cookbook default TTL for periodic checkpoints is 7 days; final is kept indefinitely. If evaluation drags, intermediate checkpoints could expire. → Bump to 30 days (`ttl_seconds=2592000`) for Run 1.
8. **`InterleavedChatDatasetBuilder.stopping_strategy`.** Default "all_exhausted" oversamples small sources by repeating — we want each row seen ~once. → Set `stopping_strategy="first_exhausted"` so the largest source (math, ~200K) dictates one epoch's step count.
9. **Off-by-one on baseline**. PROJECT.md baseline says "baseline score 45.0%" for IFEval. IFEval has 4 sub-metrics averaged. Make sure we compute the same average when comparing to baseline. `eval_ifeval.py:72-76` shows it collects all sub-metrics — we'll average `prompt_level_strict`, `prompt_level_loose`, `inst_level_strict`, `inst_level_loose` for apples-to-apples.
10. **LoRA target modules default**. If Tinker's default isn't all-linear, we silently eat a 5–10 point hit (per "LoRA Without Regret"). → Phase 0 verifies.
11. **`submission.json` regeneration**. `eval_all.py --checkpoint_path` *overwrites* submission.json on each run. If we sweep temperatures we need to save each separately before picking the winner. → Simple: rename the file after each sweep run, merge manually.
12. **Don't train on the `openai/gsm8k` test split.** Easy to accidentally load `split="test"` or the default `split=None` which may include both. → Always explicit `split="train"` in our loader; a smoke test asserts row count of math-from-gsm8k is ≤7473.

---

## File layout (what I'll add)

```
training/                       # new directory, separate from evaluation/
├── __init__.py
├── data_sources.py             # adapters, decontamination filter, mix assembly
├── data_smoke.py               # Phase 1d smoke tests
├── sft_train.py                # Run 0 + Run 1 entrypoint (chz Config via cookbook)
├── rl_train.py                 # Run 2 entrypoint
├── submit.py                   # Phase 5 sweep helper (wraps eval_all.py invocations)
└── configs/
    ├── run0_3b_smoke.py
    ├── run1_8b_main.py
    └── run2_8b_grpo_ifeval.py
```

`evaluation/train_and_publish.py` stays (PROJECT.md's toy reference). We don't modify anything else under `evaluation/`.

---

## What success looks like

- `tasks/todo.md` all boxes checked
- Run 1 best checkpoint clears 45/50/30
- Run 2 ideally pushes IFEval 5+ points above Run 1 without degrading the other two
- Final `submission.json` points to whichever of {Run 1, Run 2} won the sweep
- Report section on negative results: which sources' rows got dropped in decontamination; any run-to-run regression; the specific temperature/top_p ablations that lost

---

## Open questions for user before I start coding

1. **Go/no-go on the 3-run plan?** If you'd rather do 2 runs (drop Run 2 GRPO) or 4 runs (add GSM8K GRPO), I'll adjust.
2. **Should Run 2 GRPO be done even if Run 1 clears all three baselines comfortably?** I'd still do it — the leaderboard credit is cheap and GRPO experience is a meaty extension for the final report. But happy to hold.
3. **Is the mix composition OK (53% math / 32% code / 13% IF / 1% general)?** This is weighted toward math because GSM8K baseline is the highest bar to clear (50% from a ~10% base). Reasonable to rebalance if the Phase 0 pricing check suggests we should spend fewer tokens overall.
