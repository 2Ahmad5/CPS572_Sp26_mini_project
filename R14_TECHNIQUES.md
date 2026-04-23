# R14 — techniques and first principles

R14 is the current best clean checkpoint:
- IFEval **71.30%** / GSM8K **80.29%** / HumanEval **50.00%** → avg_norm **1.6190**
- Checkpoint: `tinker://b5b5e31c-db3f-5987-8cc4-dab7bf015a1a:train:0/sampler_weights/final`
- Base model: `meta-llama/Llama-3.1-8B`, greedy decoding (temperature=0, top_p=1.0)

R14 is built by stacking five training stages on top of `Llama-3.1-8B`:

```
base 8B → R11 (SFT) → R12 (3-way GRPO) → R13 (BoN-SFT, 113 rows) → R14 (RSFT, 355 rows)
```

Below: each technique that produces R14, why it works from first principles, and the file/config that implements it.

---

## 1. LoRA on the base model (not Instruct)

**What.** Train rank-32 LoRA adapters on Llama-3.1-8B base, on all linear layers (`train_mlp=True`).

**Why.** LoRA learns a low-rank update `ΔW = BA` to each weight matrix. The information-capacity budget for a rank-r LoRA on hidden-size h is `~r·h^2` parameters per matrix; for our setup that's roughly 1B trainable params, well above the ~10⁹ tokens our 317K-row corpus produces. *LoRA Without Regret* (Biderman 2024) shows this matches full-FT quality below the capacity ceiling, while costing far less to train and ship. Using the **base** model (not Instruct) means we control the entire instruction format ourselves — no fighting against pre-existing chat templates, important for IFEval where format is the score.

**Files.** `training/sft_train.py:r11_clean_8b` (`lora_rank=32`).

---

## 2. LR formula `lr = 5e-5 · 10 · (2000/h)^0.781`

**What.** SFT learning rate of `2.83e-4` for Llama-3.1-8B (hidden=4096); RL LR of `1.5e-5` (~10x lower).

**Why.** The same paper derives this empirically — full-FT-equivalent SFT LR scales with hidden size by `h^-0.781`, multiplied by 10× for LoRA. Picking too low underfits; too high causes loss spikes and capacity collapse. We saw smooth monotonic NLL descent (0.667 → 0.300 over 2478 steps) with no spikes — the formula was in the right zone. RL needs ~10× lower because gradient variance from sparse rewards is much larger than supervised cross-entropy.

**Files.** `training/sft_train.py:r11_clean_8b` (`learning_rate=2.83e-4`), `training/rl_train.py:r12_grpo_3way_clean_8b` (`learning_rate=1.5e-5`).

---

## 3. Multi-source SFT data mix (decontaminated)

**What.** R11 trains on `r1_clean.jsonl`, 317,218 rows from 7 sources:
- math: gsm8k train + OpenMathInstruct-2 + MetaMathQA + tulu-3-personas-math-grade
- code: tulu-3-personas-code + bigcode/self-oss-instruct-sc2-exec-filter
- IF: tulu-3-personas-instruction-following

**Why.** Multi-task transfer requires multi-task SFT data, mixed in proportions that match what we're scored on. Our R3 single-task RL experiment (IFEval-only on 3B) showed a +3.8 IF gain but −1.0 GSM and −1.8 HE — single-objective SFT/RL regresses other objectives (Tulu-3 §6.3, IFBench arXiv 2507.02833). Mixing math/code/IF in the same training set lets the LoRA share representations across tasks rather than treating them as alternatives. Each source's count was tuned to keep no single bucket overwhelming — 51% math, 28% code, 9% IF, with the remainder utility.

The conspicuously absent source is `ise-uiuc/Magicoder-Evol-Instruct-110K`. We dropped it because our deep audit (#5 below) caught it leaking 22% of HumanEval problems via Evol-Instruct paraphrases. Removing it cost ~5pp HE from the dirty baseline but is the only defensible choice under the project's "training on test data = score of 0" rule.

**Files.** `training/data_sources.py` (source registry, adapters); `training/build_dataset.py:r1_clean` (mix weights).

---

## 4. Format alignment — the `ANSWER: N` suffix

**What.** Every math training row ends with `\nANSWER: <number>`. The GSM8K eval scorer regexes for this exact tag.

**Why.** Models can solve a problem correctly and still score 0 if their output format doesn't match the scorer. R0 smoke run measured this directly: GSM8K accuracy went **9.2% → 32.5% in 76 SFT steps** purely from appending `ANSWER: N` — same model, same math reasoning, just learning to surface the final answer in the format the grader looks for. It's the highest-ROI single change in the project. The same principle drives the math RL reward (`training/rl_math_env.py:_extract_number`) — it prefers the `ANSWER:` tag and falls back to "last number in response" only when the tag is missing, which keeps the RL reward signal aligned with what wins on eval.

**Files.** `training/data_sources.py:adapt_gsm8k_train` and friends (append `ANSWER: N`); `training/rl_math_env.py:_ANSWER_TAG_RE`.

---

## 5. Decontamination: 13-gram filter at build time + Aho-Corasick fingerprint audit at submission time

**What.** Two layers of integrity:

(a) **Build-time** — `training/data_sources.py:build_deny_ngrams` constructs a deny-set of every word-level 13-gram from `google/IFEval`, `openai/gsm8k` test split, and `openai/openai_humaneval`. Any training row containing a matching 13-gram is dropped during `build_dataset.py`. This is the BigCode/StarCoder standard.

(b) **Submission-time** — `training/verify_no_test_leak_deep.py` extracts ~7,500 *distinctive fingerprints* from every test item (function signatures, `>>> example()` calls, doctest expected-output lines, docstring sentences, canonical-solution lines) and uses an **Aho-Corasick automaton** (one linear pass per training file) to detect any of them as substrings.

**Why.** The 13-gram filter is necessary but not sufficient. It catches verbatim leakage (e.g. someone copy-pasted a HumanEval problem into Magicoder), but misses *derivative* contamination — when a dataset has been "Evol-Instruct paraphrased" from a HumanEval seed and the wording is changed but distinctive examples like `make_palindrome('cat') → 'catac'` are preserved unchanged. We caught Magicoder-Evol-Instruct doing exactly this (37/164 HE items). Aho-Corasick checks 7,500 needles against a 500MB haystack in under 5 seconds, making this practical to run on every materialized JSONL before training or submission.

**Files.** `training/verify_no_test_leak.py` (basic: EXACT + SUBSTRING), `training/verify_no_test_leak_deep.py` (deep: fingerprints + AC), `training/prep_candidate_datasets.py` (HumanEval function-name blocklist + 13-gram pre-filter for new sources).

---

## 6. Tighter `max_length=2048` cap on training

**What.** Training sequences are truncated at 2048 tokens (R1 used 3072).

**Why.** The eval harness samples with `max_tokens=1024` (and `512` for IFEval). Training the model to produce 3000-token chains-of-thought teaches it to keep going past where the eval cuts it off — the answer ends up missing. Tightening to 2048 forces shorter reasoning during training, which yields shorter outputs at inference that fit inside the 1024-token eval budget. R1 at 3072-cap got GSM 62.6%; R6 at 2048-cap got 77.3% — the cap is plausibly load-bearing for that gain.

**Files.** `training/sft_train.py:r11_clean_8b` (`max_length=2048`).

---

## 7. 3-way mixed GRPO with verifiers (R12 step)

**What.** After SFT, run GRPO with three reward streams interleaved at weights `[0.34 IF, 0.33 math, 0.33 code]`:
- **IFEval verifier** — `tinker_cookbook.eval.benchmarks._ifeval_verify.verify_all_instructions`. Reward = 1 iff every constraint passes.
- **Math verifier** — extract a number from the response (`ANSWER:` regex preferred, last-number fallback) and compare to ground truth via `math.isclose`. Reward = 1 if exact match.
- **Code verifier** — extract `python` fenced code, write to a temp file, run each MBPP test in a subprocess with a 5-second timeout. Reward = 1 iff *all* tests pass.

GRPO computes a **per-group advantage** (relative to other rollouts of the same prompt), so reward shaping requires that all rollouts inside a group come from the same domain. We use `tinker_cookbook.rl.interleaved.InterleavedRLDatasetBuilder`, which produces single-domain groups within each batch but mixes the *batches* in the configured weight ratio.

**Why.** RL learns roughly "1 bit per episode" (LoRA Without Regret). With three task verifiers wired up, every gradient step nudges the policy toward winning *some* benchmark. The hard part is preventing single-task regression — our R3 experiment showed IFEval-only RL gives +3.8 IF but −1.0 GSM and −1.8 HE because the policy moves probability mass off the other tasks' competence. Interleaving keeps all three tasks "alive" in the gradient. We use `total_batches=15` to force the MBPP source (only ~120 decontaminated prompts) to cycle, otherwise training would terminate at step 7 when MBPP exhausts (this happened in R9 before the cycling fix).

**Files.** `training/rl_train.py:r12_grpo_3way_clean_8b`, `training/rl_ifeval.py`, `training/rl_math_env.py`, `training/rl_code_env.py`.

---

## 8. Subprocess code verifier (no sandbox dep)

**What.** `bon_run.py:_run_tests` writes generated code + each test case to a temp file, executes via `subprocess.run([sys.executable, path], timeout=5)`, counts success.

**Why.** The standard RL-for-code recipe (DeepCoder, AceCoder) uses Modal or SandboxFusion as a sandbox layer. Both are heavy infra dependencies. Since we control the training inputs end-to-end and run on a trusted local machine, a subprocess with a timeout is a safe-enough verifier — same trust model as running pytest. ~80 lines of code, zero external services. The 5-second timeout catches infinite loops without slowing typical solutions noticeably.

**Files.** `training/bon_run.py:_run_tests`, `training/rl_code_env.py:_run_tests` (mirrored).

---

## 9. Best-of-N rejection-sampling SFT (R13 + R14 steps)

**What.** R13 and R14 share the same recipe: sample N=16 rollouts per prompt from the previous checkpoint at `temperature=1.0`, run them through the subprocess verifier, keep only fully-verified solutions, then SFT on those. R13 used 113 rows from 120 MBPP-sanitized prompts; R14 used 355 rows from 374 prompts (MBPP-sanitized + decontaminated MBPP-full).

**Why.** This is rejection-sampling SFT (Llama-3.1 paper §3.5, also called RAFT). The model knows how to solve some MBPP problems but its sampling distribution is wide — high-temperature sampling generates diverse candidates, the verifier filters down to known-correct ones, SFT teaches the model to prefer that specific verifier-passing distribution. Three properties make this attractive:

1. **Zero contamination risk** — no new training data; the rollouts come from a model trained only on already-decontaminated data.
2. **Verifier-grounded supervision** — every retained row has been objectively shown to pass tests, so we're not amplifying confident-but-wrong outputs.
3. **Compounding** — R13 added +0.016 avg_norm; R14 (3.1× more data) added another +0.039. Same recipe, monotone.

R14's HumanEval gain from this step was +2.44pp (47.56 → 50.00) and an unexpected GSM8K cross-task transfer of +2.28pp — code-style reasoning structure helps math too.

**Files.** `training/bon_run.py` (rollout + verifier loop), `training/sft_train.py:r9_bon_8b` (the SFT continuation config, reused for both R13 and R14 by passing `--data_file` and `--load_checkpoint_path` overrides).

---

## 10. Greedy eval (`temperature=0`, `top_p=1.0`)

**What.** All evaluation runs use deterministic argmax sampling.

**Why.** Reproducibility. Two runs with the same checkpoint produce identical results, which is the only honest way to compare two checkpoints' eval scores. We tested temperature 0.0 / 0.2 / 0.5 with top-p 0.9 / 1.0 on R1 step-2500 and greedy won on avg_norm — but the bigger value was eliminating sampling noise from comparison decisions throughout the iteration. The grader will re-run our checkpoint with these same settings (per `evaluation/eval_all.py`'s `submission.json` schema), so the score we report is the score they'll see.

**Files.** `evaluation/eval_all.py` (defaults `temperature=0.0`, `top_p=1.0`).

---

## How to reproduce R14 from scratch

```bash
# 1. Build the SFT dataset (~30 min, no Tinker cost)
python -m training.build_dataset --config r1_clean --output training/data/r1_clean.jsonl --max_len_tokens 2048

# 2. Audit the dataset (under 5 min, no Tinker cost)
python -m training.verify_no_test_leak_deep --files training/data/r1_clean.jsonl

# 3. R11: SFT (~1.5-2 hr, ~$8-10)
python -m training.sft_train --config r11_clean_8b
# -> note the final checkpoint URI from the printed `Saved checkpoints` line

# 4. R12: 3-way mixed GRPO (~20 min, ~$3)
python -m training.rl_train --config r12_grpo_3way_clean_8b \
  --load_checkpoint_path "tinker://<R11_RUN_ID>:train:0/weights/final"

# 5. R13: first BoN-SFT round (~7 min rollout + 5 min SFT, ~$2)
python -m training.bon_run --checkpoint "tinker://<R12_RUN_ID>:train:0/sampler_weights/final" \
  --base_model meta-llama/Llama-3.1-8B --n_prompts 400 --n_samples 16 \
  --output training/data/r13_bon.jsonl
python -m training.sft_train --config r9_bon_8b \
  --load_checkpoint_path "tinker://<R12_RUN_ID>:train:0/weights/final" \
  --data_file training/data/r13_bon.jsonl

# 6. Build expanded MBPP prompt pool for R14
python -m training.prep_candidate_datasets --which mbpp_full

# 7. R14: second BoN-SFT round on expanded pool (~25 min rollout + 5 min SFT, ~$5-7)
python -m training.bon_run --checkpoint "tinker://<R13_RUN_ID>:train:0/sampler_weights/final" \
  --base_model meta-llama/Llama-3.1-8B --n_prompts 400 --n_samples 16 \
  --extra_prompts training/data/mbpp_full_prompts.jsonl \
  --output training/data/r14_rsft.jsonl
python -m training.sft_train --config r9_bon_8b \
  --load_checkpoint_path "tinker://<R13_RUN_ID>:train:0/weights/final" \
  --data_file training/data/r14_rsft.jsonl

# 8. Evaluate R14
python -m evaluation.eval_all \
  --checkpoint_path "tinker://<R14_RUN_ID>:train:0/sampler_weights/final" \
  --base_model meta-llama/Llama-3.1-8B \
  --output_path evaluation/submission.json
```

End-to-end: ~3-4 hours wall-clock, ~$20-25 Tinker credits.
