# Data Reproduction

Exactly what a TA needs to rebuild every training dataset that went into R16, from scratch. No pre-built JSONLs are shipped in this zip — everything is rebuilt from public Hugging Face sources by running the scripts below.

---

## 1. Raw data sources

All via `datasets.load_dataset(...)` over the Hugging Face Hub. These are the only external inputs.

### SFT sources (supervised fine-tuning on r1_clean_v2)

| HF path | Config / split | Role | # kept in final mix |
|---|---|---|---|
| `openai/gsm8k` | `main` / `train` | math | 7,466 |
| `nvidia/OpenMathInstruct-2` | default / `train_1M` | math | 100,000 |
| `meta-math/MetaMathQA` | default / `train` | math | 50,000 |
| `allenai/tulu-3-sft-personas-math-grade` | default / `train` | math | 49,951 |
| `allenai/tulu-3-sft-personas-code` | default / `train` | code | 34,991 |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | default / `train` | code | 45,000 |
| `allenai/tulu-3-sft-personas-instruction-following` | default / `train` | IF | 29,810 |
| `KodCode/KodCode-V1` | default / `train` | code | 15,000 (subsampled from a 30K filtered pool) |
| **Total** | | | **332,218 rows** |

### RL prompt sources (used for GRPO rollouts — the model sees these as *prompts only*, responds, and gets rewarded by a verifier; no supervised signal on the content)

| HF path | Config / split | Role |
|---|---|---|
| `argilla/ifeval-like-data` | `filtered` / `train` | IFEval RL prompts (with exact-match filter vs `google/IFEval`) |
| `openai/gsm8k` | `main` / `train` | math RL prompts (same source as SFT math bucket; HF train split is disjoint from test) |
| `google-research-datasets/mbpp` | `sanitized` / `train` | code RL prompts (13-gram decontaminated vs HumanEval) |

### Held-out evaluation benchmarks (never trained on, only scored against; used as deny-sets for decontamination)

| HF path | Split | Role |
|---|---|---|
| `google/IFEval` | `train` (this is the eval set despite the name) | IFEval benchmark (541 prompts) |
| `openai/gsm8k` | `main` / `test` | GSM8K benchmark (1,319 items) |
| `openai/openai_humaneval` | `test` | HumanEval benchmark (164 problems) |

---

## 2. Preprocessing / filtering decisions (per-source)

All adapter functions live in `training/data_sources.py`. Every adapter returns `{"messages": [{role, content}, ...]}` or `None` (drop). Decisions by source:

- **`openai/gsm8k` train** (`adapt_gsm8k_train`): split on the `####` sentinel, append `\nANSWER: <N>` to the assistant response. *This is load-bearing.* The eval scorer regexes for `ANSWER: N`; training data that ends with the original `#### N` format gets the answer wrong even when the reasoning is correct. Our R0 smoke run saw GSM8K go 9.2% → 32.5% in 76 steps from this single change.

- **`nvidia/OpenMathInstruct-2`** (`adapt_openmathinstruct2`): keeps rows with an `expected_answer` field, appends `\nANSWER: <expected_answer>`.

- **`meta-math/MetaMathQA`** (`adapt_metamathqa`): regex-extracts the final answer from `The answer is: N`. Appends `\nANSWER: <N>` if not already present.

- **`allenai/tulu-3-sft-personas-math-grade`** (`adapt_persona_math_grade`): Tulu's personas math already has CoT; we append `\nANSWER: <last-number-in-assistant>`.

- **`ise-uiuc/Magicoder-Evol-Instruct-110K`** (`adapt_magicoder_evol`): **REMOVED from the final mix.** We originally used it with a Python-fence filter (require ```python``` block or `def` definition). Mid-project, our deep contamination audit found this dataset contained Evol-Instruct paraphrases of 22% of HumanEval problems — distinctive docstring examples like `>>> how_many_times('aaaa', 'aa')` (HumanEval/18) preserved verbatim inside paraphrased prose. The 13-gram filter didn't catch this because Evol-Instruct rewrites the prose while keeping the examples intact. We dropped the source entirely. See `training/data_sources.py:SOURCES_EXCLUDED_FOR_STRICT_DECONTAM`.

- **`ise-uiuc/Magicoder-OSS-Instruct-75K`**: **Dropped** in earlier cost-audit (2026-04-19) — redundant with Magicoder-Evol, same distillation lineage, no measurable independent signal.

- **`HuggingFaceH4/no_robots`**: **Dropped** in earlier cost-audit — 1% slice, no measurable effect.

- **`allenai/tulu-3-sft-personas-code`** (`adapt_persona_code`): pass-through.

- **`bigcode/self-oss-instruct-sc2-exec-filter-50k`** (`adapt_bigcode_self_oss`): require a `def` in the response to drop non-Python rows.

- **`allenai/tulu-3-sft-personas-instruction-following`** (`adapt_persona_if`): pass-through.

- **`KodCode/KodCode-V1`** (handled in `training/prep_candidate_datasets.py`, not in `data_sources.py`): applied three *pre-filters* before we ever write a row to JSONL:
  1. `benchmark_similarity < 0.5` (stricter than KodCode's native 0.95 filter; their field name is `benchmark_similarity` and scores cosine-similarity vs HumanEval/MBPP/BCB/LCB canonicals).
  2. A HumanEval function-name blocklist: if the row contains `def <name>(` where `<name>` matches any HumanEval canonical (`is_palindrome`, `make_palindrome`, `is_prime`, `greatest_common_divisor`, etc.), drop it.
  3. 13-gram word-level deny-set from IFEval + GSM8K-test + HumanEval (same filter used for every other SFT source).

### Universal filters applied to *every* SFT source

In `training/build_dataset.py`:

- **13-gram word deny-set** (`training/data_sources.py:build_deny_ngrams`). Union of all 13-word-grams from `google/IFEval.prompt` + `openai/gsm8k` test question+answer + `openai/openai_humaneval` prompt+canonical_solution. Any training row with a matching 13-gram on either user or assistant content is dropped.

- **Length filter (`is_too_long`)**: `max_len_tokens=2048`. Two-stage: a fast char pre-filter (`total_chars < max*2` → keep; `> max*8` → drop) followed by exact Llama-3.1 tokenization only for the ambiguous middle band. This is 50x faster than tokenizing everything and catches the same rows.

### RL prompt-source decontamination

- **`argilla/ifeval-like-data`** (in `training/rl_ifeval.py`): load the `filtered` config / `train` split, then filter out any row whose `prompt` is an exact string match with any of the 541 `google/IFEval` prompts. Catches verbatim duplicates; the IFEval benchmark and argilla share the same constraint library so phrase-level overlap is inherent and expected.

- **MBPP sanitized** (in `training/rl_code_env.py` via `_load_mbpp_decontaminated`): apply the same 13-gram deny-set built from HumanEval. After decontamination ~120 rows survive out of MBPP's sanitized-train 427.

### BoN rollout datasets (R13, R14, R17)

These are **generated at training time** by sampling 16 rollouts per MBPP prompt at `temperature=1.0` from the prior-stage checkpoint, running each through a subprocess test-executor with a 5-second timeout, and keeping only rollouts that pass every test. Not deterministic (sampling), but the *recipe* reproduces: same seed, same checkpoint, same prompt pool, same verifier produces statistically equivalent datasets.

We don't ship these JSONLs. They get recreated by `training/bon_run.py`.

---

## 3. End-to-end rebuild commands

All commands assume `python` points at the venv Python (`.venv/Scripts/python` on Windows or `.venv/bin/python` on Linux/Mac). The `TINKER_API_KEY` environment variable must be set.

### Step 1: build the base SFT mix (r1_clean.jsonl)

```bash
python -m training.build_dataset \
    --config r1_clean \
    --output training/data/r1_clean.jsonl \
    --max_len_tokens 2048 \
    --seed 42 \
    --shuffle_buffer 10000 \
    --max_seen_factor 5.0
```

Takes ~10 min (HF streaming), produces **317,218 rows**. Source counts: math 207,417 / code 79,991 / IF 29,810.

### Step 2: build the filtered KodCode pool

```bash
python -m training.prep_candidate_datasets \
    --which kodcode \
    --kodcode_max 30000 \
    --sim_threshold 0.5
```

Takes ~5 min. Produces `training/data/kodcode_candidate.jsonl` (30,000 rows kept out of ~402,504 seen; ~371K dropped on similarity, ~1.2K on HumanEval function-name blocklist).

### Step 3: merge into r1_clean_v2

```bash
python -m training.make_r1_clean_v2
```

Reads both JSONLs, shuffles 15K KodCode rows in with the 317K r1_clean rows, writes **`training/data/r1_clean_v2.jsonl`** — the final SFT base (332,218 rows total).

### Step 4: R15 SFT (~1.5-2 hr wall-clock, ~$8-10 Tinker credits)

```bash
python -m training.sft_train --config r15_clean_v2_8b
```

This is `meta-llama/Llama-3.1-8B` with LoRA rank 32, LR 2.83e-4, batch 128, max_length 2048, 1 epoch. Logs go to `logs/r15_clean_v2_8b/`. Captures the final checkpoint URI at the end of the run.

### Step 5: R16 3-way GRPO (~20 min, ~$3)

```bash
python -m training.rl_train --config r16_grpo_3way_clean_v2_8b \
    --load_checkpoint_path "tinker://<R15_RUN_ID>:train:0/weights/final"
```

3-way mixed GRPO using the `InterleavedRLDatasetBuilder` from `tinker_cookbook`. Reward sources:
- IFEval: `argilla/ifeval-like-data` filtered × `verify_all_instructions`
- Math: `openai/gsm8k` train × numeric-match on the `ANSWER:` regex
- Code: `google-research-datasets/mbpp` sanitized (decontam'd) × subprocess test-exec

Group size 8, 48 groups/batch (16/domain), LR 1.5e-5, KL β=0.05, `max_tokens=1024`, `max_steps=15` with `total_batches=15` (forces MBPP to cycle — without this the run stops at step 7 on MBPP exhaustion).

### Step 6: evaluate

```bash
python -m evaluation.eval_all \
    --checkpoint_path "tinker://<R16_RUN_ID>:train:0/sampler_weights/final" \
    --base_model meta-llama/Llama-3.1-8B
```

Writes `evaluation/submission.json` with R16's IFEval/GSM8K/HumanEval scores.

---

## 4. Optional — reproducing the full iteration ladder

If the TA wants to reproduce *every* intermediate checkpoint (R11 through R17), not just R16:

```bash
# R11: SFT on r1_clean alone (no KodCode) — the clean baseline
python -m training.sft_train --config r11_clean_8b

# R12: 3-way GRPO on top of R11
python -m training.rl_train --config r12_grpo_3way_clean_8b \
    --load_checkpoint_path "tinker://<R11_ID>:train:0/weights/final"

# R13: BoN distillation on top of R12 (first round, 120 MBPP sanitized prompts)
python -m training.bon_run \
    --checkpoint "tinker://<R12_ID>:train:0/sampler_weights/final" \
    --base_model meta-llama/Llama-3.1-8B \
    --n_prompts 400 --n_samples 16 --max_tokens 1024 \
    --output training/data/r13_bon.jsonl
python -m training.sft_train --config r9_bon_8b \
    --load_checkpoint_path "tinker://<R12_ID>:train:0/weights/final" \
    --data_file training/data/r13_bon.jsonl

# R14: extended RSFT (MBPP sanitized + MBPP-full prompts)
python -m training.prep_candidate_datasets --which mbpp_full
python -m training.bon_run \
    --checkpoint "tinker://<R13_ID>:train:0/sampler_weights/final" \
    --base_model meta-llama/Llama-3.1-8B \
    --n_prompts 400 --n_samples 16 --max_tokens 1024 \
    --extra_prompts training/data/mbpp_full_prompts.jsonl \
    --output training/data/r14_rsft.jsonl
python -m training.sft_train --config r9_bon_8b \
    --load_checkpoint_path "tinker://<R13_ID>:train:0/weights/final" \
    --data_file training/data/r14_rsft.jsonl

# R15, R16 as above.

# R17: BoN distillation on top of R16 (regressed in our run; included for
# full reproduction but not the submission path)
python -m training.bon_run \
    --checkpoint "tinker://<R16_ID>:train:0/sampler_weights/final" \
    --base_model meta-llama/Llama-3.1-8B \
    --n_prompts 400 --n_samples 16 --max_tokens 1024 \
    --extra_prompts training/data/mbpp_full_prompts.jsonl \
    --output training/data/r17_bon.jsonl
python -m training.sft_train --config r9_bon_8b \
    --load_checkpoint_path "tinker://<R16_ID>:train:0/weights/final" \
    --data_file training/data/r17_bon.jsonl
```

Total cumulative cost from a clean start (base Llama → R17) is ~$35-40 Tinker credits.

---

## 5. Determinism notes

- **SFT data (`r1_clean.jsonl`, `r1_clean_v2.jsonl`)**: fully deterministic given `--seed 42` and `--shuffle_buffer 10000`. Re-running with the same flags produces byte-identical JSONLs (barring HF re-shuffling an upstream source, which would show up as a different row count in the per-source stats).

- **KodCode pool**: deterministic by HF streaming order. Our `prep_candidate_datasets.py` does not take a seed — it streams in dataset order and stops at `--kodcode_max` kept rows. Re-running produces the same 30K rows.

- **BoN rollout datasets (r13_bon, r14_rsft, r17_bon)**: **non-deterministic** because they come from policy sampling at `temperature=1.0`. Two runs from the same checkpoint on the same prompts will produce different golden-row sets but statistically equivalent distributions (>95% winner pass-rate on the subprocess verifier in every one of our runs).

- **Tinker LoRA training**: deterministic given the same data, same checkpoint to seed from, same hyperparameters (the cookbook passes fixed seeds through). Two independent runs of `r15_clean_v2_8b` produce nearly-identical checkpoints.

---

## 6. Integrity: how we verified "no training on test data"

We ran three independent contamination audits at increasing rigor:

1. **13-gram word filter at build time** (applied inside `data_sources.py:build_deny_ngrams` + `is_contaminated`). Dropped 548 rows across all SFT sources — the verbatim HumanEval-solution leaks in the original raw Magicoder dataset.

2. **Deep fingerprint audit** after the fact. Generated ~7,500 distinctive fingerprints per benchmark (function signatures, `>>> example()` calls, doctest expected-outputs, first docstring sentences, canonical solution lines) and scanned every materialized JSONL via Aho-Corasick. This is what caught Magicoder-Evol's HumanEval derivative contamination (37 / 164 HumanEval items matched after the 13-gram filter passed).

3. **TA-approved 8-gram filter** (Piazza-approved method from Prof. Dhingra). Reimplemented verbatim. Flagged ~10% of our training rows, but forensic inspection confirmed every flagged 8-gram is either a shared IFEval constraint-library template, an official GSM8K train/test template overlap (different names+numbers), or a generic Python idiom. No verbatim test prompt, question, or canonical solution appears in training.

The audit scripts are NOT in this zip (per instructions to keep the zip minimal), but the audit results are in the final report PDF under *Contamination*.
