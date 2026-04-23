# Exploration: Beating the Baseline on IFEval / GSM8K / HumanEval

**Baselines to clear (Llama-3.1-8B, LoRA on Tinker):** IFEval 45%, GSM8K 50%, HumanEval 30% (pass@1).
**Budget:** $250 Tinker credits per student.

Sources filtered to high-signal only (official Thinking Machines docs + cookbook code, Allen AI Tulu-3 paper, Raschka's ablations, BigCode/NVIDIA/MSR dataset cards, DeepSeekMath paper). Anything labeled "hearsay" or "community" is flagged.

---

## 1. Tinker-specific ground truth (from cloned cookbook + TML blogs)

### 1a. LoRA hyperparameters — the "LoRA Without Regret" recipe (Schulman, TML, Sep 2025)

- **LR rule:** LoRA LR = **10× full-FT LR** (SL and RL both). For very short runs (~100 steps), use 15×.
- **Target modules:** **All linear layers** — MLPs are the critical component. Attention-only LoRA is strictly worse, even at matched param count.
- **Batch size:** LoRA is **less tolerant of large batches** than full FT. Keep batch size modest.
- **Rank vs. dataset size:** LoRA matches full FT on small-to-medium SFT datasets; fails on very large ones (runs out of capacity). Rank 32–64 should be sufficient for ≤500K-example LoRA SFT.
- **RL finding (huge):** "LoRA fully matches full FT for policy gradient RL, even with ranks as low as 1." This is a strong argument for doing cheap RL after SFT.
- Source: https://thinkingmachines.ai/blog/lora/

### 1b. Tinker's exact LR formula (from `tinker_cookbook/hyperparam_utils.py`)

```
base_lr_full_ft = 5e-5
lora_multiplier = 10.0
exponent_llama = 0.781   # Qwen uses 0.0775
lr = base_lr * lora_multiplier * (2000 / hidden_size) ** exponent_llama
```

For **Llama-3.1-8B (hidden=4096):** LoRA LR ≈ 5e-5 × 10 × (2000/4096)^0.781 ≈ **~2.9e-4**.
Cookbook recipes actually default to **1e-4 to 2e-4** — slightly conservative vs. the formula.

### 1c. Cookbook default SFT config (`tinker_cookbook/recipes/chat_sl/train.py`)

| Param | Default |
|-------|---------|
| model_name | meta-llama/Llama-3.1-8B |
| lora_rank | 32 |
| learning_rate | 1e-4 |
| batch_size | 256 |
| max_length | 16384 |
| num_epochs | 1 |
| save_every / eval_every | 20 steps |
| lr_schedule | linear (with warmup) |
| adam betas | 0.9, 0.95 |

Published chat_sl result on Tulu-3 mixture: **test NLL 0.50 after 1740 steps** with LR=5e-4, batch=128, rank=64. (No downstream benchmark scores reported.)

### 1d. Cookbook Math RL recipe — directly relevant

`tinker_cookbook/recipes/math_rl/train.py` is a GRPO-style importance-sampling RL loop.
**Published results in the cookbook README:**
- **GSM8K: 90.9% accuracy after 220 steps** — group_size=64, groups_per_batch=32, LR=8e-5, max_tokens=1024.
- MATH: 76.8% after 180 steps — group_size=16, groups_per_batch=64, LR=2e-5.

Reward is **binary** via `\boxed{}` extraction + sympy/math_verify grading.

### 1e. Cookbook has IFEval + GSM8K benchmarks built in, but NOT HumanEval

- `tinker_cookbook/eval/benchmarks/ifeval.py` — loads google/IFEval, uses full verifier, returns strict-all accuracy.
- `tinker_cookbook/eval/benchmarks/gsm8k.py` — loads openai/gsm8k test, boxed-answer grading.
- HumanEval not present; **MBPP is the only built-in code benchmark**. The project's `evaluation/` folder has its own HumanEval runner (Inspect AI).

### 1f. Multi-task mixing — not first-class in cookbook

- `tinker_cookbook/distillation/datasets.py:49-82` has `CompositeDataset` for RL: interleaves N builders by `groups_per_batch_list`. Reusable for multi-task RL.
- For SFT, the idiom is "use a pre-mixed HF dataset like tulu-3-sft-mixture." No curriculum / weighted-sampling SFT builder ships.

### 1g. On-Policy Distillation (Lu et al., TML, Oct 2025) — potential extension

- Teacher grades each student token via reverse-KL; dense per-token advantages. **9–30× cheaper than SFT, ~50–100× cheaper than RL** for the same lift.
- Reported: AIME'24 60% → 74.4%; IFEval recovery to 83%.
- **Implemented in tinker-cookbook/recipes/distillation.**
- Caveat: needs a stronger teacher (Llama-3.1-70B-Instruct or similar) → extra inference spend. Experiments used Qwen3, not Llama-3.
- Source: https://thinkingmachines.ai/blog/on-policy-distillation/

---

## 2. Tulu-3 as the reference recipe (Allen AI, Nov 2024)

Tulu-3 trained **full-FT Llama-3.1-8B** and published all data + hyperparameters — gold-standard reference.

- Final scores: **IFEval 82.4% (prompt-loose), GSM8K 87.6% (8-shot CoT), HumanEval 83.9% (pass@10), MATH 43.7%, MMLU 68.2%.**
- Pipeline: SFT → DPO → **RLVR** (RL with verifiable rewards; targets IFEval + GSM8K + MATH).
- **RLVR hyperparams:** LR **3e-7**, KL β ∈ {0.1, 0.05, 0.03, 0.01}, effective batch 224, max 2048 tokens, 100k episodes. Reward = exactly the IFEval verifier / GSM8K exact-match checker.
- **RLVR gain over DPO:** +3.3 GSM8K, +1.7 MATH, +1.3 IFEval. Modest individually, but compounds.
- Sources: https://arxiv.org/abs/2411.15124 ; https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B

**Caveat:** Tulu-3 numbers use richer eval protocols than the project (8-shot CoT vs. zero-shot for GSM8K; pass@10 vs. pass@1 for HumanEval). Expect **our evaluated numbers to be lower** than Tulu-3's headline figures.

**Actionable:** Tulu-3 uses identifiable sub-mixtures on HuggingFace. These are our highest-leverage datasets — see §3.

---

## 3. Dataset recommendations (ranked, all published-result-backed)

### Math → GSM8K

1. **nvidia/OpenMathInstruct-2** (14M, CC-BY-4.0) — OpenMath2-Llama3.1-8B reached **91.7% GSM8K, 67.8% MATH**. Contamination explorer published. Use `train_1M` split.
2. **meta-math/MetaMathQA** (395K, MIT) — MetaMath-Mistral-7B: 77.7% GSM8K. Explicit no-test-contamination claim.
3. **microsoft/orca-math-word-problems-200k** (200K, MIT) — GPT-4-generated. Run n-gram check vs. GSM8K test first.
4. **allenai/tulu-3-sft-personas-math-grade** (50K) — persona-grounded grade-school math; aligns with GSM8K style.

### Code → HumanEval

1. **ise-uiuc/Magicoder-Evol-Instruct-110K** (Apache-2.0) — explicitly decontaminated via BigCode pipeline.
2. **ise-uiuc/Magicoder-OSS-Instruct-75K** (MIT) — seeded from real OSS; Python-heavy. Pairs with #1.
3. **bigcode/self-oss-instruct-sc2-exec-filter-50k** (ODC-BY) — **execution-validated** responses (very high label quality).
4. **allenai/tulu-3-sft-personas-code** (35K) — HumanEval-style Python completion prompts.

**Avoid as sole source:** `nvidia/OpenCodeInstruct` (5M, the project's "suggested" code dataset) has **no documented HumanEval decontamination** in its card. If used, run contamination check first.

### IFEval → Instruction following

1. **allenai/tulu-3-sft-personas-instruction-following** (30K, ODC-BY) — **targets the exact IFEval constraint classes** (length, format, JSON, keywords, case, punctuation). Documented in the Tulu-3 paper as the primary driver of their IFEval score. **Single highest-leverage dataset in this list.**
2. **argilla/ifeval-like-data** (filtered subset 56K) — generated via MagPie + Qwen2.5-72B, filtered by IFEval verifier itself. Check overlap with google/IFEval before training.

### Anti-forgetting / general quality

- **HuggingFaceH4/no_robots** (10K human-written, CC-BY-NC-4.0 — research-only)
- **GAIR/lima** (1K curated, CC-BY-NC-SA)

### A concrete 400–500K-example recipe

| Bucket | Datasets | Count |
|--------|----------|-------|
| Math | OpenMathInstruct-2 (200K) + MetaMathQA (100K) + tulu-3-personas-math-grade (50K) | 350K |
| Code | Magicoder-Evol-110K + Magicoder-OSS-75K (Python-only) + tulu-3-personas-code (35K) | 220K |
| IF | tulu-3-personas-instruction-following (30K) + argilla/ifeval-like filtered (30K) | 60K |
| Anti-forget | no_robots (10K) + lima (1K) | 11K |

Subsample to hit ~400K total; preserve relative ratios.

---

## 4. LoRA ablation findings (Raschka) — cross-validates TML

Source: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms

- **LoRA on all linear layers strictly beats q,v-only.** (Confirms TML.)
- **r=256 beat r=8/32/64/128/512 on Alpaca**; no clean heuristic — treat rank as a hyperparameter to tune.
- "alpha = 2×rank" rule is not optimal — r=256, α=128 (0.5×) outperformed.
- **Multi-epoch instruction tuning degrades performance** — 1 epoch is usually the right default.
- Optimizer choice barely matters (AdamW vs. SGD).
- Dropout 0.05 used but untested.
- **LIMA 1K ≈ Alpaca 50K** — data quality dominates quantity past a point.

---

## 5. RL / post-SFT extensions — ranked by lift-per-dollar

### Rank 1 (best value): **Rejection-sampling fine-tuning (RFT / STaR-style)**

- Sample N=8 completions per train problem with your SFT checkpoint; keep correct ones; run 1 more SFT epoch on them.
- No value network, no advantage estimation, no KL term. Reuses the SFT infrastructure.
- Community reports **~80% of GRPO's gain on GSM8K at ~1/10 the compute.** (Hearsay — flagged.)
- Pairs with the project's existing `evaluation/` infrastructure.

### Rank 2: **GRPO on IFEval only**

- IFEval reward is a **free Python verifier** (no judge, no rollout-of-rollouts). Per-step RL cost is dominated by generation, not reward.
- Tulu-3 showed clear IFEval gains under RLVR with **very low LR (3e-7)** and KL β ~0.05.
- Cookbook already has the Math-RL training loop; swap in `tinker_cookbook/eval/benchmarks/ifeval.py` as the env.
- **Expected lift: +3–8 points on IFEval** (extrapolated from Tulu-3 paper; Tulu-3 saw +1.3 on top of an already-DPO'd checkpoint, plain SFT→RLVR should see more).
- LoRA Without Regret: rank-1 LoRA suffices for RL → rent the smallest possible adapter.

### Rank 3: **GRPO on GSM8K**

- Also verifiable (boxed-answer regex + sympy). Cookbook recipe is literally already written and ships with 90.9% GSM8K at 220 steps as a precedent (though that's from an Instruct-tuned starting point, not a fresh SFT).
- Budget: ~2–4 H100-hours; fits in $250 if IFEval RL is skipped.

### Rank 4 (high risk): **GRPO on HumanEval**

- HumanEval is tiny (164 problems) and **known-leaky**. Training signal has to come from a surrogate corpus (MBPP, APPS, CodeContests) — the project doesn't ship a code-RL environment pointing at any of these.
- Cookbook has Code RL recipe (`tinker_cookbook/recipes/code_rl/`) using DeepCoder data — workable but expensive (sandboxed exec per rollout).
- **Skip unless HumanEval is well above baseline and math/IF are already maxed.**

### Rank 5 (advanced extension): **On-Policy Distillation**

- High compute-efficiency, implemented in cookbook. But needs a stronger teacher; Qwen3 was the tested model family, not Llama-3.1-8B.
- **Probably only worth attempting if SFT + cheap RL both hit their targets and there is remaining budget.**

### What to skip

- **DPO**: weaker on verifiable-reward tasks than RLVR (per Tulu-3 ablations). Useful for chattiness, not our three tasks.
- **Full-FT**: exceeds Tinker LoRA-only tier for most practical budgets.
- **PPO with value head**: heavier than GRPO/RLOO at comparable quality.

---

## 6. Concrete recommended training plan

**Phase 0 — dev loop (Llama-3.2-3B)**
Validate mixing, templates, and eval hooks with small-scale runs (LR from `get_lr` formula, rank 16, batch 64, ~300 steps). Confirm ≥ published 3B base on all three tasks.

**Phase 1 — SFT on Llama-3.1-8B (LoRA)**
- Dataset: mix from §3 (~400K total, ratios ~65/30/10/3 math/code/IF/general).
- Hyperparams: rank **32**, LoRA on **all linear layers**, LR **2e-4**, batch **128**, seq length **4096** (sufficient; 16384 is wasteful), **1 epoch**, linear LR schedule with 3% warmup.
- **Save every 500 steps and evaluate each** — the best checkpoint is often not the last (confirmed by project README and Raschka's 1-epoch finding).

**Phase 2 — rejection-sampling SFT pass**
- Generate 8 completions per train problem (GSM8K train + IFEval-alike train), filter for correctness, SFT for one more pass at lower LR (5e-5) for ~100 steps.

**Phase 3 — GRPO on IFEval (if budget permits)**
- Clone `recipes/math_rl/train.py`; point env at `ifeval.py` benchmark; reward = IFEval strict-all.
- LR **1e-5** (conservative; Tulu-3 used 3e-7 on full-FT — LoRA LR ~10× → ~3e-6, can be pushed higher); KL β **0.05**; group_size **16**; groups_per_batch **32**; max_tokens **1024**; 100–300 steps.
- **Re-evaluate all three tasks** after RL (project PROJECT.md explicitly warns about task degradation).

**Phase 4 (optional) — GRPO on GSM8K**
- Same hyperparams as cookbook's proven config (group_size=64, lr=8e-5, max_tokens=1024). ~220 steps.

### Guardrails
- Budget gate: do a cost dry-run on 3B before 8B.
- Contamination: run n-gram check against IFEval/GSM8K test/HumanEval on any dataset whose card doesn't explicitly document decontamination (Orca-Math, Tulu-3 personas, OpenCodeInstruct, argilla/ifeval-like).
- Eval intermediate checkpoints: PROJECT.md explicitly flags that the final checkpoint is often not the best.

---

## 7. Citations (verified live)

- LoRA Without Regret — https://thinkingmachines.ai/blog/lora/
- On-Policy Distillation — https://thinkingmachines.ai/blog/on-policy-distillation/
- Tinker docs (index) — https://tinker-docs.thinkingmachines.ai/
- tinker-cookbook (cloned locally at `research_sources/tinker-cookbook`)
- Tulu-3 paper — https://arxiv.org/abs/2411.15124
- Tulu-3-8B model card — https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
- Raschka LoRA tips — https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- DeepSeekMath (GRPO) — https://arxiv.org/abs/2402.03300
- OpenMathInstruct-2 — https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- MetaMathQA — https://huggingface.co/datasets/meta-math/MetaMathQA
- Magicoder-Evol-Instruct-110K — https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K
- Magicoder-OSS-Instruct-75K — https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- BigCode self-oss-instruct 50k — https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k
- Tulu-3 persona IF subset — https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following
- Tulu-3 persona math-grade — https://huggingface.co/datasets/allenai/tulu-3-sft-personas-math-grade
- Tulu-3 persona code — https://huggingface.co/datasets/allenai/tulu-3-sft-personas-code
- argilla/ifeval-like-data — https://huggingface.co/datasets/argilla/ifeval-like-data

## 8. Flagged unverified claims (do NOT rely on without verification)

- "RFT gets 80% of GRPO's gain at 1/10 cost on GSM8K" — community hearsay, plausible but not peer-reviewed.
- Expected RL-on-IFEval lift of "+3–8 points" — extrapolated; actual result depends heavily on SFT starting point.
- RL agent's summary of DeepSeekMath GRPO numbers was from training-data recall (Jan 2026 cutoff). Spot-check the 2402.03300 paper before citing.

---

## 9. Project-specific eval protocol (from reading `evaluation/*.py`)

All recommendations above must be filtered through these concrete eval facts:

1. **IFEval (541 prompts, Inspect AI task `inspect_evals/ifeval`).** Returns 4 sub-metrics: `prompt_level_strict`, `prompt_level_loose`, `inst_level_strict`, `inst_level_loose`. Baseline 45% is the **average** of the four. Loose is easier than strict — if ablating, targeting `prompt_strict` also lifts the other three. Training should emphasize exact constraint compliance (length counts, keyword includes, format/case/punctuation rules).

2. **GSM8K (1319 test items, `gsm8k(fewshot=0)`).** **Zero-shot** — no format priming. SFT data must explicitly teach the `\boxed{answer}` style, or Inspect AI's scorer will miss correct answers. Filter / transform training examples to end with boxed numeric answers. This alone typically swings GSM8K several points.

3. **HumanEval (164 problems, `inspect_evals/humaneval` with `sandbox="local"`).** Pass@1, greedy (temperature=0 by default), single generation per problem. Low variance — clean function-completion matters more than sampling diversity. Project ships its own HumanEval eval; cookbook has MBPP but not HumanEval.

4. **Generation settings are a submission parameter.** `submission.json` records `temperature` and `top_p`; the grader re-runs with exactly those. **Free tunable at submission time** — ablate e.g. {0.0, 0.3, 0.5} × {1.0, 0.9} on a held-out slice or full eval on final checkpoint before submitting. Often worth 1–3 points for no additional training cost. Temperature 0.0 is usually best for IFEval (deterministic constraint checking) and HumanEval (greedy pass@1), while GSM8K sometimes benefits from small temperature for self-consistency — but self-consistency isn't in the eval pipeline here, so stick near 0.

5. **Renderer is auto-selected (`get_recommended_renderer_name`).** For Llama-3.1-8B the Llama-3 renderer is chosen, wrapping prompts in `<|start_header_id|>user<|end_header_id|>...<|eot_id|><|start_header_id|>assistant<|end_header_id|>...`. **SFT data must be `{"role", "content"}` message format.** All recommended datasets in §3 support this natively, but confirm the `map_fn` in the cookbook's `SupervisedDatasetFromHFDataset` produces message lists, not raw strings.

6. **`max_tokens=1024` at eval.** Long chain-of-thought risks truncation — especially for GSM8K and HumanEval. Recommend filtering SFT rows whose responses exceed ~700–800 response tokens. Teach the model concise reasoning. OpenMathInstruct-2 in particular has some very long rationales that should be sampled away.

7. **Evaluator concurrency is high (`max_connections=512`).** Full eval on all three tasks is fast — cheap to evaluate intermediate checkpoints. Take advantage: save every 500 steps during SFT and evaluate each; PROJECT.md explicitly says the final checkpoint is often not the best.

8. **Baseline 45/50/30 was achieved by the TA with "mixed SFT on Llama-3.1-8B"** (per PROJECT.md). So clearing baseline does not require any RL or exotic extension — disciplined SFT alone is sufficient. RL/distillation are for the last 1 point per task (competitive / leaderboard credit).

## 10. Follow-ups worth pursuing next

1. **Fetch the correct `sl-hyperparams` page URL** — current guesses returned 404. The SL tutorial path likely lives under `/tutorials/advanced/sl-hyperparams/` per the docs sitemap.
2. **Verify Tulu-3 SFT hyperparameters** in the paper appendix (PDF parsing failed this run) — particularly the LR used for the 8B SFT stage.
3. **Contamination audit** for every chosen dataset vs. the three benchmarks before any real training.
4. **3B dry run** to validate the whole pipeline end-to-end, then port configs to 8B.
