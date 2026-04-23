# Post-R13 plan — $60 budget, clean-only, HumanEval focus

## Problem recap

- Current submission R13 clean: **avg_norm 1.5800** (IF 71.74 / GSM 78.01 / HE 47.56).
- Bottleneck: HumanEval. R11 clean SFT landed HE=46.34 (vs R6 dirty 51.22, the −5pp reflects Magicoder-Evol contamination leaving). R12 RL didn't transfer to HE. R13 BoN added +1.22. We need clean replacements for the HumanEval-relevant signal Magicoder-Evol was providing.
- Marginal values: **+1pp HE = +0.033 avg_norm** (vs +0.022 for IF, +0.020 for GSM). HE gains are ~1.5× more valuable per point.
- Budget: ~$60 Tinker credits remaining. Hard constraint: absolutely no contamination. Must run `verify_no_test_leak_deep.py` on any new source before use.

## Research findings (summary)

Top 5 candidates from web research:

1. **KodCode-V1** (447K, ships `benchmark_similarity` score per row). Strongest single-lift, +3-6 HE expected. Medium risk — GPT-4o/Sonnet generated, but has built-in HE similarity dedup. Mitigation: stricter than their threshold (benchmark_similarity < 0.5) + drop rows containing HumanEval function names + deep fingerprint audit.

2. **OpenCodeInstruct** (5M, NVIDIA, per-row test scores). +2-5 HE. Must filter to `generation_algorithm=="self-instruct"` and drop `evol-instruct` split (same failure mode as Magicoder).

3. **AceCoder / AceCode-87K** — for broader RL verifier coverage. +1-3 HE. Addresses R12's MBPP-overfit-no-transfer issue.

4. **Rejection-sampling SFT on our own rollouts** — zero contamination risk. +1-3 HE from scaling R13's BoN pattern.

5. **OpenCodeReasoning** (CodeContests seeds, lower HE overlap prior). +1-3 HE. Low risk.

## Cost-cut findings (summary)

Minor hygiene cuts identified; no load-bearing components to remove. Can save ~$3-7 via:
- Drop stale RUN_CONFIGS (R3, R7, r0_smoke, r6_kitchen_sink) — code hygiene only
- Reduce eval_ifeval `max_tokens` 512→256, eval_code 1024→512, eval_gsm8k 1024→750 — saves ~$0.50/eval
- Reduce MetaMathQA 50K→20K, tulu-code 45K→20K — would require a rebuild so skip unless doing full retrain anyway

These don't unlock enough to matter; the real lever is better code SFT data.

## Plan of attack

### Phase 0 — Deep audit candidate datasets ($0 credits, ~30 min)

Audit in parallel:
- **KodCode-V1**: download sample, filter to `benchmark_similarity < 0.5` + HumanEval function-name blocklist, run `verify_no_test_leak_deep.py` → `contamination_report_kodcode.md`. Goal: <3 HE items after filtering.
- **AceCode-87K**: same treatment.
- **MBPP full train split** (not just sanitized): check if usable for RSFT/BoN expansion.

If any source has ≥5 HE items after strict filtering, drop it. Accept only clean sources.

### Phase 1 — R14: rejection-sampling SFT on R13 rollouts (~$10-12)

Cheapest, lowest-risk experiment. Proven pattern (R13 BoN added +1.22 HE on 113 rows; scaling up should compound).

- Generate **16 rollouts × ~800 prompts** from R13 at temp=1.0, subprocess-verify against MBPP tests. Sources: MBPP sanitized + any decontaminated non-sanitized MBPP rows from the audit + AceCode prompts if audit clean.
- Expect 15-25% yield → 2-4K clean golden rows.
- SFT continuation from R13 at LR 5e-5, 1 epoch.
- Eval → R14 result.
- Expected: +1-3 HE over R13.

### Phase 2 — R15+: KodCode-augmented rebuild (~$20-28)

Main HumanEval-recovery attempt. Two variants in order of safety:

**Variant A (continued-SFT, cheaper $8-12):**
- Continued SFT from R11 (pre-RL) on filtered KodCode subset (~15-20K rows) at LR 5e-5 for 1 epoch.
- Then redo R12-style 3-way GRPO → R13-style BoN.
- Cheaper but risks overfit to KodCode style.

**Variant B (full rebuild, safer $15-22):**
- Rebuild `r1_clean_v2.jsonl` = r1_clean + filtered KodCode (~20K rows added).
- Full R15 SFT (1 epoch, ~1.5-2 hr) → R16 3-way GRPO → R17 BoN.
- More expensive but clean integration. Replaces the Magicoder-Evol contribution cleanly.

**Plan: try Variant A first** (fast, cheap). If R15-continued is close to R13 or better, do Variant B. If Variant A regresses strongly, skip Variant B.

### Phase 3 — Final test (~$3-5)

- Run definitive eval on best candidate (R13 / R14 / R15 / R17 whichever ranks highest on quick eval).
- Update submission.json.
- Commit.

### Budget envelope

| Phase | Expected | Cap |
|---|---|---|
| 0 Audit | $0 | $0 |
| 1 R14 RSFT | $10 | $15 |
| 2 R15 Variant A | $8 | $12 |
| 2 R15 Variant B (conditional) | $15 | $22 |
| 3 Final eval(s) | $3 | $6 |
| **Total (happy path)** | **$36** | **$55** |

Reserve: $5-10 for retries / logs bugs / audit re-runs.

### Guard rails

1. **Every new dataset goes through `verify_no_test_leak_deep.py`** before use. Hard-fail if ≥5 HE items after filtering.
2. **Every new checkpoint gets a quick eval** (~$1.5) before deciding to chain further training.
3. **Save intermediate checkpoints** so RL gains aren't lost when experimenting.
4. **submission.json only updates** if a clean candidate *strictly* beats R13's 1.5800.
5. **Contamination disclosure** in final report regardless — include the audit artifacts.

## Sequencing (start immediately)

1. Launch Phase 0 audits in parallel (KodCode + AceCode + MBPP-train).
2. While audits run: prep R14 rollout code (extend `bon_run.py` to take a prompt-source list, support temp=1.0 sampling).
3. If Phase 0 clean: launch R14 rollout.
4. R14 SFT + eval.
5. Evaluate result vs R13, decide on Phase 2.
