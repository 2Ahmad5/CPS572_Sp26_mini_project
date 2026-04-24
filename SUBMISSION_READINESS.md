# Submission readiness checklist

Against [PROJECT.md](PROJECT.md) §Submission.

## 1. `submission.json` (Autograder) — ✅ READY

```json
{
  "checkpoint_path": "tinker://10952bf4-613d-5f99-86ff-745e6775fb01:train:0/sampler_weights/final",
  "base_model": "meta-llama/Llama-3.1-8B",
  "settings": {"temperature": 0.0, "top_p": 1.0},
  "ifeval":    { "metrics": { "google/IFEval/final_acc": 0.7301 } },
  "gsm8k":     { "metrics": { "openai/gsm8k/accuracy":   0.7756 } },
  "humaneval": { "metrics": { "openai/openai_humaneval/accuracy": 0.5305 } }
}
```

- IFEval **73.01%** (baseline 45.0% → +28.01pp, **full-credit threshold met**)
- GSM8K **77.56%** (baseline 50.0% → +27.56pp, **full-credit threshold met**)
- HumanEval **53.05%** (baseline 30.0% → +23.05pp, **full-credit threshold met**)
- avg_norm **1.6473**

Produced by the standard command per README:
```
python evaluation/eval_all.py --checkpoint_path ... --base_model meta-llama/Llama-3.1-8B
```
with no `--limit` flag (full eval), greedy decoding (project-recommended).

**Checkpoint is R16** — R15 full clean SFT (r1_clean_v2.jsonl, no Magicoder-Evol contamination) + 3-way mixed GRPO (IFEval/math/code verifiers, 15 cycled steps). See `423iter.md` and `R14_TECHNIQUES.md` for full provenance.

Uploaded as-is to Gradescope Autograder. Grader re-runs the eval on the checkpoint and settings we provide; since our settings match project defaults (`temperature=0`, `top_p=1`) the results will reproduce exactly.

## 2. Final Report (PDF) — ⚠ TEAM ACTION

The report must include 5 sections. We have the raw material for all five:

### (a) Methodology — **drafted in `R14_TECHNIQUES.md`**

Covers: base model + LoRA, LR formula, multi-source SFT data mix, ANSWER-format alignment, decontamination (13-gram + Aho-Corasick), `max_length=2048`, 3-way mixed GRPO verifiers, subprocess code verifier, BoN rejection-sampling SFT, greedy eval. See `R14_TECHNIQUES.md`. Note the final submission is R16, not R14, so adjust the text — R16 is R15 SFT + 3-way GRPO (no BoN); BoN was tried as R17 and **regressed**.

### (b) Extensions — ✅ we have multiple

1. **Multi-task GRPO with a novel 3-way code verifier (`training/rl_code_env.py`).** Most code RL recipes use Modal or SandboxFusion; we use a 80-line subprocess-based verifier that lets the model train against real MBPP test-exec feedback with no external service.
2. **Aho-Corasick deep contamination audit (`training/verify_no_test_leak_deep.py`).** Detects Evol-Instruct-style derivative test-set leakage that 13-gram filters miss. Led to the discovery that `ise-uiuc/Magicoder-Evol-Instruct-110K` leaks 22% of HumanEval through paraphrased problem statements.
3. **Rejection-sampling SFT (`training/bon_run.py`).** Proven pattern — R13→R14 on the contaminated lineage gave +0.039 avg_norm from 355 model-generated, subprocess-verified MBPP rollouts.
4. **KodCode-V1 integration with strict pre-filtering.** `training/prep_candidate_datasets.py` applies `benchmark_similarity < 0.5` + HumanEval function-name blocklist + 13-gram filter. Clean HumanEval gain of +6.71pp at SFT stage (R11 46.34 → R15 53.05).

### (c) Results and Analysis — covered by `423iter.md` and the run ladder table

| Run | Description | IFEval | GSM8K | HumanEval | avg_norm |
|---|---|---|---|---|---|
| baseline 8B (no FT) | | ~22.2 | ~9.7 | ~0.0 | 0.11 |
| R6 (dirty SFT) | r1_enhanced with Magicoder-Evol | 69.58 | 77.33 | 51.22 | 1.6000 |
| R10 (dirty RL) | 3-way GRPO on R9-BoN | 71.07 | 80.67 | 55.49 | 1.6808 |
| R11 (clean SFT) | r1_clean, Magicoder removed | 70.07 | 75.51 | 46.34 | 1.5374 |
| R12 (clean +RL) | +3-way GRPO | 71.48 | 77.94 | 46.34 | 1.5640 |
| R13 (clean +RL+BoN) | +MBPP BoN distill | 71.74 | 78.01 | 47.56 | 1.5800 |
| R14 (clean +RSFT) | +expanded RSFT | 71.30 | 80.29 | 50.00 | 1.6190 |
| R15 (clean SFT + KodCode) | r1_clean_v2 | 70.62 | 77.63 | 53.05 | 1.6301 |
| **R16** ← **submission** | **+3-way GRPO** | **73.01** | **77.56** | **53.05** | **1.6473** |
| R17 (R16 +BoN) | −0.02 regression | 72.82 | 78.77 | 50.61 | 1.6269 |

Negative results worth writing up:
- **Magicoder-Evol derivative contamination.** 13-gram filter passed, deep audit caught 22% HE paraphrases. Dropping it cost −5pp HE legitimately.
- **Single-task RL regresses multi-task performance** (R3 gave +3.8 IF but −1.0 GSM / −1.8 HE on 3B). Required moving to 3-way mixed GRPO.
- **BoN didn't always help.** On R13 (HE 47.56), BoN added +2.44 HE. On R16 (HE 53.05, already KodCode-enriched), BoN *regressed* HE by −2.44pp — MBPP-style distillation pulled the code distribution away from KodCode's already-strong shape.
- **MBPP sanitized is tiny** (120 rows after HE decontam). The default `InterleavedRLDatasetBuilder` hits smallest-source exhaustion; R9 terminated at step 7 when we expected 60. Required `total_batches` override to force cycling.
- **Tinker platform outages** during R15 SFT caused a 48-min single-step stall and ~30-min total delay. Training recovered each time, but the `tinker_cookbook/utils/logtree.py` HTML writer crashed on a non-ASCII character (`√`) under Windows cp1252 — we patched the cookbook to open log files with `encoding="utf-8"`.

### (d) LLM Usage — ⚠ TEAM FILLS IN EXACTLY

Required disclosure:
- Used **Claude Code** (Claude Opus 4.7 / Sonnet) for code writing, debugging, research, planning, and iteration orchestration throughout this project.
- Specific LLM-assisted code:
  - `training/rl_code_env.py`, `training/prep_candidate_datasets.py`, `training/verify_no_test_leak*.py`, `training/final_audit.py`, `training/post_r9.py`, `training/trace_leaks.py`
  - Modifications to `training/bon_run.py` (extra_prompts support), `training/rl_train.py` (R10/R12/R16 configs, CodeRLDatasetBuilder wiring), `training/sft_train.py` (R11/R15/R8-RFT/R8-BoN/R9-BoN configs)
  - In-venv patch to `tinker_cookbook/utils/logtree.py` for UTF-8 encoding
  - All iteration markdown docs (`419iter.md`, `423iter.md`, `R14_TECHNIQUES.md`, this file)
  - The two-pass contamination audit scripts (basic + Aho-Corasick fingerprint)
- Human-authored / pre-existing:
  - `training/rl_ifeval.py`, `training/rl_math_env.py` (original project code)
  - `training/data_sources.py`, `training/build_dataset.py` (originally human; later modified with LLM assistance to add `r1_clean` config and `SOURCES_EXCLUDED_FOR_STRICT_DECONTAM`)

### (e) Tinker Feedback — ⚠ TEAM FILLS IN

Suggested answers based on our experience:

- **Hardest part / where stuck:**
  - Platform reliability — 3 Tinker connection outages during R15 SFT (48-min heartbeat failure + smaller hiccups). Training recovered but it was nerve-wracking. We added process-liveness monitoring and saved frequent checkpoints.
  - Windows encoding issue — `tinker_cookbook/utils/logtree.py` opens log files without `encoding="utf-8"`, crashing on non-ASCII chars in rollouts under cp1252. We had to patch the cookbook in-place.
  - Dataset exhaustion behavior — `InterleavedRLDatasetBuilder` silently terminates training when the smallest source runs out. R9 stopped at step 7 expecting 60. Fixed via `total_batches=` override, but a clearer warning would've saved time.
- **Used tinker-cookbook:** Yes. Helpful: `rl.interleaved.InterleavedRLDatasetBuilder`, `rl.problem_env.ProblemEnv`, `supervised.train.main`. Missing: a clear example of how to plug a subprocess-based code verifier into GRPO without depending on Modal/SandboxFusion. Also: the cookbook's `code_rl` recipe is Qwen/DeepCoder-flavored and didn't work for our Llama-3.1-8B base without meaningful adapter work.
- **One thing to make Tinker significantly better:** built-in verbose metrics streaming during rollouts (right now `metrics.jsonl` only lands after each batch; during a 50s rollout there's no feedback). That plus per-source reward monitoring in `InterleavedRLDatasetBuilder` would make debugging multi-task runs much easier.
- **Use Tinker again:** Yes, for fine-tuning research. The LoRA + managed inference combo is productive. For production training I'd want more reliability guarantees.

## 3. Code (ZIP) — team action

Files to include (match what's on the `suhas-testing-progress-4-23` branch plus this report):

- `training/` (all `.py` except `__pycache__/`)
- `evaluation/` (all `.py` + `submission.json` + audit reports in `data/`)
- `README.md`, `PROJECT.md`
- `R14_TECHNIQUES.md`, `SUBMISSION_READINESS.md` (this file)
- `419iter.md`, `423iter.md` (full iteration history)
- `tasks/r14_plan.md`, `tasks/session_summary.md`

Exclude:
- `.venv/`, `.env`, `.claude/`
- `logs/` (huge training logs)
- `training/data/*.jsonl` (huge materialized datasets — rebuildable from scratch via `training/build_dataset.py`)
- `evaluation/inspect-logs/` (huge per-eval inspect transcripts)
- `evaluation/submission_*.json` intermediate per-run submissions (already gitignored)
- `evaluation/r1_checkpoint_eval.json` / `r3_` / `r6_` (stale intermediate)

A one-line zip command that respects the above:
```
git archive --format=zip -o code.zip HEAD
```
would ship everything tracked in git (already gitignore-filtered correctly).

## Final hard-constraint check

Per PROJECT.md §Suggested Datasets: *"we will review your code submissions — any team found to have trained on test data will receive a score of **0** for the entire project."*

See `training/data/FINAL_contamination_report.md` — our end-to-end audit across all 9 training artifacts (SFT JSONLs + RL prompt sources) shows:
- IFEval: 48/541 (8.9%) — all argilla/gsm8k shared-constraint-library phrases.
- GSM8K: 2/1319 (0.15%) — coincidental generic phrases.
- HumanEval: 12/164 (7.3%) — all generic-code false positives (Python idioms like `return fib(n-1)+fib(n-2)`).

No distinctive HumanEval doctest calls, function signatures, or example outputs appear in any training artifact. The 13-gram + Aho-Corasick fingerprint audit is stricter than BigCode's 13-gram standard. **We pass both the letter and the spirit of the rule.**
