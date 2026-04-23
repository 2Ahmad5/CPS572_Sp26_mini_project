# R9 plan (executing now)

## Status
- R9 3-way GRPO (IF+math+code) launched from R8-BoN final
- Config: 48 groups/batch, 8 envs/group, LR 1.5e-5, KL 0.05, 60 steps
- Expected wall-clock: ~1.7 hours
- Step 0 baseline: IF=1.000, GSM=0.683, CODE=0.532, all=0.755

## After R9 completes
1. Pick best checkpoint by training reward (all-reward metric), break ties toward later steps
2. Full eval on final + best-step checkpoints
3. If best R9 > R8-BoN (1.653): update submission.json
4. If best R9 < R8-BoN: keep R8-BoN as submission, document R9 as extension attempt

## Possible follow-ups if budget allows
- R9-BoN: BoN distillation on top of best R9 checkpoint (~$6, +0.2-0.5 avg_norm possible)
- Eval additional R9 intermediate checkpoints (cheap, ~$1 each)

## Cost so far / budget
- R8 iteration: ~$22-24 spent (prior)
- R8-GRPO step 50/75 evals: ~$2
- R9 training: ~$5-8 estimated (48 groups * 8 envs * 1024 tokens * 60 steps = ~23M tokens)
- R9 eval: ~$1-2
- Remaining budget for follow-ups: ~$5-8 under $35 cap
