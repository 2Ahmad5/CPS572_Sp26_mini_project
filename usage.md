## What was wrong

The README says `python evaluation/eval_all.py` but the scripts use absolute imports (`from evaluation.eval_ifeval import ...`). **You need to use `-m` module syntax** instead:

```bash
# WRONG (from README):
python evaluation/eval_all.py ...

# CORRECT:
python -m evaluation.eval_all ...
python -m evaluation.train_and_publish ...
```

## Steps 1-2 completed successfully

| Step | Status | Command |
|------|--------|---------|
| Venv + deps | Already done | `source .venv/Scripts/activate` |
| API key | Set | `TINKER_API_KEY` is configured |
| Step 1: Baseline eval | Passed | `python -m evaluation.eval_all --base_models meta-llama/Llama-3.2-3B --limit 5` |
| Step 2: Toy training | Passed | `python -m evaluation.train_and_publish` |
| Step 2: Eval checkpoint | Passed | `python -m evaluation.eval_all --checkpoint_path "tinker://..." --base_model meta-llama/Llama-3.2-3B --limit 5` |

Checkpoint path: `tinker://620d6a86-2c01-565a-b9d5-b19cf4fd8fc2:train:0/sampler_weights/demo`

## Testing loop for iteration

```bash
# Always activate first
source .venv/Scripts/activate

# 1. Train (modify train_and_publish.py with real data/params)
python -m evaluation.train_and_publish --num_steps 50 --lr 1e-4 --rank 32

# 2. Quick eval (5 samples, ~30 sec) — use for fast iteration
python -m evaluation.eval_all \
  --checkpoint_path "tinker://YOUR_CHECKPOINT" \
  --base_model meta-llama/Llama-3.2-3B \
  --limit 5

# 3. Medium eval (~20 samples) — when results look promising
python -m evaluation.eval_all \
  --checkpoint_path "tinker://YOUR_CHECKPOINT" \
  --base_model meta-llama/Llama-3.2-3B \
  --limit 20

# 4. Full eval (no --limit) — for final submission
python -m evaluation.eval_all \
  --checkpoint_path "tinker://YOUR_CHECKPOINT" \
  --base_model meta-llama/Llama-3.1-8B

# You can also eval individual tasks for faster feedback:
python -m evaluation.eval_ifeval --checkpoint_path "tinker://..." --base_model meta-llama/Llama-3.2-3B --limit 10
python -m evaluation.eval_gsm8k --checkpoint_path "tinker://..." --base_model meta-llama/Llama-3.2-3B --limit 10
python -m evaluation.eval_code --checkpoint_path "tinker://..." --base_model meta-llama/Llama-3.2-3B --limit 10
```

The checkpoint path is printed after training and saved in `evaluation/checkpoint_info.json`. Use `--limit 5` for rapid iteration, then scale up once you see improvement.
