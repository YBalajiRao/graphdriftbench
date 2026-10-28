---
name: experiment-runner
description: Use when launching, resuming or logging experiments.
---

# Experiment runner

- Config: Hydra (configs/). One run = (dataset, method, seed, budget, injection schedule). Seeds 0-4.
- Launch: `python scripts/run.py experiment=<name> seed=<s>` with the `gdb` conda env.
- Output per run, never overwriting an existing run_id (abort or resume instead):
  `results/raw/<exp>/<run_id>/{config.yaml,metrics.json,acc_matrix.npy,alarms.csv}`
- Aggregate with `scripts/aggregate.py` into `results/*.csv`. These CSVs are the only source for tables and figures.
- Before launching: check free GPU memory (`nvidia-smi`; RTX 3070 has 8 GB); log peak memory (`torch.cuda.max_memory_allocated`) in metrics.json.
- Windows: `num_workers=0` in all DataLoaders; TGB data in D:\research\tgb_data; no bash-only syntax in launch scripts.
- Log the seed set for torch/numpy/random in config.yaml. Learner must never receive injection times/types or segment boundaries.
- Resuming: skip run_ids whose metrics.json exists; rerun incomplete ones into a fresh run_id.
