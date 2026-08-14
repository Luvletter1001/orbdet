# Orbdet Agent Instructions

## Shell commands

- Prefix every non-interactive shell command with `rtk`.

## Formal training gate

- Do not start formal training, a long-running schedule, a queue, `tmux`,
  `nohup`, or an automatic resume unless the user explicitly authorizes formal
  training in the current turn.
- The scripts under `scripts/smoke/` are bounded smoke launchers, not formal
  training entry points.
- Keep `FORMAL_TRAINING_NOT_STARTED.md` accurate. Do not change its status as a
  side effect of setup, diagnosis, evaluation, or code review.

## GPU and data safety

- For multi-GPU runs on physical GPUs 8 and 9, set
  `CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, and `NCCL_IB_DISABLE=1`.
- Never terminate or modify another user's GPU process.
- Treat `data/DOTA-v1.0/` as linked source data. Do not rewrite annotations or
  images in place.

