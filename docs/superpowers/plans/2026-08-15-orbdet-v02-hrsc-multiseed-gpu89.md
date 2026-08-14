# Orbdet-v0.2 HRSC Multi-Seed GPU 8/9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the missing clean-HRSC Orbdet-v0.2 seeds 42 and 2026 sequentially from a frozen local snapshot on physical GPUs 8/9.

**Architecture:** A dedicated Git worktree freezes the proven v0.2 implementation. Two inheritance-only configs and guarded launchers feed a fail-fast queue with durable success/failure markers and unique work directories.

**Tech Stack:** MMEngine Config, MMRotate, PyTorch distributed, Bash, tmux, pytest.

---

## File map

| File | Responsibility |
|---|---|
| `tests/test_orbdet_v02_multiseed_queue_contract.py` | Static seed, environment, isolation, and fail-fast contracts |
| `configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_seed42.py` | Seed 42 override only |
| `configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_seed2026.py` | Seed 2026 override only |
| `scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed42.sh` | Guarded two-rank seed 42 launcher |
| `scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed2026.sh` | Guarded two-rank seed 2026 launcher |
| `scripts/formal/run_orbdet_v0_2_hrsc_multiseed_gpu89_20260815.sh` | Sequential fail-fast queue and markers |

### Task 1: Create the frozen execution branch

- [ ] Create `.worktrees/v02-stability` from the approved design commit on branch `exp/v02-hrsc-stability`.
- [ ] Verify `.worktrees/` is ignored and run the existing v0.2/GODC unit baseline in the worktree.

Run:

```bash
rtk git check-ignore -q .worktrees
rtk git worktree add .worktrees/v02-stability -b exp/v02-hrsc-stability
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_orbdet_v0_2.py tests/test_group_orbit_determinantal_cluster.py
```

Expected: worktree created and all selected baseline tests pass.

### Task 2: Add RED multi-seed contracts

- [ ] Add tests that load both configs and assert their normalized contents
  equal the seed-3407 config after replacing only `randomness.seed` and
  `work_dir`.
- [ ] Assert launchers contain the Orbdet interpreter, worktree `PYTHONPATH`,
  GPU 8/9 and NCCL variables, unique ports 29641/29642, no `--resume`, and
  refuse existing checkpoints.
- [ ] Assert the queue calls seed 42 before 2026, uses `set -euo pipefail`, and
  writes `COMPLETE` only after both launchers while trapping errors to `FAILED`.
- [ ] Run the test and observe missing-file failure.

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_orbdet_v02_multiseed_queue_contract.py
```

Expected: FAIL because seed configs/launchers do not yet exist.

### Task 3: Implement inheritance-only seed configs

- [ ] Create both configs with exactly this shape:

```python
_base_ = './orbdet_v0_2_r50_hrsc_clean_gpu89.py'
randomness = dict(seed=42, deterministic=False)  # 2026 in second file
work_dir = ('/data1/zcy/Orbdet/work_dirs/formal/'
            'orbdet_v0_2_hrsc_clean_gpu89_seed42_20260815')
```

- [ ] Run the config-normalization test and confirm the config portion passes.

### Task 4: Implement launchers and queue

- [ ] Resolve `repo_root` from each script location so code comes from the
  frozen worktree, while keeping artifact paths under the shared project
  `work_dirs/formal` tree.
- [ ] Before launch, reject any matching active Orbdet training process and
  any existing `*.pth` in the exact work directory.
- [ ] Launch two ranks with the specified interpreter, ports, GPU and NCCL
  variables.
- [ ] Implement the queue with durable `RUNNING`, `COMPLETE`, and `FAILED`
  markers and no destructive cleanup.
- [ ] Make scripts executable and run the contract test to GREEN.

Run:

```bash
rtk chmod +x scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed42.sh scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed2026.sh scripts/formal/run_orbdet_v0_2_hrsc_multiseed_gpu89_20260815.sh
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_orbdet_v02_multiseed_queue_contract.py tests/test_orbdet_v0_2.py
```

Expected: PASS.

### Task 5: Freeze and launch

- [ ] Commit all task-1 artifacts on `exp/v02-hrsc-stability`.
- [ ] Recheck GPU 8/9 availability and absence of Orbdet training processes.
- [ ] Start one detached tmux session named
  `orbdet_v02_multiseed_gpu89_20260815` running the queue.
- [ ] Within 60 seconds, verify two ranks, expected config/work dir, live log
  progress, and memory allocation on both GPUs.

Run:

```bash
rtk git add tests/test_orbdet_v02_multiseed_queue_contract.py configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_seed42.py configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_seed2026.py scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed42.sh scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed2026.sh scripts/formal/run_orbdet_v0_2_hrsc_multiseed_gpu89_20260815.sh
rtk git commit -m "exp: queue v0.2 HRSC stability seeds"
rtk tmux new-session -d -s orbdet_v02_multiseed_gpu89_20260815 "rtk bash /data1/zcy/Orbdet/.worktrees/v02-stability/scripts/formal/run_orbdet_v0_2_hrsc_multiseed_gpu89_20260815.sh"
```

Expected: seed 42 starts first and the queue remains live.

