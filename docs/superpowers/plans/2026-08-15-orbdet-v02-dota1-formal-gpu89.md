# Orbdet-v0.2 DOTA-v1 Formal GPU 8/9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, fail-fast DOTA-v1 1× formal contract for Orbdet-v0.2 and queue it after the HRSC GODC run.

**Architecture:** A dedicated worktree contains one full-trainval configuration, a two-step smoke override, guarded two-rank launchers, and a controller that waits for the prior task's durable success marker. No validation or hidden-test loop runs during formal training.

**Tech Stack:** MMEngine Config, MMRotate, PyTorch distributed, DOTA-v1 prepared patches, Bash, tmux, pytest.

---

### Task 1: Create isolation and RED contracts

- [ ] Create `.worktrees/dota-v1-formal` on branch
  `exp/orbdet-v02-dota1-formal` from local `main`.
- [ ] Add `tests/test_orbdet_v02_dota1_formal_contract.py` asserting model,
  HBox pipeline, raw/effective data counts, 12E optimizer/scheduler, no validation,
  bounded smoke, required GPU/NCCL/env/ports, and controller order.
- [ ] Run the test and observe missing artifact failures.

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_orbdet_v02_dota1_formal_contract.py
```

Expected: FAIL because formal config and launch scripts do not exist.

### Task 2: Implement formal and smoke configs

- [ ] Create `configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89.py` with
  `OrbdetV02Detector`, 15 classes, rotation-agnostic IDs `[9, 11]`, local
  R50 weights, 1024 crop, global batch 4, full trainval, no val/test loop,
  AdamW `1e-4`, milestones 8/11, 12 epochs, and seed 3407.
- [ ] Create `configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89_smoke.py`
  inheriting formal but limiting the dataset to eight images, one epoch/two
  steps, zero workers, and a unique smoke directory.
- [ ] Build the model and initialize the formal train dataset; assert 20,995
  raw pairs and 12,757 effective nonempty samples.

### Task 3: Implement guarded launchers and controller

- [ ] Create smoke/formal launchers using ports 29645/29646, the worktree
  `PYTHONPATH`, physical GPUs 8/9, both NCCL disable variables, exact process
  guards, GPU-idle guards, and checkpoint overwrite refusal.
- [ ] Create `run_orbdet_v02_dota1_after_godc_20260815.sh` that waits for the
  GODC controller `COMPLETE`, aborts on `FAILED`, runs smoke, checks
  `epoch_1.pth`, then runs formal and writes durable markers.
- [ ] Make scripts executable; run Bash syntax checks and contract tests.

### Task 4: Verify, freeze, and arm

- [ ] Run focused tests, the complete project `tests/` suite, config/model/data
  construction, and `git diff --check`.
- [ ] Commit the DOTA worktree locally.
- [ ] Start tmux session `orbdet_v02_dota1_after_godc_gpu89_20260815`; verify
  it waits without a DOTA process or GPU allocation while task 2 is pending.
- [ ] Update the formal status file on local `main`; do not push.
