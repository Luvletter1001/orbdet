# Orbdet Clean HRSC Comparison Until 08:30 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a protocol-matched H2RBox-versus-Orbdet HRSC comparison on GPUs 8/9 while enforcing an absolute 08:30 GPU-release deadline.

**Architecture:** Keep the running baseline untouched, add a minimal candidate config that inherits its entire experiment contract, and enforce equality with automated tests. A named deadline guard controls only experiment-owned sessions and exact config processes; separate launch/test logs preserve every stage.

**Tech Stack:** Python 3.10, PyTorch 1.12.1, MMEngine 0.10.3, MMDetection 3.3.0, MMCV 2.2.0, MMRotate, pytest, Bash, tmux, GNU timeout, NCCL/DDP.

---

The Orbdet directory is not a Git repository, so worktree and commit operations
are not applicable. Isolation is provided by unique configs, work directories,
logs, ports, and tmux names.

### Task 1: Define candidate and deadline contracts with failing tests

**Files:**
- Create: `tests/test_hrsc_clean_comparison_contract.py`

- [ ] **Step 1: Write tests for the missing candidate artifacts**

The test loads the clean H2RBox config and candidate config, asserts equality of
train/val/test dataloaders, train loop, optimizer, schedulers, hooks, seed, and
image pipeline, then asserts the candidate changes only detector type,
consistency loss type/parameters, and work directory. Launcher tests require
GPUs 8/9, both NCCL safety flags, two ranks, unique ports, no resume, and the
absolute deadline epoch `1786667400` in the guard.

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q /data1/zcy/Orbdet/tests/test_hrsc_clean_comparison_contract.py
```

Expected: failure because the candidate config and launch/guard scripts do not
yet exist.

### Task 2: Implement the minimal clean candidate configuration

**Files:**
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py`
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py`

- [ ] **Step 1: Add the formal candidate config**

Inherit `h2rbox_r50_hrsc_200e_2gpu_officiallike.py`. Override only the model
type to `OrbdetDetector`, the self-supervised loss to
`OrbdetHarmonicConsistencyLoss` with the validated v0.1 parameters, and the
candidate work directory.

- [ ] **Step 2: Add the bounded smoke config**

Inherit the formal candidate, select eight training images, set zero workers,
run one epoch/two optimizer steps without validation, log every step, and save
one checkpoint under a unique smoke work directory.

### Task 3: Implement fixed launchers and deadline guard

**Files:**
- Create: `scripts/smoke/run_orbdet_v0_1_hrsc_clean_gpu89_bs2_smoke.sh`
- Create: `scripts/formal/run_orbdet_v0_1_hrsc_clean_200e_gpu89_bs2.sh`
- Create: `scripts/formal/guard_orbdet_gpu89_until_0830_20260814.sh`

- [ ] **Step 1: Add the two-rank smoke launcher**

Use physical GPUs 8/9, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, the Orbdet
conda interpreter, two ranks, port 29617, and the unique smoke work directory.

- [ ] **Step 2: Add the formal launcher**

Use the same environment contract, two ranks, port 29618, duplicate-process
protection, no resume, and the unique formal candidate work directory.

- [ ] **Step 3: Add the deadline guard**

Poll time every 30 seconds until epoch `1786667400`. At the deadline, send
Ctrl-C to the exact named baseline, candidate, and queue tmux sessions if they
exist; after an orderly grace interval, signal only processes matching the exact
baseline or candidate config path. Never match a generic `tools/train.py` command.

- [ ] **Step 4: Verify GREEN and existing tests**

Run the new contract test and `tests/test_orbdet_v0_1.py`; both must pass.

### Task 4: Start deadline guard and close the baseline

**Files:**
- Runtime log: `work_dirs/orbdet_gpu89_deadline_guard_0830_20260814.log`
- Runtime test dir: `work_dirs/test/h2rbox_hrsc_clean_bestval_gpu89_20260814/`

- [ ] **Step 1: Start the guard in its own tmux session**

Use session `orbdet_gpu89_deadline_guard_0830_20260814` and verify it records
the deadline and remaining seconds.

- [ ] **Step 2: Wait for the existing H2RBox session to finish**

Poll at intervals no longer than 45 seconds. Require epoch 200 checkpoint,
final validation, no fatal error, and GPU release.

- [ ] **Step 3: Run one-shot H2RBox held-out test**

Select `best_dota_mAP_epoch_*.pth` only from the validation work directory.
Launch `tools/test.py` on GPUs 8/9 with port 29616 and save results/predictions
to the dedicated H2RBox test directory.

### Task 5: Smoke and launch the clean Orbdet candidate

**Files:**
- Runtime smoke dir: `work_dirs/smoke/orbdet_v0_1_hrsc_clean_gpu89_bs2_20260814/`
- Runtime formal log: `work_dirs/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_20260814.launch.log`
- Runtime formal dir: `work_dirs/formal/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814/`

- [ ] **Step 1: Run the two-step DDP smoke**

Require two finite train windows, saved checkpoint, both ranks, and no fatal
error. Do not tune parameters on smoke output.

- [ ] **Step 2: Launch formal training with the remaining-time timeout**

Compute `remaining = 1786667400 - now - 300`. Start only if positive. Wrap the
formal launcher in `timeout --signal=INT --kill-after=60`, redirect to the
dedicated log, and run in session
`orbdet_hrsc_clean200e_bs2_gpu89_20260814`.

- [ ] **Step 3: Verify stability**

Require the two main ranks on GPU 8/9, finite loss/quality diagnostics, sustained
utilization, and no OOM/NCCL/traceback. Record the first validation point.

### Task 6: Final test, deadline audit, and result record

**Files:**
- Create: `resultmd/exp_orbdet_hrsc_clean/fres_orbdet_hrsc_clean_gpu89_20260814.md`
- Runtime candidate test dir: `work_dirs/test/orbdet_v0_1_hrsc_clean_bestval_gpu89_20260814/`

- [ ] **Step 1: Test the candidate if training completes before cutoff**

Select the best-validation checkpoint and run held-out test once using the
remaining deadline budget. If training was time-capped, skip test and record
the last completed epoch/checkpoint.

- [ ] **Step 2: Audit GPU release at or before 08:30**

Verify no process matching either exact comparison config remains on GPUs 8/9.
Do not interpret unrelated GPU 8/9 processes as owned by this experiment.

- [ ] **Step 3: Write the final evidence record**

Record protocols, curves, best-validation checkpoints, one-shot test metrics,
errors, deadline status, actual wall time, and a fair comparison conclusion.
Never fill a missing result with an estimate.

