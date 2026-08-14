# Orbdet-v0.2 Anchored-Symmetry GPU89 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the non-self-weighting Orbdet-v0.2 symmetry anchor and run its clean HRSC seed-3407 experiment on physical GPUs 8/9.

**Architecture:** Reuse the official H2RBox-v2 original/rotated/flipped view path, per-object fixed correspondence, PSC coder, and snap losses.  Orbdet adds detached calibration diagnostics but leaves the proven optimization objective and inference path unchanged.

**Tech Stack:** PyTorch 1.12.1, MMEngine, MMDetection/MMRotate 1.x, pytest, torchrun/NCCL, HRSC2016.

---

### Task 1: RED tests for the correctness anchor

**Files:**
- Create: `tests/test_orbdet_v0_2.py`

- [ ] Write tests requiring the official H2RBox-v2 components, the anchored
  Orbdet loss/detector, exact loss equivalence, a flip-wrong counterexample,
  detached diagnostics, and the clean GPU89 config/launcher contracts.
- [ ] Run
  `rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q tests/test_orbdet_v0_2.py`
  and confirm collection fails because the new modules do not exist.

### Task 2: Port the proven symmetry foundation

**Files:**
- Create: `mmrotate/models/detectors/h2rbox_v2.py`
- Create: `mmrotate/models/dense_heads/h2rbox_v2_head.py`
- Create: `mmrotate/models/losses/h2rbox_v2_consistency_loss.py`
- Modify: `mmrotate/models/detectors/__init__.py`
- Modify: `mmrotate/models/dense_heads/__init__.py`
- Modify: `mmrotate/models/losses/__init__.py`

- [ ] Apply the three files byte-for-byte from upstream dev-1.x commit
  `97b793577199e80f85f199b6bffb99b03017c4f4`.
- [ ] Export `H2RBoxV2Detector`, `H2RBoxV2Head`, and
  `H2RBoxV2ConsistencyLoss` through the local registries.
- [ ] Run a focused import/build check with the Orbdet interpreter.

### Task 3: GREEN anchored diagnostics without loss weighting

**Files:**
- Create: `mmrotate/models/losses/orbdet_anchored_symmetry_loss.py`
- Create: `mmrotate/models/detectors/orbdet_v0_2.py`
- Modify: `mmrotate/models/losses/__init__.py`
- Modify: `mmrotate/models/detectors/__init__.py`

- [ ] Implement `OrbdetAnchoredSymmetryLoss` as a subclass of the official
  loss.  Call `super().forward(...)` for the returned training objective, then
  record detached Gaussian qualities from snapped rotation and flip residuals.
- [ ] Implement `OrbdetV02Detector` as a prediction-preserving subclass that
  only exposes detached metrics.
- [ ] Run the focused test and confirm loss equivalence, counterexample
  separation, finite gradients, and registration all pass.

### Task 4: Clean HRSC config and launchers

**Files:**
- Create: `configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_smoke.py`
- Create: `scripts/smoke/run_orbdet_v0_2_hrsc_clean_gpu89.sh`
- Create: `scripts/formal/run_orbdet_v0_2_hrsc_clean_gpu89_seed3407.sh`

- [ ] Define the R50-FPN/PSC model, clean data split, HBox-only train
  pipeline, optimizer-step-aligned 103E schedule, seed 3407, and unique work
  directory.
- [ ] Define a two-iteration smoke override with validation disabled.
- [ ] Pin both launchers to the Orbdet interpreter, physical GPUs 8/9, two
  ranks, and disabled NCCL P2P/IB; reject duplicate matching formal jobs.
- [ ] Run the complete focused tests and existing Orbdet regression tests.

### Task 5: GPU89 smoke and formal seed 3407

**Files:**
- Output: `work_dirs/smoke/orbdet_v0_2_hrsc_clean_gpu89/`
- Output: `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814/`
- Create: `resultmd/exp_orbdet_v0_2_hrsc_clean/fres_orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814.md`

- [ ] Recheck physical GPU 8/9 ownership immediately before launch.
- [ ] Run the bounded DDP smoke and verify two ranks, finite losses, and no
  traceback/NCCL error.
- [ ] Start the formal seed-3407 job only after smoke passes.
- [ ] Monitor loss and validation checkpoints.  Stop this candidate if epoch
  24 AP50 is below 0.70; otherwise continue through 103 epochs and record the
  best clean validation AP.
- [ ] Do not evaluate `test.txt` in this task.

