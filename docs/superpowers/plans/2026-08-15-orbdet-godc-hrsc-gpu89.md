# Orbdet GODC HBox/FPN HRSC GPU 8/9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable C2 group-orbit auxiliary objective on GT-HBox FPN ROIs, prove it with tests and a two-rank smoke run, then launch a formal clean-HRSC comparison after the v0.2 stability queue succeeds.

**Architecture:** H2RBox-v2 exposes one protected auxiliary-loss hook at the point where training FPN features and original-view GT instances already coexist. A new registered adapter extracts square multi-level HBox ROIs and invokes the existing GODC core; a dedicated detector subclass enables it without changing prediction or baseline configs.

**Tech Stack:** PyTorch, MMEngine/MMDetection registries, MMRotate, MMCV RoIAlign, pytest, Bash, tmux.

---

## File map

| File | Responsibility |
|---|---|
| `mmrotate/models/detectors/h2rbox_v2.py` | Default no-op auxiliary hook |
| `mmrotate/models/losses/hbox_fpn_group_orbit_loss.py` | HBox validation, FPN RoI extraction, C2 orbit, GODC diagnostics |
| `mmrotate/models/detectors/orbdet_godc.py` | GODC-enabled v0.2 subclass |
| `mmrotate/models/losses/__init__.py` | Public adapter export |
| `mmrotate/models/detectors/__init__.py` | Public detector export |
| `tests/test_hbox_fpn_group_orbit_loss.py` | Adapter unit/gradient/empty tests |
| `tests/test_orbdet_godc.py` | Hook, config, prediction, and launcher contracts |
| `configs/orbdet/orbdet_godc_c2_r50_hrsc_clean_gpu89.py` | Formal seed-3407 comparison |
| `configs/orbdet/orbdet_godc_c2_r50_hrsc_clean_gpu89_smoke.py` | Eight-image two-step smoke |
| `scripts/smoke/run_orbdet_godc_c2_hrsc_gpu89.sh` | Guarded smoke launcher |
| `scripts/formal/run_orbdet_godc_c2_hrsc_gpu89_seed3407.sh` | Guarded formal launcher |
| `scripts/formal/run_orbdet_godc_after_v02_multiseed_20260815.sh` | Wait-success, smoke, formal controller |

### Task 1: Add RED adapter tests

- [ ] Test registry construction using a `SingleRoIExtractor` with output size
  7, channels 4, strides `[1]`, and a guard-disabled GODC core.
- [ ] With one valid HBox and a learnable feature map, assert a finite scalar,
  nonzero gradient, ROI count one, and detached finite diagnostics.
- [ ] With empty/NaN/zero-area boxes, assert a graph-connected zero, ROI count
  zero, and no exception.
- [ ] Assert invalid group `c1`, nonpositive loss weight, and nonpositive
  minimum box size are rejected.
- [ ] Run and observe import failure for the missing adapter.

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_hbox_fpn_group_orbit_loss.py
```

Expected: FAIL with missing module/class.

### Task 2: Implement the ROI adapter

- [ ] Register `HBoxFPNGroupOrbitLoss(torch.nn.Module)` with constructor:

```python
def __init__(self, roi_extractor: dict, orbit_loss: dict,
             group: str = 'c2', loss_weight: float = 0.02,
             min_box_size: float = 2.0) -> None:
```

- [ ] Convert every `gt_instances.bboxes` through `convert_to('hbox').tensor`,
  filter finite boxes with width/height at least `min_box_size`, combine via
  `bbox2roi`, and feed only `roi_extractor.num_inputs` FPN levels.
- [ ] Return `features[0].sum() * 0.0` when no valid ROI, otherwise build the
  exact configured orbit and return `loss_weight * orbit_loss(orbit)`.
- [ ] Mirror detached GODC diagnostics and a tensor `last_roi_count`.
- [ ] Export the class and run adapter tests to GREEN.

### Task 3: Add the detector hook test-first

- [ ] Test that `H2RBoxV2Detector._add_auxiliary_losses` returns the original
  loss dictionary unchanged.
- [ ] Test `OrbdetGODCDetector` is an `OrbdetV02Detector`, does not override
  `predict`, builds its adapter from the registry, emits `loss_godc`, and adds
  diagnostic keys without the substring `loss`.
- [ ] Observe RED before production changes.
- [ ] Add the no-op hook and call it once after `bbox_head.loss` in
  `H2RBoxV2Detector.loss`.
- [ ] Implement/export `OrbdetGODCDetector`; its override calls the adapter on
  original-view features and appends detached diagnostics.
- [ ] Run detector, v0.2 regression, and GODC-core tests to GREEN.

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_hbox_fpn_group_orbit_loss.py tests/test_orbdet_godc.py tests/test_orbdet_v0_2.py tests/test_group_orbit_determinantal_cluster.py
```

Expected: PASS.

### Task 4: Add formal/smoke configs and contracts

- [ ] Make the formal config inherit the seed-3407 v0.2 config and override
  only detector type, the adapter config below, and unique work dir:

```python
model = dict(
    type='OrbdetGODCDetector',
    godc_auxiliary=dict(
        type='HBoxFPNGroupOrbitLoss',
        group='c2', loss_weight=0.02, min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        orbit_loss=dict(
            type='GroupOrbitDeterminantalClusterLoss',
            determinantal_weight=1.0,
            spectral_tail_weight=0.0,
            fixed_space_weight=1.0,
            energy_guard_weight=1.0,
            variance_guard_weight=1.0)))
```

- [ ] Make smoke inherit formal and bound it to eight images, two optimizer
  steps, no validation, and its own smoke work dir.
- [ ] Assert formal equivalence outside `model.type`, adapter, and work dir;
  HBox-only training and held-out test policy; nonzero fixed weight; unique
  work dirs; required GPU/NCCL/env/ports; fail-fast controller order.
- [ ] Run config/contracts and full CPU suite to GREEN.

### Task 5: Implement launch/controller scripts

- [ ] Smoke and formal launchers refuse matching live processes and existing
  checkpoints, then use ports 29643 and 29644 on GPUs 8/9.
- [ ] Controller polls the task-1 `COMPLETE`/`FAILED` markers every 30 seconds,
  stops on failure, runs smoke, verifies `epoch_1.pth`, then runs formal.
- [ ] Controller writes its own `RUNNING`, `SMOKE_COMPLETE`, `COMPLETE`, or
  `FAILED` marker without deleting prior artifacts.
- [ ] Make scripts executable and run shell syntax checks plus contract tests.

### Task 6: Verify, commit, and arm the unattended chain

- [ ] Run focused tests, full pytest, config builds, and `git diff --check`.
- [ ] Commit GODC integration locally.
- [ ] Start the detached controller only after the code and contracts pass.
- [ ] Update `FORMAL_TRAINING_NOT_STARTED.md` to factual seed-3407-complete,
  task-1-running, task-2-armed state.
- [ ] Monitor controller/queue/process/log/GPU state within 60 seconds.

Run:

```bash
rtk tmux new-session -d -s orbdet_godc_after_v02_gpu89_20260815 "rtk bash /data1/zcy/Orbdet/scripts/formal/run_orbdet_godc_after_v02_multiseed_20260815.sh"
```

Expected: controller waits without allocating GPU while task 1 runs, then
automatically executes smoke and formal training only after task-1 success.

