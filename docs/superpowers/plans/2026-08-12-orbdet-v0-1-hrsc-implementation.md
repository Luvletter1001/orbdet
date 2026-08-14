# Orbdet-v0.1 HRSC2016 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 HBox-supervised Orbdet-v0.1，在 HRSC2016 官方 trainval/test 划分上用 GPU 8/9 完成约 12 小时预算的训练、周期评测和稳定性值守。

**Architecture:** 复用 R50-FPN H2RBox 的成熟密集候选链路，以独立 `OrbdetHarmonicConsistencyLoss` 对双视图方向一致性计算二阶圆周谐波可靠度并加权自监督框损失；`OrbdetDetector` 只负责输出训练诊断量，推理分数链保持不变。HRSC 训练通过 `qbox -> hbox -> rbox` 丢弃真实角度，测试保留 OBB GT。

**Tech Stack:** Python 3.10, PyTorch 1.12.1, MMEngine 0.10.3, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest, NCCL DDP, HRSC2016.

---

## File map

| path | responsibility |
|---|---|
| `mmrotate/models/losses/orbdet_harmonic_consistency_loss.py` | 谐波可靠度与质量加权一致性损失 |
| `mmrotate/models/detectors/orbdet.py` | Orbdet 模型注册与诊断量透传 |
| `configs/orbdet/orbdet_v0_1_r50_hrsc.py` | 正式 HRSC 配置 |
| `configs/orbdet/orbdet_v0_1_r50_hrsc_calibration.py` | 一轮吞吐标定配置 |
| `scripts/formal/run_orbdet_v0_1_hrsc_gpu89.sh` | 双卡正式启动器 |
| `scripts/smoke/run_orbdet_v0_1_hrsc_gpu89.sh` | 双卡短冒泡启动器 |
| `tests/test_orbdet_v0_1.py` | 数学、注册、数据和启动合同 |
| `resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md` | 训练状态与最终结果回执 |

Orbdet 不是 Git 仓库；每个原计划 commit 点改为保存 fresh test/log evidence，不初始化仓库。

### Task 1: RED — harmonic reliability contract

**Files:**
- Create: `tests/test_orbdet_v0_1.py`

- [ ] **Step 1: Write the missing-module tests**

```python
import math
import torch

from mmrotate.models.losses.orbdet_harmonic_consistency_loss import (
    OrbdetHarmonicConsistencyLoss, axis_harmonic_reliability)


def test_axis_harmonic_reliability_respects_period_and_side_swap():
    delta = torch.tensor([0., math.pi / 2, math.pi, -math.pi / 2])
    quality = axis_harmonic_reliability(delta, min_quality=0.25, gamma=2.)
    assert torch.allclose(quality, torch.ones_like(quality), atol=1e-6)


def test_axis_harmonic_reliability_has_configured_floor_at_ambiguous_axis():
    delta = torch.tensor([math.pi / 4, -math.pi / 4])
    quality = axis_harmonic_reliability(delta, min_quality=0.25, gamma=2.)
    assert torch.allclose(quality, torch.full_like(quality, 0.25), atol=1e-6)


def test_orbdet_loss_has_finite_gradient_and_detached_quality():
    loss_fn = OrbdetHarmonicConsistencyLoss(loss_weight=0.4)
    pred = torch.tensor([[10., 10., 4., 2., 0.2]], requires_grad=True)
    target = torch.tensor([[10., 10., 4., 2., 0.0]])
    loss = loss_fn(pred, target, weight=torch.ones(1), avg_factor=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert 0.25 <= float(loss_fn.last_quality_mean) <= 1.0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_orbdet_v0_1.py
```

Expected: collection fails with `ModuleNotFoundError` for `orbdet_harmonic_consistency_loss`.

### Task 2: GREEN — harmonic reliability loss

**Files:**
- Create: `mmrotate/models/losses/orbdet_harmonic_consistency_loss.py`
- Modify: `mmrotate/models/losses/__init__.py`
- Test: `tests/test_orbdet_v0_1.py`

- [ ] **Step 1: Implement the minimal public function**

```python
def axis_harmonic_reliability(angle_delta, min_quality=0.25, gamma=2.0):
    raw = torch.maximum(angle_delta.cos().abs(), angle_delta.sin().abs())
    floor = math.sqrt(0.5)
    normalized = ((raw - floor) / (1.0 - floor)).clamp_(0.0, 1.0)
    return min_quality + (1.0 - min_quality) * normalized.pow(gamma)
```

- [ ] **Step 2: Implement `OrbdetHarmonicConsistencyLoss`**

Build the same center/shape/periodic-angle components as
`H2RBoxConsistencyLoss`, but replace the incoming sample weight with:

```python
quality = axis_harmonic_reliability(
    pred[..., 4] - target[..., 4], self.min_quality, self.gamma).detach()
quality_weight = weight * quality
```

Store detached scalar diagnostics `last_quality_mean`, `last_quality_min`, and
`last_quality_high_frac`; empty inputs must return differentiable zero and finite diagnostics.

- [ ] **Step 3: Export the class**

Add `from .orbdet_harmonic_consistency_loss import OrbdetHarmonicConsistencyLoss`
to `mmrotate/models/losses/__init__.py`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 pytest command. Expected: `3 passed`.

### Task 3: RED/GREEN — detector registration and diagnostics

**Files:**
- Create: `mmrotate/models/detectors/orbdet.py`
- Modify: `mmrotate/models/detectors/__init__.py`
- Modify: `tests/test_orbdet_v0_1.py`

- [ ] **Step 1: Add a failing registry test**

```python
from mmrotate.registry import MODELS


def test_orbdet_detector_is_registered():
    assert MODELS.get('OrbdetDetector').__name__ == 'OrbdetDetector'
```

Run the focused test and confirm failure because `OrbdetDetector` is absent.

- [ ] **Step 2: Implement the minimal detector**

Subclass `H2RBoxDetector`. In `loss`, call `super().loss`, read diagnostics from
`self.bbox_head.loss_bbox_ss`, and add non-loss tensors named `orbdet_q_mean`,
`orbdet_q_min`, `orbdet_q_high_frac`. Do not override `predict`.

- [ ] **Step 3: Export and verify**

Export `OrbdetDetector` in the detector package and run the complete test file.
Expected: `4 passed`.

### Task 4: RED/GREEN — HRSC formal configuration contract

**Files:**
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc.py`
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc_calibration.py`
- Modify: `tests/test_orbdet_v0_1.py`

- [ ] **Step 1: Add failing configuration assertions**

Load the formal config with `Config.fromfile` and assert:

```python
assert cfg.model.type == 'OrbdetDetector'
assert cfg.model.bbox_head.num_classes == 1
assert cfg.model.bbox_head.loss_bbox_ss.type == 'OrbdetHarmonicConsistencyLoss'
assert cfg.train_dataloader.batch_size == 1
assert cfg.train_dataloader.batch_sampler is None
assert cfg.train_dataloader.dataset.ann_file == 'ImageSets/trainval.txt'
assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/test.txt'
train_conversions = [x.box_type_mapping.gt_bboxes for x in cfg.train_pipeline
                     if x.type == 'ConvertBoxType']
assert train_conversions == ['hbox', 'rbox']
assert cfg.train_pipeline[-2].type == 'mmdet.Pad'
```

Run focused test and confirm failure because the config is absent.

- [ ] **Step 2: Write the formal config**

Inherit the H2RBox DOTA model/runtime but replace all dataset fields with native
`HRSCDataset`, `data_root='/data1/zcy/Orbdet/data/hrsc/'`, official files, one class,
`800 x 800` resize/pad, AdamW, batch 1/GPU, workers 2, deterministic seed 3407.
Set an initial safe `max_epochs=216`, `val_interval=12`, milestone epochs 144 and 198;
these three values are replaced after calibration using the design formula.

- [ ] **Step 3: Write calibration config**

Inherit the formal config, disable val, set one epoch and a separate
`work_dir=work_dirs/calibration/orbdet_v0_1_hrsc_gpu89`.

- [ ] **Step 4: Parse and verify**

Run the whole test file and a direct `Config.fromfile` print. Expected: all tests pass,
and print shows `batch_size=1`, `batch_sampler=None`, `DefaultSampler`.

### Task 5: RED/GREEN — GPU 8/9 launch contracts

**Files:**
- Create: `scripts/smoke/run_orbdet_v0_1_hrsc_gpu89.sh`
- Create: `scripts/formal/run_orbdet_v0_1_hrsc_gpu89.sh`
- Modify: `tests/test_orbdet_v0_1.py`

- [ ] **Step 1: Add failing launcher text tests**

Assert both scripts contain exact assignments for `CUDA_VISIBLE_DEVICES=8,9`,
`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, the Orbdet interpreter, `--nproc_per_node=2`,
and distinct calibration/formal configs.

- [ ] **Step 2: Write launchers**

Use `set -euo pipefail`, fixed project root, fixed interpreter, explicit master ports,
and `tools/train.py ... --launcher pytorch`. The formal launcher must reject an existing
live Orbdet formal process instead of starting a duplicate.

- [ ] **Step 3: Verify contracts**

Run the whole test file. Expected: all tests pass.

### Task 6: Data and pretrained initialization

**Files:**
- Create symlink: `data/hrsc -> /data/zcy/dataset/HRSC_unzip`
- Create: `weights/resnet50-0676ba61.pth`

- [ ] **Step 1: Create and verify the read-only data link**

Create only the Orbdet-side symlink. Verify `trainval=617`, `test=453`, and disjoint IDs
with the Orbdet interpreter.

- [ ] **Step 2: Obtain official ImageNet ResNet-50 weights**

Download `https://download.pytorch.org/models/resnet50-0676ba61.pth` into Orbdet weights,
verify SHA256, and point the formal config to the absolute local path. Do not initialize
from a DOTA/HRSC detection checkpoint.

- [ ] **Step 3: Build dataset and model on CPU**

Use MMEngine Runner construction without training. Expected: 617 train samples,
453 validation samples, model type `OrbdetDetector`.

### Task 7: Full static and unit verification

**Files:** all changed Python/config/shell files.

- [ ] **Step 1: Compile**

Run `python -m py_compile` for all new Python/config files. Expected exit 0.

- [ ] **Step 2: Shell syntax**

Run `bash -n` on both launchers. Expected exit 0.

- [ ] **Step 3: Full relevant pytest**

Run `tests/test_orbdet_v0_1.py` plus `tests/test_dota1_smoke_contract.py`.
Expected: zero failures, proving no regression to the existing smoke foundation.

### Task 8: Two-GPU short smoke and calibration

**Files:**
- Output: `work_dirs/calibration/orbdet_v0_1_hrsc_gpu89/`
- Update: `resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md`

- [ ] **Step 1: Preflight resources**

Record GPU 8/9 memory/utilization and all owners. Do not stop `lzy` processes.

- [ ] **Step 2: Run bounded DDP calibration**

Launch one full epoch on GPU 8/9 with mandatory NCCL flags. Require finite losses,
presence of all three `orbdet_q_*` diagnostics, and `epoch_1.pth`.

- [ ] **Step 3: Measure throughput**

Parse steady-state iteration time after warmup. Run one full validation of the calibration
checkpoint to measure wall-clock evaluation cost; its AP is diagnostic only.

- [ ] **Step 4: Finalize schedule**

Compute total epochs so estimated train + periodic val + final eval is 11–12 hours,
with hard maximum 12.5 hours. Set validation interval to produce 12–20 AP points and set
milestones to approximately `2/3` and `11/12` of total epochs. Re-run config tests.

### Task 9: Formal training and live guard

**Files:**
- Output: `work_dirs/formal/orbdet_v0_1_hrsc_gpu89/`
- Update: `FORMAL_TRAINING_NOT_STARTED.md`
- Update: `resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md`

- [ ] **Step 1: Open the formal gate accurately**

Change status to `running`, record user authorization, config path, GPUs, start time and PID.

- [ ] **Step 2: Start exactly one formal DDP run**

Use the formal launcher in a persistent foreground execution session. Confirm both ranks enter
the training loop and GPU 8/9 memory remains safe.

- [ ] **Step 3: Monitor at least each validation boundary**

Record epoch, `loss_cls`, `loss_bbox`, `loss_centerness`, `loss_bbox_ss`,
`orbdet_q_mean`, AP50, GPU memory/utilization, checkpoint and process health.

- [ ] **Step 4: Apply stop rules**

Stop only at a completed checkpoint/validation boundary when either the three-point stability
rule in the spec passes or the 12.5-hour hard cap is reached. On NaN/crash, preserve evidence,
diagnose, test the fix, and resume only within remaining budget.

### Task 10: Final evaluation and receipt

**Files:**
- Update: `resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md`
- Update: `FORMAL_TRAINING_NOT_STARTED.md`

- [ ] **Step 1: Evaluate the best checkpoint**

Run two-GPU test on official HRSC test split and capture `dota/mAP`.

- [ ] **Step 2: Verify artifacts**

Record best checkpoint absolute path and SHA256, full log path, final AP sequence, elapsed time,
config checksum and remaining Orbdet processes.

- [ ] **Step 3: Close status truthfully**

Set formal status to `completed_stable`, `completed_budget_exhausted`, or `failed_diagnosed` based
only on evidence. The Chinese receipt must distinguish achieved results from target ranges.

- [ ] **Step 4: Fresh completion verification**

Re-run relevant tests, config parsing, artifact existence checks and process inspection before
claiming completion.
