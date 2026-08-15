# H2RBox-v2 Official Audit and R50 MS+RR 12E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先用 OpenMMLab 官方 H2RBox-v2 R50 权重审计 DOTA-v1 SS/MS 提交链路，再按官方 checkpoint 内嵌协议完成 Orbdet-v0.2 R50 MS+RR 12E；R50 在线结果确认后，80+ 目标直接转入 Swin-B，不运行 R50 36E。

**Architecture:** 官方 SS checkpoint 负责验证模型加载、15 类顺序、SS trainval 自评和 SS test merge；官方 MS+RR checkpoint 负责验证 71,888 个 MS test patches 的原图合并与 ZIP。正式训练配置继承现有 Orbdet-v0.2 推理/优化路径，但对齐官方 checkpoint 的 global batch、类别处理、RR、AdamW 和 12E scheduler；审计、smoke、formal 使用互不覆盖的状态目录。

**Tech Stack:** MMRotate 1.0.0rc1, MMEngine 0.10.3, PyTorch 1.12.1+cu113, DOTA-v1.0, Bash, pytest, two NVIDIA A40 GPUs.

---

### Task 1: Freeze official artifacts and protocol evidence

**Files:**
- Runtime: `work_dirs/audit/h2rbox_v2_dota1_official_20260815/downloads/`
- Create: `resultmd/exp_h2rbox_v2_dota1_official_audit_20260815/faudit_h2rbox_v2_dota1_official_chain.md`

- [ ] **Step 1: Verify official artifact digests**

Run:

```bash
rtk sha256sum work_dirs/audit/h2rbox_v2_dota1_official_20260815/downloads/*.pth
```

Expected:

```text
fa5ad1d2d6d030a477fe6f9a405863a76f55d463cff31b7e01c8366527888fde  ...1x_dota-fa5ad1d2.pth
5e0e53e12e0d8b07f79922b6cef9b56c142458483d2c1e2f27348f7e72e9d677  ...ms_rr-1x_dota-5e0e53e1.pth
```

- [ ] **Step 2: Read checkpoint metadata as the reproduction authority**

Assert both checkpoints contain 371 state tensors, epoch 12, and respectively `iter=76800` and `iter=409956`. Adapt old names in metadata to the current port as follows:

```text
H2RBox2Detector -> H2RBoxV2Detector / OrbdetV02Detector
H2RBox2Head -> H2RBoxV2Head
rotation_agnostic_resize_classes -> agnostic_resize_classes
use_nested_projection -> use_circumiou_loss
H2RBoxSymmetryLoss.use_snapping_loss -> H2RBoxV2ConsistencyLoss.use_snap_loss
```

The formal contract is global batch 2, `lr=5e-5`, `weight_decay=0.005`, `rotation_agnostic_classes=[1,9,11]`, resize class `[1]`, reweighted bbox loss enabled, `RandomRotate(prob=1, angle_range=180)`, and 12 epochs with milestones 8/11.

### Task 2: Add RED contract tests

**Files:**
- Create: `tests/test_h2rbox_v2_dota1_official_audit_msrr_contract.py`

- [ ] **Step 1: Add artifact, config, dataset, launcher, and safety assertions**

The test must require three audit configs, one formal config, one two-step smoke config, audit/smoke/formal launchers, exact SS/MS patch counts, exact official hashes, global batch 2, no `batch_sampler` override, and the GPU/NCCL variables.

- [ ] **Step 2: Run RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q -p no:cacheprovider tests/test_h2rbox_v2_dota1_official_audit_msrr_contract.py
```

Expected: FAIL because the new configs and launchers do not exist.

### Task 3: Implement official checkpoint audit chain

**Files:**
- Create: `configs/orbdet/h2rbox_v2_r50_dota1_official_ss_trainval_audit_gpu89.py`
- Create: `configs/orbdet/h2rbox_v2_r50_dota1_official_ss_test_submission_gpu89.py`
- Create: `configs/orbdet/h2rbox_v2_r50_dota1_official_ms_test_submission_gpu89.py`
- Create: `scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu89.sh`

- [ ] **Step 1: Implement current-code equivalent of checkpoint model metadata**

Use `H2RBoxV2Detector` and `H2RBoxV2Head` for the official audit, including:

```python
bbox_head=dict(
    rotation_agnostic_classes=[1, 9, 11],
    agnostic_resize_classes=[1],
    use_circumiou_loss=True,
    use_standalone_angle=True,
    use_reweighted_loss_bbox=True)
```

- [ ] **Step 2: Build the three inference datasets**

Expected dataset lengths are SS trainval `20,995`, SS test `10,833`, and MS test `71,888`. SS trainval uses oriented annotations for self-evaluation; both test configs use `DOTAMetric(format_only=True, merge_patches=True, iou_thr=0.1)`.

- [ ] **Step 3: Implement fail-fast sequential audit launcher**

The launcher checks the exact SHA256 values, refuses occupied GPU 8/9, runs SS trainval, SS test, then MS test on two ranks, verifies every ZIP contains exactly 15 root-level `Task1_<class>.txt` files, and writes `COMPLETE` only after all stages pass.

- [ ] **Step 4: Run GREEN and a strict checkpoint load audit**

Run the focused test and build each model/dataset. Loading each official checkpoint must report no missing and no unexpected model keys.

### Task 4: Execute and interpret official inference audit

**Files:**
- Runtime: `work_dirs/audit/h2rbox_v2_dota1_official_20260815/`
- Update: `resultmd/exp_h2rbox_v2_dota1_official_audit_20260815/faudit_h2rbox_v2_dota1_official_chain.md`

- [ ] **Step 1: Launch only when physical GPUs 8/9 have no foreign compute processes**

Run:

```bash
rtk bash scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu89.sh
```

- [ ] **Step 2: Apply local pass criteria**

Require finite inference, strict state loading, expected dataset counts, SS trainval AP near the official checkpoint's recorded trainval value, successful 937-original-image merging, two valid 15-file ZIPs, and no traceback/NCCL/NaN. Online AP remains a separate confirmation that requires uploading the ZIP.

### Task 5: Implement and smoke R50 MS+RR 12E

**Files:**
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py`
- Create: `configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_smoke.py`
- Create: `scripts/smoke/run_orbdet_v0_2_r50_dota1_ms_rr_gpu89_smoke.sh`
- Create: `scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407.sh`

- [ ] **Step 1: Implement exact training contract**

Use `/data/zcy/dataset/trainval_ms_full/` read-only, 68,325 nonempty samples, batch 1/rank on GPUs 8/9, global batch 2, AdamW `lr=5e-5`, `weight_decay=0.005`, official category flags, RR probability 1, 12E, seed 3407, and no validation/test loop on the full trainval run.

- [ ] **Step 2: Verify sampler merge and config construction**

Require `train_dataloader.batch_size == 1`, `train_dataloader.sampler.type == 'DefaultSampler'`, and `train_dataloader.batch_sampler is None`. Build the model and full filtered dataset.

- [ ] **Step 3: Run bounded two-rank smoke**

The smoke uses 4 samples and exactly 2 global optimizer steps, writes `epoch_1.pth`, and must have finite losses and gradients.

### Task 6: Start formal R50 MS+RR 12E after audit passes

**Files:**
- Update on successful launch: `FORMAL_TRAINING_NOT_STARTED.md`
- Runtime: `work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407_20260815/`
- Create: `resultmd/exp_orbdet_v02_dota1_msrr12e_20260815/flog_orbdet_v02_dota1_msrr12e_gpu89.md`

- [ ] **Step 1: Recheck GPU ownership and audit marker**

The formal launcher refuses foreign GPU processes, missing audit `COMPLETE`, duplicate jobs, resume, or existing checkpoints.

- [ ] **Step 2: Start one named local tmux session**

Use physical GPUs 8/9 with `CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, and `NCCL_IB_DISABLE=1`. Do not start a queue, automatic resume, or R50 36E continuation.

- [ ] **Step 3: Verify live evidence**

Within the first minute, require two ranks, nonzero GPU utilization, the expected `68,325` dataset length, finite first losses, and no fast config/NCCL failure.

### Task 7: Gate the 80+ Swin-B phase

**Files:**
- Update after online result: `resultmd/exp_orbdet_v02_dota1_msrr12e_20260815/fres_orbdet_v02_dota1_msrr12e_gpu89.md`

- [ ] Record R50 MS+RR online AP50 and compare with official `78.25`.
- [ ] If online 80+ remains the target, write a separate Swin-B design/plan and request explicit formal-training authorization for that run.
- [ ] Do not create or launch an R50 36E schedule.
