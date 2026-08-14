# Orbdet-v0.1 HRSC Recovery-Contract 200E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在恢复有效的 H2RBox HRSC 合同下，仅替换 Orbdet detector/loss，按相同 seed 和相同 30→60→200E 恢复边界完成严格对照。

**Architecture:** Candidate 配置直接继承恢复版 H2RBox 配置，只覆盖 model 与独立 work_dir。合同测试把 candidate 归一化回 H2RBox 后做完整配置比较；启动脚本依次完成 30E、resume 60E、resume 200E，并在最后只评 val/train、不触碰 held-out test。

**Tech Stack:** MMRotate/MMDetection、MMEngine Config、PyTorch DDP、pytest、tmux、NVIDIA A40。

---

### Task 1: 建立严格合同测试

**Files:**
- Create: `tests/test_orbdet_hrsc_recovery_contract.py`

- [ ] **Step 1: 写缺失产物测试**

测试加载恢复版 H2RBox reference，并要求 candidate、smoke、两个 launcher 存在。
它必须断言 batch、optimizer、scheduler、seed、数据 split、validation interval 和
hooks 均相等，并在归一化 detector/loss/work_dir 后要求完整配置相等。

- [ ] **Step 2: 运行 RED**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  tests/test_orbdet_hrsc_recovery_contract.py
```

Expected: FAIL，且失败原因仅为新 candidate 产物缺失。

### Task 2: 创建最小 candidate 与 smoke 配置

**Files:**
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py`
- Create: `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_smoke_gpu89.py`

- [ ] **Step 1: candidate 继承恢复母版**

```python
_base_ = './h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py'
model = dict(
    type='OrbdetDetector',
    bbox_head=dict(loss_bbox_ss=dict(
        _delete_=True,
        type='OrbdetHarmonicConsistencyLoss',
        loss_weight=0.4,
        min_quality=0.25,
        gamma=2.0,
        high_quality_thr=0.75,
        center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
        shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
        angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0)))
```

- [ ] **Step 2: smoke 限制为四图两步**

Smoke 使用 `indices=list(range(4))`、batch 1/GPU、双 rank、1E、无 validation，
checkpoint interval 和 logger interval 均为 1。

### Task 3: 创建安全 launcher

**Files:**
- Create: `scripts/smoke/run_orbdet_v0_1_hrsc_recovery_gpu89_smoke.sh`
- Create: `scripts/formal/run_orbdet_v0_1_hrsc_recovery_staged_200e_gpu89_20260814.sh`

- [ ] **Step 1: smoke launcher**

固定 GPU 8/9、双 rank、NCCL P2P/IB disable、Orbdet Python 与唯一 master port。

- [ ] **Step 2: staged formal launcher**

正式脚本依次执行：

```text
fresh -> train_cfg.max_epochs=30
resume epoch_30.pth -> train_cfg.max_epochs=60
resume epoch_60.pth -> train_cfg.max_epochs=200
epoch_200.pth -> explicit val evaluation
epoch_200.pth -> explicit train evaluation
```

每阶段检查目标 checkpoint；新 work_dir 中已有 checkpoint 时拒绝启动；禁止调用
held-out test。

- [ ] **Step 3: 运行 GREEN 与回归测试**

```bash
rtk env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/Orbdet \
  /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest -q \
  tests/test_orbdet_hrsc_recovery_contract.py \
  tests/test_h2rbox_hrsc_recovery_contract.py \
  tests/test_orbdet_v0_1.py
```

Expected: all pass。

### Task 4: 冒泡验证

**Files:**
- Update: `resultmd/exp_orbdet_hrsc_recovery/fres_orbdet_v0_1_hrsc_recovery_gpu89_20260814.md`

- [ ] **Step 1: 再查 GPU 8/9 与精确进程**

若发现其他用户占用 GPU 8/9，不启动并报告；不得终止该进程。

- [ ] **Step 2: 启动 smoke**

```bash
rtk bash scripts/smoke/run_orbdet_v0_1_hrsc_recovery_gpu89_smoke.sh
```

- [ ] **Step 3: 验证 smoke**

要求 2/2 optimizer steps、finite loss/grad/quality、存在 `epoch_1.pth`，且无
OOM、NCCL、traceback、NaN/Inf。

### Task 5: 启动并监控正式 200E

**Files:**
- Modify: `FORMAL_TRAINING_NOT_STARTED.md`
- Update: `resultmd/exp_orbdet_hrsc_recovery/fres_orbdet_v0_1_hrsc_recovery_gpu89_20260814.md`

- [ ] **Step 1: 在独立 tmux 中启动**

```bash
rtk tmux new-session -d -s orbdet_v01_recovery_staged200e_gpu89_20260814 \
  "cd /data1/zcy/Orbdet && rtk bash scripts/formal/run_orbdet_v0_1_hrsc_recovery_staged_200e_gpu89_20260814.sh 2>&1 | rtk tee work_dirs/orbdet_v0_1_hrsc_recovery_staged200e_gpu89_20260814.launch.log"
```

- [ ] **Step 2: 稳定窗口检查**

验证两个主 rank、GPU 8/9、finite loss/grad、Orbdet quality 统计、无 fatal error。

- [ ] **Step 3: 阶段与 validation 记录**

记录 30/60/90/120/150/180E 自动 val 和 200E 显式 val。正式 launcher 自行在
200E 停止，不自动延长，不运行 held-out test。

### Task 6: 最终判定

**Files:**
- Update: `resultmd/exp_orbdet_hrsc_recovery/fres_orbdet_v0_1_hrsc_recovery_gpu89_20260814.md`

- [ ] **Step 1: 完整性审计**

核对 `epoch_200.pth`、validation ledger、显式 train/val、wall time、显存和 fatal
scan。

- [ ] **Step 2: 应用预注册门槛**

```text
best candidate val mAP < 0.8710 -> Orbdet-v0.1 failed
best candidate val mAP >= 0.8710 -> single-seed gate passed
```

不使用 held-out test，也不根据结果临时改阈值或调参。

## 自审

- 规格中的数据、优化器、scheduler、seed、stage、validation 和安全约束均有对应任务。
- 没有待填参数或未定义路径。
- Orbdet 不是 Git 仓库，因此计划中的检查点由测试、日志、配置快照和独立目录承担，
  不执行 Git commit。
