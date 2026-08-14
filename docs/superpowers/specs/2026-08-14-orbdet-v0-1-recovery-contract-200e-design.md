# Orbdet-v0.1 HRSC 恢复合同 200E 严格对照设计

## 审批状态

用户已于 2026-08-14 明确批准：完全冻结恢复后的 H2RBox 合同，只替换
`OrbdetDetector + OrbdetHarmonicConsistencyLoss`，使用相同 seed 训练 200E；若
Orbdet 的最佳 validation mAP 仍低于 H2RBox 的 `0.8710`，则判定 v0.1 失败并
进入模型修改阶段。

## 科学问题

在一个已经证明不会塌缩的 HRSC 水平框监督训练合同中，Orbdet-v0.1 的 harmonic
reliability consistency loss 能否超过纯 H2RBox？

上一轮 global batch 4 / lr 5e-5 对照的 H2RBox 与 Orbdet 都塌缩到约 8% test
AP，因此只能证明 v0.1 不能挽救该坏合同，不能用于判断 v0.1 在有效合同中的价值。

## 方案选择

### 采用：镜像恢复实验的分段 200E

训练严格镜像 H2RBox 的实际执行边界：从头训练到 30E，恢复到 60E，再从 60E
恢复到 200E。这样不仅超参数相同，断点恢复方式也不成为隐藏变量。

### 未采用：单次连续 200E

实现更简单，但 H2RBox 实际在 30E/60E 发生过恢复，连续训练会多引入一个执行
差异。

### 未采用：为 Orbdet 单独调参或改数据集

可能提高结果，但会破坏因果归因；只有严格对照完成后才能进入模型修改阶段。

## 冻结合同

| 项目 | 固定值 |
|---|---|
| dataset | HRSC2016 |
| train / val / held-out test | 436 / 181 / 453 images |
| supervision | qbox -> enclosing hbox -> zero-angle rbox |
| input | 800 x 800 |
| physical GPUs | 8,9 |
| batch | 1/GPU，global 2 |
| optimizer | AdamW，lr 1e-4，weight decay 0.05 |
| gradient clipping | max norm 35 |
| warmup | 500 optimizer steps |
| scheduler | iteration-based MultiStepLR |
| LR milestones | 105060 / 144612 optimizer steps |
| scheduler endpoint | 157590 optimizer steps |
| seed | 3407，deterministic=False |
| steps/epoch | 218 |
| train stages | 0→30E，resume 30→60E，resume 60→200E |
| validation | every 30E on val only；200E 显式补一次 val |
| held-out test | 本轮禁止 |

在 200E 时仅执行 43600 optimizer steps，尚未到第一个 LR milestone；这与恢复版
H2RBox 的实际 200E 运行完全一致。

## 唯一允许的功能变量

| H2RBox reference | Orbdet-v0.1 candidate |
|---|---|
| `H2RBoxDetector` | `OrbdetDetector` |
| `H2RBoxConsistencyLoss` | `OrbdetHarmonicConsistencyLoss` |

Orbdet loss 固定为：

- `loss_weight=0.4`
- `min_quality=0.25`
- `gamma=2.0`
- `high_quality_thr=0.75`
- center/shape/angle loss 与 H2RBox 对应项一致

`OrbdetDetector` 只增加训练期 quality 统计；推理仍使用 H2RBox head 与 rotated
NMS，quality 不得再次乘到测试分数。

除 `model.type`、`loss_bbox_ss`、candidate 独立 `work_dir` 以及 launcher 的唯一
session/port/path 外，合同测试必须证明所有有效配置相等。

## 执行与安全

1. 使用 `/data/zcy/anaconda3/envs/orbdet/bin/python`。
2. 所有双卡命令固定
   `CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。
3. 正式训练前做 4 图、global batch 2、2 optimizer-step 冒泡。
4. 新建独立 work dir；若目录已有 checkpoint，launcher 必须拒绝从头覆盖。
5. 只检测并阻止本候选的精确 config 进程；禁止终止其他用户进程。
6. 正式训练由独立 tmux 托管，并保留 stage、val、错误扫描和 GPU 证据。
7. 不使用已经过期的 08:30 deadline guard。

## 判定规则

从 candidate 的 30/60/90/120/150/180E validation 和显式 200E validation 中取
最大 `dota/mAP`：

- `< 0.8710`：按用户定义判定 Orbdet-v0.1 失败，停止继续堆训练，进入模型修改。
- `>= 0.8710`：v0.1 在 seed 3407 上通过单种子门，但不能直接声称稳定优于；下一步
  应补多 seed。
- 任意 fatal error、NaN/Inf、OOM 或 NCCL failure：工程失败，与算法失败分开记录。

## 产物

- candidate config：
  `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py`
- smoke config：
  `configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_smoke_gpu89.py`
- contract test：`tests/test_orbdet_hrsc_recovery_contract.py`
- formal launcher：
  `scripts/formal/run_orbdet_v0_1_hrsc_recovery_staged_200e_gpu89_20260814.sh`
- result record：
  `resultmd/exp_orbdet_hrsc_recovery/fres_orbdet_v0_1_hrsc_recovery_gpu89_20260814.md`

Orbdet 目录不是 Git worktree，因此本次无法提交 commit；通过不可覆盖目录、合同测试、
配置快照和日志保留可追溯性。
