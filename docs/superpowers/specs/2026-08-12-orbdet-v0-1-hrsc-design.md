# Orbdet-v0.1 HRSC2016 双卡训练设计

## 1. 目标与边界

本轮直接实现并训练 `Orbdet-v0.1`，不安排独立 H2RBox 基线。任务使用
HRSC2016 单类船舶检测数据，在物理 GPU 8/9 上训练，目标总墙钟预算约 12 小时，
并值守到验证 `AP50` 进入合理区间且走势稳定。

本版本的研究边界是：

- CNN-based 密集检测，不引入 transformer query decoder；
- 训练只使用由 OBB 外接得到的 HBox，模型不得读取真实 OBB 方向；
- 测试使用官方 OBB GT；
- FMSA 草稿只吸收圆周 Fourier 方向置信度思想；
- fixed-query 实验只吸收失败经验：保留成熟候选生成，不将独立质量分数再次乘入
  推理分类分数，不复用已失败的手写 fixed-query posterior。

## 2. 数据协议

数据源固定为 `/data/zcy/dataset/HRSC_unzip`，在 Orbdet 中只创建只读符号链接
`data/hrsc`。使用官方划分：

| split | images | role |
|---|---:|---|
| `ImageSets/trainval.txt` | 617 | train |
| `ImageSets/test.txt` | 453 | val/test |

训练流水线为 `Load qbox -> Convert hbox -> Convert rbox -> Resize/Pad`。第二次转换
产生角度恒为零的旋转框，仅用于复用 H2RBox 目标接口，不恢复原始方向。验证流水线为
`Load qbox -> Convert rbox`，仅用于 OBB 指标计算。训练与测试 ID 按官方文件隔离。

输入采用 `800 x 800` 方形画布。HRSC 原图保持比例缩放后补边；方形画布避免随机旋转
视图在长方形边界上系统裁掉细长船舶。

## 3. 模型架构

### 3.1 主干

- `ResNet-50 + FPN + FCOS-style rotated dense head`；
- ImageNet ResNet-50 初始化；
- 原图弱监督分支负责分类、中心度和 HBox 外接约束；
- 随机旋转分支负责方向与形状一致性。

### 3.2 Orbdet 二阶圆周谐波可靠度

设原图预测方向旋转到增强视图后的目标为 `theta_t`，增强视图预测方向为
`theta_p`，差值为 `delta = theta_p - theta_t`。方向具有 `pi` 周期，同时框的
宽高交换产生 `pi/2` 等价轴，因此先计算：

```text
r_raw = max(|cos(delta)|, |sin(delta)|)
r = clamp((r_raw - sqrt(1/2)) / (1 - sqrt(1/2)), 0, 1)
q = q_min + (1 - q_min) * r^gamma
```

`|cos(delta)|` 等价于两个方向二阶圆周谐波的 resultant agreement；
`|sin(delta)|` 处理宽高交换分支。默认 `q_min=0.25`、`gamma=2`。

`q` 在用于损失加权前 `detach`。它只表示当前伪方向的一致性可靠度，不允许网络通过
反向传播直接抬高权重。保留 `q_min` 可避免训练早期低置信样本完全失去梯度。

### 3.3 损失和推理

原有 H2RBox 一致性损失中的中心、形状与周期角度项使用
`centerness * q` 加权，其余分类/HBox/中心度损失不变。训练日志额外记录：

- `orbdet_q_mean`
- `orbdet_q_min`
- `orbdet_q_high_frac` (`q >= 0.75`)

推理仍走标准 FCOS 密集候选与 rotated NMS，分类分数和中心度保持原路径；`q` 不进入
推理打分。这一约束直接来自 fixed-query 实验中“候选绑定不足、质量分二次相乘导致
排序塌缩”的经验。

## 4. 训练与时间预算

使用物理 GPU 8/9，固定环境：

```text
CUDA_VISIBLE_DEVICES=8,9
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
PYTHONNOUSERSITE=1
```

默认每卡 `batch_size=1`，全局 batch 为 2；不设置无类型的 `batch_sampler`。
优化器采用 AdamW，基础学习率 `1e-4`，带线性 warmup 和按实际总 epoch 比例生成的
两段衰减点。

先运行同模型、同尺寸的完整 1 epoch 标定，不在标定阶段汇报 AP。根据实测训练
`sec/epoch` 和一次完整验证耗时，计算正式 `max_epochs`，使训练、周期验证和最终评测
合计目标为 11–12 小时。正式上限不超过 12.5 小时；保存最近 3 个 checkpoint 与
`best_dota_mAP`。

验证间隔根据标定后的 epoch 总数确定，使全程产生约 12–20 个 AP 点；不得通过高频
验证挤占主要训练预算。

## 5. 稳定与停止判据

HRSC 使用 `DOTAMetric` 的 `dota/mAP`，在本协议中按 `AP50` 解读。仓库模型卡显示，
HRSC HBox-supervised/related R50 方法的 AP50 约为 0.79–0.90，因此本轮定义：

- `AP50 < 0.50`：明显异常，需要诊断；
- `0.70 <= AP50 < 0.80`：已学习但未进入合理目标区间；
- `0.80 <= AP50 <= 0.92`：合理区间；
- 连续 3 个验证点的 `max-min <= 0.015`，且最后两段最佳提升均 `< 0.005`：稳定。

若约 4 小时后 AP50 仍低于 0.50，或出现 NaN/Inf、持续 loss 爆炸、DDP 退出，立即停止
当前进程，保留日志和 checkpoint，定位原因后在剩余预算内恢复。不得用延长训练掩盖
结构错误。若提前达到合理且稳定区间，可在最近 checkpoint/验证点后干净停止；否则跑至
时间上限并如实报告未达标。

## 6. 测试与可审计性

实施必须经过以下门禁：

1. 单元测试验证谐波可靠度的周期性、宽高交换等价性、下限和有限梯度；
2. 配置合同验证训练流水线确实丢弃 OBB 方向、测试保留 OBB、类别数为 1；
3. 启动器合同验证 GPU 与 NCCL 环境变量、正式目录和参数；
4. 8/9 双卡短冒泡验证 DDP、前后向、日志诊断量和 checkpoint；
5. 正式训练前再次检查 GPU 进程归属，不终止其他用户进程；
6. 每次验证记录 epoch、AP、loss、显存和训练状态，最终写中文回执。

## 7. 产物

- `mmrotate/models/losses/orbdet_harmonic_consistency_loss.py`
- `mmrotate/models/detectors/orbdet.py`
- `configs/orbdet/orbdet_v0_1_r50_hrsc.py`
- `configs/orbdet/orbdet_v0_1_r50_hrsc_calibration.py`
- `scripts/formal/run_orbdet_v0_1_hrsc_gpu89.sh`
- `tests/test_orbdet_v0_1.py`
- `resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md`

Orbdet 当前没有 `.git`，因此不初始化仓库、不伪造 commit；文件、命令输出、日志和校验值
构成本轮审计证据。
