# Orbdet-v0.2 HRSC Clean GPU 8/9 — Seed 3407

## 状态

- 当前：正式训练运行中，目标 103E。
- 非塌缩门：通过。
- 当前最佳：epoch 24，clean validation mAP `0.8946`，AP50 `0.8950`。
- held-out test：未运行。

## 修复内容

Orbdet-v0.1 的 harmonic `q` 由被加权的同一角度残差产生，clean recovery
实验中出现 `q_mean=0.9980`、val mAP 仅 `0.0893` 的错误一致退化。

v0.2 使用官方 H2RBox-v2 的三视图方向锚点：

1. original/rotated/vertical-flipped 共享参数；
2. 固定 `bid` 聚合同一目标在三视图中的预测；
3. PSC coder + snap rotation/flip losses处理角度周期；
4. 新 `q_rot/q_flip/q_joint/q_anchored` 全部 detach，仅记录，不加权 loss，
   不参与推理。

移植来源：`yuyi1005/mmrotate` dev-1.x commit
`97b793577199e80f85f199b6bffb99b03017c4f4`。三个官方实现文件与来源
SHA256 完全一致。

## 实验合同

| 项目 | 设置 |
|---|---|
| train / val / held-out test | 436 / 181 / 453 images |
| 训练监督 | `qbox -> hbox -> rbox`，丢弃真实方向 |
| 模型 | R50-FPN + H2RBoxV2Head + PSC |
| GPU | physical 8/9，2 ranks |
| batch | 2/rank，global 4 |
| optimizer | AdamW, lr `5e-5`, wd `0.005` |
| schedule | 103E；step milestones 7440/10230 |
| seed | 3407 |
| work dir | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814` |

## 验证结果

| epoch | val mAP | AP50 | recall | 判定 |
|---:|---:|---:|---:|---|
| 12 | 0.7702 | 0.7700 | 0.858 | 正常学习 |
| 24 | **0.8946** | **0.8950** | **0.919** | 通过 0.70 early gate 与 0.88 首-seed门 |

## 工程验证

- TDD RED：新模块缺失时 collection 按预期失败。
- focused GREEN：`8 passed`。
- 全项目回归：`36 passed`，只有两个既有 warning。
- GPU89 smoke：2/2 optimizer steps；finite loss/grad/q；峰值约
  `5607 MiB/GPU`；无 traceback、NCCL error 或 rank failure。
- 正式训练稳定吞吐约 `0.314 s/step`。

## 仍未成立的结论

seed 3407 已达到 88，但不能据此声称“每次稳定 88”。必须在相同合同下完成
seed 42 和 2026，并检查三 seed 的 mean/worst/std。held-out test 只允许在模型与
checkpoint 规则冻结后做一次。

