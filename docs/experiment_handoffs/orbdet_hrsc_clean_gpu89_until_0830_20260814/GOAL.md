# 实验目标

## 核心问题

在完全相同的数据、优化器、batch、schedule、seed 和评测协议下，检验
`OrbdetHarmonicConsistencyLoss` 是否能缓解原版 H2RBox 在 HRSC 小数据集上的
方向学习退化。

## 唯一功能变量

| baseline | candidate |
|---|---|
| `H2RBoxDetector` | `OrbdetDetector` |
| `H2RBoxConsistencyLoss` | `OrbdetHarmonicConsistencyLoss` |

candidate 固定参数：

- `loss_weight=0.4`
- `min_quality=0.25`
- `gamma=2.0`
- `high_quality_thr=0.75`
- center/shape/angle loss 与 baseline 保持一致

`OrbdetDetector` 只增加训练期质量统计；推理仍沿用 H2RBox head 和 rotated NMS，
不得把 harmonic quality 再乘到分类分数。

## 固定实验合同

| item | value |
|---|---|
| dataset | HRSC2016 |
| train / val / held-out test | 436 / 181 / 453 images |
| input | 800 x 800 |
| physical GPUs | 8,9 |
| batch | 2/GPU，global 4 |
| optimizer | AdamW，lr 5e-5，weight decay 0.05 |
| gradient clipping | max norm 35 |
| warmup | 500 optimizer iterations |
| training | 200E |
| LR milestones | 133/184，gamma 0.1 |
| validation | 每 10E，只使用 val |
| checkpoint selection | 最大 val `dota/mAP` |
| test policy | best-val 选定后，held-out test 只评一次 |
| seed | 3407 |

## 时间边界

GPU 8/9 的本实验使用权最晚截止：

```text
2026-08-14 08:30:00 +08:00
Unix epoch: 1786667400
```

截止时尚未完成的本任务应标记为 `time_capped`，不是算法失败。停止操作只能匹配
本项目的精确 config/session，禁止终止其他用户进程。

## 完成判定

- `complete`：H2RBox 与 clean Orbdet 都有 best-val checkpoint 和一次 held-out
  test 结果。
- `candidate_complete_test_pending`：Orbdet 训练完成，但截止时间阻止 test。
- `time_capped`：08:30 停止了未完成的本任务。
- `failed`：配置错误、NaN/Inf、OOM、NCCL 或其他工程故障。

不预设或伪造目标 AP。最终只报告实际日志中的 mAP/AP50、recall、训练时间和显存。

