# Orbdet Tasks 1–3 GPU 8/9 最终结果

## 结论

用户授权的任务 1、2、3 已按顺序在物理 GPU 8/9 上全部跑完。任务 1 建立了
Orbdet-v0.2 clean-HRSC 三 seed 稳定性证据；任务 2 完成 GODC 的 smoke 与
103E 正式对照；任务 3 完成 DOTA-v1 12E 数据集规模训练。所有产物只保存在
本地，hidden test 未使用，也没有推送远端。

## 运行合同

| item | value |
|---|---|
| physical_gpus | `8,9` |
| distributed | 2 ranks |
| nccl_safety | `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` |
| python | `/data/zcy/anaconda3/envs/orbdet/bin/python` |
| source_policy | DOTA/HRSC source data read-only |
| remote_policy | local Git only, no push |

## Task 1 — Orbdet-v0.2 clean-HRSC 多 seed

| seed | schedule | best_epoch | val_mAP | AP50 | final_ckpt |
|---:|---:|---:|---:|---:|---|
| 3407 | 103E | 48 | 0.9029 | 0.9030 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814/epoch_103.pth` |
| 42 | 103E | 60 | 0.8992 | 0.8990 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed42_20260815/epoch_103.pth` |
| 2026 | 103E | 72 | 0.8977 | 0.8980 | `work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed2026_20260815/epoch_103.pth` |

三 seed best-val `mAP`：mean `0.8999`，worst `0.8977`，population std
`0.0022`。这支持“在相同 clean-HRSC 合同与三组随机种子下稳定接近 0.90”，
但不应外推为跨数据集或任意种子保证。

- controller: `work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/queue.log`
- completion: `work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/COMPLETE`

## Task 2 — GODC HBox/FPN C2

GODC smoke 完成 2 个 optimizer steps，`loss_godc` 有限且非零，并生成
`epoch_1.pth`。正式 seed 3407 完成 103E：

| epoch | val_mAP | AP50 |
|---:|---:|---:|
| 12 | 0.5560 | 0.5560 |
| 24 | **0.8980** | **0.8980** |
| 36 | 0.8879 | 0.8880 |
| 48 | 0.8932 | 0.8930 |
| 60 | 0.8892 | 0.8890 |
| 72 | 0.8977 | 0.8980 |
| 84 | 0.8968 | 0.8970 |
| 96 | 0.8956 | 0.8960 |

相对同 seed 的 v0.2 baseline best `0.9029`，GODC best 为 `0.8980`
（delta `-0.0049`）。因此该次实验验证了模块可训练、梯度与数值稳定，但没有
提供精度优于 baseline 的证据。

- smoke_ckpt: `work_dirs/smoke/orbdet_godc_c2_hrsc_clean_gpu89_20260815/epoch_1.pth`
- final_ckpt: `work_dirs/formal/orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815/epoch_103.pth`
- best_ckpt: `work_dirs/formal/orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815/best_dota_mAP_epoch_24.pth`
- completion: `work_dirs/formal/orbdet_godc_after_v02_gpu89_20260815/COMPLETE`

## Task 3 — Orbdet-v0.2 DOTA-v1 12E

第一次 smoke 暴露 PyTorch 1.12 CUDA 不支持 Long tensor
`index_reduce(..., reduce='mean')`。失败控制器正确阻断正式训练，原始失败产物
保存在带 `failed_long_index_reduce_0709` 后缀的目录。随后增加整数标签压缩
兼容函数与回归测试，修复提交为本地主分支 `30c3660`；修复后 smoke 与正式
训练均通过。

| field | value |
|---|---|
| source_pairs | 20,995 trainval image/annotation pairs |
| effective_samples | 12,757 with `filter_empty_gt=True` |
| classes | 15 |
| input | 1024×1024 |
| global_batch | 4 |
| schedule | 12E, milestones 8/11 |
| started_at | 2026-08-15 07:14 CST |
| completed_at | 2026-08-15 12:45 CST |
| train_duration | about 5h31m |
| validation | disabled; no independent local val split |

checkpoint 已按合同生成：

- `work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_4.pth`
- `work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_8.pth`
- `work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_12.pth`
- completion: `work_dirs/formal/orbdet_v02_dota1_after_godc_gpu89_20260815/COMPLETE`

最后一次周期日志记录为 epoch 12、iter `3180/3190`：`loss=1.0705`、
`grad_norm=2.8741`、`loss_symmetry_ss=0.0246`，随后于 12:45:31 保存
`epoch_12.pth` 并由控制器写出完成标记。因为合同关闭 validation/test，
本次不产生可合规报告的 DOTA `mAP`。

## 最终验收

- `Task 1`: 两个补充 seed 均生成 `epoch_103.pth`，队列 `COMPLETE`。
- `Task 2`: smoke 生成 `epoch_1.pth`；正式训练生成 `epoch_103.pth`，控制器 `COMPLETE`。
- `Task 3`: smoke 生成 `epoch_1.pth`；正式训练生成 `epoch_12.pth`，控制器 `COMPLETE`。
- 三份最终控制器日志未匹配到 `Traceback`、`RuntimeError`、NCCL error 或 `NaN`。
- 项目测试结果在最终提交前重新运行并记录在 Git 提交中。

## 后续建议

DOTA-v1 当前只有训练 checkpoint，没有独立 validation 指标。若要比较检测精度，
下一步应先冻结 checkpoint/推理配置，再在合规的独立标注验证集上评估，或生成
官方 test submission；不要用 trainval 自评替代泛化指标。GODC 当前应视为
“工程与优化可行、精度未胜 baseline”，后续需要跨 seed 或结构/权重消融后再决定
是否纳入主模型。
