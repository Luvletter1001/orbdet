# H2RBox-v2 DOTA-v1 官方 Checkpoint 链路审计

## 当前结论

官方 SS 与 MS+RR checkpoint 已从 OpenMMLab 模型库下载并完成哈希、metadata
与 `strict=True` 模型加载检查。2026-08-16 使用物理 GPU 4–7 完成三段正式推理
审计：SS trainval self-eval 为 `mAP=0.8131`、`AP50=0.8130`，SS 与 MS+RR
submission 均完成 patch merge，并分别生成通过完整性检查的 15 文件 ZIP。

一次沙箱内拒绝路径检查因无法访问宿主 NVIDIA driver 返回 exit 9；同一 launcher
在宿主 GPU 权限下正确返回 exit 6：`GPU 8/9 are not idle`。该次非实验性空
`FAILED` marker 已保留并改名为
`SANDBOX_DRIVER_ACCESS_CHECK_FAILED_20260815`，不会与正式审计状态混淆。

| field | SS | MS+RR |
|---|---|---|
| model_zoo_AP50 | 72.59 | 78.25 |
| checkpoint_epoch | 12 | 12 |
| checkpoint_iter | 76,800 | 409,956 |
| checkpoint_state_tensors | 371 | 371 |
| checkpoint_sha256 | `fa5ad1d2d6d030a477fe6f9a405863a76f55d463cff31b7e01c8366527888fde` | `5e0e53e12e0d8b07f79922b6cef9b56c142458483d2c1e2f27348f7e72e9d677` |
| strict_load | pass | pass |
| inference_status | pass | pass |

## 2026-08-16 GPU4567 实际结果

| field | SS | MS+RR |
|---|---|---|
| physical_gpus | 4,5,6,7 | 4,5,6,7 |
| nproc | 4 | 4 |
| patch_count | 10,833 | 71,888 |
| trainval_mAP | `0.8131` | N/A |
| trainval_AP50 | `0.8130` | N/A |
| zip_files | 15 | 15 |
| zip_test | pass | pass |
| zip_sha256 | `fd298f9028ac7805ddddf64d4515d6a3da56a9d418c6295893caa3517ced4fa7` | `75b76cb1d8a94192a41b6fbc3d5665018c573958f5b8a6c17abf0631047a6aab` |

SS 与 MS+RR ZIP 分别约 9.2 MiB 与 18 MiB。两次启动器问题均在 GPU 训练
开始前 fail-fast：第一次为 GNU `timeout` 时间格式，第二次为 GPU89 配置内嵌
`outfile_prefix`。问题已按 TDD 修复，失败目录保留在带
`failed_invalid_timeout` / `failed_outfile_prefix` 后缀的 runtime 路径，没有覆盖
正式通过结果。

官方入口：

- `https://github.com/open-mmlab/mmrotate/blob/dev-1.x/configs/h2rbox_v2/README.md`
- `https://download.openmmlab.com/mmrotate/v1.0/h2rbox_v2/`

## Checkpoint 内嵌协议

本次复现以 checkpoint 的 `meta.cfg` 为权威，因为当前 GitHub 配置与 2023 年实际
产出权重的配置存在字段和值的演化。MS+RR checkpoint 明确记录：

| field | value |
|---|---|
| global_batch | 2 |
| optimizer | AdamW |
| lr | `5e-5` |
| weight_decay | `0.005` |
| max_epochs | 12 |
| milestones | `[8, 11]` |
| rotation_agnostic_classes | `[1, 9, 11]` |
| agnostic_resize_classes | `[1]`，旧名 `rotation_agnostic_resize_classes` |
| use_reweighted_loss_bbox | `True` |
| RR | `RandomRotate(prob=1, angle_range=180)` |

SS 官方日志 epoch 12 的 trainval self-eval 为
`dota/mAP=0.8132380843`、`dota/AP50=0.8132`。该值只用于本地链路 sanity
check，不替代在线 test AP。

## 本地数据合同

| split | path | patch_count | role |
|---|---|---:|---|
| SS trainval | `data/DOTA-v1.0/trainval/` | 20,995 | labelled self-eval |
| SS test | `data/DOTA-v1.0/test/` | 10,833 | SS merge/ZIP |
| MS test | `/data/zcy/dataset/test_ms/` | 71,888 | MS merge/ZIP |
| MS trainval raw | `/data/zcy/dataset/trainval_ms_full/` | 138,883 | read-only source |
| MS trainval nonempty | same | 68,325 | formal sampler |

MS patches 的第二个文件名字段是实际窗口边长，例如 `682`、`1024`。切分工具在
原图坐标系采用不同窗口大小，推理 pipeline 再 resize 到 1024 并 rescale 回窗口
坐标，因此现有 `DOTAMetric` 加回 `x/y` offset 的 merge 逻辑适用，不需要额外
scale 除法。

## 审计通过条件

- 两份 checkpoint 均 strict load，无 missing/unexpected key。
- dataset length 严格等于 `20,995 / 10,833 / 71,888`。
- SS trainval AP 与官方日志记录处于合理邻域。
- SS 与 MS 推理均完成 patch-to-original merge。
- 两个 ZIP 均恰好包含 15 个根目录 `Task1_<class>.txt`，压缩完整性通过。
- 日志无 `Traceback`、`RuntimeError`、NCCL error 或 `NaN`。
- 在线 AP 仍需分别上传 ZIP 后确认；本地结构检查不能伪装成在线链路确认。

## 可复现入口

- contract test：
  `tests/test_h2rbox_v2_dota1_official_audit_msrr_contract.py`
- audit launcher：
  `scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu89.sh`
- runtime root：
  `work_dirs/audit/h2rbox_v2_dota1_official_20260815/`
- GPU4567 audit launcher：
  `scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu4567.sh`
- GPU4567 runtime root：
  `work_dirs/audit/h2rbox_v2_dota1_official_gpu4567_20260816/`
- SS ZIP：
  `work_dirs/audit/h2rbox_v2_dota1_official_gpu4567_20260816/ss_submission/h2rbox_v2_official_ss_task1/h2rbox_v2_official_ss_task1.zip`
- MS+RR ZIP：
  `work_dirs/audit/h2rbox_v2_dota1_official_gpu4567_20260816/ms_submission/h2rbox_v2_official_msrr_task1/h2rbox_v2_official_msrr_task1.zip`
