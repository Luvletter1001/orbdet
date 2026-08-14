# Orbdet DOTA-v1 GPU 8/9 冒泡执行回执

## 结论

Orbdet 的 CNN-based HBox-supervised OBB 基础链路已通过双卡冒泡：配置解析、
DOTA-v1 数据加载、`qbox -> hbox -> rbox` 监督转换、H2RBox 模型构建、前向、
反向、AdamW 更新、NCCL/DDP 同步、checkpoint 保存/加载和 `DOTAMetric` 评测
均实际执行。冒泡后已停止，正式训练未启动。

`dota/mAP=0.0000` 只来自随机初始化模型的 4-step smoke run，不代表模型质量，
也不能用于论文对比或方案取舍。

## Scope

| key | value |
|---|---|
| date | `2026-08-12` |
| timezone | `Asia/Shanghai (UTC+08:00)` |
| project_root | `/data1/zcy/Orbdet` |
| conda_env | `/data/zcy/anaconda3/envs/orbdet` |
| physical_gpus | `8,9` |
| visible_ranks | `GPU 0,1` after `CUDA_VISIBLE_DEVICES=8,9` |
| nccl_safety | `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` |
| model | `H2RBoxDetector`, R50-FPN, 32,147,803 params |
| supervision | `qbox -> hbox -> rbox` |
| train_subset | `8` images |
| test_subset | `4` images |
| per_gpu_batch | `1` |
| train_schedule | `1 epoch`, `4 DDP optimizer steps` |
| pretrained_init | disabled (`backbone.init_cfg=None`) |
| formal_train_status | `not_started` |

## Provenance and data

基础组件不是从含大量未提交修改的 OpenRSD 工作树直接复制，而是从干净快照
`12d3fd8b75e8b64ec53fded9cf035a2306d58874` 按 allowlist 提取。归档
SHA-256 为
`554a3a8c68564a9b287e03f54dc4950e26c55a06dd74e361d336893d9ab3c718`。

| data_key | value |
|---|---:|
| trainval_annfiles | 20,995 |
| trainval_images | 20,995 |
| paired_image_annotation | 20,995 |
| classes | 15 |
| trainval_link | `/data1/zcy/Orbdet/data/DOTA-v1.0/trainval -> /data/zcy/dataset/trainval_ss` |
| test_link | `/data1/zcy/Orbdet/data/DOTA-v1.0/test -> /data/zcy/dataset/test_ss` |

## Environment

| package | version |
|---|---|
| Python | `3.10.20` |
| PyTorch | `1.12.1+cu113` |
| TorchVision | `0.13.1+cu113` |
| NumPy | `1.26.4` |
| SciPy | `1.11.4` |
| MMCV | `2.2.0` |
| MMEngine | `0.10.3` |
| MMDetection | `3.3.0` |
| MMRotate | `1.0.0rc1` |
| OpenCV | `4.9.0` |
| charset-normalizer | `3.4.4` |

环境从 `/data/zcy/anaconda3/envs/openrsd` 本地克隆后做了三项最小隔离修复：

1. 移除与 PyTorch 1.12.1 冲突、且 CNN/H2RBox 不使用的可选
   `open-clip-torch 3.3.0`。
2. 将因上游 `.gitignore` 漏掉 `mmengine/dist/` 而不完整的本地 MMEngine
   快照移到 `vendor/openrsd_snapshot/mmengine/`，运行时使用环境内完整
   `mmengine==0.10.3`。
3. 修复 Conda clone 造成的 NumPy 1.26.4 元数据/2.2.5 运行文件混装和
   charset-normalizer 3.4.4/3.4.6 混装；原混装文件均保存在
   `vendor/environment_backups/`，没有删除。

最终 `pip check` 输出 `No broken requirements found.`，editable 映射只包含
`mmdet` 与 `mmrotate`，均指向 `/data1/zcy/Orbdet`。

## Train smoke evidence

| metric | value |
|---|---:|
| train_exit | `0` |
| distributed | `True` |
| world_size | `2` |
| completed_steps | `4/4` |
| step_losses | `3.4844, 7.7283, 6.4617, 6.0198` |
| peak_logged_memory | `824 MB/rank` |
| checkpoint | `/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89/epoch_1.pth` |
| checkpoint_size | `366.7 MiB` |
| checkpoint_sha256 | `0a970130329e470d1e62a050192612067cff55e95a50b0d5eb30ec024ecdca87` |
| train_log | `/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89/20260812_174044/20260812_174044.log` |

四个 step 的总 loss 及各分量均为有限值，日志记录
`Saving checkpoint at 1 epochs`。

## Test smoke evidence

| metric | value |
|---|---:|
| test_exit | `0` |
| distributed | `True` |
| world_size | `2` |
| completed_steps | `2/2 per rank` |
| dota/mAP | `0.0000` |
| dota/AP50 | `0.0000` |
| test_log | `/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89/20260812_174126/20260812_174126.log` |
| metric_json | `/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89/20260812_174126/20260812_174126.json` |

评测日志明确记录 checkpoint 加载成功并输出 15 类 DOTA 表格与稳定 metric key。

## Verification and stopping state

| check | result |
|---|---|
| contract_tests | `7 passed` after stop marker installation |
| config_compile | `pass` |
| shell_syntax | `pass` for train/test launchers |
| dataset_build | `pass`, train=8, test=4 |
| model_build | `pass`, 32,147,803 params |
| dataloader_batch | `pass`, one `(3, 256, 256)` image with GT |
| checkpoint_exists | `pass` |
| metric_output_exists | `pass` |
| remaining_orbdet_processes | `0` |
| formal_queue_or_background_job | `none` |

决定：停在冒泡完成状态，不启动正式训练。后续只有在用户显式授权后，才设计、
审查并启动独立的正式配置。

