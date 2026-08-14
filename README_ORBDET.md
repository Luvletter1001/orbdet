# Orbdet

Orbdet 是独立于 OpenRSD 与 OV-CapFlow 的 CNN-based HBox-supervised OBB
研究目录。当前只完成基础组件整理、独立环境创建和 DOTA-v1 GPU 8/9 双卡
冒泡；正式训练尚未启动。

## Current status

| item | value |
|---|---|
| project_root | `/data1/zcy/Orbdet` |
| conda_env | `/data/zcy/anaconda3/envs/orbdet` |
| parent_model | `H2RBoxDetector` with R50-FPN |
| supervision | `qbox -> hbox -> rbox` |
| dataset | DOTA-v1, 15 classes, 20,995 trainval pairs |
| physical_gpus | `8,9` |
| smoke_scope | 8 train images, 4 test images, 1 epoch, 4 DDP steps |
| formal_train_status | `not_started` |

## Components

- `mmrotate/`, `mmdet/`, `mmrotate_configs/`, `mmdet_configs/`, `tools/`:
  从干净 OpenRSD 快照
  `12d3fd8b75e8b64ec53fded9cf035a2306d58874` 的 Git 跟踪文件提取。
- `vendor/openrsd_snapshot/mmengine/`: 保留快照中的定制 MMEngine 源码供审计；
  该目录因上游 `.gitignore` 漏掉 `mmengine/dist/` 而不作为运行时包。
- 运行时 MMEngine 来自 `orbdet` 环境中的完整 `mmengine==0.10.3`。
- `data/DOTA-v1.0/trainval` 和 `data/DOTA-v1.0/test` 是现有数据的符号链接，
  不复制、不修改原始数据。

## Smoke entry points

- Config: `configs/orbdet/h2rbox_r50_dota1_smoke.py`
- Train smoke: `scripts/smoke/run_dota1_gpu89.sh`
- Test smoke: `scripts/smoke/test_dota1_gpu89.sh`
- Contract: `tests/test_dota1_smoke_contract.py`
- Receipt: `docs/smoke/DOTA1_GPU89_SMOKE_RECEIPT.md`

运行契约测试：

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_dota1_smoke_contract.py
```

`scripts/smoke/` 只用于边界严格的冒泡。正式训练必须先获得用户显式授权，
并创建独立、可审查的正式 config；不要通过扩大 smoke config 的 `indices` 或
`max_epochs` 绕过停止门。

