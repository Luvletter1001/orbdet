#!/usr/bin/env bash
set -euo pipefail

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_1_r50_hrsc.py' \
  >/dev/null; then
  rtk echo 'Orbdet-v0.1 HRSC formal training is already running.' >&2
  exit 2
fi

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29889 \
  /data1/zcy/Orbdet/tools/train.py \
  /data1/zcy/Orbdet/configs/orbdet/orbdet_v0_1_r50_hrsc.py \
  --launcher=pytorch \
  --work-dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_gpu89
