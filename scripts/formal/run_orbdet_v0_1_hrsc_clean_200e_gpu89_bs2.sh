#!/usr/bin/env bash
set -euo pipefail

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py' \
  >/dev/null; then
  rtk echo 'Clean Orbdet HRSC 200E GPU 8/9 training is already running.' >&2
  exit 2
fi

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29618 \
  /data1/zcy/Orbdet/tools/train.py \
  /data1/zcy/Orbdet/configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py \
  --launcher=pytorch \
  --work-dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814
