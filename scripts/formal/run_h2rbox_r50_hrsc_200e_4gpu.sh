#!/usr/bin/env bash
set -euo pipefail

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*h2rbox_r50_hrsc_200e_4gpu.py' \
  >/dev/null; then
  rtk echo 'H2RBox HRSC bs8 200E four-GPU baseline is already running.' >&2
  exit 2
fi

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29605 \
  /data1/zcy/Orbdet/tools/train.py \
  /data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_hrsc_200e_4gpu.py \
  --launcher=pytorch \
  --work-dir=/data1/zcy/Orbdet/work_dirs/formal/h2rbox_r50_hrsc_trainval_200e_gpu0123_bs8_seed3407
