#!/usr/bin/env bash
set -euo pipefail

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
  --master_port=29614 \
  /data1/zcy/Orbdet/tools/train.py \
  /data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike_smoke.py \
  --launcher=pytorch \
  --work-dir=/data1/zcy/Orbdet/work_dirs/smoke/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814
