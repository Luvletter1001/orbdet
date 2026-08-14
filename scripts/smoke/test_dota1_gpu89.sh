#!/usr/bin/env bash
rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29689 \
  /data1/zcy/Orbdet/tools/test.py \
  /data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_dota1_smoke.py \
  /data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89/epoch_1.pth \
  --launcher=pytorch \
  --work-dir=/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89

