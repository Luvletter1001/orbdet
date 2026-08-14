#!/usr/bin/env bash
set -euo pipefail

config=/data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py
work_dir=/data1/zcy/Orbdet/work_dirs/formal/h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814
checkpoint="${work_dir}/epoch_200.pth"

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py' \
  >/dev/null; then
  rtk echo 'H2RBox HRSC recovery training is already running.' >&2
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
  --master_port=29623 \
  /data1/zcy/Orbdet/tools/train.py \
  "${config}" \
  --launcher=pytorch \
  --work-dir="${work_dir}" \
  --resume \
  --cfg-options \
  train_cfg.max_epochs=200 \
  default_hooks.checkpoint.max_keep_ckpts=7 \
  default_hooks.checkpoint.save_last=True

if [[ ! -s "${checkpoint}" ]]; then
  rtk echo "Missing final checkpoint: ${checkpoint}" >&2
  exit 3
fi

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  /data1/zcy/Orbdet/tools/test.py \
  "${config}" \
  "${checkpoint}" \
  --work-dir=/data1/zcy/Orbdet/work_dirs/eval/h2rbox_hrsc_recovery_epoch200_val_gpu8_20260814 \
  --cfg-options test_dataloader.dataset.ann_file=ImageSets/val.txt

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  /data1/zcy/Orbdet/tools/test.py \
  "${config}" \
  "${checkpoint}" \
  --work-dir=/data1/zcy/Orbdet/work_dirs/eval/h2rbox_hrsc_recovery_epoch200_train_gpu8_20260814 \
  --cfg-options test_dataloader.dataset.ann_file=ImageSets/train.txt
