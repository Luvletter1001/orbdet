#!/usr/bin/env bash
set -euo pipefail

config=/data1/zcy/Orbdet/configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py
work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_2_r50_hrsc_clean_gpu89.py' \
  >/dev/null; then
  rtk echo 'Orbdet-v0.2 clean HRSC training is already running.' >&2
  exit 2
fi

shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )); then
  rtk echo "Refusing to overwrite existing checkpoints in ${work_dir}" >&2
  exit 3
fi

rtk mkdir -p "${work_dir}"

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
  --master_port=29633 \
  /data1/zcy/Orbdet/tools/train.py \
  "${config}" \
  --launcher=pytorch \
  --work-dir="${work_dir}"

