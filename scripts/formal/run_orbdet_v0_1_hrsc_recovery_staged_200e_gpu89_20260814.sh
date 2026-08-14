#!/usr/bin/env bash
set -euo pipefail

config=/data1/zcy/Orbdet/configs/orbdet/orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py
work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_recovery_bs1_stepaligned_200e_gpu89_seed3407_20260814
val_eval_dir=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_1_hrsc_recovery_epoch200_val_gpu8_20260814
train_eval_dir=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_1_hrsc_recovery_epoch200_train_gpu8_20260814

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py' \
  >/dev/null; then
  rtk echo 'Orbdet-v0.1 HRSC recovery-contract training is already running.' >&2
  exit 2
fi

shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )); then
  rtk echo "Refusing to overwrite existing checkpoints in ${work_dir}" >&2
  exit 3
fi

rtk mkdir -p "${work_dir}"

run_stage() {
  epoch_override="$1"
  shift
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
    --master_port=29625 \
    /data1/zcy/Orbdet/tools/train.py \
    "${config}" \
    --launcher=pytorch \
    --work-dir="${work_dir}" \
    "$@" \
    --cfg-options \
    "${epoch_override}" \
    default_hooks.checkpoint.max_keep_ckpts=7 \
    default_hooks.checkpoint.save_last=True
}

run_stage train_cfg.max_epochs=30
if [[ ! -s "${work_dir}/epoch_30.pth" ]]; then
  rtk echo 'Missing stage checkpoint: epoch_30.pth' >&2
  exit 4
fi

run_stage train_cfg.max_epochs=60 --resume
if [[ ! -s "${work_dir}/epoch_60.pth" ]]; then
  rtk echo 'Missing stage checkpoint: epoch_60.pth' >&2
  exit 5
fi

run_stage train_cfg.max_epochs=200 --resume
if [[ ! -s "${work_dir}/epoch_200.pth" ]]; then
  rtk echo 'Missing final checkpoint: epoch_200.pth' >&2
  exit 6
fi

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  /data1/zcy/Orbdet/tools/test.py \
  "${config}" \
  "${work_dir}/epoch_200.pth" \
  --work-dir="${val_eval_dir}" \
  --cfg-options test_dataloader.dataset.ann_file=ImageSets/val.txt

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/data1/zcy/Orbdet \
  CUDA_VISIBLE_DEVICES=8 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  /data1/zcy/Orbdet/tools/test.py \
  "${config}" \
  "${work_dir}/epoch_200.pth" \
  --work-dir="${train_eval_dir}" \
  --cfg-options test_dataloader.dataset.ann_file=ImageSets/train.txt
