#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89_smoke.py"
work_dir=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_dota1_1x_gpu89_20260815

if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_1x_gpu89' \
  >/dev/null; then
  rtk echo 'An Orbdet-v0.2 DOTA-v1 job is already running.' >&2
  exit 2
fi

gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid \
  --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing to start DOTA-v1 smoke.' >&2
  exit 4
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
  PYTHONPATH="${repo_root}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29645 \
  "${repo_root}/tools/train.py" \
  "${config}" \
  --launcher=pytorch \
  --work-dir="${work_dir}"
