#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_smoke.py"
work_dir=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_20260816
audit_complete=/data1/zcy/Orbdet/work_dirs/audit/h2rbox_v2_dota1_official_gpu4567_20260816/COMPLETE

on_error() {
  exit_code=$?
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/FAILED"
  rtk echo "Orbdet-v0.2 GPU4567 smoke failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

if [[ ! -e "${audit_complete}" ]]; then
  rtk echo "Official checkpoint audit is not complete: ${audit_complete}" >&2
  exit 2
fi
if rtk pgrep -af '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_ms_rr' >/dev/null; then
  rtk echo 'An Orbdet-v0.2 DOTA-v1 MS+RR job is already running.' >&2
  exit 3
fi
gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 4/5/6/7 are not idle; refusing to start smoke.' >&2
  exit 4
fi

shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || [[ -e "${work_dir}/COMPLETE" ]]; then
  rtk echo "Refusing to overwrite existing smoke outputs in ${work_dir}" >&2
  exit 5
fi

rtk mkdir -p "${work_dir}"
rtk touch "${work_dir}/RUNNING"
rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29663 "${repo_root}/tools/train.py" "${config}" \
  --launcher=pytorch --work-dir="${work_dir}"

if [[ ! -s "${work_dir}/epoch_1.pth" ]]; then
  rtk echo 'GPU4567 smoke did not produce epoch_1.pth.' >&2
  exit 6
fi
rtk touch "${work_dir}/COMPLETE"
rtk echo 'Orbdet-v0.2 DOTA-v1 MS+RR GPU4567 two-step smoke completed.'
