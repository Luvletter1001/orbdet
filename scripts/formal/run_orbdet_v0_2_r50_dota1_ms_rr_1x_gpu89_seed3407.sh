#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py"
work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407_20260815
audit_complete=/data1/zcy/Orbdet/work_dirs/audit/h2rbox_v2_dota1_official_20260815/COMPLETE
smoke_complete=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_20260815/COMPLETE
smoke_checkpoint=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_20260815/epoch_1.pth

on_error() {
  exit_code=$?
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/FAILED"
  rtk echo "Orbdet-v0.2 DOTA-v1 MS+RR formal failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

if [[ ! -e "${audit_complete}" ]]; then
  rtk echo "Official checkpoint audit is not complete: ${audit_complete}" >&2
  exit 2
fi
if [[ ! -e "${smoke_complete}" || ! -s "${smoke_checkpoint}" ]]; then
  rtk echo 'The required two-rank MS+RR smoke is not complete.' >&2
  exit 3
fi
if rtk pgrep -af \
  '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py' \
  >/dev/null; then
  rtk echo 'An Orbdet-v0.2 DOTA-v1 MS+RR formal job is already running.' >&2
  exit 4
fi

gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid \
  --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing to start MS+RR formal.' >&2
  exit 5
fi

shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || \
   [[ -e "${work_dir}/COMPLETE" ]]; then
  rtk echo "Refusing to overwrite existing formal outputs in ${work_dir}" >&2
  exit 6
fi

rtk mkdir -p "${work_dir}"
rtk touch "${work_dir}/RUNNING"
rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29654 \
  "${repo_root}/tools/train.py" \
  "${config}" \
  --launcher=pytorch \
  --work-dir="${work_dir}"

if [[ ! -s "${work_dir}/epoch_12.pth" ]]; then
  rtk echo 'Formal MS+RR run did not produce epoch_12.pth.' >&2
  exit 7
fi
rtk touch "${work_dir}/COMPLETE"
rtk echo 'Orbdet-v0.2 DOTA-v1 R50 MS+RR 12E formal completed.'
