#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator="${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py"
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py"
work_dir=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818
deadline='2026-08-18 08:20:00 +0800'

on_error() {
  exit_code=$?
  trap - ERR INT TERM
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/FAILED"
  rtk echo "SS seed-42 smoke failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  rtk mkdir -p "${work_dir}"
  rtk touch "${work_dir}/INTERRUPTED"
  rtk echo 'SS seed-42 smoke interrupted.' >&2
  exit 130
}
trap on_error ERR
trap on_signal INT TERM

deadline_epoch="$(rtk date -d "${deadline}" +%s)"
now_epoch="$(rtk date +%s)"
remaining_seconds=$(( deadline_epoch - now_epoch ))
if (( remaining_seconds <= 0 )); then
  rtk echo "Deadline has passed: ${deadline}" >&2
  exit 2
fi
if rtk pgrep -af '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py' >/dev/null; then
  rtk echo 'The SS seed-42 smoke is already running.' >&2
  exit 3
fi
gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing to start SS seed-42 smoke.' >&2
  exit 4
fi
shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || [[ -e "${work_dir}/RUNNING" || -e "${work_dir}/COMPLETE" || -e "${work_dir}/FAILED" || -e "${work_dir}/INTERRUPTED" ]]; then
  rtk echo "Refusing to overwrite existing outputs or markers in ${work_dir}" >&2
  exit 5
fi

rtk mkdir -p "${work_dir}"
rtk touch "${work_dir}/RUNNING"
set +e
rtk timeout --foreground --signal=TERM --kill-after=30s "${remaining_seconds}s" \
  rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=2 \
  --master_port=29671 "${repo_root}/tools/train.py" "${config}" \
  --launcher=pytorch --work-dir="${work_dir}"
train_exit=$?
set -e
if [[ ${train_exit} -eq 124 || ${train_exit} -eq 137 || ${train_exit} -eq 143 ]]; then
  trap - ERR INT TERM
  rtk touch "${work_dir}/INTERRUPTED"
  rtk echo 'SS seed-42 smoke reached the deadline.' >&2
  exit "${train_exit}"
fi
if [[ ${train_exit} -ne 0 ]]; then
  exit "${train_exit}"
fi
if [[ ! -s "${work_dir}/epoch_1.pth" || -e "${work_dir}/epoch_2.pth" ]]; then
  rtk echo 'SS seed-42 smoke checkpoint boundary was not respected.' >&2
  exit 6
fi
rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${work_dir}/epoch_1.pth" --expected-epoch 1 --expected-iter 2 \
  --config-token OrbdetV02Detector --config-token randomness
rtk touch "${work_dir}/COMPLETE"
rtk echo 'SS seed-42 smoke completed through epoch 1.'
