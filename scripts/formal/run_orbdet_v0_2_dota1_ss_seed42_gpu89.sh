#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator="${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py"
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py"
work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818
deadline='2026-08-18 08:20:00 +0800'
launch_lock=/data1/zcy/Orbdet/work_dirs/.gpu_8_9.launch_lock
launch_lock_held=0

release_launch_lock() {
  if (( launch_lock_held == 1 )); then
    if ! rtk rmdir "${launch_lock}"; then
      rtk echo "Could not remove empty launch lock: ${launch_lock}" >&2
    fi
    launch_lock_held=0
  fi
}
mark_failed() {
  rtk mkdir -p "${work_dir}"
  if [[ -e "${work_dir}/RUNNING" ]]; then
    rtk mv "${work_dir}/RUNNING" "${work_dir}/FAILED"
  else
    rtk touch "${work_dir}/FAILED"
  fi
}
mark_interrupted() {
  rtk mkdir -p "${work_dir}"
  if [[ -e "${work_dir}/RUNNING" ]]; then
    rtk mv "${work_dir}/RUNNING" "${work_dir}/INTERRUPTED"
  else
    rtk touch "${work_dir}/INTERRUPTED"
  fi
}

on_error() {
  exit_code=$?
  trap - ERR INT TERM
  mark_failed
  rtk echo "SS seed-42 formal run failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  mark_interrupted
  rtk echo 'SS seed-42 formal run interrupted.' >&2
  exit 130
}
trap on_error ERR
trap on_signal INT TERM
trap release_launch_lock EXIT

rtk mkdir -p "$(rtk dirname "${work_dir}")"
if ! rtk mkdir "${launch_lock}"; then
  rtk echo "Launch lock already exists: ${launch_lock}" >&2
  exit 7
fi
launch_lock_held=1

deadline_epoch="$(rtk date -d "${deadline}" +%s)"
if (( deadline_epoch - $(rtk date +%s) <= 0 )); then
  rtk echo "Deadline has passed: ${deadline}" >&2
  exit 2
fi
if rtk pgrep -af '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py' >/dev/null; then
  rtk echo 'The SS seed-42 formal run is already running.' >&2
  exit 3
fi
gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing to start SS seed-42 formal run.' >&2
  exit 4
fi
shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || [[ -e "${work_dir}/RUNNING" || -e "${work_dir}/COMPLETE" || -e "${work_dir}/FAILED" || -e "${work_dir}/INTERRUPTED" ]]; then
  rtk echo "Refusing to overwrite existing outputs or markers in ${work_dir}" >&2
  exit 5
fi

rtk mkdir -p "${work_dir}"
remaining_seconds=$(( deadline_epoch - $(rtk date +%s) ))
if (( remaining_seconds <= 0 )); then
  rtk echo "Deadline has passed: ${deadline}" >&2
  exit 2
fi
rtk touch "${work_dir}/RUNNING"
set +e
rtk timeout --signal=TERM --kill-after=30s "${remaining_seconds}s" \
  rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=2 \
  --master_port=29673 "${repo_root}/tools/train.py" "${config}" \
  --launcher=pytorch --work-dir="${work_dir}"
train_exit=$?
set -e
if [[ ${train_exit} -eq 124 ]]; then
  trap - ERR INT TERM
  mark_interrupted
  rtk echo 'SS seed-42 formal run reached the deadline.' >&2
  exit "${train_exit}"
fi
if [[ ${train_exit} -eq 137 || ${train_exit} -eq 143 ]]; then
  trap - ERR INT TERM
  mark_interrupted
  rtk echo "SS seed-42 formal run was interrupted with exit ${train_exit}." >&2
  exit "${train_exit}"
fi
if [[ ${train_exit} -ne 0 ]]; then
  trap - ERR INT TERM
  mark_failed
  rtk echo "SS seed-42 formal training failed with exit ${train_exit}." >&2
  exit "${train_exit}"
fi
if [[ ! -s "${work_dir}/epoch_12.pth" || -e "${work_dir}/epoch_13.pth" ]]; then
  trap - ERR INT TERM
  mark_failed
  rtk echo 'SS seed-42 formal run checkpoint boundary was not respected.' >&2
  exit 6
fi
rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${work_dir}/epoch_12.pth" --expected-epoch 12 --expected-iter 38280 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token randomness
rtk mv "${work_dir}/RUNNING" "${work_dir}/COMPLETE"
rtk echo 'SS seed-42 formal run completed through epoch 12.'
