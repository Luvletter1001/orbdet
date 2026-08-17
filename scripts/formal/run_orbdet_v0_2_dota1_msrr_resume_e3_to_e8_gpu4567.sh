#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator="${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py"
config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py"
work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818
source_checkpoint=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816/epoch_3.pth
source_sha256=7847a8991984a87ae1545a1a04f26490bcfe16213603615c24a19a299740996b
deadline='2026-08-18 08:20:00 +0800'
launch_lock=/data1/zcy/Orbdet/work_dirs/.gpu_4_5_6_7.launch_lock
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
  rtk echo "MS+RR formal continuation failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  mark_interrupted
  rtk echo 'MS+RR formal continuation interrupted.' >&2
  exit 130
}
trap on_error ERR
trap on_signal INT TERM
trap release_launch_lock EXIT

rtk mkdir -p "$(rtk dirname "${work_dir}")"
if ! rtk mkdir "${launch_lock}"; then
  rtk echo "Launch lock already exists: ${launch_lock}" >&2
  exit 9
fi
launch_lock_held=1

deadline_epoch="$(rtk date -d "${deadline}" +%s)"
now_epoch="$(rtk date +%s)"
remaining_seconds=$(( deadline_epoch - now_epoch ))
if (( remaining_seconds <= 0 )); then
  rtk echo "Deadline has passed: ${deadline}" >&2
  exit 2
fi
if [[ ! -s "${source_checkpoint}" ]]; then
  rtk echo "Required source checkpoint is missing: ${source_checkpoint}" >&2
  exit 3
fi
if [[ "$(rtk sha256sum "${source_checkpoint}" | rtk awk '{print $1}')" != "${source_sha256}" ]]; then
  rtk echo 'Source checkpoint SHA-256 does not match the approved stage-1 artifact.' >&2
  exit 4
fi
rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${source_checkpoint}" --expected-epoch 3 --expected-iter 51246 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token trainval_ms_full
if rtk pgrep -af '[p]ython.*tools/train.py.*orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py' >/dev/null; then
  rtk echo 'The MS+RR formal continuation is already running.' >&2
  exit 5
fi
gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 4/5/6/7 are not idle; refusing to start MS+RR formal continuation.' >&2
  exit 6
fi
shopt -s nullglob
existing_checkpoints=("${work_dir}"/*.pth)
if (( ${#existing_checkpoints[@]} > 0 )) || [[ -e "${work_dir}/RUNNING" || -e "${work_dir}/COMPLETE" || -e "${work_dir}/FAILED" || -e "${work_dir}/INTERRUPTED" ]]; then
  rtk echo "Refusing to overwrite existing outputs or markers in ${work_dir}" >&2
  exit 7
fi

rtk mkdir -p "${work_dir}"
rtk touch "${work_dir}/RUNNING"
set +e
rtk timeout --foreground --signal=TERM --kill-after=30s "${remaining_seconds}s" \
  rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29672 "${repo_root}/tools/train.py" "${config}" \
  --launcher=pytorch --work-dir="${work_dir}" --resume="${source_checkpoint}"
train_exit=$?
set -e
if [[ ${train_exit} -eq 124 ]]; then
  trap - ERR INT TERM
  mark_interrupted
  rtk echo 'MS+RR formal continuation reached the deadline.' >&2
  exit "${train_exit}"
fi
if [[ ${train_exit} -eq 137 || ${train_exit} -eq 143 ]]; then
  trap - ERR INT TERM
  mark_interrupted
  rtk echo "MS+RR formal continuation was interrupted with exit ${train_exit}." >&2
  exit "${train_exit}"
fi
if [[ ${train_exit} -ne 0 ]]; then
  trap - ERR INT TERM
  mark_failed
  rtk echo "MS+RR formal continuation training failed with exit ${train_exit}." >&2
  exit "${train_exit}"
fi
if [[ ! -s "${work_dir}/epoch_8.pth" || -e "${work_dir}/epoch_9.pth" ]]; then
  trap - ERR INT TERM
  mark_failed
  rtk echo 'MS+RR formal continuation checkpoint boundary was not respected.' >&2
  exit 8
fi
rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${work_dir}/epoch_8.pth" --expected-epoch 8 --expected-iter 136656 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token trainval_ms_full
rtk mv "${work_dir}/RUNNING" "${work_dir}/COMPLETE"
rtk echo 'MS+RR formal continuation completed through epoch 8.'
