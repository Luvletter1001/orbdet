#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator="${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py"
deadline_local='2026-08-18 08:20:00 +0800'
deadline_epoch="$(rtk date -d "${deadline_local}" +%s)"
minimum_eval_seconds="${MINIMUM_EVAL_SECONDS:-1500}"
stage_root=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818
checkpoint=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818/epoch_12.pth
eval_root=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_ss_gpu89_seed42_epoch12_20260818
status_dir="${eval_root}/status"
lock_path=/data1/zcy/Orbdet/work_dirs/.gpu_8_9.launch_lock
launch_lock_held=0

trainval_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py"
ss_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py"
trainval_work_dir="${eval_root}/trainval"
ss_work_dir="${eval_root}/ss_submission"
ss_prefix="${ss_work_dir}/orbdet_v0_2_ss_seed42_epoch12_task1"
ss_zip="${ss_prefix}/orbdet_v0_2_ss_seed42_epoch12_task1.zip"

rtk mkdir -p "${status_dir}"

release_launch_lock() {
  if (( launch_lock_held == 1 )); then
    if ! rtk rmdir "${lock_path}"; then
      rtk echo "Could not remove empty launch lock: ${lock_path}" >&2
    fi
    launch_lock_held=0
  fi
}

on_error() {
  exit_code=$?
  rtk touch "${status_dir}/FAILED"
  rtk echo "Epoch-12 SS post-evaluation failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  rtk touch "${status_dir}/INTERRUPTED"
  rtk echo 'Epoch-12 SS post-evaluation reached its absolute deadline.' >&2
  exit 130
}
trap on_error ERR
trap on_signal INT TERM
trap release_launch_lock EXIT

skip_if_insufficient_window() {
  local phase=$1
  local now_epoch remaining_seconds
  now_epoch="$(rtk date +%s)"
  remaining_seconds=$(( deadline_epoch - now_epoch ))
  if (( remaining_seconds < minimum_eval_seconds )); then
    rtk touch "${status_dir}/POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW"
    rtk touch "${status_dir}/${phase}_SKIPPED_INSUFFICIENT_WINDOW"
    rtk echo "Only ${remaining_seconds}s remain before ${deadline_local}; skipping ${phase}."
    exit 0
  fi
}

run_bounded() {
  local now_epoch remaining_seconds exit_code
  now_epoch="$(rtk date +%s)"
  remaining_seconds=$(( deadline_epoch - now_epoch ))
  if (( remaining_seconds <= 0 )); then
    rtk touch "${status_dir}/POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW"
    rtk echo 'POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW: deadline already reached.' >&2
    exit 0
  fi
  set +e
  rtk timeout --signal=INT --kill-after=2m "${remaining_seconds}s" "$@"
  exit_code=$?
  set -e
  if [[ ${exit_code} -eq 124 || ${exit_code} -eq 130 || ${exit_code} -eq 137 ]]; then
    on_signal
  fi
  return "${exit_code}"
}

validate_zip() {
  local zip_path=$1
  if [[ ! -s "${zip_path}" ]]; then
    rtk echo "Submission ZIP is missing or empty: ${zip_path}" >&2
    return 20
  fi
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" -c \
    'import sys, zipfile; classes=("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter"); expected={"Task1_" + name + ".txt" for name in classes}; archive=zipfile.ZipFile(sys.argv[1]); names=archive.namelist(); actual=set(names); assert len(names) == 15 and actual == expected, (names, expected); assert archive.testzip() is None; print("validated", sys.argv[1], len(names), "files")' \
    "${zip_path}"
  rtk unzip -t "${zip_path}"
}

if ! rtk mkdir "${lock_path}"; then
  rtk echo "GPU group 8,9 launch lock is held: ${lock_path}" >&2
  exit 3
fi
launch_lock_held=1

existing_status_marker="$(rtk find "${status_dir}" -mindepth 1 -maxdepth 1 -print -quit)"
if [[ -n "${existing_status_marker}" || -e "${trainval_work_dir}" || \
      -e "${ss_work_dir}" || -e "${ss_prefix}" ]]; then
  rtk echo "Existing post-evaluation output found in ${eval_root}." >&2
  exit 5
fi

if [[ ! -e "${stage_root}/COMPLETE" || ! -s "${checkpoint}" ]]; then
  rtk echo 'Epoch-12 training is not COMPLETE; refusing post-evaluation.' >&2
  exit 2
fi
gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing post-evaluation.' >&2
  exit 4
fi
skip_if_insufficient_window PRECHECK

rtk touch "${status_dir}/RUNNING"
run_bounded rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${checkpoint}" --expected-epoch 12 --expected-iter 38280 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token randomness > "${status_dir}/checkpoint_contract.json"
rtk touch "${status_dir}/CHECKPOINT_VALIDATED"

skip_if_insufficient_window TRAINVAL
run_bounded rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "${python_bin}" -m torch.distributed.launch --nproc_per_node=2 --master_port=29677 "${repo_root}/tools/test.py" "${trainval_config}" "${checkpoint}" --launcher=pytorch --cfg-options "work_dir=${trainval_work_dir}"
rtk touch "${status_dir}/TRAINVAL_COMPLETE"

skip_if_insufficient_window SS_SUBMISSION
run_bounded rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 "${python_bin}" -m torch.distributed.launch --nproc_per_node=2 --master_port=29678 "${repo_root}/tools/test.py" "${ss_config}" "${checkpoint}" --launcher=pytorch --cfg-options "work_dir=${ss_work_dir}" "test_evaluator.outfile_prefix=${ss_prefix}"
validate_zip "${ss_zip}"
rtk touch "${status_dir}/SS_SUBMISSION_COMPLETE"
rtk touch "${status_dir}/COMPLETE"
rtk echo 'Epoch-12 SS trainval and submission post-evaluation completed.'
