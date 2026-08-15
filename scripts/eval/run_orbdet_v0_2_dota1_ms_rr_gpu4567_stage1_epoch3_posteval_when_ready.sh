#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
stage_root=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816
stage_complete="${stage_root}/COMPLETE"
window_root=/data1/zcy/Orbdet/work_dirs/controllers/orbdet_v0_2_dota1_ms_rr_gpu4567_six_hour_20260816
watcher_root=/data1/zcy/Orbdet/work_dirs/controllers/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_posteval_watcher_20260816
eval_complete=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/status/COMPLETE
launcher="${repo_root}/scripts/eval/run_orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_posteval.sh"
deadline_local='2026-08-16 09:50:00 +0800'
deadline_epoch="$(rtk date -d "${deadline_local}" +%s)"
minimum_eval_seconds="${MINIMUM_EVAL_SECONDS:-1800}"
poll_seconds="${POLL_SECONDS:-30}"

rtk mkdir -p "${watcher_root}"
if [[ -e "${watcher_root}/RUNNING" || -e "${watcher_root}/COMPLETE" || \
      -e "${watcher_root}/TIME_LIMIT_REACHED" ]]; then
  rtk echo "Existing post-evaluation watcher marker found in ${watcher_root}." >&2
  exit 2
fi
rtk touch "${watcher_root}/RUNNING"

while [[ ! -e "${stage_complete}" ]]; do
  if [[ -e "${stage_root}/FAILED" || -e "${stage_root}/INTERRUPTED" || \
        -e "${window_root}/FAILED" || \
        -e "${window_root}/TIME_LIMIT_REACHED" ]]; then
    rtk touch "${watcher_root}/BLOCKED_UPSTREAM"
    rtk echo 'Stage-1 stopped before epoch 3; post-evaluation will not run.' >&2
    exit 10
  fi
  now_epoch="$(rtk date +%s)"
  if (( now_epoch >= deadline_epoch )); then
    rtk touch "${watcher_root}/TIME_LIMIT_REACHED"
    rtk echo 'Post-evaluation watcher reached the six-hour safety deadline.' >&2
    exit 124
  fi
  rtk sleep "${poll_seconds}"
done

# COMPLETE is written only after torchrun returns, but allow a short grace
# period for driver contexts to disappear. Never terminate a process here.
gpu_wait_deadline=$(( $(rtk date +%s) + 120 ))
gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid \
  --format=csv,noheader,nounits)"
while [[ -n "${gpu_processes}" ]]; do
  now_epoch="$(rtk date +%s)"
  if (( now_epoch >= gpu_wait_deadline )); then
    rtk touch "${watcher_root}/GPU_BUSY"
    rtk echo 'GPU 4/5/6/7 stayed busy; no process was modified.' >&2
    exit 11
  fi
  rtk sleep 5
  gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
done

now_epoch="$(rtk date +%s)"
remaining_seconds=$(( deadline_epoch - now_epoch ))
if (( remaining_seconds < minimum_eval_seconds )); then
  rtk touch "${watcher_root}/SKIPPED_INSUFFICIENT_WINDOW"
  rtk echo "Only ${remaining_seconds}s remain; preserving the allocation boundary." >&2
  exit 0
fi

set +e
rtk timeout --signal=INT --kill-after=2m "${remaining_seconds}s" \
  rtk bash "${launcher}"
exit_code=$?
set -e

if [[ ${exit_code} -eq 124 || ${exit_code} -eq 130 || \
      ${exit_code} -eq 137 ]]; then
  rtk touch "${watcher_root}/TIME_LIMIT_REACHED"
  rtk echo 'Stage-3 post-evaluation was stopped at the safety deadline.' >&2
  exit "${exit_code}"
fi
if [[ ${exit_code} -ne 0 || ! -e "${eval_complete}" ]]; then
  rtk touch "${watcher_root}/FAILED"
  rtk echo "Stage-3 post-evaluation failed with exit ${exit_code}." >&2
  exit "${exit_code}"
fi

rtk touch "${watcher_root}/COMPLETE"
rtk echo 'Stage-3 post-evaluation completed inside the six-hour window.'
