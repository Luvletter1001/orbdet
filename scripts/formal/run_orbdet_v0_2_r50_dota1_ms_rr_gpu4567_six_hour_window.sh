#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
window_root=/data1/zcy/Orbdet/work_dirs/controllers/orbdet_v0_2_dota1_ms_rr_gpu4567_six_hour_20260816
audit_launcher="${repo_root}/scripts/eval/run_h2rbox_v2_dota1_official_checkpoint_audit_gpu4567.sh"
smoke_launcher="${repo_root}/scripts/smoke/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_smoke.sh"
stage_launcher="${repo_root}/scripts/formal/run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_3e.sh"

rtk mkdir -p "${window_root}"
if [[ -e "${window_root}/RUNNING" || -e "${window_root}/COMPLETE" ]]; then
  rtk echo "Existing six-hour controller marker found in ${window_root}" >&2
  exit 2
fi
rtk touch "${window_root}/RUNNING"

set +e
rtk timeout --signal=INT --kill-after=5m 345m \
  rtk bash -c "cd '${repo_root}' && rtk bash '${audit_launcher}' && rtk bash '${smoke_launcher}' && rtk bash '${stage_launcher}'"
exit_code=$?
set -e

if [[ ${exit_code} -eq 124 || ${exit_code} -eq 130 || ${exit_code} -eq 137 ]]; then
  rtk touch "${window_root}/TIME_LIMIT_REACHED"
  rtk echo "GPU4567 six-hour controller reached its time limit (${exit_code})." >&2
  exit "${exit_code}"
fi
if [[ ${exit_code} -ne 0 ]]; then
  rtk touch "${window_root}/FAILED"
  rtk echo "GPU4567 six-hour controller failed with exit ${exit_code}." >&2
  exit "${exit_code}"
fi

rtk touch "${window_root}/COMPLETE"
rtk echo 'GPU4567 six-hour audit, smoke, and stage1 chain completed.'
