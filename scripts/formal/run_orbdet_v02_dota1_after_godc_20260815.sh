#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
status_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v02_dota1_after_godc_gpu89_20260815
smoke_work_dir=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_2_dota1_1x_gpu89_20260815

on_error() {
  exit_code=$?
  rtk touch "${status_dir}/FAILED"
  rtk echo "Orbdet-v0.2 DOTA-v1 controller failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

rtk mkdir -p "${status_dir}"
rtk touch "${status_dir}/RUNNING"

while [[ ! -e "/data1/zcy/Orbdet/work_dirs/formal/orbdet_godc_after_v02_gpu89_20260815/COMPLETE" ]]; do
  if [[ -e "/data1/zcy/Orbdet/work_dirs/formal/orbdet_godc_after_v02_gpu89_20260815/FAILED" ]]; then
    rtk echo 'Upstream GODC chain failed; stopping DOTA-v1 chain.' >&2
    exit 5
  fi
  rtk sleep 30
done

rtk bash "${repo_root}/scripts/smoke/run_orbdet_v0_2_dota1_gpu89.sh"
if [[ ! -f "${smoke_work_dir}/epoch_1.pth" ]]; then
  rtk echo 'DOTA-v1 smoke did not produce epoch_1.pth.' >&2
  exit 6
fi
rtk touch "${status_dir}/SMOKE_COMPLETE"
rtk bash "${repo_root}/scripts/formal/run_orbdet_v0_2_dota1_1x_gpu89_seed3407.sh"
rtk touch "${status_dir}/COMPLETE"
rtk echo 'Orbdet-v0.2 DOTA-v1 smoke and formal training completed.'
