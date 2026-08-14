#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
upstream_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815
status_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_godc_after_v02_gpu89_20260815
smoke_work_dir=/data1/zcy/Orbdet/work_dirs/smoke/orbdet_godc_c2_hrsc_clean_gpu89_20260815

on_error() {
  exit_code=$?
  rtk touch "${status_dir}/FAILED"
  rtk echo "Orbdet-GODC controller failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

rtk mkdir -p "${status_dir}"
rtk touch "${status_dir}/RUNNING"

while [[ ! -e "/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/COMPLETE" ]]; do
  if [[ -e "/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_hrsc_multiseed_gpu89_20260815/FAILED" ]]; then
    rtk echo 'Upstream v0.2 multi-seed queue failed; stopping GODC chain.' >&2
    exit 5
  fi
  rtk sleep 30
done

rtk bash "${repo_root}/scripts/smoke/run_orbdet_godc_c2_hrsc_gpu89.sh"
if [[ ! -f "${smoke_work_dir}/epoch_1.pth" ]]; then
  rtk echo 'GODC smoke did not produce epoch_1.pth.' >&2
  exit 6
fi
rtk touch "${status_dir}/SMOKE_COMPLETE"
rtk bash "${repo_root}/scripts/formal/run_orbdet_godc_c2_hrsc_gpu89_seed3407.sh"
rtk touch "${status_dir}/COMPLETE"
rtk echo 'Orbdet-GODC HRSC smoke and formal training completed.'
