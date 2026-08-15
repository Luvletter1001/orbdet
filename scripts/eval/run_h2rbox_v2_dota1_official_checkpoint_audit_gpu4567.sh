#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
audit_root=/data1/zcy/Orbdet/work_dirs/audit/h2rbox_v2_dota1_official_gpu4567_20260816
download_root=/data1/zcy/Orbdet/work_dirs/audit/h2rbox_v2_dota1_official_20260815/downloads

ss_checkpoint="${download_root}/h2rbox_v2-le90_r50_fpn-1x_dota-fa5ad1d2.pth"
msrr_checkpoint="${download_root}/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth"
ss_sha256=fa5ad1d2d6d030a477fe6f9a405863a76f55d463cff31b7e01c8366527888fde
msrr_sha256=5e0e53e12e0d8b07f79922b6cef9b56c142458483d2c1e2f27348f7e72e9d677

ss_trainval_config="${repo_root}/configs/orbdet/h2rbox_v2_r50_dota1_official_ss_trainval_audit_gpu89.py"
ss_test_config="${repo_root}/configs/orbdet/h2rbox_v2_r50_dota1_official_ss_test_submission_gpu89.py"
ms_test_config="${repo_root}/configs/orbdet/h2rbox_v2_r50_dota1_official_ms_test_submission_gpu89.py"

ss_trainval_work_dir="${audit_root}/ss_trainval"
ss_submission_work_dir="${audit_root}/ss_submission"
ms_submission_work_dir="${audit_root}/ms_submission"
ss_zip="${ss_submission_work_dir}/h2rbox_v2_official_ss_task1/h2rbox_v2_official_ss_task1.zip"
ms_zip="${ms_submission_work_dir}/h2rbox_v2_official_msrr_task1/h2rbox_v2_official_msrr_task1.zip"

rtk mkdir -p "${audit_root}"

on_error() {
  exit_code=$?
  rtk touch "${audit_root}/FAILED"
  rtk echo "Official H2RBox-v2 GPU4567 audit failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

validate_zip() {
  zip_path=$1
  if [[ ! -s "${zip_path}" ]]; then
    rtk echo "Submission ZIP is missing or empty: ${zip_path}" >&2
    return 20
  fi
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" -c \
    'import sys, zipfile; classes=("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter"); expected={"Task1_"+name+".txt" for name in classes}; archive=zipfile.ZipFile(sys.argv[1]); actual=set(archive.namelist()); assert actual == expected, (actual, expected); assert archive.testzip() is None; print("validated", sys.argv[1], len(actual), "files")' \
    "${zip_path}"
  rtk unzip -t "${zip_path}"
}

if [[ ! -s "${ss_checkpoint}" || ! -s "${msrr_checkpoint}" ]]; then
  rtk echo 'Official checkpoint is missing or empty.' >&2
  exit 2
fi
if [[ "$(rtk sha256sum "${ss_checkpoint}" | rtk awk '{print $1}')" != "${ss_sha256}" ]]; then
  rtk echo 'Official SS checkpoint SHA256 mismatch.' >&2
  exit 3
fi
if [[ "$(rtk sha256sum "${msrr_checkpoint}" | rtk awk '{print $1}')" != "${msrr_sha256}" ]]; then
  rtk echo 'Official MS+RR checkpoint SHA256 mismatch.' >&2
  exit 4
fi
if rtk pgrep -af '[p]ython.*tools/test.py.*h2rbox_v2_r50_dota1_official' >/dev/null; then
  rtk echo 'An official H2RBox-v2 DOTA-v1 audit is already running.' >&2
  exit 5
fi

gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 4/5/6/7 are not idle; refusing to start official audit.' >&2
  exit 6
fi
if [[ -e "${audit_root}/COMPLETE" || \
      -e "${audit_root}/SS_TRAINVAL_COMPLETE" || \
      -e "${audit_root}/SS_SUBMISSION_COMPLETE" || \
      -e "${audit_root}/MS_SUBMISSION_COMPLETE" || \
      -e "${ss_submission_work_dir}/h2rbox_v2_official_ss_task1" || \
      -e "${ms_submission_work_dir}/h2rbox_v2_official_msrr_task1" ]]; then
  rtk echo 'Existing GPU4567 audit outputs found; refusing to overwrite them.' >&2
  exit 7
fi

rtk touch "${audit_root}/RUNNING"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29660 "${repo_root}/tools/test.py" "${ss_trainval_config}" \
  "${ss_checkpoint}" --launcher=pytorch --work-dir="${ss_trainval_work_dir}"
rtk touch "${audit_root}/SS_TRAINVAL_COMPLETE"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29661 "${repo_root}/tools/test.py" "${ss_test_config}" \
  "${ss_checkpoint}" --launcher=pytorch --work-dir="${ss_submission_work_dir}"
validate_zip "${ss_zip}"
rtk touch "${audit_root}/SS_SUBMISSION_COMPLETE"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29662 "${repo_root}/tools/test.py" "${ms_test_config}" \
  "${msrr_checkpoint}" --launcher=pytorch --work-dir="${ms_submission_work_dir}"
validate_zip "${ms_zip}"
rtk touch "${audit_root}/MS_SUBMISSION_COMPLETE"
rtk touch "${audit_root}/COMPLETE"
rtk echo 'Official H2RBox-v2 DOTA-v1 GPU4567 audit completed.'
