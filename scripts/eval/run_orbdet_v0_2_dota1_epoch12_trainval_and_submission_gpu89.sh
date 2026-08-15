#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
checkpoint=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_12.pth
trainval_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py"
submission_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py"
trainval_work_dir=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_epoch12_trainval_raw20995_gpu89_20260815
submission_work_dir=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_epoch12_test_submission_gpu89_20260815
submission_prefix=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_epoch12_test_submission_gpu89_20260815/dota_v1_task1_epoch12
submission_zip="${submission_prefix}/dota_v1_task1_epoch12.zip"
status_dir=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_epoch12_gpu89_20260815

rtk mkdir -p "${status_dir}"

on_error() {
  exit_code=$?
  rtk touch "${status_dir}/FAILED"
  rtk echo "Orbdet-v0.2 DOTA-v1 epoch-12 evaluation failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
trap on_error ERR

if [[ ! -s "${checkpoint}" ]]; then
  rtk echo "Checkpoint is missing or empty: ${checkpoint}" >&2
  exit 2
fi

if rtk pgrep -af \
  '[p]ython.*tools/test.py.*orbdet_v0_2_r50_dota1_epoch12' \
  >/dev/null; then
  rtk echo 'An Orbdet-v0.2 DOTA-v1 epoch-12 evaluation is already running.' >&2
  exit 3
fi

gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid \
  --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 8/9 are not idle; refusing to start DOTA-v1 evaluation.' >&2
  exit 4
fi

if [[ -e "${status_dir}/COMPLETE" || \
      -e "${status_dir}/TRAINVAL_COMPLETE" || \
      -e "${status_dir}/SUBMISSION_COMPLETE" || \
      -e "${submission_prefix}" ]]; then
  rtk echo 'Existing evaluation outputs found; refusing to overwrite them.' >&2
  exit 5
fi

rtk touch "${status_dir}/RUNNING"

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${repo_root}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29647 \
  "${repo_root}/tools/test.py" \
  "${trainval_config}" \
  "${checkpoint}" \
  --launcher=pytorch \
  --work-dir="${trainval_work_dir}"

rtk touch "${status_dir}/TRAINVAL_COMPLETE"

rtk env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${repo_root}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=8,9 \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29648 \
  "${repo_root}/tools/test.py" \
  "${submission_config}" \
  "${checkpoint}" \
  --launcher=pytorch \
  --work-dir="${submission_work_dir}"

rtk test -s "${submission_zip}"
rtk env PYTHONNOUSERSITE=1 \
  /data/zcy/anaconda3/envs/orbdet/bin/python -c \
  'import sys, zipfile; expected={"Task1_"+name+".txt" for name in ("plane", "baseball-diamond", "bridge", "ground-track-field", "small-vehicle", "large-vehicle", "ship", "tennis-court", "basketball-court", "storage-tank", "soccer-ball-field", "roundabout", "harbor", "swimming-pool", "helicopter")}; actual=set(zipfile.ZipFile(sys.argv[1]).namelist()); assert actual == expected, (actual, expected)' \
  "${submission_zip}"

rtk touch "${status_dir}/SUBMISSION_COMPLETE"
rtk touch "${status_dir}/COMPLETE"
rtk echo 'Orbdet-v0.2 DOTA-v1 trainval evaluation and test submission completed.'
