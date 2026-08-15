#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator="${repo_root}/tools/analysis_tools/validate_checkpoint_contract.py"
stage_root=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816
checkpoint="${stage_root}/epoch_3.pth"
stage_complete="${stage_root}/COMPLETE"
eval_root=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816
status_dir="${eval_root}/status"

trainval_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_trainval_eval.py"
ss_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ss_test_submission.py"
ms_config="${repo_root}/configs/orbdet/orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ms_test_submission.py"
trainval_work_dir="${eval_root}/trainval"
ss_work_dir="${eval_root}/ss_submission"
ms_work_dir="${eval_root}/ms_submission"
ss_prefix="${ss_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_ss_task1"
ms_prefix="${ms_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1"
ss_zip="${ss_prefix}/orbdet_v0_2_msrr_stage1_epoch3_ss_task1.zip"
ms_zip="${ms_prefix}/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1.zip"

rtk mkdir -p "${status_dir}"

on_error() {
  exit_code=$?
  rtk touch "${status_dir}/FAILED"
  rtk echo "Orbdet-v0.2 stage-3 post-evaluation failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}
on_signal() {
  trap - ERR INT TERM
  rtk touch "${status_dir}/INTERRUPTED"
  rtk echo 'Orbdet-v0.2 stage-3 post-evaluation reached the allocation deadline.' >&2
  exit 130
}
trap on_error ERR
trap on_signal INT TERM

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
  rtk sha256sum "${zip_path}"
}

if [[ ! -e "${stage_complete}" || ! -s "${checkpoint}" ]]; then
  rtk echo 'Stage-1 epoch 3 is not complete; refusing post-evaluation.' >&2
  exit 2
fi
if rtk pgrep -af \
  '[p]ython.*tools/test.py.*orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3' \
  >/dev/null; then
  rtk echo 'The Orbdet-v0.2 stage-3 post-evaluation is already running.' >&2
  exit 3
fi
gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid \
  --format=csv,noheader,nounits)"
if [[ -n "${gpu_processes}" ]]; then
  rtk echo 'GPU 4/5/6/7 are not idle; refusing post-evaluation.' >&2
  exit 4
fi
if [[ -e "${status_dir}/RUNNING" || -e "${status_dir}/COMPLETE" || \
      -e "${status_dir}/TRAINVAL_COMPLETE" || \
      -e "${status_dir}/SS_SUBMISSION_COMPLETE" || \
      -e "${status_dir}/MS_SUBMISSION_COMPLETE" || \
      -e "${ss_prefix}" || -e "${ms_prefix}" ]]; then
  rtk echo "Existing stage-3 evaluation outputs found in ${eval_root}." >&2
  exit 5
fi

rtk touch "${status_dir}/RUNNING"

rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
  "${checkpoint}" --expected-epoch 3 --expected-iter 51246 \
  --expected-state-tensors 371 --config-token OrbdetV02Detector \
  --config-token trainval_ms_full | \
  rtk tee "${status_dir}/checkpoint_contract.json"
rtk touch "${status_dir}/CHECKPOINT_VALIDATED"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29665 "${repo_root}/tools/test.py" "${trainval_config}" \
  "${checkpoint}" --launcher=pytorch --work-dir="${trainval_work_dir}"
rtk touch "${status_dir}/TRAINVAL_COMPLETE"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29666 "${repo_root}/tools/test.py" "${ss_config}" \
  "${checkpoint}" --launcher=pytorch --work-dir="${ss_work_dir}"
validate_zip "${ss_zip}"
rtk touch "${status_dir}/SS_SUBMISSION_COMPLETE"

rtk env PYTHONNOUSERSITE=1 PYTHONPATH="${repo_root}" \
  MPLCONFIGDIR=/tmp/zcy-codex/mplconfig OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  "${python_bin}" -m torch.distributed.launch --nproc_per_node=4 \
  --master_port=29667 "${repo_root}/tools/test.py" "${ms_config}" \
  "${checkpoint}" --launcher=pytorch --work-dir="${ms_work_dir}"
validate_zip "${ms_zip}"
rtk touch "${status_dir}/MS_SUBMISSION_COMPLETE"
rtk touch "${status_dir}/COMPLETE"
rtk echo 'Orbdet-v0.2 stage-3 trainval, SS, and MS+RR post-evaluation completed.'
