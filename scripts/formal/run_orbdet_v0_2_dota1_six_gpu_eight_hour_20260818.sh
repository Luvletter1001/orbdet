#!/usr/bin/env bash
set -euo pipefail

repo_root="$(rtk realpath "$(rtk dirname "${BASH_SOURCE[0]}")/../..")"
python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
deadline='2026-08-18 08:20:00 +0800'
controller_root=/data1/zcy/Orbdet/work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818
primary_work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818
secondary_work_dir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818

ms_smoke_launcher="${repo_root}/scripts/smoke/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh"
ms_formal_launcher="${repo_root}/scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh"
ss_smoke_launcher="${repo_root}/scripts/smoke/run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh"
ss_formal_launcher="${repo_root}/scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh"
ss_posteval_launcher="${repo_root}/scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh"
ms_posteval_launcher="${repo_root}/scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh"

deadline_epoch="$(rtk date -d "${deadline}" +%s)"
ms_pid=''
ms_pgid=''
ss_pid=''
ss_pgid=''
sync_pid=''
sync_pgid=''
ss_exit=0
ss_exit_captured=0
ss_stopped_for_priority=0
controller_active=0

at_deadline() {
  (( $(rtk date +%s) >= deadline_epoch ))
}

mark_status() {
  local status=$1
  if (( controller_active == 1 )) && [[ -e "${controller_root}/RUNNING" ]]; then
    rtk mv "${controller_root}/RUNNING" "${controller_root}/${status}"
    controller_active=0
  fi
}

child_is_running() {
  local target_pid=$1
  local job_pid
  local job_pids
  [[ -n "${target_pid}" ]] || return 1
  job_pids="$(jobs -pr)"
  while IFS= read -r job_pid; do
    if [[ "${job_pid}" == "${target_pid}" ]]; then
      return 0
    fi
  done <<<"${job_pids}"
  return 1
}

owned_session_is_verified() {
  local wrapper_pid=$1
  local pgid=$2
  local process_row
  local leader_pid
  local parent_pid
  local actual_pgid
  local session_id
  [[ "${wrapper_pid}" =~ ^[0-9]+$ && "${pgid}" =~ ^[0-9]+$ ]] || return 1
  child_is_running "${wrapper_pid}" || return 1
  process_row="$(rtk ps -o pid=,ppid=,pgid=,sid= -p "${pgid}")"
  read -r leader_pid parent_pid actual_pgid session_id <<<"${process_row}"
  [[ "${leader_pid}" == "${pgid}" &&
     "${parent_pid}" == "${wrapper_pid}" &&
     "${actual_pgid}" == "${pgid}" &&
     "${session_id}" == "${pgid}" ]]
}

owned_group_has_members() {
  local pgid=$1
  rtk ps -eo pgid= | rtk awk -v target="${pgid}" \
    '$1 == target { found = 1 } END { exit !found }'
}

signal_owned_group() {
  local wrapper_pid=$1
  local pgid=$2
  if ! owned_session_is_verified "${wrapper_pid}" "${pgid}"; then
    return 1
  fi
  rtk kill -TERM -- "-${pgid}" 2>/dev/null || true
}

wait_for_owned_group_shutdown() {
  local pgid=$1
  local sample
  for sample in {1..30}; do
    if ! owned_group_has_members "${pgid}"; then
      return 0
    fi
    rtk sleep 1
  done
  if owned_group_has_members "${pgid}"; then
    rtk kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
}

terminate_owned_children() {
  local ms_group_owned=0
  local ss_group_owned=0
  local sync_group_owned=0
  if signal_owned_group "${ms_pid}" "${ms_pgid}"; then
    ms_group_owned=1
  fi
  if signal_owned_group "${ss_pid}" "${ss_pgid}"; then
    ss_group_owned=1
  fi
  if signal_owned_group "${sync_pid}" "${sync_pgid}"; then
    sync_group_owned=1
  fi
  if (( ms_group_owned == 1 )); then
    wait_for_owned_group_shutdown "${ms_pgid}"
  fi
  if (( ss_group_owned == 1 )); then
    wait_for_owned_group_shutdown "${ss_pgid}"
  fi
  if (( sync_group_owned == 1 )); then
    wait_for_owned_group_shutdown "${sync_pgid}"
  fi
  if [[ -n "${ms_pid}" ]]; then
    wait "${ms_pid}" 2>/dev/null || true
    ms_pid=''
    ms_pgid=''
  fi
  if [[ -n "${ss_pid}" ]]; then
    wait "${ss_pid}" 2>/dev/null || true
    ss_pid=''
    ss_pgid=''
  fi
  if [[ -n "${sync_pid}" ]]; then
    wait "${sync_pid}" 2>/dev/null || true
    sync_pid=''
    sync_pgid=''
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR INT TERM
  terminate_owned_children
  if at_deadline; then
    rtk touch "${controller_root}/TIME_LIMIT_REACHED"
    mark_status INTERRUPTED
  else
    mark_status FAILED
  fi
  rtk echo "DOTA six-GPU controller failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}

on_signal() {
  trap - ERR INT TERM
  terminate_owned_children
  if at_deadline; then
    rtk touch "${controller_root}/TIME_LIMIT_REACHED"
  fi
  mark_status INTERRUPTED
  rtk echo 'DOTA six-GPU controller interrupted.' >&2
  exit 130
}

fail_controller() {
  local marker=$1
  local exit_code=$2
  local status=${3:-FAILED}
  trap - ERR INT TERM
  terminate_owned_children
  rtk touch "${controller_root}/${marker}"
  if at_deadline; then
    rtk touch "${controller_root}/TIME_LIMIT_REACHED"
    status=INTERRUPTED
  fi
  mark_status "${status}"
  exit "${exit_code}"
}

wait_for_idle() {
  local gpu_list=$1
  local gpu_processes
  local empty_samples=0
  local now_epoch
  local remaining_seconds

  while (( empty_samples < 2 )); do
    now_epoch="$(rtk date +%s)"
    if (( now_epoch >= deadline_epoch )); then
      rtk touch "${controller_root}/TIME_LIMIT_REACHED"
      return 124
    fi
    case "${gpu_list}" in
      4,5,6,7)
        if ! gpu_processes="$(rtk nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits)"; then
          rtk echo 'Could not query GPU 4/5/6/7 process state.' >&2
          return 3
        fi
        ;;
      8,9)
        if ! gpu_processes="$(rtk nvidia-smi -i 8,9 --query-compute-apps=pid --format=csv,noheader,nounits)"; then
          rtk echo 'Could not query GPU 8/9 process state.' >&2
          return 3
        fi
        ;;
      *)
        rtk echo "Unsupported GPU list: ${gpu_list}" >&2
        return 2
        ;;
    esac
    if [[ -z "${gpu_processes}" ]]; then
      empty_samples=$(( empty_samples + 1 ))
    else
      empty_samples=0
    fi
    if (( empty_samples < 2 )); then
      now_epoch="$(rtk date +%s)"
      remaining_seconds=$(( deadline_epoch - now_epoch ))
      if (( remaining_seconds <= 30 )); then
        rtk touch "${controller_root}/TIME_LIMIT_REACHED"
        return 124
      fi
      rtk sleep 30
    fi
  done
}

scan_for_fatal_training_output() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - \
    "${controller_root}" "${primary_work_dir}" <<'PY'
import pathlib
import re
import sys

roots = [pathlib.Path(value) for value in sys.argv[1:]]
patterns = (
    re.compile(r'Traceback \(most recent call last\)'),
    re.compile(r'RuntimeError:'),
    re.compile(r'CUDA out of memory', re.IGNORECASE),
    re.compile(r'NCCL.*(?:error|failure|abort)', re.IGNORECASE),
    re.compile(r'(?<![A-Za-z])NaN(?![A-Za-z])', re.IGNORECASE),
    # Case-sensitive and token-bounded so ordinary "INFO" is not an Inf hit.
    re.compile(r'(?<![A-Za-z])(?:Inf|-Inf|\+Inf)(?![A-Za-z])'),
)
for root in roots:
    if not root.exists():
        continue
    candidates = [root] if root.is_file() else root.rglob('*')
    for path in candidates:
        if not path.is_file() or path.suffix not in {'.log', '.json'}:
            continue
        try:
            content = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for pattern in patterns:
            if pattern.search(content):
                print(f'fatal training output in {path}: {pattern.pattern}',
                      file=sys.stderr)
                raise SystemExit(1)
PY
}

validate_first_200_steps() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${primary_work_dir}" <<'PY'
import json
import math
import pathlib
import sys

work_dir = pathlib.Path(sys.argv[1])
records = []
for path in work_dir.rglob('scalars.json') if work_dir.exists() else ():
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        continue
    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get('step'), int):
            records.append(record)

gate_step = 51446
try:
    gate_record = next(
        record for record in records if record['step'] == gate_step)
except StopIteration:
    raise SystemExit(10)
for name in ('loss', 'grad_norm', 'time'):
    value = gate_record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(f'gate scalar {name} is missing or non-numeric', file=sys.stderr)
        raise SystemExit(11)
    if not math.isfinite(float(value)):
        print(f'gate scalar {name} is non-finite: {value}', file=sys.stderr)
        raise SystemExit(12)
print(f'validated exact gate step {gate_step}')
PY
}

measure_primary_time_median() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${primary_work_dir}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

work_dir = pathlib.Path(sys.argv[1])
priority_time_threshold = 0.341
samples = []
for path in work_dir.rglob('scalars.json') if work_dir.exists() else ():
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        continue
    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        step = record.get('step') if isinstance(record, dict) else None
        value = record.get('time') if isinstance(record, dict) else None
        if (isinstance(step, int) and not isinstance(step, bool) and
                step >= 51446 and isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value)) and
                float(value) > 0.0):
            samples.append((step, float(value)))
samples.sort(key=lambda item: item[0])
recent_stable_samples = [value for _, value in samples[-20:]]
if len(recent_stable_samples) < 5:
    raise SystemExit(10)
median_time = statistics.median(recent_stable_samples)
print(f'{median_time:.9f}')
raise SystemExit(20 if median_time > priority_time_threshold else 0)
PY
}

latest_complete_ss_checkpoint() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${secondary_work_dir}" <<'PY'
import pathlib
import re
import sys
import zipfile

work_dir = pathlib.Path(sys.argv[1])
complete = []
for path in work_dir.glob('epoch_*.pth') if work_dir.exists() else ():
    match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
    if match is None or not path.is_file() or path.stat().st_size == 0:
        continue
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                continue
    except (OSError, zipfile.BadZipFile):
        continue
    complete.append((int(match.group(1)), path))
if not complete:
    raise SystemExit(10)
epoch, path = max(complete, key=lambda item: item[0])
print(f'{epoch}\t{path}')
PY
}

stop_secondary_for_primary_priority() {
  local boundary_checkpoint=$1
  if ! signal_owned_group "${ss_pid}" "${ss_pgid}"; then
    return 1
  fi
  wait_for_owned_group_shutdown "${ss_pgid}"
  set +e
  wait "${ss_pid}"
  ss_exit=$?
  set -e
  ss_pid=''
  ss_pgid=''
  ss_exit_captured=1
  ss_stopped_for_priority=1
  rtk printf '%s\n' "${boundary_checkpoint}" \
    >"${controller_root}/ss_priority_stop_checkpoint"
  rtk touch "${controller_root}/SS_STOPPED_FOR_PRIMARY_PRIORITY"
}

start_managed_job() {
  local launcher=$1
  local log_path=$2
  local label=$3
  local pid_variable=$4
  local pgid_variable=$5
  local ready_path="${controller_root}/${label}_process_group.pgid"
  local wrapper_pid
  local pgid=''
  local sample

  rtk setsid --wait rtk bash -c '
    set -euo pipefail
    ready_path=$1
    launcher=$2
    pgid="$(rtk ps -o pgid= -p "$$")"
    session_id="$(rtk ps -o sid= -p "$$")"
    pgid="${pgid//[[:space:]]/}"
    session_id="${session_id//[[:space:]]/}"
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${session_id}" != "${pgid}" ]]; then
      rtk echo "Could not establish a dedicated owned process group." >&2
      exit 70
    fi
    rtk printf "%s\n" "${pgid}" >"${ready_path}"
    exec rtk bash "${launcher}"
  ' owned-session "${ready_path}" "${launcher}" >"${log_path}" 2>&1 &
  wrapper_pid=$!
  printf -v "${pid_variable}" '%s' "${wrapper_pid}"

  for sample in {1..50}; do
    if [[ -s "${ready_path}" ]]; then
      pgid="$(rtk sed -n '1p' "${ready_path}")"
      break
    fi
    if ! child_is_running "${wrapper_pid}"; then
      break
    fi
    rtk sleep 0.1
  done
  printf -v "${pgid_variable}" '%s' "${pgid}"
  if ! owned_session_is_verified "${wrapper_pid}" "${pgid}"; then
    rtk echo "Could not verify owned process group for ${label}." >&2
    return 71
  fi
}

run_synchronously() {
  local launcher=$1
  local log_path=$2
  local label=$3
  local exit_code
  if ! start_managed_job "${launcher}" "${log_path}" "${label}" \
      sync_pid sync_pgid; then
    return 71
  fi
  set +e
  wait "${sync_pid}"
  exit_code=$?
  sync_pid=''
  sync_pgid=''
  set -e
  return "${exit_code}"
}

rtk mkdir -p "$(rtk dirname "${controller_root}")"
if [[ -e "${controller_root}" ]]; then
  rtk echo "Refusing to overwrite existing controller state in ${controller_root}" >&2
  exit 2
fi
rtk mkdir "${controller_root}"
rtk touch "${controller_root}/RUNNING"
controller_active=1
trap on_error ERR
trap on_signal INT TERM

set +e
wait_for_idle 4,5,6,7
idle_exit=$?
set -e
if (( idle_exit != 0 )); then
  if (( idle_exit == 124 )); then
    fail_controller GPU4567_IDLE_WAIT_INTERRUPTED 124 INTERRUPTED
  fi
  fail_controller GPU4567_IDLE_QUERY_FAILED "${idle_exit}"
fi
set +e
wait_for_idle 8,9
idle_exit=$?
set -e
if (( idle_exit != 0 )); then
  if (( idle_exit == 124 )); then
    fail_controller GPU89_IDLE_WAIT_INTERRUPTED 124 INTERRUPTED
  fi
  fail_controller GPU89_IDLE_QUERY_FAILED "${idle_exit}"
fi

if ! run_synchronously "${ms_smoke_launcher}" \
    "${controller_root}/ms_smoke.log" ms_smoke; then
  fail_controller MS_SMOKE_FAILED 20
fi

rtk date --iso-8601=seconds >"${controller_root}/ms_started_at"
if ! start_managed_job "${ms_formal_launcher}" \
    "${controller_root}/ms_formal.log" ms ms_pid ms_pgid; then
  fail_controller MS_SESSION_START_FAILED 24
fi
rtk printf '%s\n' "${ms_pid}" >"${controller_root}/ms_wrapper.pid"
rtk printf '%s\n' "${ms_pgid}" >"${controller_root}/ms_process_group.pgid"
rtk printf '%s\n' "${ms_pgid}" >"${controller_root}/ms_session_leader.pid"
rtk echo 'Waiting for exact resumed scalar gate '"'\"step\": 51446'"'.'

while true; do
  if ! scan_for_fatal_training_output; then
    fail_controller MS_FATAL_OUTPUT_DETECTED 21
  fi
  set +e
  validate_first_200_steps
  gate_exit=$?
  set -e
  if (( gate_exit == 0 )); then
    break
  fi
  if (( gate_exit != 10 )); then
    fail_controller MS_NONFINITE_SCALARS 22
  fi
  if ! child_is_running "${ms_pid}"; then
    set +e
    wait "${ms_pid}"
    ms_early_exit=$?
    set -e
    ms_pid=''
    ms_pgid=''
    rtk printf '%s\n' "${ms_early_exit}" >"${controller_root}/ms_exit_code"
    if (( ms_early_exit == 0 )); then
      ms_early_exit=23
    fi
    fail_controller MS_EXITED_BEFORE_STEP_51446 "${ms_early_exit}"
  fi
  if at_deadline; then
    fail_controller TIME_LIMIT_REACHED 124 INTERRUPTED
  fi
  rtk sleep 10
done
rtk touch "${controller_root}/MS_FIRST_200_STEPS_VALIDATED"

if ! run_synchronously "${ss_smoke_launcher}" \
    "${controller_root}/ss_smoke.log" ss_smoke; then
  fail_controller SS_SMOKE_FAILED 30
fi

rtk date --iso-8601=seconds >"${controller_root}/ss_started_at"
if ! start_managed_job "${ss_formal_launcher}" \
    "${controller_root}/ss_formal.log" ss ss_pid ss_pgid; then
  fail_controller SS_SESSION_START_FAILED 31
fi
rtk printf '%s\n' "${ss_pid}" >"${controller_root}/ss_wrapper.pid"
rtk printf '%s\n' "${ss_pgid}" >"${controller_root}/ss_process_group.pgid"
rtk printf '%s\n' "${ss_pgid}" >"${controller_root}/ss_session_leader.pid"

priority_breach_count=0
priority_stop_requested=0
priority_stop_baseline_epoch=0
priority_poll_seconds=30
next_priority_poll="$(rtk date +%s)"
ss_observed_epoch=0
ss_latest_checkpoint=NONE
while child_is_running "${ms_pid}" || child_is_running "${ss_pid}"; do
  if at_deadline; then
    fail_controller TIME_LIMIT_REACHED 124 INTERRUPTED
  fi
  now_epoch="$(rtk date +%s)"

  checkpoint_info=''
  set +e
  checkpoint_info="$(latest_complete_ss_checkpoint)"
  checkpoint_exit=$?
  set -e
  if (( checkpoint_exit == 0 )); then
    checkpoint_epoch="${checkpoint_info%%$'\t'*}"
    checkpoint_path="${checkpoint_info#*$'\t'}"
    if (( checkpoint_epoch > ss_observed_epoch )); then
      ss_observed_epoch=${checkpoint_epoch}
      ss_latest_checkpoint=${checkpoint_path}
    fi
  elif (( checkpoint_exit != 10 )); then
    fail_controller SS_CHECKPOINT_SCAN_FAILED 41
  fi

  if (( priority_stop_requested == 1 &&
        ss_observed_epoch > priority_stop_baseline_epoch )) && \
      child_is_running "${ms_pid}" && child_is_running "${ss_pid}"; then
    if ! stop_secondary_for_primary_priority "${ss_latest_checkpoint}"; then
      fail_controller SS_PRIORITY_STOP_FAILED 42
    fi
  fi

  if (( priority_stop_requested == 0 && now_epoch >= next_priority_poll )) && \
      child_is_running "${ms_pid}" && child_is_running "${ss_pid}"; then
    primary_time_median=''
    set +e
    primary_time_median="$(measure_primary_time_median)"
    median_exit=$?
    set -e
    next_priority_poll=$(( now_epoch + priority_poll_seconds ))
    if [[ -n "${primary_time_median}" ]]; then
      rtk printf '%s\n' "${primary_time_median}" \
        >"${controller_root}/primary_recent_time_median"
    fi
    if (( median_exit == 20 )); then
      priority_breach_count=$(( priority_breach_count + 1 ))
    elif (( median_exit == 0 || median_exit == 10 )); then
      priority_breach_count=0
    else
      fail_controller MS_PRIORITY_SCALAR_SCAN_FAILED 43
    fi
    rtk printf '%s\n' "${priority_breach_count}" \
      >"${controller_root}/primary_priority_breach_count"

    if (( priority_breach_count >= 3 )); then
      priority_stop_requested=1
      priority_stop_baseline_epoch=${ss_observed_epoch}
      rtk touch "${controller_root}/SS_STOP_REQUESTED"
      rtk printf '%s\n' "${ss_latest_checkpoint}" \
        >"${controller_root}/ss_stop_requested_checkpoint"
    fi
  fi
  rtk sleep 5
done

set +e
if [[ -n "${ms_pid}" ]]; then
  wait "${ms_pid}"
  ms_exit=$?
  ms_pid=''
  ms_pgid=''
fi
if (( ss_exit_captured == 0 )) && [[ -n "${ss_pid}" ]]; then
  wait "${ss_pid}"
  ss_exit=$?
  ss_pid=''
  ss_pgid=''
  ss_exit_captured=1
fi
set -e
rtk printf '%s\n' "${ms_exit}" >"${controller_root}/ms_exit_code"
rtk printf '%s\n' "${ss_exit}" >"${controller_root}/ss_exit_code"

training_failed=0
training_interrupted=0
if (( ms_exit != 0 )); then
  training_failed=1
  if (( ms_exit == 124 || ms_exit == 130 || ms_exit == 137 || ms_exit == 143 )); then
    training_interrupted=1
    rtk touch "${controller_root}/MS_TRAINING_INTERRUPTED"
  else
    rtk touch "${controller_root}/MS_TRAINING_FAILED"
  fi
elif [[ ! -s "${primary_work_dir}/epoch_8.pth" || ! -e "${primary_work_dir}/COMPLETE" || -e "${primary_work_dir}/epoch_9.pth" ]]; then
  training_failed=1
  rtk touch "${controller_root}/MS_FINAL_CONTRACT_FAILED"
fi
if (( ss_exit != 0 )); then
  training_failed=1
  if (( ss_exit == 124 || ss_exit == 130 || ss_exit == 137 || ss_exit == 143 )); then
    training_interrupted=1
    rtk touch "${controller_root}/SS_TRAINING_INTERRUPTED"
  else
    rtk touch "${controller_root}/SS_TRAINING_FAILED"
  fi
elif [[ ! -s "${secondary_work_dir}/epoch_12.pth" || ! -e "${secondary_work_dir}/COMPLETE" || -e "${secondary_work_dir}/epoch_13.pth" ]]; then
  training_failed=1
  rtk touch "${controller_root}/SS_FINAL_CONTRACT_FAILED"
fi

# Post-evaluation launchers self-gate on the remaining window. Evaluation skips
# or failures are recorded but never turn successful bounded training into a lie.
if (( ss_exit == 0 )) && [[ -s "${secondary_work_dir}/epoch_12.pth" && -e "${secondary_work_dir}/COMPLETE" && ! -e "${secondary_work_dir}/epoch_13.pth" ]]; then
  if ! run_synchronously "${ss_posteval_launcher}" \
      "${controller_root}/ss_posteval.log" ss_posteval; then
    rtk touch "${controller_root}/SS_POSTEVAL_FAILED"
  fi
fi
if (( ms_exit == 0 )) && [[ -s "${primary_work_dir}/epoch_8.pth" && -e "${primary_work_dir}/COMPLETE" && ! -e "${primary_work_dir}/epoch_9.pth" ]]; then
  if ! run_synchronously "${ms_posteval_launcher}" \
      "${controller_root}/ms_posteval.log" ms_posteval; then
    rtk touch "${controller_root}/MS_POSTEVAL_FAILED"
  fi
fi

if (( training_failed != 0 )); then
  if at_deadline; then
    rtk touch "${controller_root}/TIME_LIMIT_REACHED"
    mark_status INTERRUPTED
    exit 124
  fi
  if (( training_interrupted != 0 )); then
    mark_status INTERRUPTED
    exit 130
  fi
  mark_status FAILED
  exit 40
fi

mark_status COMPLETE
rtk echo 'DOTA six-GPU bounded training controller completed both training contracts.'
