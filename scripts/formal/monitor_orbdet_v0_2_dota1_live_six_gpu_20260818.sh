#!/usr/bin/env bash
set -euo pipefail

controller_helper=/data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh
ORBDET_CONTROLLER_LIBRARY_ONLY=1
source "${controller_helper}"
unset ORBDET_CONTROLLER_LIBRARY_ONLY

validate_registered_leader() {
  local expected_pid=$1
  local registered_sid=$2
  local expected_command=$3
  local process_row leader_pid actual_pgid actual_sid executable_basename
  local current_monitor_sid
  local -a process_argv=()

  [[ "${expected_pid}" =~ ^[0-9]+$ && "${registered_sid}" =~ ^[0-9]+$ ]] || return 1
  process_row="$(rtk ps -o pid=,pgid=,sid= \
    -p "${expected_pid}" 2>/dev/null)" || return 1
  read -r leader_pid actual_pgid actual_sid <<<"${process_row}"
  current_monitor_sid="$(rtk ps -o sid= -p "$$")"
  current_monitor_sid="${current_monitor_sid//[[:space:]]/}"
  [[ "${leader_pid}" == "${expected_pid}" &&
     "${leader_pid}" == "${actual_pgid}" &&
     "${leader_pid}" == "${actual_sid}" &&
     "${actual_sid}" == "${registered_sid}" &&
     "${actual_sid}" != "${current_monitor_sid}" ]] || return 1
  [[ -r "/proc/${expected_pid}/cmdline" ]] || return 1
  mapfile -d '' -t process_argv <"/proc/${expected_pid}/cmdline" || return 1
  [[ "${#process_argv[@]}" == 3 ]] || return 1
  executable_basename="$(rtk basename "${process_argv[0]}")"
  [[ "${executable_basename}" == bash &&
     "${process_argv[1]}" == -c &&
     "${process_argv[2]}" == "${expected_command}" ]]
}

update_priority_observation() {
  local latest_step=$1
  local median_time=$2
  priority_observation_advanced=0
  if (( latest_step > last_priority_step )); then
    last_priority_step=${latest_step}
    priority_observation_advanced=1
    if rtk awk -v median="${median_time}" \
        -v threshold="${priority_time_threshold}" \
        'BEGIN { exit !(median > threshold) }'; then
      priority_breach_count=$(( priority_breach_count + 1 ))
    else
      priority_breach_count=0
    fi
  fi
}

python_bin=/data/zcy/anaconda3/envs/orbdet/bin/python
checkpoint_validator=/data1/zcy/Orbdet/tools/analysis_tools/validate_checkpoint_contract.py
deadline='2026-08-18 08:20:00 +0800'
deadline_epoch="$(rtk date -d "${deadline}" +%s)"
poll_seconds=30
priority_time_threshold=0.341
recent_limit=20

monitor_root=/data1/zcy/Orbdet/work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/live_monitor
ms_session=orbdet_dota1_msrr_e3e8_gpu4567_20260818
ss_session=orbdet_dota1_ss_seed42_gpu89_20260818
ms_launcher=/data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh
ss_launcher=/data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh
ms_expected_command='rtk bash /data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh > /data1/zcy/Orbdet/work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/ms_formal.log 2>&1'
ss_expected_command='rtk bash /data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh > /data1/zcy/Orbdet/work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/ss_formal.log 2>&1'
ms_workdir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818
ss_workdir=/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818
ms_posteval_launcher=/data1/zcy/Orbdet/scripts/eval/run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh
ss_posteval_launcher=/data1/zcy/Orbdet/scripts/eval/run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh
ms_posteval_status=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_epoch8_20260818/status
ss_posteval_status=/data1/zcy/Orbdet/work_dirs/eval/orbdet_v0_2_dota1_ss_gpu89_seed42_epoch12_20260818/status

ms_pane_pid=''
ms_sid=''
ss_pane_pid=''
ss_sid=''
ms_scalars_path=''
ss_scalars_path=''
monitor_sid="$(rtk ps -o sid= -p "$$")"
monitor_sid="${monitor_sid//[[:space:]]/}"
monitor_active=0
fatal_monitor_condition=0
priority_breach_count=0
last_priority_step=-1
priority_observation_advanced=0
priority_stop_requested=0
priority_stop_baseline_epoch=0
ss_stopped_for_priority=0
ms_outcome=RUNNING
ss_outcome=RUNNING
ms_posteval_outcome=PENDING
ss_posteval_outcome=PENDING
ss_posteval_started_marker="${monitor_root}/SS_POSTEVAL_STARTED"
ms_posteval_started_marker="${monitor_root}/MS_POSTEVAL_STARTED"
declare -A seen_checkpoints=()

at_deadline() {
  (( $(rtk date +%s) >= deadline_epoch ))
}

atomic_write() {
  local target=$1
  local value=$2
  local temporary="${target}.tmp.$$.$RANDOM"
  rtk printf '%s\n' "${value}" >"${temporary}"
  rtk mv "${temporary}" "${target}"
}

append_log() {
  local message=$1
  rtk printf '%s %s\n' "$(rtk date --iso-8601=seconds)" "${message}" \
    >>"${monitor_root}/monitor.log"
}

mark_monitor() {
  local terminal_status=$1
  if (( monitor_active == 1 )) && [[ -e "${monitor_root}/RUNNING" ]]; then
    rtk mv "${monitor_root}/RUNNING" "${monitor_root}/${terminal_status}"
    monitor_active=0
  fi
}

record_fatal_condition() {
  local marker=$1
  local detail=$2
  fatal_monitor_condition=1
  rtk touch "${monitor_root}/${marker}"
  append_log "FATAL ${marker}: ${detail}"
}

attach_tmux_session() {
  local label=$1
  local session_name=$2
  local expected_command=$3
  local pid_variable=$4
  local sid_variable=$5
  local panes pane_count pane_pid process_row
  local leader_pid actual_pgid actual_sid

  rtk tmux has-session -t "=${session_name}"
  panes="$(rtk tmux list-panes -t "=${session_name}" -F '#{pane_id}')"
  pane_count="$(rtk printf '%s\n' "${panes}" | rtk awk 'NF { count += 1 } END { print count + 0 }')"
  if (( pane_count != 1 )); then
    rtk echo "Expected exactly one pane in ${session_name}; found ${pane_count}." >&2
    return 20
  fi
  pane_pid="$(rtk tmux display-message -p -t "=${session_name}" '#{pane_pid}')"
  [[ "${pane_pid}" =~ ^[0-9]+$ ]] || return 21
  process_row="$(rtk ps -o pid=,pgid=,sid= -p "${pane_pid}")"
  read -r leader_pid actual_pgid actual_sid <<<"${process_row}"
  if ! validate_registered_leader "${pane_pid}" "${actual_sid}" "${expected_command}"; then
    rtk echo "Untrusted process topology for ${session_name}." >&2
    return 22
  fi
  if [[ "${actual_sid}" == "${monitor_sid}" ]]; then
    rtk echo "Refusing monitor session SID for ${session_name}." >&2
    return 23
  fi
  printf -v "${pid_variable}" '%s' "${pane_pid}"
  printf -v "${sid_variable}" '%s' "${actual_sid}"
  atomic_write "${monitor_root}/${label}_attachment" \
    "session=${session_name} pane_pid=${pane_pid} sid=${actual_sid} argv0=bash argv1=-c command=${expected_command}"
  append_log "ATTACHED ${label} pane_pid=${pane_pid} sid=${actual_sid} exact_command=${expected_command}"
}

verify_live_registration() {
  local label=$1
  local pane_pid=$2
  local registered_sid=$3
  local expected_command=$4
  if ! session_has_members "${registered_sid}"; then
    return 10
  fi
  if ! validate_registered_leader "${pane_pid}" "${registered_sid}" "${expected_command}"; then
    record_fatal_condition "${label}_LIVE_TOPOLOGY_INVALID" \
      "registered SID ${registered_sid} still has members but its leader/token changed"
    return 11
  fi
  return 0
}

query_gpu_state() {
  local label=$1
  local gpu_list=$2
  local gpu_processes
  if ! gpu_processes="$(rtk nvidia-smi -i "${gpu_list}" \
      --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    record_fatal_condition "${label}_GPU_QUERY_FAILED" \
      "read-only nvidia-smi query failed for ${gpu_list}"
    return 1
  fi
  atomic_write "${monitor_root}/${label}_gpu_processes" "${gpu_processes:-IDLE}"
}

gpu_group_is_idle() {
  local gpu_list=$1
  local gpu_processes
  gpu_processes="$(rtk nvidia-smi -i "${gpu_list}" \
    --query-compute-apps=pid --format=csv,noheader,nounits)" || return 2
  [[ -z "${gpu_processes}" ]]
}

bind_current_scalars_path() {
  local workdir=$1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${workdir}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = list(root.rglob('scalars.json')) if root.exists() else []
if not paths:
    print(f'no scalars.json beneath {root}', file=sys.stderr)
    raise SystemExit(10)
newest_mtime = max(path.stat().st_mtime_ns for path in paths)
newest = [path for path in paths if path.stat().st_mtime_ns == newest_mtime]
if len(newest) != 1:
    print(f'ambiguous newest scalars.json paths: {newest}', file=sys.stderr)
    raise SystemExit(11)
print(newest[0].resolve())
PY
}

latest_scalar_step() {
  local scalars_path=$1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${scalars_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
steps = []
for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
    try:
        record = json.loads(line)
    except (TypeError, ValueError):
        continue
    step = record.get('step', record.get('iter')) if isinstance(record, dict) else None
    if isinstance(step, int) and not isinstance(step, bool):
        steps.append(step)
if not steps:
    print(f'bound scalar file has no integer steps: {path}', file=sys.stderr)
    raise SystemExit(12)
print(max(steps))
PY
}

read_latest_scalars() {
  local scalars_path=$1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${scalars_path}" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(10)
try:
    lines = [line for line in path.read_text(
        encoding='utf-8', errors='replace').splitlines() if line.strip()]
except OSError as error:
    print(error, file=sys.stderr)
    raise SystemExit(11)
if not lines:
    raise SystemExit(10)
try:
    record = json.loads(lines[-1])
except (TypeError, ValueError) as error:
    print(f'latest scalar JSON is invalid: {error}', file=sys.stderr)
    raise SystemExit(11)
if not isinstance(record, dict):
    print('latest scalar JSON is not an object', file=sys.stderr)
    raise SystemExit(11)
for name in ('loss', 'grad_norm', 'time'):
    value = record.get(name)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        print(f'latest scalar {name} is not numeric and finite: {value}',
              file=sys.stderr)
        raise SystemExit(12)
epoch = record.get('epoch', 'NA')
iteration = record.get('iter', record.get('step', 'NA'))
print(f'path={path} epoch={epoch} iter={iteration} time={record["time"]}')
PY
}

scan_exact_fatal_patterns() {
  local workdir=$1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${workdir}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
log_patterns = (
    re.compile(r'(?<![A-Za-z])Traceback(?![A-Za-z])'),
    re.compile(r'(?<![A-Za-z])RuntimeError(?![A-Za-z])'),
    re.compile(r'(?<![A-Za-z])NCCL error(?![A-Za-z])', re.IGNORECASE),
    re.compile(r'(?<![A-Za-z])CUDA out of memory(?![A-Za-z])', re.IGNORECASE),
)
json_nonfinite = re.compile(
    r'(?<![A-Za-z])(?:NaN|Infinity|-Infinity)(?![A-Za-z])')
for path in root.rglob('*') if root.exists() else ():
    if not path.is_file() or path.suffix not in {'.log', '.json'}:
        continue
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        continue
    patterns = log_patterns + ((json_nonfinite,) if path.suffix == '.json' else ())
    for pattern in patterns:
        if pattern.search(content):
            print(f'{path}: {pattern.pattern}')
            raise SystemExit(20)
PY
}

record_checkpoint_inventory() {
  local label=$1
  local workdir=$2
  local inventory line key
  inventory="$(rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${workdir}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
entries = []
for path in root.glob('epoch_*.pth') if root.exists() else ():
    match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
    if match is None or not path.is_file():
        continue
    stat = path.stat()
    entries.append((int(match.group(1)), path.name, stat.st_size,
                    stat.st_mtime_ns))
for _, name, st_size, st_mtime_ns in sorted(entries):
    print(f'{name}\t{st_size}\t{st_mtime_ns}')
PY
)"
  atomic_write "${monitor_root}/${label}_checkpoint_inventory" "${inventory}"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    key="${label}:${line}"
    if [[ ! -v "seen_checkpoints[${key}]" ]]; then
      seen_checkpoints["${key}"]=1
      append_log "NEW_CHECKPOINT ${label} ${line}"
    fi
  done <<<"${inventory}"
}

measure_recent_ms_median() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - \
    "${ms_scalars_path}" "${recent_limit}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

path = pathlib.Path(sys.argv[1])
recent_limit = int(sys.argv[2])
records = []
try:
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
except OSError as error:
    print(error, file=sys.stderr)
    raise SystemExit(11)
for line in lines:
    try:
        record = json.loads(line)
    except (TypeError, ValueError):
        continue
    value = record.get('time') if isinstance(record, dict) else None
    iteration = record.get('step', record.get('iter')) if isinstance(record, dict) else None
    if (isinstance(iteration, int) and not isinstance(iteration, bool) and
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) > 0.0):
        records.append((iteration, float(value)))
records.sort(key=lambda item: item[0])
if len(records) < recent_limit:
    raise SystemExit(10)
recent = [value for _, value in records[-recent_limit:]]
latest_step = records[-1][0]
print(f'{latest_step}\t{statistics.median(recent):.9f}')
PY
}

latest_complete_ss_checkpoint() {
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" - "${ss_workdir}" <<'PY'
import pathlib
import re
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
complete = []
for path in root.glob('epoch_*.pth') if root.exists() else ():
    match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
    if match is None or not path.is_file() or path.stat().st_size == 0:
        continue
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist() or archive.testzip() is not None:
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

terminate_registered_sid() {
  local label=$1
  local registered_pid=$2
  local registered_sid=$3
  local expected_command=$4
  if ! session_has_members "${registered_sid}"; then
    return 0
  fi
  if ! validate_registered_leader "${registered_pid}" "${registered_sid}" \
      "${expected_command}"; then
    rtk touch "${monitor_root}/${label}_DEADLINE_LEADER_UNVERIFIED"
    append_log "SIGNAL_REFUSED ${label} exact argv validation failed for registered_sid=${registered_sid}"
    return 72
  fi
  append_log "DEADLINE_VERIFIED ${label} sid=${registered_sid}"
  local registered_sid_guard="${registered_sid}"
  [[ "${registered_sid_guard}" =~ ^[0-9]+$ &&
     "${registered_sid_guard}" != "${monitor_sid}" ]] || return 70
  signal_owned_session "${registered_sid}" TERM
  wait_for_owned_session_shutdown "${registered_sid}"
  if session_has_members "${registered_sid}"; then
    return 71
  fi
  append_log "DEADLINE_STOPPED ${label} sid=${registered_sid}"
}

handle_deadline() {
  local termination_failed=0
  trap - ERR INT TERM
  rtk touch "${monitor_root}/TIME_LIMIT_REACHED"
  if [[ -n "${ms_sid}" ]]; then
    if ! terminate_registered_sid MS "${ms_pane_pid}" "${ms_sid}" \
        "${ms_expected_command}"; then
      termination_failed=1
    fi
  fi
  if [[ -n "${ss_sid}" ]]; then
    if ! terminate_registered_sid SS "${ss_pane_pid}" "${ss_sid}" \
        "${ss_expected_command}"; then
      termination_failed=1
    fi
  fi
  if (( termination_failed != 0 )); then
    mark_monitor FAILED
    rtk echo 'One or more registered SIDs could not be emptied at the deadline.' >&2
    exit 125
  fi
  mark_monitor INTERRUPTED
  exit 124
}

on_error() {
  local exit_code=$?
  trap - ERR INT TERM
  if at_deadline; then
    handle_deadline
  fi
  mark_monitor FAILED
  rtk echo "Live monitor failed with exit ${exit_code}." >&2
  exit "${exit_code}"
}

on_signal() {
  trap - ERR INT TERM
  if at_deadline; then
    handle_deadline
  fi
  mark_monitor INTERRUPTED
  rtk echo 'Live monitor interrupted; registered training sessions were not signaled before the deadline.' >&2
  exit 130
}

validate_ms_training_contract() {
  local checkpoint="${ms_workdir}/epoch_8.pth"
  [[ -e "${ms_workdir}/COMPLETE" && -s "${checkpoint}" &&
     ! -e "${ms_workdir}/epoch_9.pth" ]] || return 1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
    "${checkpoint}" --expected-epoch 8 --expected-iter 136656 \
    --expected-state-tensors 371 --config-token OrbdetV02Detector \
    --config-token trainval_ms_full \
    >"${monitor_root}/ms_checkpoint_contract.json"
  rtk touch "${monitor_root}/MS_TRAINING_CONTRACT_VERIFIED"
  rtk sha256sum "${checkpoint}" >"${monitor_root}/ms_checkpoint.sha256"
}

validate_ss_training_contract() {
  local checkpoint="${ss_workdir}/epoch_12.pth"
  [[ -e "${ss_workdir}/COMPLETE" && -s "${checkpoint}" &&
     ! -e "${ss_workdir}/epoch_13.pth" ]] || return 1
  rtk env PYTHONNOUSERSITE=1 "${python_bin}" "${checkpoint_validator}" \
    "${checkpoint}" --expected-epoch 12 --expected-iter 38280 \
    --expected-state-tensors 371 --config-token OrbdetV02Detector \
    --config-token randomness \
    >"${monitor_root}/ss_checkpoint_contract.json"
  rtk touch "${monitor_root}/SS_TRAINING_CONTRACT_VERIFIED"
  rtk sha256sum "${checkpoint}" >"${monitor_root}/ss_checkpoint.sha256"
}

observe_training_outcome() {
  local label=$1
  local registered_sid=$2
  local workdir=$3
  local validator=$4
  local outcome_variable=$5
  local current_outcome
  printf -v current_outcome '%s' "${!outcome_variable}"
  [[ "${current_outcome}" == RUNNING ]] || return 0
  if session_has_members "${registered_sid}"; then
    return 0
  fi
  if [[ "${label}" == SS && "${ss_stopped_for_priority}" == 1 ]]; then
    printf -v "${outcome_variable}" '%s' PRIORITY_STOPPED
    atomic_write "${monitor_root}/${label}_training_outcome" PRIORITY_STOPPED
  elif [[ -e "${workdir}/FAILED" ]]; then
    printf -v "${outcome_variable}" '%s' FAILED
    atomic_write "${monitor_root}/${label}_training_outcome" FAILED
  elif [[ -e "${workdir}/INTERRUPTED" ]]; then
    printf -v "${outcome_variable}" '%s' INTERRUPTED
    atomic_write "${monitor_root}/${label}_training_outcome" INTERRUPTED
  elif "${validator}"; then
    printf -v "${outcome_variable}" '%s' VERIFIED
    atomic_write "${monitor_root}/${label}_training_outcome" VERIFIED
  else
    printf -v "${outcome_variable}" '%s' FAILED
    atomic_write "${monitor_root}/${label}_training_outcome" VANISHED_WITHOUT_FORMAL_TERMINAL
    record_fatal_condition "${label}_SESSION_VANISHED" \
      "registered SID ${registered_sid} emptied without a valid formal terminal contract"
  fi
}

poll_training_progress() {
  local label=$1
  local workdir=$2
  local scalars_path=$3
  local scalar_result scalar_exit fatal_result fatal_exit
  set +e
  scalar_result="$(read_latest_scalars "${scalars_path}" 2>&1)"
  scalar_exit=$?
  set -e
  if (( scalar_exit == 0 )); then
    atomic_write "${monitor_root}/${label}_latest_progress" "${scalar_result}"
    append_log "PROGRESS ${label} ${scalar_result}"
  elif (( scalar_exit != 10 )); then
    record_fatal_condition "${label}_SCALAR_VALIDATION_FAILED" "${scalar_result}"
  fi
  set +e
  fatal_result="$(scan_exact_fatal_patterns "${workdir}" 2>&1)"
  fatal_exit=$?
  set -e
  if (( fatal_exit == 20 )); then
    record_fatal_condition "${label}_FATAL_OUTPUT_DETECTED" "${fatal_result}"
  elif (( fatal_exit != 0 )); then
    record_fatal_condition "${label}_FATAL_SCAN_FAILED" "${fatal_result}"
  fi
  record_checkpoint_inventory "${label}" "${workdir}"
}

poll_primary_priority() {
  local median_result median_time median_exit latest_step
  local checkpoint_info checkpoint_exit latest_ss_epoch
  if (( priority_stop_requested == 0 )); then
    set +e
    median_result="$(measure_recent_ms_median)"
    median_exit=$?
    set -e
    if (( median_exit == 0 )); then
      latest_step="${median_result%%$'\t'*}"
      median_time="${median_result#*$'\t'}"
      if [[ ! "${latest_step}" =~ ^[0-9]+$ ]]; then
        record_fatal_condition MS_PRIORITY_SCALAR_SCAN_FAILED \
          "invalid latest step from bound scalar file: ${median_result}"
        return
      fi
      update_priority_observation "${latest_step}" "${median_time}"
      if (( priority_observation_advanced == 1 )); then
        atomic_write "${monitor_root}/primary_recent_time_median" \
          "step=${latest_step} median=${median_time}"
        atomic_write "${monitor_root}/last_priority_step" "${last_priority_step}"
      fi
    elif (( median_exit == 10 )); then
      priority_observation_advanced=0
    else
      record_fatal_condition MS_PRIORITY_SCALAR_SCAN_FAILED \
        "median scan exited ${median_exit}"
      priority_observation_advanced=0
    fi
    if (( priority_observation_advanced == 1 )); then
      atomic_write "${monitor_root}/primary_priority_breach_count" \
        "${priority_breach_count}"
    fi
    if (( priority_breach_count >= 3 )); then
      priority_stop_requested=1
      set +e
      checkpoint_info="$(latest_complete_ss_checkpoint)"
      checkpoint_exit=$?
      set -e
      if (( checkpoint_exit == 0 )); then
        priority_stop_baseline_epoch="${checkpoint_info%%$'\t'*}"
      elif (( checkpoint_exit == 10 )); then
        priority_stop_baseline_epoch=0
      else
        record_fatal_condition SS_CHECKPOINT_SCAN_FAILED \
          "priority baseline scan exited ${checkpoint_exit}"
        return
      fi
      rtk touch "${monitor_root}/SS_STOP_REQUESTED"
      atomic_write "${monitor_root}/ss_priority_stop_baseline" \
        "epoch=${priority_stop_baseline_epoch} checkpoint=${checkpoint_info:-NONE}"
    fi
  fi

  if (( priority_stop_requested == 1 && ss_stopped_for_priority == 0 )); then
    set +e
    checkpoint_info="$(latest_complete_ss_checkpoint)"
    checkpoint_exit=$?
    set -e
    if (( checkpoint_exit == 0 )); then
      latest_ss_epoch="${checkpoint_info%%$'\t'*}"
      if (( latest_ss_epoch > priority_stop_baseline_epoch )); then
        if ! verify_live_registration SS "${ss_pane_pid}" "${ss_sid}" \
            "${ss_expected_command}"; then
          record_fatal_condition SS_PRIORITY_STOP_REFUSED \
            'SS leader was not the registered launcher at the strict checkpoint boundary'
          return
        fi
        signal_owned_session "${ss_sid}" TERM
        if ! wait_for_owned_session_shutdown "${ss_sid}"; then
          record_fatal_condition SS_PRIORITY_STOP_FAILED \
            "registered SS SID ${ss_sid} did not empty"
          return
        fi
        ss_stopped_for_priority=1
        ss_outcome=PRIORITY_STOPPED
        atomic_write "${monitor_root}/ss_priority_stop_checkpoint" "${checkpoint_info}"
        rtk touch "${monitor_root}/SS_STOPPED_FOR_PRIMARY_PRIORITY"
        atomic_write "${monitor_root}/SS_training_outcome" PRIORITY_STOPPED
      fi
    elif (( checkpoint_exit != 10 )); then
      record_fatal_condition SS_CHECKPOINT_SCAN_FAILED \
        "priority follow-up scan exited ${checkpoint_exit}"
    fi
  fi
}

record_posteval_terminal() {
  local label=$1
  local status_dir=$2
  local outcome_variable=$3
  if [[ -e "${status_dir}/COMPLETE" ]]; then
    printf -v "${outcome_variable}" '%s' COMPLETE
    atomic_write "${monitor_root}/${label}_posteval_outcome" COMPLETE
    return 0
  fi
  if [[ -e "${status_dir}/POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW" ]]; then
    printf -v "${outcome_variable}" '%s' POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW
    atomic_write "${monitor_root}/${label}_posteval_outcome" \
      POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW
    return 0
  fi
  return 1
}

run_posteval_once() {
  local label=$1
  local gpu_list=$2
  local launcher=$3
  local status_dir=$4
  local started_marker=$5
  local outcome_variable=$6
  local posteval_exit
  # Safety invariant: no post-evaluation after deadline.
  if at_deadline; then
    return 124
  fi
  if record_posteval_terminal "${label}" "${status_dir}" "${outcome_variable}"; then
    return 0
  fi
  if [[ -e "${started_marker}" ]]; then
    record_fatal_condition "${label}_POSTEVAL_NONTERMINAL" \
      'post-evaluation was already attempted without an allowed terminal marker'
    printf -v "${outcome_variable}" '%s' FAILED
    return 1
  fi
  if ! gpu_group_is_idle "${gpu_list}"; then
    return 10
  fi
  rtk touch "${started_marker}"
  set +e
  rtk bash "${launcher}" >"${monitor_root}/${label}_posteval.log" 2>&1
  posteval_exit=$?
  set -e
  if (( posteval_exit != 0 )) || \
      ! record_posteval_terminal "${label}" "${status_dir}" "${outcome_variable}"; then
    printf -v "${outcome_variable}" '%s' FAILED
    atomic_write "${monitor_root}/${label}_posteval_outcome" FAILED
    record_fatal_condition "${label}_POSTEVAL_FAILED" \
      "reviewed launcher exited ${posteval_exit} without an allowed terminal marker"
    return 1
  fi
}

all_registered_sids_empty() {
  ! session_has_members "${ms_sid}" && ! session_has_members "${ss_sid}"
}

if [[ "${ORBDET_LIVE_MONITOR_LIBRARY_ONLY:-0}" == 1 ]]; then
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
  fi
  exit 0
fi

rtk mkdir -p "$(rtk dirname "${monitor_root}")"
if [[ -e "${monitor_root}" ]]; then
  rtk echo "Refusing to overwrite existing monitor state in ${monitor_root}" >&2
  exit 2
fi
rtk mkdir "${monitor_root}"
rtk touch "${monitor_root}/RUNNING"
monitor_active=1
trap on_error ERR
trap on_signal INT TERM

attach_tmux_session MS "${ms_session}" "${ms_expected_command}" ms_pane_pid ms_sid
attach_tmux_session SS "${ss_session}" "${ss_expected_command}" ss_pane_pid ss_sid
[[ "${ms_sid}" != "${ss_sid}" ]]
ms_scalars_path="$(bind_current_scalars_path "${ms_workdir}")"
ss_scalars_path="$(bind_current_scalars_path "${ss_workdir}")"
last_priority_step="$(latest_scalar_step "${ms_scalars_path}")"
atomic_write "${monitor_root}/MS_scalars_binding" "${ms_scalars_path}"
atomic_write "${monitor_root}/SS_scalars_binding" "${ss_scalars_path}"
atomic_write "${monitor_root}/last_priority_step" "${last_priority_step}"
atomic_write "${monitor_root}/registered_sids" "MS=${ms_sid} SS=${ss_sid}"
append_log "START deadline=${deadline} monitor_sid=${monitor_sid}"

while true; do
  if at_deadline; then
    handle_deadline
  fi

  if [[ "${ms_outcome}" == RUNNING ]]; then
    set +e
    verify_live_registration MS "${ms_pane_pid}" "${ms_sid}" \
      "${ms_expected_command}"
    ms_live_exit=$?
    set -e
    if (( ms_live_exit == 0 )); then
      poll_training_progress MS "${ms_workdir}" "${ms_scalars_path}"
      query_gpu_state MS 4,5,6,7 || true
    fi
  fi
  if [[ "${ss_outcome}" == RUNNING ]]; then
    set +e
    verify_live_registration SS "${ss_pane_pid}" "${ss_sid}" \
      "${ss_expected_command}"
    ss_live_exit=$?
    set -e
    if (( ss_live_exit == 0 )); then
      poll_training_progress SS "${ss_workdir}" "${ss_scalars_path}"
      query_gpu_state SS 8,9 || true
    fi
  fi

  if [[ "${ms_outcome}" == RUNNING && "${ss_outcome}" == RUNNING ]] && \
      session_has_members "${ms_sid}" && session_has_members "${ss_sid}"; then
    poll_primary_priority
  fi

  observe_training_outcome MS "${ms_sid}" "${ms_workdir}" \
    validate_ms_training_contract ms_outcome
  observe_training_outcome SS "${ss_sid}" "${ss_workdir}" \
    validate_ss_training_contract ss_outcome

  if [[ "${ms_outcome}" != RUNNING && "${ss_outcome}" != RUNNING ]] && \
      all_registered_sids_empty; then
    if (( ss_stopped_for_priority == 1 )) || \
        [[ "${ms_outcome}" == INTERRUPTED || "${ss_outcome}" == INTERRUPTED ||
           "${ms_outcome}" == PRIORITY_STOPPED || "${ss_outcome}" == PRIORITY_STOPPED ]]; then
      mark_monitor INTERRUPTED
      exit 130
    fi
    if (( fatal_monitor_condition != 0 )) || \
        [[ "${ms_outcome}" != VERIFIED || "${ss_outcome}" != VERIFIED ]]; then
      mark_monitor FAILED
      exit 40
    fi

    if [[ "${ss_posteval_outcome}" == PENDING ]]; then
      run_posteval_once SS 8,9 "${ss_posteval_launcher}" \
        "${ss_posteval_status}" "${ss_posteval_started_marker}" \
        ss_posteval_outcome || true
    fi
    if at_deadline; then
      handle_deadline
    fi
    if [[ "${ms_posteval_outcome}" == PENDING ]]; then
      run_posteval_once MS 4,5,6,7 "${ms_posteval_launcher}" \
        "${ms_posteval_status}" "${ms_posteval_started_marker}" \
        ms_posteval_outcome || true
    fi
    if [[ "${ss_posteval_outcome}" == FAILED ||
          "${ms_posteval_outcome}" == FAILED ]]; then
      mark_monitor FAILED
      exit 50
    fi
    if [[ "${ss_posteval_outcome}" != PENDING &&
          "${ms_posteval_outcome}" != PENDING ]]; then
      mark_monitor COMPLETE
      append_log 'COMPLETE both training and allowed post-evaluation outcomes recorded'
      exit 0
    fi
  fi

  rtk sleep "${poll_seconds}"
done
