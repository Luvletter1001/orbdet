from pathlib import Path
import os
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
MONITOR = (
    ROOT / 'scripts/formal/' /
    'monitor_orbdet_v0_2_dota1_live_six_gpu_20260818.sh')
CONTROLLER = (
    '/data1/zcy/Orbdet/scripts/formal/'
    'run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh')
MS_SESSION = 'orbdet_dota1_msrr_e3e8_gpu4567_20260818'
SS_SESSION = 'orbdet_dota1_ss_seed42_gpu89_20260818'
MS_LAUNCHER = (
    '/data1/zcy/Orbdet/scripts/formal/'
    'run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh')
SS_LAUNCHER = (
    '/data1/zcy/Orbdet/scripts/formal/'
    'run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh')
MS_WORKDIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818')
SS_WORKDIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ss_gpu89_seed42_20260818')
MONITOR_ROOT = (
    '/data1/zcy/Orbdet/work_dirs/controllers/'
    'orbdet_dota1_sixgpu_8h_20260818/live_monitor')
DEADLINE = '2026-08-18 08:20:00 +0800'
MS_COMMAND = (
    f'rtk bash {MS_LAUNCHER} > '
    '/data1/zcy/Orbdet/work_dirs/controllers/'
    'orbdet_dota1_sixgpu_8h_20260818/ms_formal.log 2>&1')
SS_COMMAND = (
    f'rtk bash {SS_LAUNCHER} > '
    '/data1/zcy/Orbdet/work_dirs/controllers/'
    'orbdet_dota1_sixgpu_8h_20260818/ss_formal.log 2>&1')


def monitor_text():
    return MONITOR.read_text()


def test_monitor_is_strict_attach_only_and_fail_closed():
    text = monitor_text()
    assert text.startswith('#!/usr/bin/env bash\nset -euo pipefail\n')
    for exact in (
            CONTROLLER, MS_SESSION, SS_SESSION, MS_LAUNCHER, SS_LAUNCHER,
            MS_WORKDIR, SS_WORKDIR, MONITOR_ROOT, DEADLINE, MS_COMMAND,
            SS_COMMAND):
        assert exact in text
    assert 'ORBDET_CONTROLLER_LIBRARY_ONLY=1' in text
    assert 'source "${controller_helper}"' in text
    assert 'ORBDET_LIVE_MONITOR_LIBRARY_ONLY' in text
    assert 'tmux has-session' in text
    assert text.count('-t "=${session_name}"') == 3
    assert "'#{pane_pid}'" in text
    assert "'#{pane_id}'" in text
    assert 'pane_count != 1' in text
    assert 'leader_pid' in text
    assert 'actual_pgid' in text
    assert 'actual_sid' in text
    assert 'monitor_sid' in text
    assert '/proc/${expected_pid}/cmdline' in text
    assert "mapfile -d ''" in text
    assert '[[ "${#process_argv[@]}" == 3 ]]' in text
    assert 'rtk basename "${process_argv[0]}"' in text
    assert '"${process_argv[1]}" == -c' in text
    assert '"${process_argv[2]}" == "${expected_command}"' in text
    assert '*"${launcher_token}"*' not in text
    assert 'Refusing to overwrite existing monitor state' in text
    assert 'rtk mkdir "${monitor_root}"' in text
    assert 'rtk touch "${monitor_root}/RUNNING"' in text
    assert 'rtk mv "${monitor_root}/RUNNING"' in text
    assert 'trap on_error ERR' in text
    assert 'trap on_signal INT TERM' in text
    assert 'tmux new-session' not in text
    assert 'tmux kill-session' not in text
    assert 'nohup' not in text
    assert 'tools/train.py' not in text
    assert 'torch.distributed' not in text
    assert '--resume' not in text
    assert 'rtk rm ' not in text


def test_monitor_checks_progress_fatals_and_checkpoint_inventory():
    text = monitor_text()
    for required in (
            'poll_seconds=30', 'scalars.json', 'math.isfinite', 'loss',
            'grad_norm', 'time', 'Traceback', 'RuntimeError', 'NCCL error',
            'CUDA out of memory', 'NaN', 'Infinity', 'monitor.log',
            'checkpoint_inventory', 'st_size', 'st_mtime_ns',
            'nvidia-smi -i "${gpu_list}"', 'query_gpu_state MS 4,5,6,7',
            'query_gpu_state SS 8,9'):
        assert required in text
    assert 'RuntimeInfo' not in text
    assert 'sha256sum' in text
    assert text.index('sha256sum') > text.index('TRAINING_CONTRACT_VERIFIED')


def test_monitor_enforces_primary_priority_at_strict_epoch_boundary():
    text = monitor_text()
    for required in (
            'priority_time_threshold=0.341', 'recent_limit=20',
            'statistics.median', 'priority_breach_count >= 3',
            'ms_scalars_path', 'bind_current_scalars_path',
            'last_priority_step', 'latest_step > last_priority_step',
            'priority_observation_advanced',
            'last_priority_step="$(latest_scalar_step "${ms_scalars_path}")"',
            'SS_STOP_REQUESTED', 'priority_stop_baseline_epoch',
            'latest_ss_epoch > priority_stop_baseline_epoch',
            'signal_owned_session "${ss_sid}" TERM',
            'wait_for_owned_session_shutdown "${ss_sid}"',
            'SS_STOPPED_FOR_PRIMARY_PRIORITY'):
        assert required in text
    assert text.index('SS_STOP_REQUESTED') < text.index(
        'latest_ss_epoch > priority_stop_baseline_epoch')
    median_body = text.split('measure_recent_ms_median() {', 1)[1].split(
        '\n}', 1)[0]
    assert "root.rglob('scalars.json')" not in median_body


def test_exact_proc_argv_accepts_only_declared_bash_c_command(tmp_path):
    expected = 'rtk timeout 60s rtk sleep 60 & wait'
    good = subprocess.Popen(
        ['/usr/bin/bash', '-c', expected], start_new_session=True)
    bad = subprocess.Popen(
        ['/usr/bin/bash', '-c', expected, expected], start_new_session=True)
    harness = r'''
set -euo pipefail
ORBDET_LIVE_MONITOR_LIBRARY_ONLY=1
source "$1"
validate_registered_leader "$2" "$2" "$3"
'''
    refusal_harness = r'''
set -euo pipefail
ORBDET_LIVE_MONITOR_LIBRARY_ONLY=1
source "$1"
declare -F terminate_registered_sid >/dev/null
monitor_root=$4
rtk mkdir -p "${monitor_root}"
if terminate_registered_sid TEST "$2" "$2" "$3"; then
  exit 91
fi
session_has_members "$2"
'''
    try:
        for process in (good, bad):
            for _ in range(40):
                if os.getsid(process.pid) == process.pid:
                    break
                time.sleep(0.05)
        accepted = subprocess.run(
            ['bash', '-c', harness, 'argv-test', str(MONITOR),
             str(good.pid), expected], capture_output=True, text=True,
            timeout=5, check=False)
        embedded_extra = subprocess.run(
            ['bash', '-c', harness, 'argv-test', str(MONITOR),
             str(bad.pid), expected], capture_output=True, text=True,
            timeout=5, check=False)
        assert accepted.returncode == 0, accepted.stderr
        assert embedded_extra.returncode != 0
        refused_signal = subprocess.run(
            ['bash', '-c', refusal_harness, 'argv-test', str(MONITOR),
             str(bad.pid), expected, str(tmp_path / 'refusal')],
            capture_output=True, text=True, timeout=5, check=False)
        assert refused_signal.returncode == 0, refused_signal.stderr
        assert bad.poll() is None
    finally:
        for process in (good, bad):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def test_priority_breaches_require_three_distinct_increasing_steps():
    harness = r'''
set -euo pipefail
ORBDET_LIVE_MONITOR_LIBRARY_ONLY=1
source "$1"
priority_time_threshold=0.341
priority_breach_count=0
last_priority_step=-1
update_priority_observation 100 0.400
update_priority_observation 100 0.400
update_priority_observation 100 0.400
same_step_count=${priority_breach_count}
update_priority_observation 101 0.400
update_priority_observation 102 0.400
rtk printf '%s %s %s\n' "${same_step_count}" \
  "${priority_breach_count}" "${last_priority_step}"
'''
    result = subprocess.run(
        ['bash', '-c', harness, 'priority-test', str(MONITOR)],
        capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1 3 102'


def test_monitor_verifies_exact_training_and_posteval_contracts():
    text = monitor_text()
    required = (
        'epoch_12.pth', 'epoch_13.pth', '--expected-epoch 12',
        '--expected-iter 38280', '--expected-state-tensors 371',
        '--config-token OrbdetV02Detector', '--config-token randomness',
        'epoch_8.pth', 'epoch_9.pth', '--expected-epoch 8',
        '--expected-iter 136656', '--config-token trainval_ms_full',
        'run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh',
        'run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh',
        'SS_POSTEVAL_STARTED', 'MS_POSTEVAL_STARTED',
        'POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW', 'TRAINING_CONTRACT_VERIFIED',
    )
    for token in required:
        assert token in text
    assert text.count('SS_POSTEVAL_STARTED') == 1
    assert text.count('MS_POSTEVAL_STARTED') == 1
    assert 'if at_deadline; then' in text
    assert 'no post-evaluation after deadline' in text


def test_monitor_deadline_only_signals_registered_sids_and_finishes_atomically():
    text = monitor_text()
    for required in (
            'TIME_LIMIT_REACHED', 'terminate_registered_sid',
            'signal_owned_session "${registered_sid}" TERM',
            'wait_for_owned_session_shutdown "${registered_sid}"',
            'session_has_members "${registered_sid}"',
            'mark_monitor INTERRUPTED', 'mark_monitor COMPLETE',
            'mark_monitor FAILED', 'ss_stopped_for_priority'):
        assert required in text
    assert 'kill -TERM -- "-${registered_sid}"' not in text
    assert 'kill -KILL -- "-${registered_sid}"' not in text
    assert 'rtk kill ' not in text
    assert text.count('if ! terminate_registered_sid') == 2


def test_registered_sid_topology_validation_and_real_shutdown_are_bounded(tmp_path):
    assert MONITOR.exists(), 'monitor implementation is required'
    token = 'orbdet-live-monitor-owned-topology-token'
    # Keep the owned test token inside ps's displayed argv prefix.
    launcher = Path('/tmp') / f'{token}-{id(tmp_path)}.sh'
    launcher.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'rtk timeout 60s rtk sleep 60\n')
    launcher.chmod(0o755)
    harness = r'''
set -euo pipefail
ORBDET_LIVE_MONITOR_LIBRARY_ONLY=1
source "$1"
topology_pid=''
topology_sid=''
cleanup() {
  if [[ -n "${topology_sid}" ]] && session_has_members "${topology_sid}"; then
    signal_owned_session "${topology_sid}" TERM
    wait_for_owned_session_shutdown "${topology_sid}" || true
  fi
  if [[ -n "${topology_pid}" ]]; then
    wait "${topology_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
rtk setsid --wait rtk bash "$2" &
topology_pid=$!
for sample in {1..40}; do
  topology_sid="$(rtk ps -ww --ppid "${topology_pid}" \
    -o pid=,pgid=,sid= | rtk awk \
    '$1 == $2 && $1 == $3 { print $1; exit }')"
  [[ -n "${topology_sid}" ]] && break
  rtk sleep 0.05
done
[[ "${topology_sid}" =~ ^[0-9]+$ ]]
group_count="$(rtk ps --sid "${topology_sid}" -o sid=,pgid= | \
  rtk awk -v sid="${topology_sid}" \
  '$1 == sid { groups[$2] = 1 } END { print length(groups) }')"
if (( group_count < 2 )); then
  rtk echo 'nested timeout did not form a second process group' >&2
  exit 91
fi
terminate_owned_session "${topology_pid}" "${topology_sid}"
wait "${topology_pid}" 2>/dev/null || true
topology_pid=''
if session_has_members "${topology_sid}"; then
  rtk echo 'registered session still has members' >&2
  exit 92
fi
trap - EXIT
'''
    started = time.monotonic()
    result = subprocess.run(
        ['bash', '-c', harness, 'topology-test', str(MONITOR),
         str(launcher), token],
        capture_output=True, text=True,
        timeout=10, check=False)
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert elapsed < 10
    launcher.unlink(missing_ok=True)
