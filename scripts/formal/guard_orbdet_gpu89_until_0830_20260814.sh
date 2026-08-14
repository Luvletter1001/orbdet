#!/usr/bin/env bash
set -euo pipefail

DEADLINE_EPOCH=1786667400
PROJECT_ROOT=/data1/zcy/Orbdet
SESSIONS=(
  h2rbox_hrsc200e_bs2_gpu89_20260814
  orbdet_hrsc_clean200e_bs2_gpu89_20260814
  orbdet_hrsc_clean_queue_gpu89_20260814
)
CONFIGS=(
  "$PROJECT_ROOT/configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py"
  "$PROJECT_ROOT/configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py"
)

now="$(rtk date +%s)"
remaining=$((DEADLINE_EPOCH - now))
rtk echo "deadline_epoch=$DEADLINE_EPOCH remaining_seconds=$remaining"

while ((now < DEADLINE_EPOCH)); do
  remaining=$((DEADLINE_EPOCH - now))
  if ((remaining > 30)); then
    rtk sleep 30
  else
    rtk sleep "$remaining"
  fi
  now="$(rtk date +%s)"
done

rtk echo "deadline_reached_epoch=$now"
for session in "${SESSIONS[@]}"; do
  if rtk tmux has-session -t "$session" 2>/dev/null; then
    rtk echo "sending_interrupt_to_session=$session"
    rtk tmux send-keys -t "$session" C-c
  fi
done

# Allow an orderly checkpoint/exit, then signal only exact experiment configs.
rtk sleep 30
for config in "${CONFIGS[@]}"; do
  pattern="[p]ython.*tools/(train|test)\.py.*${config}"
  while read -r pid _; do
    [[ -n "$pid" ]] || continue
    rtk echo "sending_term_to_pid=$pid config=$config"
    rtk kill -TERM "$pid"
  done < <(rtk pgrep -af "$pattern" || :)
done

rtk sleep 30
for config in "${CONFIGS[@]}"; do
  pattern="[p]ython.*tools/(train|test)\.py.*${config}"
  while read -r pid _; do
    [[ -n "$pid" ]] || continue
    rtk echo "sending_kill_to_pid=$pid config=$config"
    rtk kill -KILL "$pid"
  done < <(rtk pgrep -af "$pattern" || :)
done

rtk echo 'deadline_guard_complete=1'
