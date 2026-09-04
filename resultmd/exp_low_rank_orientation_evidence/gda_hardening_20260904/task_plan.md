# GDA hardening task plan

## Goal

Prove and repair GDA Plan-B correctness defects without starting formal
training.

## Phases

- [x] Freeze and stop invalid P1; preserve logs/checkpoint.
- [x] Obtain independent adversarial review and user approval.
- [x] Write and self-review the hardening design.
- [x] Write regression tests and verify RED.
- [x] Implement object-identity and empty-graph repair; verify GREEN.
- [x] Implement numerical safety; verify GREEN.
- [x] Implement DDP reduction, inference/analysis separation, masks, and
  diagnostics; verify GREEN.
- [x] Correct controlled configs and experiment documentation.
- [x] Run focused and relevant regression verification.
- [x] Incorporate the independent adversarial review and complete final
  self-review/report.

## Constraints

- Prefix non-interactive shell commands with `rtk`.
- Use `/data/zcy/anaconda3/envs/orbdet/bin/python` with
  `PYTHONNOUSERSITE=1`.
- Do not start formal training, a queue, tmux, nohup, or resume.
- Do not modify linked DOTA source data.
- Preserve unrelated untracked planning files in the worktree.

## Errors

| Error | Resolution |
|---|---|
| Existing root `task_plan.md` belongs to a prior low-rank task | Use this task-specific directory; do not overwrite prior files. |
| `git add` could not create the shared worktree `index.lock` in the managed sandbox | Retry only the scoped add/commit command with approved escalation. |
| CPU/Gloo could not open a loopback socket inside the managed sandbox | Set `GLOO_SOCKET_IFNAME=lo` and rerun only the CPU test with scoped escalation. |
| Full suite first reported one Gloo failure inside the restricted sandbox | Re-ran the identical full suite with scoped loopback permission: 174 passed. |
| Sandboxed `nvidia-smi` could not reach the driver | Scoped read-only rerun succeeded; a transient non-GDA process exited before inspection. |
