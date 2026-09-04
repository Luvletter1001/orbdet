# GDA hardening progress

## 2026-09-04

- Reviewed GDA Plan-B v1.1 at commit `8492065`.
- Reproduced synthetic cross-view identity corruption.
- Reproduced NaN backward gradients at exact-zero `u2`.
- Independent reviewer reproduced identity corruption with the real DOTA
  pipeline and found additional flip-ID, DDP, initialization, mask, and method
  interpretation gaps.
- User approved the revised staged hardening plan.
- Sent SIGTERM only to P1 process group 3079663 after validating its command
  and working directory. Confirmed the process and GPU allocation exited.
- Preserved `launch.log` and `epoch_4.pth`; no formal run was started.
- Added the approved hardening design and task-specific planning records.
- Verified all six validation rows and the final epoch-7 iteration-220 row
  against the preserved launch log; wrote the `INVALID_DIAGNOSTIC` marker.
- Git commit initially failed because the managed sandbox exposes the shared
  worktree index as read-only; retry requires the standard scoped escalation.
- Object-identity RED: three focused tests failed on rank keys `[0,1]`,
  renumbered keys `[0,1,2]`, and flip IDs `[7.6,8.6,9.6]`.
- Object-identity GREEN: compaction now intersects true `bid.long()` keys and
  flip reconstruction restarts at one; `tests/test_gda_plan_b.py` reports
  `10 passed`.
- Numerical RED: the boundary test failed on non-finite Sigma for extreme raw
  `t`; the previously isolated exact-zero case also yielded NaN u2 gradients.
- Numerical GREEN: smooth `t` bounds and a canonical zero-radius u2 fallback
  give finite forward/backward values; the focused numerical plus Gate-A run
  reports `14 passed`.
- Reduction/mask RED: tests proved an empty intersection returned a detached
  tensor, the loss rejected an orientation-validity mask, and the distributed
  helpers/global diagnostics were absent. The first Gloo attempt was blocked
  by the managed sandbox's loopback policy; a scoped CPU-only rerun was used.
- Reduction/mask GREEN: empty views stay in autograd, rotation-agnostic objects
  are excluded, differentiable sums use DDP mean-count scaling, diagnostic
  numerators are globally reduced, and CUDA-path Python mask synchronizations
  were removed. Plan-B + CPU/Gloo reports `14 passed`; Plan-B alone reports
  `13 passed` with only an upstream MMCV import warning.
- Inference/init RED: all three B-T6 tests failed: ordinary eval executed the
  probe, no explicit analysis method existed, and same-seed shared weights
  differed.
- Inference/init GREEN: ordinary eval bypasses the probe, explicit
  `forward_gda_probe` emits analysis maps, probe construction preserves the RNG
  stream, and probe modules are excluded from parent `init_cfg` traversal until
  shared named overrides finish. B-T6 reports `3 passed`.
- Control-config RED: both contract tests failed because the single-GPU null
  control did not exist and the GDA config still inherited the batch-2 E0
  recipe directly.
- Control-config GREEN: the GPU4 null control and GDA arm both use global batch
  four, and their flattened configs differ only in `model` and `work_dir`;
  `tests/test_gda_control_contract.py` reports `2 passed`.
- Teacher-aggregation RED: real PSC tests showed the head stashed one decoded
  channel instead of the three encoded channels, and no decode-after-pooling
  path existed.
- Teacher-aggregation GREEN: the head now stashes detached PSC encodings,
  compaction averages those encodings per true object, and decoding happens
  afterward like the parent loss. Focused tests report `2 passed`; full Plan-B
  reports `17 passed`.
- Added a tiny end-to-end detector loss test covering three-view construction,
  target assignment, object compaction, auxiliary loss insertion, backward,
  and finite probe gradients.
- Corrected method documentation: this implementation is a coupled auxiliary
  regularizer, not a GDA inference head or direct main-loss denoiser; teacher
  agreement is not ground-truth chamber accuracy.
- Final focused suite: `33 passed, 3 warnings`.
- Related low-rank/evaluation regression suite: `35 passed, 1 warning`.
- Full repository suite initially had one sandbox-only Gloo socket failure
  (`173 passed` otherwise); scoped loopback rerun completed with
  `174 passed, 9 warnings`.
- `git diff --check` passed. Process search found no GDA P1 or `tools/train.py`
  process. A read-only GPU query briefly observed PID 3444809, which exited
  before process inspection and was not modified.
- No GPU smoke, formal training, queue, tmux, nohup, or resume was started.
