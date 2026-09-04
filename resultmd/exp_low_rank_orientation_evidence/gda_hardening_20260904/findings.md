# GDA hardening findings

- Rank-based compaction is invalid when a non-tail object lacks positives in
  one view.
- Parent view IDs share integer identity and use `.2/.4/.6` suffixes.
- Flip reconstruction in the GDA subclass currently violates that contract.
- Exact-zero `u2` gives NaN angle gradients; extreme `t` overflows.
- P1 and E0 differ in world size and effective optimization schedule.
- Current P1 validation through epoch 6 is 0.0088, 0.0240, 0.0319, 0.0419,
  0.0578, 0.0756; this run is diagnostic-only.
- Independent review found all logged epoch-1-to-5 gradient-norm windows above
  the clip threshold, so loss descent is not evidence of harmless coupling.
- A future DDP repair must divide differentiable local sums by the global mean
  denominator to account for DDP's subsequent gradient averaging.
- Ordinary inference and explicit probe evidence collection need separate
  paths.
- True parent-style IDs are identical in their integer part across views;
  tensor `searchsorted` intersection preserves identities without host lists.
- Preserving RNG only around probe construction is insufficient: MMEngine's
  generic initializer visits extra probe convolutions before the named
  `conv_cls` override. Temporarily excluding probe modules during parent
  initialization is required for exact same-seed shared-state parity.
- PSC decode is nonlinear: with point angles 0.2 and 1.0 radians, averaging
  per-point `sin(2*decode)` gives 0.6494 while decoding the pooled PSC encoding
  gives 0.9320. The latter matches the parent object's compaction semantics.

## Final verification findings

- Focused GDA/Gate-A/control/DDP suite: 33 passed.
- Related low-rank/evaluation suite: 35 passed.
- Full repository test suite: 174 passed under CPU-only execution.
- Remaining warnings are pre-existing upstream/parent warnings: MMCV import
  relocation, PyTorch meshgrid indexing, parent H2RBox `index_reduce_`, and
  MMCV `file_client_args` deprecation.
- `git diff --check` is clean.
- No GDA training process remains.
- No GPU memory smoke, gradient-calibration experiment, or new formal training
  was run. Batch-four memory fit and useful auxiliary weights remain unproven.
