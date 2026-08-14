# Group-Orbit Determinantal Cluster Core Design

## Status

Approved in conversation on 2026-08-15. This specification covers the
mathematical core and tests only. It does not authorize formal training.

## Context

Orbdet-v0.2 is an MMRotate/H2RBox-v2 project. Its current symmetry component
operates on predicted angles from original, rotated, and vertically flipped
views. It does not construct instance feature orbits, estimate a stabilizer
group, or impose a rank constraint on an orbit matrix.

The approved first stage adds a reusable, registered loss and exact planar
group-orbit builder. It remains disconnected from the default v0.2 training
objective so mathematical correctness can be tested independently of ROI
selection and loss-weight tuning.

## Objectives

1. Build exact finite planar group orbits for feature tensors using grid-free
   `torch.rot90` and `torch.flip` operations.
2. Measure approximate rank-one structure through a small Gram matrix.
3. Distinguish rank-one collinearity from strict group invariance.
4. Expose detached stability diagnostics without self-weighting the loss.
5. Reject zero-energy, constant-feature, empty-support, and identity-group
   shortcuts.
6. Keep all operations differentiable and safe for mixed-precision features.

## Non-goals

- No automatic ROI extraction from HBoxes in this stage.
- No learned group classifier or orientation head.
- No modification of Orbdet-v0.2 configs, detector prediction, NMS, or scores.
- No formal or smoke training launch.
- No claim that generic transformation-plus-low-rank modeling is new.

## Files

- Create `mmrotate/models/losses/group_orbit_determinantal_cluster_loss.py`.
- Export it from `mmrotate/models/losses/__init__.py`.
- Create `tests/test_group_orbit_determinantal_cluster.py`.
- Preserve the research record in
  `docs/future_work/inspirations/2026-08-15-group-orbit-determinantal-cluster.md`.

## Planar group-orbit builder

The public utility

```python
build_planar_group_orbit(features: Tensor, group: str) -> Tensor
```

accepts `[N, C, H, W]` and returns `[N, K, C, H, W]`. Supported groups are:

| name | elements | order |
|---|---|---:|
| `c2` | identity, 180-degree rotation | 2 |
| `c4` | identity, 90/180/270-degree rotations | 4 |
| `d1` | identity, horizontal reflection | 2 |
| `d2` | identity, 180-degree rotation, horizontal and vertical reflections | 4 |

`c4` requires square spatial support because exact 90-degree rotations of a
rectangular tensor change its shape. The builder rejects unknown groups,
non-4D input, zero spatial dimensions, and the trivial group `c1`.

This exact builder is intentionally limited. Arbitrary-axis reflection and
continuous rotation require interpolation and belong in the later ROI adapter.

## Loss contract

`GroupOrbitDeterminantalClusterLoss` is registered in `MODELS`. Its forward
method accepts a precomputed orbit `[N, K, ...]`, an optional broadcastable
support mask, optional sample weights, `avg_factor`, and the standard MMEngine
reduction override.

After masking, every orbit member is flattened to obtain
$O\in\mathbb{R}^{K\times D}$ and

$$
G=OO^\top\in\mathbb{R}^{K\times K}.
$$

Gram construction and eigendecomposition run in float32 when the feature
dtype is float16 or bfloat16. The result remains connected to the input
autograd graph.

### Determinantal component

$$
L_{\wedge^2}=
\frac{(\operatorname{tr}G)^2-\operatorname{tr}(G^2)}
{2(\operatorname{tr}G)^2+\varepsilon}.
$$

This is zero, outside the zero-energy degeneracy, exactly when the orbit matrix
has rank at most one.

### Spectral-tail diagnostic/component

$$
L_{\mathrm{tail}}=1-
\frac{\lambda_{\max}(G)}{\operatorname{tr}G+\varepsilon}.
$$

Its weight is configurable. It is also recorded as a detached diagnostic.

### Fixed-space component

$$
L_{\mathrm{fix}}=
\frac{\sum_h\lVert x_h-\bar{x}\rVert^2}
{\sum_h\lVert x_h\rVert^2+\varepsilon}.
$$

This rejects rank-one orbits whose members differ only by scale. It is kept
separate from the determinantal component for ablation.

### Degeneracy guards

Per-sample mean-square energy and spatial/feature variance are compared with
configurable positive floors:

$$
L_{\mathrm{guard}}=
\max(0,e_{min}-e)+\max(0,v_{min}-v).
$$

An empty support therefore produces a finite positive penalty instead of a
false perfect symmetry score. Constant nonzero features trigger the variance
guard. The loss requires `K >= 2`, so an identity-only orbit cannot win.

### Stability diagnostic

For descending singular values computed from the Gram eigenvalues,

$$
q_{\mathrm{gap}}=
\frac{\sigma_1-\sigma_2}{\sigma_1+\varepsilon}.
$$

The module records detached batch means for the determinantal, tail,
fixed-space, energy-guard, variance-guard, and gap quantities. None of these
diagnostics reweights its own objective.

## Reduction and empty batches

- `none` returns one value per sample.
- `sum` sums weighted samples.
- `mean` follows MMEngine-style weighted averaging and honors `avg_factor`.
- A genuinely empty batch returns a graph-connected scalar zero for `mean` or
  `sum`, and an empty vector for `none`; diagnostics are reset to finite zeros.
- Non-finite input is rejected with a clear `ValueError` rather than silently
  contaminating training.

## Test-first acceptance criteria

Tests are written and observed failing before the production module exists.
The final test suite must establish:

1. Exact element order and values for `c2`, `c4`, `d1`, and `d2`.
2. A constructed invariant orbit has near-zero determinantal, spectral-tail,
   and fixed-space terms.
3. A generic asymmetric feature has a larger objective than its symmetric
   counterpart under the same nontrivial group.
4. A scaled rank-one orbit has zero determinantal term but positive
   fixed-space residual.
5. Zero, constant, and empty-support inputs receive finite guard penalties.
6. Mask broadcasting, reductions, and sample weights behave as documented.
7. Float32 gradients are finite and nonzero for a non-invariant orbit.
8. Float16/bfloat16 Gram computation does not introduce NaNs where supported.
9. The class can be built from the MMRotate registry.
10. Existing Orbdet-v0.1 and v0.2 unit tests still pass.

## Later integration boundary

The next stage may add an HBox ROI adapter that samples FPN features, evaluates
multiple nontrivial group hypotheses, and emits quotient-valued orientation
evidence. That stage needs a separate design because support selection, group
assignment, and loss weighting each introduce an independent collapse mode.

## Safety

No dataset file, checkpoint, work directory, or formal-training status marker
is modified. No GPU process or training launcher is started. Git ignores local
data, weights, work directories, logs, bytecode, and environment recovery
snapshots.
