# Low-Rank Channel-Consistent Orientation Evidence

## Goal

Add a small, model-agnostic PyTorch module that extracts an orientation and a
confidence signal from an instance ROI feature map.  It is a diagnostic-only
component: it must not modify detector predictions, losses, NMS, data, or
formal-training status.

## Problem

The existing GODC experiment constructed a C2 orbit by rotating an already
extracted FPN ROI and penalized rank and fixed-space residuals.  That makes the
ROI invariant to a spatial rotation, so it can remove the orientation evidence
needed by an OBB detector.  The replacement must use low rank to *measure*
cross-channel agreement on a visual axis, not to force invariance.

## Chosen mechanism

For one ROI feature tensor `F` of shape `[N, C, H, W]`, compute central
finite-difference gradients `gx, gy`.  A fixed two-dimensional Hann support
window gives each channel an axial evidence vector

`m_c = [sum w (gx^2 - gy^2), sum w (2 gx gy)]`.

Stacking the vectors yields `M` of shape `[N, C, 2]`.  Its right singular
vector associated with the largest singular value is the visual axial phase:

`theta = 0.5 * atan2(v_y, v_x)`.

The following detached diagnostics are reported per instance:

- `confidence = sigma1 / (sigma1 + sigma2 + eps)`, measuring channelwise
  rank-one agreement;
- `anisotropy = norm(mean_c(m_c)) / (mean_c(norm(m_c)) + eps)`;
- `energy = mean(gx^2 + gy^2)`;
- `valid`, requiring energy and anisotropy floors before the angle can be
  interpreted.

The sign of a singular vector is fixed by its dot product with the mean axial
vector; this avoids an arbitrary 180-degree flip while preserving the intended
pi-periodic angle.

## Scope boundary

This first increment exposes a pure functional module only.  It accepts
features directly and has no ROI extraction, detector hook, loss weight,
training configuration, checkpoint, or evaluation-launcher change.  A later
approved design may attach it to an HBox ROI and test whether its diagnostics
predict angle error.

## Acceptance tests

1. Synthetic horizontal and vertical stripe features decode axes near 0 and
   pi/2 modulo pi.
2. Rotating a synthetic stripe by 45 degrees rotates the decoded axis by 45
   degrees modulo pi.
3. Multi-channel copies with arbitrary positive amplitudes retain high
   low-rank confidence.
4. Orthogonal channel populations lower confidence or anisotropy.
5. Constant features are finite, low energy, and invalid rather than producing
   a claimed orientation.
6. The returned tensors preserve gradient connectivity, while diagnostics can
   be detached by the caller.

## Non-claims

This module does not establish a new paper contribution, prove a reduction in
P50/P90, or reproduce any external method.  Its novelty relative to structure
tensors, p-atic evidence, and low-rank feature analysis requires a dedicated
literature check before being claimed.
