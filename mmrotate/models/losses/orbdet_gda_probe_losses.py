# Copyright (c) OpenMMLab. All rights reserved.
"""GDA probe losses (Plan-B W1).

Pure tensor math on per-object compacted probe rows.  The detector
(:class:`OrbdetGDADetector`) does view slicing, target assignment and
object compaction; this module only sees the resulting per-object tensors.

Three terms, all acting solely on the probe branch outputs:

1. Envelope regression (tight views only: ori + flp): the HBox-derived
   envelope supervises the V4 invariants of the predicted Sigma --
   gauge-equivalent angle encodings are never penalized (Gate-A test
   A7/A9).  The rot view's gt is a rotated HBox (a *loose* OBB whose
   envelope is systematically too large), so it is excluded from
   regression and only joins cross-view consistency.
2. Cross-view envelope consistency: ``rotate_sigma``/``reflect_sigma``
   map the ori-view Sigma into the rotated/flipped frames; consistency is
   measured on envelopes only (V4-invariant ring).
3. Chamber bit classification (softly gated by the detached predicted
   anisotropy a -- near-square instances get small weight, the
   pre-registered B1/B2 motivation; no learnable confidence channel):
   * main-head-anchored: per-instance target = chamber bit of the
     *detached* baseline-head angle in the same view (empirically
     correct on elongated instances; cross-head distillation cannot
     self-reinforce).  Masked where |sin 2*theta_main| is small (the
     main head's own orbit-boundary unreliability zone).
   * cross-view equivariant: targets generated from the *detached*
     ori-view prediction through the known view transform
     (psi -> psi + 2*rot for rotation, psi -> -psi for reflection).

v1.1 note: an earlier GT-encoded chamber target (sign of sin(2*theta_gt)
in the rot view) was dropped after the P0c smoke root-caused it as
image-uniform under HBox supervision (theta_gt = rot for every object).

Every returned loss is a scalar tensor; diagnostics carry no gradient and
their names never contain 'loss' (logger-only, mmdet convention).
"""
import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmrotate.models.utils.gda_symmetric_space import (
    envelope_consistency_loss, envelope_from_sigma, reflect_sigma,
    rotate_sigma, tapsi_to_sigma)
from mmrotate.registry import MODELS


def _smooth_l1(x: Tensor, beta: float) -> Tensor:
    ax = x.abs()
    return torch.where(ax < beta, 0.5 * ax * ax / beta, ax - 0.5 * beta)


@MODELS.register_module()
class OrbdetGDAProbeLoss(nn.Module):
    """GDA probe objective.

    Args:
        a0 (float): Gate center on the detached anisotropy a.
            Default log(1.15) (B1 operating band edge).
        tau (float): Gate temperature.  Default 0.10.
        w_env, w_xview, w_bit_gt, w_bit_x (float): term weights.
        beta (float): Smooth-L1 beta for envelope terms.
        bit_min_abs_sin (float): minimum |sin 2*theta_main| for the detached
            main-head chamber target to be considered reliable.
        t_min, t_max (float): smooth bounds for the log-scale coordinate.
        u2_min_radius (float): radius below which the undefined double-angle
            carrier uses a finite canonical fallback.
    """

    def __init__(self,
                 a0: float = math.log(1.15),
                 tau: float = 0.10,
                 w_env: float = 1.0,
                 w_xview: float = 0.5,
                 w_bit_gt: float = 0.2,
                 w_bit_x: float = 0.1,
                 beta: float = 0.05,
                 bit_min_abs_sin: float = 0.3,
                 t_min: float = -6.0,
                 t_max: float = 14.0,
                 u2_min_radius: float = 1e-4):
        super().__init__()
        if not t_min < 0 < t_max:
            raise ValueError('t_min and t_max must straddle zero')
        if u2_min_radius <= 0:
            raise ValueError('u2_min_radius must be positive')
        self.a0 = float(a0)
        self.tau = float(tau)
        self.w_env = float(w_env)
        self.w_xview = float(w_xview)
        self.w_bit_gt = float(w_bit_gt)
        self.w_bit_x = float(w_bit_x)
        self.beta = float(beta)
        self.bit_min_abs_sin = float(bit_min_abs_sin)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.u2_min_radius = float(u2_min_radius)

    # -- representation helpers -----------------------------------------

    def rows_to_sigma(self, rows: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Probe rows (..., 6) -> (Sigma (...,2,2), a (...,), psi (...,)).

        ``a_raw`` is clamped to [-8, 8] before the softplus: entry-wise
        float32 assembly of Sigma loses SPD through cosh/sinh
        cancellation for a >~ 8 (a = 8 already means an absurd aspect
        ratio of e^8 ~ 2980).  Gradients vanish beyond the clamp, which
        is the intended safety behaviour.
        """
        raw_t = rows[..., 0]
        # Legal boxes in a 1024-pixel frame occupy roughly
        # log(w*h/12) in [-2.5, 11.4].  A smooth asymmetric bound with margin
        # preserves unit slope at zero while preventing exp overflow.
        t = torch.where(
            raw_t >= 0,
            self.t_max * torch.tanh(raw_t / self.t_max),
            (-self.t_min) * torch.tanh(raw_t / (-self.t_min)))
        a = F.softplus(rows[..., 1].clamp(-8.0, 8.0))
        u2x, u2y = rows[..., 2], rows[..., 3]
        radius_sq = u2x.square() + u2y.square()
        valid = radius_sq >= self.u2_min_radius ** 2
        # atan2(0, 0) has NaN derivatives.  The angle is undefined in this
        # region, so use a canonical finite direction with zero local angle
        # gradient until the probe leaves the safety ball.
        safe_x = torch.where(valid, u2x, torch.ones_like(u2x))
        safe_y = torch.where(valid, u2y, torch.zeros_like(u2y))
        psi = torch.atan2(safe_y, safe_x)
        return tapsi_to_sigma(t, a, psi), a, psi

    def gate(self, a_detached: Tensor) -> Tensor:
        """Soft gate in [0, 1] from *detached* anisotropy (no gradient)."""
        return torch.sigmoid((a_detached - self.a0) / self.tau)

    def chamber_targets_xview(self, psi_ori_detached: Tensor,
                              rot: Tensor) -> Tuple[Tensor, Tensor]:
        """Equivariant chamber targets from the detached ori prediction.

        Rotation by ``rot`` maps psi -> psi + 2*rot; vertical reflection
        maps psi -> -psi.  The chamber bit is sign(sin psi).
        Returns long class indices in {0, 1} for the rot and flp views.
        """
        s_rot = torch.sin(psi_ori_detached + 2.0 * rot)
        s_flp = -torch.sin(psi_ori_detached)
        return (s_rot > 0).long(), (s_flp > 0).long()

    # -- main objective --------------------------------------------------

    def forward(self, rows3: Tensor, env3: Tensor, sin2_main3: Tensor,
                rot: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute probe losses.

        Args:
            rows3 (Tensor): (M, 3, 6) probe rows, view dim order
                (ori, rot, flp).
            env3 (Tensor): (M, 3, 2) GT-derived envelope targets (W, H)
                per view.  Only views 0 (ori) and 2 (flp) are used for
                regression -- their HBox gt is tight; the rot view's gt
                is a rotated HBox (a *loose* OBB), so the rot view only
                participates in cross-view consistency (term 2).
            sin2_main3 (Tensor): (M, 3) sin(2*theta_main) per view, from
                the *detached* baseline-head angle -- the per-instance
                chamber anchor (empirically reliable for elongated
                instances, see the Gate-B evidence).
            rot (Tensor): scalar view rotation of the rot view.

        Returns:
            tuple[dict, dict]: scalar losses (keys contain 'loss') and
            detached diagnostics (keys never contain 'loss').
        """
        m = rows3.size(0)
        device = rows3.device
        zero = rows3.sum() * 0.0
        if m == 0:
            losses = dict(
                gda_loss_env=zero,
                gda_loss_xview=zero,
                gda_loss_bit=zero)
            return losses, dict(
                gda_gate_mean=zero.detach(), gda_bit_acc=zero.detach(),
                gda_nobj=torch.zeros((), device=device),
                gda_raw_t_min=zero.detach(), gda_raw_t_max=zero.detach(),
                gda_u2_radius_mean=zero.detach(),
                gda_u2_fallback_frac=zero.detach())

        rows_ori, rows_rot, rows_flp = rows3[:, 0], rows3[:, 1], rows3[:, 2]
        sig_ori, a_ori, psi_ori = self.rows_to_sigma(rows_ori)
        sig_rot, _, _ = self.rows_to_sigma(rows_rot)
        sig_flp, _, _ = self.rows_to_sigma(rows_flp)

        # 1. envelope regression against GT-derived envelopes, tight views
        #    only (ori + flp; the rot view's rotated-HBox gt is loose).
        #    V4-invariant: gauge-equivalent encodings penalized identically.
        loss_env = (envelope_consistency_loss(sig_ori, env3[:, 0], self.beta)
                    + envelope_consistency_loss(sig_flp, env3[:, 2],
                                                self.beta)).mean() / 2.0

        # 2. cross-view envelope consistency on the V4-invariant ring.
        w_env, h_env = envelope_from_sigma(rotate_sigma(sig_ori, rot))
        w_tgt, h_tgt = envelope_from_sigma(sig_rot)
        loss_x_rot = (_smooth_l1(w_env - w_tgt, self.beta)
                      + _smooth_l1(h_env - h_tgt, self.beta)).mean()
        w_env, h_env = envelope_from_sigma(reflect_sigma(sig_ori))
        w_tgt, h_tgt = envelope_from_sigma(sig_flp)
        loss_x_flp = (_smooth_l1(w_env - w_tgt, self.beta)
                      + _smooth_l1(h_env - h_tgt, self.beta)).mean()
        loss_xview = (loss_x_rot + loss_x_flp) / 2.0

        # 3. chamber bit.  Soft gate from detached anisotropy.
        g = self.gate(a_ori.detach())  # (M,), no gradient by construction
        logits_ori = rows_ori[:, 4:6]
        logits_rot = rows_rot[:, 4:6]
        logits_flp = rows_flp[:, 4:6]

        # 3a. main-head-anchored bit: per-instance target from the
        #     detached baseline-head angle (the baseline is empirically
        #     correct on elongated instances; Gate-B B2), in every view.
        #     Masked where the main head itself sits near the orbit
        #     boundary (|sin 2*theta_main| too small to be reliable).
        loss_bit_gt = zero
        bit_acc = zero.detach()
        n_bit = 0
        for v, logits_v in enumerate((logits_ori, logits_rot, logits_flp)):
            s2 = sin2_main3[:, v]
            mask_v = s2.abs() >= self.bit_min_abs_sin
            if not mask_v.any():
                continue
            tgt_v = (s2[mask_v] > 0).long()
            ce_v = F.cross_entropy(logits_v[mask_v], tgt_v,
                                   reduction='none')
            loss_bit_gt = loss_bit_gt + (g[mask_v] * ce_v).sum()
            n_bit = n_bit + int(mask_v.sum())
            with torch.no_grad():
                bit_acc = bit_acc + (
                    logits_v[mask_v].argmax(dim=-1) == tgt_v).to(
                        ce_v.dtype).sum()
        if n_bit > 0:
            loss_bit_gt = loss_bit_gt / n_bit
            bit_acc = bit_acc / n_bit

        # 3b. cross-view equivariant bit (targets from detached ori pred).
        tgt_rot, tgt_flp = self.chamber_targets_xview(psi_ori.detach(), rot)
        ce_rot = F.cross_entropy(logits_rot, tgt_rot, reduction='none')
        ce_flp = F.cross_entropy(logits_flp, tgt_flp, reduction='none')
        loss_bit_x = ((g * ce_rot).mean() + (g * ce_flp).mean()) / 2.0

        losses = dict(
            gda_loss_env=self.w_env * loss_env,
            gda_loss_xview=self.w_xview * loss_xview,
            gda_loss_bit=self.w_bit_gt * loss_bit_gt
            + self.w_bit_x * loss_bit_x)
        diagnostics = dict(
            gda_gate_mean=g.mean().detach(),
            gda_bit_acc=bit_acc.detach(),
            gda_nobj=torch.tensor(float(m), device=device).detach(),
            gda_raw_t_min=rows3[..., 0].amin().detach(),
            gda_raw_t_max=rows3[..., 0].amax().detach(),
            gda_u2_radius_mean=torch.linalg.vector_norm(
                rows3[..., 2:4], dim=-1).mean().detach(),
            gda_u2_fallback_frac=(
                torch.linalg.vector_norm(rows3[..., 2:4], dim=-1)
                < self.u2_min_radius).to(rows3.dtype).mean().detach())
        return losses, diagnostics
