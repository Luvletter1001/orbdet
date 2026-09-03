# Copyright (c) OpenMMLab. All rights reserved.
"""GDA probe losses (Plan-B W1).

Pure tensor math on per-object compacted probe rows.  The detector
(:class:`OrbdetGDADetector`) does view slicing, target assignment and
object compaction; this module only sees the resulting per-object tensors.

Three terms, all acting solely on the probe branch outputs:

1. Envelope regression (every view): the HBox-derived envelope supervises
   the V4 invariants of the predicted Sigma -- gauge-equivalent angle
   encodings are never penalized (Gate-A test A7/A9).
2. Cross-view envelope consistency: ``rotate_sigma``/``reflect_sigma``
   map the ori-view Sigma into the rotated/flipped frames; consistency is
   measured on envelopes only (V4-invariant ring).
3. Chamber bit classification:
   * GT-anchored: only where the GT encoding carries a reliable bit
     (|sin 2*theta_gt| >= bit_min_abs_sin, which automatically excludes
     the horizontal/vertical HBox views) AND softly gated by the detached
     predicted anisotropy a (near-square instances get weight ~0 --
     pre-registered B1/B2 motivation, no learnable confidence channel);
   * cross-view equivariant: targets generated from the *detached*
     ori-view prediction through the known view transform
     (psi -> psi + 2*rot for rotation, psi -> -psi for reflection),
     preventing self-reinforcement.

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
        bit_min_abs_sin (float): minimum |sin 2*theta_gt| for the
            GT-anchored chamber target to be considered reliable.
    """

    def __init__(self,
                 a0: float = math.log(1.15),
                 tau: float = 0.10,
                 w_env: float = 1.0,
                 w_xview: float = 0.5,
                 w_bit_gt: float = 0.2,
                 w_bit_x: float = 0.1,
                 beta: float = 0.05,
                 bit_min_abs_sin: float = 0.3):
        super().__init__()
        self.a0 = float(a0)
        self.tau = float(tau)
        self.w_env = float(w_env)
        self.w_xview = float(w_xview)
        self.w_bit_gt = float(w_bit_gt)
        self.w_bit_x = float(w_bit_x)
        self.beta = float(beta)
        self.bit_min_abs_sin = float(bit_min_abs_sin)

    # -- representation helpers -----------------------------------------

    @staticmethod
    def rows_to_sigma(rows: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Probe rows (..., 6) -> (Sigma (...,2,2), a (...,), psi (...,)).

        ``a_raw`` is clamped to [-8, 8] before the softplus: entry-wise
        float32 assembly of Sigma loses SPD through cosh/sinh
        cancellation for a >~ 8 (a = 8 already means an absurd aspect
        ratio of e^8 ~ 2980).  Gradients vanish beyond the clamp, which
        is the intended safety behaviour.
        """
        t = rows[..., 0]
        a = F.softplus(rows[..., 1].clamp(-8.0, 8.0))
        psi = torch.atan2(rows[..., 3], rows[..., 2])
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

    def forward(self, rows3: Tensor, env3: Tensor, sin2_gt3: Tensor,
                rot: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute probe losses.

        Args:
            rows3 (Tensor): (M, 3, 6) probe rows, view dim order
                (ori, rot, flp).
            env3 (Tensor): (M, 3, 2) GT-derived envelope targets (W, H)
                per view.
            sin2_gt3 (Tensor): (M, 3) sin(2*theta_gt) per view.
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
                gda_nobj=torch.zeros((), device=device))

        rows_ori, rows_rot, rows_flp = rows3[:, 0], rows3[:, 1], rows3[:, 2]
        sig_ori, a_ori, psi_ori = self.rows_to_sigma(rows_ori)
        sig_rot, _, _ = self.rows_to_sigma(rows_rot)
        sig_flp, _, _ = self.rows_to_sigma(rows_flp)

        # 1. envelope regression against GT-derived envelopes, all views --
        #    V4-invariant: gauge-equivalent encodings penalized identically.
        loss_env = (envelope_consistency_loss(sig_ori, env3[:, 0], self.beta)
                    + envelope_consistency_loss(sig_rot, env3[:, 1], self.beta)
                    + envelope_consistency_loss(sig_flp, env3[:, 2],
                                                self.beta)).mean() / 3.0

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

        # 3a. GT-anchored bit: only where the GT encoding is reliable
        #     (|sin 2*theta_gt| large enough) -- HBox views (theta_gt = 0)
        #     are automatically masked out.
        mask_gt = sin2_gt3[:, 1].abs() >= self.bit_min_abs_sin
        if mask_gt.any():
            tgt_gt = (sin2_gt3[mask_gt, 1] > 0).long()
            ce = F.cross_entropy(logits_rot[mask_gt], tgt_gt,
                                 reduction='none')
            loss_bit_gt = (g[mask_gt] * ce).sum() / mask_gt.sum().clamp_min(
                1).to(ce.dtype)
            with torch.no_grad():
                pred = logits_rot[mask_gt].argmax(dim=-1)
                bit_acc = (pred == tgt_gt).to(ce.dtype).mean()
        else:
            loss_bit_gt = zero
            bit_acc = zero.detach()

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
            gda_nobj=torch.tensor(float(m), device=device).detach())
        return losses, diagnostics
