# Copyright (c) OpenMMLab. All rights reserved.
"""H2RBoxGDAHead: H2RBoxV2Head with a parallel GDA probe branch (Plan-B W0).

The probe predicts the KAK coordinates of the Gaussian symmetric-space
representation (see ``mmrotate/models/utils/gda_symmetric_space.py``) plus
a 2-logit chamber classifier, from the same FPN features:

    per-point probe channels = (t, a_raw, u2x, u2y, bit_logit0, bit_logit1)

with ``a = softplus(a_raw) >= 0`` and ``psi = atan2(u2y, u2x)`` (the u2
double-angle carrier, periodic by construction).  The covariance is
assembled downstream by ``tapsi_to_sigma``.

Design contract (pinned by tests/test_gda_plan_b.py):

* The baseline path is untouched: ``forward`` returns the parent's
  4-tuple unchanged, bit-for-bit; probe outputs are stashed on
  ``self.last_gda_probe`` (per-level tensors of shape (B, 6, H, W)) and
  consumed by ``OrbdetGDADetector._add_auxiliary_losses``.
* ``gda_probe.detach_feats=True`` detaches the FPN input of the probe
  tower, giving the ablation arm where no probe gradient can reach the
  shared backbone/neck.
* The probe never participates in inference; ``predict`` behaviour is
  inherited unchanged.
"""
from typing import List, Tuple

import torch
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmrotate.registry import MODELS
from .h2rbox_v2_head import H2RBoxV2Head

GDA_PROBE_CHANNELS = 6  # (t, a_raw, u2x, u2y, bit_logit0, bit_logit1)


@MODELS.register_module()
class H2RBoxGDAHead(H2RBoxV2Head):
    """H2RBoxV2Head with a parallel GDA (t, a, u2, chamber) probe branch.

    Args:
        gda_probe (dict, optional): Probe config.  Keys:
            ``enabled`` (bool, default True), ``detach_feats`` (bool,
            default False), ``loss`` (dict, OrbdetGDAProbeLoss config).
            ``None`` disables the probe entirely (exact parent behaviour).
    """

    def __init__(self, *args, gda_probe: dict = None, **kwargs):
        self.gda_cfg = {} if gda_probe is None else dict(gda_probe)
        self.gda_cfg.setdefault('enabled', gda_probe is not None)
        self.gda_cfg.setdefault('detach_feats', False)
        super().__init__(*args, **kwargs)
        if self.gda_cfg.get('enabled') and 'loss' in self.gda_cfg:
            self.loss_gda_probe = MODELS.build(self.gda_cfg['loss'])
        else:
            self.loss_gda_probe = None
        self._gda_probe_out: List[Tensor] = []
        self._gda_main_angle: List[Tensor] = []

    # -- construction ----------------------------------------------------

    def _init_layers(self):
        """Add the probe tower after the parent towers are built."""
        super()._init_layers()
        if not self.gda_cfg.get('enabled'):
            return
        # GN groups must divide the channel count (tiny test heads use
        # feat_channels < 32).
        num_groups = 32
        while self.feat_channels % num_groups:
            num_groups //= 2
        norm_cfg = dict(type='GN', num_groups=num_groups, requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.gda_probe_tower = nn.Sequential(
            ConvModule(
                self.in_channels,
                self.feat_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.feat_channels,
                self.feat_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.gda_probe_predictor = nn.Conv2d(
            self.feat_channels, GDA_PROBE_CHANNELS, 3, padding=1)

    def init_weights(self):
        super().init_weights()
        if not self.gda_cfg.get('enabled'):
            return
        for m in self.gda_probe_tower.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:  # ConvModule with norm -> bias=None
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.gda_probe_predictor.weight, mean=0, std=0.01)
        # bias prior: t=0; a_raw=-2 -> a=softplus(-2)=0.127 (aspect~1.14,
        # gate ~0.5 on the [1.1, 1.3) band edge); u2 ~ N(0, 0.01); bit=0.
        with torch.no_grad():
            bias = self.gda_probe_predictor.bias
            bias.zero_()
            bias[1] = -2.0

    # -- forward ---------------------------------------------------------

    @property
    def gda_enabled(self) -> bool:
        return bool(self.gda_cfg.get('enabled'))

    @property
    def last_gda_probe(self) -> List[Tensor]:
        return self._gda_probe_out

    @property
    def last_gda_main_angle(self) -> List[Tensor]:
        """Detached decoded baseline-head angle per level (train only)."""
        return self._gda_main_angle

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], ...]:
        self._gda_probe_out = []
        self._gda_main_angle = []
        return super().forward(x)

    def forward_single(self, x: Tensor, scale, stride) -> Tuple[Tensor, ...]:
        outs = super().forward_single(x, scale, stride)
        if self.gda_cfg.get('enabled'):
            feat = x.detach() if self.gda_cfg.get('detach_feats') else x
            probe = self.gda_probe_predictor(self.gda_probe_tower(feat))
            self._gda_probe_out.append(probe)
            if self.training:
                # Stash the baseline head's decoded angle (detached): the
                # per-instance chamber anchor for probe loss term 3a.
                # Decode is pointwise; reshape map <-> flat is exact.
                angle_pred = outs[2]
                n, e, hh, ww = angle_pred.shape
                ang = self.angle_coder.decode(
                    angle_pred.permute(0, 2, 3, 1).reshape(-1, e),
                    keepdim=True).reshape(n, hh, ww, 1).permute(0, 3, 1, 2)
                self._gda_main_angle.append(ang.detach())
        return outs
