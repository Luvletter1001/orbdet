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
* Ordinary inference skips the probe entirely; analysis code must call
  ``forward_gda_probe`` explicitly.
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
        # Module constructors initialize parameters immediately. Preserve the
        # global RNG stream so adding the probe does not change the baseline
        # head's later init_weights result under the same seed.
        with torch.random.fork_rng(devices=[]):
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
        if not self.gda_cfg.get('enabled'):
            return super().init_weights()
        # BaseModule's generic Conv initializer traverses every registered
        # child before applying named overrides such as ``conv_cls``. If the
        # probe remains registered, its random draws shift those baseline
        # overrides even under the same seed. Temporarily exclude only the
        # probe modules, initialize the untouched parent graph, then restore
        # and initialize the probe from the subsequent RNG stream.
        tower = self._modules.pop('gda_probe_tower')
        predictor = self._modules.pop('gda_probe_predictor')
        try:
            super().init_weights()
        finally:
            self.add_module('gda_probe_tower', tower)
            self.add_module('gda_probe_predictor', predictor)
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
        """Detached baseline-head angle encoding per level (train only).

        The detector mean-pools these encodings per physical object before
        decoding, exactly matching the parent self-supervision path.
        """
        return self._gda_main_angle

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], ...]:
        self._gda_probe_out = []
        self._gda_main_angle = []
        return super().forward(x)

    def _forward_gda_probe_single(self, x: Tensor) -> Tensor:
        feat = x.detach() if self.gda_cfg.get('detach_feats') else x
        return self.gda_probe_predictor(self.gda_probe_tower(feat))

    def forward_gda_probe(self, x: Tuple[Tensor]) -> List[Tensor]:
        """Explicitly emit probe maps for analysis outside ordinary predict."""
        if not self.gda_enabled:
            self._gda_probe_out = []
            return []
        self._gda_probe_out = [
            self._forward_gda_probe_single(feat) for feat in x
        ]
        return self._gda_probe_out

    def forward_single(self, x: Tensor, scale, stride) -> Tuple[Tensor, ...]:
        outs = super().forward_single(x, scale, stride)
        if self.gda_cfg.get('enabled') and self.training:
            probe = self._forward_gda_probe_single(x)
            self._gda_probe_out.append(probe)
            # Preserve the parent's semantics: compact encoded point outputs
            # by object first, then decode the compacted encoding.
            self._gda_main_angle.append(outs[2].detach())
        return outs
