# Copyright (c) OpenMMLab. All rights reserved.
"""Plan-B tests for the GDA probe head / loss / detector wiring.

B-T1  probe rows -> Sigma is SPD and all probe losses are finite,
      including extreme a_raw inputs (numerical safety)
B-T2  loss-level equivariance: rotating the view by phi and the target
      by the same phi leaves the envelope loss invariant
B-T3  gauge zero-penalty: V4-equivalent encodings of the same rectangle
      give identical envelope losses
B-T4  gate: sigmoid((a-a0)/tau) is centered at a0, monotone, and carries
      no gradient (built from a detached input)
B-T5  chamber bit: cross-view targets follow psi -> psi + 2*rot /
      psi -> -psi, are gradient-free, and the bit loss backpropagates
      only into the bit logits (not into u2)
B-T6  head parity: probe-disabled head is bit-identical to the parent;
      probe-enabled head returns identical baseline outputs; with
      detach_feats=True a probe-only backward touches no parent weight
B-T7  compact_probe_by_object: cross-view grouping by bid *rank* (bid
      integers differ across views), mean-pooling, bcnt==3 filtering
"""
import math

import torch

from mmrotate.models.utils.gda_symmetric_space import (
    envelope_consistency_loss, envelope_from_sigma, envelope_from_wh_theta,
    rotate_sigma, wh_theta_to_sigma)


# ---------------------------------------------------------------- B-T1

def test_b_t1_sigma_spd_and_losses_finite():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    torch.manual_seed(0)
    loss_mod = OrbdetGDAProbeLoss()
    rows = torch.randn(7, 3, 6)
    rows[..., 1] = torch.tensor([-30., -5., -2., 0., 1., 5., 30.])[:, None]
    sig, a, psi = loss_mod.rows_to_sigma(rows.reshape(-1, 6))
    eig = torch.linalg.eigvalsh(sig)
    assert (eig > 0).all(), 'Sigma must be SPD even for extreme a_raw'
    assert torch.isfinite(sig).all()
    # the safety clamp activates: a_raw = +-30 lands inside softplus([-8, 8])
    # (softplus(8) = 8.0003 > 8, hence the 8.01 tolerance)
    assert (a <= 8.01).all() and (a > 0).all()
    env3 = torch.rand(7, 3, 2) * 100 + 1
    sin2_3 = torch.rand(7, 3) * 2 - 1
    rot = torch.tensor(0.7)
    losses, diag = loss_mod(rows, env3, sin2_3, rot)
    for v in losses.values():
        assert torch.isfinite(v).all()
    for v in diag.values():
        assert torch.isfinite(v).all()
    # empty-batch contract
    losses0, _ = loss_mod(rows[:0], env3[:0], sin2_3[:0], rot)
    for v in losses0.values():
        assert v.item() == 0.0 and torch.isfinite(v).all()


def test_b_t1_zero_u2_and_extreme_t_have_finite_forward_and_backward():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    loss_mod = OrbdetGDAProbeLoss()
    raw_t = torch.tensor([-100., -20., 0., 20., 100.])
    rows = torch.zeros(5, 3, 6)
    rows[..., 0] = raw_t[:, None]
    rows[..., 1] = -2.0
    # Cover exact zero and radii on both sides of the intended 1e-4 safety
    # boundary.  The old implementation evaluates atan2(0, 0).
    rows[:, :, 2] = torch.tensor(
        [0., 5e-5, 2e-4, -5e-5, -2e-4])[:, None]
    rows = rows.requires_grad_(True)
    sigma, _, _ = loss_mod.rows_to_sigma(rows.reshape(-1, 6))
    env3 = torch.full((5, 3, 2), 20.0)
    sin2_main3 = torch.ones(5, 3)
    losses, diagnostics = loss_mod(
        rows, env3, sin2_main3, torch.tensor(0.5))

    assert torch.isfinite(sigma).all()
    assert all(torch.isfinite(value).all() for value in losses.values())
    for name in ('gda_raw_t_min', 'gda_raw_t_max',
                 'gda_u2_radius_mean', 'gda_u2_fallback_frac'):
        assert name in diagnostics
        assert torch.isfinite(diagnostics[name]).all()

    sum(losses.values()).backward()
    assert rows.grad is not None
    assert torch.isfinite(rows.grad).all()


# ---------------------------------------------------------------- B-T2

def test_b_t2_envelope_loss_view_rotation_equivariant():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    torch.manual_seed(1)
    loss_mod = OrbdetGDAProbeLoss()
    for _ in range(20):
        w = float(torch.rand(()) * 50 + 5)
        h = float(torch.rand(()) * 40 + 1)
        th = float(torch.rand(()) * math.pi - math.pi / 2)
        phi = float(torch.rand(()) * math.pi)
        sig = wh_theta_to_sigma(torch.tensor(w), torch.tensor(h),
                                torch.tensor(th))
        env_ref = envelope_from_wh_theta(torch.tensor(w), torch.tensor(h),
                                         torch.tensor(th + phi))
        env_rot = envelope_from_sigma(rotate_sigma(sig, torch.tensor(phi)))
        assert torch.allclose(env_rot[0], env_ref[0], atol=1e-4)
        assert torch.allclose(env_rot[1], env_ref[1], atol=1e-4)
        # and the consistency loss is invariant under joint rotation
        env_t = torch.stack([env_ref[0], env_ref[1]])
        l_rot = envelope_consistency_loss(rotate_sigma(sig, torch.tensor(phi)),
                                          env_t)
        env_o = envelope_from_wh_theta(torch.tensor(w), torch.tensor(h),
                                       torch.tensor(th))
        l_ori = envelope_consistency_loss(
            sig, torch.stack([env_o[0], env_o[1]]))
        assert torch.allclose(l_rot, l_ori, atol=1e-4)


# ---------------------------------------------------------------- B-T3

def test_b_t3_gauge_equivalent_encodings_same_loss():
    torch.manual_seed(2)
    env_t = torch.tensor([30.0, 12.0])
    base = dict(w=23.0, h=9.0, theta=0.35)
    variants = [
        (base['w'], base['h'], base['theta']),
        (base['h'], base['w'], base['theta'] + math.pi / 2),
        (base['w'], base['h'], base['theta'] + math.pi),
    ]
    losses = []
    for w, h, th in variants:
        sig = wh_theta_to_sigma(torch.tensor(w), torch.tensor(h),
                                torch.tensor(th))
        losses.append(envelope_consistency_loss(sig, env_t))
    assert torch.allclose(losses[0], losses[1], atol=1e-5)
    assert torch.allclose(losses[0], losses[2], atol=1e-5)


# ---------------------------------------------------------------- B-T4

def test_b_t4_gate_centered_monotone_gradfree():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    loss_mod = OrbdetGDAProbeLoss()
    a = torch.linspace(-1, 3, 50, requires_grad=True)
    g = loss_mod.gate(a.detach())
    assert not g.requires_grad, 'gate must be gradient-free'
    assert torch.all(g[1:] >= g[:-1]), 'gate must be non-decreasing'
    # strictly increasing where it matters (float32 sigmoid saturates
    # to exactly 1.0 in the far tail, which is fine)
    mid = (a - loss_mod.a0).abs() < 0.3
    gm = g[mid]
    assert torch.all(gm[1:] > gm[:-1])
    g_at_a0 = loss_mod.gate(torch.tensor(loss_mod.a0))
    assert abs(g_at_a0.item() - 0.5) < 1e-6
    assert g[0].item() < 0.25 and g[-1].item() > 0.99


# ---------------------------------------------------------------- B-T5

def test_b_t5_chamber_targets_equivariant_and_gradfree():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    loss_mod = OrbdetGDAProbeLoss()
    torch.manual_seed(3)
    psi = torch.rand(16) * 2 * math.pi
    rot = torch.tensor(0.43)
    psi_req = psi.clone().requires_grad_(True)
    t_rot, t_flp = loss_mod.chamber_targets_xview(psi_req.detach(), rot)
    assert not t_rot.requires_grad and not t_flp.requires_grad
    assert (t_rot == (torch.sin(psi + 2 * rot) > 0).long()).all()
    assert (t_flp == (torch.sin(psi) < 0).long()).all()

    # the bit loss backpropagates only into the bit logits, not into u2
    loss_iso = OrbdetGDAProbeLoss(w_env=0.0, w_xview=0.0, w_bit_gt=0.0,
                                  w_bit_x=1.0)
    rows = torch.randn(5, 3, 6).requires_grad_(True)
    env3 = torch.rand(5, 3, 2) * 50 + 1
    sin2_3 = torch.rand(5, 3) * 2 - 1
    losses, _ = loss_iso(rows, env3, sin2_3, rot)
    losses['gda_loss_bit'].backward()
    grad = rows.grad
    assert grad is not None
    assert torch.all(grad[:, :, 2:4] == 0), 'u2 must get no bit-loss grad'
    assert torch.any(grad[:, :, 4:6] != 0), 'bit logits must get grad'


# ---------------------------------------------------------------- B-T6

def _tiny_head_cfg(**extra):
    cfg = dict(
        num_classes=2,
        in_channels=4,
        feat_channels=4,
        stacked_convs=1,
        strides=[8, 16],
        regress_ranges=[(-1, 64), (64, 128)],
        center_sampling=False,
        norm_on_bbox=False,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                      gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                             loss_weight=1.0),
    )
    cfg.update(extra)
    return cfg


def test_b_t6_head_parity_and_detach_isolation():
    from mmrotate.utils import register_all_modules
    register_all_modules()
    from mmrotate.models.dense_heads.h2rbox_gda_head import H2RBoxGDAHead
    from mmrotate.models.dense_heads.h2rbox_v2_head import H2RBoxV2Head
    torch.manual_seed(4)
    parent = H2RBoxV2Head(**_tiny_head_cfg())
    parent.init_weights()
    feats = [torch.randn(2, 4, 8, 8), torch.randn(2, 4, 4, 4)]

    # probe disabled: exact parent behaviour, same parameters
    child_off = H2RBoxGDAHead(**_tiny_head_cfg())
    child_off.load_state_dict(parent.state_dict())
    child_off.eval()
    parent.eval()
    with torch.no_grad():
        outs_p = parent(feats)
        outs_c = child_off(feats)
    for op, oc in zip(outs_p, outs_c):
        for lp, lc in zip(op, oc):
            assert torch.equal(lp, lc), 'probe-off must be bit-identical'
    assert child_off.last_gda_probe == []

    # Probe enabled: ordinary eval keeps baseline outputs identical and must
    # not execute/stash the analysis-only probe.
    # init_weights is exercised explicitly (ConvModule-with-norm convs
    # have bias=None -- a crash surface that broke the first smoke).
    cfg = _tiny_head_cfg(gda_probe=dict(enabled=True, detach_feats=True))
    child_on = H2RBoxGDAHead(**cfg)
    child_on.init_weights()
    child_on.load_state_dict(parent.state_dict(), strict=False)
    child_on.eval()
    with torch.no_grad():
        outs_c2 = child_on(feats)
    for op, oc in zip(outs_p, outs_c2):
        for lp, lc in zip(op, oc):
            assert torch.equal(lp, lc), 'baseline outputs must be untouched'
    assert child_on.last_gda_probe == []

    # Explicit analysis is the only eval-time path that emits probe maps.
    with torch.no_grad():
        probe_maps = child_on.forward_gda_probe(feats)
    assert len(probe_maps) == 2
    assert probe_maps[0].shape[1] == 6

    # detach_feats: a probe-only backward reaches no parent weight
    child_on.train()
    child_on.zero_grad()
    child_on(feats)
    probe_loss = sum(
        p.float().pow(2).mean() for p in child_on.last_gda_probe)
    probe_loss.backward()
    parent_param_names = dict(parent.named_parameters()).keys()
    for name, p in child_on.named_parameters():
        if name in parent_param_names:
            assert p.grad is None or torch.all(p.grad == 0), \
                f'parent param {name} must not receive probe gradient'
    assert any(p.grad is not None and torch.any(p.grad != 0)
               for n, p in child_on.named_parameters()
               if n.startswith('gda_probe')), 'probe tower must train'


def test_b_t6_ordinary_eval_never_executes_probe_tower(monkeypatch):
    from mmrotate.models.dense_heads.h2rbox_gda_head import H2RBoxGDAHead
    child = H2RBoxGDAHead(**_tiny_head_cfg(
        gda_probe=dict(enabled=True, detach_feats=True)))
    child.eval()
    feats = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4)]
    calls = []
    original = child.gda_probe_tower.forward

    def counted_forward(value):
        calls.append(value.shape)
        return original(value)

    monkeypatch.setattr(child.gda_probe_tower, 'forward', counted_forward)
    with torch.no_grad():
        child(feats)
    assert calls == []
    with torch.no_grad():
        child.forward_gda_probe(feats)
    assert len(calls) == len(feats)


def test_b_t6_probe_construction_preserves_shared_initialization_rng():
    from mmrotate.models.dense_heads.h2rbox_gda_head import H2RBoxGDAHead
    from mmrotate.models.dense_heads.h2rbox_v2_head import H2RBoxV2Head

    torch.manual_seed(46)
    parent = H2RBoxV2Head(**_tiny_head_cfg())
    parent.init_weights()
    torch.manual_seed(46)
    child = H2RBoxGDAHead(**_tiny_head_cfg(
        gda_probe=dict(enabled=True, detach_feats=True)))
    child.init_weights()

    child_state = child.state_dict()
    for name, value in parent.state_dict().items():
        assert torch.equal(value, child_state[name]), name


def test_b_t6_stashes_main_psc_encoding_before_object_mean_decode():
    from mmrotate.utils import register_all_modules
    register_all_modules()
    from mmrotate.models.dense_heads.h2rbox_gda_head import H2RBoxGDAHead
    cfg = _tiny_head_cfg(
        angle_coder=dict(
            type='PSCCoder', angle_version='le90', dual_freq=False,
            num_step=3, thr_mod=0),
        gda_probe=dict(enabled=True, detach_feats=True))
    head = H2RBoxGDAHead(**cfg)
    head.train()
    feats = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4)]
    outputs = head(feats)

    assert len(head.last_gda_main_angle) == len(feats)
    for encoded, predicted in zip(head.last_gda_main_angle, outputs[2]):
        assert encoded.shape[1] == head.angle_coder.encode_size
        assert torch.equal(encoded, predicted.detach())


def test_b_t6_main_teacher_decodes_after_pooling_psc_encodings():
    from mmrotate.models.detectors.orbdet_gda import \
        decode_compacted_main_sin2
    from mmrotate.models.task_modules.coders.angle_coder import PSCCoder
    coder = PSCCoder(
        angle_version='le90', dual_freq=False, num_step=3, thr_mod=0)
    point_angles = torch.tensor([[0.2], [1.0]])
    point_encodings = coder.encode(point_angles)
    pooled = point_encodings.mean(dim=0).reshape(1, 1, -1)

    got = decode_compacted_main_sin2(coder, pooled)
    expected = torch.sin(2.0 * coder.decode(
        pooled.reshape(-1, coder.encode_size))).reshape(1, 1)
    old_order = torch.sin(2.0 * coder.decode(point_encodings)).mean()

    assert torch.allclose(got, expected)
    assert not torch.allclose(got.squeeze(), old_order)


# ---------------------------------------------------------------- B-T7

def test_b_t7_compacts_by_true_bid_identity_when_middle_object_is_missing():
    from mmrotate.models.detectors.orbdet_gda import compact_probe_by_object
    # Parent-style ids share the integer object identity across views and use
    # only the fractional suffix to encode ori/rot/flp.  Object 2 is absent
    # from the rotated view; object 3 must not shift into object 2's rank.
    rows0 = torch.tensor([[10.], [20.], [30.]])
    bids0 = torch.tensor([1.2, 2.2, 3.2])
    rows1 = torch.tensor([[100.], [300.]])
    bids1 = torch.tensor([1.4, 3.4])
    rows2 = torch.tensor([[1000.], [2000.], [3000.]])
    bids2 = torch.tensor([1.6, 2.6, 3.6])
    extras = [rows0.clone(), rows1.clone(), rows2.clone()]

    rows3, ext3, keys = compact_probe_by_object(
        [rows0, rows1, rows2], [bids0, bids1, bids2], extras)

    assert torch.equal(torch.as_tensor(keys), torch.tensor([1, 3]))
    expected = torch.tensor([[10., 100., 1000.],
                             [30., 300., 3000.]])
    assert torch.equal(rows3.squeeze(-1), expected)
    assert torch.equal(ext3.squeeze(-1), expected)


def test_b_t7_mean_pools_duplicate_points_without_losing_global_image_ids():
    from mmrotate.models.detectors.orbdet_gda import compact_probe_by_object
    # IDs 1/2 belong to image 0 and ID 3 belongs to image 1. Object 1 has two
    # positive points per view; pooling must not merge or renumber images.
    bids = [torch.tensor([1.2, 1.2, 2.2, 3.2]),
            torch.tensor([1.4, 1.4, 2.4, 3.4]),
            torch.tensor([1.6, 1.6, 2.6, 3.6])]
    rows = [torch.tensor([[1.], [3.], [20.], [30.]]),
            torch.tensor([[10.], [14.], [200.], [300.]]),
            torch.tensor([[100.], [106.], [2000.], [3000.]])]
    rows3, _, keys = compact_probe_by_object(rows, bids, rows)
    assert torch.equal(torch.as_tensor(keys), torch.tensor([1, 2, 3]))
    assert torch.equal(rows3[:, :, 0],
                       torch.tensor([[2., 12., 103.],
                                     [20., 200., 2000.],
                                     [30., 300., 3000.]]))


def test_b_t7_rebuilt_flip_ids_match_parent_integer_identity():
    from mmengine.structures import InstanceData
    from mmrotate.models.detectors.orbdet_gda import OrbdetGDADetector
    from mmrotate.structures.bbox import RotatedBoxes

    def _instances(n):
        inst = InstanceData()
        boxes = torch.tensor([[100. + i, 100., 20., 10., 0.]
                              for i in range(n)])
        inst.bboxes = RotatedBoxes(boxes)
        inst.labels = torch.zeros(n, dtype=torch.long)
        return inst

    gt_ori = [_instances(2), _instances(1)]
    gt_rot = [_instances(2), _instances(1)]
    proxy = type('DetectorProxy', (), {'crop_size': (1024, 1024)})()
    gt_flp = OrbdetGDADetector._rebuild_flipped_view(
        proxy, gt_ori, gt_rot)

    got = torch.cat([g.bid for g in gt_flp])
    assert torch.allclose(got, torch.tensor([1.6, 2.6, 3.6]))


def test_b_t7_empty_intersection_keeps_every_view_in_autograd_graph():
    from mmrotate.models.detectors.orbdet_gda import compact_probe_by_object
    rows = [torch.randn(2, 6, requires_grad=True),
            torch.randn(0, 6, requires_grad=True),
            torch.randn(2, 6, requires_grad=True)]
    bids = [torch.tensor([1.2, 2.2]), torch.empty(0),
            torch.tensor([1.6, 2.6])]
    extras = [torch.randn(2, 3, requires_grad=True),
              torch.randn(0, 3, requires_grad=True),
              torch.randn(2, 3, requires_grad=True)]

    rows3, ext3, keys = compact_probe_by_object(rows, bids, extras)
    assert rows3.shape == (0, 3, 6)
    assert ext3.shape == (0, 3, 3)
    assert torch.as_tensor(keys).numel() == 0
    assert rows3.requires_grad and ext3.requires_grad
    (rows3.sum() + ext3.sum()).backward()
    assert all(value.grad is not None for value in rows + extras)


def test_b_t9_orientation_agnostic_objects_contribute_no_probe_loss():
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    torch.manual_seed(9)
    loss_mod = OrbdetGDAProbeLoss()
    rows = torch.randn(3, 3, 6, requires_grad=True)
    env3 = torch.rand(3, 3, 2) * 50 + 1
    sin2_main3 = torch.rand(3, 3) * 2 - 1
    valid = torch.tensor([False, True, False])

    masked, _ = loss_mod(
        rows, env3, sin2_main3, torch.tensor(0.4),
        valid_object_mask=valid)
    reference, _ = loss_mod(
        rows[1:2], env3[1:2], sin2_main3[1:2], torch.tensor(0.4))
    for name in masked:
        assert torch.allclose(masked[name], reference[name], atol=1e-6)

    sum(masked.values()).backward()
    assert torch.all(rows.grad[[0, 2]] == 0)
    assert torch.any(rows.grad[1] != 0)


# ---------------------------------------------------------------- B-T8

def test_b_t8_chamber_target_from_detached_main_angle():
    """v1.1 3a: chamber target = bit of the detached main-head angle."""
    import torch.nn.functional as F
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        OrbdetGDAProbeLoss
    torch.manual_seed(5)
    loss_mod = OrbdetGDAProbeLoss(w_env=0.0, w_xview=0.0, w_bit_gt=1.0,
                                  w_bit_x=0.0)
    m = 6
    rows = torch.zeros(m, 3, 6)
    rows[..., 1] = 2.0    # a = softplus(2) ~ 2.13 -> gate ~ 1
    rows[..., 4] = 3.0    # logit0 >> logit1 -> always predicts class 0
    rows[..., 5] = -3.0
    rows = rows.requires_grad_(True)
    # per-instance main-head sin(2*theta): instances 0,2 bit=1; 1,3,4
    # bit=0; instance 5 sits on the orbit boundary (|0.1| < 0.3, masked)
    col = torch.tensor([0.9, -0.9, 0.9, -0.9, -0.9, 0.1])
    sin2_main3 = col[:, None].expand(m, 3).contiguous()
    env3 = torch.rand(m, 3, 2) * 50 + 1
    losses, diag = loss_mod(rows, env3, sin2_main3, torch.tensor(0.5))

    # expected: gate ~ 1, per-view identical -> mean over 5 masked
    # instances: targets [1,0,1,0,0], predictions all class 0
    a = F.softplus(torch.tensor(2.0))
    g = torch.sigmoid((a - loss_mod.a0) / loss_mod.tau)
    logp = torch.log_softmax(torch.tensor([3.0, -3.0]), dim=-1)
    ce = torch.stack([-logp[1], -logp[0], -logp[1], -logp[0], -logp[0]])
    expected = (g * ce).mean()
    assert torch.allclose(losses['gda_loss_bit'], expected, atol=1e-5), \
        (losses['gda_loss_bit'].item(), expected.item())
    assert abs(diag['gda_bit_acc'].item() - 0.6) < 1e-6  # 3/5 masked-in
    # gradient flows only into the bit logits
    losses['gda_loss_bit'].backward()
    assert torch.all(rows.grad[:, :, :4] == 0)
    assert torch.any(rows.grad[:, :, 4:6] != 0)
