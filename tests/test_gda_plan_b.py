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

    # probe enabled: baseline outputs still identical, probe stashed
    cfg = _tiny_head_cfg(gda_probe=dict(enabled=True, detach_feats=True))
    child_on = H2RBoxGDAHead(**cfg)
    child_on.load_state_dict(parent.state_dict(), strict=False)
    child_on.eval()
    with torch.no_grad():
        outs_c2 = child_on(feats)
    for op, oc in zip(outs_p, outs_c2):
        for lp, lc in zip(op, oc):
            assert torch.equal(lp, lc), 'baseline outputs must be untouched'
    assert len(child_on.last_gda_probe) == 2
    assert child_on.last_gda_probe[0].shape[1] == 6

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


# ---------------------------------------------------------------- B-T7

def test_b_t7_compact_by_bid_rank():
    from mmrotate.models.detectors.orbdet_gda import compact_probe_by_object
    # bid integers differ across views (parent accumulates the offset);
    # identity is the rank of the bid integer within the view.
    rows0 = torch.tensor([[1., 10.], [2., 20.], [3., 30.]])
    bids0 = torch.tensor([1.2, 1.2, 2.2])  # obj0 twice, obj1 once
    rows1 = torch.tensor([[4., 40.], [5., 50.], [6., 60.]])
    bids1 = torch.tensor([5.4, 6.4, 6.4])  # obj0 once, obj1 twice
    rows2 = torch.tensor([[7., 70.], [8., 80.]])
    bids2 = torch.tensor([9.6, 10.6])      # obj0, obj1
    ext0 = torch.tensor([[100., 0.1], [101., 0.1], [102., 0.2]])
    ext1 = torch.tensor([[200., 0.3], [201., 0.4], [202., 0.4]])
    ext2 = torch.tensor([[300., 0.5], [301., 0.6]])

    rows3, ext3, keep = compact_probe_by_object(
        [rows0, rows1, rows2], [bids0, bids1, bids2], [ext0, ext1, ext2])
    assert keep == [0, 1]
    assert rows3.shape == (2, 3, 2)
    # obj0: mean of rows0[0:2], rows1[0], rows2[0]
    assert torch.allclose(rows3[0, 0], torch.tensor([1.5, 15.]))
    assert torch.allclose(rows3[0, 1], torch.tensor([4., 40.]))
    assert torch.allclose(rows3[0, 2], torch.tensor([7., 70.]))
    # extras are mean-pooled (point-independent per object in production)
    assert ext3[0, 0, 0].item() == 100.5   # mean of [100, 101]
    assert ext3[1, 1, 0].item() == 201.5   # mean of [201, 202]

    # an object missing from one view is dropped (bcnt == 3)
    rows3b, _, keep_b = compact_probe_by_object(
        [rows0[:2], rows1, rows2[:1]],
        [bids0[:2], bids1, bids2[:1]],
        [ext0[:2], ext1, ext2[:1]])
    assert keep_b == [0]
    assert rows3b.shape == (1, 3, 2)

    # empty intersection -> zero-sized tensors, no crash
    rows3c, _, keep_c = compact_probe_by_object(
        [rows0[:0], rows1, rows2],
        [bids0[:0], bids1, bids2],
        [ext0[:0], ext1, ext2])
    assert keep_c == [] and rows3c.shape == (0, 3, 2)
