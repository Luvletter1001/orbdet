import torch

from tools.analysis_tools.eval_filtered_orientation_map import (
    apply_filter, fuse_suspicion, ranks01)


def test_ranks01_orders_ascending():
    r = ranks01([10.0, 0.0, 5.0])
    assert r[1] == 0.0 and r[0] == 1.0
    assert abs(r[2] - 0.5) < 1e-9


def test_fuse_suspicion_prefers_jointly_bad():
    # instance 2 is worst in both signals -> highest fused suspicion
    fused = fuse_suspicion([0.0, 5.0, 9.0], [0.1, 0.2, 0.9])
    assert fused[2] > fused[1] > fused[0]


def test_apply_filter_remove_and_downweight():
    bboxes = torch.zeros(4, 5)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([0, 1, 2, 3])
    flagged = torch.tensor([False, True, True, False])
    b, s, l = apply_filter(bboxes, scores, labels, flagged, 'remove', 0.3)
    assert torch.allclose(s, torch.tensor([0.9, 0.6]), atol=1e-6)
    assert l.tolist() == [0, 3]
    b, s, l = apply_filter(bboxes, scores, labels, flagged, 'downweight', 0.3)
    assert abs(s[1] - 0.24) < 1e-6 and abs(s[2] - 0.21) < 1e-6
    assert s[0] == 0.9 and s[3] == 0.6
    import pytest
    with pytest.raises(ValueError):
        apply_filter(bboxes, scores, labels, flagged, 'bogus', 0.3)


def test_periodic_angle_error_deg():
    import math
    from tools.analysis_tools.eval_filtered_orientation_map import (
        _periodic_angle_error_deg)
    err = _periodic_angle_error_deg(
        torch.tensor([math.radians(80.0), math.radians(-80.0)]),
        torch.tensor([0.0, 0.0]))
    assert torch.allclose(err, torch.tensor([80.0, 80.0]), atol=1e-4)


def test_voc_ap_11point():
    from tools.analysis_tools.eval_filtered_orientation_map import voc_ap
    # recall 1.0 reached at precision 0.5; earlier point (r=0, p=1) only
    # covers level 0
    ap = voc_ap([0.0, 0.5, 1.0], [1.0, 0.5, 0.5])
    assert abs(ap - (1.0 + 0.5 * 10) / 11) < 1e-9


def test_oriented_class_ap_angle_gate():
    import math
    from tools.analysis_tools.eval_filtered_orientation_map import (
        oriented_class_ap)
    # square-ish GT at angle 0; p1 (score .9) rotated 30 deg (IoU ~0.77,
    # angle err 30), p2 (score .8) exact
    gt = [torch.tensor([[50., 50., 20., 20., 0.]])]
    p1 = dict(image_index=0, score=0.9,
              bbox=torch.tensor([50., 50., 20., 20., math.radians(30)]))
    p2 = dict(image_index=0, score=0.8,
              bbox=torch.tensor([50., 50., 20., 20., 0.0]))
    entries = [p1, p2]
    # standard matching: p1 is TP (IoU>=0.5), perfect AP
    assert abs(oriented_class_ap(gt, entries, 0.5, None) - 1.0) < 1e-9
    # 45-deg gate: 30 deg error still passes
    assert abs(oriented_class_ap(gt, entries, 0.5, 45.0) - 1.0) < 1e-9
    # 15-deg gate: p1 becomes FP, p2 TP at rank 2 -> AP 0.5
    assert abs(oriented_class_ap(gt, entries, 0.5, 15.0) - 0.5) < 1e-9
    # no GT -> None
    assert oriented_class_ap([torch.zeros(0, 5)], entries, 0.5, 15.0) is None


def test_le90_regularization_unifies_equivalent_encodings():
    # the same physical rectangle encoded as (w>h, theta) vs (w<h, theta+90)
    # must yield the same canonical angle, otherwise the angle gate punishes
    # geometrically identical boxes
    import math
    from tools.analysis_tools.collect_low_rank_orientation_evidence import (
        _regularize_le90)
    b1 = torch.tensor([[0., 0., 20., 10., math.radians(30)]])
    b2 = torch.tensor([[0., 0., 10., 20., math.radians(120)]])
    a1 = _regularize_le90(b1)[:, 4]
    a2 = _regularize_le90(b2)[:, 4]
    diff = (a1 - a2 + math.pi / 2) % math.pi - math.pi / 2
    assert abs(float(diff[0])) < 1e-5
