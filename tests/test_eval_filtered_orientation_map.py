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
