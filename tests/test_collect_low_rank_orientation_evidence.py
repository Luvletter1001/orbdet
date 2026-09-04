import json
import math

import pytest
import torch

from tools.analysis_tools import collect_low_rank_orientation_evidence as C


def test_build_row_unique_identity_and_finite_scalars():
    evidence = dict(
        low_rank_angle=torch.tensor(0.1),
        low_rank_confidence=torch.tensor(0.9),
        low_rank_anisotropy=torch.tensor(0.8),
        low_rank_energy=torch.tensor(1.5),
        low_rank_valid=torch.tensor(True),
        low_rank_sigma1=torch.tensor(2.0),
        low_rank_sigma2=torch.tensor(0.1),
        fpn_level=torch.tensor(2, dtype=torch.long))
    geometry = dict(
        gt_width=10.0,
        gt_height=12.0,
        gt_area=120.0,
        gt_aspect_ratio=1.2)
    match = dict(
        pred_index=3,
        pred_score=0.77,
        rotated_iou=0.66,
        pred_angle=0.10,
        gt_angle=0.05,
        e2_deg=2.86,
        e4_deg=2.86)
    row = C.build_row('run', 'IMG1', 0, 5, evidence, geometry, match)
    assert (row['run_name'], row['image_id'],
            row['gt_index']) == ('run', 'IMG1', 0)
    assert row['label'] == 5
    assert row['evidence_available'] is True
    assert row['matched'] is True
    assert type(row['low_rank_valid']) is bool
    assert row['low_rank_valid'] is True
    for key in (
            'low_rank_angle', 'low_rank_confidence', 'low_rank_anisotropy',
            'low_rank_energy', 'low_rank_sigma1', 'low_rank_sigma2'):
        assert isinstance(row[key], float)
        assert math.isfinite(row[key])
    assert isinstance(row['fpn_level'], int)
    assert row['large_angle_error'] is False
    json.dumps(row, allow_nan=False)


def test_build_row_marks_unavailable_and_unmatched():
    row = C.build_row('run', 'IMG', 1, 0)
    assert row['evidence_available'] is False
    assert row['matched'] is False
    assert not any(key.startswith('low_rank_') for key in row)
    assert 'pred_index' not in row
    json.dumps(row, allow_nan=False)


def test_build_row_flags_large_angle_error():
    match = dict(
        pred_index=0,
        pred_score=0.9,
        rotated_iou=0.8,
        pred_angle=0.0,
        gt_angle=0.0,
        e2_deg=20.0,
        e4_deg=5.0)
    row = C.build_row('run', 'IMG', 0, 0, None, None, match)
    assert row['matched'] is True
    assert row['large_angle_error'] is True


def test_score_desc_one_to_one_matching_no_stealing():
    gt = torch.tensor(
        [[0., 0., 10., 10., 0.], [30., 0., 10., 10., 0.]],
        dtype=torch.float32)
    gt_labels = torch.tensor([0, 0])
    # Shuffled input order: idx0 score .8 overlaps gt0 only; idx1 score .7
    # matches the far gt1; idx2 score .9 matches gt0.
    pred = torch.tensor(
        [[0., 0., 9., 9., 0.], [30., 0., 10., 10., 0.],
         [0., 0., 10., 10., 0.]],
        dtype=torch.float32)
    scores = torch.tensor([0.8, 0.7, 0.9])
    labels = torch.tensor([0, 0, 0])
    matches = C.match_rotated_predictions(
        pred,
        scores,
        labels,
        gt,
        gt_labels,
        score_threshold=0.05,
        iou_threshold=0.5)
    pair = {
        int(g): int(p)
        for g, p in zip(
            matches['gt_index'].tolist(), matches['pred_index'].tolist())
    }
    # Highest score (idx2) claims gt0; gt0 is never stolen by the later idx0.
    assert pair[0] == 2
    assert pair[1] == 1
    assert len(pair) == 2
    assert matches['gt_index'].numel() == 2


def test_matching_ignores_other_class_and_low_score():
    gt = torch.tensor([[0., 0., 10., 10., 0.]], dtype=torch.float32)
    gt_labels = torch.tensor([0])
    pred = torch.tensor(
        [[0., 0., 10., 10., 0.], [0., 0., 10., 10., 0.]],
        dtype=torch.float32)
    scores = torch.tensor([0.9, 0.01])
    labels = torch.tensor([1, 0])  # idx0 wrong class, idx1 below score thr
    matches = C.match_rotated_predictions(
        pred,
        scores,
        labels,
        gt,
        gt_labels,
        score_threshold=0.05,
        iou_threshold=0.5)
    assert matches['gt_index'].numel() == 0


def test_le90_e2_period_pi_e4_period_half_pi():
    gt = torch.tensor([0.0])
    e2, e4 = C.e2_e4_deg(torch.tensor([math.radians(80.0)]), gt)
    assert abs(float(e2) - 80.0) < 1e-4   # period pi
    assert abs(float(e4) - 10.0) < 1e-4   # period pi/2
    e2b, e4b = C.e2_e4_deg(torch.tensor([math.radians(-80.0)]), gt)
    assert abs(float(e2b) - 80.0) < 1e-4
    assert abs(float(e4b) - 10.0) < 1e-4


def test_rejects_existing_outputs(tmp_path):
    out = tmp_path / 'evidence.jsonl'
    manifest = tmp_path / 'evidence.manifest.json'
    C.reject_existing_outputs(out, manifest)  # absent -> fine
    out.write_text('x')
    with pytest.raises(FileExistsError):
        C.reject_existing_outputs(out, manifest)
    out.unlink()
    manifest.write_text('x')
    with pytest.raises(FileExistsError):
        C.reject_existing_outputs(out, manifest)


def test_manifest_records_all_hashes_and_thresholds(tmp_path):
    cfg = tmp_path / 'cfg.py'
    ckpt = tmp_path / 'ckpt.pth'
    low_rank = tmp_path / 'low_rank.py'
    holdout = tmp_path / 'holdout.json'
    for path in (cfg, ckpt, low_rank, holdout):
        path.write_text(path.name)
    manifest = C.build_manifest(
        run_name='r',
        config_path=cfg,
        checkpoint_path=ckpt,
        low_rank_function_path=low_rank,
        holdout_manifest_path=holdout,
        output_path=tmp_path / 'evidence.jsonl',
        score_threshold=0.05,
        iou_threshold=0.50,
        max_images=None,
        image_count=8,
        row_count=20,
        matched_count=11,
        valid_count=9,
        roi_mode='hbox',
        extra_cues=False)
    for key in (
            'config_sha256', 'checkpoint_sha256',
            'low_rank_function_sha256', 'holdout_manifest_sha256'):
        assert isinstance(manifest[key], str) and len(manifest[key]) == 64
    assert manifest['score_threshold'] == 0.05
    assert manifest['iou_threshold'] == 0.50
    assert manifest['roi_mode'] == 'hbox'
    assert (manifest['row_count'], manifest['matched_count'],
            manifest['valid_count']) == (20, 11, 9)


def test_square_hboxes_centered_with_max_side():
    hboxes = torch.tensor(
        [[10., 20., 110., 40.],   # 100 x 20 -> square side 100
         [0., 0., 30., 70.]],     # 30 x 70  -> square side 70
        dtype=torch.float32)
    squares = C._square_hboxes(hboxes)
    assert squares.shape == (2, 4)
    # centers preserved
    assert torch.allclose(
        (squares[:, 0] + squares[:, 2]) / 2,
        torch.tensor([60., 15.]))
    assert torch.allclose(
        (squares[:, 1] + squares[:, 3]) / 2,
        torch.tensor([30., 35.]))
    # side == max(width, height), both axes equal
    assert torch.allclose(squares[:, 2] - squares[:, 0],
                          torch.tensor([100., 70.]))
    assert torch.allclose(squares[:, 3] - squares[:, 1],
                          squares[:, 2] - squares[:, 0])
    with pytest.raises(ValueError):
        C._square_hboxes(torch.zeros(3, 5))


def test_parse_args_roi_mode_default_and_explicit(tmp_path):
    cfg = tmp_path / 'cfg.py'
    ckpt = tmp_path / 'ckpt.pth'
    out = tmp_path / 'evidence.jsonl'
    base = [str(cfg), '--checkpoint', str(ckpt), '--run-name', 'r',
            '--output', str(out)]
    args = C.parse_args(base)
    assert args.roi_mode == 'hbox'
    args = C.parse_args(base + ['--roi-mode', 'square'])
    assert args.roi_mode == 'square'
    with pytest.raises(SystemExit):
        C.parse_args(base + ['--roi-mode', 'rotated'])


def test_feature_hboxes_rejects_unknown_roi_mode():
    with pytest.raises(ValueError):
        C._feature_hboxes_and_geometry([], roi_mode='rotated')
    # default mode stays hbox and accepts empty input
    hboxes, keep, geometries = C._feature_hboxes_and_geometry([])
    assert hboxes == [] and keep == [] and geometries == []


def test_spatial_activation_axis_recovers_blob_long_axis():
    # elongated activation blob at +30 degrees inside a 32x32 RoI
    theta = math.radians(30.0)
    yy, xx = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing='ij')
    ca, sa = math.cos(-theta), math.sin(-theta)
    u = ca * (xx - 16) - sa * (yy - 16)   # along long axis
    v = sa * (xx - 16) + ca * (yy - 16)
    blob = ((u.abs() <= 10) & (v.abs() <= 2)).float()
    features = torch.stack([blob, blob * 0.5 + 0.01])[None]  # [1, 2, 32, 32]
    axis, eccentricity = C.spatial_activation_axis(features)
    err = abs(math.degrees(float(axis[0])) - 30.0 + 90) % 180 - 90
    assert abs(err) < 3.0
    assert float(eccentricity[0]) > 0.8
    # a square blob is isotropic: low eccentricity, axis meaningless
    square = torch.ones(1, 2, 32, 32)
    square[:, :, 8:24, 8:24] = 2.0
    _, ecc_square = C.spatial_activation_axis(square)
    assert float(ecc_square[0]) < 0.1
    with pytest.raises(ValueError):
        C.spatial_activation_axis(torch.zeros(3, 3))


def test_image_patch_axis_follows_pixel_edges():
    # bright horizontal bar (long axis along x) in a square crop
    images = torch.zeros(1, 3, 64, 64)
    images[:, :, 28:36, 8:56] = 1.0
    boxes = [torch.tensor([[4., 4., 60., 60.]])]
    axis = C.image_patch_axis(images, boxes, out_size=14)
    assert axis.numel() == 1
    err = abs((math.degrees(float(axis[0])) + 90) % 180 - 90)
    assert err < 5.0
    # empty input yields empty output
    assert C.image_patch_axis(images, [torch.zeros(0, 4)]).numel() == 0


def test_low_rank_roi_evidence_is_finite_and_level_in_range():
    from mmrotate.utils import register_all_modules
    from mmrotate.registry import MODELS
    register_all_modules()
    extractor = MODELS.build(
        dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]))
    torch.manual_seed(0)
    features = tuple(
        torch.randn(1, 256, size, size)
        for size in (128, 64, 32, 16, 8))
    hboxes = [torch.tensor([[10., 10., 90., 90.]])]
    result = C.low_rank_evidence_for_hboxes(extractor, features, hboxes)
    assert result['instance_index'].numel() == 1
    idx = 0
    for key in (
            'low_rank_angle', 'low_rank_confidence', 'low_rank_anisotropy',
            'low_rank_energy', 'low_rank_sigma1', 'low_rank_sigma2'):
        value = float(result[key][idx])
        assert math.isfinite(value)
    level = int(result['fpn_level'][idx])
    assert 0 <= level <= 4
    assert result['low_rank_valid'].dtype == torch.bool
