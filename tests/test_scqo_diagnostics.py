import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from mmrotate.evaluation.functional import scqo_diagnostics as diagnostics
from mmrotate.evaluation.functional.scqo_diagnostics import (
    binary_auroc, binary_average_precision, cross_validated_logistic_probe,
    expected_calibration_error, match_rotated_predictions,
    periodic_angle_error)


def test_periodic_angle_error_uses_shortest_pi_path():
    pred = torch.tensor([math.pi / 2 - 0.01, -math.pi / 2 + 0.02])
    target = torch.tensor([-math.pi / 2 + 0.01, math.pi / 2 - 0.01])

    error = periodic_angle_error(pred, target, period=math.pi)

    assert torch.allclose(error, torch.tensor([0.02, 0.03]), atol=1e-6)


def test_periodic_angle_error_preserves_dtype_and_device():
    pred = torch.tensor([0.25, -0.25], dtype=torch.float64)
    target = torch.zeros_like(pred)

    error = periodic_angle_error(pred, target)

    assert error.dtype == pred.dtype
    assert error.device == pred.device


@pytest.mark.parametrize('period', [0.0, -1.0, math.inf, math.nan])
def test_periodic_angle_error_rejects_invalid_period(period):
    values = torch.zeros(1)

    with pytest.raises(ValueError, match='period'):
        periodic_angle_error(values, values, period=period)


def test_periodic_angle_error_rejects_misaligned_or_nonfinite_inputs():
    with pytest.raises(ValueError, match='shapes'):
        periodic_angle_error(torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match='dtype'):
        periodic_angle_error(
            torch.zeros(1), torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError, match='floating'):
        periodic_angle_error(
            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match='finite'):
        periodic_angle_error(torch.tensor([math.nan]), torch.zeros(1))


def test_periodic_angle_error_rejects_period_unrepresentable_in_input_dtype():
    values = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(ValueError, match='representable'):
        periodic_angle_error(values, values, period=1e-300)


def test_periodic_angle_error_avoids_overflow_between_finite_extremes():
    maximum = torch.finfo(torch.float32).max
    pred = torch.tensor([maximum, -maximum])
    target = torch.tensor([-maximum, maximum])

    error = periodic_angle_error(pred, target)

    assert error.dtype == pred.dtype
    assert torch.isfinite(error).all()
    assert (error >= 0).all()
    assert (error <= math.pi / 2).all()


def test_matching_is_score_ordered_one_to_one_and_class_aware():
    gt = torch.tensor([[10., 10., 8., 4., 0.], [10., 10., 8., 4., 0.]])
    gt_labels = torch.tensor([0, 1])
    pred = gt.repeat_interleave(2, dim=0)
    pred_labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    result = match_rotated_predictions(
        pred,
        scores,
        pred_labels,
        gt,
        gt_labels,
        score_threshold=0.05,
        iou_threshold=0.5)

    assert torch.equal(result['gt_index'], torch.tensor([0, 1]))
    assert torch.equal(result['pred_index'], torch.tensor([0, 2]))
    assert torch.allclose(result['iou'], torch.ones(2))


def test_matching_accepts_iou_boundary_and_breaks_score_ties_by_index():
    gt = torch.tensor([[10., 10., 8., 4., 0.]])
    pred = gt.repeat(2, 1)

    result = match_rotated_predictions(
        pred,
        torch.tensor([0.8, 0.8]),
        torch.tensor([0, 0]),
        gt,
        torch.tensor([0]),
        iou_threshold=1.0)

    assert torch.equal(result['pred_index'], torch.tensor([0]))
    assert torch.equal(result['gt_index'], torch.tensor([0]))
    assert result['iou'].item() == pytest.approx(1.0)


def test_matching_supports_aligned_float64_boxes_with_float32_iou_output():
    box = torch.tensor([[10., 10., 8., 4., 0.]], dtype=torch.float64)

    result = match_rotated_predictions(
        box, torch.tensor([0.9], dtype=torch.float64), torch.tensor([0]),
        box.clone(), torch.tensor([0]))

    assert torch.equal(result['pred_index'], torch.tensor([0]))
    assert result['iou'].dtype == torch.float32
    assert result['iou'].item() == pytest.approx(1.0)


def test_matching_preserves_small_float64_offsets_at_large_centers():
    pred = torch.tensor([[1e6, 1e6, 0.01, 0.01, 0.0]], dtype=torch.float64)
    gt = torch.tensor([[1e6 + 0.02, 1e6, 0.01, 0.01, 0.0]],
                      dtype=torch.float64)

    result = match_rotated_predictions(
        pred, torch.tensor([0.9], dtype=torch.float64), torch.tensor([0]), gt,
        torch.tensor([0]))

    assert result['pred_index'].numel() == 0
    assert result['gt_index'].numel() == 0
    assert result['iou'].numel() == 0


@pytest.mark.parametrize(('width', 'height'), [(0.0, 1.0), (1.0, -1.0)])
def test_matching_rejects_nonpositive_box_dimensions(width, height):
    box = torch.tensor([[0.0, 0.0, width, height, 0.0]])

    with pytest.raises(ValueError, match='positive'):
        match_rotated_predictions(box, torch.tensor([0.9]), torch.tensor([0]),
                                  box.clone(), torch.tensor([0]))


def test_matching_rejects_dimensions_that_underflow_in_float32_iou():
    box = torch.tensor([[0.0, 0.0, 1e-300, 1e-300, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match='float32'):
        match_rotated_predictions(box, torch.tensor([0.9],
                                                    dtype=torch.float64),
                                  torch.tensor([0]), box.clone(),
                                  torch.tensor([0]))


def test_matching_rejects_relative_offsets_that_overflow_float64():
    pred = torch.tensor([[-1e308, 0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    gt = torch.tensor([[1e308, 0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match='relative'):
        match_rotated_predictions(pred, torch.tensor([0.9],
                                                     dtype=torch.float64),
                                  torch.tensor([0]), gt, torch.tensor([0]))


@pytest.mark.parametrize('empty_side', ['prediction', 'ground_truth', 'both'])
def test_matching_empty_inputs_return_well_typed_empty_results(empty_side):
    box = torch.tensor([[10., 10., 8., 4., 0.]])
    pred = box
    scores = torch.tensor([0.9])
    pred_labels = torch.tensor([0])
    gt = box.clone()
    gt_labels = torch.tensor([0])
    if empty_side in ('prediction', 'both'):
        pred = torch.empty((0, 5))
        scores = torch.empty(0)
        pred_labels = torch.empty(0, dtype=torch.long)
    if empty_side in ('ground_truth', 'both'):
        gt = torch.empty((0, 5))
        gt_labels = torch.empty(0, dtype=torch.long)

    result = match_rotated_predictions(pred, scores, pred_labels, gt,
                                       gt_labels)

    assert set(result) == {'gt_index', 'pred_index', 'iou'}
    assert result['gt_index'].shape == (0, )
    assert result['pred_index'].shape == (0, )
    assert result['iou'].shape == (0, )
    assert result['gt_index'].dtype == torch.long
    assert result['pred_index'].dtype == torch.long
    assert result['iou'].dtype == torch.float32
    assert all(value.device == gt.device for value in result.values())


@pytest.mark.parametrize(('parameter', 'value'),
                         [('score_threshold', -0.1), ('score_threshold', 1.1),
                          ('score_threshold', math.nan),
                          ('iou_threshold', -0.1), ('iou_threshold', 1.1),
                          ('iou_threshold', math.inf)])
def test_matching_rejects_invalid_thresholds(parameter, value):
    box = torch.tensor([[10., 10., 8., 4., 0.]])
    kwargs = {parameter: value}

    with pytest.raises(ValueError, match=parameter):
        match_rotated_predictions(box, torch.tensor([0.9]), torch.tensor([0]),
                                  box, torch.tensor([0]), **kwargs)


def test_matching_rejects_bad_shapes_labels_dtypes_and_nonfinite_values():
    box = torch.tensor([[10., 10., 8., 4., 0.]])
    score = torch.tensor([0.9])
    label = torch.tensor([0])

    with pytest.raises(ValueError, match='shape'):
        match_rotated_predictions(box[:, :4], score, label, box, label)
    with pytest.raises(ValueError, match='align'):
        match_rotated_predictions(box.repeat(2, 1), score, label, box, label)
    with pytest.raises(ValueError, match='integer'):
        match_rotated_predictions(box, score, label.float(), box, label)
    with pytest.raises(ValueError, match='dtype'):
        match_rotated_predictions(box.double(), score.double(), label, box,
                                  label)
    with pytest.raises(ValueError, match='finite'):
        match_rotated_predictions(box, torch.tensor([math.nan]), label, box,
                                  label)


def test_binary_metrics_have_known_perfect_values():
    target = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])

    assert binary_auroc(target, score) == 1.0
    assert binary_average_precision(target, score) == 1.0
    assert expected_calibration_error(target, score, bins=2) < 0.2


def test_binary_metrics_handle_ties_independently_of_order():
    target = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.9, 0.1, 0.1])

    assert binary_auroc(target, score) == pytest.approx(0.5)
    assert binary_average_precision(target, score) == pytest.approx(0.5)
    reversed_ap = binary_average_precision(target[::-1], score[::-1])
    assert reversed_ap == pytest.approx(0.5)


def test_expected_calibration_error_includes_probability_endpoints():
    target = np.array([0, 1])
    probability = np.array([0.0, 1.0])

    assert expected_calibration_error(target, probability, bins=2) == 0.0


@pytest.mark.parametrize('metric', [binary_auroc, binary_average_precision])
def test_binary_ranking_metrics_reject_missing_classes(metric):
    with pytest.raises(ValueError, match='both binary classes'):
        metric(np.zeros(2), np.array([0.1, 0.2]))


@pytest.mark.parametrize(('target', 'score'),
                         [(np.array([0, 1]), np.array([0.1])),
                          (np.array([0, 0.5]), np.array([0.1, 0.9])),
                          (np.array([0, 1]), np.array([0.1, math.nan]))])
def test_binary_metrics_reject_misaligned_nonbinary_or_nonfinite_inputs(
        target, score):
    with pytest.raises(ValueError):
        binary_auroc(target, score)


@pytest.mark.parametrize('bins', [0, -1, 1.5, math.inf])
def test_expected_calibration_error_rejects_invalid_bins(bins):
    with pytest.raises(ValueError, match='bins'):
        expected_calibration_error([0, 1], [0.1, 0.9], bins=bins)


def test_expected_calibration_error_rejects_values_outside_probability_range():
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        expected_calibration_error([0, 1], [-0.1, 1.0])


def test_grouped_probe_is_deterministic_and_separates_signal():
    groups = np.repeat(np.arange(30), 4)
    target = np.tile(np.array([0, 0, 1, 1]), 30)
    signal = target.astype(np.float64) * 2 - 1
    features = np.stack((signal + np.linspace(-0.1, 0.1, signal.size),
                         np.sin(np.arange(signal.size))),
                        axis=1)

    first = cross_validated_logistic_probe(
        features, target, groups, folds=3, l2=1e-2, max_iter=100)
    second = cross_validated_logistic_probe(
        features, target, groups, folds=3, l2=1e-2, max_iter=100)

    assert np.allclose(first, second)
    assert binary_auroc(target, first) > 0.99
    for group in np.unique(groups):
        assert np.all(np.isfinite(first[groups == group]))


def test_grouped_probe_handles_constant_features_and_zero_mad():
    groups = np.repeat(np.arange(30), 2)
    target = np.tile(np.array([0, 1]), 30)
    features = np.ones((target.size, 2), dtype=np.float64)

    probability = cross_validated_logistic_probe(features, target, groups)

    assert np.all(np.isfinite(probability))
    assert np.allclose(probability, 0.5)


def test_group_fold_assignments_use_canonical_identity_and_hash_once(
        monkeypatch):
    real_sha256 = diagnostics.hashlib.sha256
    hashed_payloads = []

    def counting_sha256(payload):
        hashed_payloads.append(payload)
        return real_sha256(payload)

    monkeypatch.setattr(diagnostics.hashlib, 'sha256', counting_sha256)
    groups = [0.0, -0.0, 1, 1.0, '1', '1']

    assignments = diagnostics._group_fold_assignments(groups, folds=17)

    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[4] == assignments[5]
    assert len(hashed_payloads) == 3
    assert len(set(hashed_payloads)) == 3


def test_probe_arrays_preserve_mixed_group_id_types():
    features = np.arange(8, dtype=np.float64).reshape(4, 2)
    target = np.array([0, 1, 0, 1])

    _, _, groups = diagnostics._probe_arrays(features, target,
                                             [1, '1', 2, '2'])

    assert groups.dtype == object
    assert isinstance(groups[0], int)
    assert isinstance(groups[1], str)


@pytest.mark.parametrize('group', [
    True,
    np.bool_(False), b'group', None,
    object(), math.nan, math.inf, -math.inf
])
def test_group_hash_rejects_unstable_or_nonfinite_ids(group):
    with pytest.raises(ValueError, match='stable finite scalar'):
        diagnostics._fold_for_group(group, folds=3)


def test_group_fold_assignments_are_stable_across_python_hash_seeds():
    script = (
        'import json; '
        'from mmrotate.evaluation.functional.scqo_diagnostics import '
        '_group_fold_assignments; '
        'values = [0.0, -0.0, 1, 1.0, "1", "1"]; '
        'print(json.dumps(_group_fold_assignments(values, 2**63-1).tolist()))')
    outputs = []
    for seed in ('1', '777'):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1',
            PYTHONHASHSEED=seed,
            PYTHONNOUSERSITE='1')
        output = subprocess.check_output([sys.executable, '-c', script],
                                         env=env,
                                         text=True)
        outputs.append(json.loads(output))

    assert outputs[0] == outputs[1]
    assignments = outputs[0]
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[4] == assignments[5]
    assert assignments[2] != assignments[4]


@pytest.mark.parametrize(('parameter', 'value'), [('folds', 1), ('folds', 2.5),
                                                  ('folds', math.inf),
                                                  ('l2', -1.0),
                                                  ('l2', math.nan),
                                                  ('l2', math.inf),
                                                  ('max_iter', 0),
                                                  ('max_iter', 2.5),
                                                  ('max_iter', math.inf)])
def test_grouped_probe_rejects_invalid_hyperparameters(parameter, value):
    features = np.arange(24, dtype=np.float64).reshape(12, 2)
    target = np.tile(np.array([0, 1]), 6)
    groups = np.repeat(np.arange(6), 2)
    kwargs = {parameter: value}

    with pytest.raises(ValueError, match=parameter):
        cross_validated_logistic_probe(features, target, groups, **kwargs)


def test_grouped_probe_rejects_bad_shapes_labels_and_nonfinite_features():
    features = np.arange(24, dtype=np.float64).reshape(12, 2)
    target = np.tile(np.array([0, 1]), 6)
    groups = np.repeat(np.arange(6), 2)

    with pytest.raises(ValueError, match=r'\[N, D\]'):
        cross_validated_logistic_probe(features[:, 0], target, groups)
    with pytest.raises(ValueError, match='groups'):
        cross_validated_logistic_probe(features, target, groups[:-1])
    with pytest.raises(ValueError, match='binary'):
        cross_validated_logistic_probe(features,
                                       target.astype(float) + 0.5, groups)
    features[0, 0] = math.nan
    with pytest.raises(ValueError, match='finite'):
        cross_validated_logistic_probe(features, target, groups)


def test_diagnostic_functions_are_exported_from_functional_package():
    from mmrotate.evaluation import functional

    names = (
        'periodic_angle_error',
        'match_rotated_predictions',
        'binary_auroc',
        'binary_average_precision',
        'expected_calibration_error',
        'cross_validated_logistic_probe',
    )
    assert all(name in functional.__all__ for name in names)
    assert all(callable(getattr(functional, name)) for name in names)
