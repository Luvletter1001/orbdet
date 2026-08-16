# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import math
import operator
from numbers import Real
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmrotate.structures.bbox import rbbox_overlaps


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be finite')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _positive_integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def periodic_angle_error(prediction: Tensor,
                         target: Tensor,
                         period: float = math.pi) -> Tensor:
    """Return the shortest absolute angular error for a periodic angle."""
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError('prediction and target must be tensors')
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if prediction.device != target.device:
        raise ValueError('prediction and target devices must match')
    if prediction.dtype != target.dtype:
        raise ValueError('prediction and target dtype must match')
    if not prediction.is_floating_point():
        raise ValueError('prediction and target must be floating tensors')
    period = _finite_real(period, 'period')
    if period <= 0:
        raise ValueError('period must be positive and finite')
    if (not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()):
        raise ValueError('prediction and target must be finite')

    delta = prediction - target
    return torch.remainder(delta + period / 2, period).sub(period / 2).abs()


def _validate_matching_inputs(pred_boxes: Tensor, pred_scores: Tensor,
                              pred_labels: Tensor, gt_boxes: Tensor,
                              gt_labels: Tensor) -> None:
    tensors = (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError('matching inputs must be tensors')
    if (pred_boxes.ndim != 2 or pred_boxes.shape[1] != 5 or gt_boxes.ndim != 2
            or gt_boxes.shape[1] != 5):
        raise ValueError('box tensors must have shape [N, 5]')
    if pred_scores.ndim != 1 or pred_labels.ndim != 1 or gt_labels.ndim != 1:
        raise ValueError('scores and labels must have shape [N]')
    if (pred_boxes.shape[0] != pred_scores.shape[0]
            or pred_boxes.shape[0] != pred_labels.shape[0]
            or gt_boxes.shape[0] != gt_labels.shape[0]):
        raise ValueError('boxes, scores, and labels must align')
    device = gt_boxes.device
    if any(value.device != device for value in tensors):
        raise ValueError('matching input devices must align')
    if not pred_boxes.is_floating_point() or not gt_boxes.is_floating_point():
        raise ValueError('box tensors must be floating')
    if pred_boxes.dtype != gt_boxes.dtype:
        raise ValueError('prediction and ground-truth box dtype must match')
    if not pred_scores.is_floating_point():
        raise ValueError('prediction scores must be floating')
    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }
    if (pred_labels.dtype not in integer_dtypes
            or gt_labels.dtype not in integer_dtypes):
        raise ValueError('prediction and ground-truth labels must be integer')
    if pred_labels.dtype != gt_labels.dtype:
        raise ValueError('prediction and ground-truth label dtype must match')
    if (not torch.isfinite(pred_boxes).all()
            or not torch.isfinite(pred_scores).all()
            or not torch.isfinite(gt_boxes).all()):
        raise ValueError('boxes and scores must be finite')


def _empty_matches(device: torch.device) -> Dict[str, Tensor]:
    return dict(
        gt_index=torch.empty(0, device=device, dtype=torch.long),
        pred_index=torch.empty(0, device=device, dtype=torch.long),
        iou=torch.empty(0, device=device, dtype=torch.float32))


def match_rotated_predictions(pred_boxes: Tensor,
                              pred_scores: Tensor,
                              pred_labels: Tensor,
                              gt_boxes: Tensor,
                              gt_labels: Tensor,
                              score_threshold: float = 0.05,
                              iou_threshold: float = 0.5) -> Dict[str, Tensor]:
    """Greedily match score-ordered predictions to same-class GT boxes."""
    score_threshold = _finite_real(score_threshold, 'score_threshold')
    iou_threshold = _finite_real(iou_threshold, 'iou_threshold')
    if not 0 <= score_threshold <= 1:
        raise ValueError('score_threshold must lie in [0, 1]')
    if not 0 <= iou_threshold <= 1:
        raise ValueError('iou_threshold must lie in [0, 1]')
    _validate_matching_inputs(pred_boxes, pred_scores, pred_labels, gt_boxes,
                              gt_labels)

    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return _empty_matches(gt_boxes.device)

    keep = pred_scores >= score_threshold
    original_index = keep.nonzero(as_tuple=False).reshape(-1)
    if original_index.numel() == 0:
        return _empty_matches(gt_boxes.device)
    kept_scores = pred_scores[original_index].detach().to(
        device='cpu', dtype=torch.float64).numpy()
    order = torch.as_tensor(
        np.argsort(-kept_scores, kind='mergesort'),
        device=original_index.device)
    selected = original_index[order]

    available_gt = torch.ones(
        gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    gt_indices = []
    pred_indices = []
    matched_ious = []
    for pred_index in selected.tolist():
        candidates = ((gt_labels == pred_labels[pred_index])
                      & available_gt).nonzero(as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            continue
        overlaps = rbbox_overlaps(
            pred_boxes[pred_index:pred_index + 1].float(),
            gt_boxes[candidates].float())[0]
        best_value, best_offset = overlaps.max(dim=0)
        if best_value.item() < iou_threshold:
            continue
        gt_index = candidates[best_offset].item()
        available_gt[gt_index] = False
        gt_indices.append(gt_index)
        pred_indices.append(pred_index)
        matched_ious.append(best_value)

    if not gt_indices:
        return _empty_matches(gt_boxes.device)
    return dict(
        gt_index=torch.tensor(
            gt_indices, device=gt_boxes.device, dtype=torch.long),
        pred_index=torch.tensor(
            pred_indices, device=gt_boxes.device, dtype=torch.long),
        iou=torch.stack(matched_ious).to(dtype=torch.float32))


def _binary_arrays(target, score) -> Tuple[np.ndarray, np.ndarray]:
    try:
        raw_target = np.asarray(target).reshape(-1)
        raw_score = np.asarray(score).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError('target and score must be array-like') from error
    if raw_target.shape != raw_score.shape or raw_target.size == 0:
        raise ValueError('target and score must be non-empty and aligned')
    if not np.isrealobj(raw_target) or not np.isrealobj(raw_score):
        raise ValueError('binary targets and finite scores are required')
    try:
        target_is_finite = np.isfinite(raw_target).all()
        score = raw_score.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'binary targets and finite scores are required') from error
    if (not target_is_finite or not np.isfinite(score).all()
            or not np.isin(raw_target, [0, 1]).all()):
        raise ValueError('binary targets and finite scores are required')
    target = raw_target.astype(np.int64, copy=False)
    if np.unique(target).size != 2:
        raise ValueError('both binary classes are required')
    return target, score


def binary_auroc(target, score) -> float:
    """Compute exact binary AUROC, assigning average ranks to ties."""
    target, score = _binary_arrays(target, score)
    order = np.argsort(score, kind='mergesort')
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = target == 1
    n_pos = int(positives.sum())
    n_neg = target.size - n_pos
    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_average_precision(target, score) -> float:
    """Compute threshold-exact binary average precision without sklearn."""
    target, score = _binary_arrays(target, score)
    order = np.argsort(-score, kind='mergesort')
    sorted_target = target[order]
    sorted_score = score[order]
    true_positive = np.cumsum(sorted_target)
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_score)),
                           target.size - 1]
    positives_at_threshold = true_positive[threshold_ends]
    recall_increment = np.diff(np.r_[0, positives_at_threshold])
    precision = positives_at_threshold / (threshold_ends + 1)
    return float((recall_increment * precision).sum() / sorted_target.sum())


def expected_calibration_error(target, probability, bins=10) -> float:
    """Compute equal-width expected calibration error."""
    bins = _positive_integer(bins, 'bins')
    target, probability = _binary_arrays(target, probability)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('probabilities must lie in [0, 1]')
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = target.size
    value = 0.0
    for index in range(bins):
        right_closed = index == bins - 1
        mask = ((probability >= edges[index])
                & ((probability <= edges[index + 1]) if right_closed else
                   (probability < edges[index + 1])))
        if mask.any():
            value += (
                mask.sum() / total *
                abs(target[mask].mean() - probability[mask].mean()))
    return float(value)


def _fold_for_group(group, folds: int) -> int:
    digest = hashlib.sha256(str(group).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % folds


def _probe_arrays(
        features, target,
        groups: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_features = np.asarray(features)
    raw_target = np.asarray(target).reshape(-1)
    groups = np.asarray(groups)
    if (raw_features.ndim != 2 or raw_features.shape[0] != raw_target.size
            or raw_features.shape[0] == 0 or raw_features.shape[1] == 0):
        raise ValueError('features must have non-empty shape [N, D]')
    if groups.shape != raw_target.shape:
        raise ValueError('groups must align with target')
    if not np.isrealobj(raw_features) or not np.isrealobj(raw_target):
        raise ValueError('features must be finite and targets must be binary')
    try:
        features = raw_features.astype(np.float64, copy=False)
        target_is_finite = np.isfinite(raw_target).all()
    except (TypeError, ValueError) as error:
        raise ValueError(
            'features must be finite and targets must be binary') from error
    if not np.isfinite(features).all():
        raise ValueError('features must be finite')
    if not target_is_finite or not np.isin(raw_target, [0, 1]).all():
        raise ValueError('target must contain binary labels')
    target = raw_target.astype(np.int64, copy=False)
    if np.unique(target).size != 2:
        raise ValueError('both binary classes are required')
    if np.issubdtype(groups.dtype, np.number):
        try:
            if not np.isfinite(groups).all():
                raise ValueError('groups must be finite')
        except TypeError as error:
            raise ValueError(
                'groups must contain stable scalar IDs') from error
    return features, target, groups


def cross_validated_logistic_probe(features,
                                   target,
                                   groups: Sequence,
                                   folds=3,
                                   l2=1e-2,
                                   max_iter=100):
    """Return deterministic grouped out-of-fold logistic probabilities."""
    folds = _positive_integer(folds, 'folds', minimum=2)
    l2 = _finite_real(l2, 'l2')
    if l2 < 0:
        raise ValueError('l2 must be non-negative and finite')
    max_iter = _positive_integer(max_iter, 'max_iter')
    features, target, groups = _probe_arrays(features, target, groups)

    assignments = np.array([_fold_for_group(item, folds) for item in groups],
                           dtype=np.int64)
    probability = np.full(target.size, np.nan, dtype=np.float64)
    for fold in range(folds):
        test = assignments == fold
        train = ~test
        if not test.any() or np.unique(target[train]).size != 2:
            raise ValueError(
                'every fold needs test rows and two train classes')
        median = np.median(features[train], axis=0)
        mad = np.median(np.abs(features[train] - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)
        normalized_train = (features[train] - median) / scale
        normalized_test = (features[test] - median) / scale
        if (not np.isfinite(normalized_train).all()
                or not np.isfinite(normalized_test).all()):
            raise ValueError('standardized features must be finite')
        x_train = torch.tensor(normalized_train, dtype=torch.float64)
        y_train = torch.tensor(target[train], dtype=torch.float64)
        x_test = torch.tensor(normalized_test, dtype=torch.float64)
        weight = torch.zeros(
            features.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias],
                                      lr=1.0,
                                      max_iter=max_iter,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            logits = x_train @ weight + bias
            loss = (
                F.binary_cross_entropy_with_logits(logits, y_train) +
                0.5 * l2 * weight.square().sum())
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            probability[test] = torch.sigmoid(x_test @ weight +
                                              bias).cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('grouped probe left non-finite predictions')
    return probability


__all__ = [
    'periodic_angle_error', 'match_rotated_predictions', 'binary_auroc',
    'binary_average_precision', 'expected_calibration_error',
    'cross_validated_logistic_probe'
]
