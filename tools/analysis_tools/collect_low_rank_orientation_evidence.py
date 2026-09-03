#!/usr/bin/env python3
"""Collect per-GT low-rank orientation evidence from a frozen checkpoint.

P0 diagnostic only: this script never trains, mutates weights, writes source
data, or touches the DOTA official test split. It runs a frozen model over the
fixed grouped holdout, extracts GT-HBox FPN RoIs, evaluates the pure
``low_rank_channel_orientation_evidence`` primitive on each RoI, performs
one-to-one rotated prediction matching, and publishes an immutable JSONL plus
manifest in a single exclusive step.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from numbers import Real
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.structures.bbox import BaseBoxes, bbox2roi
from mmrotate.models.losses.low_rank_orientation_evidence import (
    low_rank_channel_orientation_evidence)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.structures.bbox import rbbox_overlaps
from mmrotate.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[2]
LOW_RANK_FUNCTION_PATH = (
    ROOT / 'mmrotate' / 'models' / 'losses' /
    'low_rank_orientation_evidence.py')
GROUPED_HOLDOUT_MANIFEST = Path(
    '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407/'
    'manifests/holdout.json')

MANIFEST_SCHEMA_VERSION = 1
ROW_SCHEMA_VERSION = 1
ROI_OUTPUT_SIZE = 14
ROI_SAMPLING_RATIO = 2
ROI_OUT_CHANNELS = 256
FEATMAP_STRIDES = [8, 16, 32, 64, 128]
MIN_BOX_SIZE = 2.0
ROI_MODES = ('hbox', 'square')
LARGE_ERROR_DEG = 15.0

_BASE_ROW_KEYS = {
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched'
}
_EVIDENCE_KEYS = {
    'low_rank_angle', 'low_rank_confidence', 'low_rank_anisotropy',
    'low_rank_energy', 'low_rank_valid', 'low_rank_sigma1',
    'low_rank_sigma2', 'fpn_level',
    # optional feature/image cues (only present with --extra-cues)
    'spatial_axis', 'spatial_eccentricity', 'image_axis'
}
_GEOMETRY_KEYS = {'gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio'}
_MATCH_KEYS = {
    'pred_index', 'pred_score', 'rotated_iou', 'pred_angle', 'gt_angle',
    'e2_deg', 'e4_deg', 'large_angle_error'
}
_RESERVED_ROW_KEYS = _BASE_ROW_KEYS | _GEOMETRY_KEYS | _MATCH_KEYS


def sha256_file(path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be finite')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _scalar(value):
    """Convert a finite numeric scalar (or 0-d tensor) to a python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('row evidence must be scalar')
        value = value.item()
    if isinstance(value, bool):
        return value
    if not isinstance(value, Real):
        raise TypeError('row evidence must be a finite numeric scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('row evidence scalar must be finite')
    return value


def _box_tensor(boxes) -> torch.Tensor:
    """Return an ``[N, 5]`` rotated-box tensor from container or tensor."""
    if isinstance(boxes, BaseBoxes):
        tensor = boxes.tensor
    elif isinstance(boxes, torch.Tensor):
        tensor = boxes
    else:
        raise TypeError('boxes must be BaseBoxes or torch.Tensor')
    if tensor.ndim != 2 or tensor.shape[1] != 5:
        raise ValueError('rotated boxes must have shape [N, 5]')
    return tensor


def _regularize_le90(boxes: torch.Tensor) -> torch.Tensor:
    """Return a long-edge canonical copy without mutating the source."""
    return RotatedBoxes(boxes.clone()).regularize_boxes('le90')


def periodic_angle_error(prediction: torch.Tensor, target: torch.Tensor,
                         period: float = math.pi) -> torch.Tensor:
    """Return the shortest absolute angular error on a periodic circle."""
    if not isinstance(prediction, torch.Tensor) or not isinstance(
            target, torch.Tensor):
        raise TypeError('prediction and target must be tensors')
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if prediction.dtype != target.dtype:
        raise ValueError('prediction and target dtype must match')
    period = _finite_real(period, 'period')
    if period <= 0:
        raise ValueError('period must be positive and finite')
    reduced_prediction = torch.remainder(prediction.double(), period)
    reduced_target = torch.remainder(target.double(), period)
    direct_error = (reduced_prediction - reduced_target).abs()
    wrapped_error = (period - direct_error).clamp_min(0.0)
    error = torch.minimum(direct_error, wrapped_error).to(prediction.dtype)
    return error


def _validate_matching_inputs(pred_boxes, pred_scores, pred_labels, gt_boxes,
                              gt_labels) -> None:
    tensors = (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError('matching inputs must be tensors')
    if (pred_boxes.ndim != 2 or pred_boxes.shape[1] != 5
            or gt_boxes.ndim != 2 or gt_boxes.shape[1] != 5):
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
    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }
    if (pred_labels.dtype not in integer_dtypes
            or gt_labels.dtype not in integer_dtypes):
        raise ValueError('prediction and ground-truth labels must be integer')
    if (not torch.isfinite(pred_boxes).all()
            or not torch.isfinite(pred_scores).all()
            or not torch.isfinite(gt_boxes).all()):
        raise ValueError('boxes and scores must be finite')
    if ((pred_boxes[:, 2:4] <= 0).any() or (gt_boxes[:, 2:4] <= 0).any()):
        raise ValueError('box widths and heights must be positive')


def _empty_matches(device) -> Dict[str, torch.Tensor]:
    return dict(
        gt_index=torch.empty(0, device=device, dtype=torch.long),
        pred_index=torch.empty(0, device=device, dtype=torch.long),
        iou=torch.empty(0, device=device, dtype=torch.float32))


def _relative_float32_boxes(pred_box, candidate_boxes):
    relative_pred = pred_box.detach().to(dtype=torch.float64).clone()
    relative_candidates = candidate_boxes.detach().to(
        dtype=torch.float64).clone()
    origin = relative_pred[:, :2].clone()
    relative_pred[:, :2] -= origin
    relative_candidates[:, :2] -= origin
    relative_pred = relative_pred.float()
    relative_candidates = relative_candidates.float()
    return relative_pred, relative_candidates


def match_rotated_predictions(pred_boxes: torch.Tensor,
                              pred_scores: torch.Tensor,
                              pred_labels: torch.Tensor,
                              gt_boxes: torch.Tensor,
                              gt_labels: torch.Tensor,
                              score_threshold: float = 0.05,
                              iou_threshold: float = 0.5
                              ) -> Dict[str, torch.Tensor]:
    """Greedily match score-ordered predictions to same-class GT one-to-one."""
    import numpy as np
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
    gt_indices, pred_indices, matched_ious = [], [], []
    for pred_index in selected.tolist():
        candidates = ((gt_labels == pred_labels[pred_index])
                      & available_gt).nonzero(as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            continue
        relative_pred, relative_gt = _relative_float32_boxes(
            pred_boxes[pred_index:pred_index + 1], gt_boxes[candidates])
        overlaps = rbbox_overlaps(relative_pred, relative_gt)[0]
        if not torch.isfinite(overlaps).all():
            raise ValueError('rotated IoU values must be finite')
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


def e2_e4_deg(pred_angle: torch.Tensor,
              gt_angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Le90 canonical e2 (period pi) and e4 (period pi/2) errors in degrees."""
    e2 = periodic_angle_error(pred_angle, gt_angle, period=math.pi)
    e4 = periodic_angle_error(pred_angle, gt_angle, period=math.pi / 2)
    return torch.rad2deg(e2), torch.rad2deg(e4)


def _validated_scale_factor(value) -> Tuple[float, float]:
    """Return a finite positive ``(scale_x, scale_y)`` pair."""
    message = 'scale_factor must contain two finite positive numeric values'
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ValueError(message)
    if isinstance(value, torch.Tensor):
        components = value.detach().cpu().tolist()
    else:
        try:
            components = list(value)
        except TypeError as error:
            raise ValueError(message) from error
    if len(components) != 2:
        raise ValueError(message)
    validated = []
    for component in components:
        if isinstance(component, torch.Tensor):
            component = component.item()
        if isinstance(component, bool) or not isinstance(component, Real):
            raise ValueError(message)
        component = float(component)
        if not math.isfinite(component) or component <= 0:
            raise ValueError(message)
        validated.append(component)
    return tuple(validated)


def _feature_space_gt_instances(samples):
    """Clone original-space GT and rescale only the FPN-evidence view."""
    feature_instances = []
    for sample in samples:
        original = sample.gt_instances
        if not isinstance(original.bboxes, BaseBoxes):
            raise TypeError(
                'feature-space GT bboxes must be BaseBoxes before prediction')
        scale_factor = _validated_scale_factor(
            sample.metainfo.get('scale_factor'))
        cloned = copy.deepcopy(original)
        cloned_boxes = original.bboxes.clone()
        cloned_boxes.rescale_(scale_factor)
        cloned.bboxes = cloned_boxes
        feature_instances.append(cloned)
    return feature_instances


def _validate_mapping_keys(values: Mapping, name: str) -> None:
    if any(not isinstance(key, str) for key in values):
        raise TypeError(f'{name} keys must be strings')


def build_row(run_name: str,
              image_id: str,
              gt_index: int,
              label: int,
              evidence: Optional[Mapping] = None,
              geometry: Optional[Mapping] = None,
              match: Optional[Mapping] = None) -> Dict:
    """Build one immutable-schema JSON row for a ground-truth instance."""
    row = dict(
        run_name=str(run_name),
        image_id=str(image_id),
        gt_index=int(gt_index),
        label=int(label),
        evidence_available=evidence is not None,
        matched=match is not None)
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise TypeError('evidence must be a mapping or None')
        _validate_mapping_keys(evidence, 'evidence')
        collisions = _RESERVED_ROW_KEYS.intersection(evidence)
        if collisions:
            raise ValueError(
                'evidence cannot overwrite reserved row keys: '
                + ', '.join(sorted(collisions)))
        unknown = set(evidence).difference(_EVIDENCE_KEYS)
        if unknown:
            raise ValueError(
                f'unexpected evidence keys: {", ".join(sorted(unknown))}')
        for key, value in evidence.items():
            if key == 'low_rank_valid':
                row[key] = bool(_scalar(value))
            elif key == 'fpn_level':
                row[key] = int(_scalar(value))
            else:
                row[key] = _scalar(value)
    if geometry is not None:
        if not isinstance(geometry, Mapping):
            raise TypeError('geometry must be a mapping or None')
        unknown = set(geometry).difference(_GEOMETRY_KEYS)
        if unknown:
            raise ValueError(
                f'unexpected geometry keys: {", ".join(sorted(unknown))}')
        row.update({key: _scalar(value) for key, value in geometry.items()})
    if match is not None:
        if not isinstance(match, Mapping):
            raise TypeError('match must be a mapping or None')
        e2_value = _scalar(match['e2_deg'])
        row.update(
            pred_index=int(match['pred_index']),
            pred_score=_scalar(match['pred_score']),
            rotated_iou=_scalar(match['rotated_iou']),
            pred_angle=_scalar(match['pred_angle']),
            gt_angle=_scalar(match['gt_angle']),
            e2_deg=e2_value,
            e4_deg=_scalar(match['e4_deg']),
            large_angle_error=bool(e2_value > LARGE_ERROR_DEG))
    json.dumps(row, allow_nan=False)
    return row


def _hbox_geometry(hboxes: torch.Tensor) -> List[Dict]:
    """Return feature-space HBox geometry for every ground-truth instance."""
    geometries = []
    for index in range(hboxes.shape[0]):
        width = _scalar(hboxes[index, 2] - hboxes[index, 0])
        height = _scalar(hboxes[index, 3] - hboxes[index, 1])
        short_side = max(min(width, height), 1e-8)
        geometries.append(
            dict(
                gt_width=width,
                gt_height=height,
                gt_area=width * height,
                gt_aspect_ratio=max(width, height) / short_side))
    return geometries


def _square_hboxes(hboxes: torch.Tensor) -> torch.Tensor:
    """Replace each HBox by its centered square with side = max(width, height).

    RoIAlign resamples every RoI to a fixed ``output_size`` square. For an
    elongated [x0, y0, x1, y1] RoI the x/y resampling factors differ, which
    amplifies the pooled gradient component along the HBox long side by
    (W/H)**2 inside the structure tensor and flips the recovered axis by 90
    degrees. Pooling a centered square keeps the resampling isotropic so the
    low-rank axis reflects feature geometry rather than RoI shape.
    """
    if hboxes.ndim != 2 or hboxes.shape[1] != 4:
        raise ValueError('hboxes must have shape [N, 4]')
    center_x = (hboxes[:, 0] + hboxes[:, 2]) / 2
    center_y = (hboxes[:, 1] + hboxes[:, 3]) / 2
    half = torch.maximum(
        hboxes[:, 2] - hboxes[:, 0], hboxes[:, 3] - hboxes[:, 1]) / 2
    return torch.stack(
        (center_x - half, center_y - half,
         center_x + half, center_y + half),
        dim=1)


def spatial_activation_axis(roi_features: torch.Tensor,
                            eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """Axis (mod pi) of the RoI activation-energy blob via spatial moments.

    Unlike the gradient structure tensor (which cannot distinguish an edge's
    two perpendicular readings), the activation *support* is informative about
    the object itself: the long axis of the activation ellipse is the object's
    long axis, with no 90-degree ambiguity. Uses the feature content ("where
    the object responds") rather than gradient math only.

    Returns ``(axis, eccentricity)``: ``axis`` in (-pi/2, pi/2] with 0 along
    the +x direction (same convention as ``low_rank_angle``), ``eccentricity``
    in [0, 1] measuring how elongated the blob is (0 = isotropic).
    """
    if roi_features.ndim != 4:
        raise ValueError('roi_features must have shape [N, C, H, W]')
    work = roi_features.float() if roi_features.dtype in (
        torch.float16, torch.bfloat16) else roi_features
    energy = work.square().mean(dim=1)  # [N, H, W]
    # suppress uniform background: only above-median response counts
    background = energy.flatten(1).median(dim=1).values[:, None, None]
    energy = (energy - background).clamp_min(0)
    n = energy.shape[0]
    height, width = energy.shape[-2:]
    ys = torch.arange(height, device=energy.device, dtype=energy.dtype)
    xs = torch.arange(width, device=energy.device, dtype=energy.dtype)
    total = energy.sum(dim=(-2, -1)) + eps
    cy = (energy * ys[None, :, None]).sum(dim=(-2, -1)) / total
    cx = (energy * xs[None, None, :]).sum(dim=(-2, -1)) / total
    dy = ys[None, :, None] - cy[:, None, None]
    dx = xs[None, None, :] - cx[:, None, None]
    cov_yy = (energy * dy.square()).sum(dim=(-2, -1)) / total
    cov_xx = (energy * dx.square()).sum(dim=(-2, -1)) / total
    cov_xy = (energy * dx * dy).sum(dim=(-2, -1)) / total
    axis = 0.5 * torch.atan2(2 * cov_xy, cov_xx - cov_yy)
    # plain wrap into (-pi/2, pi/2]: this axis IS the blob long axis,
    # no -90 degree edge rotation (unlike the gradient structure tensor)
    axis = torch.remainder(axis + math.pi / 2, math.pi) - math.pi / 2
    trace = cov_xx + cov_yy
    delta = ((cov_xx - cov_yy).square() + 4 * cov_xy.square()).sqrt()
    lam_max = 0.5 * (trace + delta)
    lam_min = (0.5 * (trace - delta)).clamp_min(0)
    eccentricity = 1 - lam_min / (lam_max + eps)
    return axis, eccentricity


def image_patch_axis(images: torch.Tensor,
                     hboxes_per_image: Sequence[torch.Tensor],
                     out_size: int = ROI_OUTPUT_SIZE,
                     eps: float = 1e-6) -> torch.Tensor:
    """Structure-tensor axis from the *input image* itself under each box.

    Independent evidence source: the real pixel edges (hull sides, wing
    edges, lane markings) inside the same square region. ``images`` is the
    preprocessed batch tensor [B, C, H, W] and ``hboxes_per_image`` the
    per-image boxes in that coordinate frame (square boxes keep the crop
    isotropic). Per-channel means only shift gradients away; per-channel
    gradient moments are summed so std scaling is harmless.
    """
    from mmcv.ops import roi_align

    reference = images
    cleaned = []
    for image_index, hboxes in enumerate(hboxes_per_image):
        if hboxes is None or hboxes.numel() == 0:
            cleaned.append(reference.new_empty((0, 4)))
            continue
        cleaned.append(hboxes.to(device=reference.device,
                                 dtype=torch.float32))
    if sum(box.shape[0] for box in cleaned) == 0:
        return reference.new_empty((0,))
    rois = bbox2roi(cleaned)
    patches = roi_align(
        reference, rois, (out_size, out_size), 1.0,
        ROI_SAMPLING_RATIO, 'avg', True)
    work = patches.mean(dim=1, keepdim=True)  # grayscale [N, 1, H, W]
    gx = 0.5 * (work[:, :, 1:-1, 2:] - work[:, :, 1:-1, :-2])
    gy = 0.5 * (work[:, :, 2:, 1:-1] - work[:, :, :-2, 1:-1])
    height, width = gx.shape[-2:]
    wy = torch.hann_window(
        height + 2, periodic=False, device=work.device, dtype=work.dtype
    )[1:-1]
    wx = torch.hann_window(
        width + 2, periodic=False, device=work.device, dtype=work.dtype
    )[1:-1]
    support = wy[:, None] * wx[None, :]
    axial_x = (support * (gx.square() - gy.square())).sum(dim=(-2, -1))
    axial_y = (support * (2.0 * gx * gy)).sum(dim=(-2, -1))
    gradient_phase = 0.5 * torch.atan2(axial_y[:, 0], axial_x[:, 0])
    # edges run along the object: axis = gradient direction - 90 degrees
    return torch.remainder(
        gradient_phase + math.pi, math.pi) - math.pi / 2


def build_roi_extractor():
    """Build the fixed GT-HBox FPN RoI extractor."""
    return MODELS.build(
        dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=ROI_OUTPUT_SIZE,
                sampling_ratio=ROI_SAMPLING_RATIO),
            out_channels=ROI_OUT_CHANNELS,
            featmap_strides=FEATMAP_STRIDES))


def low_rank_evidence_for_hboxes(roi_extractor, features: Tuple[torch.Tensor,
                                                                ...],
                                 hboxes_per_image: Sequence[torch.Tensor],
                                 extra_cues: bool = False
                                 ) -> Dict[str, torch.Tensor]:
    """Extract RoIs and evaluate the low-rank evidence per instance.

    With ``extra_cues=True`` additionally returns ``spatial_axis`` /
    ``spatial_eccentricity`` from the activation-support second moments.
    """
    reference = features[0]
    empty_long = reference.new_empty((0,), dtype=torch.long)
    cleaned = []
    for image_index, hboxes in enumerate(hboxes_per_image):
        if hboxes is None or hboxes.numel() == 0:
            cleaned.append(reference.new_empty((0, 4)))
            continue
        if hboxes.ndim != 2 or hboxes.shape[1] != 4:
            raise ValueError('hboxes must have shape [N, 4]')
        cleaned.append(hboxes.to(device=reference.device, dtype=torch.float32))
    counts = [box.shape[0] for box in cleaned]
    if sum(counts) == 0:
        return dict(
            batch_index=empty_long,
            instance_index=empty_long.clone(),
            low_rank_angle=reference.new_empty((0,)),
            low_rank_confidence=reference.new_empty((0,)),
            low_rank_anisotropy=reference.new_empty((0,)),
            low_rank_energy=reference.new_empty((0,)),
            low_rank_valid=torch.empty(
                (0,), dtype=torch.bool, device=reference.device),
            low_rank_sigma1=reference.new_empty((0,)),
            low_rank_sigma2=reference.new_empty((0,)),
            fpn_level=empty_long.clone())

    rois = bbox2roi(cleaned).to(
        device=reference.device, dtype=reference.dtype)
    roi_features = roi_extractor(
        tuple(feature.detach() for feature in features), rois)
    evidence = low_rank_channel_orientation_evidence(
        roi_features, min_energy=1e-6, min_anisotropy=0.1, eps=1e-6)
    fpn_level = roi_extractor.map_roi_levels(
        rois, roi_extractor.num_inputs)

    batch_index = rois[:, 0].long()
    instance_index = torch.cat([
        torch.arange(count, device=reference.device, dtype=torch.long)
        for count in counts if count > 0
    ]) if any(count > 0 for count in counts) else empty_long
    result = dict(
        batch_index=batch_index,
        instance_index=instance_index,
        low_rank_angle=evidence['axis_angle'].detach(),
        low_rank_confidence=evidence['confidence'].detach(),
        low_rank_anisotropy=evidence['anisotropy'].detach(),
        low_rank_energy=evidence['energy'].detach(),
        low_rank_valid=evidence['valid'].detach().to(torch.bool),
        low_rank_sigma1=evidence['singular_values'][:, 0].detach(),
        low_rank_sigma2=evidence['singular_values'][:, 1].detach(),
        fpn_level=fpn_level.detach().long())
    if extra_cues:
        spatial_axis, spatial_eccentricity = spatial_activation_axis(
            roi_features.detach())
        result['spatial_axis'] = spatial_axis.detach()
        result['spatial_eccentricity'] = spatial_eccentricity.detach()
    return result


def _feature_hboxes_and_geometry(feature_instances, roi_mode: str = 'hbox'):
    """Convert feature-space RBox GT to HBox, applying the size guard.

    ``roi_mode='square'`` replaces each kept HBox by its centered square
    (side = max(width, height)) before RoI extraction, removing the
    anisotropic RoIAlign resampling that otherwise biases the low-rank
    gradient statistics. Recorded geometry always describes the original
    HBox regardless of the pooling mode.
    """
    if roi_mode not in ROI_MODES:
        raise ValueError(f'roi_mode must be one of {ROI_MODES}')
    hboxes_per_image = []
    keep_indices = []
    geometries = []
    for instances in feature_instances:
        rbox = _box_tensor(instances.bboxes)
        hbox_container = instances.bboxes.convert_to('hbox')
        hboxes = hbox_container.tensor.to(dtype=torch.float32)
        if hboxes.numel() == 0:
            hboxes = hboxes.reshape(0, 4)
        geometries.append(_hbox_geometry(hboxes))
        width = hboxes[:, 2] - hboxes[:, 0]
        height = hboxes[:, 3] - hboxes[:, 1]
        valid = (
            torch.isfinite(hboxes).all(dim=1)
            & (width >= MIN_BOX_SIZE) & (height >= MIN_BOX_SIZE))
        keep = valid.nonzero(as_tuple=False).reshape(-1)
        keep_indices.append(keep)
        kept = hboxes[keep]
        if roi_mode == 'square':
            kept = _square_hboxes(kept)
        hboxes_per_image.append(kept)
        # rbox is referenced only to keep the original-space contract clear.
        del rbox
    return hboxes_per_image, keep_indices, geometries


def _match_by_gt(pred, gt_boxes, gt_labels, score_threshold, iou_threshold):
    pred_boxes = _box_tensor(pred.bboxes)
    matches = match_rotated_predictions(
        pred_boxes,
        pred.scores,
        pred.labels,
        gt_boxes,
        gt_labels,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    canonical_pred = _regularize_le90(pred_boxes)
    canonical_gt = _regularize_le90(gt_boxes)
    result = {}
    for offset in range(matches['gt_index'].numel()):
        gt_index = int(matches['gt_index'][offset])
        pred_index = int(matches['pred_index'][offset])
        pred_angle = canonical_pred[pred_index, 4:pred_index + 5]
        gt_angle = canonical_gt[gt_index, 4:gt_index + 5]
        e2, e4 = e2_e4_deg(pred_angle, gt_angle)
        result[gt_index] = dict(
            pred_index=pred_index,
            pred_score=pred.scores[pred_index],
            rotated_iou=matches['iou'][offset],
            pred_angle=pred_angle[0],
            gt_angle=gt_angle[0],
            e2_deg=e2[0],
            e4_deg=e4[0])
    return result


def reject_existing_outputs(output_path, manifest_path) -> None:
    """Refuse to overwrite any prior evidence artifact."""
    for path in (Path(output_path), Path(manifest_path)):
        if os.path.lexists(path):
            raise FileExistsError(f'{path} must not already exist')


def _probability_argument(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            'threshold must be a finite number in [0, 1]') from error
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(
            'threshold must be a finite number in [0, 1]')
    return parsed


def _positive_integer_argument(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            'max-images must be a positive integer') from error
    if str(parsed) != value.strip() or parsed <= 0:
        raise argparse.ArgumentTypeError(
            'max-images must be a positive integer')
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=_positive_integer_argument)
    parser.add_argument(
        '--holdout-manifest', type=Path, default=GROUPED_HOLDOUT_MANIFEST)
    parser.add_argument(
        '--score-threshold', type=_probability_argument, default=0.05)
    parser.add_argument(
        '--iou-threshold', type=_probability_argument, default=0.50)
    parser.add_argument(
        '--roi-mode',
        choices=ROI_MODES,
        default='hbox',
        help="RoI geometry fed to RoIAlign: 'hbox' keeps the tight GT HBox; "
             "'square' pools the centered square (side=max(W,H)) to avoid "
             'anisotropic-resampling bias in the low-rank structure tensor')
    parser.add_argument(
        '--extra-cues',
        action='store_true',
        help='additionally record feature/image cues per instance: '
             'spatial_axis + spatial_eccentricity (activation-support '
             'second moments) and image_axis (structure tensor on the raw '
             'input-image crop)')
    return parser.parse_args(argv)


def _limit_raw_batch(data, limit: Optional[int]):
    if limit is None:
        return data
    batch_size = len(data['data_samples'])
    if batch_size <= limit:
        return data
    limited = dict(data)
    limited['inputs'] = data['inputs'][:limit]
    limited['data_samples'] = data['data_samples'][:limit]
    return limited


def _validate_grouped_holdout(cfg) -> None:
    val_loader = cfg.val_dataloader
    dataset = val_loader.dataset
    data_root = dataset.get('data_root') if isinstance(
        dataset, Mapping) else getattr(dataset, 'data_root', None)
    if data_root is None or 'holdout' not in str(data_root):
        raise ValueError(
            'val_dataloader dataset data_root must be the grouped holdout '
            'directory; refusing to run over trainval or official test')


def _write_rows(stream, runner, model, roi_extractor, run_name: str,
                max_images: Optional[int], score_threshold: float,
                iou_threshold: float, roi_mode: str,
                extra_cues: bool) -> Dict[str, int]:
    image_count = row_count = matched_count = valid_count = 0
    collected_rows = []
    with torch.inference_mode():
        for data in runner.val_dataloader:
            remaining = (
                None if max_images is None else max_images - image_count)
            if remaining is not None and remaining <= 0:
                break
            data = _limit_raw_batch(data, remaining)
            processed = model.data_preprocessor(data, training=False)
            if not isinstance(processed, Mapping):
                raise TypeError('data preprocessor must return a mapping')
            inputs = processed['inputs']
            samples = list(processed['data_samples'])
            if remaining is not None and len(samples) > remaining:
                inputs = inputs[:remaining]
                samples = samples[:remaining]
            if not samples:
                continue

            gt_snapshots = [
                dict(
                    image_id=str(sample.metainfo['img_id']),
                    boxes=_box_tensor(
                        sample.gt_instances.bboxes).detach().clone(),
                    labels=sample.gt_instances.labels.detach().clone())
                for sample in samples
            ]
            feature_instances = _feature_space_gt_instances(samples)
            features = model.extract_feat(inputs)
            hboxes_per_image, keep_indices, geometries = (
                _feature_hboxes_and_geometry(feature_instances, roi_mode))
            raw_evidence = low_rank_evidence_for_hboxes(
                roi_extractor, features, hboxes_per_image,
                extra_cues=extra_cues)
            if extra_cues:
                # Same concatenated RoI order as raw_evidence rows.
                image_axes = image_patch_axis(inputs, hboxes_per_image)
            evidence_by_image = [dict() for _ in samples]
            for row_id in range(raw_evidence['batch_index'].numel()):
                b = int(raw_evidence['batch_index'][row_id])
                kept_position = int(raw_evidence['instance_index'][row_id])
                original_gt_index = int(keep_indices[b][kept_position])
                evidence_by_image[b][original_gt_index] = {
                    key: raw_evidence[key][row_id]
                    for key in _EVIDENCE_KEYS
                    if key in raw_evidence
                }
                if extra_cues:
                    evidence_by_image[b][original_gt_index][
                        'image_axis'] = image_axes[row_id]
            predictions = list(model.predict(inputs, samples, rescale=True))
            if len(predictions) != len(samples):
                raise ValueError('predictions and samples must align')

            for batch_index, (snapshot, prediction) in enumerate(
                    zip(gt_snapshots, predictions)):
                gt_boxes = snapshot['boxes']
                gt_labels = snapshot['labels']
                match_by_gt = _match_by_gt(
                    prediction.pred_instances, gt_boxes, gt_labels,
                    score_threshold, iou_threshold)
                evidence_by_gt = evidence_by_image[batch_index]
                geometry_by_gt = geometries[batch_index]
                for gt_index, label in enumerate(gt_labels.tolist()):
                    evidence = evidence_by_gt.get(gt_index)
                    match = match_by_gt.get(gt_index)
                    row = build_row(
                        run_name,
                        snapshot['image_id'],
                        gt_index,
                        label,
                        evidence,
                        geometry_by_gt[gt_index],
                        match)
                    collected_rows.append(row)
                    row_count += 1
                    matched_count += int(match is not None)
                    valid_count += int(
                        match is not None and evidence is not None
                        and bool(evidence['low_rank_valid']))
                image_count += 1
                if image_count % 100 == 0:
                    print(
                        f'[collect] processed {image_count} images, '
                        f'{row_count} rows, {matched_count} matched, '
                        f'{valid_count} valid',
                        flush=True)
            if max_images is not None and image_count >= max_images:
                break
    # Publish rows in a stable global (image_id, gt_index) order regardless of
    # the dataloader / batch traversal order.
    collected_rows.sort(key=lambda row: (row['image_id'], row['gt_index']))
    for row in collected_rows:
        stream.write(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True) + '\n')
    return dict(
        image_count=image_count,
        row_count=row_count,
        matched_count=matched_count,
        valid_count=valid_count)


def build_manifest(*, run_name, config_path, checkpoint_path,
                   low_rank_function_path, holdout_manifest_path, output_path,
                   score_threshold, iou_threshold, max_images, image_count,
                   row_count, matched_count, valid_count, roi_mode,
                   extra_cues) -> Dict:
    """Assemble the immutable manifest with all frozen-input hashes."""
    def _digest(path):
        path = Path(path)
        return sha256_file(path) if path.exists() else None

    manifest = dict(
        schema_version=MANIFEST_SCHEMA_VERSION,
        row_schema_version=ROW_SCHEMA_VERSION,
        run_name=str(run_name),
        config=str(Path(config_path).resolve()),
        config_sha256=_digest(config_path),
        checkpoint=str(Path(checkpoint_path).resolve()),
        checkpoint_sha256=_digest(checkpoint_path),
        low_rank_function=str(Path(low_rank_function_path).resolve()),
        low_rank_function_sha256=_digest(low_rank_function_path),
        holdout_manifest=str(Path(holdout_manifest_path).resolve()),
        holdout_manifest_sha256=_digest(holdout_manifest_path),
        output=str(Path(output_path).resolve()),
        output_sha256=_digest(output_path),
        score_threshold=float(score_threshold),
        iou_threshold=float(iou_threshold),
        max_images=max_images,
        image_count=int(image_count),
        row_count=int(row_count),
        matched_count=int(matched_count),
        valid_count=int(valid_count),
        roi_mode=str(roi_mode),
        extra_cues=bool(extra_cues))
    return manifest


def _write_json(path: Path, value: Mapping) -> None:
    with path.open('x', encoding='utf-8') as stream:
        stream.write(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=parent, prefix=f'.{prefix}.', suffix='.tmp')
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _publish_exclusive(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f'{destination} must not already exist') from error


def _rollback_own_link(source: Path, destination: Path) -> None:
    try:
        if os.path.lexists(destination) and os.path.samefile(
                source, destination):
            destination.unlink()
    except OSError:
        pass


def _publish_pair(output_temp, output_path, manifest_temp, manifest_path):
    _publish_exclusive(output_temp, output_path)
    try:
        _publish_exclusive(manifest_temp, manifest_path)
    except BaseException:
        _rollback_own_link(output_temp, output_path)
        raise


def collect(args) -> Dict:
    """Collect evidence without mutating the model, checkpoint, or data."""
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    holdout_manifest_path = Path(args.holdout_manifest).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = output_path.with_suffix('.manifest.json')
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError('config and checkpoint must exist')
    if not LOW_RANK_FUNCTION_PATH.is_file():
        raise FileNotFoundError(
            f'low-rank function missing: {LOW_RANK_FUNCTION_PATH}')
    if not holdout_manifest_path.is_file():
        raise FileNotFoundError(
            f'holdout manifest missing: {holdout_manifest_path}')
    reject_existing_outputs(output_path, manifest_path)

    run_name = str(args.run_name)
    if not run_name.strip():
        raise ValueError('run_name must be a non-empty string')
    max_images = args.max_images
    score_threshold = _finite_real(args.score_threshold, 'score_threshold')
    iou_threshold = _finite_real(args.iou_threshold, 'iou_threshold')
    roi_mode = str(args.roi_mode)
    if roi_mode not in ROI_MODES:
        raise ValueError(f'roi_mode must be one of {ROI_MODES}')
    extra_cues = bool(args.extra_cues)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_digest = sha256_file(config_path)
    checkpoint_digest = sha256_file(checkpoint_path)
    output_temp = manifest_temp = None
    try:
        output_temp = _temporary_path(output_path.parent, output_path.name)
        with tempfile.TemporaryDirectory(
                prefix='orbdet_lowrank_runner_') as runner_workspace:
            register_all_modules()
            cfg = Config.fromfile(config_path)
            _validate_grouped_holdout(cfg)
            # Frozen single-process read-only inference: never inherit the
            # training config's distributed launcher (which would require
            # RANK/WORLD_SIZE torchrun environment variables).
            cfg.launcher = 'none'
            cfg.load_from = None
            cfg.resume = False
            cfg.work_dir = runner_workspace
            runner = Runner.from_cfg(cfg)
            runner.load_checkpoint(str(checkpoint_path))
            if (sha256_file(config_path) != config_digest
                    or sha256_file(checkpoint_path) != checkpoint_digest):
                raise RuntimeError(
                    'config or checkpoint changed while loading the model')
            model = runner.model.eval()
            device = next(model.parameters()).device
            roi_extractor = build_roi_extractor().to(device)
            roi_extractor.eval()

            with output_temp.open('x', encoding='utf-8') as stream:
                counts = _write_rows(
                    stream, runner, model, roi_extractor, run_name,
                    max_images, score_threshold, iou_threshold, roi_mode,
                    extra_cues)
                stream.flush()
                os.fsync(stream.fileno())

        manifest = build_manifest(
            run_name=run_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            low_rank_function_path=LOW_RANK_FUNCTION_PATH,
            holdout_manifest_path=holdout_manifest_path,
            output_path=output_temp,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_images=max_images,
            roi_mode=roi_mode,
            extra_cues=extra_cues,
            **counts)
        # The content digest is computed on the staged temp file (same bytes
        # as the hard-linked release); record the final published path.
        manifest['output'] = str(output_path)
        manifest_temp = _temporary_path(output_path.parent, manifest_path.name)
        _write_json(manifest_temp, manifest)
        _publish_pair(output_temp, output_path, manifest_temp, manifest_path)
        if sha256_file(output_path) != manifest['output_sha256']:
            raise RuntimeError(
                'published evidence digest differs from staged digest')
        return manifest
    finally:
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def main():
    args = parse_args()
    manifest = collect(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
