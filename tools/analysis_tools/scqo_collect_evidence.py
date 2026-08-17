#!/usr/bin/env python3
"""Collect per-instance SCQO evidence from a frozen checkpoint."""

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.structures.bbox import BaseBoxes
from mmrotate.evaluation.functional.scqo_diagnostics import (
    match_rotated_predictions, periodic_angle_error)
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules

MANIFEST_SCHEMA_VERSION = 1
ROW_SCHEMA_VERSION = 1

_BASE_ROW_KEYS = {
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched'
}
_GEOMETRY_KEYS = {'gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio'}
_MATCH_KEYS = {
    'pred_index', 'pred_score', 'rotated_iou', 'pred_angle', 'gt_angle',
    'angle_error_deg', 'angle_error_c4_deg', 'large_angle_error'
}
_RESERVED_EVIDENCE_KEYS = _BASE_ROW_KEYS | _GEOMETRY_KEYS | _MATCH_KEYS
_ADAPTER_IDENTITY_KEYS = {
    'batch_index', 'instance_index', 'label', 'square_hbox'
}
_FROZEN_HRSC_VAL_PIPELINE_TYPES = (
    'mmdet.LoadImageFromFile',
    'mmdet.FixShapeResize',
    'mmdet.LoadAnnotations',
    'ConvertBoxType',
    'mmdet.PackDetInputs',
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value):
    """Convert a finite numeric or boolean scalar to a JSON scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('row evidence must be scalar')
        value = value.item()
    if isinstance(value, bool):
        return value
    if not isinstance(value, Real):
        raise TypeError(
            'row evidence must be a finite numeric or boolean scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('row evidence scalar must be finite')
    return value


def _box_tensor(boxes) -> torch.Tensor:
    """Return an ``[N, 5]`` tensor from a rotated box container or tensor."""
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
    """Return a long-edge canonical copy without mutating source boxes."""
    return RotatedBoxes(boxes.clone()).regularize_boxes('le90')


def _config_field(value, key: str):
    if isinstance(value, Mapping):
        return value[key]
    return getattr(value, key)


def _validate_hrsc_val_pipeline(cfg) -> None:
    """Require image-only resize before GT loading and no later geometry."""
    message = (
        'SCQO collection requires the frozen HRSC validation pipeline: '
        'LoadImageFromFile -> FixShapeResize(800x800, keep_ratio=True) -> '
        'LoadAnnotations(qbox) -> ConvertBoxType(rbox) -> PackDetInputs '
        'with scale_factor metadata')
    try:
        val_dataloader = _config_field(cfg, 'val_dataloader')
        dataset = _config_field(val_dataloader, 'dataset')
        pipeline = _config_field(dataset, 'pipeline')
        transforms = list(pipeline)
        transform_types = tuple(
            _config_field(transform, 'type') for transform in transforms)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(message) from error

    if transform_types != _FROZEN_HRSC_VAL_PIPELINE_TYPES:
        raise ValueError(message)
    resize = transforms[1]
    annotations = transforms[2]
    convert = transforms[3]
    packed = transforms[4]
    try:
        resize_matches = (
            _config_field(resize, 'width') == 800
            and _config_field(resize, 'height') == 800
            and _config_field(resize, 'keep_ratio') is True)
        annotations_match = (
            _config_field(annotations, 'with_bbox') is True
            and _config_field(annotations, 'box_type') == 'qbox')
        convert_matches = (
            _config_field(convert,
                          'box_type_mapping') == dict(gt_bboxes='rbox'))
        meta_keys = _config_field(packed, 'meta_keys')
        pack_matches = ('scale_factor' in meta_keys)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(message) from error
    if not (resize_matches and annotations_match and convert_matches
            and pack_matches):
        raise ValueError(message)


def _validated_scale_factor(value) -> tuple:
    """Return a finite positive ``(scale_x, scale_y)`` pair."""
    message = 'scale_factor must contain two finite positive numeric values'
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ValueError(message)
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(message)
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
            if component.numel() != 1:
                raise ValueError(message)
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


def build_evidence_row(run_name: str, image_id: str, gt_index: int, label: int,
                       evidence: Optional[Dict], geometry: Optional[Dict],
                       match: Optional[Dict]) -> Dict:
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
        collisions = _RESERVED_EVIDENCE_KEYS.intersection(evidence)
        if collisions:
            names = ', '.join(sorted(collisions))
            raise ValueError(
                f'evidence cannot overwrite reserved row keys: {names}')
        row.update({key: _scalar(value) for key, value in evidence.items()})
    if geometry is not None:
        if not isinstance(geometry, Mapping):
            raise TypeError('geometry must be a mapping or None')
        _validate_mapping_keys(geometry, 'geometry')
        unexpected = set(geometry).difference(_GEOMETRY_KEYS)
        if unexpected:
            names = ', '.join(sorted(unexpected))
            raise ValueError(f'unexpected geometry keys: {names}')
        row.update({key: _scalar(value) for key, value in geometry.items()})
    if match is not None:
        if not isinstance(match, Mapping):
            raise TypeError('match must be a mapping or None')
        row.update(
            pred_index=int(match['pred_index']),
            pred_score=_scalar(match['pred_score']),
            rotated_iou=_scalar(match['rotated_iou']),
            pred_angle=_scalar(match['pred_angle']),
            gt_angle=_scalar(match['gt_angle']),
            angle_error_deg=_scalar(match['angle_error_deg']),
            angle_error_c4_deg=_scalar(match['angle_error_c4_deg']),
            large_angle_error=bool(_scalar(match['angle_error_deg']) > 15.0))
    json.dumps(row, allow_nan=False)
    return row


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
        '--score-threshold', type=_probability_argument, default=0.05)
    parser.add_argument(
        '--iou-threshold', type=_probability_argument, default=0.5)
    return parser.parse_args(argv)


def _evidence_cfg() -> Dict:
    return dict(
        type='SCQOFPNInstanceEvidence',
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        evidence=dict(
            type='SCQOStabilizerEvidence',
            roi_size=14,
            channels=256,
            block_size=2,
            negative_seed=3407,
            min_support_fraction=0.95))


def _row_evidence(result: Mapping, row_index: int) -> Dict:
    return {
        key: value[row_index]
        for key, value in result.items() if key not in _ADAPTER_IDENTITY_KEYS
    }


def _validated_probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite number in [0, 1]')
    return value


def _validated_max_images(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError('max_images must be a positive integer')
    value = int(value)
    if value <= 0:
        raise ValueError('max_images must be a positive integer')
    return value


def _validated_run_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('run_name must be a non-empty string')
    return value


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _limit_raw_batch(data, limit: Optional[int]):
    if limit is None:
        return data
    if not isinstance(data, Mapping):
        raise TypeError('validation batches must be mappings')
    if 'inputs' not in data or 'data_samples' not in data:
        raise ValueError('validation batches need inputs and data_samples')
    try:
        batch_size = len(data['data_samples'])
    except TypeError as error:
        raise ValueError('validation data_samples must be batched') from error
    if batch_size <= limit:
        return data
    limited = dict(data)
    limited['inputs'] = data['inputs'][:limit]
    limited['data_samples'] = data['data_samples'][:limit]
    return limited


def _validate_evidence_identity(result: Mapping, samples) -> None:
    required = {'batch_index', 'instance_index', 'label', 'fpn_level'}
    missing = required.difference(result)
    if missing:
        names = ', '.join(sorted(missing))
        raise KeyError(f'evidence result lacks identity fields: {names}')
    count = result['batch_index'].numel()
    for key in required:
        value = result[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(
                f'evidence {key} must be a one-dimensional tensor')
        if value.dtype != torch.long:
            raise ValueError(f'evidence {key} must use torch.long dtype')
        if value.numel() != count:
            raise ValueError('evidence identity fields must align')
    fpn_levels = result['fpn_level']
    fpn_level_count = len(_evidence_cfg()['roi_extractor']['featmap_strides'])
    if bool(((fpn_levels < 0) | (fpn_levels >= fpn_level_count)).any()):
        raise ValueError('evidence fpn_level is out of range')
    for key, value in result.items():
        if key in _ADAPTER_IDENTITY_KEYS:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError(f'evidence {key} must have a row dimension')
        if value.shape[0] != count:
            raise ValueError('evidence row fields must align')
    seen = set()
    for row_index in range(count):
        batch_index = int(result['batch_index'][row_index])
        instance_index = int(result['instance_index'][row_index])
        if not 0 <= batch_index < len(samples):
            raise ValueError('evidence batch_index is out of range')
        gt = samples[batch_index].gt_instances
        if not 0 <= instance_index < len(gt.labels):
            raise ValueError('evidence instance_index is out of range')
        identity = (batch_index, instance_index)
        if identity in seen:
            raise ValueError('evidence instance identities must be unique')
        seen.add(identity)
        if int(result['label'][row_index]) != int(gt.labels[instance_index]):
            raise ValueError('evidence labels must preserve GT identity')


def _evidence_by_image(result: Mapping, samples):
    _validate_evidence_identity(result, samples)
    grouped = [dict() for _ in samples]
    for row_index in range(result['batch_index'].numel()):
        batch_index = int(result['batch_index'][row_index])
        instance_index = int(result['instance_index'][row_index])
        grouped[batch_index][instance_index] = _row_evidence(result, row_index)
    return grouped


def _match_by_gt(pred, gt_boxes, gt_labels, score_threshold: float,
                 iou_threshold: float) -> Dict[int, Dict]:
    pred_boxes = _box_tensor(pred.bboxes)
    gt_boxes = _box_tensor(gt_boxes)
    matches = match_rotated_predictions(
        pred_boxes,
        pred.scores,
        pred.labels,
        gt_boxes,
        gt_labels,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    canonical_pred_boxes = _regularize_le90(pred_boxes)
    canonical_gt_boxes = _regularize_le90(gt_boxes)
    result = {}
    for offset in range(matches['gt_index'].numel()):
        gt_index = int(matches['gt_index'][offset])
        pred_index = int(matches['pred_index'][offset])
        pred_angle = canonical_pred_boxes[pred_index, 4]
        gt_angle = canonical_gt_boxes[gt_index, 4]
        error = periodic_angle_error(pred_angle, gt_angle, period=math.pi)
        error_c4 = periodic_angle_error(
            pred_angle, gt_angle, period=math.pi / 2)
        result[gt_index] = dict(
            pred_index=pred_index,
            pred_score=pred.scores[pred_index],
            rotated_iou=matches['iou'][offset],
            pred_angle=pred_angle,
            gt_angle=gt_angle,
            angle_error_deg=torch.rad2deg(error),
            angle_error_c4_deg=torch.rad2deg(error_c4))
    return result


def _geometry(gt_box: torch.Tensor) -> Dict:
    width = _scalar(gt_box[2])
    height = _scalar(gt_box[3])
    short_side = max(min(width, height), 1e-8)
    return dict(
        gt_width=width,
        gt_height=height,
        gt_area=width * height,
        gt_aspect_ratio=max(width, height) / short_side)


def _write_rows(stream, runner, model, evidence_module, run_name: str,
                max_images: Optional[int], score_threshold: float,
                iou_threshold: float) -> Dict[str, int]:
    row_count = 0
    image_count = 0
    matched_count = 0
    with torch.inference_mode():
        for data in runner.val_dataloader:
            remaining = (None if max_images is None else max_images -
                         image_count)
            if remaining is not None and remaining <= 0:
                break
            data = _limit_raw_batch(data, remaining)
            processed = model.data_preprocessor(data, training=False)
            if not isinstance(processed, Mapping):
                raise TypeError(
                    'model data preprocessor must return a mapping')
            inputs = processed['inputs']
            samples = list(processed['data_samples'])
            if remaining is not None and len(samples) > remaining:
                inputs = inputs[:remaining]
                samples = samples[:remaining]
            if not samples:
                continue
            # Prediction rescaling and every reported quantity use this
            # immutable original-coordinate snapshot.
            gt_snapshots = [
                dict(
                    image_id=str(sample.metainfo['img_id']),
                    boxes=_box_tensor(
                        sample.gt_instances.bboxes).detach().clone(),
                    labels=sample.gt_instances.labels.detach().clone())
                for sample in samples
            ]
            feature_gt_instances = _feature_space_gt_instances(samples)
            features = model.extract_feat(inputs)
            raw_evidence = evidence_module(
                features, feature_gt_instances,
                [sample.metainfo for sample in samples])
            evidence_by_image = _evidence_by_image(raw_evidence, samples)
            predictions = list(model.predict(inputs, samples, rescale=True))
            if len(predictions) != len(samples):
                raise ValueError('predictions and data samples must align')

            for batch_index, (gt_snapshot, prediction) in enumerate(
                    zip(gt_snapshots, predictions)):
                pred = prediction.pred_instances
                gt_boxes = _box_tensor(gt_snapshot['boxes'])
                gt_labels = gt_snapshot['labels']
                match_by_gt = _match_by_gt(pred, gt_boxes, gt_labels,
                                           score_threshold, iou_threshold)
                evidence_by_gt = evidence_by_image[batch_index]
                for gt_index, label in enumerate(gt_labels.tolist()):
                    match = match_by_gt.get(gt_index)
                    row = build_evidence_row(run_name, gt_snapshot['image_id'],
                                             gt_index, label,
                                             evidence_by_gt.get(gt_index),
                                             _geometry(gt_boxes[gt_index]),
                                             match)
                    stream.write(
                        json.dumps(
                            row,
                            allow_nan=False,
                            ensure_ascii=False,
                            sort_keys=True) + '\n')
                    row_count += 1
                    matched_count += int(match is not None)
                image_count += 1
            if max_images is not None and image_count >= max_images:
                break
    return dict(
        image_count=image_count,
        row_count=row_count,
        matched_count=matched_count)


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


def _publish_exclusive(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise FileExistsError(
            f'{destination} must not already exist') from error


def _rollback_own_link(source: Path, destination: Path) -> None:
    try:
        if _path_entry_exists(destination) and os.path.samefile(
                source, destination):
            destination.unlink()
    except OSError:
        pass


def _publish_pair(output_temp: Path, output_path: Path, manifest_temp: Path,
                  manifest_path: Path) -> None:
    _publish_exclusive(output_temp, output_path)
    try:
        _publish_exclusive(manifest_temp, manifest_path)
    except BaseException:
        _rollback_own_link(output_temp, output_path)
        raise


def _temporary_path(parent: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=parent, prefix=f'.{prefix}.', suffix='.tmp')
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def collect(args) -> Dict:
    """Collect evidence without mutating the model, checkpoint, or outputs."""
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    requested_output = Path(args.output)
    requested_manifest = requested_output.with_suffix('.manifest.json')
    output_path = requested_output.resolve()
    manifest_path = output_path.with_suffix('.manifest.json')
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError('config and checkpoint must exist')
    destinations = (requested_output, requested_manifest, output_path,
                    manifest_path)
    if any(_path_entry_exists(path) for path in destinations):
        raise FileExistsError('output and manifest must not already exist')

    run_name = _validated_run_name(getattr(args, 'run_name', None))
    max_images = _validated_max_images(getattr(args, 'max_images', None))
    score_threshold = _validated_probability(
        getattr(args, 'score_threshold', 0.05), 'score_threshold')
    iou_threshold = _validated_probability(
        getattr(args, 'iou_threshold', 0.5), 'iou_threshold')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_digest = sha256(config_path)
    checkpoint_digest = sha256(checkpoint_path)
    output_temp = None
    manifest_temp = None
    try:
        output_temp = _temporary_path(output_path.parent, output_path.name)
        with tempfile.TemporaryDirectory(
                prefix='orbdet_scqo_runner_') as runner_workspace:
            register_all_modules()
            cfg = Config.fromfile(config_path)
            _validate_hrsc_val_pipeline(cfg)
            cfg.load_from = None
            cfg.resume = False
            cfg.work_dir = runner_workspace
            runner = Runner.from_cfg(cfg)
            runner.load_checkpoint(str(checkpoint_path))
            if (sha256(config_path) != config_digest
                    or sha256(checkpoint_path) != checkpoint_digest):
                raise RuntimeError(
                    'config or checkpoint changed while loading the model')
            model = runner.model
            model.eval()
            evidence_module = MODELS.build(_evidence_cfg()).to(
                next(model.parameters()).device)
            evidence_module.eval()

            with output_temp.open('x', encoding='utf-8') as stream:
                counts = _write_rows(stream, runner, model, evidence_module,
                                     run_name, max_images, score_threshold,
                                     iou_threshold)
                stream.flush()
                os.fsync(stream.fileno())

        manifest = dict(
            schema_version=MANIFEST_SCHEMA_VERSION,
            row_schema_version=ROW_SCHEMA_VERSION,
            run_name=run_name,
            config=str(config_path),
            config_sha256=config_digest,
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=checkpoint_digest,
            output=str(output_path),
            output_sha256=sha256(output_temp),
            image_count=counts['image_count'],
            row_count=counts['row_count'],
            matched_count=counts['matched_count'],
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_images=max_images)
        manifest_temp = _temporary_path(output_path.parent, manifest_path.name)
        _write_json(manifest_temp, manifest)
        _publish_pair(output_temp, output_path, manifest_temp, manifest_path)
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
