#!/usr/bin/env python3
"""Report the pre-registered SCQO Plan A evidence gate."""

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from mmrotate.evaluation.functional.scqo_diagnostics import (
    binary_average_precision, binary_auroc, cross_validated_logistic_probe,
    expected_calibration_error)

SUMMARY_SCHEMA_VERSION = 1
_SUMMARY_NAME = 'summary.json'
_MARKDOWN_NAME = 'fres_scqo_plan_a_hrsc_audit.md'
_LOCK_NAME = '.scqo-report.lock'
_INCLUSIVE_TOLERANCE = 1e-12
FEATURES = ('c2_determinantal', 'c2_spectral_tail', 'c2_fixed_space',
            'c2_q_gap', 'c4_determinantal', 'c4_spectral_tail',
            'c4_fixed_space', 'c4_q_gap', 'a2', 'a4', 'c4_negative_margin',
            'energy', 'variance', 'support_fraction')
_BASE_FIELDS = {
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched'
}
_EVIDENCE_FIELDS = set(FEATURES) | {
    'valid', 'fpn_level', 'gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio'
}
_REQUIRED_MATCH_FIELDS = {
    'rotated_iou', 'angle_error_deg', 'angle_error_c4_deg', 'large_angle_error'
}


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', nargs='+', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        result = int(value)
        if float(value) != result:
            raise ValueError(f'{name} must be an integer')
    else:
        raise ValueError(f'{name} must be an integer')
    if result < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return result


def _reject_json_constant(value: str):
    raise ValueError(f'non-finite JSON constant is forbidden: {value}')


def _assert_finite_json(value, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{location} keys must be strings')
            _assert_finite_json(item, f'{location}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f'{location}[{index}]')
    elif isinstance(value, Real) and not isinstance(value, (bool, Integral)):
        if not math.isfinite(float(value)):
            raise ValueError(f'{location} must be finite')


def _validate_row(row, location: str) -> Dict:
    if not isinstance(row, dict):
        raise ValueError(f'{location} must contain a JSON object')
    _assert_finite_json(row, location)
    missing = _BASE_FIELDS.difference(row)
    if missing:
        raise ValueError(
            f'{location} lacks required fields: {", ".join(sorted(missing))}')
    run_name = row['run_name']
    image_id = row['image_id']
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(f'{location}.run_name must be a non-empty string')
    if not isinstance(image_id, str) or not image_id:
        raise ValueError(f'{location}.image_id must be a non-empty string')
    row['gt_index'] = _integer(row['gt_index'], f'{location}.gt_index')
    row['label'] = _integer(row['label'], f'{location}.label')
    for name in ('evidence_available', 'matched'):
        if not isinstance(row[name], bool):
            raise ValueError(f'{location}.{name} must be boolean')
    for name in ('valid', 'large_angle_error'):
        if name in row and not isinstance(row[name], bool):
            raise ValueError(f'{location}.{name} must be boolean')

    if row['evidence_available']:
        missing = _EVIDENCE_FIELDS.difference(row)
        if missing:
            raise ValueError(f'{location} lacks evidence fields: '
                             f'{", ".join(sorted(missing))}')
        if not isinstance(row['valid'], bool):
            raise ValueError(f'{location}.valid must be boolean')
        row['fpn_level'] = _integer(row['fpn_level'], f'{location}.fpn_level')
        for name in FEATURES:
            _finite_number(row[name], f'{location}.{name}')
        for name in ('gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio'):
            if _finite_number(row[name], f'{location}.{name}') <= 0:
                raise ValueError(f'{location}.{name} must be positive')
        support_fraction = _finite_number(row['support_fraction'],
                                          f'{location}.support_fraction')
        if not 0 <= support_fraction <= 1:
            raise ValueError(f'{location}.support_fraction must lie in [0, 1]')

    if row['matched']:
        missing = _REQUIRED_MATCH_FIELDS.difference(row)
        if missing:
            raise ValueError(f'{location} lacks match fields: '
                             f'{", ".join(sorted(missing))}')
        if 'pred_index' in row:
            row['pred_index'] = _integer(row['pred_index'],
                                         f'{location}.pred_index')
        for name in ('rotated_iou', 'angle_error_deg', 'angle_error_c4_deg'):
            _finite_number(row[name], f'{location}.{name}')
        for name in ('pred_score', 'pred_angle', 'gt_angle'):
            if name in row:
                _finite_number(row[name], f'{location}.{name}')
        for name in ('pred_score', 'rotated_iou'):
            if name not in row:
                continue
            value = float(row[name])
            if not 0 <= value <= 1:
                raise ValueError(f'{location}.{name} must lie in [0, 1]')
        for name in ('angle_error_deg', 'angle_error_c4_deg'):
            if float(row[name]) < 0:
                raise ValueError(f'{location}.{name} must be non-negative')
        if not isinstance(row['large_angle_error'], bool):
            raise ValueError(f'{location}.large_angle_error must be boolean')
        expected_large = float(row['angle_error_deg']) > 15.0
        if row['large_angle_error'] != expected_large:
            raise ValueError(
                f'{location}.large_angle_error is inconsistent with '
                'angle_error_deg')
    return row


def _resolved_sources(paths: Sequence[Path]) -> Tuple[Path, ...]:
    resolved = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            path = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                f'evidence source does not exist: {path}') from error
        if not path.is_file():
            raise ValueError(
                f'evidence source must be a readable file: {path}')
        resolved.append(path)
    return tuple(sorted(resolved, key=str))


def load_rows(paths: Sequence[Path]):
    runs = defaultdict(list)
    seen = set()
    resolved = _resolved_sources(paths)
    for path in resolved:
        try:
            with path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, start=1):
                    location = f'{path}:{line_number}'
                    if not line.strip():
                        raise ValueError(
                            f'{location} blank JSONL rows are forbidden')
                    try:
                        row = json.loads(
                            line, parse_constant=_reject_json_constant)
                    except (json.JSONDecodeError, ValueError) as error:
                        raise ValueError(
                            f'{location} is not valid finite JSON: {error}') \
                            from error
                    row = _validate_row(row, location)
                    identity = (row['run_name'], row['image_id'],
                                int(row['gt_index']))
                    if identity in seen:
                        raise ValueError(
                            f'duplicate evidence identity: {identity!r}')
                    seen.add(identity)
                    runs[row['run_name']].append(row)
        except (OSError, UnicodeError) as error:
            raise OSError(f'cannot read evidence source: {path}') from error
    if not runs:
        raise ValueError('evidence sources contain no rows')
    for rows in runs.values():
        rows.sort(key=lambda row: (row['image_id'], int(row['gt_index'])))
    return dict(sorted(runs.items())), resolved


def _reliability_curve(target, probability, bins=10):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for index in range(bins):
        mask = ((probability >= edges[index]) &
                ((probability <= edges[index + 1]) if index == bins - 1 else
                 (probability < edges[index + 1])))
        result.append(
            dict(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=int(mask.sum()),
                confidence=(float(probability[mask].mean())
                            if mask.any() else None),
                error_rate=(float(target[mask].mean())
                            if mask.any() else None)))
    return result


def _probe(rows: Sequence[Mapping], feature_names: Sequence[str]):
    x = np.asarray([[float(row[name]) for name in feature_names]
                    for row in rows],
                   dtype=np.float64)
    target = np.asarray([float(row['angle_error_deg']) > 15.0 for row in rows],
                        dtype=np.int64)
    groups = np.asarray([row['image_id'] for row in rows], dtype=object)
    probability = cross_validated_logistic_probe(
        x, target, groups, folds=3, l2=1e-2, max_iter=100)
    return dict(
        probability=probability,
        auroc=binary_auroc(target, probability),
        auprc=binary_average_precision(target, probability),
        ece=expected_calibration_error(target, probability, bins=10),
        reliability=_reliability_curve(target, probability, bins=10))


def _masked_auroc(target, probability, mask):
    if int(mask.sum()) < 10 or np.unique(target[mask]).size != 2:
        return None
    return binary_auroc(target[mask], probability[mask])


def _tertile_aurocs(rows, target, probability, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    edges = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    index = np.digitize(values, edges, right=True)
    return [
        _masked_auroc(target, probability, index == bucket)
        for bucket in range(3)
    ]


def _category_aurocs(rows, target, probability, field):
    values = np.asarray([row[field] for row in rows], dtype=object)
    return {
        str(value): _masked_auroc(target, probability, values == value)
        for value in sorted(set(values.tolist()), key=str)
    }


def _angle_summary(values: Iterable[float]):
    values = np.asarray(list(values), dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9)),
        within_5=float((values <= 5.0).mean()),
        within_10=float((values <= 10.0).mean()),
        within_15=float((values <= 15.0).mean()))


def _reject_by_fpn(rows):
    available = [row for row in rows if row['evidence_available']]
    levels = sorted({int(row['fpn_level']) for row in available})
    result = {}
    for level in levels:
        selected = [row for row in available if int(row['fpn_level']) == level]
        result[str(level)] = dict(
            count=len(selected),
            reject_fraction=float(
                np.mean([not row['valid'] for row in selected])))
    return result


def _insufficient(rows, eligible, reason):
    return dict(
        gate_b='INSUFFICIENT',
        total_rows=len(rows),
        eligible_rows=len(eligible),
        reject_by_fpn=_reject_by_fpn(rows),
        reason=reason)


def _inclusive_at_least(value, threshold):
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=_INCLUSIVE_TOLERANCE)


def _gate_b(combined_auroc, best_baseline, texture_aurocs):
    if any(value is None for value in texture_aurocs):
        return 'INSUFFICIENT'
    gain = combined_auroc - best_baseline
    numeric_pass = (
        _inclusive_at_least(combined_auroc, 0.65)
        and _inclusive_at_least(gain, 0.05))
    return ('PASS' if numeric_pass
            and all(value > 0.5 for value in texture_aurocs) else 'FAIL')


def summarize_run(rows):
    eligible = [
        row for row in rows if row['evidence_available'] and row['matched']
        and float(row['rotated_iou']) >= 0.5 and row['valid']
    ]
    target = np.asarray(
        [float(row['angle_error_deg']) > 15.0 for row in eligible],
        dtype=np.int64)
    positive_count = int(target.sum())
    negative_count = len(eligible) - positive_count
    if len(eligible) < 60 or positive_count < 15 or negative_count < 15:
        return _insufficient(
            rows, eligible,
            'need at least 60 eligible rows and 15 examples per class')

    try:
        combined = _probe(eligible, FEATURES)
        energy = _probe(eligible, ('energy', ))
        variance = _probe(eligible, ('variance', ))
        aspect_ratio = _probe(eligible, ('gt_aspect_ratio', ))
    except (RuntimeError, ValueError) as error:
        return _insufficient(rows, eligible,
                             f'grouped probe is not computable: {error}')
    best_baseline = max(energy['auroc'], variance['auroc'])
    texture_aurocs = _tertile_aurocs(eligible, target, combined['probability'],
                                     'variance')
    gate = _gate_b(combined['auroc'], best_baseline, texture_aurocs)

    return dict(
        gate_b=gate,
        total_rows=len(rows),
        eligible_rows=len(eligible),
        large_error_rows=positive_count,
        small_error_rows=negative_count,
        combined_auroc=combined['auroc'],
        combined_auprc=combined['auprc'],
        combined_ece=combined['ece'],
        combined_reliability=combined['reliability'],
        energy_auroc=energy['auroc'],
        variance_auroc=variance['auroc'],
        aspect_ratio_auroc=aspect_ratio['auroc'],
        best_baseline_auroc=best_baseline,
        auroc_gain=combined['auroc'] - best_baseline,
        angle_error_e2=_angle_summary(row['angle_error_deg']
                                      for row in eligible),
        angle_error_e4=_angle_summary(row['angle_error_c4_deg']
                                      for row in eligible),
        stratified_aurocs=dict(
            texture=texture_aurocs,
            aspect_ratio=_tertile_aurocs(eligible, target,
                                         combined['probability'],
                                         'gt_aspect_ratio'),
            area=_tertile_aurocs(eligible, target, combined['probability'],
                                 'gt_area'),
            label=_category_aurocs(eligible, target, combined['probability'],
                                   'label'),
            fpn_level=_category_aurocs(eligible, target,
                                       combined['probability'], 'fpn_level')),
        reject_by_fpn=_reject_by_fpn(rows))


def _format_metric(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def _markdown(summary):
    lines = [
        '# SCQO Plan A HRSC 只读证据审计', '',
        '本报告只读取冻结 checkpoint 和 HRSC validation；没有训练、调参或测试集评估。', '',
        '| run | Gate B | eligible | AUROC | baseline | gain | e2 median | e2 P90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for name, item in summary['runs'].items():
        angle = item.get('angle_error_e2', {})
        lines.append(f'| {name} | {item["gate_b"]} | '
                     f'{item.get("eligible_rows", 0)} | '
                     f'{_format_metric(item.get("combined_auroc"))} | '
                     f'{_format_metric(item.get("best_baseline_auroc"))} | '
                     f'{_format_metric(item.get("auroc_gain"))} | '
                     f'{_format_metric(angle.get("median"), 2)} | '
                     f'{_format_metric(angle.get("p90"), 2)} |')
    lines.extend([
        '', 'Gate B 要求：combined AUROC ≥ 0.65、相对最佳能量/方差单变量基线'
        '提高 ≥ 0.05，并且低、中、高三个纹理桶的 AUROC 均 > 0.5。', '',
        'PASS 只授权继续讨论 Plan B；FAIL/INSUFFICIENT 均不得启动 E4。', ''
    ])
    return '\n'.join(lines)


def _report_paths(output_dir: Path):
    return output_dir / _SUMMARY_NAME, output_dir / _MARKDOWN_NAME


def _validate_output_directory(output_dir: Path) -> bool:
    """Validate the destination and return whether it already exists."""
    if not _path_entry_exists(output_dir):
        return False
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileExistsError(
            f'report output must be absent or an existing regular directory: '
            f'{output_dir}')
    existing = [
        path for path in _report_paths(output_dir) if _path_entry_exists(path)
    ]
    if existing:
        names = ', '.join(path.name for path in existing)
        raise FileExistsError(f'report outputs already exist: {names}')
    return True


def _same_entry(path: Path, identity) -> bool:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (identity.st_dev,
                                                identity.st_ino)


def _unlink_owned_entry(path: Path, identity) -> None:
    if _same_entry(path, identity):
        path.unlink()


def _rmdir_owned_entry(path: Path, identity) -> None:
    if _same_entry(path, identity):
        try:
            path.rmdir()
        except OSError:
            pass


def _write_staged_file(output_dir: Path, prefix: str, text: str):
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=output_dir)
    identity = os.fstat(descriptor)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        return path, identity
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        _unlink_owned_entry(path, identity)
        raise


def _write_report_directory(output_dir: Path, summary: Mapping) -> None:
    summary_text = json.dumps(
        summary, allow_nan=False, ensure_ascii=False, indent=2,
        sort_keys=True) + '\n'
    markdown_text = _markdown(summary)
    output_dir = Path(os.path.abspath(output_dir))
    existed = _validate_output_directory(output_dir)
    created_identity = None
    if not existed:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_dir.mkdir()
            created_identity = os.lstat(output_dir)
        except FileExistsError:
            _validate_output_directory(output_dir)

    lock_path = output_dir / _LOCK_NAME
    lock_descriptor = None
    lock_identity = None
    staged = []
    published = []
    try:
        try:
            lock_descriptor = os.open(lock_path,
                                      os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                      0o600)
        except FileExistsError as error:
            raise FileExistsError(
                f'report output is already being written: {output_dir}') \
                from error
        lock_identity = os.fstat(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = None
        _validate_output_directory(output_dir)

        staged.append(
            _write_staged_file(output_dir, f'.{_SUMMARY_NAME}.tmp-',
                               summary_text))
        staged.append(
            _write_staged_file(output_dir, f'.{_MARKDOWN_NAME}.tmp-',
                               markdown_text))
        for (source,
             source_identity), destination in zip(staged,
                                                  _report_paths(output_dir)):
            os.link(source, destination, follow_symlinks=False)
            published.append((destination, source_identity))
    except BaseException:
        for path, identity in reversed(published):
            _unlink_owned_entry(path, identity)
        raise
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        for path, identity in staged:
            _unlink_owned_entry(path, identity)
        if lock_identity is not None:
            _unlink_owned_entry(lock_path, lock_identity)
        if created_identity is not None:
            _rmdir_owned_entry(output_dir, created_identity)


def report(evidence_paths: Sequence[Path], output_dir: Path):
    output_dir = Path(os.path.abspath(output_dir))
    _validate_output_directory(output_dir)
    runs, sources = load_rows(evidence_paths)
    summary = dict(
        schema_version=SUMMARY_SCHEMA_VERSION,
        evidence=[str(path) for path in sources],
        runs={
            name: summarize_run(rows)
            for name, rows in runs.items()
        })
    json.dumps(summary, allow_nan=False)
    _write_report_directory(output_dir, summary)
    return summary


def main():
    args = parse_args()
    summary = report(args.evidence, args.output_dir)
    print(
        json.dumps(
            summary, allow_nan=False, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
