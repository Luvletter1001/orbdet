#!/usr/bin/env python3
"""Offline audit for frozen low-rank orientation evidence JSONL.

Reads the immutable evidence produced by
``collect_low_rank_orientation_evidence.py`` and reports coverage, low-rank
vs detector angular errors, confidence quintiles, hard-error ranking metrics,
class / aspect / FPN stratification, class concentration, and the single P0
decision (Gate A). It never touches the model, checkpoint, or source data.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

LARGE_ERROR_DEG = 15.0
QUINTILE_COUNT = 5
ASPECT_BINS = ('[1,1.5)', '[1.5,3)', '[3,6)', '[6,inf)')

_BASE_REQUIRED = {
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched', 'gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio'
}
_MATCH_REQUIRED = {
    'pred_index', 'pred_score', 'rotated_iou', 'pred_angle', 'gt_angle',
    'e2_deg', 'e4_deg', 'large_angle_error'
}
_EVIDENCE_REQUIRED = {
    'low_rank_angle', 'low_rank_confidence', 'low_rank_anisotropy',
    'low_rank_energy', 'low_rank_valid', 'low_rank_sigma1',
    'low_rank_sigma2', 'fpn_level'
}
_NUMERIC_FIELDS = (
    'gt_width', 'gt_height', 'gt_area', 'gt_aspect_ratio', 'low_rank_angle',
    'low_rank_confidence', 'low_rank_anisotropy', 'low_rank_energy',
    'low_rank_sigma1', 'low_rank_sigma2', 'pred_score', 'rotated_iou',
    'pred_angle', 'gt_angle', 'e2_deg', 'e4_deg')


def load_rows(path) -> List[Dict]:
    """Load and strictly validate an evidence JSONL file."""
    path = Path(path)
    rows = []
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'line {line_number}: row must be an object')
            rows.append(row)
    _validate_rows(rows)
    return rows


def _validate_rows(rows: Sequence[Dict]) -> None:
    seen = set()
    for index, row in enumerate(rows):
        missing = _BASE_REQUIRED.difference(row)
        if missing:
            raise ValueError(
                f'row {index}: missing fields {sorted(missing)}')
        if bool(row['matched']):
            missing_match = _MATCH_REQUIRED.difference(row)
            if missing_match:
                raise ValueError(
                    f'row {index}: matched row missing {sorted(missing_match)}')
        if bool(row['evidence_available']):
            missing_evidence = _EVIDENCE_REQUIRED.difference(row)
            if missing_evidence:
                raise ValueError(
                    f'row {index}: available-evidence row missing '
                    f'{sorted(missing_evidence)}')
        identity = (row['run_name'], row['image_id'], row['gt_index'])
        if identity in seen:
            raise ValueError(f'duplicate row identity: {identity}')
        seen.add(identity)
        for key in _NUMERIC_FIELDS:
            if key in row:
                value = row[key]
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)) or not math.isfinite(value):
                    raise ValueError(
                        f'row {index}: field {key} must be finite number')


def quantile(values, q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError('cannot compute quantile of empty sequence')
    return float(np.quantile(array, q))


def wrap_period_rad(value: np.ndarray, period: float) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float64) + period / 2.0) % period
            ) - period / 2.0


def _is_valid_matched(row: Dict) -> bool:
    return bool(row['matched'] and row['evidence_available']
                and row['low_rank_valid'])


def _low_rank_self_error_deg(row: Dict) -> float:
    diff = row['low_rank_angle'] - row['gt_angle']
    wrapped = wrap_period_rad([diff], math.pi)[0]
    return abs(math.degrees(wrapped))


def binary_auroc(target, score) -> float:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError('AUROC/AUPRC require both positive and negative '
                         'examples')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_score = score[order]
    rank_values = np.arange(1, len(score) + 1, dtype=np.float64)
    start = 0
    while start < len(sorted_score):
        end = start + 1
        while (end < len(sorted_score)
               and sorted_score[end] == sorted_score[start]):
            end += 1
        ranks[order[start:end]] = rank_values[start:end].mean()
        start = end
    positive_rank_sum = ranks[target == 1].sum()
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2.0)
        / (positive * negative))


def binary_average_precision(target, score) -> float:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError('AUROC/AUPRC require both positive and negative '
                         'examples')
    order = np.argsort(-score, kind='mergesort')
    sorted_target = target[order]
    cumulative = np.cumsum(sorted_target)
    precision_at = cumulative / np.arange(1, len(sorted_target) + 1)
    recall_prev = np.concatenate(([0.0], cumulative[:-1] / positive))
    recall_now = cumulative / positive
    return float(
        np.sum((recall_now - recall_prev) * precision_at))


def aspect_bin(aspect_ratio: float) -> str:
    if aspect_ratio < 1.5:
        return '[1,1.5)'
    if aspect_ratio < 3.0:
        return '[1.5,3)'
    if aspect_ratio < 6.0:
        return '[3,6)'
    return '[6,inf)'


def _error_arrays(rows: Sequence[Dict]):
    low_rank_err = np.array(
        [_low_rank_self_error_deg(row) for row in rows], dtype=np.float64)
    detector_err = np.array([row['e2_deg'] for row in rows], dtype=np.float64)
    confidence = np.array(
        [row['low_rank_confidence'] for row in rows], dtype=np.float64)
    return low_rank_err, detector_err, confidence


def _confidence_quintiles(valid_rows: Sequence[Dict]) -> List[Dict]:
    if len(valid_rows) < QUINTILE_COUNT:
        raise ValueError(
            'need at least five valid matched rows for confidence quintiles')
    _, detector_err, confidence = _error_arrays(valid_rows)
    order = np.argsort(confidence, kind='mergesort')
    grouped_indices = np.array_split(order, QUINTILE_COUNT)
    bins = []
    for bin_index, indices in enumerate(grouped_indices):
        if len(indices) == 0:
            raise ValueError(f'quintile {bin_index} is empty')
        e2 = detector_err[indices]
        bins.append(
            dict(
                bin=bin_index,
                count=int(len(indices)),
                confidence_lo=float(confidence[indices].min()),
                confidence_hi=float(confidence[indices].max()),
                e2_p50=quantile(e2, 0.50),
                e2_p90=quantile(e2, 0.90),
                e2_mean=float(e2.mean())))
    return bins


def _stratify(valid_rows: Sequence[Dict], key_fn) -> Dict:
    groups: Dict[object, List[Dict]] = {}
    for row in valid_rows:
        groups.setdefault(key_fn(row), []).append(row)
    result = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        lr_err, det_err, _ = _error_arrays(group)
        result[key] = dict(
            valid_count=len(group),
            low_rank_p50=quantile(lr_err, 0.50),
            low_rank_p90=quantile(lr_err, 0.90),
            detector_p50=quantile(det_err, 0.50),
            detector_p90=quantile(det_err, 0.90))
    return result


def _class_concentration(valid_rows: Sequence[Dict]) -> Dict[str, float]:
    total = len(valid_rows)
    class_counts: Dict[int, int] = {}
    improved_counts: Dict[int, int] = {}
    improved_total = 0
    for row in valid_rows:
        label = int(row['label'])
        class_counts[label] = class_counts.get(label, 0) + 1
        if _low_rank_self_error_deg(row) < float(row['e2_deg']):
            improved_counts[label] = improved_counts.get(label, 0) + 1
            improved_total += 1
    valid_share = (max(class_counts.values()) / total) if total else 0.0
    improvement_share = (
        max(improved_counts.values()) / improved_total
        if improved_total else 0.0)
    return dict(
        valid_max_class_share=float(valid_share),
        p90_improvement_max_class_share=float(improvement_share))


def analyze(rows: Sequence[Dict]) -> Dict:
    """Compute the full offline audit summary from validated rows."""
    total = len(rows)
    matched_rows = [row for row in rows if bool(row['matched'])]
    evidence_rows = [
        row for row in rows if bool(row['evidence_available'])]
    valid_rows = [row for row in rows if _is_valid_matched(row)]
    if not matched_rows:
        raise ValueError('dataset has no matched rows')

    lr_err, det_err, confidence = _error_arrays(valid_rows)
    target = (det_err > LARGE_ERROR_DEG).astype(np.int64)
    if len(valid_rows) > 0:
        auroc = binary_auroc(target, 1.0 - confidence)
        auprc = binary_average_precision(target, 1.0 - confidence)
    else:
        raise ValueError('dataset has no valid+matched rows')

    summary = dict(
        counts=dict(
            total=total,
            matched=len(matched_rows),
            evidence_available=len(evidence_rows),
            valid_matched=len(valid_rows),
            coverage=(len(valid_rows) / len(matched_rows))
            if matched_rows else 0.0),
        low_rank_self_error=dict(
            p50=quantile(lr_err, 0.50), p90=quantile(lr_err, 0.90)),
        detector_e2=dict(
            p50=quantile(det_err, 0.50), p90=quantile(det_err, 0.90)),
        confidence_quintiles=_confidence_quintiles(valid_rows),
        hard_error=dict(
            threshold_deg=LARGE_ERROR_DEG,
            auroc=auroc,
            auprc=auprc,
            positive=int(target.sum()),
            negative=int((target == 0).sum())),
        by_class={
            str(key): value
            for key, value in _stratify(
                valid_rows, lambda row: int(row['label'])).items()
        },
        by_aspect_bin=_stratify(
            valid_rows, lambda row: aspect_bin(row['gt_aspect_ratio'])),
        by_fpn_level={
            int(key): value
            for key, value in _stratify(
                valid_rows, lambda row: int(row['fpn_level'])).items()
        },
        concentration=_class_concentration(valid_rows))
    evaluate = evaluate_gate_a(summary)
    summary['gate_a'] = evaluate
    return summary


def evaluate_gate_a(summary: Dict) -> Dict:
    """Apply the four conjunctive Gate A conditions."""
    counts = summary['counts']
    conditions = {}

    enough_matches = counts['matched'] >= 500
    aspect = summary.get('by_aspect_bin', {})
    non_empty_bins_supported = all(
        bin_['valid_count'] >= 50 for bin_ in aspect.values()) and bool(
            aspect)
    conditions['matched_and_bin_support'] = bool(
        enough_matches and non_empty_bins_supported)

    quintiles = summary['confidence_quintiles']
    bottom_p90 = quintiles[0]['e2_p90']
    top_p90 = quintiles[-1]['e2_p90']
    if bottom_p90 > 0:
        relative_gain = (bottom_p90 - top_p90) / bottom_p90
    else:
        relative_gain = 0.0
    conditions['top_quintile_p90_relative_gain_10pct'] = bool(
        relative_gain >= 0.10)

    lr_p90 = summary['low_rank_self_error']['p90']
    detector_p90 = summary['detector_e2']['p90']
    conditions['low_rank_p90_within_5deg_of_detector'] = bool(
        lr_p90 <= detector_p90 + 5.0)

    concentration = summary['concentration']
    conditions['class_not_concentrated'] = bool(
        concentration['valid_max_class_share'] <= 0.40
        and concentration['p90_improvement_max_class_share'] <= 0.40)

    passed = all(conditions.values())
    return dict(
        conditions=conditions,
        passed=passed,
        conclusion=('P0 pass / eligible for frozen-feature cross-fit only'
                    if passed else 'P0 fail / diagnostic-only'))


def render_markdown(summary: Dict) -> str:
    counts = summary['counts']
    gate = summary['gate_a']
    lines = [
        '# Low-Rank Orientation Evidence — P0 Frozen Audit',
        '',
        f"- matched: {counts['matched']}",
        f"- evidence_available: {counts['evidence_available']}",
        f"- valid+matched: {counts['valid_matched']}",
        f"- coverage (valid/matched): {counts['coverage']:.4f}",
        '',
        '## Angular error on valid+matched subset (degrees)',
        f"- low-rank self P50/P90: "
        f"{summary['low_rank_self_error']['p50']:.4f} / "
        f"{summary['low_rank_self_error']['p90']:.4f}",
        f"- detector e2 P50/P90: "
        f"{summary['detector_e2']['p50']:.4f} / "
        f"{summary['detector_e2']['p90']:.4f}",
        '',
        '## Confidence quintiles (low -> high confidence)',
        '| bin | count | e2 P50 | e2 P90 | e2 mean |',
        '|---|---|---|---|---|']
    for bin_ in summary['confidence_quintiles']:
        lines.append(
            f"| {bin_['bin']} | {bin_['count']} | {bin_['e2_p50']:.4f} "
            f"| {bin_['e2_p90']:.4f} | {bin_['e2_mean']:.4f} |")
    hard = summary['hard_error']
    lines += [
        '',
        '## Hard-error ranking (target e2 > 15°, score 1-confidence)',
        f"- AUROC: {hard['auroc']:.4f}",
        f"- AUPRC: {hard['auprc']:.4f}",
        f"- positive/negative: {hard['positive']}/{hard['negative']}",
        '',
        '## Concentration',
        f"- max class share of valid instances: "
        f"{summary['concentration']['valid_max_class_share']:.4f}",
        f"- max class share of P90 improvement: "
        f"{summary['concentration']['p90_improvement_max_class_share']:.4f}",
        '',
        '## Gate A',
    ]
    for name, ok in gate['conditions'].items():
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    lines += ['', f"**Conclusion: {gate['conclusion']}**", '']
    return '\n'.join(lines)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    rows = load_rows(args.evidence)
    summary = analyze(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'summary.json'
    markdown_path = output_dir / 'fres_low_rank_orientation_evidence_p0.md'
    with summary_path.open('w', encoding='utf-8') as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False)
    with markdown_path.open('w', encoding='utf-8') as stream:
        stream.write(render_markdown(summary))
    print(json.dumps(summary['gate_a'], ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
