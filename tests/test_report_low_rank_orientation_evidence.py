import json
import math

import pytest

from tools.analysis_tools import report_low_rank_orientation_evidence as R


def _row(index,
         confidence=0.5,
         e2_deg=10.0,
         lr_self_deg=5.0,
         label=0,
         aspect=2.0,
         level=1,
         matched=True,
         valid=True):
    gt_angle = 0.0
    return dict(
        run_name='r',
        image_id=f'I{index // 5}',
        gt_index=index % 5,
        label=label,
        evidence_available=True,
        matched=matched,
        low_rank_angle=math.radians(lr_self_deg),
        low_rank_confidence=confidence,
        low_rank_anisotropy=0.8,
        low_rank_energy=1.0,
        low_rank_valid=valid,
        low_rank_sigma1=2.0,
        low_rank_sigma2=0.1,
        fpn_level=level,
        gt_width=10.0,
        gt_height=10.0,
        gt_area=100.0,
        gt_aspect_ratio=aspect,
        pred_index=0,
        pred_score=0.9,
        rotated_iou=0.7,
        pred_angle=math.radians(e2_deg),
        gt_angle=gt_angle,
        e2_deg=e2_deg,
        e4_deg=min(e2_deg, 90.0 - e2_deg),
        large_angle_error=e2_deg > 15.0)


def _write(tmp_path, rows):
    path = tmp_path / 'evidence.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    return path


def test_quantiles_p50_p90_on_valid_matched_subset(tmp_path):
    rows = []
    for i in range(100):
        rows.append(_row(i, e2_deg=float(i), lr_self_deg=float(i) / 2))
    summary = R.analyze(R.load_rows(_write(tmp_path, rows)))
    assert summary['counts']['valid_matched'] == 100
    assert abs(summary['detector_e2']['p50'] - 49.5) < 1e-6
    assert abs(summary['detector_e2']['p90'] - 89.1) < 1e-6
    assert abs(summary['low_rank_self_error']['p50'] - 24.75) < 1e-6


def test_confidence_quintiles_are_equal_frequency_and_ordered(tmp_path):
    rows = []
    for i in range(100):
        confidence = i / 100.0
        rows.append(
            _row(i, confidence=confidence, e2_deg=float(99 - i)))
    summary = R.analyze(R.load_rows(_write(tmp_path, rows)))
    bins = summary['confidence_quintiles']
    assert len(bins) == 5
    assert sum(bin_['count'] for bin_ in bins) == 100
    assert all(bin_['count'] == 20 for bin_ in bins)
    for bin_ in bins:
        for key in ('e2_p50', 'e2_p90', 'e2_mean'):
            assert math.isfinite(bin_[key])
    # Lowest-confidence bin has the largest detector e2 P90.
    assert bins[0]['e2_p90'] > bins[-1]['e2_p90']


def test_class_concentration_reports_largest_share(tmp_path):
    rows = []
    for i in range(80):
        rows.append(
            _row(i, label=0, e2_deg=20.0 if i % 2 == 0 else 5.0))
    for i in range(80, 100):
        rows.append(
            _row(i, label=1, e2_deg=20.0 if i % 2 == 0 else 5.0))
    summary = R.analyze(R.load_rows(_write(tmp_path, rows)))
    assert abs(summary['concentration']['valid_max_class_share'] - 0.8) < 1e-9


def test_aspect_bins_and_fpn_levels(tmp_path):
    rows = [
        _row(0, aspect=1.2, level=0, e2_deg=20.0),
        _row(1, aspect=2.0, level=0, e2_deg=5.0),
        _row(2, aspect=4.0, level=0, e2_deg=20.0),
        _row(3, aspect=8.0, level=0, e2_deg=5.0),
        _row(4, level=4, e2_deg=20.0),
        _row(5, level=4, e2_deg=5.0)
    ]
    summary = R.analyze(R.load_rows(_write(tmp_path, rows)))
    aspect = summary['by_aspect_bin']
    assert set(aspect) == {'[1,1.5)', '[1.5,3)', '[3,6)', '[6,inf)'}
    levels = summary['by_fpn_level']
    assert set(levels) == {0, 4}


def test_hard_error_auroc_needs_both_classes(tmp_path):
    rows = [_row(i, e2_deg=5.0) for i in range(20)]
    with pytest.raises(ValueError):
        R.analyze(R.load_rows(_write(tmp_path, rows)))


def test_reject_missing_field(tmp_path):
    row = _row(0)
    del row['low_rank_confidence']
    with pytest.raises(ValueError):
        R.load_rows(_write(tmp_path, [row]))


def test_reject_duplicate_identity(tmp_path):
    rows = [_row(0), _row(0)]
    rows[1]['image_id'] = rows[0]['image_id']
    with pytest.raises(ValueError):
        R.load_rows(_write(tmp_path, rows))


def test_reject_non_finite(tmp_path):
    row = _row(0)
    row['e2_deg'] = float('nan')
    path = tmp_path / 'evidence.jsonl'
    # Standard JSON emits a bare NaN token; the loader must reject it.
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    with pytest.raises(ValueError):
        R.load_rows(path)


def _passing_summary():
    return dict(
        counts=dict(
            total=600,
            matched=600,
            evidence_available=580,
            valid_matched=560,
            coverage=560 / 600),
        low_rank_self_error=dict(p50=4.0, p90=8.0),
        detector_e2=dict(p50=6.0, p90=10.0),
        confidence_quintiles=[
            dict(count=112, e2_p50=12.0, e2_p90=20.0, e2_mean=12.0),
            dict(count=112, e2_p50=10.0, e2_p90=17.0, e2_mean=10.0),
            dict(count=112, e2_p50=8.0, e2_p90=14.0, e2_mean=8.0),
            dict(count=112, e2_p50=6.0, e2_p90=12.0, e2_mean=6.0),
            dict(count=112, e2_p50=5.0, e2_p90=10.0, e2_mean=5.0)],
        by_aspect_bin={
            '[1,1.5)': dict(valid_count=140),
            '[1.5,3)': dict(valid_count=140),
            '[3,6)': dict(valid_count=140),
            '[6,inf)': dict(valid_count=140)},
        concentration=dict(
            valid_max_class_share=0.30,
            p90_improvement_max_class_share=0.30))


def test_gate_a_passes_when_all_conditions_hold():
    result = R.evaluate_gate_a(_passing_summary())
    assert result['passed'] is True
    assert all(result['conditions'].values())


def test_gate_a_fails_on_each_broken_condition():
    # Condition 1: too few matches.
    summary = _passing_summary()
    summary['counts']['matched'] = 100
    assert R.evaluate_gate_a(summary)['passed'] is False

    # Condition 2: top quintile not at least 10% better than bottom.
    summary = _passing_summary()
    summary['confidence_quintiles'][0]['e2_p90'] = 10.0
    summary['confidence_quintiles'][-1]['e2_p90'] = 9.5
    assert R.evaluate_gate_a(summary)['passed'] is False

    # Condition 3: low-rank P90 more than 5 deg worse than detector.
    summary = _passing_summary()
    summary['low_rank_self_error']['p90'] = 16.0
    assert R.evaluate_gate_a(summary)['passed'] is False

    # Condition 4a: one class dominates valid instances.
    summary = _passing_summary()
    summary['concentration']['valid_max_class_share'] = 0.45
    assert R.evaluate_gate_a(summary)['passed'] is False

    # Condition 4b: one class dominates observed P90 improvement.
    summary = _passing_summary()
    summary['concentration']['p90_improvement_max_class_share'] = 0.45
    assert R.evaluate_gate_a(summary)['passed'] is False

    # Condition 1b: a non-empty aspect bin has too few valid instances.
    summary = _passing_summary()
    summary['by_aspect_bin']['[6,inf)']['valid_count'] = 10
    assert R.evaluate_gate_a(summary)['passed'] is False
