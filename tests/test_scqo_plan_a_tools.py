import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData

from mmdet.models.detectors.base import BaseDetector
from mmdet.structures import DetDataSample
from mmrotate.structures import RotatedBoxes

from tools.analysis_tools import scqo_collect_evidence as collector
from tools.analysis_tools import scqo_report_evidence as reporter
from tools.analysis_tools.scqo_collect_evidence import (_evidence_cfg, _scalar,
                                                        build_evidence_row,
                                                        collect, parse_args,
                                                        sha256)

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / 'tools/analysis_tools/scqo_collect_evidence.py'
REPORTER = ROOT / 'tools/analysis_tools/scqo_report_evidence.py'


def _arguments(tmp_path, *, max_images=None):
    config = tmp_path / 'config.py'
    checkpoint = tmp_path / 'checkpoint.pth'
    config.write_text('model = {}\n')
    checkpoint.write_bytes(b'frozen-checkpoint')
    return argparse.Namespace(
        config=config,
        checkpoint=checkpoint,
        run_name='v02',
        output=tmp_path / 'evidence.jsonl',
        max_images=max_images,
        score_threshold=0.05,
        iou_threshold=0.5)


def _instances(boxes, labels, scores=None):
    instances = SimpleNamespace(
        bboxes=RotatedBoxes(
            torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)),
        labels=torch.as_tensor(labels, dtype=torch.long))
    if scores is not None:
        instances.scores = torch.as_tensor(scores, dtype=torch.float32)
    return instances


def _sample(image_id, gt_boxes, gt_labels, pred_boxes, pred_scores,
            pred_labels):
    prediction = SimpleNamespace(
        pred_instances=_instances(pred_boxes, pred_labels, pred_scores))
    return SimpleNamespace(
        metainfo=dict(img_id=image_id, img_shape=(64, 64)),
        gt_instances=_instances(gt_boxes, gt_labels),
        prediction=prediction)


class _FakeModel(torch.nn.Module):

    def __init__(self, *, fail_on_extract=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.fail_on_extract = fail_on_extract
        self.eval_called = False
        self.preprocessed_batch_sizes = []
        self.predicted_image_ids = []

    def eval(self):
        self.eval_called = True
        return super().eval()

    def data_preprocessor(self, data, training=False):
        assert training is False
        assert torch.is_inference_mode_enabled()
        self.preprocessed_batch_sizes.append(len(data['data_samples']))
        return data

    def extract_feat(self, inputs):
        assert torch.is_inference_mode_enabled()
        if self.fail_on_extract:
            raise RuntimeError('synthetic inference failure')
        return (inputs.float(), )

    def predict(self, inputs, samples, rescale=False):
        assert rescale is False
        assert torch.is_inference_mode_enabled()
        assert inputs.shape[0] == len(samples)
        self.predicted_image_ids.extend(sample.metainfo['img_id']
                                        for sample in samples)
        return [sample.prediction for sample in samples]


class _BoxConvertingModel(_FakeModel):

    def __init__(self, results):
        super().__init__()
        self.results = results
        self.box_types_after_predict = None

    def predict(self, inputs, samples, rescale=False):
        assert rescale is False
        assert torch.is_inference_mode_enabled()
        self.predicted_image_ids.extend(sample.metainfo['img_id']
                                        for sample in samples)
        predictions = BaseDetector.add_pred_to_datasample(
            self, samples, self.results)
        self.box_types_after_predict = [(type(sample.gt_instances.bboxes),
                                         type(sample.pred_instances.bboxes))
                                        for sample in samples]
        return predictions


class _FakeEvidence(torch.nn.Module):

    def __init__(self, *, on_forward=None):
        super().__init__()
        self.on_forward = on_forward
        self.eval_called = False
        self.to_device = None
        self.seen_image_ids = []

    def to(self, device):
        self.to_device = torch.device(device)
        return self

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, features, gt_instances, metas):
        assert torch.is_inference_mode_enabled()
        if self.on_forward is not None:
            self.on_forward()
        self.seen_image_ids.extend(meta['img_id'] for meta in metas)
        rows = []
        for batch_index, meta in enumerate(metas):
            if meta['img_id'] in {'img-0', 'img-1'}:
                rows.append((batch_index, 0, 2 + 2 * batch_index,
                             0.7 + 0.1 * batch_index))
        return dict(
            batch_index=torch.tensor([row[0] for row in rows],
                                     dtype=torch.long),
            instance_index=torch.tensor([row[1] for row in rows],
                                        dtype=torch.long),
            label=torch.tensor(
                [gt_instances[row[0]].labels[row[1]] for row in rows],
                dtype=torch.long),
            square_hbox=torch.zeros((len(rows), 4)),
            fpn_level=torch.tensor([row[2] for row in rows], dtype=torch.long),
            energy=torch.tensor([row[3] for row in rows]),
            valid=torch.ones(len(rows), dtype=torch.bool))


class _RotatedBoxContractEvidence(_FakeEvidence):

    def __init__(self):
        super().__init__()
        self.box_types = []

    def forward(self, features, gt_instances, metas):
        self.box_types.extend(
            type(instances.bboxes) for instances in gt_instances)
        for instances in gt_instances:
            if not isinstance(instances.bboxes, RotatedBoxes):
                raise TypeError('evidence adapter requires RotatedBoxes')
            instances.bboxes.convert_to('hbox')
        return super().forward(features, gt_instances, metas)


def _install_fake_runtime(monkeypatch,
                          samples,
                          *,
                          fail_on_extract=False,
                          on_load=None,
                          on_evidence=None,
                          model=None,
                          evidence=None):
    if model is None:
        model = _FakeModel(fail_on_extract=fail_on_extract)
    if evidence is None:
        evidence = _FakeEvidence(on_forward=on_evidence)
    cfg = SimpleNamespace(load_from='old.pth', resume=True, work_dir='old')
    runner = SimpleNamespace(
        model=model,
        val_dataloader=[
            dict(
                inputs=torch.zeros(len(samples), 1, 1, 1),
                data_samples=samples)
        ],
        loaded_checkpoint=None,
        loaded_bytes=None)
    calls = dict(
        registered=0, config_path=None, evidence_cfg=None, work_dir=None)

    def register():
        calls['registered'] += 1

    class FakeConfig:

        @staticmethod
        def fromfile(path):
            calls['config_path'] = Path(path)
            return cfg

    class FakeRunner:

        @staticmethod
        def from_cfg(received_cfg):
            assert received_cfg is cfg
            calls['work_dir'] = received_cfg.work_dir
            return runner

    def load_checkpoint(path):
        runner.loaded_checkpoint = Path(path)
        runner.loaded_bytes = runner.loaded_checkpoint.read_bytes()
        if on_load is not None:
            on_load()

    def build(config):
        calls['evidence_cfg'] = config
        return evidence

    runner.load_checkpoint = load_checkpoint
    monkeypatch.setattr(collector, 'register_all_modules', register)
    monkeypatch.setattr(collector, 'Config', FakeConfig)
    monkeypatch.setattr(collector, 'Runner', FakeRunner)
    monkeypatch.setattr(collector, 'MODELS', SimpleNamespace(build=build))
    return dict(
        calls=calls, cfg=cfg, runner=runner, model=model, evidence=evidence)


def _single_sample(image_id='img-0'):
    return _sample(image_id, [[10, 10, 8, 8, 0]], [0], [[10, 10, 8, 8, 0]],
                   [0.9], [0])


def _real_box_sample():
    sample = DetDataSample()
    sample.set_metainfo(dict(img_id='img-0', img_shape=(64, 64)))
    ground_truth = InstanceData()
    ground_truth.bboxes = RotatedBoxes(
        torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.1]]))
    ground_truth.labels = torch.tensor([0], dtype=torch.long)
    sample.gt_instances = ground_truth

    prediction = InstanceData()
    prediction.bboxes = RotatedBoxes(
        torch.tensor([[10.0, 10.0, 8.0, 4.0, 0.2]]))
    prediction.scores = torch.tensor([0.9])
    prediction.labels = torch.tensor([0], dtype=torch.long)
    return sample, prediction


def test_collector_help_and_row_schema():
    completed = subprocess.run(
        [sys.executable, str(COLLECTOR), '--help'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True)

    assert completed.returncode == 0
    assert '--checkpoint' in completed.stdout
    row = build_evidence_row(
        run_name='v02',
        image_id='100000001',
        gt_index=0,
        label=0,
        evidence={
            'energy': 1.0,
            'valid': True
        },
        geometry={
            'gt_width': 8.0,
            'gt_height': 4.0,
            'gt_area': 32.0,
            'gt_aspect_ratio': 2.0,
        },
        match=None)
    assert row == {
        'run_name': 'v02',
        'image_id': '100000001',
        'gt_index': 0,
        'label': 0,
        'evidence_available': True,
        'matched': False,
        'energy': 1.0,
        'valid': True,
        'gt_width': 8.0,
        'gt_height': 4.0,
        'gt_area': 32.0,
        'gt_aspect_ratio': 2.0,
    }


def test_sha256_hashes_file_bytes(tmp_path):
    path = tmp_path / 'payload.bin'
    payload = b'orbdet-scqo\x00evidence'
    path.write_bytes(payload)

    assert sha256(path) == hashlib.sha256(payload).hexdigest()


def test_box_tensor_accepts_only_baseboxes_or_n_by_five_tensors():
    tensor = torch.zeros((2, 5))
    boxes = RotatedBoxes(tensor)

    assert collector._box_tensor(boxes) is boxes.tensor
    assert collector._box_tensor(tensor) is tensor
    with pytest.raises(TypeError, match='BaseBoxes or torch.Tensor'):
        collector._box_tensor(SimpleNamespace(tensor=tensor))
    with pytest.raises(ValueError, match=r'\[N, 5\]'):
        collector._box_tensor(torch.zeros((2, 4)))
    with pytest.raises(ValueError, match=r'\[N, 5\]'):
        collector._box_tensor(torch.zeros(5))


@pytest.mark.parametrize('value', [
    torch.tensor([1.0, 2.0]),
    math.nan,
    math.inf,
    -math.inf,
    torch.tensor(math.nan),
    '1.0',
    [1.0],
    object(),
])
def test_scalar_rejects_nonscalar_nonfinite_or_nonnumeric_values(value):
    with pytest.raises((TypeError, ValueError), match='scalar|finite|numeric'):
        _scalar(value)


@pytest.mark.parametrize('reserved_key', [
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched', 'pred_index'
])
def test_row_evidence_cannot_overwrite_reserved_identity(reserved_key):
    with pytest.raises(ValueError, match='reserved'):
        build_evidence_row('run', 'image', 0, 1, {reserved_key: 3}, None, None)


def test_fixed_evidence_configuration_matches_plan_a_contract():
    assert _evidence_cfg() == dict(
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


@pytest.mark.parametrize('extra', [
    ['--score-threshold', '-0.01'],
    ['--score-threshold', 'nan'],
    ['--score-threshold', '1.01'],
    ['--iou-threshold', '-0.01'],
    ['--iou-threshold', 'inf'],
    ['--iou-threshold', '1.01'],
    ['--max-images', '0'],
    ['--max-images', '-1'],
    ['--max-images', '1.5'],
])
def test_cli_rejects_invalid_thresholds_and_max_images(extra):
    argv = [
        'config.py', '--checkpoint', 'checkpoint.pth', '--run-name', 'run',
        '--output', 'evidence.jsonl', *extra
    ]

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_cli_accepts_probability_boundaries_and_positive_image_limit():
    args = parse_args([
        'config.py', '--checkpoint', 'checkpoint.pth', '--run-name', 'run',
        '--output', 'evidence.jsonl', '--score-threshold', '0',
        '--iou-threshold', '1', '--max-images', '1'
    ])

    assert args.score_threshold == 0.0
    assert args.iou_threshold == 1.0
    assert args.max_images == 1


@pytest.mark.parametrize('missing', ['config', 'checkpoint'])
def test_collector_requires_config_and_checkpoint_before_model_setup(
        tmp_path, monkeypatch, missing):
    args = _arguments(tmp_path)
    getattr(args, missing).unlink()
    monkeypatch.setattr(
        collector, 'register_all_modules',
        lambda: pytest.fail('model registration must not start'))

    with pytest.raises(
            FileNotFoundError, match='config and checkpoint must exist'):
        collect(args)


@pytest.mark.parametrize('existing', ['output', 'manifest'])
def test_collector_refuses_existing_destinations_before_loading_model(
        tmp_path, monkeypatch, existing):
    args = _arguments(tmp_path)
    destination = (
        args.output
        if existing == 'output' else args.output.with_suffix('.manifest.json'))
    destination.write_text('immutable\n')
    monkeypatch.setattr(
        collector, 'register_all_modules',
        lambda: pytest.fail('model registration must not start'))

    with pytest.raises(FileExistsError, match='must not already exist'):
        collect(
            argparse.Namespace(
                config=args.config,
                checkpoint=args.checkpoint,
                output=args.output))
    assert destination.read_text() == 'immutable\n'


@pytest.mark.parametrize('existing', ['output', 'manifest'])
def test_collector_refuses_dangling_destination_links_before_loading_model(
        tmp_path, monkeypatch, existing):
    args = _arguments(tmp_path)
    destination = (
        args.output
        if existing == 'output' else args.output.with_suffix('.manifest.json'))
    destination.symlink_to(tmp_path / f'missing-{existing}')
    monkeypatch.setattr(
        collector, 'register_all_modules',
        lambda: pytest.fail('model registration must not start'))

    with pytest.raises(FileExistsError, match='must not already exist'):
        collect(args)


@pytest.mark.parametrize(('field', 'value'), [
    ('score_threshold', -0.1),
    ('score_threshold', math.nan),
    ('score_threshold', 1.1),
    ('iou_threshold', -0.1),
    ('iou_threshold', math.inf),
    ('iou_threshold', 1.1),
    ('max_images', 0),
    ('max_images', -1),
    ('max_images', True),
])
def test_collect_rejects_invalid_direct_arguments_before_model_setup(
        tmp_path, monkeypatch, field, value):
    args = _arguments(tmp_path, max_images=1)
    setattr(args, field, value)
    monkeypatch.setattr(
        collector, 'register_all_modules',
        lambda: pytest.fail('model registration must not start'))

    with pytest.raises(ValueError, match=field):
        collect(args)


def test_collector_runs_frozen_inference_and_preserves_per_image_identity(
        tmp_path, monkeypatch):
    args = _arguments(tmp_path, max_images=2)
    samples = [
        _sample('img-0', [[10, 10, 8, 8, 0], [30, 30, 6, 4, 0.25]], [0, 1],
                [[10, 10, 8, 8, 0], [10, 10, 8, 8, math.pi / 2 - 0.05]],
                [0.99, 0.9], [3, 0]),
        _sample('img-1', [[20, 20, 4, 4, 0.2]], [2], [[20, 20, 4, 4, 0.3]],
                [0.8], [2]),
        _sample('img-must-not-be-consumed', [[40, 40, 4, 4, 0]], [4],
                [[40, 40, 4, 4, 0]], [0.7], [4]),
    ]
    runtime = _install_fake_runtime(monkeypatch, samples)

    manifest = collect(args)

    manifest_path = args.output.with_suffix('.manifest.json')
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    written_manifest = json.loads(manifest_path.read_text())
    assert manifest == written_manifest
    assert [(row['image_id'], row['gt_index'], row['label'])
            for row in rows] == [('img-0', 0, 0), ('img-0', 1, 1),
                                 ('img-1', 0, 2)]
    assert runtime['model'].preprocessed_batch_sizes == [2]
    assert runtime['model'].predicted_image_ids == ['img-0', 'img-1']
    assert runtime['evidence'].seen_image_ids == ['img-0', 'img-1']
    first, unmatched, second_image = rows
    assert first['evidence_available'] is True
    assert first['fpn_level'] == 2.0
    assert first['energy'] == pytest.approx(0.7)
    assert first['matched'] is True
    assert first['pred_index'] == 1
    assert first['pred_score'] == pytest.approx(0.9)
    assert first['rotated_iou'] > 0.9
    assert first['angle_error_deg'] == pytest.approx(87.1352, abs=1e-3)
    assert first['angle_error_c4_deg'] == pytest.approx(2.8648, abs=1e-3)
    assert first['large_angle_error'] is True
    assert first['gt_width'] == 8.0
    assert first['gt_height'] == 8.0
    assert first['gt_area'] == 64.0
    assert first['gt_aspect_ratio'] == 1.0
    assert unmatched['evidence_available'] is False
    assert unmatched['matched'] is False
    assert 'pred_index' not in unmatched
    assert second_image['fpn_level'] == 4.0
    assert second_image['pred_index'] == 0
    assert manifest['schema_version'] == 1
    assert manifest['row_schema_version'] == 1
    assert manifest['image_count'] == 2
    assert manifest['row_count'] == 3
    assert manifest['matched_count'] == 2
    assert manifest['score_threshold'] == 0.05
    assert manifest['iou_threshold'] == 0.5
    assert manifest['max_images'] == 2
    assert manifest['config'] == str(args.config.resolve())
    assert manifest['checkpoint'] == str(args.checkpoint.resolve())
    assert manifest['output'] == str(args.output.resolve())
    assert manifest['config_sha256'] == sha256(args.config)
    assert manifest['checkpoint_sha256'] == hashlib.sha256(
        runtime['runner'].loaded_bytes).hexdigest()
    assert manifest['output_sha256'] == sha256(args.output)
    assert runtime['calls']['registered'] == 1
    assert runtime['calls']['config_path'] == args.config.resolve()
    assert runtime['calls']['evidence_cfg'] == _evidence_cfg()
    assert runtime['cfg'].load_from is None
    assert runtime['cfg'].resume is False
    assert runtime['model'].eval_called
    assert runtime['evidence'].eval_called
    assert runtime['evidence'].to_device == torch.device('cpu')
    assert not Path(runtime['calls']['work_dir']).exists()
    assert not list(tmp_path.glob('.*.tmp'))


def test_collector_extracts_evidence_before_real_prediction_box_conversion(
        tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    sample, prediction = _real_box_sample()
    model = _BoxConvertingModel([prediction])
    evidence = _RotatedBoxContractEvidence()
    _install_fake_runtime(
        monkeypatch, [sample], model=model, evidence=evidence)

    manifest = collect(args)

    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert evidence.box_types == [RotatedBoxes]
    assert model.box_types_after_predict == [(torch.Tensor, torch.Tensor)]
    assert isinstance(sample.gt_instances.bboxes, torch.Tensor)
    assert isinstance(sample.pred_instances.bboxes, torch.Tensor)
    assert manifest['row_count'] == 1
    assert manifest['matched_count'] == 1
    assert args.output.with_suffix('.manifest.json').is_file()
    assert len(rows) == 1
    row = rows[0]
    assert row['image_id'] == 'img-0'
    assert row['gt_index'] == 0
    assert row['label'] == 0
    assert row['evidence_available'] is True
    assert row['fpn_level'] == 2.0
    assert row['energy'] == pytest.approx(0.7)
    assert row['matched'] is True
    assert row['pred_index'] == 0
    assert row['rotated_iou'] > 0.8
    assert row['angle_error_deg'] == pytest.approx(math.degrees(0.1), abs=1e-5)
    assert row['angle_error_c4_deg'] == pytest.approx(
        math.degrees(0.1), abs=1e-5)
    assert row['gt_width'] == 8.0
    assert row['gt_height'] == 4.0
    assert row['gt_area'] == 32.0
    assert row['gt_aspect_ratio'] == 2.0


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('batch_index', torch.tensor([0.9]), 'torch.long'),
    ('instance_index', torch.tensor([0.9]), 'torch.long'),
    ('label', torch.tensor([0.9]), 'torch.long'),
    ('fpn_level', torch.tensor([1.5]), 'torch.long'),
    ('fpn_level', torch.tensor([-1]), 'fpn_level'),
    ('fpn_level', torch.tensor([5]), 'fpn_level'),
])
def test_evidence_identity_requires_long_indices_and_valid_fpn_levels(
        field, value, message):
    sample = _single_sample()
    result = dict(
        batch_index=torch.tensor([0]),
        instance_index=torch.tensor([0]),
        label=torch.tensor([0]),
        square_hbox=torch.zeros((1, 4)),
        fpn_level=torch.tensor([0]),
        energy=torch.tensor([1.0]),
        valid=torch.tensor([True]))
    result[field] = value

    with pytest.raises(ValueError, match=message):
        collector._evidence_by_image(result, [sample])


def test_inference_failure_publishes_neither_output_nor_manifest_and_cleans_up(
        tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    runtime = _install_fake_runtime(
        monkeypatch, [_single_sample()], fail_on_extract=True)

    with pytest.raises(RuntimeError, match='synthetic inference failure'):
        collect(args)

    assert not args.output.exists()
    assert not args.output.with_suffix('.manifest.json').exists()
    assert not list(tmp_path.glob('.*.tmp'))
    assert not Path(runtime['calls']['work_dir']).exists()


def test_max_images_does_not_fetch_a_batch_beyond_the_limit(
        tmp_path, monkeypatch):
    args = _arguments(tmp_path, max_images=1)
    runtime = _install_fake_runtime(monkeypatch, [_single_sample()])
    first_batch = runtime['runner'].val_dataloader[0]

    def bounded_loader():
        yield first_batch
        pytest.fail('collector fetched a batch beyond max_images')

    runtime['runner'].val_dataloader = bounded_loader()

    manifest = collect(args)

    assert manifest['image_count'] == 1


@pytest.mark.parametrize('changed_file', ['config', 'checkpoint'])
def test_collector_rejects_inputs_changed_while_checkpoint_is_loaded(
        tmp_path, monkeypatch, changed_file):
    args = _arguments(tmp_path)

    def mutate_loaded_input():
        path = getattr(args, changed_file)
        if changed_file == 'config':
            path.write_text('model = {"changed": True}\n')
        else:
            path.write_bytes(b'changed-checkpoint')

    runtime = _install_fake_runtime(
        monkeypatch, [_single_sample()], on_load=mutate_loaded_input)

    with pytest.raises(RuntimeError, match='changed while loading'):
        collect(args)

    assert not args.output.exists()
    assert not args.output.with_suffix('.manifest.json').exists()
    assert not Path(runtime['calls']['work_dir']).exists()


def test_atomic_publication_never_overwrites_a_racing_output(
        tmp_path, monkeypatch):
    args = _arguments(tmp_path)

    def create_racing_output():
        args.output.write_text('racing-writer\n')

    _install_fake_runtime(
        monkeypatch, [_single_sample()], on_evidence=create_racing_output)

    with pytest.raises(FileExistsError):
        collect(args)

    assert args.output.read_text() == 'racing-writer\n'
    assert not args.output.with_suffix('.manifest.json').exists()
    assert not list(tmp_path.glob('.*.tmp'))


def test_collector_cannot_start_training_or_overwrite_status():
    text = COLLECTOR.read_text()
    forbidden = ('tools/train.py', 'train_step(', 'optim_wrapper', 'tmux',
                 'nohup', 'FORMAL_TRAINING_NOT_STARTED.md')
    assert all(token not in text for token in forbidden)


def _synthetic_report_rows(run_name='synthetic'):
    rows = []
    for image_index in range(30):
        for local in range(4):
            target = local >= 2
            rows.append(
                dict(
                    run_name=run_name,
                    image_id=str(image_index),
                    gt_index=local,
                    label=0,
                    evidence_available=True,
                    matched=True,
                    rotated_iou=0.8,
                    angle_error_deg=20.0 if target else 2.0,
                    angle_error_c4_deg=5.0 if target else 2.0,
                    large_angle_error=target,
                    c2_determinantal=0.1,
                    c2_spectral_tail=0.1,
                    c2_fixed_space=0.1,
                    c2_q_gap=0.9,
                    c4_determinantal=float(target),
                    c4_spectral_tail=float(target),
                    c4_fixed_space=float(target),
                    c4_q_gap=float(not target),
                    energy=1.0,
                    variance=0.5 + 0.001 * image_index,
                    energy_guard=0.0,
                    variance_guard=0.0,
                    support_fraction=1.0,
                    fpn_level=image_index % 3,
                    gt_width=8.0 + image_index,
                    gt_height=4.0,
                    gt_area=(8.0 + image_index) * 4.0,
                    gt_aspect_ratio=(8.0 + image_index) / 4.0,
                    a2=float(not target),
                    a4=float(target),
                    negative_energy=1.0,
                    c4_negative_margin=float(target),
                    valid=True))
    return rows


def _write_jsonl(path, rows):
    path.write_text(
        ''.join(
            json.dumps(row, allow_nan=False, sort_keys=True) + '\n'
            for row in rows),
        encoding='utf-8')


def _run_reporter(source_paths, output):
    return subprocess.run([
        sys.executable,
        str(REPORTER), *(str(path) for path in source_paths), '--output-dir',
        str(output)
    ],
                          cwd=ROOT,
                          check=False,
                          capture_output=True,
                          env={
                              **os.environ, 'PYTHONPATH': str(ROOT)
                          },
                          text=True)


def test_reporter_writes_pass_evidence_without_persisting_probe(tmp_path):
    source = tmp_path / 'rows.jsonl'
    _write_jsonl(source, _synthetic_report_rows())
    output = tmp_path / 'report'

    completed = _run_reporter([source], output)

    assert completed.returncode == 0, completed.stderr
    summary_text = (output / 'summary.json').read_text(encoding='utf-8')
    summary = json.loads(summary_text)
    result = summary['runs']['synthetic']
    assert result['gate_b'] == 'PASS'
    assert result['combined_auroc'] >= 0.65
    assert result['auroc_gain'] >= 0.05
    assert all(value > 0.5 for value in result['stratified_aurocs']['texture'])
    assert set(result['stratified_aurocs']) == {
        'texture', 'aspect_ratio', 'area', 'label', 'fpn_level'
    }
    assert result['angle_error_e2']['median'] == pytest.approx(11.0)
    assert result['angle_error_e4']['median'] == pytest.approx(3.5)
    reliability = result['combined_reliability']
    assert sum(item['count']
               for item in reliability) == result['eligible_rows']
    recomputed_ece = sum(item['count'] / result['eligible_rows'] *
                         abs(item['error_rate'] - item['confidence'])
                         for item in reliability if item['count'])
    assert recomputed_ece == pytest.approx(result['combined_ece'])
    markdown = (output /
                'fres_scqo_plan_a_hrsc_audit.md').read_text(encoding='utf-8')
    assert '冻结 checkpoint' in markdown
    assert 'HRSC validation' in markdown
    assert 'PASS 只授权继续讨论 Plan B' in markdown
    assert 'FAIL/INSUFFICIENT 均不得启动 E4' in markdown
    assert 'NaN' not in summary_text
    assert not list(output.glob('*.pth'))
    assert not list(output.glob('*.pkl'))

    repeated = _run_reporter([source], output)
    assert repeated.returncode != 0
    assert 'already exist' in repeated.stderr


def _minimal_row(run_name='run', image_id='image', gt_index=0):
    return dict(
        run_name=run_name,
        image_id=image_id,
        gt_index=gt_index,
        label=0,
        evidence_available=False,
        matched=False)


def _load_rows(tmp_path, rows, name='rows.jsonl'):
    source = tmp_path / name
    _write_jsonl(source, rows)
    return reporter.load_rows([source])


def test_reporter_help_lists_required_inputs():
    completed = subprocess.run(
        [sys.executable, str(REPORTER), '--help'],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={
            **os.environ, 'PYTHONPATH': str(ROOT)
        },
        text=True)

    assert completed.returncode == 0
    assert 'evidence' in completed.stdout
    assert '--output-dir' in completed.stdout


def test_reporter_merges_same_run_deterministically_across_sources(tmp_path):
    rows = [_minimal_row(image_id=str(index)) for index in range(20)]
    first = tmp_path / 'b.jsonl'
    second = tmp_path / 'a.jsonl'
    _write_jsonl(first, rows[::2])
    _write_jsonl(second, rows[1::2])

    forward = reporter.report([first, second], tmp_path / 'forward')
    reverse = reporter.report([second, first], tmp_path / 'reverse')

    assert forward == reverse
    assert forward['evidence'] == sorted(forward['evidence'])
    assert forward['runs']['run']['total_rows'] == 20
    assert forward['runs']['run']['gate_b'] == 'INSUFFICIENT'
    assert (tmp_path / 'forward' /
            'summary.json').read_bytes() == (tmp_path / 'reverse' /
                                             'summary.json').read_bytes()


def test_reporter_rejects_duplicate_identity_across_sources(tmp_path):
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    row = _minimal_row()
    _write_jsonl(first, [row])
    _write_jsonl(second, [row])

    with pytest.raises(ValueError, match='duplicate evidence identity'):
        reporter.report([first, second], tmp_path / 'report')

    assert not (tmp_path / 'report').exists()


@pytest.mark.parametrize(('payload', 'message'), [
    ('', 'no rows'),
    ('\n', 'blank JSONL'),
    ('not-json\n', 'valid finite JSON'),
    ('[]\n', 'JSON object'),
    ('{"run_name": NaN}\n', 'non-finite JSON'),
])
def test_reporter_rejects_empty_blank_malformed_or_nonfinite_jsonl(
        tmp_path, payload, message):
    source = tmp_path / 'bad.jsonl'
    source.write_text(payload, encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        reporter.load_rows([source])


def test_reporter_rejects_missing_or_non_file_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match='does not exist'):
        reporter.load_rows([tmp_path / 'missing.jsonl'])
    with pytest.raises(ValueError, match='readable file'):
        reporter.load_rows([tmp_path])


@pytest.mark.parametrize('field', [
    'run_name', 'image_id', 'gt_index', 'label', 'evidence_available',
    'matched'
])
def test_reporter_requires_base_identity_fields(tmp_path, field):
    row = _minimal_row()
    del row[field]

    with pytest.raises(ValueError, match='required fields'):
        _load_rows(tmp_path, [row])


@pytest.mark.parametrize(('field', 'value'), [
    ('evidence_available', 1),
    ('matched', 0),
    ('valid', 1),
    ('large_angle_error', 1),
])
def test_reporter_requires_strict_boolean_fields_even_when_optional(
        tmp_path, field, value):
    row = _minimal_row()
    row[field] = value

    with pytest.raises(ValueError, match='boolean'):
        _load_rows(tmp_path, [row])


@pytest.mark.parametrize('field', [
    *reporter.FEATURES, 'valid', 'fpn_level', 'gt_width', 'gt_height',
    'gt_area', 'gt_aspect_ratio'
])
def test_reporter_requires_complete_available_evidence(tmp_path, field):
    row = _synthetic_report_rows()[0]
    del row[field]

    with pytest.raises(ValueError, match='evidence fields'):
        _load_rows(tmp_path, [row])


@pytest.mark.parametrize('field', [
    'rotated_iou', 'angle_error_deg', 'angle_error_c4_deg', 'large_angle_error'
])
def test_reporter_requires_complete_match_fields(tmp_path, field):
    row = _synthetic_report_rows()[0]
    del row[field]

    with pytest.raises(ValueError, match='match fields'):
        _load_rows(tmp_path, [row])


def test_reporter_rejects_nonfinite_values_in_unrecognized_fields(tmp_path):
    source = tmp_path / 'nonfinite.jsonl'
    source.write_text(
        json.dumps(_minimal_row())[:-1] + ', "extra": Infinity}\n',
        encoding='utf-8')

    with pytest.raises(ValueError, match='non-finite JSON'):
        reporter.load_rows([source])


def test_reporter_eligibility_and_fpn_rejection_use_registered_contract():
    rows = _synthetic_report_rows()[:4]
    rows[0]['evidence_available'] = False
    rows[1]['matched'] = False
    rows[2]['rotated_iou'] = 0.49
    rows[3]['valid'] = False

    result = reporter.summarize_run(rows)

    assert result['eligible_rows'] == 0
    assert result['gate_b'] == 'INSUFFICIENT'
    assert sum(item['count'] for item in result['reject_by_fpn'].values()) == 3
    assert sum(
        item['count'] * item['reject_fraction']
        for item in result['reject_by_fpn'].values()) == pytest.approx(1.0)


def test_reporter_requires_fifteen_examples_in_each_error_class():
    rows = _synthetic_report_rows()[:60]
    changed = 0
    for row in rows:
        if row['large_angle_error'] and changed < 16:
            row['large_angle_error'] = False
            row['angle_error_deg'] = 2.0
            changed += 1

    result = reporter.summarize_run(rows)

    assert len(rows) == 60
    assert sum(row['large_angle_error'] for row in rows) == 14
    assert result['gate_b'] == 'INSUFFICIENT'
    assert '15 examples per class' in result['reason']


def test_reporter_passes_image_identity_as_group_to_probe(monkeypatch):
    rows = _synthetic_report_rows()
    captured = []

    def capture_groups(features, target, groups, **kwargs):
        captured.append(tuple(groups))
        return np.linspace(0.01, 0.99, len(target))

    monkeypatch.setattr(reporter, 'cross_validated_logistic_probe',
                        capture_groups)

    reporter._probe(rows, ('energy', ))

    assert captured == [tuple(row['image_id'] for row in rows)]
    assert len(set(captured[0])) == 30


def test_exact_gate_b_thresholds():
    assert reporter._gate_b(0.65, 0.60, [0.51, 0.51, 0.51]) == 'PASS'
    assert reporter._gate_b(0.70, 0.65, [0.51, 0.51, 0.51]) == 'PASS'
    assert reporter._gate_b(0.85, 0.80, [0.51, 0.51, 0.51]) == 'PASS'
    assert reporter._gate_b(0.6499, 0.50, [0.51, 0.51, 0.51]) == 'FAIL'
    assert reporter._gate_b(0.70, 0.650001, [0.51, 0.51, 0.51]) == 'FAIL'
    assert reporter._gate_b(0.70, 0.60, [0.50, 0.51, 0.51]) == 'FAIL'
    assert reporter._gate_b(0.70, 0.60, [None, 0.51, 0.51]) == 'INSUFFICIENT'


def _fake_probe(rows, feature_names):
    target = np.asarray([float(row['angle_error_deg']) > 15.0 for row in rows],
                        dtype=np.int64)
    if tuple(feature_names) == reporter.FEATURES:
        probability = np.where(target == 1, 0.9, 0.1)
        auroc = 1.0
    else:
        probability = np.full(target.shape, 0.5)
        auroc = 0.5
    return dict(
        probability=probability,
        auroc=auroc,
        auprc=reporter.binary_average_precision(target, probability),
        ece=reporter.expected_calibration_error(target, probability),
        reliability=reporter._reliability_curve(target, probability))


def test_tied_or_one_class_variance_tertiles_are_insufficient(monkeypatch):
    monkeypatch.setattr(reporter, '_probe', _fake_probe)
    tied = _synthetic_report_rows()
    for row in tied:
        row['variance'] = 0.5

    tied_result = reporter.summarize_run(tied)

    assert tied_result['gate_b'] == 'INSUFFICIENT'
    assert tied_result['stratified_aurocs']['texture'].count(None) == 2

    one_class = _synthetic_report_rows()
    for row in one_class:
        row['variance'] = float(row['large_angle_error'])

    one_class_result = reporter.summarize_run(one_class)

    assert one_class_result['gate_b'] == 'INSUFFICIENT'
    assert None in one_class_result['stratified_aurocs']['texture']


def test_reporter_marks_nonseparating_evidence_fail(monkeypatch):
    rows = _synthetic_report_rows()

    def chance_probe(rows, feature_names):
        target = np.asarray(
            [float(row['angle_error_deg']) > 15.0 for row in rows],
            dtype=np.int64)
        probability = np.full(target.shape, 0.5)
        return dict(
            probability=probability,
            auroc=0.5,
            auprc=0.5,
            ece=0.0,
            reliability=reporter._reliability_curve(target, probability))

    monkeypatch.setattr(reporter, '_probe', chance_probe)

    result = reporter.summarize_run(rows)

    assert result['gate_b'] == 'FAIL'


def test_reporter_cli_writes_into_existing_evidence_directory(tmp_path):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    manifest = output / 'evidence.manifest.json'
    unrelated = output / 'notes.txt'
    _write_jsonl(source, [_minimal_row()])
    manifest.write_text('{"immutable": true}\n', encoding='utf-8')
    unrelated.write_text('preserve-me\n', encoding='utf-8')
    evidence_bytes = source.read_bytes()
    manifest_bytes = manifest.read_bytes()

    completed = _run_reporter([source], output)

    assert completed.returncode == 0, completed.stderr
    assert source.read_bytes() == evidence_bytes
    assert manifest.read_bytes() == manifest_bytes
    assert unrelated.read_text(encoding='utf-8') == 'preserve-me\n'
    assert (output / 'summary.json').is_file()
    assert (output / 'fres_scqo_plan_a_hrsc_audit.md').is_file()


@pytest.mark.parametrize('destination_kind',
                         ['file', 'directory_symlink', 'dangling_symlink'])
def test_reporter_refuses_file_or_symlink_output_path(tmp_path,
                                                      destination_kind):
    source = tmp_path / 'rows.jsonl'
    _write_jsonl(source, [_minimal_row()])
    output = tmp_path / 'report'
    if destination_kind == 'file':
        output.write_text('keep-me\n', encoding='utf-8')
    elif destination_kind == 'directory_symlink':
        target = tmp_path / 'target'
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    elif destination_kind == 'dangling_symlink':
        output.symlink_to(tmp_path / 'missing-target')

    with pytest.raises(FileExistsError, match='regular directory'):
        reporter.report([source], output)

    assert os.path.lexists(output)


@pytest.mark.parametrize('report_name',
                         ['summary.json', 'fres_scqo_plan_a_hrsc_audit.md'])
@pytest.mark.parametrize('entry_kind', ['file', 'directory', 'dangling_link'])
def test_reporter_refuses_existing_report_entries_in_evidence_directory(
        tmp_path, report_name, entry_kind):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    destination = output / report_name
    if entry_kind == 'file':
        destination.write_text('preserve-me\n', encoding='utf-8')
    elif entry_kind == 'directory':
        destination.mkdir()
    else:
        destination.symlink_to(output / 'missing-target')

    with pytest.raises(FileExistsError, match='already exist'):
        reporter.report([source], output)

    assert os.path.lexists(destination)


def test_reporter_publishes_neither_file_if_rendering_fails(
        tmp_path, monkeypatch):
    source = tmp_path / 'rows.jsonl'
    _write_jsonl(source, [_minimal_row()])
    output = tmp_path / 'report'

    def fail_render(summary):
        raise RuntimeError('synthetic rendering failure')

    monkeypatch.setattr(reporter, '_markdown', fail_render)

    with pytest.raises(RuntimeError, match='synthetic rendering failure'):
        reporter.report([source], output)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob('.*.tmp-*'))


def test_reporter_rolls_back_first_report_when_second_link_fails(
        tmp_path, monkeypatch):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    evidence_bytes = source.read_bytes()
    real_link = reporter.os.link
    calls = 0

    def fail_second_link(source_path, destination_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('synthetic second-link failure')
        return real_link(source_path, destination_path, **kwargs)

    monkeypatch.setattr(reporter.os, 'link', fail_second_link)

    with pytest.raises(OSError, match='synthetic second-link failure'):
        reporter.report([source], output)

    assert source.read_bytes() == evidence_bytes
    assert not os.path.lexists(output / 'summary.json')
    assert not os.path.lexists(output / 'fres_scqo_plan_a_hrsc_audit.md')
    assert not list(output.glob('.*.tmp-*'))
    assert not os.path.lexists(output / '.scqo-report.lock')


@pytest.mark.parametrize('racing_kind', ['file', 'directory', 'symlink'])
def test_reporter_never_overwrites_a_racing_first_report_entry(
        tmp_path, monkeypatch, racing_kind):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    real_link = reporter.os.link
    racing_path = output / 'summary.json'

    def race_first_link(source_path, destination_path, **kwargs):
        destination_path = Path(destination_path)
        if destination_path.name == 'summary.json':
            if racing_kind == 'file':
                destination_path.write_text('racer\n', encoding='utf-8')
            elif racing_kind == 'directory':
                destination_path.mkdir()
            else:
                destination_path.symlink_to(output / 'missing-racer')
        return real_link(source_path, destination_path, **kwargs)

    monkeypatch.setattr(reporter.os, 'link', race_first_link)

    with pytest.raises(FileExistsError):
        reporter.report([source], output)

    assert os.path.lexists(racing_path)
    if racing_kind == 'file':
        assert racing_path.read_text(encoding='utf-8') == 'racer\n'
    assert not os.path.lexists(output / 'fres_scqo_plan_a_hrsc_audit.md')
    assert not list(output.glob('.*.tmp-*'))
    assert not os.path.lexists(output / '.scqo-report.lock')


def test_reporter_rolls_back_our_first_link_but_preserves_racing_second_link(
        tmp_path, monkeypatch):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    real_link = reporter.os.link
    markdown = output / 'fres_scqo_plan_a_hrsc_audit.md'

    def race_second_link(source_path, destination_path, **kwargs):
        destination_path = Path(destination_path)
        if destination_path.name == markdown.name:
            destination_path.symlink_to(output / 'missing-racer')
        return real_link(source_path, destination_path, **kwargs)

    monkeypatch.setattr(reporter.os, 'link', race_second_link)

    with pytest.raises(FileExistsError):
        reporter.report([source], output)

    assert not os.path.lexists(output / 'summary.json')
    assert markdown.is_symlink()
    assert not list(output.glob('.*.tmp-*'))
    assert not os.path.lexists(output / '.scqo-report.lock')


def test_reporter_preserves_racing_directory_replacing_created_output(
        tmp_path, monkeypatch):
    source = tmp_path / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    output = tmp_path / 'audit'
    displaced = tmp_path / 'displaced-owned-directory'
    racer_identity = {}

    def replace_directory_then_fail(path, flags, mode=0o777, *, dir_fd=None):
        assert Path(path) == output / '.scqo-report.lock'
        output.rename(displaced)
        output.mkdir()
        racer_identity['value'] = os.lstat(output)
        raise OSError('synthetic lock-open failure')

    monkeypatch.setattr(reporter.os, 'open', replace_directory_then_fail)

    with pytest.raises(OSError, match='synthetic lock-open failure'):
        reporter.report([source], output)

    assert output.is_dir()
    current = os.lstat(output)
    expected = racer_identity['value']
    assert (current.st_dev, current.st_ino) == (expected.st_dev,
                                                expected.st_ino)
    assert displaced.is_dir()
    assert not list(output.iterdir())


def test_reporter_preserves_racing_temp_replacement_on_fsync_failure(
        tmp_path, monkeypatch):
    output = tmp_path / 'audit'
    output.mkdir()
    source = output / 'evidence.jsonl'
    _write_jsonl(source, [_minimal_row()])
    evidence_bytes = source.read_bytes()
    racer = {}

    def replace_temp_then_fail(descriptor):
        candidates = list(output.glob('.summary.json.tmp-*'))
        assert len(candidates) == 1
        path = candidates[0]
        path.unlink()
        path.write_text('racing-temp\n', encoding='utf-8')
        racer['path'] = path
        racer['identity'] = os.lstat(path)
        raise OSError('synthetic staged fsync failure')

    monkeypatch.setattr(reporter.os, 'fsync', replace_temp_then_fail)

    with pytest.raises(OSError, match='synthetic staged fsync failure'):
        reporter.report([source], output)

    racer_path = racer['path']
    assert racer_path.read_text(encoding='utf-8') == 'racing-temp\n'
    current = os.lstat(racer_path)
    expected = racer['identity']
    assert (current.st_dev, current.st_ino) == (expected.st_dev,
                                                expected.st_ino)
    assert source.read_bytes() == evidence_bytes
    assert not os.path.lexists(output / 'summary.json')
    assert not os.path.lexists(output / 'fres_scqo_plan_a_hrsc_audit.md')
    assert not os.path.lexists(output / '.scqo-report.lock')


def test_all_plan_a_tools_cannot_start_training_or_overwrite_status():
    text = COLLECTOR.read_text() + REPORTER.read_text()
    forbidden = ('tools/train.py', 'train_step(', 'optim_wrapper', 'tmux',
                 'nohup', 'FORMAL_TRAINING_NOT_STARTED.md')
    assert all(token not in text for token in forbidden)
