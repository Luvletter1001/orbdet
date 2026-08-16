import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.analysis_tools import scqo_collect_evidence as collector
from tools.analysis_tools.scqo_collect_evidence import (_evidence_cfg, _scalar,
                                                        build_evidence_row,
                                                        collect, parse_args,
                                                        sha256)

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / 'tools/analysis_tools/scqo_collect_evidence.py'


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
        bboxes=SimpleNamespace(
            tensor=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)),
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


def _install_fake_runtime(monkeypatch,
                          samples,
                          *,
                          fail_on_extract=False,
                          on_load=None,
                          on_evidence=None):
    model = _FakeModel(fail_on_extract=fail_on_extract)
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
