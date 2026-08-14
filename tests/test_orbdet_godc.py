import copy
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmrotate.models.detectors.h2rbox_v2 import H2RBoxV2Detector
from mmrotate.models.detectors.orbdet_v0_2 import OrbdetV02Detector
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py')
FORMAL_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_godc_c2_r50_hrsc_clean_gpu89.py')
SMOKE_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_godc_c2_r50_hrsc_clean_gpu89_smoke.py')
SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/run_orbdet_godc_c2_hrsc_gpu89.sh')
FORMAL_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_godc_c2_hrsc_gpu89_seed3407.sh')
CONTROLLER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_godc_after_v02_multiseed_20260815.sh')
FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815')
SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_godc_c2_hrsc_clean_gpu89_20260815')


def _adapter_cfg():
    return dict(
        type='HBoxFPNGroupOrbitLoss',
        group='c2',
        loss_weight=0.02,
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=4,
            featmap_strides=[1]),
        orbit_loss=dict(
            type='GroupOrbitDeterminantalClusterLoss',
            energy_guard_weight=0.0,
            variance_guard_weight=0.0))


def _instances():
    instances = InstanceData()
    instances.bboxes = RotatedBoxes(
        torch.tensor([[10., 10., 12., 8., 0.]]))
    instances.labels = torch.zeros(1, dtype=torch.long)
    return instances


def test_h2rbox_v2_default_auxiliary_hook_is_exact_noop():
    losses = {'loss_bbox': torch.tensor(1.0)}

    returned = H2RBoxV2Detector._add_auxiliary_losses(
        None, losses, (), [])

    assert returned is losses
    assert list(returned) == ['loss_bbox']


def test_godc_detector_inherits_v02_and_preserves_prediction_path():
    register_all_modules()
    detector_cls = MODELS.get('OrbdetGODCDetector')

    assert detector_cls is not None
    assert issubclass(detector_cls, OrbdetV02Detector)
    assert 'predict' not in detector_cls.__dict__


def test_godc_auxiliary_hook_emits_one_loss_and_detached_diagnostics():
    register_all_modules()
    detector_cls = MODELS.get('OrbdetGODCDetector')
    detector = object.__new__(detector_cls)
    torch.nn.Module.__init__(detector)
    detector.godc_auxiliary = MODELS.build(_adapter_cfg())
    feature = torch.randn(1, 4, 20, 20, requires_grad=True)
    losses = {'loss_bbox': feature.sum() * 0.0}

    returned = detector._add_auxiliary_losses(
        losses, (feature, ), [_instances()])
    total = returned['loss_godc'] + returned['loss_bbox']
    total.backward()

    assert returned is losses
    assert returned['loss_godc'].shape == ()
    assert torch.isfinite(returned['loss_godc'])
    assert float(returned['loss_godc']) > 0.0
    diagnostic_names = [name for name in returned if name.startswith('godc_')]
    assert diagnostic_names
    assert all('loss' not in name for name in diagnostic_names)
    assert all(not returned[name].requires_grad for name in diagnostic_names)
    assert feature.grad is not None
    assert float(feature.grad.abs().sum()) > 0.0


def test_formal_config_changes_only_detector_adapter_and_work_dir():
    base = Config.fromfile(BASE_CONFIG).to_dict()
    formal = Config.fromfile(FORMAL_CONFIG).to_dict()

    assert formal['model']['type'] == 'OrbdetGODCDetector'
    adapter = formal['model']['godc_auxiliary']
    assert adapter['type'] == 'HBoxFPNGroupOrbitLoss'
    assert adapter['group'] == 'c2'
    assert adapter['loss_weight'] == 0.02
    assert adapter['roi_extractor']['featmap_strides'] == [8, 16, 32, 64,
                                                           128]
    assert adapter['roi_extractor']['roi_layer']['output_size'] == 7
    assert adapter['orbit_loss']['type'] == (
        'GroupOrbitDeterminantalClusterLoss')

    normalized = copy.deepcopy(formal)
    normalized['model'].pop('godc_auxiliary')
    normalized['model']['type'] = base['model']['type']
    normalized['work_dir'] = base['work_dir']
    assert normalized == base
    assert formal['work_dir'] == FORMAL_WORK_DIR
    assert formal['randomness']['seed'] == 3407


def test_formal_model_builds_with_godc_components():
    register_all_modules()
    cfg = Config.fromfile(FORMAL_CONFIG)

    model = MODELS.build(cfg.model)

    assert model.__class__.__name__ == 'OrbdetGODCDetector'
    assert model.godc_auxiliary.__class__.__name__ == (
        'HBoxFPNGroupOrbitLoss')
    assert model.godc_auxiliary.orbit_loss.__class__.__name__ == (
        'GroupOrbitDeterminantalClusterLoss')


def test_smoke_config_is_eight_images_two_steps_without_validation():
    formal = Config.fromfile(FORMAL_CONFIG)
    smoke = Config.fromfile(SMOKE_CONFIG)

    assert smoke.model == formal.model
    assert smoke.train_dataloader.batch_size == 2
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.train_dataloader.persistent_workers is False
    assert smoke.train_dataloader.dataset.indices == 8
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.val_cfg is None
    assert smoke.val_dataloader is None
    assert smoke.val_evaluator is None
    assert smoke.work_dir == SMOKE_WORK_DIR


def test_gpu89_launchers_and_controller_are_fail_fast():
    contracts = (
        (SMOKE_LAUNCHER, SMOKE_CONFIG.name, SMOKE_WORK_DIR, 29643),
        (FORMAL_LAUNCHER, FORMAL_CONFIG.name, FORMAL_WORK_DIR, 29644),
    )
    for launcher, config_name, work_dir, port in contracts:
        text = launcher.read_text()
        for required in (
                'set -euo pipefail', 'repo_root=',
                'PYTHONNOUSERSITE=1', 'PYTHONPATH="${repo_root}"',
                'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
                'NCCL_IB_DISABLE=1',
                '/data/zcy/anaconda3/envs/orbdet/bin/python',
                '--nproc_per_node=2', f'--master_port={port}',
                '--launcher=pytorch', config_name, work_dir,
                'existing_checkpoints'):
            assert required in text, required
        assert '--resume' not in text

    controller = CONTROLLER.read_text()
    assert 'set -euo pipefail' in controller
    assert 'sleep 30' in controller
    assert ('orbdet_v0_2_hrsc_multiseed_gpu89_20260815/COMPLETE'
            in controller)
    assert ('orbdet_v0_2_hrsc_multiseed_gpu89_20260815/FAILED'
            in controller)
    assert controller.index(SMOKE_LAUNCHER.name) < controller.index(
        FORMAL_LAUNCHER.name)
    assert 'epoch_1.pth' in controller
    assert 'SMOKE_COMPLETE' in controller
    assert 'trap on_error ERR' in controller
    assert 'rm -' not in controller
