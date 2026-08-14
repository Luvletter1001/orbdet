import math
from pathlib import Path

import torch
from mmengine.config import Config

from mmrotate.models.detectors.h2rbox_v2 import H2RBoxV2Detector
from mmrotate.models.losses.h2rbox_v2_consistency_loss import (
    H2RBoxV2ConsistencyLoss)
from mmrotate.models.losses.orbdet_anchored_symmetry_loss import (
    OrbdetAnchoredSymmetryLoss)
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89.py')
SMOKE_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_hrsc_clean_gpu89_smoke.py')
SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/run_orbdet_v0_2_hrsc_clean_gpu89.sh')
FORMAL_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v0_2_hrsc_clean_gpu89_seed3407.sh')


def _loss_kwargs():
    return dict(
        use_snap_loss=True,
        loss_rot=dict(
            type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
        loss_flp=dict(
            type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1))


def _symmetry_inputs(flip_is_correct=True):
    rotation = 0.8
    pred_ori = torch.tensor([0.3], requires_grad=True)
    pred_rot = torch.tensor([0.3 + rotation], requires_grad=True)
    flip_angle = -0.3 if flip_is_correct else 0.3
    pred_flp = torch.tensor([flip_angle], requires_grad=True)
    target_ori = torch.tensor([0.0])
    target_rot = torch.tensor([rotation])
    return pred_ori, pred_rot, pred_flp, target_ori, target_rot


def test_anchored_loss_is_exact_official_objective():
    official = H2RBoxV2ConsistencyLoss(**_loss_kwargs())
    anchored = OrbdetAnchoredSymmetryLoss(**_loss_kwargs())
    inputs = _symmetry_inputs(flip_is_correct=False)

    official_value = official(*inputs)
    anchored_value = anchored(*inputs)

    assert torch.allclose(anchored_value, official_value, atol=1e-7)
    anchored_value.backward()
    for tensor in inputs[:3]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_flip_anchor_rejects_rotation_consistent_absolute_error():
    loss_fn = OrbdetAnchoredSymmetryLoss(**_loss_kwargs())

    correct_loss = loss_fn(*_symmetry_inputs(flip_is_correct=True))
    correct_q = loss_fn.last_quality_joint.clone()
    wrong_loss = loss_fn(*_symmetry_inputs(flip_is_correct=False))
    wrong_q = loss_fn.last_quality_joint.clone()

    assert float(correct_loss) < 1e-7
    assert float(wrong_loss) > 0.0
    assert float(correct_q) > float(wrong_q)
    assert not loss_fn.last_quality_rot.requires_grad
    assert not loss_fn.last_quality_flip.requires_grad
    assert not loss_fn.last_quality_joint.requires_grad


def test_anchored_quality_is_bounded_and_validates_temperature():
    loss_fn = OrbdetAnchoredSymmetryLoss(
        quality_temperature=0.25, **_loss_kwargs())
    loss_fn(*_symmetry_inputs(flip_is_correct=False))

    for value in (
            loss_fn.last_quality_rot, loss_fn.last_quality_flip,
            loss_fn.last_quality_joint):
        assert 0.0 <= float(value) <= 1.0

    try:
        OrbdetAnchoredSymmetryLoss(
            quality_temperature=0.0, **_loss_kwargs())
    except ValueError:
        pass
    else:
        raise AssertionError('non-positive quality temperature was accepted')


def test_orbdet_v02_detector_preserves_h2rbox_v2_prediction_path():
    detector_cls = MODELS.get('OrbdetV02Detector')

    assert detector_cls is not None
    assert issubclass(detector_cls, H2RBoxV2Detector)
    assert 'predict' not in detector_cls.__dict__


def test_clean_hrsc_config_has_anchored_training_contract():
    cfg = Config.fromfile(FORMAL_CONFIG)

    assert cfg.model.type == 'OrbdetV02Detector'
    assert cfg.model.bbox_head.type == 'H2RBoxV2Head'
    assert cfg.model.bbox_head.angle_coder.type == 'PSCCoder'
    assert cfg.model.bbox_head.angle_coder.num_step == 3
    assert cfg.model.bbox_head.loss_symmetry_ss.type == (
        'OrbdetAnchoredSymmetryLoss')
    assert cfg.model.bbox_head.loss_symmetry_ss.use_snap_loss is True
    assert cfg.model.backbone.init_cfg.checkpoint.endswith(
        'weights/resnet50-0676ba61.pth')

    assert cfg.train_dataloader.batch_size == 2
    assert cfg.train_dataloader.dataset.ann_file == 'ImageSets/train.txt'
    assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/val.txt'
    assert cfg.test_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert cfg.val_dataloader.dataset.test_mode is True
    assert cfg.test_dataloader.dataset.test_mode is True

    train_conversions = [
        item.box_type_mapping.gt_bboxes for item in cfg.train_pipeline
        if item.type == 'ConvertBoxType'
    ]
    val_conversions = [
        item.box_type_mapping.gt_bboxes for item in cfg.val_pipeline
        if item.type == 'ConvertBoxType'
    ]
    assert train_conversions == ['hbox', 'rbox']
    assert val_conversions == ['rbox']

    assert cfg.train_cfg.max_epochs == 103
    assert cfg.train_cfg.val_interval == 12
    assert cfg.param_scheduler[1].by_epoch is False
    assert list(cfg.param_scheduler[1].milestones) == [7440, 10230]
    assert cfg.randomness.seed == 3407
    assert cfg.default_hooks.checkpoint.save_best == 'dota/mAP'


def test_formal_model_builds_with_registered_v02_components():
    register_all_modules()
    cfg = Config.fromfile(FORMAL_CONFIG)

    model = MODELS.build(cfg.model)

    assert model.__class__.__name__ == 'OrbdetV02Detector'
    assert model.bbox_head.__class__.__name__ == 'H2RBoxV2Head'
    assert model.bbox_head.angle_coder.__class__.__name__ == 'PSCCoder'
    assert model.bbox_head.loss_symmetry_ss.__class__.__name__ == (
        'OrbdetAnchoredSymmetryLoss')


def test_smoke_config_is_bounded_and_disables_validation():
    cfg = Config.fromfile(SMOKE_CONFIG)

    assert cfg.train_cfg.max_epochs == 1
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None
    assert '/smoke/' in cfg.work_dir


def test_gpu89_launchers_pin_safe_distributed_contract():
    smoke_text = SMOKE_LAUNCHER.read_text()
    formal_text = FORMAL_LAUNCHER.read_text()

    for text in (smoke_text, formal_text):
        assert 'PYTHONNOUSERSITE=1' in text
        assert 'PYTHONPATH=/data1/zcy/Orbdet' in text
        assert 'CUDA_VISIBLE_DEVICES=8,9' in text
        assert 'NCCL_P2P_DISABLE=1' in text
        assert 'NCCL_IB_DISABLE=1' in text
        assert '/data/zcy/anaconda3/envs/orbdet/bin/python' in text
        assert '--nproc_per_node=2' in text
        assert '--launcher=pytorch' in text

    assert 'orbdet_v0_2_r50_hrsc_clean_gpu89_smoke.py' in smoke_text
    assert 'orbdet_v0_2_r50_hrsc_clean_gpu89.py' in formal_text
    assert '[p]ython.*tools/train.py.*orbdet_v0_2' in formal_text
    assert 'ImageSets/test.txt' not in formal_text

