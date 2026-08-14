import math
from pathlib import Path

import torch
from mmengine import ConfigDict
from mmengine.config import Config

from mmrotate.models.detectors.h2rbox import H2RBoxDetector
from mmrotate.models.losses.orbdet_harmonic_consistency_loss import (
    OrbdetHarmonicConsistencyLoss, axis_harmonic_reliability)
from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / 'configs/orbdet/orbdet_v0_1_r50_hrsc.py'
CALIBRATION_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_1_r50_hrsc_calibration.py')
CALIBRATION_LAUNCHER = (
    ROOT / 'scripts/smoke/run_orbdet_v0_1_hrsc_gpu89.sh')
FORMAL_LAUNCHER = ROOT / 'scripts/formal/run_orbdet_v0_1_hrsc_gpu89.sh'


def test_axis_harmonic_reliability_respects_period_and_side_swap():
    delta = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])

    quality = axis_harmonic_reliability(
        delta, min_quality=0.25, gamma=2.0)

    assert torch.allclose(quality, torch.ones_like(quality), atol=1e-6)


def test_axis_harmonic_reliability_has_floor_at_ambiguous_axis():
    delta = torch.tensor([math.pi / 4, -math.pi / 4])

    quality = axis_harmonic_reliability(
        delta, min_quality=0.25, gamma=2.0)

    assert torch.allclose(
        quality, torch.full_like(quality, 0.25), atol=1e-6)


def test_harmonic_reliability_rejects_invalid_hyperparameters():
    delta = torch.zeros(1)

    for min_quality, gamma in ((-0.1, 2.0), (1.1, 2.0), (0.25, 0.0)):
        try:
            axis_harmonic_reliability(
                delta, min_quality=min_quality, gamma=gamma)
        except ValueError:
            continue
        raise AssertionError(
            f'expected ValueError for min_quality={min_quality}, gamma={gamma}')


def test_orbdet_loss_has_finite_gradient_and_detached_quality():
    loss_fn = OrbdetHarmonicConsistencyLoss(loss_weight=0.4)
    pred = torch.tensor(
        [[10.0, 10.0, 4.0, 2.0, 0.2]], requires_grad=True)
    target = torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.0]])

    loss = loss_fn(
        pred, target, weight=torch.ones(1), avg_factor=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert 0.25 <= float(loss_fn.last_quality_mean) <= 1.0
    assert not loss_fn.last_quality_mean.requires_grad


def test_orbdet_loss_handles_empty_batch_with_finite_diagnostics():
    loss_fn = OrbdetHarmonicConsistencyLoss(loss_weight=0.4)
    pred = torch.empty((0, 5), requires_grad=True)
    target = torch.empty((0, 5))

    loss = loss_fn(
        pred, target, weight=torch.empty(0), avg_factor=1)
    loss.backward()

    assert float(loss) == 0.0
    assert pred.grad is not None
    assert torch.isfinite(loss_fn.last_quality_mean)
    assert torch.isfinite(loss_fn.last_quality_min)
    assert torch.isfinite(loss_fn.last_quality_high_frac)


def test_orbdet_detector_is_registered_without_predict_override():
    detector_cls = MODELS.get('OrbdetDetector')

    assert detector_cls is not None
    assert issubclass(detector_cls, H2RBoxDetector)
    assert 'predict' not in detector_cls.__dict__


def test_hrsc_config_enforces_hbox_train_and_obb_eval_contract():
    cfg = Config.fromfile(FORMAL_CONFIG)

    assert cfg.model.type == 'OrbdetDetector'
    assert cfg.model.bbox_head.num_classes == 1
    assert (cfg.model.bbox_head.loss_bbox_ss.type ==
            'OrbdetHarmonicConsistencyLoss')
    assert cfg.model.backbone.init_cfg.checkpoint.endswith(
        'weights/resnet50-0676ba61.pth')
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.batch_sampler is None
    assert cfg.train_dataloader.sampler.type == 'DefaultSampler'
    assert cfg.train_dataloader.dataset.type == 'HRSCDataset'
    assert (cfg.train_dataloader.dataset.ann_file ==
            'ImageSets/trainval.txt')
    assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert cfg.val_dataloader.dataset.test_mode is True

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
    assert cfg.train_pipeline[-2].type == 'mmdet.Pad'
    assert tuple(cfg.train_pipeline[-2].size) == (800, 800)
    assert cfg.train_pipeline[-1].type == 'mmdet.PackDetInputs'
    assert cfg.val_evaluator.type == 'DOTAMetric'
    assert cfg.val_evaluator.metric == 'mAP'


def test_hrsc_calibration_is_one_epoch_without_validation():
    cfg = Config.fromfile(CALIBRATION_CONFIG)

    assert cfg.model.type == 'OrbdetDetector'
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None
    assert '/calibration/' in cfg.work_dir


def test_formal_schedule_matches_measured_twelve_hour_budget():
    cfg = Config.fromfile(FORMAL_CONFIG)

    assert cfg.train_cfg.max_epochs == 510
    assert cfg.train_cfg.val_interval == 30
    assert cfg.param_scheduler[1].end == 510
    assert list(cfg.param_scheduler[1].milestones) == [340, 468]
    assert cfg.default_hooks.checkpoint.interval == 30
    assert cfg.default_hooks.checkpoint.max_keep_ckpts == 3
    assert cfg.default_hooks.checkpoint.save_best == 'dota/mAP'


def test_gpu89_launchers_pin_nccl_and_matching_configs():
    calibration_text = CALIBRATION_LAUNCHER.read_text()
    formal_text = FORMAL_LAUNCHER.read_text()

    for text in (calibration_text, formal_text):
        assert 'PYTHONNOUSERSITE=1' in text
        assert 'PYTHONPATH=/data1/zcy/Orbdet' in text
        assert 'CUDA_VISIBLE_DEVICES=8,9' in text
        assert 'NCCL_P2P_DISABLE=1' in text
        assert 'NCCL_IB_DISABLE=1' in text
        assert '/data/zcy/anaconda3/envs/orbdet/bin/python' in text
        assert '--nproc_per_node=2' in text
        assert '--launcher=pytorch' in text

    assert 'orbdet_v0_1_r50_hrsc_calibration.py' in calibration_text
    assert 'work_dirs/calibration/orbdet_v0_1_hrsc_gpu89' in calibration_text
    assert 'orbdet_v0_1_r50_hrsc.py' in formal_text
    assert 'work_dirs/formal/orbdet_v0_1_hrsc_gpu89' in formal_text
    assert 'pgrep' in formal_text
    assert "[p]ython.*tools/train.py.*orbdet_v0_1_r50_hrsc.py" in formal_text


def test_h2rbox_prediction_keeps_post_nms_lengths_without_square_classes():
    register_all_modules()
    cfg = Config.fromfile(FORMAL_CONFIG)
    head = MODELS.build(cfg.model.bbox_head)
    head.eval()
    features = [
        torch.randn(1, 256, size, size) for size in (8, 4, 2, 1, 1)
    ]
    img_meta = dict(
        img_shape=(64, 64),
        ori_shape=(64, 64),
        scale_factor=(1.0, 1.0),
    )
    test_cfg = ConfigDict(
        nms_pre=50,
        min_bbox_size=0,
        score_thr=0.0,
        nms=ConfigDict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=10,
    )

    with torch.no_grad():
        cls_scores, bbox_preds, angle_preds, centernesses = head(features)
        results = head.predict_by_feat(
            cls_scores,
            bbox_preds,
            angle_preds,
            score_factors=centernesses,
            batch_img_metas=[img_meta],
            cfg=test_cfg,
            rescale=False)

    assert len(results) == 1
    assert len(results[0].bboxes) == len(results[0].scores)
    assert len(results[0].scores) <= 10
