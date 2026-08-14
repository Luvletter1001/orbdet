import copy
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = (
    ROOT /
    'configs/orbdet/h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py')
CANDIDATE_CONFIG = (
    ROOT /
    'configs/orbdet/'
    'orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py')
SMOKE_CONFIG = (
    ROOT /
    'configs/orbdet/'
    'orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_smoke_gpu89.py')
SMOKE_LAUNCHER = (
    ROOT /
    'scripts/smoke/run_orbdet_v0_1_hrsc_recovery_gpu89_smoke.sh')
FORMAL_LAUNCHER = (
    ROOT /
    'scripts/formal/'
    'run_orbdet_v0_1_hrsc_recovery_staged_200e_gpu89_20260814.sh')

FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_1_hrsc_recovery_bs1_stepaligned_200e_gpu89_'
    'seed3407_20260814')
SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_1_hrsc_recovery_bs1_gpu89_20260814')


def _load(path: Path) -> Config:
    assert path.is_file(), f'missing required artifact: {path}'
    return Config.fromfile(path)


def test_candidate_changes_only_detector_loss_and_work_dir():
    reference = _load(REFERENCE_CONFIG)
    candidate = _load(CANDIDATE_CONFIG)

    assert candidate.model.type == 'OrbdetDetector'
    loss = candidate.model.bbox_head.loss_bbox_ss
    assert loss.type == 'OrbdetHarmonicConsistencyLoss'
    assert loss.loss_weight == 0.4
    assert loss.min_quality == 0.25
    assert loss.gamma == 2.0
    assert loss.high_quality_thr == 0.75
    assert loss.center_loss_cfg == dict(
        type='mmdet.L1Loss', loss_weight=0.0)
    assert loss.shape_loss_cfg == dict(
        type='mmdet.IoULoss', loss_weight=1.0)
    assert loss.angle_loss_cfg == dict(
        type='mmdet.L1Loss', loss_weight=1.0)
    assert candidate.work_dir == FORMAL_WORK_DIR

    normalized = copy.deepcopy(candidate.to_dict())
    expected = reference.to_dict()
    normalized['model'] = copy.deepcopy(expected['model'])
    normalized['work_dir'] = expected['work_dir']
    assert normalized == expected


def test_frozen_recovery_contract_matches_completed_h2rbox_run():
    candidate = _load(CANDIDATE_CONFIG)

    assert candidate.train_dataloader.batch_size == 1
    assert candidate.train_dataloader.dataset.ann_file == 'ImageSets/train.txt'
    assert candidate.val_dataloader.dataset.ann_file == 'ImageSets/val.txt'
    assert candidate.test_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert candidate.optim_wrapper.optimizer.lr == 1e-4
    assert candidate.randomness.seed == 3407
    assert candidate.randomness.deterministic is False
    assert candidate.train_cfg.max_epochs == 60
    assert candidate.train_cfg.val_interval == 30
    assert candidate.default_hooks.checkpoint.interval == 30

    warmup, schedule = candidate.param_scheduler
    assert warmup.by_epoch is False
    assert warmup.end == 500
    assert schedule.by_epoch is False
    assert schedule.end == 510 * 309
    assert list(schedule.milestones) == [340 * 309, 468 * 309]


def test_smoke_is_four_images_two_global_batch_steps():
    candidate = _load(CANDIDATE_CONFIG)
    smoke = _load(SMOKE_CONFIG)

    assert smoke.model == candidate.model
    assert smoke.train_dataloader.batch_size == 1
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.train_dataloader.persistent_workers is False
    assert list(smoke.train_dataloader.dataset.indices) == list(range(4))
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.val_cfg is None
    assert smoke.val_dataloader is None
    assert smoke.val_evaluator is None
    assert smoke.default_hooks.logger.interval == 1
    assert smoke.default_hooks.checkpoint.interval == 1
    assert smoke.default_hooks.checkpoint.max_keep_ckpts == 1
    assert smoke.work_dir == SMOKE_WORK_DIR


def _assert_gpu89_ddp_launcher(text: str, config_name: str,
                               work_dir: str, port: int) -> None:
    for required in (
            'PYTHONNOUSERSITE=1', 'PYTHONPATH=/data1/zcy/Orbdet',
            'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1',
            '/data/zcy/anaconda3/envs/orbdet/bin/python',
            '--nproc_per_node=2', f'--master_port={port}',
            '--launcher=pytorch', config_name, work_dir):
        assert required in text


def test_smoke_and_formal_launchers_are_exactly_scoped():
    assert SMOKE_LAUNCHER.is_file(), f'missing: {SMOKE_LAUNCHER}'
    assert FORMAL_LAUNCHER.is_file(), f'missing: {FORMAL_LAUNCHER}'

    smoke_text = SMOKE_LAUNCHER.read_text()
    formal_text = FORMAL_LAUNCHER.read_text()
    _assert_gpu89_ddp_launcher(
        smoke_text,
        'orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_smoke_gpu89.py',
        SMOKE_WORK_DIR,
        29624)
    _assert_gpu89_ddp_launcher(
        formal_text,
        'orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py',
        FORMAL_WORK_DIR,
        29625)

    assert 'pgrep -af' in formal_text
    assert ('[p]ython.*tools/train.py.*'
            'orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py'
            in formal_text)
    assert 'train_cfg.max_epochs=30' in formal_text
    assert 'train_cfg.max_epochs=60' in formal_text
    assert 'train_cfg.max_epochs=200' in formal_text
    assert formal_text.count('--resume') == 2
    assert 'epoch_30.pth' in formal_text
    assert 'epoch_60.pth' in formal_text
    assert 'epoch_200.pth' in formal_text
    assert 'test_dataloader.dataset.ann_file=ImageSets/val.txt' in formal_text
    assert 'test_dataloader.dataset.ann_file=ImageSets/train.txt' in formal_text
    assert 'ImageSets/test.txt' not in formal_text
    assert 'NCCL_P2P_DISABLE=1' in formal_text
    assert 'NCCL_IB_DISABLE=1' in formal_text
