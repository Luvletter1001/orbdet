from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_CONFIG = (
    ROOT /
    'configs/orbdet/h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py')

RECOVERY_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814')


def test_recovery_contract_restores_proven_optimizer_and_aligns_by_steps():
    assert RECOVERY_CONFIG.is_file(), f'missing: {RECOVERY_CONFIG}'
    cfg = Config.fromfile(RECOVERY_CONFIG)

    assert cfg.model.type == 'H2RBoxDetector'
    assert cfg.model.bbox_head.loss_bbox_ss.type == 'H2RBoxConsistencyLoss'

    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.dataset.ann_file == 'ImageSets/train.txt'
    assert cfg.val_dataloader.dataset.ann_file == 'ImageSets/val.txt'
    assert cfg.test_dataloader.dataset.ann_file == 'ImageSets/test.txt'

    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert cfg.param_scheduler[0].type == 'LinearLR'
    assert cfg.param_scheduler[0].by_epoch is False
    assert cfg.param_scheduler[0].end == 500

    step_scheduler = cfg.param_scheduler[1]
    assert step_scheduler.type == 'MultiStepLR'
    assert step_scheduler.by_epoch is False
    assert step_scheduler.end == 510 * 309
    assert list(step_scheduler.milestones) == [340 * 309, 468 * 309]

    assert cfg.train_cfg.max_epochs == 60
    assert cfg.train_cfg.val_interval == 30
    assert cfg.default_hooks.checkpoint.interval == 30
    assert cfg.randomness.seed == 3407
    assert cfg.work_dir == RECOVERY_WORK_DIR
