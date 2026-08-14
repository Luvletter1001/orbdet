from pathlib import Path

from mmengine.config import Config

from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89.py')
SMOKE_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89_smoke.py')
SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/run_orbdet_v0_2_dota1_gpu89.sh')
FORMAL_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v0_2_dota1_1x_gpu89_seed3407.sh')
CONTROLLER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v02_dota1_after_godc_20260815.sh')
FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815')
SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_dota1_1x_gpu89_20260815')


def test_formal_config_is_full_dota_hbox_only_one_x_contract():
    cfg = Config.fromfile(FORMAL_CONFIG)

    assert cfg.model.type == 'OrbdetV02Detector'
    assert cfg.model.crop_size == (1024, 1024)
    assert cfg.model.bbox_head.type == 'H2RBoxV2Head'
    assert cfg.model.bbox_head.num_classes == 15
    assert cfg.model.bbox_head.rotation_agnostic_classes == [9, 11]
    assert cfg.model.bbox_head.angle_coder.type == 'PSCCoder'
    assert cfg.model.backbone.init_cfg.checkpoint.endswith(
        'weights/resnet50-0676ba61.pth')

    transforms = cfg.train_dataloader.dataset.pipeline
    conversions = [
        item.box_type_mapping.gt_bboxes for item in transforms
        if item.type == 'ConvertBoxType'
    ]
    assert conversions == ['hbox', 'rbox']
    assert cfg.train_dataloader.batch_size == 2
    assert cfg.train_dataloader.dataset.type == 'DOTADataset'
    assert cfg.train_dataloader.dataset.ann_file == 'trainval/annfiles/'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == (
        'trainval/images/')
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is True
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.train_cfg.val_interval == 999
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None
    assert cfg.test_cfg is None
    assert cfg.test_dataloader is None
    assert cfg.test_evaluator is None

    assert cfg.optim_wrapper.optimizer.type == 'AdamW'
    assert cfg.optim_wrapper.optimizer.lr == 0.0001
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.05
    assert cfg.param_scheduler[1].by_epoch is True
    assert cfg.param_scheduler[1].milestones == [8, 11]
    assert cfg.default_hooks.checkpoint.interval == 4
    assert cfg.default_hooks.checkpoint.get('save_best') is None
    assert cfg.randomness.seed == 3407
    assert cfg.work_dir == FORMAL_WORK_DIR


def test_formal_model_and_all_12757_nonempty_training_samples_build():
    register_all_modules()
    cfg = Config.fromfile(FORMAL_CONFIG)

    model = MODELS.build(cfg.model)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    assert model.__class__.__name__ == 'OrbdetV02Detector'
    assert model.bbox_head.num_classes == 15
    raw_annotations = list(
        Path(cfg.train_dataloader.dataset.data_root,
             cfg.train_dataloader.dataset.ann_file).glob('*.txt'))
    assert len(raw_annotations) == 20995
    assert len(dataset) == 12757


def test_smoke_is_same_model_and_exactly_two_global_steps():
    formal = Config.fromfile(FORMAL_CONFIG)
    smoke = Config.fromfile(SMOKE_CONFIG)

    assert smoke.model == formal.model
    assert smoke.train_dataloader.batch_size == 2
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.train_dataloader.persistent_workers is False
    assert smoke.train_dataloader.dataset.indices == 8
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.param_scheduler[0].end == 2
    assert smoke.work_dir == SMOKE_WORK_DIR


def test_launchers_pin_gpu89_and_controller_is_fail_fast():
    contracts = (
        (SMOKE_LAUNCHER, SMOKE_CONFIG.name, SMOKE_WORK_DIR, 29645),
        (FORMAL_LAUNCHER, FORMAL_CONFIG.name, FORMAL_WORK_DIR, 29646),
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
                'gpu_processes', 'existing_checkpoints'):
            assert required in text, required
        assert '--resume' not in text
        assert 'tools/test.py' not in text

    controller = CONTROLLER.read_text()
    assert 'set -euo pipefail' in controller
    assert 'sleep 30' in controller
    assert 'orbdet_godc_after_v02_gpu89_20260815/COMPLETE' in controller
    assert 'orbdet_godc_after_v02_gpu89_20260815/FAILED' in controller
    assert controller.index(SMOKE_LAUNCHER.name) < controller.index(
        FORMAL_LAUNCHER.name)
    assert 'epoch_1.pth' in controller
    assert 'SMOKE_COMPLETE' in controller
    assert 'trap on_error ERR' in controller
    assert 'rm -' not in controller
