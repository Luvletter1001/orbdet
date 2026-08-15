import hashlib
import subprocess
from pathlib import Path

import torch
from mmengine.config import Config

from mmrotate.registry import DATASETS, MODELS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(
    '/data1/zcy/Orbdet/work_dirs/audit/'
    'h2rbox_v2_dota1_official_20260815')
DOWNLOAD_ROOT = AUDIT_ROOT / 'downloads'
SS_CHECKPOINT = (
    DOWNLOAD_ROOT / 'h2rbox_v2-le90_r50_fpn-1x_dota-fa5ad1d2.pth')
MSRR_CHECKPOINT = (
    DOWNLOAD_ROOT /
    'h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth')

SS_TRAINVAL_CONFIG = (
    ROOT / 'configs/orbdet/'
    'h2rbox_v2_r50_dota1_official_ss_trainval_audit_gpu89.py')
SS_TEST_CONFIG = (
    ROOT / 'configs/orbdet/'
    'h2rbox_v2_r50_dota1_official_ss_test_submission_gpu89.py')
MS_TEST_CONFIG = (
    ROOT / 'configs/orbdet/'
    'h2rbox_v2_r50_dota1_official_ms_test_submission_gpu89.py')
FORMAL_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py')
SMOKE_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_smoke.py')

AUDIT_LAUNCHER = (
    ROOT / 'scripts/eval/'
    'run_h2rbox_v2_dota1_official_checkpoint_audit_gpu89.sh')
SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/'
    'run_orbdet_v0_2_r50_dota1_ms_rr_gpu89_smoke.sh')
FORMAL_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407.sh')

GPU4567_AUDIT_ROOT = Path(
    '/data1/zcy/Orbdet/work_dirs/audit/'
    'h2rbox_v2_dota1_official_gpu4567_20260816')
GPU4567_FORMAL_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py')
GPU4567_SMOKE_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_smoke.py')
GPU4567_STAGE1_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py')
GPU4567_AUDIT_LAUNCHER = (
    ROOT / 'scripts/eval/'
    'run_h2rbox_v2_dota1_official_checkpoint_audit_gpu4567.sh')
GPU4567_SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/'
    'run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_smoke.sh')
GPU4567_STAGE1_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_3e.sh')
GPU4567_WINDOW_LAUNCHER = (
    ROOT / 'scripts/formal/'
    'run_orbdet_v0_2_r50_dota1_ms_rr_gpu4567_six_hour_window.sh')

GPU4567_FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_seed3407_20260816')
GPU4567_SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_20260816')
GPU4567_STAGE1_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816')

SS_SHA256 = 'fa5ad1d2d6d030a477fe6f9a405863a76f55d463cff31b7e01c8366527888fde'
MSRR_SHA256 = '5e0e53e12e0d8b07f79922b6cef9b56c142458483d2c1e2f27348f7e72e9d677'
FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407_20260815')
SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_20260815')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_official_model_contract(cfg: Config) -> None:
    assert cfg.model.type == 'H2RBoxV2Detector'
    assert cfg.model.bbox_head.type == 'H2RBoxV2Head'
    assert cfg.model.bbox_head.rotation_agnostic_classes == [1, 9, 11]
    assert cfg.model.bbox_head.agnostic_resize_classes == [1]
    assert cfg.model.bbox_head.use_circumiou_loss is True
    assert cfg.model.bbox_head.use_standalone_angle is True
    assert cfg.model.bbox_head.use_reweighted_loss_bbox is True


def test_official_checkpoints_are_exact_and_embed_expected_protocol():
    assert _sha256(SS_CHECKPOINT) == SS_SHA256
    assert _sha256(MSRR_CHECKPOINT) == MSRR_SHA256

    ss = torch.load(SS_CHECKPOINT, map_location='cpu')
    msrr = torch.load(MSRR_CHECKPOINT, map_location='cpu')

    assert len(ss['state_dict']) == 371
    assert len(msrr['state_dict']) == 371
    assert (ss['meta']['epoch'], ss['meta']['iter']) == (12, 76800)
    assert (msrr['meta']['epoch'], msrr['meta']['iter']) == (12, 409956)

    embedded = Config.fromstring(msrr['meta']['cfg'], '.py')
    assert embedded.train_dataloader.batch_size == 2
    assert embedded.optim_wrapper.optimizer.lr == 0.00005
    assert embedded.optim_wrapper.optimizer.weight_decay == 0.005
    assert embedded.model.bbox_head.rotation_agnostic_classes == [1, 9, 11]
    assert embedded.model.bbox_head.rotation_agnostic_resize_classes == [1]
    assert embedded.model.bbox_head.use_reweighted_loss_bbox is True
    rr = [
        transform for transform in embedded.train_dataloader.dataset.pipeline
        if transform.type == 'RandomRotate'
    ]
    assert len(rr) == 1
    assert rr[0].prob == 1
    assert rr[0].angle_range == 180


def test_official_audit_configs_use_exact_model_and_dataset_contracts():
    ss_trainval = Config.fromfile(SS_TRAINVAL_CONFIG)
    ss_test = Config.fromfile(SS_TEST_CONFIG)
    ms_test = Config.fromfile(MS_TEST_CONFIG)

    for cfg in (ss_trainval, ss_test, ms_test):
        _assert_official_model_contract(cfg)
        assert cfg.test_cfg.type == 'TestLoop'
        assert cfg.test_dataloader.batch_size == 2
        assert cfg.test_dataloader.sampler.type == 'DefaultSampler'
        assert cfg.test_dataloader.sampler.shuffle is False

    assert ss_trainval.test_dataloader.dataset.data_root == (
        '/data1/zcy/Orbdet/data/DOTA-v1.0/')
    assert ss_trainval.test_dataloader.dataset.ann_file == 'trainval/annfiles/'
    assert ss_trainval.test_evaluator.metric == 'mAP'
    assert ss_trainval.test_evaluator.iou_thrs == 0.5

    assert ss_test.test_dataloader.dataset.data_prefix.img_path == (
        'test/images/')
    assert ms_test.test_dataloader.dataset.data_root == (
        '/data/zcy/dataset/test_ms/')
    assert ms_test.test_dataloader.dataset.data_prefix.img_path == 'images/'
    for cfg in (ss_test, ms_test):
        assert cfg.test_evaluator.format_only is True
        assert cfg.test_evaluator.merge_patches is True
        assert cfg.test_evaluator.iou_thr == 0.1


def test_official_audit_datasets_have_expected_patch_counts():
    register_all_modules()
    expected = {
        SS_TRAINVAL_CONFIG: 20995,
        SS_TEST_CONFIG: 10833,
        MS_TEST_CONFIG: 71888,
    }
    for config_path, expected_len in expected.items():
        cfg = Config.fromfile(config_path)
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        assert len(dataset) == expected_len


def test_msrr_formal_config_matches_checkpoint_update_contract():
    cfg = Config.fromfile(FORMAL_CONFIG)

    assert cfg.model.type == 'OrbdetV02Detector'
    assert cfg.model.bbox_head.rotation_agnostic_classes == [1, 9, 11]
    assert cfg.model.bbox_head.agnostic_resize_classes == [1]
    assert cfg.model.bbox_head.use_circumiou_loss is True
    assert cfg.model.bbox_head.use_standalone_angle is True
    assert cfg.model.bbox_head.use_reweighted_loss_bbox is True

    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.sampler.type == 'DefaultSampler'
    assert cfg.train_dataloader.sampler.shuffle is True
    assert cfg.train_dataloader.batch_sampler is None
    assert cfg.train_dataloader.dataset.data_root == (
        '/data/zcy/dataset/trainval_ms_full/')
    assert cfg.train_dataloader.dataset.ann_file == 'annfiles/'
    assert cfg.train_dataloader.dataset.data_prefix.img_path == 'images/'
    assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is True

    transforms = cfg.train_dataloader.dataset.pipeline
    assert [item.type for item in transforms] == [
        'mmdet.LoadImageFromFile', 'mmdet.LoadAnnotations',
        'ConvertBoxType', 'ConvertBoxType', 'mmdet.Resize',
        'mmdet.RandomFlip', 'RandomRotate', 'mmdet.PackDetInputs'
    ]
    assert transforms[2].box_type_mapping.gt_bboxes == 'hbox'
    assert transforms[3].box_type_mapping.gt_bboxes == 'rbox'
    assert transforms[6].prob == 1
    assert transforms[6].angle_range == 180

    assert cfg.optim_wrapper.optimizer.lr == 0.00005
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.005
    assert cfg.train_cfg.max_epochs == 12
    assert cfg.param_scheduler[1].milestones == [8, 11]
    assert cfg.val_cfg is None
    assert cfg.test_cfg is None
    assert cfg.work_dir == FORMAL_WORK_DIR


def test_msrr_formal_model_dataset_and_smoke_build():
    register_all_modules()
    formal = Config.fromfile(FORMAL_CONFIG)
    smoke = Config.fromfile(SMOKE_CONFIG)

    # Registry construction mutates nested train/test cfg entries in this
    # MMEngine version, so compare the frozen config contract first.
    assert smoke.model == formal.model
    model = MODELS.build(formal.model)
    dataset = DATASETS.build(formal.train_dataloader.dataset)

    assert model.__class__.__name__ == 'OrbdetV02Detector'
    assert len(dataset) == 68325
    assert smoke.train_dataloader.batch_size == 1
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.train_dataloader.persistent_workers is False
    assert smoke.train_dataloader.dataset.indices == 4
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.param_scheduler[0].end == 2
    assert smoke.work_dir == SMOKE_WORK_DIR


def test_launchers_are_fail_fast_and_gpu89_safe():
    audit = AUDIT_LAUNCHER.read_text()
    for required in (
            'set -euo pipefail', SS_SHA256, MSRR_SHA256,
            SS_TRAINVAL_CONFIG.name, SS_TEST_CONFIG.name, MS_TEST_CONFIG.name,
            'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1', '--nproc_per_node=2',
            '--master_port=29650', '--master_port=29651',
            '--master_port=29652', 'SS_TRAINVAL_COMPLETE',
            'SS_SUBMISSION_COMPLETE', 'MS_SUBMISSION_COMPLETE',
            'COMPLETE', 'FAILED'):
        assert required in audit, required
    assert audit.index(SS_TRAINVAL_CONFIG.name) < audit.index(
        SS_TEST_CONFIG.name) < audit.index(MS_TEST_CONFIG.name)
    assert 'tools/train.py' not in audit
    assert 'rm -' not in audit

    launchers = (
        (SMOKE_LAUNCHER, SMOKE_CONFIG.name, SMOKE_WORK_DIR, 29653),
        (FORMAL_LAUNCHER, FORMAL_CONFIG.name, FORMAL_WORK_DIR, 29654),
    )
    for launcher, config_name, work_dir, port in launchers:
        text = launcher.read_text()
        for required in (
                'set -euo pipefail', config_name, work_dir,
                'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
                'NCCL_IB_DISABLE=1', '--nproc_per_node=2',
                f'--master_port={port}', 'gpu_processes',
                'existing_checkpoints'):
            assert required in text, required
        assert '--resume' not in text
        assert 'tools/test.py' not in text
        assert 'rm -' not in text

    assert str(AUDIT_ROOT / 'COMPLETE') in FORMAL_LAUNCHER.read_text()


def test_gpu4567_six_hour_stage_contract():
    formal = Config.fromfile(GPU4567_FORMAL_CONFIG)
    smoke = Config.fromfile(GPU4567_SMOKE_CONFIG)
    stage1 = Config.fromfile(GPU4567_STAGE1_CONFIG)

    assert formal.train_dataloader.batch_size == 1
    assert formal.train_dataloader.sampler.type == 'DefaultSampler'
    assert formal.train_dataloader.batch_sampler is None
    assert formal.optim_wrapper.optimizer.lr == 0.0001
    assert formal.optim_wrapper.optimizer.weight_decay == 0.005
    assert formal.train_cfg.max_epochs == 12
    assert formal.default_hooks.checkpoint.interval == 1
    assert formal.default_hooks.checkpoint.save_last is True
    assert formal.work_dir == GPU4567_FORMAL_WORK_DIR

    assert smoke.train_dataloader.batch_size == 1
    assert smoke.train_dataloader.dataset.indices == 8
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.param_scheduler[0].end == 2
    assert smoke.work_dir == GPU4567_SMOKE_WORK_DIR

    assert stage1.train_cfg.max_epochs == 3
    assert stage1.default_hooks.checkpoint.interval == 1
    assert stage1.work_dir == GPU4567_STAGE1_WORK_DIR
    assert stage1.resume is False

    launchers = (
        (GPU4567_AUDIT_LAUNCHER, '--master_port=29660'),
        (GPU4567_SMOKE_LAUNCHER, '--master_port=29663'),
        (GPU4567_STAGE1_LAUNCHER, '--master_port=29664'),
    )
    for launcher, port in launchers:
        text = launcher.read_text()
        for required in (
                'set -euo pipefail', 'CUDA_VISIBLE_DEVICES=4,5,6,7',
                'NCCL_P2P_DISABLE=1', 'NCCL_IB_DISABLE=1',
                '--nproc_per_node=4', port, 'gpu_processes'):
            assert required in text, (launcher, required)
        assert 'rm -' not in text

    audit = GPU4567_AUDIT_LAUNCHER.read_text()
    assert '--master_port=29661' in audit
    assert '--master_port=29662' in audit
    assert str(GPU4567_AUDIT_ROOT) in audit

    stage_launcher = GPU4567_STAGE1_LAUNCHER.read_text()
    assert GPU4567_STAGE1_CONFIG.name in stage_launcher
    assert GPU4567_STAGE1_WORK_DIR in stage_launcher
    assert 'epoch_3.pth' in stage_launcher
    assert '--resume' not in stage_launcher

    window = GPU4567_WINDOW_LAUNCHER.read_text()
    assert '345m' in window
    assert '5h45m' not in window
    assert 'timeout' in window
    assert window.index(GPU4567_AUDIT_LAUNCHER.name) < window.index(
        GPU4567_SMOKE_LAUNCHER.name) < window.index(
            GPU4567_STAGE1_LAUNCHER.name)
    assert 'CUDA_VISIBLE_DEVICES=8,9' not in window
    assert 'rm -' not in window
    timeout_probe = subprocess.run(
        ['timeout', '345m', 'true'], check=False, capture_output=True)
    assert timeout_probe.returncode == 0, timeout_probe.stderr
