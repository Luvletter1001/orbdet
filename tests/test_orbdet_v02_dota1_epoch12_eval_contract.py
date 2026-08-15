from pathlib import Path

from mmengine.config import Config

from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_2_r50_dota1_1x_gpu89.py')
TRAINVAL_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py')
SUBMISSION_CONFIG = (
    ROOT / 'configs/orbdet/'
    'orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py')
EVAL_LAUNCHER = (
    ROOT / 'scripts/eval/'
    'run_orbdet_v0_2_dota1_epoch12_trainval_and_submission_gpu89.sh')
CHECKPOINT = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_12.pth')
TRAINVAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_epoch12_trainval_raw20995_gpu89_20260815')
SUBMISSION_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_epoch12_test_submission_gpu89_20260815')
SUBMISSION_PREFIX = f'{SUBMISSION_WORK_DIR}/dota_v1_task1_epoch12'


def test_trainval_eval_is_raw_all_patch_dota_v1_self_eval():
    train = Config.fromfile(TRAIN_CONFIG)
    cfg = Config.fromfile(TRAINVAL_CONFIG)

    assert cfg.model == train.model
    assert cfg.test_cfg.type == 'TestLoop'
    assert cfg.test_dataloader.batch_size == 2
    assert cfg.test_dataloader.dataset.type == 'DOTADataset'
    assert cfg.test_dataloader.dataset.ann_file == 'trainval/annfiles/'
    assert cfg.test_dataloader.dataset.data_prefix.img_path == (
        'trainval/images/')
    assert cfg.test_dataloader.dataset.test_mode is True
    assert cfg.test_dataloader.dataset.get('filter_cfg') is None

    pipeline_types = [
        transform.type for transform in cfg.test_dataloader.dataset.pipeline
    ]
    assert pipeline_types == [
        'mmdet.LoadImageFromFile', 'mmdet.Resize',
        'mmdet.LoadAnnotations', 'ConvertBoxType', 'mmdet.PackDetInputs'
    ]
    assert cfg.test_evaluator.type == 'DOTAMetric'
    assert cfg.test_evaluator.metric == 'mAP'
    assert cfg.test_evaluator.iou_thrs == 0.5
    assert cfg.test_evaluator.eval_mode == '11points'
    assert cfg.work_dir == TRAINVAL_WORK_DIR


def test_trainval_eval_builds_all_20995_patches_including_empty_tiles():
    register_all_modules()
    cfg = Config.fromfile(TRAINVAL_CONFIG)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    assert len(dataset) == 20995


def test_submission_config_uses_unlabelled_test_and_merges_patches():
    train = Config.fromfile(TRAIN_CONFIG)
    cfg = Config.fromfile(SUBMISSION_CONFIG)

    assert cfg.model == train.model
    assert cfg.test_cfg.type == 'TestLoop'
    assert cfg.test_dataloader.batch_size == 2
    assert cfg.test_dataloader.dataset.type == 'DOTADataset'
    assert cfg.test_dataloader.dataset.get('ann_file', '') == ''
    assert cfg.test_dataloader.dataset.data_prefix.img_path == 'test/images/'
    assert cfg.test_dataloader.dataset.test_mode is True

    pipeline_types = [
        transform.type for transform in cfg.test_dataloader.dataset.pipeline
    ]
    assert pipeline_types == [
        'mmdet.LoadImageFromFile', 'mmdet.Resize', 'mmdet.PackDetInputs'
    ]
    assert cfg.test_evaluator.type == 'DOTAMetric'
    assert cfg.test_evaluator.format_only is True
    assert cfg.test_evaluator.merge_patches is True
    assert cfg.test_evaluator.iou_thr == 0.1
    assert cfg.test_evaluator.outfile_prefix == SUBMISSION_PREFIX
    assert cfg.work_dir == SUBMISSION_WORK_DIR


def test_submission_config_builds_all_10833_official_test_patches():
    register_all_modules()
    cfg = Config.fromfile(SUBMISSION_CONFIG)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    assert len(dataset) == 10833


def test_eval_launcher_is_sequential_fail_fast_and_gpu89_safe():
    text = EVAL_LAUNCHER.read_text()

    for required in (
            'set -euo pipefail', 'trap on_error ERR', CHECKPOINT,
            TRAINVAL_CONFIG.name, SUBMISSION_CONFIG.name,
            TRAINVAL_WORK_DIR, SUBMISSION_WORK_DIR, SUBMISSION_PREFIX,
            'PYTHONNOUSERSITE=1', 'PYTHONPATH="${repo_root}"',
            'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1',
            '/data/zcy/anaconda3/envs/orbdet/bin/python',
            '--nproc_per_node=2', '--master_port=29647',
            '--master_port=29648', '--launcher=pytorch',
            'tools/test.py', 'TRAINVAL_COMPLETE',
            'SUBMISSION_COMPLETE', 'COMPLETE', 'FAILED'):
        assert required in text, required

    assert text.index(TRAINVAL_CONFIG.name) < text.index(
        SUBMISSION_CONFIG.name)
    assert 'tools/train.py' not in text
    assert 'rtk test -' not in text
    assert 'rm -' not in text
