from pathlib import Path

from mmengine.config import Config

from mmrotate.registry import DATASETS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs/orbdet'
SCRIPT_DIR = ROOT / 'scripts/eval'

STAGE_CONFIG = (
    CONFIG_DIR / 'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py')
TRAINVAL_CONFIG = (
    CONFIG_DIR /
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_trainval_eval.py')
SS_CONFIG = (
    CONFIG_DIR /
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ss_test_submission.py')
MS_CONFIG = (
    CONFIG_DIR /
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ms_test_submission.py')
LAUNCHER = (
    SCRIPT_DIR /
    'run_orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_posteval.sh')
WATCHER = (
    SCRIPT_DIR /
    'run_orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_posteval_when_ready.sh')

STAGE_ROOT = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816')
CHECKPOINT = f'{STAGE_ROOT}/epoch_3.pth'
EVAL_ROOT = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816')
TRAINVAL_WORK_DIR = f'{EVAL_ROOT}/trainval'
SS_WORK_DIR = f'{EVAL_ROOT}/ss_submission'
MS_WORK_DIR = f'{EVAL_ROOT}/ms_submission'
SS_PREFIX = f'{SS_WORK_DIR}/orbdet_v0_2_msrr_stage1_epoch3_ss_task1'
MS_PREFIX = f'{MS_WORK_DIR}/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1'


def test_stage3_eval_configs_preserve_trained_model_contract():
    stage = Config.fromfile(STAGE_CONFIG)

    for path in (TRAINVAL_CONFIG, SS_CONFIG, MS_CONFIG):
        cfg = Config.fromfile(path)
        assert cfg.model == stage.model
        assert cfg.test_cfg.type == 'TestLoop'
        assert cfg.test_dataloader.batch_size == 2
        assert cfg.test_dataloader.sampler.type == 'DefaultSampler'
        assert cfg.test_dataloader.sampler.shuffle is False


def test_stage3_trainval_is_raw_20995_patch_dota_v1_eval():
    cfg = Config.fromfile(TRAINVAL_CONFIG)

    assert cfg.test_dataloader.dataset.data_root == (
        '/data1/zcy/Orbdet/data/DOTA-v1.0/')
    assert cfg.test_dataloader.dataset.ann_file == 'trainval/annfiles/'
    assert cfg.test_dataloader.dataset.data_prefix.img_path == (
        'trainval/images/')
    assert cfg.test_dataloader.dataset.test_mode is True
    assert cfg.test_dataloader.dataset.get('filter_cfg') is None
    assert [item.type for item in cfg.test_dataloader.dataset.pipeline] == [
        'mmdet.LoadImageFromFile', 'mmdet.Resize',
        'mmdet.LoadAnnotations', 'ConvertBoxType', 'mmdet.PackDetInputs'
    ]
    assert cfg.test_evaluator.type == 'DOTAMetric'
    assert cfg.test_evaluator.metric == 'mAP'
    assert cfg.test_evaluator.iou_thrs == 0.5
    assert cfg.test_evaluator.eval_mode == '11points'
    assert cfg.work_dir == TRAINVAL_WORK_DIR


def test_stage3_submission_configs_cover_ss_and_ms_official_test_sets():
    ss = Config.fromfile(SS_CONFIG)
    ms = Config.fromfile(MS_CONFIG)

    assert ss.test_dataloader.dataset.data_root == (
        '/data1/zcy/Orbdet/data/DOTA-v1.0/')
    assert ss.test_dataloader.dataset.data_prefix.img_path == 'test/images/'
    assert ms.test_dataloader.dataset.data_root == '/data/zcy/dataset/test_ms/'
    assert ms.test_dataloader.dataset.data_prefix.img_path == 'images/'
    for cfg, work_dir, prefix in (
            (ss, SS_WORK_DIR, SS_PREFIX), (ms, MS_WORK_DIR, MS_PREFIX)):
        assert cfg.test_dataloader.dataset.get('ann_file', '') == ''
        assert cfg.test_dataloader.dataset.test_mode is True
        assert cfg.test_evaluator.type == 'DOTAMetric'
        assert cfg.test_evaluator.format_only is True
        assert cfg.test_evaluator.merge_patches is True
        assert cfg.test_evaluator.iou_thr == 0.1
        assert cfg.test_evaluator.outfile_prefix == prefix
        assert cfg.work_dir == work_dir


def test_stage3_eval_datasets_have_expected_patch_counts():
    register_all_modules()
    for path, expected in (
            (TRAINVAL_CONFIG, 20995), (SS_CONFIG, 10833),
            (MS_CONFIG, 71888)):
        cfg = Config.fromfile(path)
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        assert len(dataset) == expected


def test_stage3_posteval_launcher_is_sequential_fail_fast_and_gpu4567_safe():
    text = LAUNCHER.read_text()

    for required in (
            'set -euo pipefail', 'trap on_error ERR', STAGE_ROOT,
            'checkpoint="${stage_root}/epoch_3.pth"',
            'stage_complete="${stage_root}/COMPLETE"',
            TRAINVAL_CONFIG.name, SS_CONFIG.name, MS_CONFIG.name, EVAL_ROOT,
            'trainval_work_dir="${eval_root}/trainval"',
            'ss_work_dir="${eval_root}/ss_submission"',
            'ms_work_dir="${eval_root}/ms_submission"',
            'ss_prefix="${ss_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_ss_task1"',
            'ms_prefix="${ms_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1"',
            'CUDA_VISIBLE_DEVICES=4,5,6,7',
            'NCCL_P2P_DISABLE=1', 'NCCL_IB_DISABLE=1',
            '--nproc_per_node=4', '--master_port=29665',
            '--master_port=29666', '--master_port=29667',
            'TRAINVAL_COMPLETE', 'SS_SUBMISSION_COMPLETE',
            'MS_SUBMISSION_COMPLETE', 'COMPLETE', 'FAILED',
            'archive.testzip() is None'):
        assert required in text, required
    assert text.index(TRAINVAL_CONFIG.name) < text.index(
        SS_CONFIG.name) < text.index(MS_CONFIG.name)
    assert 'tools/train.py' not in text
    assert '--resume' not in text
    assert 'rm -' not in text


def test_stage3_posteval_watcher_obeys_absolute_window_and_never_trains():
    text = WATCHER.read_text()

    for required in (
            'set -euo pipefail', '2026-08-16 09:50:00 +0800',
            'MINIMUM_EVAL_SECONDS:-1800', 'POLL_SECONDS:-30',
            STAGE_ROOT, 'stage_complete="${stage_root}/COMPLETE"',
            '${stage_root}/FAILED', '${stage_root}/INTERRUPTED',
            'TIME_LIMIT_REACHED',
            'SKIPPED_INSUFFICIENT_WINDOW', LAUNCHER.name,
            'nvidia-smi -i 4,5,6,7'):
        assert required in text, required
    assert 'tools/train.py' not in text
    assert '--resume' not in text
    assert 'rm -' not in text
