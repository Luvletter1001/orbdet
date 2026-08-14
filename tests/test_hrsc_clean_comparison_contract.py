import copy
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = (
    ROOT / 'configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py')
CANDIDATE_CONFIG = (
    ROOT / 'configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py')
SMOKE_CONFIG = (
    ROOT /
    'configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py')
SMOKE_LAUNCHER = (
    ROOT / 'scripts/smoke/run_orbdet_v0_1_hrsc_clean_gpu89_bs2_smoke.sh')
FORMAL_LAUNCHER = (
    ROOT / 'scripts/formal/run_orbdet_v0_1_hrsc_clean_200e_gpu89_bs2.sh')
DEADLINE_GUARD = (
    ROOT / 'scripts/formal/guard_orbdet_gpu89_until_0830_20260814.sh')

FORMAL_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814')
SMOKE_WORK_DIR = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_1_hrsc_clean_gpu89_bs2_20260814')


def _load(path: Path) -> Config:
    assert path.is_file(), f'missing required artifact: {path}'
    return Config.fromfile(path)


def test_candidate_matches_baseline_experiment_contract():
    baseline = _load(BASELINE_CONFIG)
    candidate = _load(CANDIDATE_CONFIG)

    for field in (
            'train_dataloader', 'val_dataloader', 'test_dataloader',
            'train_cfg', 'val_cfg', 'test_cfg', 'optim_wrapper',
            'param_scheduler', 'default_hooks', 'randomness', 'val_evaluator',
            'test_evaluator'):
        assert candidate[field] == baseline[field], field

    assert candidate.train_cfg.max_epochs == 200
    assert candidate.train_cfg.val_interval == 10
    assert candidate.train_dataloader.dataset.ann_file == 'ImageSets/train.txt'
    assert candidate.val_dataloader.dataset.ann_file == 'ImageSets/val.txt'
    assert candidate.test_dataloader.dataset.ann_file == 'ImageSets/test.txt'
    assert candidate.work_dir == FORMAL_WORK_DIR


def test_candidate_changes_only_detector_and_consistency_loss():
    baseline = _load(BASELINE_CONFIG)
    candidate = _load(CANDIDATE_CONFIG)

    assert candidate.model.type == 'OrbdetDetector'
    candidate_loss = candidate.model.bbox_head.loss_bbox_ss
    assert candidate_loss.type == 'OrbdetHarmonicConsistencyLoss'
    assert candidate_loss.loss_weight == 0.4
    assert candidate_loss.min_quality == 0.25
    assert candidate_loss.gamma == 2.0
    assert candidate_loss.high_quality_thr == 0.75

    normalized = copy.deepcopy(candidate.model)
    normalized.type = baseline.model.type
    normalized_loss = normalized.bbox_head.loss_bbox_ss
    normalized_loss.type = baseline.model.bbox_head.loss_bbox_ss.type
    for key in ('min_quality', 'gamma', 'high_quality_thr'):
        normalized_loss.pop(key)

    assert normalized == baseline.model


def test_smoke_config_is_bounded_to_eight_images_and_two_steps():
    formal = _load(CANDIDATE_CONFIG)
    smoke = _load(SMOKE_CONFIG)

    assert smoke.model == formal.model
    assert smoke.train_dataloader.batch_size == 2
    assert smoke.train_dataloader.num_workers == 0
    assert smoke.train_dataloader.persistent_workers is False
    assert list(smoke.train_dataloader.dataset.indices) == list(range(8))
    assert smoke.train_cfg.max_epochs == 1
    assert smoke.val_cfg is None
    assert smoke.val_dataloader is None
    assert smoke.val_evaluator is None
    assert smoke.default_hooks.logger.interval == 1
    assert smoke.default_hooks.checkpoint.interval == 1
    assert smoke.default_hooks.checkpoint.max_keep_ckpts == 1
    assert smoke.work_dir == SMOKE_WORK_DIR


def _assert_two_rank_launcher(text: str, config_name: str,
                              work_dir: str, port: int) -> None:
    for required in (
            'PYTHONNOUSERSITE=1', 'PYTHONPATH=/data1/zcy/Orbdet',
            'CUDA_VISIBLE_DEVICES=8,9', 'NCCL_P2P_DISABLE=1',
            'NCCL_IB_DISABLE=1',
            '/data/zcy/anaconda3/envs/orbdet/bin/python',
            '--nproc_per_node=2', f'--master_port={port}',
            '--launcher=pytorch', config_name, work_dir):
        assert required in text
    assert '--resume' not in text


def test_launchers_pin_gpu89_nccl_ports_and_unique_workdirs():
    assert SMOKE_LAUNCHER.is_file(), f'missing: {SMOKE_LAUNCHER}'
    assert FORMAL_LAUNCHER.is_file(), f'missing: {FORMAL_LAUNCHER}'

    smoke_text = SMOKE_LAUNCHER.read_text()
    formal_text = FORMAL_LAUNCHER.read_text()
    _assert_two_rank_launcher(
        smoke_text,
        'orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py',
        SMOKE_WORK_DIR,
        29617)
    _assert_two_rank_launcher(
        formal_text,
        'orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py',
        FORMAL_WORK_DIR,
        29618)
    assert 'pgrep -af' in formal_text
    assert ('[p]ython.*tools/train.py.*'
            'orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py' in formal_text)


def test_deadline_guard_is_absolute_and_exactly_scoped():
    assert DEADLINE_GUARD.is_file(), f'missing: {DEADLINE_GUARD}'
    text = DEADLINE_GUARD.read_text()

    for required in (
            '1786667400', 'sleep 30',
            'h2rbox_hrsc200e_bs2_gpu89_20260814',
            'orbdet_hrsc_clean200e_bs2_gpu89_20260814',
            'orbdet_hrsc_clean_queue_gpu89_20260814',
            'h2rbox_r50_hrsc_200e_2gpu_officiallike.py',
            'orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py',
            'tmux send-keys', 'pgrep -af'):
        assert required in text

    for forbidden in ('pkill tools/train.py', 'killall python'):
        assert forbidden not in text
