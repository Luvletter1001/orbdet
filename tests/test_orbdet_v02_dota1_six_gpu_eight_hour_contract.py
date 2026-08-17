from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/orbdet'
SCRIPTS = ROOT / 'scripts'
MS_BASE = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py'
MS_RESUME = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py'
MS_SMOKE = CFG / 'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e4_smoke.py'
SS_BASE = CFG / 'orbdet_v0_2_r50_dota1_1x_gpu89.py'
SS42 = CFG / 'orbdet_v0_2_r50_dota1_ss_seed42_gpu89.py'
SS42_SMOKE = CFG / 'orbdet_v0_2_r50_dota1_ss_seed42_gpu89_smoke.py'
DEADLINE = '2026-08-18 08:20:00 +0800'


def test_ms_resume_preserves_global_batch_and_targets_epoch8():
    base = Config.fromfile(MS_BASE)
    cfg = Config.fromfile(MS_RESUME)
    assert cfg.model == base.model
    assert cfg.train_dataloader == base.train_dataloader
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.optim_wrapper.optimizer.lr == 1e-4
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.005
    assert cfg.train_cfg.max_epochs == 8
    assert cfg.train_cfg.val_interval == 999
    assert cfg.randomness.seed == 3407
    assert cfg.default_hooks.checkpoint.interval == 1
    assert 'resume_e3_to_e8_20260818' in cfg.work_dir


def test_ms_resume_smoke_is_exactly_two_steps_after_epoch3():
    cfg = Config.fromfile(MS_SMOKE)
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_dataloader.persistent_workers is False
    assert cfg.train_cfg.max_epochs == 4
    assert cfg.default_hooks.logger.interval == 1


def test_ss42_changes_only_seed_hooks_and_work_dir():
    base = Config.fromfile(SS_BASE)
    cfg = Config.fromfile(SS42)
    assert cfg.model == base.model
    assert cfg.train_dataloader == base.train_dataloader
    assert cfg.optim_wrapper == base.optim_wrapper
    assert cfg.param_scheduler == base.param_scheduler
    assert cfg.train_cfg == base.train_cfg
    assert cfg.randomness.seed == 42
    assert cfg.default_hooks.checkpoint.interval == 1
    assert 'seed42_20260818' in cfg.work_dir


def test_ss42_smoke_is_two_global_steps():
    cfg = Config.fromfile(SS42_SMOKE)
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.param_scheduler[0].end == 2


def test_launchers_pin_resources_resume_and_deadline():
    cases = (
        ('formal/run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh',
         'CUDA_VISIBLE_DEVICES=4,5,6,7', '--nproc_per_node=4',
         '--resume="${source_checkpoint}"'),
        ('formal/run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh',
         'CUDA_VISIBLE_DEVICES=8,9', '--nproc_per_node=2', '--resume'),
    )
    for rel, gpu, ranks, resume in cases:
        text = (SCRIPTS / rel).read_text()
        assert 'set -euo pipefail' in text
        assert gpu in text
        assert ranks in text
        assert 'NCCL_P2P_DISABLE=1' in text
        assert 'NCCL_IB_DISABLE=1' in text
        assert DEADLINE in text
        assert 'existing_checkpoints' in text
        assert 'gpu_processes' in text
        if resume.startswith('--resume='):
            assert resume in text
        else:
            assert resume not in text


def test_controller_orders_primary_before_secondary_and_never_overreaches():
    path = SCRIPTS / 'formal/run_orbdet_v0_2_dota1_six_gpu_eight_hour_20260818.sh'
    text = path.read_text()
    assert DEADLINE in text
    assert 'nvidia-smi -i 4,5,6,7' in text
    assert 'nvidia-smi -i 8,9' in text
    assert text.index('msrr_resume_e3_to_e8') < text.index('ss_seed42')
    assert 'epoch_8.pth' in text
    assert 'epoch_12.pth' in text
    assert 'epoch_9.pth' in text
    assert 'TIME_LIMIT_REACHED' in text
    assert 'rm -' not in text
