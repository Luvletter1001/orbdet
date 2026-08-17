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
MS_POSTEVAL = (
    SCRIPTS / 'eval/' /
    'run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh')
SS_POSTEVAL = (
    SCRIPTS / 'eval/' /
    'run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh')
TRAIN_LAUNCHERS = (
    SCRIPTS / 'smoke/' /
    'run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh',
    SCRIPTS / 'smoke/' /
    'run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh',
    SCRIPTS / 'formal/' /
    'run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh',
    SCRIPTS / 'formal/' /
    'run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh',
)


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


def test_training_deadline_is_refreshed_immediately_before_group_timeout():
    refresh = 'remaining_seconds=$(( deadline_epoch - $(rtk date +%s) ))'
    timeout = (
        'rtk timeout --signal=TERM --kill-after=30s '
        '"${remaining_seconds}s"')
    for path in TRAIN_LAUNCHERS:
        text = path.read_text()
        assert '--foreground' not in text
        assert text.count(refresh) == 1
        assert text.count(timeout) == 1
        refresh_index = text.index(refresh)
        timeout_index = text.index(timeout)
        assert refresh_index > text.index('existing_checkpoints=')
        assert refresh_index > text.index('gpu_processes=')
        assert refresh_index < text.index('rtk touch "${work_dir}/RUNNING"')
        assert 0 < timeout_index - refresh_index < 500


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
    assert 'epoch_13.pth' in text
    assert 'TIME_LIMIT_REACHED' in text
    assert text.count(
        'fail_controller TIME_LIMIT_REACHED 124 INTERRUPTED') >= 2
    required_launchers = (
        'run_orbdet_v0_2_dota1_msrr_resume_e3_to_e4_gpu4567_smoke.sh',
        'run_orbdet_v0_2_dota1_msrr_resume_e3_to_e8_gpu4567.sh',
        'run_orbdet_v0_2_dota1_ss_seed42_gpu89_smoke.sh',
        'run_orbdet_v0_2_dota1_ss_seed42_gpu89.sh',
        'run_orbdet_v0_2_dota1_ss_seed42_epoch12_posteval_gpu89.sh',
        'run_orbdet_v0_2_dota1_msrr_epoch8_posteval_gpu4567.sh',
    )
    for launcher in required_launchers:
        assert launcher in text
    assert '51446' in text
    assert 'math.isfinite' in text
    assert 'loss' in text
    assert 'grad_norm' in text
    assert 'time' in text
    assert 'wait "${ms_pid}"' in text
    assert 'wait "${ss_pid}"' in text
    assert 'jobs -pr' in text
    assert 'kill -0' not in text
    assert 'setsid --wait' in text
    assert 'ms_session_leader.pid' in text
    assert 'ss_session_leader.pid' in text
    assert 'kill -TERM -- "-${pgid}"' in text
    assert 'kill -TERM "${pid}"' not in text
    assert text.index('ss_posteval_launcher') < text.index(
        'ms_posteval_launcher')
    assert 'tmux' not in text
    assert 'nohup' not in text
    assert 'rm -' not in text


def test_bounded_posteval_launchers_pin_resources_validate_contracts_and_outputs():
    cases = (
        (MS_POSTEVAL,
         '/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818/epoch_8.pth',
         'CUDA_VISIBLE_DEVICES=4,5,6,7', '--nproc_per_node=4',
         ('--master_port=29674', '--master_port=29675', '--master_port=29676'),
         '/data1/zcy/Orbdet/work_dirs/.gpu_4_5_6_7.launch_lock',
         ('--expected-epoch 8', '--expected-iter 136656',
          '--config-token OrbdetV02Detector',
          '--config-token trainval_ms_full'),
         ('orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_trainval_eval.py',
          'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ss_test_submission.py',
          'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ms_test_submission.py'),
         ('TRAINVAL_COMPLETE', 'SS_SUBMISSION_COMPLETE',
          'MS_SUBMISSION_COMPLETE'),
         ('skip_if_insufficient_window TRAINVAL',
          'skip_if_insufficient_window SS_SUBMISSION',
          'skip_if_insufficient_window MS_SUBMISSION')),
        (SS_POSTEVAL,
         '/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818/epoch_12.pth',
         'CUDA_VISIBLE_DEVICES=8,9', '--nproc_per_node=2',
         ('--master_port=29677', '--master_port=29678'),
         '/data1/zcy/Orbdet/work_dirs/.gpu_8_9.launch_lock',
         ('--expected-epoch 12', '--expected-iter 38280',
          '--config-token OrbdetV02Detector', '--config-token randomness'),
         ('orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py',
          'orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py'),
         ('TRAINVAL_COMPLETE', 'SS_SUBMISSION_COMPLETE'),
         ('skip_if_insufficient_window TRAINVAL',
          'skip_if_insufficient_window SS_SUBMISSION')),
    )
    for (path, checkpoint, gpu, ranks, ports, lock, validator, configs,
         markers, phase_gates) in cases:
        text = path.read_text()
        for required in (
                'set -euo pipefail', DEADLINE,
                'MINIMUM_EVAL_SECONDS:-1500',
                'POSTEVAL_SKIPPED_INSUFFICIENT_WINDOW', checkpoint, gpu,
                ranks, lock, 'NCCL_P2P_DISABLE=1', 'NCCL_IB_DISABLE=1',
                'launch_lock_held=0', 'rtk mkdir "${lock_path}"',
                'rtk rmdir "${lock_path}"', 'trap release_launch_lock EXIT',
                'nvidia-smi -i', '--expected-state-tensors 371',
                'checkpoint_contract.json', 'archive.testzip() is None',
                'names=archive.namelist()', 'len(names) == 15',
                'actual == expected', 'COMPLETE', 'timeout --signal=INT',
                'existing_status_marker="$(rtk find "${status_dir}"',
                '-mindepth 1 -maxdepth 1 -print -quit)',
                '[[ -n "${existing_status_marker}"',
                'FAILED', 'INTERRUPTED', 'CHECKPOINT_VALIDATED',
                *ports, *validator, *configs, *markers, *phase_gates):
            assert required in text, required
        assert [text.index(name) for name in configs] == sorted(
            text.index(name) for name in configs)
        assert [text.index(gate) for gate in phase_gates] == sorted(
            text.index(gate) for gate in phase_gates)
        for gate, port in zip(phase_gates, ports):
            assert text.index(gate) < text.index(port)
        assert text.index('existing_status_marker=') < text.index(
            'gpu_processes=')
        assert 'tools/train.py' not in text
        assert '--resume' not in text
        assert 'rm -' not in text
        assert 'flock' not in text
