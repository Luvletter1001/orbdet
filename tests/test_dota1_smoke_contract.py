import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version
import warnings

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/orbdet/h2rbox_r50_dota1_smoke.py"
FORMAL_CONFIG = ROOT / "configs/orbdet/orbdet_v0_1_r50_hrsc.py"
CALIBRATION_CONFIG = (
    ROOT / "configs/orbdet/orbdet_v0_1_r50_hrsc_calibration.py")
TRAIN_SCRIPT = ROOT / "scripts/smoke/run_dota1_gpu89.sh"
TEST_SCRIPT = ROOT / "scripts/smoke/test_dota1_gpu89.sh"
STOP_MARKER = ROOT / "FORMAL_TRAINING_NOT_STARTED.md"
AGENT_RULES = ROOT / "AGENTS.md"


def test_smoke_artifacts_exist():
    assert CONFIG.is_file(), f"missing smoke config: {CONFIG}"
    assert TRAIN_SCRIPT.is_file(), f"missing train launcher: {TRAIN_SCRIPT}"
    assert TEST_SCRIPT.is_file(), f"missing test launcher: {TEST_SCRIPT}"


def test_complete_mmengine_runtime_is_not_shadowed_by_partial_source():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mmengine; "
                "from mmengine.runner import Runner; "
                "print(mmengine.__file__)"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/site-packages/mmengine/" in result.stdout


def test_numpy_runtime_matches_metadata_and_pycocotools_abi():
    import numpy
    import pycocotools.mask  # noqa: F401

    assert numpy.__version__ == "1.26.4"
    assert version("numpy") == numpy.__version__


def test_requests_has_a_working_charset_detector():
    import charset_normalizer

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import requests  # noqa: F401

    dependency_warnings = [
        warning
        for warning in caught
        if "character detection dependency" in str(warning.message)
    ]
    assert charset_normalizer.__version__ == "3.4.4"
    assert not dependency_warnings


def test_config_is_a_bounded_hbox_supervised_h2rbox_run():
    cfg = Config.fromfile(CONFIG)

    assert cfg.model.type == "H2RBoxDetector"
    assert cfg.model.backbone.init_cfg is None
    assert tuple(cfg.model.crop_size) == (256, 256)
    assert tuple(cfg.model.bbox_head.crop_size) == (256, 256)

    transforms = cfg.train_dataloader.dataset.pipeline
    box_mappings = [
        transform["box_type_mapping"]["gt_bboxes"]
        for transform in transforms
        if transform["type"] == "ConvertBoxType"
    ]
    assert box_mappings == ["hbox", "rbox"]

    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.num_workers == 0
    assert cfg.train_dataloader.persistent_workers is False
    assert cfg.train_dataloader.batch_sampler is None
    assert cfg.train_dataloader.sampler.type == "DefaultSampler"
    assert cfg.train_dataloader.dataset.indices <= 8

    assert cfg.train_cfg.type == "EpochBasedTrainLoop"
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.val_cfg is None
    assert cfg.val_dataloader is None
    assert cfg.val_evaluator is None

    assert cfg.test_dataloader.batch_size == 1
    assert cfg.test_dataloader.num_workers == 0
    assert cfg.test_dataloader.dataset.indices <= 4
    assert cfg.test_evaluator.type == "DOTAMetric"
    assert cfg.default_hooks.checkpoint.interval == 1
    assert "/work_dirs/smoke/" in cfg.work_dir


def test_launchers_are_fixed_to_gpu89_two_rank_smoke_only():
    train_text = TRAIN_SCRIPT.read_text()
    test_text = TEST_SCRIPT.read_text()

    required_tokens = (
        "rtk env",
        "CUDA_VISIBLE_DEVICES=8,9",
        "NCCL_P2P_DISABLE=1",
        "NCCL_IB_DISABLE=1",
        "/data/zcy/anaconda3/envs/orbdet/bin/python",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--master_port=29689",
        "--launcher=pytorch",
        "h2rbox_r50_dota1_smoke.py",
    )
    for token in required_tokens:
        assert token in train_text
        assert token in test_text

    assert "tools/train.py" in train_text
    assert "tools/test.py" in test_text
    assert "epoch_1.pth" in test_text

    forbidden_tokens = (
        "--resume",
        "nohup",
        "tmux",
        "run_train_queue",
        "formal_train",
    )
    for token in forbidden_tokens:
        assert token not in train_text
        assert token not in test_text


def test_formal_training_authorization_is_recorded_during_lifecycle():
    assert STOP_MARKER.is_file()
    assert AGENT_RULES.is_file()

    marker_text = STOP_MARKER.read_text()
    rules_text = AGENT_RULES.read_text()
    assert any(
        f"formal_train_status: {status}" in marker_text
        for status in ('authorized_preflight', 'running', 'complete'))
    assert "user_explicit_authorization_required: satisfied" in marker_text
    assert "authorized_scope: orbdet_v0.2_hrsc_clean_gpu89" in marker_text
    assert "Do not start formal training" in rules_text
    assert "explicitly authorizes" in rules_text

    project_configs = sorted((ROOT / "configs/orbdet").glob("*.py"))
    for required_config in (CONFIG, FORMAL_CONFIG, CALIBRATION_CONFIG):
        assert required_config in project_configs
    assert (ROOT / 'configs/orbdet/'
            'orbdet_v0_2_r50_hrsc_clean_gpu89.py') in project_configs
