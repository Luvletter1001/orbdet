"""Configuration contract for an honest single-GPU GDA comparison."""
from pathlib import Path

from mmengine import Config


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / 'configs' / 'orbdet'


def _load(name: str) -> Config:
    return Config.fromfile(CONFIG_DIR / name)


def test_single_gpu_control_matches_e0_global_batch_and_update_schedule():
    e0 = _load('orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu89.py')
    control = _load('orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu4_control.py')

    assert control.intended_world_size == 1
    assert control.global_batch_size == 4
    assert control.train_dataloader.batch_size == 4
    assert e0.train_dataloader.batch_size * 2 == control.global_batch_size
    assert control.optim_wrapper == e0.optim_wrapper
    assert control.param_scheduler == e0.param_scheduler
    assert control.train_cfg == e0.train_cfg
    assert control.train_dataloader.dataset == e0.train_dataloader.dataset
    assert control.randomness == e0.randomness


def test_gda_and_single_gpu_null_control_differ_only_in_model_and_work_dir():
    control = _load('orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu4_control.py')
    gda = _load('orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4.py')

    assert gda.intended_world_size == control.intended_world_size == 1
    assert gda.global_batch_size == control.global_batch_size == 4
    assert gda.train_dataloader.batch_size == control.train_dataloader.batch_size

    control_dict = control.to_dict()
    gda_dict = gda.to_dict()
    allowed = {'model', 'work_dir'}
    differing = {
        key for key in control_dict.keys() | gda_dict.keys()
        if control_dict.get(key) != gda_dict.get(key)
    }
    assert differing == allowed

    assert control.model.type == 'OrbdetV02Detector'
    assert gda.model.type == 'OrbdetGDADetector'
    assert gda.model.bbox_head.gda_probe.enabled is True
