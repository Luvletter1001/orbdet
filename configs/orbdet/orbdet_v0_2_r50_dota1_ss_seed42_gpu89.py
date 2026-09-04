_base_ = './orbdet_v0_2_r50_dota1_1x_gpu89.py'

randomness = dict(seed=42, deterministic=False)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=4,
        save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ss_gpu89_seed42_20260818')
