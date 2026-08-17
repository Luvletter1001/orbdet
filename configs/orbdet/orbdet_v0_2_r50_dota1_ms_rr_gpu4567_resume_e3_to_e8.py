_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=999)
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=6,
        save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818')
