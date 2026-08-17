_base_ = './orbdet_v0_2_r50_dota1_ms_rr_gpu4567_resume_e3_to_e8.py'

train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=8))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=999)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_last=True))
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_resume_e3_to_e4_20260818')
