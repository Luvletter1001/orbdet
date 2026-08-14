_base_ = './orbdet_v0_1_r50_hrsc.py'

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)
val_cfg = None
val_dataloader = None
val_evaluator = None

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=100)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/calibration/'
    'orbdet_v0_1_hrsc_gpu89')
