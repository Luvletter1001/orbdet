_base_ = './orbdet_v0_1_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py'

# Four images at one image/rank give exactly two global optimizer steps.
train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=list(range(4))))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
val_cfg = None
val_dataloader = None
val_evaluator = None

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best=None))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'orbdet_v0_1_hrsc_recovery_bs1_gpu89_20260814')
