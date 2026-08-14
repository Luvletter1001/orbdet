_base_ = './h2rbox_r50_hrsc_200e_2gpu_officiallike.py'

# Eight images across two ranks at batch 2 gives exactly two optimizer steps.
train_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(indices=8))

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
        end=2)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/smoke/'
    'h2rbox_r50_hrsc_200e_gpu89_bs2_20260814')
