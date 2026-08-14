_base_ = './h2rbox_r50_hrsc_100e_4gpu.py'

# Clean H2RBox baseline: train/val select the checkpoint and keep HRSC test
# untouched until the final one-shot evaluation.
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(ann_file='ImageSets/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='ImageSets/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='ImageSets/test.txt'))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)

# Closest official HRSC weak-supervision optimization recipe, extended to the
# user-selected 200E horizon while preserving the 8/11 decay proportions.
optim_wrapper = dict(optimizer=dict(lr=0.00005))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[133, 184],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=20,
        save_best='dota/mAP',
        rule='greater'))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'h2rbox_r50_hrsc_train_val_200e_gpu89_bs2_seed3407_20260814')
