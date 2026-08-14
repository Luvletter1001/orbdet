_base_ = './h2rbox_r50_hrsc_200e_2gpu_officiallike.py'

# Recovery gate for the collapsed clean baseline.  Preserve the clean
# train/val/test split and model, while restoring the optimizer granularity of
# the successful trainval run: one image per rank and AdamW at 1e-4.
train_dataloader = dict(batch_size=1)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=60, val_interval=30)

optim_wrapper = dict(optimizer=dict(lr=0.0001))

# The successful run used 309 optimizer steps/epoch, with LR drops at epochs
# 340 and 468 and a 510E endpoint.  Express those boundaries directly in
# optimizer steps so the clean 436-image split cannot silently shorten them.
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
        end=510 * 309,
        by_epoch=False,
        milestones=[340 * 309, 468 * 309],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=30,
        max_keep_ckpts=2,
        save_best='dota/mAP',
        rule='greater'))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'h2rbox_r50_hrsc_recovery_bs1_stepaligned_gpu89_seed3407_20260814')
