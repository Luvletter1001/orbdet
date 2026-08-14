_base_ = './h2rbox_r50_hrsc_100e_4gpu.py'

# Final four-GPU baseline protocol requested after the bs16 diagnostic run.
# Eight images per rank gives global batch 32 and 14 optimizer steps/epoch.
train_dataloader = dict(batch_size=8)
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)

# H2RBox reference AdamW LR is 1e-4 at global batch 16.
optim_wrapper = dict(optimizer=dict(lr=0.0002))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=200,
        by_epoch=True,
        milestones=[133, 184],
        gamma=0.1)
]

# Retain all 10E snapshots so transient failures remain diagnosable.
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=14),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=20,
        save_best='dota/mAP',
        rule='greater'))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'h2rbox_r50_hrsc_trainval_200e_gpu0123_bs8_seed3407')
