_base_ = './orbdet_v0_1_r50_hrsc.py'

# Pure H2RBox baseline.  Keep the same CNN, FPN, HRSC transforms, image size
# and HBox-only training targets as OrbDet-v0.1, but restore the original
# H2RBox detector and consistency loss.
model = dict(
    type='H2RBoxDetector',
    bbox_head=dict(
        loss_bbox_ss=dict(
            _delete_=True,
            type='H2RBoxConsistencyLoss',
            loss_weight=0.4,
            center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0))))

# Paper protocol: tune on train/val and keep the official test split held out.
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(ann_file='ImageSets/train.txt'))
val_dataloader = dict(dataset=dict(ann_file='ImageSets/val.txt'))
test_dataloader = dict(dataset=dict(ann_file='ImageSets/test.txt'))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

# Four ranks x sixteen images gives a global batch of 64.  H2RBox's reference
# AdamW recipe uses 1e-4 at global batch 16, so use linear LR scaling here.
# Scheduler ratios follow 8/11 of 12E.
optim_wrapper = dict(optimizer=dict(lr=0.0004))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[67, 92],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=10,
        save_best='dota/mAP',
        rule='greater'))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'h2rbox_r50_hrsc_trainval_100e_gpu0123_bs16_seed3407')
