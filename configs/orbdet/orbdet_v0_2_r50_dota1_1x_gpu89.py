_base_ = './orbdet_v0_2_r50_hrsc_clean_gpu89.py'

data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'
image_size = (1024, 1024)

model = dict(
    crop_size=image_size,
    bbox_head=dict(
        num_classes=15,
        rotation_agnostic_classes=[9, 11]))

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=image_size,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

# The prepared DOTA tree has no independent labelled validation split.
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=12, val_interval=999)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))

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
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=4,
        max_keep_ckpts=3,
        save_last=True))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815')
