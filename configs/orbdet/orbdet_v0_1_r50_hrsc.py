_base_ = [
    '../../mmrotate_configs/h2rbox/'
    'h2rbox-le90_r50_fpn_adamw-1x_dota.py'
]

data_root = '/data1/zcy/Orbdet/data/hrsc/'
file_client_args = dict(backend='disk')
image_size = (800, 800)

model = dict(
    type='OrbdetDetector',
    crop_size=image_size,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                '/data1/zcy/Orbdet/weights/resnet50-0676ba61.pth'))),
    bbox_head=dict(
        num_classes=1,
        crop_size=image_size,
        square_classes=[],
        rotation_agnostic_classes=[],
        loss_bbox_ss=dict(
            _delete_=True,
            type='OrbdetHarmonicConsistencyLoss',
            loss_weight=0.4,
            min_quality=0.25,
            gamma=2.0,
            high_quality_thr=0.75,
            center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0))))

train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # The only training target exposed to the detector is the enclosing HBox.
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Pad',
        size=image_size,
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
    # Keep official OBBs in original coordinates for metric computation.
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad',
        size=image_size,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/trainval.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/test.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=510, val_interval=30)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
        end=510,
        by_epoch=True,
        milestones=[340, 468],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=30,
        max_keep_ckpts=3,
        save_best='dota/mAP',
        rule='greater'))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = '/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_gpu89'
