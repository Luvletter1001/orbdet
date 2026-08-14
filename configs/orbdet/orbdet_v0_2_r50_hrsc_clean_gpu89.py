_base_ = [
    '../../mmrotate_configs/_base_/datasets/hrsc.py',
    '../../mmrotate_configs/_base_/schedules/schedule_6x.py',
    '../../mmrotate_configs/_base_/default_runtime.py'
]

angle_version = 'le90'
data_root = '/data1/zcy/Orbdet/data/hrsc/'
file_client_args = dict(backend='disk')
image_size = (800, 800)

# Stability-recovery model: the optimization path is the official H2RBox-v2
# symmetry objective. Orbdet adds detached diagnostics only.
model = dict(
    type='OrbdetV02Detector',
    crop_size=image_size,
    view_range=(0.25, 0.75),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                '/data1/zcy/Orbdet/weights/resnet50-0676ba61.pth'))),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxV2Head',
        num_classes=1,
        in_channels=256,
        angle_version=angle_version,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=False,
        use_circumiou_loss=True,
        use_standalone_angle=True,
        use_reweighted_loss_bbox=False,
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_symmetry_ss=dict(
            type='OrbdetAnchoredSymmetryLoss',
            use_snap_loss=True,
            quality_temperature=0.25,
            high_quality_thr=0.75,
            loss_rot=dict(
                type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1))),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# The train transform irreversibly removes OBB direction before model input.
train_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

# Evaluation keeps the original OBB annotation coordinates.
val_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=file_client_args),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(
        type='HRSCDataset',
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        data_prefix=dict(sub_data_root='FullDataSet/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
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

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=0.5)
test_evaluator = val_evaluator

# 436 clean-train images / global batch 4 = 109 updates per epoch. 103 clean
# epochs match the 11,160 updates of the official 72E trainval recipe.
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=103, val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.005),
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
        end=11160,
        by_epoch=False,
        milestones=[7440, 10230],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(
        type='CheckpointHook',
        interval=12,
        max_keep_ckpts=5,
        save_best='dota/mAP',
        rule='greater',
        save_last=True))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_hrsc_clean_gpu89_seed3407_20260814')

