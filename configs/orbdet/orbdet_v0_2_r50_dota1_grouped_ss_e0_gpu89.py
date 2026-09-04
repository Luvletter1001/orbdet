# Recovered verbatim from the formal E0 run dir
# (work_dirs/formal/orbdet_v0_2_dota1_grouped_ss_e0_gpu89_seed3407/,
#  config sha256 bf02d0855c12e4d377f85d2f7df4283613613a58e4cb65840cb81aca143a32be).
# This flattened dump is the authoritative source of the E0 recipe; the
# original source config did not survive under configs/.  GDA Plan-B
# configs inherit from this file.
angle_version = 'le90'
data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'
dataset_type = 'HRSCDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=4, max_keep_ckpts=3, save_last=True, type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
grouped_holdout_root = '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407/holdout/'
grouped_root = '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407'
grouped_train_root = '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407/train/'
image_size = (
    1024,
    1024,
)
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='/data1/zcy/Orbdet/weights/resnet50-0676ba61.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    bbox_head=dict(
        angle_coder=dict(
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0,
            type='PSCCoder'),
        angle_version='le90',
        bbox_coder=dict(angle_version='le90', type='DistanceAnglePointCoder'),
        center_sample_radius=1.5,
        center_sampling=True,
        centerness_on_reg=True,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='mmdet.IoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        loss_symmetry_ss=dict(
            high_quality_thr=0.75,
            loss_flp=dict(
                beta=0.1, loss_weight=0.05, type='mmdet.SmoothL1Loss'),
            loss_rot=dict(
                beta=0.1, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
            quality_temperature=0.25,
            type='OrbdetAnchoredSymmetryLoss',
            use_snap_loss=True),
        norm_on_bbox=True,
        num_classes=15,
        rotation_agnostic_classes=[
            9,
            11,
        ],
        scale_angle=False,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='H2RBoxV2Head',
        use_circumiou_loss=True,
        use_hbbox_loss=False,
        use_reweighted_loss_bbox=False,
        use_standalone_angle=True),
    crop_size=(
        1024,
        1024,
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='mmdet.FPN'),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.1, type='nms_rotated'),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=None,
    type='OrbdetV02Detector',
    view_range=(
        0.25,
        0.75,
    ))
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=3407)
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        512,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        data_root=
        '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407/train/',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='hbox'),
                type='ConvertBoxType'),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='hbox'), type='ConvertBoxType'),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        data_root=
        '/data1/zcy/Orbdet/work_dirs/derived/orbdet_v04r_dota1_grouped_seed3407/holdout/',
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    eval_mode='11points', iou_thrs=0.5, metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_2_dota1_grouped_ss_e0_gpu89_seed3407'
