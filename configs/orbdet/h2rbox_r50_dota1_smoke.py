_base_ = [
    '../../mmrotate_configs/h2rbox/'
    'h2rbox-le90_r50_fpn_adamw-1x_dota.py'
]

data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'

model = dict(
    crop_size=(256, 256),
    backbone=dict(init_cfg=None),
    bbox_head=dict(crop_size=(256, 256)))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(256, 256), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(256, 256), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        indices=8,
        pipeline=train_pipeline))

val_cfg = None
val_dataloader = None
val_evaluator = None

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        indices=4,
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(type='DOTAMetric', metric='mAP')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=999)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05))

default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=1, max_keep_ckpts=1))

randomness = dict(seed=3407, deterministic=False)
load_from = None
resume = False
work_dir = '/data1/zcy/Orbdet/work_dirs/smoke/dota1_h2rbox_gpu89'

