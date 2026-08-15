_base_ = './orbdet_v0_2_r50_dota1_1x_gpu89.py'

data_root = '/data/zcy/dataset/trainval_ms_full/'
image_size = (1024, 1024)

# Reproduce the optimization contract embedded in the official 78.25 AP50
# checkpoint while retaining Orbdet-v0.2's detached diagnostics.
model = dict(
    bbox_head=dict(
        rotation_agnostic_classes=[1, 9, 11],
        agnostic_resize_classes=[1],
        use_circumiou_loss=True,
        use_standalone_angle=True,
        use_reweighted_loss_bbox=True))

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
    dict(type='RandomRotate', prob=1, angle_range=180),
    dict(type='mmdet.PackDetInputs')
]

# One sample/rank on two ranks preserves the official global batch of two and
# therefore its optimizer-step count (~409,956 over 12 epochs).
train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        img_shape=image_size,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

optim_wrapper = dict(
    optimizer=dict(lr=0.00005, weight_decay=0.005))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89_seed3407_20260815')
