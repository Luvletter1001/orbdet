_base_ = './orbdet_v0_2_r50_dota1_1x_gpu89.py'

data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'
image_size = (1024, 1024)

# Raw all-patch trainval self-evaluation. This intentionally keeps empty-GT
# tiles, so dataset_len is 20,995 rather than the 12,757 non-empty tiles used
# by the training sampler.
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
    # Load after resize so the evaluation boxes remain in original coordinates.
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_cfg = dict(_delete_=True, type='TestLoop')
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
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=image_size,
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    _delete_=True,
    type='DOTAMetric',
    metric='mAP',
    iou_thrs=0.5,
    eval_mode='11points')

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_epoch12_trainval_raw20995_gpu89_20260815')
