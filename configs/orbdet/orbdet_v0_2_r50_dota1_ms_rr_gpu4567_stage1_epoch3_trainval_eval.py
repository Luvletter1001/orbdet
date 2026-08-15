_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py'

data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'
image_size = (1024, 1024)

# Raw all-patch trainval self-evaluation. Empty-GT tiles are intentionally
# retained, giving the auditable 20,995-patch DOTA-v1.0 evaluation universe.
test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
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
    'orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/trainval')
