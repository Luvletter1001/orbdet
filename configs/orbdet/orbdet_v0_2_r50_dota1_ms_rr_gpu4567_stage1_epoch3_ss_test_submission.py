_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_stage1_3e.py'

data_root = '/data1/zcy/Orbdet/data/DOTA-v1.0/'
image_size = (1024, 1024)

test_pipeline = [
    dict(
        type='mmdet.LoadImageFromFile',
        file_client_args=dict(backend='disk')),
    dict(type='mmdet.Resize', scale=image_size, keep_ratio=True),
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
        data_prefix=dict(img_path='test/images/'),
        img_shape=image_size,
        test_mode=True,
        pipeline=test_pipeline))

submission_work_dir = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/ss_submission')
submission_prefix = (
    f'{submission_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_ss_task1')
test_evaluator = dict(
    _delete_=True,
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    iou_thr=0.1,
    outfile_prefix=submission_prefix)
work_dir = submission_work_dir
