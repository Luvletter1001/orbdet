_base_ = './orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py'

model = dict(
    type='H2RBoxV2Detector',
    bbox_head=dict(
        rotation_agnostic_classes=[1, 9, 11],
        agnostic_resize_classes=[1],
        use_circumiou_loss=True,
        use_standalone_angle=True,
        use_reweighted_loss_bbox=True,
        loss_symmetry_ss=dict(
            _delete_=True,
            type='H2RBoxV2ConsistencyLoss',
            use_snap_loss=True,
            loss_rot=dict(
                type='mmdet.SmoothL1Loss', loss_weight=1.0, beta=0.1),
            loss_flp=dict(
                type='mmdet.SmoothL1Loss', loss_weight=0.05, beta=0.1))))

submission_work_dir = (
    '/data1/zcy/Orbdet/work_dirs/audit/'
    'h2rbox_v2_dota1_official_20260815/ss_submission')
submission_prefix = (
    f'{submission_work_dir}/h2rbox_v2_official_ss_task1')
test_evaluator = dict(outfile_prefix=submission_prefix)
work_dir = submission_work_dir
