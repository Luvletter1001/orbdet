_base_ = './h2rbox_r50_hrsc_recovery_bs1_stepaligned_60e_gpu89.py'

# Strict single-variable comparison against the recovered H2RBox contract.
# Every data, optimization, schedule, hook, and randomness setting is inherited.
model = dict(
    type='OrbdetDetector',
    bbox_head=dict(
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

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_1_hrsc_recovery_bs1_stepaligned_200e_gpu89_'
    'seed3407_20260814')
