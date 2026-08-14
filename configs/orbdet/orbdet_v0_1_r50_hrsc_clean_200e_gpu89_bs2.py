_base_ = './h2rbox_r50_hrsc_200e_2gpu_officiallike.py'

# Clean causal comparison: preserve the complete H2RBox experiment contract
# and change only the detector wrapper and self-supervised consistency loss.
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
    'orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814')
