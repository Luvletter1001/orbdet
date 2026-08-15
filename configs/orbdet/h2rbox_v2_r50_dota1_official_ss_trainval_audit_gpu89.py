_base_ = './orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py'

# Current-code equivalent of the model contract embedded in the official
# H2RBox-v2 model-zoo checkpoints.  Training-only loss settings are retained so
# strict model construction is reproducible; inference has no Orbdet additions.
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

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/audit/'
    'h2rbox_v2_dota1_official_20260815/ss_trainval')
