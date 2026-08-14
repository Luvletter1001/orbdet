_base_ = './orbdet_v0_2_r50_hrsc_clean_gpu89.py'

model = dict(
    type='OrbdetGODCDetector',
    godc_auxiliary=dict(
        type='HBoxFPNGroupOrbitLoss',
        group='c2',
        loss_weight=0.02,
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        orbit_loss=dict(
            type='GroupOrbitDeterminantalClusterLoss',
            determinantal_weight=1.0,
            spectral_tail_weight=0.0,
            fixed_space_weight=1.0,
            energy_guard_weight=1.0,
            variance_guard_weight=1.0)))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_godc_c2_hrsc_clean_gpu89_seed3407_20260815')
