# GDA Plan-B P1 config: E0 baseline + parallel GDA probe head.
# Single intended difference vs E0: detector/head type + gda_probe dict.
# Training requires explicit per-round authorization (see
# resultmd/exp_low_rank_orientation_evidence/gda_plan_b_implementation.md).
_base_ = './orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu89.py'

model = dict(
    type='OrbdetGDADetector',
    bbox_head=dict(
        type='H2RBoxGDAHead',
        gda_probe=dict(
            enabled=True,
            detach_feats=False,
            loss=dict(
                type='OrbdetGDAProbeLoss',
                # a0 = log(1.15), tau = 0.10 (B1 operating band edge);
                # w_env=1.0, w_xview=0.5, w_bit_gt=0.2, w_bit_x=0.1,
                # beta=0.05, bit_min_abs_sin=0.3  (module defaults)
            ))))

work_dir = ('/data1/zcy/Orbdet/work_dirs/formal/'
            'orbdet_gda_probe_dota1_grouped_ss_e0_gpu4_seed3407')
