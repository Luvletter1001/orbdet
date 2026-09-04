# GDA Plan-B controlled candidate: single-GPU null control + GDA probe.
# The only differences from the matching GPU4 control are the detector/head
# type, gda_probe dict, and work directory.  Formal use requires a separate
# bounded memory smoke and explicit authorization.
# Training requires explicit per-round authorization (see
# resultmd/exp_low_rank_orientation_evidence/gda_plan_b_implementation.md).
_base_ = './orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu4_control.py'

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
