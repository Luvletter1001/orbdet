_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567.py'

# Six-hour allocation stage.  Resume this checkpoint later with the parent
# 12E config; do not interpret this stage marker as completion of all 12E.
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=3, val_interval=999)

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_gpu4567_seed3407_stage1_3e_20260816')
