_base_ = './orbdet_v0_2_r50_dota1_ms_rr_1x_gpu89.py'

# Four ranks with one sample/rank give global batch 4.  Linearly scale the
# official global-batch-2 learning rate (5e-5) while preserving AdamW decay.
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, weight_decay=0.005))

# Save every epoch so a bounded GPU allocation always leaves a resumable
# checkpoint.  The full contract remains 12 epochs.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=4,
        save_last=True))

work_dir = (
    '/data1/zcy/Orbdet/work_dirs/formal/'
    'orbdet_v0_2_r50_dota1_ms_rr_1x_gpu4567_seed3407_20260816')
