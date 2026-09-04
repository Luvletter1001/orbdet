# Single-GPU null control for the historical two-GPU E0 recipe.
#
# E0 used two ranks with batch_size=2 per rank.  This control uses one rank
# with batch_size=4 so global batch, optimizer updates per epoch, LR milestones,
# and 500-iteration warmup exposure match.  A bounded GPU memory smoke is still
# required before any formal use.
_base_ = './orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu89.py'

intended_world_size = 1
global_batch_size = 4

train_dataloader = dict(batch_size=4)

work_dir = ('/data1/zcy/Orbdet/work_dirs/controlled/'
            'orbdet_v0_2_dota1_grouped_ss_e0_gpu4_control_seed3407')
