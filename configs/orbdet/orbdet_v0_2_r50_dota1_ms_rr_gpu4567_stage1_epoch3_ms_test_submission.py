_base_ = (
    './orbdet_v0_2_r50_dota1_ms_rr_gpu4567_stage1_epoch3_ss_test_submission.py')

data_root = '/data/zcy/dataset/test_ms/'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img_path='images/')))

submission_work_dir = (
    '/data1/zcy/Orbdet/work_dirs/eval/'
    'orbdet_v0_2_dota1_ms_rr_gpu4567_stage1_epoch3_20260816/ms_submission')
submission_prefix = (
    f'{submission_work_dir}/orbdet_v0_2_msrr_stage1_epoch3_msrr_task1')
test_evaluator = dict(outfile_prefix=submission_prefix)
work_dir = submission_work_dir
