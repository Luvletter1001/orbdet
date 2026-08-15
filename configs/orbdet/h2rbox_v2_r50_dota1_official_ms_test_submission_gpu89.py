_base_ = './h2rbox_v2_r50_dota1_official_ss_test_submission_gpu89.py'

data_root = '/data/zcy/dataset/test_ms/'

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img_path='images/')))

submission_work_dir = (
    '/data1/zcy/Orbdet/work_dirs/audit/'
    'h2rbox_v2_dota1_official_20260815/ms_submission')
submission_prefix = (
    f'{submission_work_dir}/h2rbox_v2_official_msrr_task1')
test_evaluator = dict(outfile_prefix=submission_prefix)
work_dir = submission_work_dir
