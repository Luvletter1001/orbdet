# 目录与产物清单

## 项目与环境

```text
project_root: /data1/zcy/Orbdet
python: /data/zcy/anaconda3/envs/orbdet/bin/python
physical_gpus: 8,9
deadline: 2026-08-14 08:30:00 +08:00
deadline_epoch: 1786667400
```

## 当前 H2RBox baseline（已存在、只读配置）

```text
config:
/data1/zcy/Orbdet/configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py

launcher:
/data1/zcy/Orbdet/scripts/formal/run_h2rbox_r50_hrsc_200e_gpu89_bs2.sh

tmux:
h2rbox_hrsc200e_bs2_gpu89_20260814

work_dir:
/data1/zcy/Orbdet/work_dirs/formal/h2rbox_r50_hrsc_train_val_200e_gpu89_bs2_seed3407_20260814

launch_log:
/data1/zcy/Orbdet/work_dirs/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814.launch.log

one_shot_test_dir:
/data1/zcy/Orbdet/work_dirs/test/h2rbox_hrsc_clean_bestval_gpu89_20260814
```

## clean Orbdet candidate（已完成）

```text
formal_config:
/data1/zcy/Orbdet/configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2.py

smoke_config:
/data1/zcy/Orbdet/configs/orbdet/orbdet_v0_1_r50_hrsc_clean_200e_gpu89_bs2_smoke.py

smoke_launcher:
/data1/zcy/Orbdet/scripts/smoke/run_orbdet_v0_1_hrsc_clean_gpu89_bs2_smoke.sh

formal_launcher:
/data1/zcy/Orbdet/scripts/formal/run_orbdet_v0_1_hrsc_clean_200e_gpu89_bs2.sh

deadline_guard:
/data1/zcy/Orbdet/scripts/formal/guard_orbdet_gpu89_until_0830_20260814.sh

smoke_work_dir:
/data1/zcy/Orbdet/work_dirs/smoke/orbdet_v0_1_hrsc_clean_gpu89_bs2_20260814

formal_work_dir:
/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_seed3407_20260814

formal_launch_log:
/data1/zcy/Orbdet/work_dirs/orbdet_v0_1_hrsc_clean_200e_gpu89_bs2_20260814.launch.log

one_shot_test_dir:
/data1/zcy/Orbdet/work_dirs/test/orbdet_v0_1_hrsc_clean_bestval_gpu89_20260814

result_record:
/data1/zcy/Orbdet/resultmd/exp_orbdet_hrsc_clean/fres_orbdet_hrsc_clean_gpu89_20260814.md
```

## 规格与实现计划

```text
/data1/zcy/Orbdet/docs/superpowers/specs/2026-08-14-orbdet-clean-comparison-until-0830-design.md
/data1/zcy/Orbdet/docs/superpowers/plans/2026-08-14-orbdet-clean-comparison-until-0830.md
```

## 禁止覆盖的历史产物

```text
/data1/zcy/Orbdet/work_dirs/formal/orbdet_v0_1_hrsc_gpu89
/data1/zcy/Orbdet/work_dirs/formal/h2rbox_r50_hrsc_trainval_200e_gpu0123_bs8_seed3407
/data1/zcy/Orbdet/resultmd/orbdet_v0_1_hrsc_gpu89_20260812.md
/data1/zcy/Orbdet/resultmd/exp_h2rbox_hrsc_baseline/fres_h2rbox_hrsc_bs8_200e_stopped_e186_seed3407.md
```
