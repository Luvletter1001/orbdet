# Orbdet HRSC 干净对照执行入口

这是本次 GPU 8/9 限时实验的本地交接目录。后续执行者先读以下三个文件：

1. [GOAL.md](GOAL.md)：实验目标、唯一变量和成功/停止条件。
2. [PLAN.md](PLAN.md)：严格执行顺序、验证门和 08:30 截止策略。
3. [WORKDIRS.md](WORKDIRS.md)：项目根目录、配置、日志、checkpoint 和结果目录。

详细设计与实现计划仍保存在：

- `docs/superpowers/specs/2026-08-14-orbdet-clean-comparison-until-0830-design.md`
- `docs/superpowers/plans/2026-08-14-orbdet-clean-comparison-until-0830.md`

## 当前交接状态

- 纯 H2RBox 200E baseline 已完成，不能重复启动；best-val epoch 80 已对
  held-out test 唯一评测一次。
- config：`configs/orbdet/h2rbox_r50_hrsc_200e_2gpu_officiallike.py`
- work dir：
  `work_dirs/formal/h2rbox_r50_hrsc_train_val_200e_gpu89_bs2_seed3407_20260814/`
- launch log：
  `work_dirs/h2rbox_r50_hrsc_200e_gpu89_bs2_20260814.launch.log`
- clean Orbdet candidate 合同测试、双卡两步冒泡、正式 200E 和唯一一次
  held-out test 均已完成。
- 最终状态：`complete`。H2RBox / clean Orbdet best-val mAP 分别为
  0.1104 / 0.1020，held-out test mAP 分别为 0.0842 / 0.0803。
- GPU 8/9 已释放；实验提前完成后 deadline guard 已关闭。
- 最终记录：
  `resultmd/exp_orbdet_hrsc_clean/fres_orbdet_hrsc_clean_gpu89_20260814.md`。

进入项目：

```bash
cd /data1/zcy/Orbdet
```

所有非交互 shell 命令必须以 `rtk` 开头；所有 Python 命令使用：

```text
/data/zcy/anaconda3/envs/orbdet/bin/python
```
