# Orbdet-v0.2 DOTA-v1 MS E8 与 SS seed-42 E12 评测结果

## 结论

GPU 4–7 的 MS+RR 续训完整到达 epoch 8，GPU 8/9 的 SS seed 42 完整到达
epoch 12；两个 checkpoint 均通过 epoch、iter、371 tensors、config token 与
SHA256 强验证。用户随后明确授权补评，trainval 自评和三份 DOTA Task1 在线提交
ZIP 已全部完成。

本页 mAP 是对用于训练的 raw trainval 20,995 patches 的诊断性自评，不是独立
validation，也不是 DOTA 在线 test 成绩。ZIP 只生成并校验，没有自动上传。

| run | epoch / iter | trainval mAP | AP50 | checkpoint SHA256 |
|---|---:|---:|---:|---|
| MS+RR seed 3407 | 8 / 136,656 | 0.7574 | 0.7570 | `92a598aa332fb4ef9744b9eb390b080c4cab3e7971297b9880a96c47807c6baa` |
| SS seed 42 | 12 / 38,280 | 0.7623 | 0.7620 | `8d5f6af3a18dbc0b6b9587b5d1927262390d3685d42595574eb38cb989a095bc` |

## 逐类 AP

| class | MS E8 | SS seed 42 E12 | MS − SS |
|---|---:|---:|---:|
| plane | 0.903 | 0.904 | -0.001 |
| baseball-diamond | 0.738 | 0.695 | +0.043 |
| bridge | 0.488 | 0.504 | -0.016 |
| ground-track-field | 0.568 | 0.600 | -0.032 |
| small-vehicle | 0.792 | 0.768 | +0.024 |
| large-vehicle | 0.843 | 0.845 | -0.002 |
| ship | 0.881 | 0.877 | +0.004 |
| tennis-court | 0.905 | 0.907 | -0.002 |
| basketball-court | 0.787 | 0.833 | -0.046 |
| storage-tank | 0.747 | 0.745 | +0.002 |
| soccer-ball-field | 0.655 | 0.672 | -0.017 |
| roundabout | 0.702 | 0.708 | -0.006 |
| harbor | 0.737 | 0.733 | +0.004 |
| swimming-pool | 0.795 | 0.820 | -0.025 |
| helicopter | 0.819 | 0.824 | -0.005 |

## 在线提交文件

所有 ZIP 均含根目录下 15 个唯一 `Task1_*.txt`，`zipfile.testzip()` 返回
`None`，`unzip -t` 无错误。

| model / test mode | bytes | SHA256 | path |
|---|---:|---|---|
| MS E8 / SS test | 14,912,732 | `e059a60fec42950b2737c0d6fbe8b3e04075c2c6ec66d14326bce94afee51cbc` | `work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_epoch8_20260818/ss_submission/orbdet_v0_2_msrr_epoch8_ss_task1/orbdet_v0_2_msrr_epoch8_ss_task1.zip` |
| MS E8 / MS+RR test | 29,915,872 | `d7e3c787f69c854d6fe5af19fc31083b95fd1efd3842f55c5d19dd53516d4a9f` | `work_dirs/eval/orbdet_v0_2_dota1_ms_rr_gpu4567_epoch8_20260818/ms_submission/orbdet_v0_2_msrr_epoch8_msrr_task1/orbdet_v0_2_msrr_epoch8_msrr_task1.zip` |
| SS seed 42 E12 / SS test | 13,928,054 | `d3e181678e9a28698c6c0ba52d81420779fb1c660325ac14f7bdd7b9cd643815` | `work_dirs/eval/orbdet_v0_2_dota1_ss_gpu89_seed42_epoch12_20260818/ss_submission/orbdet_v0_2_ss_seed42_epoch12_task1/orbdet_v0_2_ss_seed42_epoch12_task1.zip` |

## 训练与结果分析

- MS+RR 的 epoch median loss 从 E3 `1.1730` 降至 E8 `1.0771`，末 100 个
  记录点 median 为 `1.0505`；median grad norm 从 `3.7013` 降至 `2.5223`，
  median `q_joint` 从 `0.9656` 升至 `0.9826`。训练稳定且仍有收敛迹象。
- MS trainval mAP 从 E3 `0.6851` 提升至 E8 `0.7574`，绝对提高 `0.0723`。
  15 类中仅 ground-track-field 相比 E3 低 `0.003`，其余类别普遍提升；增幅较大
  的是 helicopter、soccer-ball-field、baseball-diamond、harbor 和 ship。
- SS seed 42 E12 为 `0.7623`，比旧 seed 3407 E12 的 `0.7770` 低 `0.0147`。
  seed 42 的 E12 median training loss `1.0493` 也高于 seed 3407 的 `1.0101`，
  二者方向一致，说明当前 SS 结果存在不可忽略的随机种子波动。
- MS E8 比 SS seed 42 E12 低 `0.0049`，但该比较不是同训练预算：MS 只到 E8，
  且尚未完成 milestones 8/11 后的低学习率阶段。因此不能据此判定 MS 无效；应完成
  MS E9–12 后再做最终比较。
- MS 相对 SS 更强的类别主要是 baseball-diamond（`+0.043`）和 small-vehicle
  （`+0.024`）；SS 更强的类别主要是 basketball-court（`+0.046`）、
  ground-track-field（`+0.032`）和 swimming-pool（`+0.025`）。

## 值守异常边界

训练值守器在 SS 正常结束的状态切换处以 exit 10 失败，因此没有自动启动后评测；
两项独立训练启动器仍正常完成并写出强验证 checkpoint。本次后评测是在用户重新
明确授权后执行。异常不影响 checkpoint 或上述指标，但说明 live monitor 的
“session 正常消失 → formal terminal contract”分支仍需单独修复和回归。
