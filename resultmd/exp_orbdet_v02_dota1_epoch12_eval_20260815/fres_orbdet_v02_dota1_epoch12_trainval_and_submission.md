# Orbdet-v0.2 DOTA-v1 Epoch 12 自评与在线提交包

## 结论

本次使用的是 **DOTA-v1.0**、15 类、1024×1024 patch。Orbdet-v0.2
epoch-12 checkpoint 已完成 raw all-patch trainval 自评，并对官方无标签 test
patch 完成推理、原图坐标回映、逐类旋转 NMS 合并和 Task1 ZIP 打包。

- trainval self-eval：`dota/mAP=0.7769502401`，`dota/AP50=0.7770`。
- online submission ZIP：
  `work_dirs/eval/orbdet_v0_2_dota1_epoch12_test_submission_gpu89_20260815/dota_v1_task1_epoch12/dota_v1_task1_epoch12.zip`。
- ZIP SHA256：
  `edd0cd1065b70a20ffbe48079d42f39f2fb6077763ed7bc01087ba8f39e46e16`。
- 在线分数尚未产生；只有上传官方评测服务器后才能报告 test AP。

## 模型与评测合同

| field | value |
|---|---|
| dataset | DOTA-v1.0 |
| classes | 15 |
| checkpoint | `work_dirs/formal/orbdet_v0_2_dota1_1x_gpu89_seed3407_20260815/epoch_12.pth` |
| checkpoint_sha256 | `64321e7a8619b554951ea3d4eda17e65d850c1a5c42733842dcb6101e23c630e` |
| model | Orbdet-v0.2, R50-FPN, H2RBoxV2Head + PSC |
| training_supervision | `qbox -> hbox -> rbox`;真实方向未参与训练 |
| input | 1024×1024 |
| physical_gpus | 8,9 |
| distributed | 2 ranks |
| nccl_safety | `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` |
| eval_commit | `2fe1db2` |
| validation_fix_commit | `ffc7c80` |

## Trainval self-eval 口径

评测使用 raw all-patch trainval 共 `20,995` 张切片，包含 `8,238` 张空标注
切片；训练 sampler 当时只使用 `filter_empty_gt=True` 后的 `12,757` 张非空
切片。因此这里不是 filtered-12757 口径。

模型已使用整个 trainval 训练，故该结果是训练集自评，只能用于检查拟合程度、
方向恢复和推理链路，不能作为独立验证集泛化成绩，也不能与论文 val/test AP
直接比较。

评测参数为 `IoU=0.5`、DOTA-v1/VOC07 `11points` AP：

| class | gts | dets | recall | AP50 |
|---|---:|---:|---:|---:|
| plane | 18,788 | 66,923 | 0.966 | 0.904 |
| baseball-diamond | 1,087 | 29,199 | 0.896 | 0.691 |
| bridge | 4,183 | 114,219 | 0.770 | 0.533 |
| ground-track-field | 733 | 27,943 | 0.783 | 0.594 |
| small-vehicle | 58,854 | 321,515 | 0.907 | 0.807 |
| large-vehicle | 43,071 | 309,003 | 0.938 | 0.861 |
| ship | 76,153 | 193,099 | 0.924 | 0.882 |
| tennis-court | 5,923 | 27,858 | 0.979 | 0.907 |
| basketball-court | 1,180 | 15,103 | 0.956 | 0.863 |
| storage-tank | 13,670 | 86,949 | 0.830 | 0.768 |
| soccer-ball-field | 827 | 24,060 | 0.894 | 0.684 |
| roundabout | 971 | 25,861 | 0.907 | 0.728 |
| harbor | 15,468 | 96,285 | 0.839 | 0.747 |
| swimming-pool | 3,731 | 22,918 | 0.932 | 0.826 |
| helicopter | 1,189 | 25,641 | 0.950 | 0.858 |
| **mAP** | — | — | — | **0.777** |

权威机器记录：
`work_dirs/eval/orbdet_v0_2_dota1_epoch12_trainval_raw20995_gpu89_20260815/20260815_131201/20260815_131201.json`。

## 官方 test submission

官方 test 口径为 `10,833` 张无标签 patch。`DOTAMetric` 使用
`format_only=True`、`merge_patches=True`、merge NMS IoU `0.1`，把 patch
预测映射回原始大图后输出 DOTA-v1 Task1 格式。

ZIP 自检结果：

- 压缩包大小约 `11.9 MiB`；解压后总计 `32,171,128` bytes。
- 恰好包含 15 个根目录文件，名称均为 `Task1_<class>.txt`。
- `unzip -t` 对 15 个文件全部返回 `OK`，无压缩数据错误。
- 控制器已写出 `TRAINVAL_COMPLETE`、`SUBMISSION_COMPLETE` 和 `COMPLETE`，
  未写出 `FAILED`。
- 控制器日志未匹配到 `Traceback`、`RuntimeError`、NCCL error 或 `NaN`。

## 可复现入口

- trainval config：
  `configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_trainval_eval_gpu89.py`
- submission config：
  `configs/orbdet/orbdet_v0_2_r50_dota1_epoch12_test_submission_gpu89.py`
- 串行 launcher：
  `scripts/eval/run_orbdet_v0_2_dota1_epoch12_trainval_and_submission_gpu89.sh`
- controller log：
  `work_dirs/eval/orbdet_v0_2_dota1_epoch12_gpu89_20260815/controller.log`

## 验证

- 合同 RED：配置与 launcher 不存在时，新增 5 项测试按预期失败。
- 合同 GREEN：两个 dataset 分别构建为 `20,995` 与 `10,833`。
- 两条 4-sample GPU runtime smoke 均成功；smoke ZIP 含 15 个 Task1 文件。
- 全量启动前项目回归：`73 passed`。
- 发现 `rtk test -s` 与包装器不兼容后，新增失败断言并改用
  Bash `[[ -s ... ]]`；focused tests 恢复为 `5 passed`。

## 上传说明

可直接把上面的 ZIP 上传至 DOTA-v1 oriented detection（Task 1）在线评测。
不要把 ZIP 内的 txt 再套一层目录或重新改名。在线返回的 AP/mAP 应另行记录，
并与本页 trainval self-eval 明确区分。
