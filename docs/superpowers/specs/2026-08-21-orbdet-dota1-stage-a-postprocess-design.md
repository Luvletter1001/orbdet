# Orbdet DOTA-v1 Stage A：无训练预测缓存与后处理诊断设计

## 状态与授权边界

本设计已获得用户对 Stage A 的口头批准，当前等待用户审阅本书面规格。只有书面
规格再次确认后才进入实施计划和代码修改。

Stage A 只覆盖：预测缓存、提交 score 精度、baseball-diamond 推理策略消融、
跨 patch merge/NMS/fusion 工具和诊断报告。明确禁止：

- 启动或续训任何模型；
- 修改 checkpoint；
- 改写 `data/DOTA-v1.0/` 或 `/data/zcy/dataset/` 源数据；
- 上传 DOTA 在线系统；
- 用在线 test 分数选择阈值；
- 覆盖现有 E12 提交文件或评测目录。

## 目标

在冻结 E12 checkpoint 的前提下，回答四个可证伪问题：

1. 两位小数 score 是否因大量并列排序而损失 AP；
2. baseball-diamond 的 `angle=0` 和 `resize=0.85` 各自对 AP50/AP75 有何影响；
3. 跨 patch hard NMS 是否丢失可用于提高高 IoU 定位的多尺度几何共识；
4. 能否把一次推理的原始 patch 结果安全缓存，使后续 merge 实验无需重复模型前向。

Stage A 的成功定义是建立可复现工具、真实诊断数值和候选排序，不把 full-trainval
诊断包装成独立泛化提升，也不承诺在线 AP 增益。

## 已知证据

- 在线 E12 MS+RR：`AP50=0.7730023495`、`AP75=0.4780261817`。
- raw trainval patch：`AP50=0.8010`、`AP75=0.4960`。
- 当前 serializer 使用 `round(score, 2)`；各 Task1 文件只有 69–87 个不同分数值。
- baseball-diamond 本地 `AP50=0.821`、`AP75=0.135`。
- 推理对 baseball-diamond 强制 `angle=0`，并把宽高同时乘以 `0.85`；理想包含
  情况下 `0.85²=0.7225 < 0.75`。
- 当前 E12 使用完整 trainval 训练，任何 trainval 消融只能作为机制诊断。

## 总体数据流

```text
frozen checkpoint + fixed config
    -> patch inference
    -> versioned raw cache (float32 scores/boxes + metadata)
    -> deterministic offline postprocessor
       -> score precision variant
       -> hard-NMS threshold variant
       -> rotated fusion variant
    -> labelled diagnostic metrics or unlabelled Task1 ZIP
    -> integrity manifest and Chinese result report
```

baseball-diamond 的 angle/resize 策略发生在 patch NMS 之前，不能从已经强制归零的
现有 Task1 文件恢复。因此四个策略使用独立、冻结配置运行；每次结果也写入各自缓存。
跨 patch merge 和 serializer 实验则全部复用对应缓存。

## 组件设计

### 1. 向后兼容的 score precision

为 `DOTAMetric` 增加显式 `score_decimals` 参数：

- 默认值为 `2`，保持现有输出行为；
- Stage A 候选使用 `6`；
- 只接受 `[1, 10]` 内整数，非法值 fail-fast；
- NMS、fusion 和缓存始终使用未量化 float32 score；
- 量化只发生在最终 `Task1_*.txt` 序列化边界；
- 坐标继续保留两位小数，本阶段不改变几何格式。

相同 score 下使用确定性次级键排序：`original_image_id`、`patch_id`、原始行号。
该规则用于可复现，不用人工 jitter 伪造不存在的置信度差异。

### 2. 版本化 raw cache

缓存保存 evaluator 实际收到的 patch 结果，不保存模型或任意可执行对象。目录包含：

| file | content |
|---|---|
| `manifest.json` | schema、checkpoint/config SHA256、dataset、classes、box type、patch count、创建命令 |
| `predictions.npz` 或分片 `predictions_*.npz` | `img_id`、float32 boxes/scores、int64 labels、确定性 row id |
| `annotations.npz` | 仅 labelled 诊断集保存 GT/ignore；test cache 不生成 |
| `SHA256SUMS` | manifest 与所有数据分片哈希 |

实现优先选择无代码执行语义的 NumPy 分片，不使用不受信任加载风险较高的 pickle。
分片大小以单文件可校验、内存峰值可控为准；schema 中固定 offset/length 索引。

缓存写入要求：

- 输出目录必须不存在；
- 所有数组必须 finite、shape 合法、类别索引在 `[0, 14]`；
- `img_id` 唯一 patch 数必须与 dataloader 合同一致；
- 写完后重新读取 manifest、校验 SHA256 和总记录数，才写 `COMPLETE`；
- 中断只留下 `INCOMPLETE`，离线 merge 拒绝读取。

### 3. baseball-diamond 推理策略解耦

训练用 `rotation_agnostic_classes` 保持 checkpoint 合同不变。新增独立 test-only
配置，不改变 state dict：

- `test_rotation_agnostic_classes`：未设置时回退到训练列表；
- `test_agnostic_resize_classes`：未设置时回退到现有列表；
- `test_agnostic_resize_factor`：默认 `0.85`，候选允许 `[0.85, 0.90, 0.95, 1.00]`。

首轮只运行四个能分离原因的预注册策略：

| variant | baseball angle | baseball resize |
|---|---|---:|
| `B0_current` | force 0 | 0.85 |
| `B1_pred_angle` | predicted | 0.85 |
| `B2_no_shrink` | force 0 | 1.00 |
| `B3_pred_angle_no_shrink` | predicted | 1.00 |

只有四个主策略完成后，才允许把 `0.90/0.95` 作为尺寸敏感性报告；它们不能依据
online test 选择。storage-tank 和 roundabout 本轮保持当前策略，避免一次改变多个类。

### 4. 离线 merge 与 rotated fusion

新增独立分析工具读取 COMPLETE cache，按 original image 和 class 聚合。第一阶段
支持两种互斥方法：

1. `hard_nms`：复现当前 rotated NMS，并固定扫描 `0.05/0.10/0.15/0.20`；
2. `rotated_wbf`：按 rotated IoU 聚类，使用 canonical `le90` 框和
   `exp(i*2*theta)` 做轴方向平均。

WBF 首轮固定 `fusion_iou_thr=0.30`，采用按 score 降序的确定性 greedy
clustering，不在 Stage A 内搜索该阈值。几何权重由 full-precision score 与可选
patch-edge 权重组成。首轮 fused score 使用 cluster 内最大原始 score，避免同时
改变几何和置信度标度；consensus re-ranking 作为后续单独 variant，不能与首轮
WBF 混合解释。

patch-edge 权重默认关闭。开启时只根据检测中心到最近 patch 边界的距离 `d` 和
文件名记录的 patch window side `s` 计算，不读取 GT：

`edge_weight = 0.25 + 0.75 * clip(d / (0.10 * s), 0, 1)`。

同一 cluster 中来自同一 patch 的多框只保留最高 score 框参与融合权重，避免单个
patch 重复计票。空类、单框 cluster、角度跨 `-pi/2` 边界和近方形框均有确定性
行为。

### 5. 指标与结果解释

labelled cache 输出：

- overall `AP50`、`AP75` 和两阈值平均；
- 15 类 AP50/AP75；
- baseball-diamond 的 recall 与 AP；
- score tie 数、unique score 数和最大 tie；
- merge 前后框数、cluster size、角度/中心/尺寸方差。

test cache 只生成结构完整的候选 ZIP 和无 GT 统计，禁止写“提升”。当前 full-trainval
已经参与训练，且仓库内没有可信的原图级 merged GT，因此：score 量化和 baseball
策略可报告 patch-level `trainval_diagnostic` AP；跨 patch NMS/WBF 只报告预测数量、
cluster 和几何一致性，不报告 AP。上述数值不能用于论文泛化主张或最终阈值冻结。
独立 DOTA train-to-val 合同属于后续阶段，不在 Stage A 内训练。

## 实验矩阵

| exp_id | unique change | output | interpretation |
|---|---|---|---|
| A0 | current policy, 2 decimals, NMS 0.10 | reproduction | 验证工具不改变基线 |
| A1 | 6 decimals only | AP/tie delta | 检验 score 量化 |
| A2–A4 | B1/B2/B3 policy | per-class AP50/AP75 | 分离 angle 与 shrink 影响 |
| A5–A7 | hard NMS 0.05/0.15/0.20 | merge statistics | NMS sensitivity |
| A8 | WBF, edge off | geometry diagnostic | 检验多尺度几何共识 |
| A9 | WBF, edge on | geometry diagnostic | 检验 patch 边界影响 |

每个实验只改变表中一项。A0 未复现前，后续结果不进入比较表。

## 测试设计

### 单元测试

- 默认 `score_decimals=2` 与现有 Task1 文本内容一致；
- 六位格式保留排序且不产生 scientific notation；
- cache 分片 round-trip 数值逐元素一致；
- manifest/hash/shape/class/finite 任一不符均拒绝 merge；
- test-only angle/resize 配置不改变 checkpoint keys；
- 四个 baseball variant 在合成框上产生预期角度和尺寸；
- `0.85 × 0.85` 合成框验证 IoU 上限示例；
- `89°/-89°` 的 WBF 使用双角平均后保持接近同一轴；
- 两个邻近真实目标不能因过宽 cluster 条件被错误融合；
- hard NMS `0.10` 与当前实现逐元素一致；
- ZIP 恰好包含 15 个唯一 `Task1_*.txt`，CRC 通过。

### 集成验证

1. 小型 synthetic cache 不使用 GPU，覆盖全部 variant；
2. 有界小样本推理验证 cache 与直接 evaluator 输出一致；
3. E12 raw trainval 完整诊断只在 GPU 空闲且前台硬超时下运行；
4. MS test 候选生成前重新核验 checkpoint SHA256；
5. 最终确认无残留 GPU 进程、无训练 checkpoint、无网络上传。

## 输出与不可覆盖规则

所有新产物写入：

`work_dirs/diagnostics/orbdet_v0_2_dota1_stage_a_postprocess_20260821/`

每个 `exp_id` 使用独立子目录。已有 E12 ZIP、训练目录和在线结果报告只读。候选 ZIP
文件名必须包含 variant，不得复用
`orbdet_v0_2_msrr_epoch12_msrr_task1.zip`。

最终结果追加到现有 per-experiment `fres_*` 记录，包含真实数字、命令、SHA256、
失败项和“不上传”声明；未知结果不填写预测值。

## 失败与停止条件

- A0 无法复现当前文本内容或 AP：停止，先修 cache/merge 等价性；
- 任一 variant 出现非有限框、类别丢失、patch 数不符或 ZIP 结构错误：该 variant
  失败，不生成可提交候选；
- WBF 在 synthetic 邻近目标测试中发生错误合并：停止 WBF，不通过调低单个样本
  分数掩盖；
- trainval 只显示收益但缺少独立 validation：只记录 hypothesis，不晋级为方法；
- 任何步骤需要训练或在线上传：停止并请求新授权。

## 交付物

Stage A 完成时应具有：

1. 向后兼容的 score precision 配置与测试；
2. 可审计、可复用的 NumPy raw cache；
3. 四个 baseball test-only 策略及逐类诊断；
4. hard NMS sweep 与周期正确的 rotated WBF 工具；
5. 原始基线复现证据、候选 ZIP 完整性证据和中文实验报告；
6. 明确声明没有训练、没有上传、没有改写源数据。
