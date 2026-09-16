# Wholly-WOOD 宿主真实接口审计（C0）

审计日期：2026-09-16
仓库：`/data1/zcy/Orbdet/work_dirs/derived/whollywood_official_20260916`
commit：`4a77e648ee2911bbd9eaeb941f4fe7916d1d4e72`（2025-02-14）
克隆方式：`git clone --depth 1 https://github.com/VisionXLab/whollywood.git`，dirty 状态：干净（无本地修改）

本文件只记录**从源码和配置核读到的事实**，不推断未核读的行为。

---

## 1. 官方配置入口（已核读）

| 协议 | 配置文件 | 顶层模型 |
|---|---|---|
| HBB | `configs/whollywood/whollywood-le90_r50_fpn-1x_dota.py` | `H2RBoxV2PDetector` |
| Point（阶段 s1，P2R 伪标签） | `...-1x_dota-s1.py` | `Point2RBoxHDR` |
| Point（阶段 s2，检测器） | `...-1x_dota-s2.py` | `H2RBoxV2PDetector` |
| 官方混合 | `...-1x_dota_mix-s1.py` / `-s2.py` | 同 s1/s2 |

**Point 协议确认为 s1 → s2 两阶段**，不是单阶段端到端。README 另指向更新的 Point2RBox-v2 仓库；首版锁定本仓库原版，不无记录切换。

## 2. 模型与 head 的真实类

| 项 | HBB / s2 | Point s1 |
|---|---|---|
| detector 源文件 | `mmrotate/models/detectors/h2rbox_v2p.py` | `mmrotate/models/detectors/point2rbox_hdr.py` |
| head 类 | `H2RBoxV2PHead` | `Point2RBoxHDRHead` |
| head 源文件 | `mmrotate/models/dense_heads/h2rbox_v2p_head.py` | `mmrotate/models/dense_heads/point2rbox_hdr_head.py` |
| 角度编码 | `PSCCoder` | `PSCCoder` |
| 框编码 | `DistanceAnglePointCoder`（`angle_version='le90'`） | 同左 |
| strides | `[8,16,32,64,128]` | `[8]`（`regress_ranges=[(-1, 1e8)]`） |
| FPN | `out_indices=(1,2,3)`、`start_level=0` | `out_indices=(0,1,2,3)`、`start_level=1` |

## 3. 数据流水线（已核读，HBB 与 Point 共用前半段）

```text
LoadImageFromFile
LoadAnnotations(with_bbox=True, box_type='qbox')     # 原标注是四点 QBB
ConvertBoxType(gt_bboxes -> 'hbox')                  # 转水平框
ConvertBoxType(gt_bboxes -> 'rbox')                  # 再表示为 (x,y,w,h,theta)
Resize(scale=(1024,1024), keep_ratio=True)
RandomFlip(prob=0.75, direction=['horizontal','vertical','diagonal'])
[RBox2Point(dummy=48, partial=...)]                  # 仅 Point 协议
[RandomShift(prob=0.5, max_shift_px=16)]             # 仅 Point 协议
PackDetInputs
```

即：**HBB 协议的弱标签是 QBB 转成的水平框，再以 `(x,y,w,h,0)` 表示。**

## 4. RBox2Point 的真实语义（关键，且与直觉相反）

源文件：`mmrotate/datasets/transforms/transforms.py:47`

```python
def transform(self, results):
    max_idx = int(round(results['gt_bboxes'].tensor.shape[0] * self.partial))
    results['gt_bboxes'].tensor[:max_idx, 2] = self.dummy   # w
    results['gt_bboxes'].tensor[:max_idx, 3] = self.dummy   # h
    results['gt_bboxes'].tensor[:max_idx, 4] = 0            # angle
```

核读结论：

1. **`partial` 是"转点的对象比例"，且按标注行顺序取前 `round(n*partial)` 个**，不是随机抽样，也不涉及哈希。
2. 官方取值：`partial=1`（纯 Point s1）、`partial=0.7`（官方 mix s1）。**不存在 `partial=.5` 的官方配置**；把 mix 当成固定 50% 是错的，实际是 70%。
3. **转换并不删除强几何，而是把 w/h 覆写为 `dummy=48`、angle 覆写为 0**。所以"点对象"在张量里仍是一个 48×48、角度 0 的框，只是中心点是真的。
   → `dummy=48` 是官方机制占位，绝不能当真实尺度监督；本项目的 Point 载荷必须只保留中心点。
4. 该变换在 `Resize` 之后、`RandomFlip` 之前，且**每 epoch 结果相同**（无随机），满足"分配一次、不按 epoch 改标签强弱"的要求。但它依赖标注行顺序，与阶段一 `assign_kinds` 的哈希序是**两套不同机制**，跨臂比较时必须写明用的是哪一套。

## 5. 对象身份：`bid_targets` 确实存在（已核读）

- `H2RBoxV2PHead.loss_by_feat`：`labels, bbox_targets, angle_targets, bid_targets = self.get_targets(...)`（源文件第 213 行）
- `Point2RBoxHDRHead`：同样返回 `bid_targets`（第 257 行）

两个臂都通过 `torch.unique(pos_bid_targets, return_inverse=True)` 按 bid 聚合同一对象的多点预测（第 322–343 行）。

**缺口**：`bid` 是**每次 `get_targets` 现场生成、按图内标注顺序编号**的运行时 ID，不是跨运行持久 ID。视图变换、P2R 伪框、teacher cache、KD 条目之间无法自动对齐。
→ 按阶段一计划要求，**必须实现显式 ID 保序**（`image_key#row`），不得用同类最近中心补配。`geometry_aux.protocols.annotations.stable_object_id` 已提供该 ID。

## 6. P2R 合成图案机制（仅 s1）

`Point2RBoxHDR` 构建期加载 `basic_pattern='data/basic_patterns/dota'`（配置中 s1 指定 `use_setsk=True`、`use_setrc=False`，`dense_cls=[4,5,6,9]`、`square_cls=[1,9,11]`），前向中把合成图案叠加到图像上（源文件第 165 行）。
s1 产出的伪框是 s2 的输入；若普通宿主已使用该伪框做框回归，再把它当"新教师信息"重复计入就是重复计数。

## 7. 推理入口

标准 MMRotate `predict` 路径，`nms=dict(type='nms_rotated', iou_threshold=0.1)`，输出统一为 OBB。旁路在推理期必须整体关闭，不读任何 teacher 缓存。

## 8. 本轮未做 / 阻塞项

- **未运行**任何 Wholly-WOOD 训练或 s1 伪标签生成，仅完成源码与配置核读。
- 未核验 `data/basic_patterns/dota` 在本机是否已存在（依赖官方数据准备脚本，本轮未执行）。
- 未安装/改用官方环境的 Python 依赖；本仓库的 `mmrotate` 是自带的，不能覆盖 Orbdet 现有 runtime 的 mmrotate 包。
- `LICENSE` 与数据准备脚本的存在性将在下一步核读，未在本文件中断言。
