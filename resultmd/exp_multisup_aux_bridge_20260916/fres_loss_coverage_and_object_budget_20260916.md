# 第一步+第二步执行结果：损失覆盖核对表 + 对象预算对齐

日期：2026-09-16。本轮**只读**：核读 host 源码。未训练、未改码、未动运行中实验。

范围：`derived/whollywood_official_20260916`（commit `4a77e64`）的 s1/s2 两个 head + 两个 detector。
**本报告修正并取代** `fplan_direction_bypass_vs_teacher_20260916.md` 第 3 节的"发现 2"。

---

## 0. 先更正上一轮的一个错误结论

上一轮我写：*"`bmsk = bcnt[bidx] == 2` 是静默对象过滤……183343 个对象里有多少被丢掉从未算过"*，暗示可能是 bug 且丢弃量可观。

**这个判断是错的，现予撤回。** 本轮读到 detector 后事实是：

`mmrotate/models/detectors/h2rbox_v2p.py:133-192`（逐字核读）：

```python
# Crop original images and gts
batch_inputs, batch_gt_instances = self.rotate_crop(batch_inputs, 0, self.crop_size, ...)
offset = 1
for gt_instances in batch_gt_instances:
    gt_instances.bid = torch.arange(0, len(gt_instances.bboxes), 1, ...) + offset + 0.2   # ori
    offset += len(gt_instances.bboxes)
# rotated
gt_instances.bid = torch.arange(0, len(...), 1, ...) + offset + 0.4                       # rot
# flipped
gt_instances.bid = torch.arange(0, len(...), 1, ...) + offset + 0.6                       # flp

if torch.rand(1) < 0.95:
    batch_inputs_all = torch.cat((batch_inputs, batch_inputs_rot))     # ori + rot  = 2 视图
    batch_data_samples_all = [...]  # batch_gt_instances + batch_gt_rot
else:
    batch_inputs_all = torch.cat((batch_inputs, batch_inputs_flp))     # ori + flp  = 2 视图
    batch_data_samples_all = [...]  # batch_gt_instances + batch_gt_flp
```

关键：**每一步只构造 2 个视图**（95% 是 ori+rot，5% 是 ori+flp）。`bid` 的小数部分编码视图类型（+0.2 / +0.4 / +0.6），而 `compacted_bid_targets.long()` 截断后三者坍缩为同一整数。

→ **`bcnt` = 该对象在几个视图里获得了正样本点**，按构造**恒为 2**。故 `bmsk = bcnt == 2` 的含义是"该对象在两个视图中都有正样本点"，**正是注释所写的 "eliminate bboxes without correspondence" 的本来意图，不是 bug，也不是大规模静默过滤。**

**并且找到了这条注释的对照证据**——s1 同源代码里写的是正确的：

| 文件 | 注释 | 代码 |
| --- | --- | --- |
| `point2rbox_hdr_head.py:327` | `(bcnt is supposed to be 2, for original and transformed)` | `bmsk = bcnt[bidx] == 2` |
| `h2rbox_v2p_head.py:327` | `(bcnt is supposed to be 3, for ori, rot, and flp)` | `bmsk = bcnt[bidx] == 2` |

两处代码相同、注释互相矛盾；s1 的注释与代码一致，**s2 的注释是陈旧/复制粘贴缺陷**（该代码路径永远只建 2 个视图，不存在 ori+rot+flp 三视图的情形）。这是一个真实的文档缺陷，但不影响数值行为。

---

## 第一步：损失覆盖核对表

粒度口径：**逐点**=按正样本点（`pos_inds`）归约；**逐对象**=先按 `bid` 分组再做 `index_reduce_('mean')`。

### s1 `Point2RBoxHDRHead`（`loss_by_feat`，951 行文件）

| 损失 | 约束维度 | 归约粒度 | 是否被 mask 过滤 | 本轮配置取值 |
| --- | --- | --- | --- | --- |
| `loss_cls` | 类别 | 逐点（`avg_factor=num_pos`） | 否 | FocalLoss, w=1.0 |
| `loss_bbox` | 中心+尺寸 | 逐点（`avg_factor=centerness_denorm`） | 否 | IoULoss, w=1.0 |
| `loss_centerness` | 中心质量 | 逐点（`avg_factor=num_pos`） | 否 | CrossEntropy, **w=0.0** |
| `loss_ss_bbox` | **跨视图 尺寸/中心一致性** | **逐对象**（`unique(bid)` + `index_reduce_('mean')`） | 是（`bmsk=bcnt==2`） | 见 `ss_info[0]=='sca'` 分支 |
| `loss_ss_symmetry` | **跨视图 角度一致性** | **逐对象**（同上） | 是（`bmsk`；另有 `square_mask` 对 `[1,9,11]` 置零） | 有 |
| `loss_angle` | 角度（直接监督） | 逐点 | — | **`None`（未启用）** |

注：`loss_bbox` 末尾有 `+ 0 * pos_angle_preds.sum()`，是把角度留在计算图里的技巧项，无梯度贡献。

### s2 `H2RBoxV2PHead`（本次实际要挂旁路的 host）

| 损失 | 约束维度 | 归约粒度 | 是否被 mask 过滤 | 本轮配置取值 |
| --- | --- | --- | --- | --- |
| `loss_cls` | 类别 | 逐点（`avg_factor=num_pos`） | 否 | FocalLoss, w=1.0 |
| `loss_bbox` | 中心+尺寸（**经 `nested_projection` 投影到目标角度下的 HBB**，配 `use_circumiou_loss=True`） | 逐点（`avg_factor=centerness_denorm`） | 否 | IoULoss, w=1.0 |
| `loss_centerness` | 中心质量 | 逐点（`avg_factor=num_pos`） | 否 | CrossEntropy, **w=1.0** |
| `loss_symmetry_ss` | **跨视图 角度一致性（仅角度）** | **逐对象**（`unique(bid)` + `index_reduce_('mean')`） | 是（`bmsk`；`agnostic_mask` 对 `[1,9,11]` 置零） | SmoothL1, w=0.2, beta=0.1 |
| `loss_angle` | 角度（直接监督） | — | — | **配置中不存在 → `None`** |
| （`use_reweighted_loss_bbox`） | 把 `loss_symmetry_ss` 灌进 `loss_bbox` 的标量门 | 标量（`.item()`，**非可微**） | — | **`False`（关闭）** |

### 核对结论：一个候选空档 + 一个已确认的真实杠杆

**候选空档（跨视图尺寸/中心一致性）**：s1 的跨视图一致性覆盖 **scale（`loss_ss_bbox`）与 angle（`loss_ss_symmetry`）两个维度**；s2 的跨视图一致性**只覆盖 angle**。而 s2 的 `loss_bbox` 是把预测 OBB 投影到**目标角度**下的 HBB 再与弱标签比（`nested_projection`：`da = pred_angle - target_angle`，再按 `|cos da|,|sin da|` 放大 `pred_wh`），**是逐视图对各自弱标签的监督，不是跨视图自一致**。

→ 即：**s2 中"跨视图的尺寸/中心一致性"没有人做，而官方 s1 做了。** 这比"我们发明了一个新损失"强得多：它是**官方两个阶段之间的一致性缺口**，且完全无需教师。

**但必须标注不确定性**：s2 用 `use_circumiou_loss=True` 让每个视图各自拿到一份（旋转后的）弱目标，对框形成了双份弱监督，**可能部分替代了跨视图框一致性**。因此这是**候选空档，需先验证是否被替代**，不能直接当已确认缺口。

**已确认的真实杠杆**：`use_reweighted_loss_bbox`（`math.exp(-loss_symmetry_ss.item()) * loss_bbox`），当前 `False`。零新代码、官方开关、非可微标量门——与我们原始 scores-into-loss 问题同族。

### 未核读项（诚实标注）

- `Point2RBox-v2` E2E head（`Point2RBoxV2Head`）在上游仓，本地未 clone，**源码未核读**；其损失清单来自配置文件：`loss_overlap=GaussianOverlapLoss(10.0)`、`loss_voronoi=VoronoiWatershedLoss(5.0)`、`loss_ss=Point2RBoxV2ConsistencyLoss(1.0)`、`loss_bbox=GDLoss(gwd,5.0)`、`loss_bbox_edg=EdgeLoss(0.3)`、`loss_cls=FocalLoss(1.0)`。维度/粒度未验证。
- s1 的 `ss_info` 多模式（`'sca'` 等）只读到入口，**未穷举全部模式**。

---

## 第二步：对象预算对齐

### 修正后的机制（取代上一轮的错误说法）

- `bcnt` = 对象在**几个视图**中获得正样本点 → 按构造=2
- `bmsk = bcnt == 2` = 保留**两个视图都有正样本点**的对象 = 有对应关系的对象
- **被排除的只有"在一个视图里拿不到正样本点的对象"**，即旋转/裁剪后落到画幅外的对象。这个集合是真存在的，但**有界**，且恰好就是**无法构成一致性对的集合**——排除它们是正确行为，不是丢失监督。

### 由此得到三条对 C1/C2 有实际影响的结论

**1. 有效批大小是配置值的 2 倍（必须修正预算算术）**

```python
batch_inputs_all = torch.cat((batch_inputs, batch_inputs_rot))   # 或 flp
feat = self.extract_feat(batch_inputs_all)
losses = self.bbox_head.loss(feat, batch_data_samples_all)
```

head 每步实际处理 `2 × batch_size` 张图。s2 配置 `batch_size=4` → **每步 8 张**。
→ 我此前 dry-run 的"每批对象数/KD 正点数"若按 4 张推算，**全部偏低一半**。C1/C2 的预算与"每对象质量"口径必须按 2× 复核。

**2. `bid` 的语义与硬约束（D1 适配器必须遵守）**

`bid = arange(n_objects) + 全局累积 offset + {0.2 ori / 0.4 rot / 0.6 flp}`

- **整数部分 = 对象**，**小数部分 = 视图**
- `offset` 在 batch 内**跨图累积** → bid 在整个 batch 内唯一，但**依赖图序与批序**，因此是**运行时非持久 ID**（与本轮之前审计一致）
- → **D1 适配器不得把 `bid` 原值当持久身份**：必须取 `.long()` 作为对象键，并把视图维度**显式外置**。直接复用 raw bid 会把"同一对象的不同视图"误当成不同对象。

**3. `rotate_crop` 是中心裁剪，被排除的对象由"旋转出画幅"导致**

```python
crop_h = (h - size_h) // 2      # 中心裁剪
crop_w = (w - size_w) // 2
# 旋转：xy = (xy - ctr)·tf.T + ctr,  a = a + rot   （wh 不变）
# 裁剪：batch_gt_instances[i].bboxes = 减去 (crop_w, crop_h) 平移，不做裁剪剔除
```

`crop_size=(1024,1024)`，而 train pipeline 已 `Resize(1024,1024,keep_ratio=True)` → 裁剪窗口基本等于画幅本身。真正造成排除的是**旋转**：`rot ∈ [0.25π, 0.75π]`（45°–135°，相当大），绕中心旋转会把靠近画面边缘的对象中心推出画幅 → 该视图拿不到正样本点 → 被 `bmsk` 排除。
**旋转保持到中心的距离**，所以纯径向不会出界；出界来自"正方形不是旋转不变的"。

### 未测的量（明确说明）

**"实际保留比例"本轮没有测**。它需要真实 `split_ss_dota` + 走一遍 FCOS 正样本分配（受 `strides=[8,16,32,64,128]`、`center_sample_radius=1.5`、`regress_ranges` 共同决定），**不能只从标注 manifest 推出**——我上一轮写的"用真实 manifest 算"是不准确的。

可测方案（如需）：在 DEV 上用真实 batch 加一行 instrumentation 打印 `bcnt` 分布与 `bmsk.sum()`，**约 1 次前向、无需训练**。是否要跑，等确认。

---

## 对上一轮决策备忘的影响

- 上一轮备忘第 3 节"发现 1（我们的 `object_mean` 不是新东西）"**依然成立**：host 确实已做 `index_reduce_('mean')` 逐对象归约。
- "发现 2（静默过滤/可能是 bug）"**作废**，替换为：s2 的注释缺陷 + **有效批大小翻倍** + **bid 视图编码约束**。
- "发现 3（`use_reweighted_loss_bbox` 是真空档）"**依然成立**，且现在有了第一步的覆盖表支撑：它确实是 s2 上**唯一**既空着、又官方留了开关的位置。
- 新增候选（本次覆盖表得出）：**s2 缺跨视图尺寸/中心一致性，而官方 s1 有** —— 需先验证是否被 `use_circumiou_loss` 双份弱监督替代。
