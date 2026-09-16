# 方向决策：走教师网络，还是改做 host 旁路？

日期：2026-09-16。本轮**只读**：核读官方 host 源码。未训练、未改码、未动运行中的实验。

问题：还走教师模型网络（KD），还是直接做 Wholly-WOOD 改进版 + 旁路、顺带兼容其他网络？

## 结论

**走旁路，不走教师网络。** 但"加旁路"的槽位**不是空的**——动手前必须先看清已被占了什么，否则会重复造轮子。而且这轮核读发现：**我们原来的 "scores-into-loss" 问题在这个 host 里是活的，且官方已经留好了开关、当前处于关闭状态、完全不需要教师。** 所以这不是换课题，是把原问题**搬到有官方杠杆的 host 上**。

---

## 1. 教师路线为何不该是主线（三条独立证据）

**(a) 理论上撞天花板。** 弱监督 HHB 2 DOF vs OBB 3 DOF 的包络欠定（1-D fibre）：**损失侧任何再加权都无法把质量集中到真角度**。教师提供的是同一维度上更准的值，属于"再加权"范畴，撞同一堵墙。

**(b) 经验上不显著。** 三种子 base vs C_equal：Δmean 均值 +0.9582pp，样本 SD 1.4458，t=1.148；**效应全部来自 seed42**（+2.6271 vs +0.1582/+0.0892）；Δmean 0.96pp **低于噪声底线 1.70pp**。而跨 seed SD 收窄 2.31–3.56x。
→ KD 在这里的作用更像**方差稳定器，不是均值抬升器**。用"mean ΔAP"去找它的价值，是用错了判据。

**(c) 工程上代价高、收益被 (a) 封顶。** 教师来源只有三条路：自训 s1（12E）、下载 ai4rs unofficial（差 1.9–2.9pp + COCO→DOTA 转换）、或走 E2E 根本不需要教师。三条都为着一个按 (a) 撞墙的机制付费。

## 2. 决定性事实：这个 host 的旋转压根不靠教师

核读 `whollywood-le90_r50_fpn-1x_dota-s1.py` / `-s2.py`：

- **s1**：`loss_angle=None`（显式）。角度来自 `basic_pattern='data/basic_patterns/dota'` + `use_setsk=True` 的**合成图案**，以及 `RBox2Point(dummy=48, partial=1)`。
- **s2**：配置里**没有 `loss_angle`**（head 默认 `None`），角度信号**唯一来源是 `loss_symmetry_ss`**（`SmoothL1Loss(weight=0.2, beta=0.1)`），配 `view_range=(0.25,0.75)`、`use_circumiou_loss=True`。

→ **两个阶段都没有角度监督。Wholly-WOOD 解决旋转的方式本来就是"一致性旁路"，不是教师。** 给它加教师是**正交**的动作，不是补它缺的东西。

## 3. 但旁路里已经有一件东西（重要，必须认下来）

`H2RBoxV2PHead.loss_by_feat`（`h2rbox_v2p_head.py:330-371`，逐字核读）：

```python
_, bidx, bcnt = torch.unique(compacted_bid_targets.long(), return_inverse=True, return_counts=True)
bmsk = bcnt[bidx] == 2                     # 注释：bcnt is supposed to be 3, for ori, rot, and flp

# The reduce all sample points of each object
compacted_angle_targets = torch.empty_like(bid).index_reduce_(0, idx, pos_angle_targets[:, 0], 'mean', include_self=False)[bmsk].view(-1, 2)
compacted_angle_preds   = torch.empty(...).index_reduce_(0, idx, pos_angle_preds, 'mean', include_self=False)[bmsk].view(-1, 2, C)
compacted_angle_preds = self.angle_coder.decode(compacted_angle_preds, keepdim=False)

if b_flp:
    d_ang = compacted_angle_preds[:, 0] + compacted_angle_preds[:, 1]
else:
    d_ang = (compacted_angle_preds[:, 0] - compacted_angle_preds[:, 1]) - (compacted_angle_targets[:, 0] - compacted_angle_targets[:, 1])

if self.use_snap_loss:
    d_ang = (d_ang + torch.pi / 2) % torch.pi - torch.pi / 2
if compacted_agnostic_mask is not None:
    d_ang[compacted_agnostic_mask] = 0

loss_symmetry_ss = self.loss_symmetry_ss(d_ang, torch.zeros_like(d_ang))
```

**三点必须诚实记录：**

1. **host 已经在做"逐对象均值归约"**：`index_reduce_(..., 'mean', ...)` 按对象分组（`idx` 来自 `bid_targets`）取均值，**这正是我们以为新造的 `object_mean`**。
   → 诚实结论：**我们的 `object_mean` 相对这个 host 不是新东西，是同一思想的推广**（推广点是 `angle_valid`、宽高交换等价框、以及把归约抽成 host 无关的 core）。新颖性表述必须据此下调，否则站不住。

2. ~~**`bmsk = bcnt[bidx] == 2` 是静默对象过滤**~~ —— **此条已于同日作废，见下方更正。**

   > **更正（2026-09-16，执行第一/二步时发现）**：读到 detector（`h2rbox_v2p.py:133-192`）后确认，每步**只构造 2 个视图**（95% ori+rot、5% ori+flp），`bid` 的小数部分编码视图（+0.2/+0.4/+0.6），`.long()` 截断后坍缩为同一整数 → **`bcnt` 按构造恒为 2**，`bmsk == 2` 是"两视图都有正样本点"的**对应关系过滤器**，**是正确行为，不是 bug，也不是大规模静默过滤**。
   > 对照证据：s1 同源代码注释写的是 `(bcnt is supposed to be 2, for original and transformed)` 且代码 `== 2`——**s2 的注释 "supposed to be 3" 才是缺陷**（该路径不存在三视图）。
   > 详见 `fres_loss_coverage_and_object_budget_20260916.md` 第 0 节。

3. **`bid_targets` 既做分组键又是运行时非持久 ID** → 与之前审计一致：它是 `object_ids` 的天然来源，但**不能作为持久身份**依赖。补充：`bid` 的小数部分是**视图编码**，D1 必须取 `.long()` 并把视图维度显式外置，否则会把同一对象的不同视图误当不同对象。

4. **（新增，替代原第 2 条的影响）有效批大小是配置值的 2 倍**：`batch_inputs_all = torch.cat((batch_inputs, batch_inputs_rot))` → head 每步处理 `2 × batch_size` 张图；s2 配置 `batch_size=4` 即**每步 8 张**。此前 dry-run 若按 4 张推算，**每批对象数/KD 正点数全部偏低一半**，C1/C2 预算口径须按 2× 复核。

## 4. 真正空着、且官方已留开关的槽位（这是本轮的正面发现）

```python
if self.use_reweighted_loss_bbox:
    loss_bbox = math.exp(-loss_symmetry_ss.item()) * loss_bbox
```

- 这是**把一致性"分数"灌进 bbox loss 的标量门控**——**正是我们原始 "scores-into-loss" 问题的同族**。
- 注意 `.item()`：**非可微的标量门**，不是梯度路径。这本身就是个值得研究的取舍点。
- **当前配置 `use_reweighted_loss_bbox=False`**（s2 config 显式关闭）。
- **零新代码、官方开关、不需要教师** → 成本最低、最可证伪。

> **定位修正（2026-09-16 12:40）**：按"重加权(A)／新增约束(B)／换信号源(C)"分类（见 `fres_loss_coverage_and_object_budget_20260916.md` 追加节），本条属 **A 类标量重加权**——它**调制已有项、不新增约束**。故按包络欠定结论，**它不能突破 1-D fibre，只改优化动力学**。上文"最真实的杠杆"应下调为"**优化动力学杠杆**"，不宜作为机制性改善的首选。真正的机制入口是 B 类（布局约束的界限框）。

→ 所以原研究问题没有作废，它在**这个 host 里活着，而且被官方认可到留了开关的程度**。改做 host 旁路 = 把问题搬到有杠杆的地方，不是放弃问题。

## 5. "兼容其他网络"这件事

结构上已经支持：host 只负责（a）产出可微 OBB（b）提供对象身份；`geometry_aux` 核心零 mm* 依赖（sys.modules 实测为空）。P1 是第一个 host 实例，Wholly-WOOD 是第二个。

但"兼容"要变成可交付物，**缺的不是更多 host，而是一份 host 接入契约**（最小接口 `student_boxes[N,5]` + `object_ids[N]` + `angle_valid[N]`，外加"归约粒度"声明）。且必须在适配层**显式建立持久身份**，不得依赖 `bid_targets`。

## 6. 风险与代价（必须说清）

- **目标漂移**：原问题是"修复已有机制"（纠错），改为"提出并验证新旁路"需要新颖性论证与更强对照。验收标准随之改变。
- **重复造轮子**：`loss_symmetry_ss` 已占跨视图角度一致性 + 逐对象均值。新旁路必须先证明它与内置不重叠。
- **判据必须换**：三种子证据表明 equal-weight 的价值在**收窄跨 seed 方差**。若仍以 mean ΔAP 为唯一主指标，会重犯"用错判据"的错误。建议 **mean 与跨 seed SD 双主指标预注册**；且 n=3 时 SD 本身不可靠，n 太小不下结论。
- **新颖性下调**：见 3.1。

## 7. 建议的落地顺序（按成本从零开始）

1. **覆盖核对表**（零成本、只读）：逐项列出 s1/s2/Point2RBox-v2 每个损失——约束哪一维（中心/尺寸/角度/重叠）、归约粒度（逐点/逐对象）、是否被 mask 过滤。**只有确认存在空档才动代码。**
2. **对象预算对齐**（只读）：修正 `bcnt == 2` 的口径（**已于 2026-09-16 完成，见 `fres_loss_coverage_and_object_budget_20260916.md`**）。注：实际保留比例**不能只从 manifest 推出**，需真实 batch 走一遍正样本分配（1 次前向、无需训练），是否要跑待确认。
3. **只动一个真正空着的槽位**，按证据强度排序：
   - (i) ~~`use_reweighted_loss_bbox=True` —— 官方开关、零新代码、直击原问题。**首选。**~~ → **已下调为 A 类（只动动力学），非机制首选**。见 `fres_loss_coverage_and_object_budget_20260916.md` 追加节。
   - **(i') 布局约束的界限框（重叠上界／Voronoi 下界）作为第二组 `teacher_boxes` 塞进现有 core —— B 类、不改合同、成本最低的机制入口。**
   - (ii) 逐对象等质量归约的推广（含 `angle_valid`、宽高交换等价框）—— 注意这是推广而非新发明。
   - (iii) **跨视图尺寸/中心一致性**（覆盖表已确认：s1 有 `loss_ss_bbox`，**s2 只做角度、缺这一维**）—— 比"新发明"强，因为它是**官方两个阶段之间的一致性缺口**。但需先验证是否已被 s2 的 `use_circumiou_loss` 双份弱监督部分替代。
4. **判据预注册**（mean + 跨 seed SD 双主指标）后再考虑多 seed 长训。
5. **暂不建 D1/D2 九配置矩阵** —— 那是教师路线的产物，路线改了需重设计。

## 8. 待用户决定

1. **第二槽填什么**：跨视图自一致 / 第一阶段伪标签 / 仍用外部教师？（决定要不要自训 s1）
2. **是否接受目标由"修复已有机制"改为"提出并验证新旁路"**（决定验收标准与是否补文献新颖性论证）。
3. ~~**是否先执行第 7 节的第 1、2 步**~~ → **已于 2026-09-16 执行完毕**（覆盖表 + 对象预算口径），结果见 `fres_loss_coverage_and_object_budget_20260916.md`。**待决**：是否再跑"1 次前向测 `bcnt` 实际保留比例"（无需训练）。
