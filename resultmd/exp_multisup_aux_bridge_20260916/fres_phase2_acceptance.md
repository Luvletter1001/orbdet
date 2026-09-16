# 阶段二验收：Wholly-WOOD 多监督输入试点（本轮为部分交付）

日期：2026-09-16
宿主开发根（HOST）：`/data1/zcy/Orbdet/work_dirs/runtime/whollywood_aux_dev_20260916`
交付记录（REPORT）：`/data1/zcy/Orbdet/resultmd/exp_multisup_aux_bridge_20260916/`

**本轮只完成了 C0（接口审计）与 C1（标注预算协议）。宿主 adapter、九配置矩阵、跨输入训练均未做，且缺少合法 Point 教师，属于明确阻塞，不是"已完成可开训"。**

---

## 1. 已完成

### Task C0：锁定实际宿主（不再猜 API）

官方仓库已 clone 到 `/data1/zcy/Orbdet/work_dirs/derived/whollywood_official_20260916`：

- commit `4a77e648ee2911bbd9eaeb941f4fe7916d1d4e72`（2025-02-14），dirty 状态干净
- `LICENSE` 存在；`data/basic_patterns/` 存在
- 未覆盖 Orbdet 根或任何现有 runtime

接口审计报告：`REPORT/faudit_whollywood_interface.md`。三条最关键的核读结论：

1. **Point 确实分 s1 → s2 两阶段**（`Point2RBoxHDR` → `H2RBoxV2PDetector`），不能称单阶段端到端。
2. **`RBox2Point` 的 `partial` 是按标注行顺序取前 `round(n*partial)` 个对象**，把 w/h 覆写为 `dummy=48`、angle 覆写为 0 —— **不删除强几何**。官方取值是 `partial=1`（纯 Point）与 `partial=0.7`（官方 mix）；**不存在 `partial=.5`**。
3. **宿主有 `bid_targets`**：`H2RBoxV2PHead.loss_by_feat` 与 `Point2RBoxHDRHead` 的 `get_targets` 均返回 `(labels, bbox_targets, angle_targets, bid_targets)`。但 bid 是运行时按图内顺序生成的，**不是跨运行持久 ID** → 必须显式 ID 保序，已由 `stable_object_id` 提供。

### Task C1：标注信息与对象身份

新增（HOST）：

```text
geometry_aux/protocols/annotations.py      稳定ID、合法行过滤、按类型严格载荷、QBB→OBB兼容转换
geometry_aux/protocols/mixed_manifest.py   seed2026 SHA256 的按类 1:1 point/hbb 分配与 manifest
tests/test_aux_annotation_budget.py
scripts/analysis/export_aux_protocol.py    只读 --dry-run
geometry_aux/（核心，固定阶段一交付版本）
```

阶段一核心已固化进 HOST，摘要：
`phase1_core_sha256 = 538a330a237156b599b8a712f316b7524ebb94978be5d302df9c791874586d69`

测试：

```bash
cd /data1/zcy/Orbdet && rtk proxy env PYTHONNOUSERSITE=1 PYTHONPATH=HOST \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/zcy/anaconda3/envs/orbdet/bin/python \
  -m pytest HOST/tests/test_aux_annotation_budget.py -q
```

`13 passed`

覆盖：ID 跨图唯一与可逆解析、difficulty/ignore 过滤保序、各 kind 载荷长度严格、point 模型视图不含 w/h/angle（注入即拒绝）、QBB→OBB 能恢复有序矩形且拒绝低于边长下限与非矩形、1:1 分配与输入顺序无关、奇数类保留 floor 规则并报告实际比例、manifest 摘要篡改可被检测、point 条目携带几何即拒绝、写盘不覆盖。

### 只读 dry-run（真实数据，未写盘、未训练）

```bash
rtk proxy env PYTHONNOUSERSITE=1 PYTHONPATH=HOST python \
  HOST/scripts/analysis/export_aux_protocol.py \
  --image-list .../teacher_cache_c.train_keys.txt \
  --ann-dir .../orbdet_v04r_dota1_grouped_seed3407/train/annfiles --dry-run
```

| 项 | 值 |
|---|---|
| 训练图 | 10746（缺失标注 0） |
| 合法对象 | 183343 |
| 类别数 | 15 |
| point / hbb | 91668 / 91675（奇数类导致 hbb 多 7 个） |
| 各类比例范围 | 0.4990 – 0.5000 |
| `object_budget_manifest_sha256` | `8d5fd75423d2122f8a347f0b3d5e82d73d0ce170029c83fcf15a9c1b6b4e007f` |

图集沿用现有 10746 训练图清单，**未接入官方 trainval 额外图**，避免与旧 C 混表。

## 2. 未完成项与具体原因

| 任务 | 状态 | 具体原因 |
|---|---|---|
| C2 教师预算与缓存 | **阻塞** | **没有合法 Point 教师**。现有 HBB C 缓存按协议不得用于纯点实验（阶段二计划明文禁止）。没有 Point 协议教师 → 按计划要求"交付导出代码和明确阻塞，不能换强教师跑通"。本轮不生成任何教师。 |
| D1 宿主桥接 adapter | 未做 | 需要先安装/核验官方宿主环境；且 KD 目标依赖 C2 的教师。接口已知（`bid_targets`、PSCCoder、`DistanceAnglePointCoder`、le90），可以下一步直接写。 |
| D2 九配置矩阵 | 未做 | 依赖 D1；且配置存在不等于试验完成，本轮无训练授权。 |
| `tests/test_aux_whollywood_bridge.py` | 未做 | 桥接 adapter 不存在，无法写有意义的桥接测试；写了就是空壳。 |
| `scripts/smoke/whollywood_aux_two_step.py` | 未做 | 同上。 |
| `scripts/analysis/check_aux_matrix.py` | 未做 | 依赖九个配置存在。 |
| R 全部验收项 | 部分 | 仅覆盖到协议层（point 载荷无 OBB、分配可复现、每对象单一 kind）。与宿主相关的 disabled/grad 等价、GT 评价隔离、四类型推理合法 OBB **均未实测**。 |

## 3. 与本轮授权边界的一致性

- 未启动任何正式训练、长队列、tmux/nohup 或自动恢复。
- 阶段一那次 GPU 检查仅 2 步真实更新、batch 4、无 Runner.train、无 checkpoint，且使用当时空闲的 GPU 2，未占用 8/9。
- 未修改任何只读快照、未就地转换 `data/DOTA-v1.0/` 标注。
- 未伪造 AP，未伪造已有缓存。

## 4. 下一步（需用户明确授权后才会动）

1. 决定 Point 教师来源：训练一个 Point 协议教师，或改用仅 HBB 协议的三臂试点（HBB 协议本身合法，可先用现有 C 缓存）。
2. 安装并核验官方宿主环境，写 D1 adapter（接口已查明，工作量可控）。
3. 先跑 HBB 三臂，再 Point-only，最后固定 1:1 混合；seed42 试点先行。
4. 补阶段一遗留的"推理去旁路"实测。
