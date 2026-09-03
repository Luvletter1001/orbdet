# GDA Plan-B：群分解角度探针头的实现与筛选实验计划

日期：2026-09-04
分支：`codex/hbox-angle-benchmark`（worktree `/data1/zcy/Orbdet/.worktrees/codex-hbox-angle-benchmark`）
前置报告：[fres_gda_group_decomposed_angle_gate_a.md](fres_gda_group_decomposed_angle_gate_a.md)（设计动机、Gate-A 12/12、Gate-B B1/B2 判定）
状态：**计划已冻结，等待授权**。代码编写与 CPU 单元测试不需要授权；**一切 GPU 作业（smoke 与正式训练）执行前需你当轮明确授权**。

## 0. 一句话

在 H2RBoxV2Head 上并联一个输出 KAK 坐标 (t, a, u₂) 的探针头，用包络一致性损失监督连续部分、用跨视图等变分类监督腔室比特、用 detach â 软门控抑制近方形实例的腔室梯度噪声；先 smoke 后单 seed 筛选，达标才进双 seed 正式对比。

## 1. 预注册成功判据（先于实验写下，与前置报告 §3 一致并操作化）

主对照：E0 基线（`formal/orbdet_v0_2_dota1_grouped_ss_e0_gpu89_seed3407/epoch_12.pth`，checkpoint sha256 `f7eb56a9…`），同 grouped holdout（manifest sha256 `10993fda…`，4073 图），同匹配口径（IoU≥0.5）。

| # | 指标 | 口径 | 通过线 |
|---|---|---|---|
| G1 | aspect≥2 实例 e2 P90 与 >15° 占比 | 缓存重算（同 §1.3 分解脚本） | P90 非劣（+0.5° 内）且 >15° 占比不升 |
| G2 | aspect∈[1.1,1.3) 翻转率 | 逐实例 flip 统计 | 相对基线 4.8% 下降 ≥30%（相对） |
| G3 | AP50（DOTA1 holdout） | 标准 mAP 评测 | 非劣于基线 −0.2pt |
| G4 | 商空间口径列（近各向同性实例按 S¹/C₄ 计分） | eval 附加列 | 只报告，不作为主张 |

**明确不承诺**：P50 改善（连续部分已 1.86–2.01°，无可争取空间）；任何来自 aspect≥1.3 真实翻转（全体 0.14%）的 AP 收益。

**失败纪律**：P1 任一 GATE 不过 → 停止，写死因分析进 resultmd，不进入 P2。探针头为纯增量分支，失败时配置开关关闭即回退，基线不受污染。

## 2. 现状资产（全部已核实路径）

| 资产 | 位置 |
|---|---|
| Gate-A 对称空间模块 | worktree `mmrotate/models/utils/gda_symmetric_space.py`（提交 `2d04ced`），API：`wh_theta_to_sigma / tapsi_to_sigma / sigma_to_tapsi / sigma_to_wh_theta / rotate_sigma / reflect_sigma / envelope_from_sigma / orbit_coordinate / chamber_bit / envelope_consistency_loss` |
| Gate-A 测试 | worktree `tests/test_gda_gate_a.py`（12/12，全量回归 153 passed） |
| v0.2 基线检测器 | `OrbdetV02Detector`（`mmrotate/models/detectors/orbdet_v0_2.py`）+ `H2RBoxV2Head`（`mmrotate/models/dense_heads/h2rbox_v2_head.py`，cls/bbox/angle/centerness 四分支 + ss 分支，`PSCCoder` 角度编码 le90） |
| v0.2 基线损失 | `OrbdetAnchoredSymmetryLoss`（`mmrotate/models/losses/orbdet_anchored_symmetry_loss.py`） |
| E0 训练配置 | 扁平化转储 `formal/orbdet_v0_2_dota1_grouped_ss_e0_gpu89_seed3407/orbdet_v0_2_r50_dota1_grouped_ss_e0_gpu89.py`（configs/ 下无源文件，以此为母本复制） |
| E0 耗时基准 | 单卡 12 epoch ≈ 5h39m（2026-08-26 13:16→18:55 日志） |
| 评测脚本 | worktree `tools/analysis_tools/eval_anchor_orbit.py`、`tools/analysis_tools/gate_b2_cross_seed.py`（提交 `6a4e1c6`）、`eval_filtered_orientation_map.py` |
| 证据缓存 | seed3407 / seed42 各一份（`work_dirs/analysis/low_rank_orientation_evidence_p0_square_cues_e0*_2026090{3,4}/evidence.jsonl`），供免训练预检 |

## 3. 实现分解（W0–W4）

### W0 探针头 `H2RBoxGDAHead`（新文件 `mmrotate/models/dense_heads/h2rbox_gda_head.py`）

- 继承 `H2RBoxV2Head`，主干/FPN/原四分支**完全不动**；新增并联小卷积分支（结构复刻 angle 分支：3×3 conv ×4 + 输出层），每点输出 4 通道：`(t_raw, a_raw, u2x, u2y)`；
- 参数化：`t = t_raw`；`a = softplus(a_raw)`（保证 a≥0 且在 a=0 处光滑）；`ψ = atan2(u2y, u2x)`（u₂ 双角载体，天然周期连续，规避角度回绕）；Σ = `tapsi_to_sigma(t, a, ψ)`，全程走 Gate-A 已验证函数；
- 配置开关 `gda_probe=dict(enabled=..., detach_feats=...)`：`detach_feats=False` 为默认（共享特征，拿梯度卫生收益）；`True` 为消融臂（探针完全不影响基线，用于隔离"共享主干被探针梯度污染"的风险对照）；
- 注册进 `mmrotate/models/dense_heads/__init__.py`；检测器仍为 `OrbdetV02Detector`，ss 视图管线不变。

### W1 损失装配（`mmrotate/models/losses/orbdet_gda_probe_losses.py`）

三项损失，全部只作用在探针分支输出上：

1. **包络回归损失**（弱视图，监督 t, a 与轨道坐标 |θ|）：`envelope_consistency_loss(Σ_pred, GT_hbox_wh)`——Gate-A 已有实现与解析测试（A6/A7/A9/A10）；
2. **跨视图包络一致性**（弱↔强视图）：`rotate_sigma(Σ_weak, rot)` 与 `Σ_ss` 各自过 `envelope_from_sigma` 后做 Smooth-L1——H2RBox 一致性项在 Σ 空间的改写，规范等价预测零惩罚；
3. **腔室比特等变交叉熵**（唯一离散量）：探针额外 2-logit 分类输出；目标由**配对视图 detach 预测**经已知旋转等变映射生成：`bit_target_ss = sign(sin(2θ̂_weak + 2·rot))`（θ̂ 来自 detach 的弱视图探针，sign 由 `chamber_bit` 给出）。防自强化机制与 v0.2 的 detach 诊断同构。
4. **内蕴软门控**：腔室损失与跨视图一致性损失的逐实例权重 `w = σ((â_detached − a₀)/τ)`，`â = a.detach()`（探针自身几何量，无可学习置信度通道——规避 v0.1 死因）；默认 a₀=log(1.15)、τ=0.10（B1 工作点：aspect 1.1–1.2 捕获 76.7–92.1% 翻转）；P0b 在证据缓存上验证权重分布合理性（纯 CPU）。

### W2 解析测试（`tests/test_gda_plan_b.py`，先写测试后接实现）

| 测试 | 钉住的声明 |
|---|---|
| B-T1 | 探针输出→Σ 处处 SPD、有限（含 a_raw 极值输入） |
| B-T2 | 等变合同：图像旋转 φ 后探针损失不变量行为正确（ψ→ψ+2φ，包络项不变，腔室目标按公式翻转） |
| B-T3 | 规范等价零惩罚：同一矩形的 V₄ 等价编码经探针损失给出相同值 |
| B-T4 | 门控行为：a→0 时腔室损失权重→0，a 大时→1；detach 正确（门控无梯度回传） |
| B-T5 | 腔室目标生成防自强化：目标对弱视图预测无梯度 |
| B-T6 | `detach_feats=True` 消融臂下基线四分支梯度与 v0.2 逐位一致 |

### W3 配置（worktree `configs/orbdet/`）

- `orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4.py`：以 E0 扁平配置为母本，仅改 `bbox_head.type='H2RBoxGDAHead'` + 追加 `gda_probe` 参数字典；数据划分、grouped holdout、优化器、12 epoch、seed=3407 全部与 E0 一致——**唯一变量是探针头**；
- `_smoke` 变体：沿用项目 smoke 惯例（短程、小数据、快验证）；
- GPU：`CUDA_VISIBLE_DEVICES=4`（当前空闲，开跑时复核占用）。

### W4 评测装配

- 解码仅发生在算指标时：`sigma_to_wh_theta`（le90）→ 与基线同管线出 AP50 与逐实例角度；
- 评测产物复用现有脚本链：缓存采集 → `eval_anchor_orbit.py` 双根口径 → §1.3 误差分解 → G1–G4 判定表；商空间口径列按前置报告 §3-5 实现为 eval 附加列，不进标准 mAP。

## 4. 分阶段执行与 GATE

| 阶段 | 内容 | 通过线 | 预算 | 授权 |
|---|---|---|---|---|
| **P0a** | W0–W2 代码 + 全部解析测试；全量回归零失败 | B-T1–T6 全过 + 既有 153 测试零回归 | 纯 CPU，无需 GPU | 无需授权（写代码） |
| **P0b** | 免训练预检：证据缓存上验证门控权重分布、探针参数化数值域 | 权重直方图合理（近方形≈0、长条≈1），无 NaN 域 | CPU 分钟级 | 无需授权 |
| **P0c** | smoke 训练（W3 smoke 配置） | loss 全有限、腔室分类准确率从随机（≈50%）起步上升、包络损失下降趋势与基线 snap 损失同量级 | GPU4，<30 min | **需授权** |
| **P1** | 单 seed 正式筛选（W3 主配置，12 epoch） | §1 G1–G3 全过 | GPU4，≈5.7h（E0 实测基准） | **需授权** |
| **P2** | 双 seed 正式对比（seed3407 + seed42，配置由 P1 配置换种子派生，命名沿用 e0/e0prime 惯例） | 双 seed 均满足 G1–G3；B2 脚本复核跨种子腔室不一致率在 [1.0,1.1) 桶相对基线 16.55% 下降 | GPU4，≈12h | **需授权** |

每个 GATE 的输出物：P0c/P1/P2 各出一份评测表（同前置报告格式），写进 `resultmd/exp_low_rank_orientation_evidence/` 并提交。

## 5. 风险与回退

| 风险 | 依据 | 对策 |
|---|---|---|
| 探针梯度经共享主干污染基线 | 唯一真实风险通道 | `detach_feats=True` 消融臂隔离；G3 AP50 非劣线兜底 |
| 腔室分类在近方形上自强化 | B2：这批实例 GT 腔室本身任选 | 门控把权重压到≈0；目标恒 detach；商空间口径列诚实计分 |
| 包络项信号弱于 snap 特手工实现 | snap 是 V₄ 不变量环的手工版，理论上等价 | P0c 观察收敛趋势；不达标在 smoke 阶段即停 |
| 收益不及预期 | 收益性质是损失卫生（37.8% 带噪信号降噪）+ 诚实计分，非角度跃迁 | 预注册不承诺 P50；G2 是唯一翻转主张；失败按纪律写死因 |

## 6. 新颖性自检（写作前必做）

组合主张保持"**对称空间表示 × HBox 弱监督 × 内蕴稳定子门控 × 商空间评测**"四元组（前置报告 §7）；动笔前按 SCQO 文档纪律做系统查重，此前不用"首次"。已知邻接工作：BCGE（TCYB 2025，全监督协方差回归）、GauCho（CVPR 2025）、GWD/KLD、Bou et al. 2024 结构张量编码、H2RBox/-v2。

## 7. 执行顺序小结

1. （无需授权）W0–W2 代码与测试、P0b 预检；
2. → 你授权 → P0c smoke；
3. → 你看 smoke 结果、再授权 → P1 单 seed；
4. → 你看 P1 判定表、再授权 → P2 双 seed。

每阶段结束我只报告一次判定表，中间不打扰。
