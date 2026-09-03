# GDA Plan-B P0c 冒烟判定报告

日期：2026-09-04
代码：worktree `codex/hbox-angle-benchmark` 提交 `007b7a0` + `7a32ea4`（init_weights 修复）
配置：`configs/orbdet/orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4_smoke.py`（8 图、1 epoch）+ `--cfg-options` 加长版（30 epoch = 120 迭代）
前置文档：[gda_plan_b_implementation.md](gda_plan_b_implementation.md)

## 0. 结论先行

冒烟**通过 2/3 门控，第 3 条（腔室准确率趋势）在本尺度无判定力——且已根因到设计层**，不是实现 bug。据此提出 Plan-B v1.1 小补丁（三处，见 §3），补丁落地并重跑冒烟后才申请 P1 授权。

## 1. 运行事实

| 项 | 值 |
|---|---|
| 首次冒烟 | 崩溃于 `init_weights`（ConvModule 带 norm 时 conv.bias=None），已修复并被 B-T6 钉住（提交 `7a32ea4`） |
| 修复后冒烟 | 4 迭代 + 加长 120 迭代，均干净跑完并保存 checkpoint，无 NaN/inf |
| 基线损失 | loss_cls/bbox/centerness/symmetry_ss 全部正常下降（总 loss 13.3→6.0），v0.2 诊断量（orbdet_q_*）完整输出——**基线目标未被污染** |
| 探针损失 | gda_loss_env 4.30→2.52（与 loss_bbox 同量级同趋势）、gda_loss_xview 0.57→~0.5、gda_loss_bit 0.073→0.049 全程有限 |
| 门控诊断 | gda_gate_mean 0.45→0.26~0.36（8 图过拟合上漂移，量级合理） |
| 资源 | GPU4 单卡，峰值显存 9.4GB，~1.0s/iter（含探针开销） |

## 2. 预注册门控逐条判定

| 门控 | 结果 | 判定 |
|---|---|---|
| 损失全有限 | 120 迭代无 NaN/inf，grad_norm 130–390（clip 35 生效） | **通过** |
| 包络损失趋势与基线同量级 | 4.30→2.52 vs loss_bbox 5.50→~1 | **通过** |
| 腔室分类准确率从随机（≈50%）起步上升 | 在 0.00–0.89 间二值跳变，无趋势 | **本尺度无判定力**（根因见 §3.2），转交 v1.1 后在 P1 判定 |

## 3. 冒烟暴露的两个设计问题（这是冒烟的价值所在）

### 3.1 rot 视图包络回归目标是松框

rot 视图的 GT 是"旋转后的 HBox"——一个绕物体的**松** OBB。它的轴对齐包络 ⊋ 该视图的紧包络。ori/flp 视图的 HBox 是紧的（vflip 保紧性），rot 视图不是。v1.0 在全部三视图做包络回归，等于向 rot 视图注入系统性偏大的尺度目标。

**v1.1 修正**：包络回归只在 ori+flp（紧 HBox 视图）；rot 视图的包络信号完全由跨视图一致性项（`rotate_sigma(Σ_ori, rot)` ↔ `Σ_rot`）承担——这正是 H2RBox-v2"弱视图监督、其余视图一致性"的哲学，也消除了松框偏差。

### 3.2 HBox 监督下 GT 腔室比特是图像级均匀的

rot 视图 GT 角 = 0 + rot（标量，全图所有实例相同）→ GT 腔室目标 sign(sin 2·rot) 与物体真实腔室**无关**，只是旋转 HBox 的编码约定。后果：(a) bit_acc 随批量目标类二值跳变（0.0/0.8），smoke 尺度无判定力——实测现象与此精确吻合；(b) 更严重的，对 aspect≥2 的实例它可能**惩罚经验上正确的腔室**（门控在这些实例上权重≈1）。

**v1.1 修正**：3a 的 GT 编码目标替换为**基线头角度的 detach 腔室**：`bit* = sign(sin(2·θ_main.detach()))`——实例级、在 aspect≥2 上经验正确（翻转率 0.00% 实测）、跨头蒸馏天然防自强化、且不引入新的可学习通道。3b（跨视图等变一致性，detach 目标）保留不变。这与 B2 的结论一致：腔室部分可学——让它向已经学会它的基线头对齐，同时用软门控保护近方形实例不被任何一方的腔室猜测惩罚。

### 3.3 v1.1 补丁清单（三处 + 测试更新）

1. `OrbdetGDAProbeLoss.forward`：包络回归项仅取视图 0（ori）与 2（flp）；签名不变。
2. 3a 目标源改为基线头 detach 角度的腔室比特：检测器 `_view_tensors` 增取主头 decode 角度（`pos_decoded_angle_preds`，v0.2 本就算好，detach 后经 `bid` 对齐传入）；`bit_min_abs_sin` 掩码改施于 |sin(2·θ_main.detach())|（主头在轨道坐标 ±45° 边界附近自身不可靠处屏蔽）。
3. 测试更新：B-T5 保持（3b 等变不变）；新增 B-T8：3a 目标 = 主头 detach 比特（无梯度、逐实例、掩码正确）。
4. P0b/报告中的门控语义不变（软权重仍作用于全部腔室项）。

## 4. 冒烟后状态与授权闸

- P0a（代码+解析测试 7/7、回归 160）、P0b（门控预检）、P0c（冒烟）已完成；
- **下一步是 v1.1 补丁（纯代码+CPU 测试，无需授权）→ 快速冒烟复跑（GPU4 <15min，需你点头）→ P1 单 seed 正式筛选（≈5.7h，需单独授权）**。
