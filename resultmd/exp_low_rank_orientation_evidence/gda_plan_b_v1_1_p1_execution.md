# GDA Plan-B v1.1 → P1 长训执行方案

日期：2026-09-04
分支：`codex/hbox-angle-benchmark`，v1.1 提交 `8492065`（测试 8/8，全量回归 161 passed）
授权状态：用户已授权"改好代码直接进入长训"（2026-09-04 03:12）
前置：[gda_plan_b_implementation.md](gda_plan_b_implementation.md)（总方案）、[gda_plan_b_p0c_smoke_report.md](gda_plan_b_p0c_smoke_report.md)（冒烟判定与 v1.1 根因）

## 1. v1.1 补丁内容（相对 P0c 冒烟版的三处改动）

| # | 改动 | 根因（P0c §3） |
|---|---|---|
| 1 | 包络回归只在 ori+flp 紧视图（`/2.0`），rot 视图仅参与跨视图一致性 | rot 视图 GT 是旋转后的松 HBox，其包络系统性偏大 |
| 2 | 腔室 3a 目标 = **主头 decode 角度的 detach 腔室比特**（逐实例、三视图全用），掩码 |sin 2θ_main| < 0.3 | 旧 GT 编码目标 = sign(sin 2·rot)，HBox 监督下图像级均匀、与物体真实腔室无关 |
| 3 | 探针头训练时顺带 stash 主头 decode 角度（detach，逐层），检测器经 bid rank 对齐传入损失 | 2 的数据通道 |

防自强化：3a 是跨头蒸馏（主头梯度不经过该目标），3b 目标由配对视图 detach 生成；门控 `w=σ((â−a0)/τ)` 作用于全部腔室项（a0=log 1.15，τ=0.10，P0b 预检分布合理）。

## 2. P1 启动参数（冻结）

| 项 | 值 |
|---|---|
| 配置 | `configs/orbdet/orbdet_gda_probe_r50_dota1_grouped_ss_e0_gpu4.py`（唯一变量 = 探针头；数据划分/优化器/12 epoch/seed=3407 与 E0 逐项一致） |
| work_dir | `work_dirs/formal/orbdet_gda_probe_dota1_grouped_ss_e0_gpu4_seed3407` |
| GPU | CUDA_VISIBLE_DEVICES=4（开跑时空闲） |
| 代码哈希 | worktree `8492065` |
| 基线对照 | E0：`formal/orbdet_v0_2_dota1_grouped_ss_e0_gpu89_seed3407/epoch_12.pth`（sha256 `f7eb56a9…`） |
| 预算 | ≈5.7h（E0 实测 5h39m，探针开销 <5%） |

## 3. 启动保护（代替独立冒烟复跑）

按用户指令跳过独立冒烟复跑、直接长训，但**前 5 分钟按冒烟标准盯日志**：出现 NaN/inf、gda_* 缺失、loss 爆炸（>50）或崩溃，立即杀进程、回滚到问题定位，不空烧 5.7 小时。通过后再进入无人值守。

## 4. P1 完成后评测协议（只读，自动执行）

1. 用采集脚本在相同 grouped holdout（manifest sha256 `10993fda…`，4073 图）对 P1 epoch_12.pth 前向，产出证据缓存（同 E0 口径：IoU≥0.5 匹配、le90、双根口径列）；
2. 复算预注册判定表（`gda_plan_b_implementation.md` §1）：
   - **G1** aspect≥2 的 e2 P90 非劣（+0.5° 内）且 >15° 占比不升；
   - **G2** aspect∈[1.1,1.3) 翻转率相对基线 4.8% 下降 ≥30%；
   - **G3** AP50 非劣 −0.2pt；
   - **G4** 商空间口径列只报告不主张；
3. 附带诊断：â 与 GT aspect 的相关性（探针是否学到真实各向异性）、gda_bit_acc 的 epoch 级轨迹、门控权重分布（对照 P0b）；
4. 出 P1 判定表写入 `resultmd/exp_low_rank_orientation_evidence/gda_plan_b_p1_result.md` 并提交。

## 5. 分支规则

- G1–G3 全过 → 报告并申请 P2（双 seed 正式对比，≈12h）授权；
- 任一不过 → 停止，写死因分析，不进入 P2；
- 训练中断/异常 → 保留现场日志，报告后等指示。

## 6. 风险登记（不变，重申）

收益性质是损失卫生（37.8% 带噪信号降噪）+ 诚实计分；不承诺 P50；AP 主张只允许来自门控翻转下降。探针消融臂（detach_feats=True）配置已备，如 G3 受损可立即切换定位污染通道。
