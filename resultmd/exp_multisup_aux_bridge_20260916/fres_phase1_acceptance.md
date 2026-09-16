# 阶段一验收：通用几何 KD 核心与 P1 桥接

日期：2026-09-16
开发根（DEV）：`/data1/zcy/Orbdet/work_dirs/runtime/multisup_aux_dev_20260916`
交付记录（REPORT）：`/data1/zcy/Orbdet/resultmd/exp_multisup_aux_bridge_20260916/`

**本报告只陈述代码兼容性与已实际运行的检查结果。没有做任何 AP 评测，不得被解读为精度提升。**

---

## 1. 修改/新增文件

新增（DEV）：

```text
geometry_aux/__init__.py        核心导出，不隐式导入 P1 或 Wholly
geometry_aux/types.py           AuxBatch 与形状/坐标/设备/ID 合同
geometry_aux/identity.py        object_mean_weights（改名 teacher_bids -> object_ids）
geometry_aux/losses.py          per_item_geometry / geometry_aux_loss
geometry_aux/provenance.py      validate_teacher_scope，教师监督预算校验
geometry_aux/adapters/p1.py     build_p1_aux_batch
tests/test_geometry_aux_core.py
tests/test_geometry_aux_provenance.py
tests/test_geometry_aux_p1.py
scripts/smoke/multisup_aux_p1_two_step.py
```

修改（DEV，兼容包装）：

```text
gda_support_profile/geometry_distill.py
  - object_mean_weights：保留 (image_indices, teacher_bids, *, dtype=None) 签名，
    实现改为委托核心
  - teacher_geometry_loss：保留原签名与原语义，per-row 几何改由核心
    per_item_geometry 计算；新增可选 reduction / image_indices / object_ids
  - 新增 aux_geometry_loss：持有 AuxBatch 的宿主代码可直接用核心入口
  - _add_auxiliary_losses：改为把身份与 reduction 一并传入，不再外部预算权重
```

**未改动**：backbone、head coder、assigner、原优化器、原 teacher 策略。
**只读快照未被修改**：`gda_object_equal_20260916`、`gda_e0_seed2026_20260915`、`c_equal_paired_20260915`、`data/DOTA-v1.0/`。

## 2. 核心隔离性

```bash
PYTHONPATH=DEV python -c "import geometry_aux; import sys; \
  assert not [m for m in sys.modules if m.split('.')[0] in ('mmrotate','mmdet','mmcv','mmengine')]"
```

结果：`mm* modules after import: []` → **核心导入不需要 P1、MMRotate registry 或 teacher 文件**。

## 3. 测试命令与实际结果

```bash
cd DEV && rtk proxy env PYTHONNOUSERSITE=1 PYTHONPATH=DEV OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 /data/zcy/anaconda3/envs/orbdet/bin/python -m pytest \
  tests/test_geometry_aux_core.py tests/test_geometry_aux_provenance.py -q
```

`40 passed`

```bash
... -m pytest tests/test_gda_object_mean_reduction.py \
                tests/test_gda_teacher_geometry_distill.py \
                tests/test_geometry_aux_core.py -q
```

`38 passed`（含对项目既有的"逐位等于 c_equal_paired_20260915 旧实现"测试）

```bash
... -m pytest tests/test_geometry_aux_p1.py -q
```

`10 passed`

覆盖的性质：

- 计划给定的两个权重性质测试（不等对象数 + 跨图同 ID 不合并；一对象一点退化为 1）
- **默认路径逐位等于旧 P1**：scalar 与 student 梯度均 `torch.equal`
- object_mean 等于"逐对象 legacy loss 的平均值"，含梯度（atol/rtol 1e-12）
- 点顺序置换不变；宽高交换等价框等价；angle-invalid 行角度梯度为 0
- 空 batch 返回与 student 相连的有限零（两种 reduction）
- teacher 与权重张量均无梯度；未知 reduction 拒绝；形状/kind/负 ID/NaN/设备不一致拒绝
- P1 adapter 的 bid 必须整且 ≥1；angle_valid 必须是 bool；adapter 路径与检测器路径逐位一致
- **禁用路径**：不打开 cache（用抛异常的桩验证）、state_dict 与宿主一致、loss 直接委托宿主且不含 `loss_teacher_geometry`

## 4. 有界 GPU 集成检查（真实运行）

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=DEV python \
  scripts/smoke/multisup_aux_p1_two_step.py \
  --config configs/orbdet/gda_object_equal_20260916/object_equal_seed42.py \
  --output REPORT/fres_phase1_p1_smoke_gpu2_seed42.json \
  --physical-gpu 2 --seed 42
```

使用 GPU **2**（8/9 当时占用 ~23.9 GiB，未抢占）。batch 4，2 步真实更新，无 Runner.train、无 checkpoint。

实测（来自 json）：

| 项 | 值 |
|---|---|
| reduction | object_mean |
| KD 正点 / 对象数 | 243 / 23 |
| 权重均值 | 0.99999994（合同要求 1） |
| **每对象质量 min / max** | **10.565215 / 10.565218**（逐对象等质量） |
| weight detached | true |
| teacher 目标 detached | true |
| student 未 detach | true |
| bbox / angle 梯度范数 | 0.06291 / 0.00245（均 > 0） |
| 主 head 参数实际更新 | 4/4（两步均更新） |
| GDA 预算 | applied 0.0944/0.0963 ≤ budget |

**结论**：辅助梯度确实到达保留的主几何参数，教师冻结，对象归约按对象等质量生效。

## 5. 未完成 / 已知偏差

1. **推理去旁路检查未执行**。本次未在 GPU 上跑推理对比（需一次完整 forward + 后处理）；代码层面已保证 disabled 时 `_geometry_teacher_cache is None` 且 `loss()` 直接委托宿主，但"输出等于同权重宿主推理"这一条**未实测**。
2. **`teacher_geometry_loss` 的劫持语义变更**：外部分析脚本（`work_dirs/analysis/gda_transfer_support_20260915/transfer_support_real_batch.py`，只读）通过替换 `module.teacher_geometry_loss` 来拦截。为保持其可用，检测器仍走模块级 `teacher_geometry_loss`（未改为直接调 `geometry_aux_loss`）。因此**检测器路径经由兼容包装**，核心的 `AuxBatch` 入口由测试与未来的 Wholly adapter 使用。这是有意为之的取舍，不是遗漏。
3. 空正点时的零损失在**核心层**已测试；检测器层的"整 batch 无 KD 正点"分支沿用原有早退逻辑，本次未单独实测。
4. 未启动任何正式训练，未产生 AP。

## 6. 阶段门判定

满足：核心独立于 P1、旧默认 loss/梯度逐位一致、对象 ID 正确、辅助梯度到保留主参数。
**未完全满足**："推理无需 teacher cache"仅代码层保证，未实测。建议在下次有空闲卡时补一次推理对比，再宣布阶段一完全通过。
