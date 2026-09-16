# 自我审查：代码是否符合预期 / 冒烟能否运行 / 能否开训

日期：2026-09-16
审查对象：阶段一 DEV 树 `work_dirs/runtime/multisup_aux_dev_20260916`、阶段二 HOST 树 `whollywood_aux_dev_20260916`

本文件按"已验证 / 本轮修掉 / 发现但未修 / 结论"四段写，**不把"代码看起来对"当作"已验证"**。

---

## 1. 已验证（每条都有执行证据）

| # | 断言 | 证据 |
|---|---|---|
| 1 | 默认路径（point_mean）与旧 P1 **逐位**一致，含 student 梯度 | `torch.equal` 对比从 `c_equal_paired_20260915` AST 抽出的旧函数；`68 passed` |
| 2 | object_mean 等于"逐对象 legacy loss 均值"，含梯度 | atol/rtol 1e-12 对比通过 |
| 3 | 核心不依赖任何 mm* 框架 | `import geometry_aux` 后 `sys.modules` 中 mm* 为空 |
| 4 | object_mean 真正实现逐对象等质量 | GPU2 实测：243 正点 / 23 对象，每对象质量 **10.565215 与 10.565218** |
| 5 | 辅助梯度到达保留主参数、教师冻结 | bbox 0.0629 / angle 0.00245 均 >0；teacher detached=true；主 head 4/4 参数更新 |
| 6 | **正式启动环境能导入新代码**（审查中发现的最高风险项，已排除） | 用 `env -i PATH=... PYTHONNOUSERSITE=1 PYTHONPATH=WORKTREE`、`cwd=WORKTREE` 复现正式环境，走 `import_modules_from_strings(**cfg.custom_imports)` + `register_all_modules()`，`geometry_aux` 与桥接模块均从 DEV 正确解析 |
| 7 | 检测器类在 registry 中可被 `MODELS.build` 解析 | `MODELS.build(cfg.model)` → `GDAProjectionProfileGeometryDistillDetector`，cache 已载入，`reduction=object_mean` |
| 8 | 禁用路径不打开 cache、state_dict 与宿主一致、loss 直接委托宿主 | 用抛异常的桩 + config 构造三种方式验证 |
| 9 | 启动脚本的导入语义 | 读 `run_gda_m1_arm_20260909.py`：`env` 含 `PYTHONPATH=str(WORKTREE)`，`Popen(cmd, cwd=WORKTREE)`，命令为 `tools/train.py`。由于 `geometry_aux` 与 `gda_support_profile` 同为 WORKTREE 下的兄弟目录，**新增的导入边不会引入新的路径前提** |

## 2. 本轮自我审查发现并已修掉的问题

1. **测试过宽**：`test_enabled_bridge_does_open_and_validate_the_cache` 用 `pytest.raises(Exception)`。这会让任何无关异常都被当成"cache 确实被打开了"而通过。先实测真实异常类型为 `FileNotFoundError`（MRO: OSError），已收紧为该具体类型。
2. **公开 API 无覆盖**：`aux_geometry_loss`（我新增的、给未来 Wholly adapter 用的重导出）当时没有任何测试调用它，属于"未验证的公开面"。已新增测试，断言其在两种 reduction 下等于核心 `geometry_aux_loss`。
   → 该文件从 10 项增至 **11 项通过**。

## 3. 发现但**未修**的问题（明确列出，不含糊）

1. **`validate_teacher_scope` 的软点**：若两侧 manifest 都未声明 `train_image_keys`/`train_keys`，图像白名单包含性检查会被跳过（`if teacher_images and protocol_images`）。真正的守卫是 `train_image_manifest_sha256` 相等，因此这不是漏洞，但"没有声明键集合的教师"不会因键而被拒。已记录，未改，因为改动会影响已定的字段语义。
2. **`AnnotationRecord.validate` 的可读性**：obb 正尺寸判断写成 `A and B or A and C`，逻辑正确（等价于 `(A and B) or (A and C)`），但应加括号。属风格问题，未改。
3. **阶段一遗留项**：推理期关闭旁路后"输出等于同权重宿主推理"**仍未实测**，只有代码层保证。
4. **`per_item_geometry` 的空张量分支**：当 `student_boxes` 为空时不校验 `teacher_boxes` 形状；真实路径上 `geometry_aux_loss` 会先调 `batch.validate()`，合同仍成立，但该函数单独调用时守卫不完整。
5. **源码字符串扫描测试偏弱**：`test_p1_adapter_source_has_no_cache_or_registry_dependency` 用字符串匹配，改名可绕过。风险低，未改。

## 4. 冒烟测试能否运行

**能，且已经跑过一次**（证据：`REPORT/fres_phase1_p1_smoke_gpu2_seed42.json`，243 正点 / 23 对象 / 每对象等质量）。

重跑时有两个操作性前提，容易踩：

1. 脚本用 `open('x')` 落盘，**同一 `--output` 重跑会 `FileExistsError`**。换文件名即可（这是防静默覆盖的有意设计）。
2. **不要指向 GPU 8 / 9**。此刻 `gda_object_equal_20260916` 的两个 arm（seed42→GPU8、seed3407→GPU9）**仍在运行**，已历时约 8 小时 48 分。本轮用 GPU 2。

## 5. 能否开始训练

### 阶段二：不能。三重硬阻塞

1. **没有合法 Point 教师。** 现有 C 缓存是 HBB 协议的，阶段二计划明文禁止用于纯点实验；没有它就无法定义 Point 臂的 KD 目标。
2. **D1 宿主 adapter 与九个配置不存在**，`test_aux_whollywood_bridge.py`、`check_aux_matrix.py` 也未写（adapter 不存在时写了就是空壳）。
3. **阶段二的门槛本身未满足**：计划要求"P1 对象等权结果已判读"。实测 `fres_object_equal_pair.json` 为
   `"complete": false`、`"paired_seed_count": 0`，两臂状态均为 `train` —— **结果还没出来**。

按计划原文，此时正确动作是"交付可审阅代码和具体缺项，不伪造 AP、不自动扩展队列"。

### 阶段一（HBB KD 路径）：技术上已就绪，但**不应自动开训**

代码侧已具备开训条件：正式导入链已验证、模型可构建、默认路径逐位兼容。不开训的理由是流程与资源，不是代码：

1. 本轮任务范围是"改框架代码"，计划禁止因文档列了实验矩阵就自动开训。
2. GPU 8/9 被在跑的正式实验占用，无空闲"原配"卡。
3. 用同一 `object_mean` 配置从 DEV 再起一跑，只会与在跑的 arm 重复，**不增加任何信息**。

**唯一值得做的低成本验证**（建议，待授权）：等 `gda_object_equal` 收工后，从 DEV 用同一 config + 同一 seed 跑 1 步，把 `log_vars` 与历史日志逐项对比。这能在几乎零成本下把"重构未改变行为"从"逐位单元测试"升级为"与真实记录运行一致"。

## 6. 一句话结论

代码符合预期，默认路径已用逐位测试与真实 GPU 冒烟双重确认；阶段一可以开训但没必要现在开；**阶段二不能开训，因为教师预算与宿主 adapter 都还缺，而且它自己的前置门槛（对象等权结果判读）还没到。**
