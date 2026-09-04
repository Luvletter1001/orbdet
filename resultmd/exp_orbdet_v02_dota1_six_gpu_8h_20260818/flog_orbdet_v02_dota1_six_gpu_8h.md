# Orbdet-v0.2 DOTA-v1 六卡八小时阶段值守交接

## 2026-08-18 02:37 +08:00 交接状态

用户已明确授权 GPU 4–9 的八小时有界正式阶段，并要求完成稳定值守后离线，
于 09:00 手工返回检查 checkpoint。本记录只描述交接瞬间；后续权威状态以
值守目录的终态 marker、checkpoint 强验证 JSON 和 `monitor.log` 为准。

| 项目 | 状态 |
|---|---|
| hard deadline | `2026-08-18 08:20:00 +08:00` |
| MS+RR | GPU 4–7，epoch 5，absolute iter 72,588，约 0.315 s/step |
| SS seed 42 | GPU 8–9，epoch 5，iter 14,100，约 0.470 s/step |
| primary slowdown gate | recent median 约 0.315 s，breach count 0 |
| source policy | DOTA 源数据只读；不覆盖已有 checkpoint |
| continuation policy | 不启动 MS epoch 9，不自动 resume，不建立下一轮队列 |

## 运行入口与产物

- MS tmux：`orbdet_dota1_msrr_e3e8_gpu4567_20260818`
- SS tmux：`orbdet_dota1_ss_seed42_gpu89_20260818`
- unattended monitor tmux：`orbdet_dota1_live_monitor_20260818`
- MS work dir：
  `work_dirs/formal/orbdet_v0_2_dota1_ms_rr_gpu4567_seed3407_resume_e3_to_e8_20260818/`
- SS work dir：
  `work_dirs/formal/orbdet_v0_2_dota1_ss_gpu89_seed42_20260818/`
- monitor root：
  `work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/live_monitor/`
- monitor log：
  `work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/live_monitor/monitor.log`

值守器已精确绑定：

- MS：pane/SID `3344507`，starttime ticks `199824051`；
- SS：pane/SID `3348515`，starttime ticks `199839073`。

绑定同时校验单 pane、`PID=PGID=SID`、完整 `/proc/<pid>/cmdline` argv 与进程
birth token。截止或主任务保护仅允许向这两个已登记 SID 内、再次验证归属的 PGID
发信号，禁止认领替换进程或修改外部 GPU 任务。

## 无人值守策略

1. 每 30 秒读取完整换行终止的 scalar JSON，检查有限 loss、grad norm 与 step time；
2. 记录新 checkpoint 的名称、大小和时间；训练终态后执行 epoch/iter/371 tensors、
   config token 与 SHA256 强验证；
3. MS 最近 20 点 median time 连续三个**新 step**观测高于 `0.341 s` 时，等待
   SS 的下一个完整 epoch checkpoint，再仅停止登记的 SS SID；
4. SS 完成 12E 后运行有界 trainval 诊断与 SS Task1 ZIP；MS 完成 8E 后仅在
   剩余时间门槛允许时运行后评测，否则写出明确 skip marker；
5. 08:20 到达后不启动新评测，清空仍存活的登记 SID 并验证无残留；不自动续训。

首次失败的只读 attachment 尝试因 `rtk tmux display-message` 不返回 pane PID 而
安全退出，已完整保存在：

`work_dirs/controllers/orbdet_dota1_sixgpu_8h_20260818/live_monitor_failed_attach_023008/`

正式值守改用经测试的 `rtk tmux list-panes -F '#{pane_pid}'`，连续三个轮询周期
均正常，`RUNNING` marker 与 breach count `0` 已确认。
