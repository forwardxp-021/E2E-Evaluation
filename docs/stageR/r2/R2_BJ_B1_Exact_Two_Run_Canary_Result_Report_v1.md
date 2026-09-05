# R2-BJ-B1 唯一一次 HLC V4 工程 canary 结果报告

## 冻结结论

本次一次性授权已消耗。生产入口严格按 `BASELINE → TREATMENT` 调用 `runner.run()` 两次，两臂 runner 均完成技术执行，预算由 2 递减为 0，未发生第三次调用，也没有 architecture failure audit。

冻结 post-run analyzer 随后在读取 V4 capture 参数时因 `KeyError:'capture_end_abs_s'` fail-closed。因此本阶段唯一合法结果状态为：

`R2_BJ_B1_CANARY_INFRASTRUCTURE_FAILURE_STOPPED`

该故障发生在两臂技术执行完成后的冻结分析阶段。它不能支持或否定 V4 scientific mechanism、endpoint、F_match、engineering 或 official safety 结论；禁止修复后重跑、手工拼接结果或执行剩余 14 runs。

## 授权与执行

- 授权文件 SHA256：`152836a99372cee8469ed62898ba4e3da378b33902b6db55d0947ec38702b31e`
- attempt ledger 中 canonical authorization SHA256：`74fb6ad290c7b833e730b6eb5ca836307b167549394bc55f4bdef5ad63f45770`
- outcome 暴露前本地提交：`312cb7c94545957c653b6161afd7699785bb447f`
- outcome 暴露前远端提交：`c0785b4292c7533dd07d4ff4a36bf076530bd8e2`
- 执行顺序：run order 1 baseline，run order 2 treatment
- `runner.run()`：2 次
- 消耗预算：2；剩余预算：0
- technical rerun：未授权且未执行
- identity replacement：未授权且未执行

## 两臂技术完成与 telemetry

| arm | runner report | realized trace | planner gate | controller-visible | actual LQR | actual-shadow |
|---|---:|---:|---:|---:|---:|---:|
| BASELINE | succeeded | 80（0–79） | 80/80 PASS | 80（0–79） | 79（0–78） | 79/79，最大绝对差 0 |
| TREATMENT | succeeded | 80（0–79） | 80/80 PASS | 80（0–79） | 79（0–78） | 79/79，最大绝对差 0 |

`controller_visible_telemetry` 仍仅表示 planner reference steering，不被改称 actual controller command。`actual_lqr_controller_telemetry` 是被动记录的 TwoStageController/LQR 实际返回及独立 frozen shadow；两臂 `behavior_changed=false`。

## 基础设施故障

冻结 analyzer SHA256 为 `b0a3daf7cc2234c5c77ad3800e0d15feecc377aca87ffff3141bbb51e8423da6`。它请求 `global_parameters.capture.capture_end_abs_s`；冻结 V4 参数文件 SHA256 为 `95b6b726a42f9501f6f5401e8b2e5e179cadb489b74087a09667889efd31a158`，其中对应字段命名为 `nominal_capture_end_abs_s`。本轮未修改 analyzer 或 V4 参数，也未再次执行 analyzer。

## 数据处置

原始 simulation、metric、trace 与 telemetry 文件仅保留在本地 production roots，不提交 Git。提交内容只包含授权记录、小型账本、SHA/行数审计、冻结 analyzer 自动结果和本报告。protected CSV 未修改，其 SHA256 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## 后续边界

- 剩余 14 runs：未授权、未执行。
- TSB：未启动。
- R2-C：未启动。
- confirmatory smoke：未启动。
- RBR：未启动。
- 下一步仅等待 Scientific Owner 对该不可重试的 infrastructure failure 作出处置。
