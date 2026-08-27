# R1 残余基准启用协议 v0.2

状态：`SCIENTIFIC_OWNER_APPROVED_FOR_IMPLEMENTATION_SMOKE`。

这是 v0.1 的版本化状态更新，不替换 v0.1，也不是完整 R1 protocol freeze。R0 v1.0 protocol、SAP、decision table、历史结果及 training authorization 都未修改。

## 获批范围

- 绑定 scientific-owner A--H 决策；冻结 context 与 mechanism measurement contract v1.0。
- 实现 HLC Stage7L deterministic external-planner architecture 与 TSB `PiecewiseLongitudinalProfileGenerator` 的 technical-smoke core。
- 严格隔离、outcome-blind 的 6+6 scenario technical smoke，最多 48 rollouts。

## 继续禁止

- RBR-A/B/C training、representation evaluation、embedding/BDD/probe/checkpoint 读取；
- 正式 R1 development roster rollout、R4 数据、根据表示 outcome 选参；
- 修改 R0 protocol/SAP/decision table 或历史 R0 result。

## 技术烟雾后的状态

初版的 logical arm 计划为 48，但因 baseline 重建循环，实际产生 72 次 trajectory-core construction calls，超过批准上限。结果已降级为 `NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`，不构成合规 smoke 证据；没有 candidate 同时通过 frozen F_match、mechanism pair gate 与 kinematic integrity，正式 roster readiness 为 `NOT_READY`。这不是 protocol 或 threshold 的修改，也不授权补跑。详见 `R1_Technical_Smoke_Execution_Scope_Correction_v1.1.md` 与 `R1_Development_Roster_Freeze_Readiness_v0.2.md`。

R-IP 保持 `SECONDARY_CONDITIONAL_NOT_REQUIRED_FOR_INITIAL_ENABLEMENT`；RBR-A/B/C 均保持 `NOT_AUTHORIZED`。

## 工具复核命令与通过标准

```bash
waymo_dev/bin/python -m unittest tests/test_r1_context_mechanism.py -v
waymo_dev/bin/python tools/r1_build_canonical_precontext.py --input <pre_context_input.json> --output <new_context.json>
```

前一命令应通过 14 个合成边界测试；后一命令只接受完整的 10-frame pre-context，输出两种 pair identity hash 和 canonical context record。`stageR_execute_r1_technical_smoke.py` 已修正为 baseline reuse，但当前无重跑授权，必须拒绝覆盖已有 smoke artifact；不得以该命令再次扩展或补偿 v1.1 已记录的 72-call 不合规执行。
