# S1R Protocol Correction Impact Audit v1

状态：**S1R_IMPACT_AUDIT_SUPPORTS_PROTOCOL_AMENDMENT**

文档性质：`PROPOSED_ONLY`；`OWNER_APPROVAL_REQUIRED = TRUE`

审计基线：`7218b390d50c89c2f8d3070a3a40eb67b8c982ac`

## Overall scientific verdict

科学上支持一个 prospective Owner amendment。理由不是旧规则造成容量不足，而是 data role 与当前 claim 的偏差路径不匹配：nuPlan 在本研究中承担 frozen closed-loop intervention 的 controlled validation platform；普通历史 rollout 或 benchmark engineering 本身不会让 RBR-vs-H comparison 偏向某一方法。真正需要隔离的是 outcome 对当前 encoder、H、BDD、calibration、operating point、cohort 与 sample-size 决策的反馈。

这一修正不降低 applicability、mechanism、F_match、safety、technical completeness、全 denominator、no-survivor、SESSION cluster 或统计校准要求。它也不把 TSB 升级为 scientific-qualified benchmark。

审计已读取并绑定 `RBR-64_博士研究总体方案_v2.3.md`（状态为 controlled-validation freeze draft），并以任务正文作为审计范围补充。v2.3 本身不等于 Owner-approved S1 amendment；冻结 S1 只有在 Scientific Owner 批准独立、版本化的 amendment 后才改变。

## Frozen contract impact

影响矩阵共 29 条：KEEP_AS_IS=17，WORDING_UPDATE_ONLY=3，OWNER_AMENDMENT_REQUIRED=6，DEPRECATE_AND_REPLACE=3，UNKNOWN=0。

必须保持：BDD Primary、HLC closure、TSB candidate 与负面限制、U-only encoder firewall、H strong challenger、no survivor selection、SESSION clustering、paired/unpaired claim boundary、Primary metric/operating point及统计失败规则。

必须 amendment：Q 的职责、Q20/Q12 的角色解释、Q→D 自动转换、E 对所有历史 use 的绝对排除、canonical role vocabulary、S2 capacity blocker 的未来语义。旧文件全部保持原样，通过新的 versioned amendment supersede。

## 248 SESSION reclassification

| 互斥主分类 | sessions | 未来用途解释 |
|---|---:|---|
| E0 metadata-only/reservation-only | 1 | outcome 未暴露；reservation 仍需 Owner 处置 |
| E1 unrelated historical use | 176 | 可透明标记后进入 Claim B 的 V 上界 |
| E2 benchmark engineering | 66 | Claim B 可用；Claim A 不能称 unseen |
| E3 Primary method development | 0 observed | 若后续发现，必须从对应 confirmatory claim 排除 |
| E4 outcome-driven adaptation | 0 observed | 严格排除；当前 Primary 从未授权 |
| E5 untouched confirmatory | 5 | C 的 identity-level 候选 |
| UNKNOWN | 0 identity class | decision-use provenance 仍有文档局限 |

旧 strict audit 的 242 exposed / 243 conflict / 5 remaining 完整保留。新解释下，exposure-policy controlled-V 上界为 248，但**当前 certified V=0**；不能从规则过严跳到“248 全部可执行”。C 候选仍只有 5，且未做完整 applicability certification。

## Claim A 与 Claim B

Claim A（TSB generator generalizes to unseen sessions）需要排除至少 66 个有明确 benchmark-engineering exposure 的 sessions，并进一步处理 reservation 与 method-development provenance。

Claim B（frozen controlled intervention 下 RBR-BDD vs H-BDD utility）允许 E0/E1/E2 进入 V，前提是 PRIMARY_FREEZE_POINT 在 outcome access 前完成、E3/E4 不进入对应 confirmatory claim、全 denominator 保留且 SESSION-aware inference 生效。这两种 claim 不等价。

## 19 个明确答案

1. **v2.3 是否改变论文核心科学问题？** 否；它改变数据角色与有效性边界。
2. **BDD Primary 是否改变？** 否。
3. **HLC scientific state 是否改变？** 否；仍 closed by scope，impossibility 未建立。
4. **TSB scientific state 是否改变？** 否；仍 frozen development candidate，未 scientific-qualified。
5. **RBR encoder U-only firewall 是否改变？** 否，必须严格保留。
6. **H strong challenger 是否改变？** 否；仍 development-informed，并须在 Primary 前冻结。
7. **No survivor selection 是否改变？** 否。
8. **SESSION-level clustering 是否改变？** 否；eligibility 与 dependence 是不同问题。
9. **historical exposure → unusable 是否过严？** 对 Claim B 是；它没有区分 outcome 的 decision use。
10. **哪些 exposure 构成 claim-relevant contamination？** E3 直接方法开发与 E4 outcome-driven adaptation；尤其是 encoder/H/BDD/kernel/calibration/operating point/cohort/sample-size 的结果反馈。
11. **248 中多少可进入未来 V？** exposure-policy 上界 248；当前 certified 数为 0，须先 amendment 和重新 preflight。
12. **多少只能用于 benchmark development？** 对 Claim B 没有证据要求任何 session 永久 B-only；对 Claim A，66 个 E2 不能充当 unseen。
13. **多少应排除 confirmatory C？** 当前 243 个非 E5 session 不属于 untouched C 候选；E3/E4 若后续发现也必须排除。
14. **多少可作 untouched C 候选？** 5 个 identity-level 候选；未构造 C，样本量未定义。
15. **Q→D→E 哪些保留/替换？** 保留训练隔离、calibration/evaluation 分离和 freeze 后不适配；用 U/B/V/C 替换 absolute-fresh Q、自动 Q→D 和全历史-use禁入 E。
16. **是否建议 U/B/V/C？** 是，Owner amendment 后采用。
17. **S2 capacity blocker 如何处理？** Amendment 前仍是有效 blocker；amendment 后其 capacity 部分降级为旧 strict-firewall historical result，工程 binding/reset/precontext blocker仍在，必须重跑 metadata-only preflight。
18. **是否需要 Owner-approved S1 amendment？** 是。
19. **amendment 前 simulation 是否仍为 0？** 是，且 training/evaluation/E access 同样保持未授权。

## 对 S2/S3/S4 的影响

- S2：amendment 前不启动；之后先重做 role-aware metadata preflight，分别给 B/V/C 计数，并继续修复独立的 execution/reset/precontext blockers。
- S3：RBR training 仍只由 U 授权；nuPlan V/C outcome 不得用于 encoder selection。
- S4：Primary 只能在 freeze point 与 V/C identity binding 后执行；V 提供 controlled utility，C 提供相对 untouched robustness，不能事后扩样或换方法。

## Zero-run proof

`SIMULATION=0; RUNNER_RUN=0; TSB_ROLLOUT=0; HLC_ROLLOUT=0; RBR_TRAINING=0; NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0; EVALUATION_EXECUTION=0`。历史文件只提取 identity keys，不分析 scientific values。Protected CSV 未修改。

## Reproduction

在当前绑定输入保持不变时，对空目录运行 `python tools/s1r_protocol_correction_audit.py --output-dir <EMPTY_OUTPUT_DIR>`，应生成与本包逐字节一致的六份产物。代码检查使用 `python -m py_compile tools/s1r_protocol_correction_audit.py` 与 `python tools/check_no_tmp_dependencies.py`。
