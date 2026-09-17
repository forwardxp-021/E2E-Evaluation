# S1R Proposed Data Role Model v1 — U/B/V/C

状态：`PROPOSED_ONLY`。不得据此执行 simulation、training 或 evaluation。

## U — Representation Training

只用于 RBR encoder training/validation。Architecture、objective、checkpoint、seed handling 与 representation interface 只能由 U 的冻结目标决定。TSB/HLC/nuPlan V/C outcome、H 表现或 Primary BDD outcome 均不得进入 encoder selection。

## B — Benchmark Development

用于 TSB/HLC/controller、mechanism、F_match、safety、applicability 与 technical completeness。B exposure 必须透明标记。它排除 Claim A“TSB 对 unseen sessions 泛化”的 unseen 身份，但不自动排除 Claim B“在冻结 controlled intervention 下 RBR-BDD vs H-BDD utility”。

## V — Controlled Evaluation Pool

进入 V 前必须到达 `PRIMARY_FREEZE_POINT`，并通过既有 applicability、technical completeness、route/reference 与全 denominator 规则。按 exposure policy 的理论上界是 248 sessions；当前认证数仍为 0，因为 amendment、production/reset/precontext 与 eligibility 尚未闭合。

## C — Confirmatory Robustness

优先使用 E5，相对 untouched，用于验证 V 结果能否复现。当前仅有 5 个 identity-level 候选，远不足以自行定义 C sample size。C 不是论文唯一合法证据来源，也未被构造或访问。

## PRIMARY_FREEZE_POINT

在任何 V outcome access 前必须一次性冻结并绑定 SHA：

1. RBR encoder architecture、objective、checkpoint、seed rule 与 representation interface；
2. H feature list、implementation、normalization、preprocessing 与 readout；
3. BDD statistic、scaler、kernel/bandwidth rule、null/calibration procedure；
4. Primary metric、operating point、FPR gate、paired/unpaired design；
5. scenario eligibility、failure retention、denominator、sample-size 与 stopping rule；
6. analysis plan、cluster-aware inference、missingness/failure policy 与 multiplicity family；
7. U/B/V/C identity bindings和 exposure taxonomy classification。

Freeze 后禁止依据 V/C outcome 添加或删除 H 特征、切换 encoder/kernel、重校 operating point、筛选 favorable scenarios、扩样或替换 Primary。任何必要修复必须先判定为 infrastructure-only，且不得观察或利用 Primary scientific value。

## Q/D/E 处置

- 保留：U-only encoder firewall、calibration 与 evaluation 分离、Primary 后不适配、E/C 不扩样、全 denominator、cluster-aware inference。
- 替换：absolute-fresh Q、Q PASS 自动变 D、所有历史使用均永久排除 E。
- 新语义：Q 的机制/测量工作归 B；D 的方法开发职能分配到 B 或显式 pre-freeze method development；大规模 controlled comparison 归 V；相对 untouched robustness 归 C。
