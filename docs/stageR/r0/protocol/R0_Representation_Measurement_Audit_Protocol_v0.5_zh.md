# R0 Representation & Measurement Audit Protocol v0.5（StageR 分支集成稿）

> 项目：E2E-Evaluation / 博士论文 Representation-V2
>
> 文档状态：`PARAMETERIZATION_PREP_DRAFT`
>
> Active development branch：`20260825_stageR_new`
>
> Remote branch HEAD（2026-08-25核验）：`460832bde6266f1367a10bfe00e9b3bc176740ce`
>
> Generation-1 historical source branch：`20260611_stage7_conclusion`（remote HEAD：`0f6fefd4363bdfcdeec37f3f7d38782516ba72dd`）
>
> 当前研究状态：
>
> ```text
> RBR_DIRECTION_FROZEN
> R0_SCIENTIFIC_SCOPE_FROZEN
> R0_PROTOCOL_V0_3_METHOD_AND_OPERATIONAL_REVIEW_PASSED
> R0_PROTOCOL_V0_5_STAGER_BRANCH_INTEGRATED_DRAFT
> R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
> RBR_ARCHITECTURE_NOT_FROZEN
> RBR_TRAINING_NOT_AUTHORIZED
> ```
>
> 本协议 v0.5 在 v0.4 的基础上完成 StageR active branch 与本地 Work/Codex 数据流集成；v0.4 已补齐冻结前最后两项操作边界：

1. 明确区分自然行为/morphology benchmark 与受控 planner treatment 的 prospective roster / mechanism-gate 规则，禁止根据 rollout 后的机制强弱筛选 confirmation 样本；
2. 将 RBR-A/B/C 的训练授权状态写成独立、机器可审计的 manifest，并绑定 protocol / decision table / asset / split / SAP 的 SHA 与 fallback。

它仍不是最终 v1.0 冻结稿，不授权训练 RBR-A/B/C，不改变 Stage6/Stage7/Stage7L 已冻结历史结论。

---


## 0A. StageR branch governance

Representation-V2 / R0 之后的新增协议、审计工具、RBR-A/B/C 开发和相关 machine-readable manifests 均以 `20260825_stageR_new` 作为 active development branch。Generation-1 冻结证据继续追溯到其历史 branch/commit，不因新分支建立而重写。

分支治理规则：

- `20260825_stageR_new`：StageR / Representation-V2 唯一主动开发分支；
- `20260611_stage7_conclusion`：Generation-1 历史来源分支，只读参考；
- R0 v1.0 freeze 时必须绑定 active branch name + exact local HEAD SHA + dirty status；
- checkpoint / tensor / historical output 的 source-of-truth 仍以实际 artifact SHA/config/log 为准，不能仅根据当前 branch 推断；
- 若 active branch 在 v1.0 freeze 前继续产生提交，最终 protocol/asset/SAP manifest 必须重新生成 SHA；
- 未经 training authorization，不得因为进入新分支而提前启动 RBR training。

## 0. v0.5 修订摘要

v0.5 继承 v0.4，并完成 StageR active branch 与本地 Work/Codex 数据流集成；v0.4 继承 v0.3 已完成的方法与操作修订，并补充冻结前最后两项边界。当前完整冻结前修订要点如下：

1. 将 v0.2 的模块级单值 `scientific_outcome/module_specific_outcome` 改为**逐假设记录**：每个 hypothesis 独立给出 `SUPPORTED / NOT_SUPPORTED / MIXED / INCONCLUSIVE / NOT_EVALUABLE`，模块仅保留非推断性的 `module_summary`。机器表采用“一条假设一行”。
2. 将 R0 行政闭环与 RBR-A/B/C **candidate-specific training authorization** 分离；`BLOCKED` 不再自动满足训练授权。
3. 将未来确认资产拆成 `FUTURE_R4_RESERVED_POOL` 与 `FUTURE_R4_CONFIRMATION_ROSTER`，消除 R0_AUDIT_HOLDOUT 在 R0 结束后“重新设计”的时间矛盾。
4. 将 D0-B 重命名为 `MATCHED_NATURAL_POSITION_RETENTION_STUDY`，明确其默认是 matched quasi-experimental evidence；只有 D0-C pooling 在相同 hidden sequence 上可称严格正交。
5. 将 D2 的 `COUPLING_SIGNAL_PRESENT` 改为 `EGO_CONTEXT_PAIRING_SENSITIVITY`；自然驾驶数据上的 shuffle 仅支持配对敏感性，不支持因果 interaction claim。
6. 将 `R_event_stat` 从 Primary readout family 移至 `Secondary non-BDD mechanistic comparator`。
7. D4 明确禁止使用 old64/A/B/C/ego13/RBR 的 embedding、BDD、probe outcome 筛选 audit/future roster；“difficulty适中”改为仅基于机制非退化、类别比例、独立单位、runnability 与预期功效的 outcome-blind feasibility。
8. 保留 v0.2 已完成的三层证据隔离思想，并细化为四个资产角色：`R0_DEVELOPMENT`、`R0_AUDIT_HOLDOUT`、`FUTURE_R4_RESERVED_POOL`、`FUTURE_R4_CONFIRMATION_ROSTER`。
9. D4 增加 **natural morphology vs controlled planner treatment** 的独立选择合同：受控 treatment 的 confirmation roster 必须在 rollout 前冻结，rollout 后只能对整套 frozen roster 执行 mechanism gate，不得以机制效果弱为由筛除样本后再做 embedding confirmation。
10. 最终训练授权 manifest 必须直接输出 `RBR_A/B/C_TRAINING_AUTHORIZATION`，并绑定所有冻结资产/协议 SHA、fallback ID 与授权理由。

---

# 1. R0 总目标

R0 不回答“哪一个新模型最好”，而回答六个在训练 Representation-V2 之前必须澄清的问题：

1. **D0 Temporal Contract**：Generation-1 的 Waymo 训练时间合同与 nuPlan/Stage7L 评估时间合同是否一致；若不一致，序列长度、event 位置或 pooling 是否造成可测的信息衰减？
2. **D1 Information & Geometry**：old64/A/B/C 中究竟保存了哪些纵向、横向、交互语义；64D 是否存在明显表示塌缩或低有效秩？
3. **D2 Context/Response Leakage**：learned representation 主要对 ego response、traffic context、ego-context pairing / conditional association、route/log/scenario shortcut 中哪些因素敏感？
4. **D3 Measurement Readout**：行为信息若存在于 64D 中，当前 full64 BDD/MMD 是否能够可靠读出；是否存在 task-signal dilution？
5. **D4 Residual Benchmark Protocol**：如何构造 handcrafted descriptors 已等价、但仍存在机制可确认时序/交互差异的 residual benchmark，并冻结后续 prospective confirmation 规则？
6. **D5 External Asset Feasibility**：DriveDNA、Person2Drive、StyleDrive 中哪些资产合法、可取得、可映射、值得进入后续外部验证？

R0 的输出必须是**决策与设计约束**，而不只是探索性图表。

---

# 2. R0 非目标

R0 明确不做以下事项：

- 不训练新的 RBR-A/B/C；
- 不重新训练 old64/A/B/C；
- 不改变 Generation-1 checkpoint；
- 不修改 Stage6/Stage7/Stage7L 已冻结 primary definition、null、threshold 或结论；
- 不把 ego13 改成 Primary 来“修复”历史结果；
- 不根据已解盲 nuPlan/Stage7L 结果搜索最佳 latent dimension/projection；
- 不建立新的最终 prospective confirmation 结论；
- 不因 R0 结果覆盖、重写或删除既有冻结资产；
- 不使用未来 R4 outcome 选择 temporal view、projection rank、kernel、bandwidth、equivalence margin、noninferiority margin 或 readout。

R0 允许使用已解盲 Stage6/Stage7/Stage7L 数据做**诊断、功效估计和开发决策**，但这些数据属于 development evidence，不得作为 RBR-V2 最终 prospective confirmation。

---

# 3. R0 结果状态模型：执行状态与逐假设科学结果分离

R0 不再使用一个模块级单值字段同时表达多个科学结论。每个模块由三层结果组成。

## 3.1 模块执行状态

```text
execution_status:
  COMPLETE
  BLOCKED
```

- `COMPLETE`：预冻结协议要求的分析、质量检查与统计输出已按计划完成。
- `BLOCKED`：资产、实现合同、样本量、对齐或统计前提不足，无法按计划完成。

`BLOCKED` 是执行事实，不等于科学假设“不成立”，也不自动允许进入训练。

## 3.2 逐假设科学结果

每个预注册 hypothesis 独立输出：

```text
hypothesis_result:
  SUPPORTED
  NOT_SUPPORTED
  MIXED
  INCONCLUSIVE
  NOT_EVALUABLE
```

定义：

- `SUPPORTED`：达到预冻结最小效应、稳定性及适用证据等级要求。
- `NOT_SUPPORTED`：分析可评估且完成，但未达到预冻结标准，或证据明确反对该假设。
- `MIXED`：不同 seed、representation、task、domain 或 metric 方向冲突。
- `INCONCLUSIVE`：现有设计无法区分主要解释，或不确定性过大。
- `NOT_EVALUABLE`：因模块/子实验 BLOCKED 或关键前提不满足，该 hypothesis 无法有效评估。

示例：

```json
{
  "module": "D2",
  "execution_status": "COMPLETE",
  "hypothesis_results": {
    "D2_RESPONSE_SENSITIVITY": "SUPPORTED",
    "D2_CONTEXT_SENSITIVITY": "SUPPORTED",
    "D2_PAIRING_SENSITIVITY": "MIXED",
    "D2_SHORTCUT_RISK": "SUPPORTED"
  },
  "module_summary": "CONTEXT_SHORTCUT_RISK"
}
```

## 3.3 module_summary 的边界

`module_summary` 只是便于工程决策的压缩标签，不替代 hypothesis-level 结果，不具有独立统计含义。一个模块可同时有多个被支持的 hypothesis。

例如 D2 可同时出现：

- `RESPONSE_SENSITIVE`；
- `CONTEXT_SENSITIVE`；
- `EGO_CONTEXT_PAIRING_SENSITIVITY`；
- `CONTEXT_SHORTCUT_RISK`。

机器可读结果表必须采用**一条 hypothesis 一行**，不得只保存一个 D0/D1/D2 模块总行。

所有结果还必须记录：

```text
evidence_level
primary_evidence
secondary_evidence
limitations
next_action
```

证据等级至少区分：

```text
R0_AUDIT_HOLDOUT_EVIDENCE
DEVELOPMENT_DIAGNOSTIC_EVIDENCE
```


# 4. 数据资产分层与反重复使用规则

## 4.1 四个资产角色

### A. `R0_DEVELOPMENT`

用途：

- 已解盲方法开发；
- variance / effect-size estimation；
- power analysis；
- probe/readout candidate development；
- matching strategy development；
- preliminary threshold/margin proposal；
- diagnostic plots。

允许包含：

- 已解盲 Stage6/Stage7/Stage7L；
- 已在 Stage6V 等阶段分析过的 Waymo test；
- 已知 outcome 的 nuPlan assets。

**重要：Waymo test 已经在 Stage6V 解锁并分析过，因此在 R0 中只能称为 `historical held-out development evidence`，不得继续称为 untouched test。**

### B. `R0_AUDIT_HOLDOUT`

用途：

- 在 R0 设计选择、threshold/margin、probe/readout capacity 冻结后，验证 R0 诊断决策是否稳定；
- 仍属于 R0 内部 audit，不是最终 R4 confirmation。

要求：

- 在 R0 关键门槛冻结前锁定；
- 与 R0_DEVELOPMENT 尽可能 `scenario/log-disjoint`；
- 若数据具有 driver identity，应优先 `driver-disjoint`；
- 不用于 tuning / threshold selection；
- 只按一次性 frozen analysis plan 评估。

### C. `FUTURE_R4_RESERVED_POOL`

用途：

- 在 R0 阶段预先封存未来 R4 可使用的数据源、scenario/token pool、日志池或受控生成规则；
- 防止 R1/R2/R3 根据模型表现再去寻找“更适合”的 future confirmation 数据。

要求：

- 在 RBR 正式训练前锁定 pool 来源或生成规则；
- R0/R1/R2/R3 不读取任何 RBR outcome 来决定 pool 内样本的保留/删除；
- pool 可大于最终 roster。

### D. `FUTURE_R4_CONFIRMATION_ROSTER`

用途：

- RBR-V2 最终 prospective qualification / confirmation。

形成时点：

- 可在 R1 residual mechanism / matching / runnability 规则稳定后，从 `FUTURE_R4_RESERVED_POOL` 中形成；
- 形成过程只允许使用 pre-treatment/context/matching/equivalence/mechanism/runnability 信息；
- 禁止读取 old64/A/B/C/ego13/RBR embedding、BDD、probe outcome 来筛选 roster。

要求：

- model / readout / kernel / threshold / margin / roster 全部冻结后才能解盲；
- 必须 scenario/log-disjoint；
- 如使用 human/external data，独立单位与身份隔离另行冻结。

## 4.2 如果没有足够数据建立 R0_AUDIT_HOLDOUT

R0 仍可执行，但所有科学结论必须标记：

```text
EVIDENCE_LEVEL = DEVELOPMENT_DIAGNOSTIC_EVIDENCE
```

不得使用：

```text
confirmatory
prospective confirmation
validated on untouched holdout
```

等措辞。

## 4.3 资产在时间上的冻结顺序

正确顺序：

1. 建立 asset inventory；
2. 生成 candidate grouping keys；
3. 在查看新的 R0 metric 结果前冻结 `R0_DEVELOPMENT / R0_AUDIT_HOLDOUT`；
4. 同期锁定 `FUTURE_R4_RESERVED_POOL` 的数据源/token pool/生成规则；
5. 仅 `R0_DEVELOPMENT` 参与 R0 方法、门槛与 readout 开发；
6. 冻结 R0 statistical analysis plan；
7. 一次性运行 `R0_AUDIT_HOLDOUT`；
8. 进入 R1 后，只能按已冻结的 mechanism/matching/runnability 规则从 `FUTURE_R4_RESERVED_POOL` 形成 `FUTURE_R4_CONFIRMATION_ROSTER`；
9. roster 形成过程中不得读取任何 RBR outcome；
10. `FUTURE_R4_CONFIRMATION_ROSTER` 持续封存直至 R4 正式授权。

---

# 5. 已确认事实与待核事实

## 5.1 已确认代码事实

当前 Waymo 5-neighbor builder：

- 默认 `window_len=80`；
- `dt=0.1 s`；
- ego 每帧 8D；
- 5 个 semantic neighbor slots；
- 每个 neighbor 每帧 15D；
- context 每帧维度 `8 + 5×15 = 83`。

nuPlan context adapter：

- 复用 Stage5D 的 ego 8D / neighbor 15D / 83D context 公式；
- semantic slot 在时序中发生 tracked-object identity switch 时，acceleration/yaw-rate temporal derivative 使用 reset/invalidation 逻辑，不跨不同 agent identity 做差分；
- static 与 safety-critical derived formulas 与 temporal derivative parity 分开审计。

## 5.2 待 R0 本地核验事实

以下内容虽然在 handover/报告中出现，但 R0 不预设其实现细节：

- old64/A/B/C **实际训练时**每个 checkpoint 对应的 `T` 是否全部为 80；
- Stage6T candidate trainer 是否存在 crop/pad/resample；
- Stage7L/nuPlan **实际送入模型**的 tensor shape；
- 150 帧序列是否直接送入 GRU；
- embedding 是否只使用 final hidden state；
- 是否能够无训练恢复 full hidden sequence；
- mask 在 80/150、slot switch、缺失 neighbor 情况下如何消费；
- B/C inference pipeline 是否与 Waymo training preprocessing 完全同构；
- full64 MMD kernel/bandwidth 的实际实现规则；
- ego13 当前正式实现的精确 13D feature list、normalization、missing-value policy；
- A/B/C 3408/3409 checkpoint 的确切路径、SHA、训练配置与可用性。

所有上述项目在本地证据核实前标记：

```text
TO_BE_VERIFIED_IN_R0
```

---

# 6. 冻结资产原则与 seed 设计

R0 开始执行前必须建立 `r0_asset_inventory.csv`，至少记录：

- asset role；
- path；
- SHA256；
- git commit / code SHA；
- dataset split；
- tensor shape；
- row/scenario/log/driver count；
- whether unblinded；
- evidence tier；
- allowed use；
- preprocessing version；
- mask/schema version；
- representation seed；
- checkpoint training config SHA。

至少纳入：

- old64 checkpoint；
- A-3407 / A-3408 / A-3409；
- B-3407 / B-3408 / B-3409；
- C-3407 / C-3408 / C-3409；
- Waymo Dynamic-v2 train/val/test manifests；
- Stage6J/K assets；
- Stage6P release trial assets；
- Stage6S-v3 assets；
- Stage7L rollout/context/embedding/BDD assets；
- ego13 implementation/config；
- current MMD/BDD implementation；
- relevant context feature schema。

### Seed 规则

- seed 3407：`PRIMARY_SEED`
- seeds 3408/3409：`REPLICATION_SEEDS`
- 核心机制结论优先要求至少 `2/3 seeds` 方向一致；
- 若 3407 与 3408/3409 强烈冲突，hypothesis_result 至少为 `MIXED`，不得只报告 primary favorable result；
- old64 若仅有单 seed，则显式标记：

```text
SEED_REPLICATION_NOT_AVAILABLE
```

且不得把其单 seed 稳定性与 A/B/C 三 seed 稳定性直接等同。

R0 只读历史资产。任何为了审计而生成的新文件必须写入独立 `outputs/r0_*` 路径。

---

# 7. 独立单位、分组切分与聚类规则

所有 probe、bootstrap、permutation、CI、matching 与 hypothesis test 必须预冻结四个字段：

```text
independence_unit
group_split_unit
bootstrap_cluster
permutation_unit
```

通用原则：

1. 同一 log 或 scenario 切出的多个 window 不得当作统计独立样本。
2. 若多个 planner/treatment rollout 来自同一 scenario，paired analysis 的 independent unit 为 scenario pair；cluster 至少不低于 log。
3. 若同一 log 含多个 scenario，bootstrap 默认优先按 `log` cluster；如科学问题要求 scenario-level paired randomization，则 permutation 可按 scenario pair，但 CI 仍需报告 log-cluster sensitivity。
4. Waymo window-level probe 的 split 必须按 source-level grouping key 阻止同一 scenario/segment 泄漏到 train/val/audit holdout。
5. 若未来 external data 含 driver identity，则 driver-level generalization 的 group split unit 必须为 driver，而非 episode。

任何偏离上述规则的分析必须写入 `r0_protocol_deviation_log.csv`。

---

# 8. D0 — Temporal Contract Audit

## 8.1 科学问题

Generation-1 模型在 Waymo 训练时见到的 temporal support 与 nuPlan/Stage7L 推理时实际 temporal support 是否不同？若不同，**长度、event位置/保留距离、pooling** 三个因素中哪些会导致行为信息衰减？

## 8.2 科学假设

- **H-TLENGTH**：在 event 内容和相对支持尽量等价时，80 与 150 temporal contract 本身造成可测信息差异。
- **H-TPOS**：固定总长度后，event 距序列末端越远，last-state readout 对该 event 的信息保留越弱。
- **H-TPOOL**：在同一条 150 帧 hidden sequence 上，last-state pooling 相对固定无学习 pooling 存在 task-signal dilution。
- **H-TNONE**：上述 temporal factors 均不足以解释实质性信息差异。

## 8.3 第一阶段：合同核验

对 old64/A/B/C 每个 seed 输出：

- training tensor `T,D`；
- inference tensor `T,D`；
- model forward signature；
- GRU/encoder return object；
- pooling logic；
- mask logic；
- crop/pad/resample logic；
- event position distribution；
- checkpoint config SHA；
- hidden sequence 是否可无训练提取。

若无法从 config 恢复，必须从 dataset tensor、training log、checkpoint metadata 和代码交叉确认。

## 8.4 D0-A：Length effect（受控程度依构造质量分级）

目标：在不混入 event presence / event position / pooling 变化的前提下，尽可能单独测试 temporal length contract。

原则：

- 选择能够在 80 与 150 合同中保留**同一 event 内容**的 episode；
- event anchor、event phase、primary motion segment 必须匹配；
- pooling 固定；
- preprocessing/mask 固定；
- 若通过 pad 构造 150，需要明确 pad value、mask 与是否为训练时可见模式；
- 若通过真实上下文扩展到 150，需要把新增 pre/post context 作为内容变量记录。

主比较示例：

```text
L80_matched_event
L150_same_event_support
```

若不能在不改变内容的情况下构造严格可比的 80/150，则该比较不能称为 pure length effect，必须降级为 descriptive。

D0-A 每个比较必须标记 `CONTROLLED_LENGTH_STUDY` 或 `CONTENT_CONFOUNDED_LENGTH_DIAGNOSTIC`；只有前者可以支持较强 length-effect 解释。

## 8.5 D0-B：MATCHED_NATURAL_POSITION_RETENTION_STUDY（默认 matched quasi-experimental）

目标：在自然 episode 中通过 matching 尽量固定 event family、duration、magnitude、场景背景与总长度，研究 event 位置/尾随帧数与信息保留之间的关联。该设计默认不是严格正交实验，不作强因果解释。

预定义位置：

```text
P_early
P_middle
P_late
```

要求：

- 总长度相同；
- event family 相同；
- event duration / magnitude 尽可能 matching；
- background context 通过 matching/stratification 控制；
- 不允许根据 Stage7L detection 结果调整 early/middle/late 边界。

主指标：probe information retention 或 task-aligned separation 随 `frames_after_event` 的变化。结果解释为 matched natural retention association。若另行构造同一 event 内容的受控时间平移实验，可作为 secondary controlled study，但必须同时报告输入 OOD 审计。

## 8.6 D0-C：Pooling effect（严格正交主实验）

目标：对**同一条 hidden sequence**仅改变 pooling/readout。

Primary fixed non-learned pooling candidates：

```text
last
mean
max
```

可选 secondary：

```text
masked_mean
predefined_recent_k_mean
```

任何 learned attention / outcome-tuned weighting 不属于 R0 primary pooling audit。

## 8.7 Content-window descriptive diagnostics

以下保留，但不得用于单独证明“final-state 遗忘”：

- `V_first80`
- `V_last80`
- `V_event80`
- `V_overlap80`
- `V_full_native`

这些 view 同时改变输入内容、event presence、event position 或支持范围，只能标记：

```text
CONTENT_WINDOW_DESCRIPTIVE_DIAGNOSTIC
```

## 8.8 两套 semantic probe 合同

D0 必须并行运行两类 probe，以区分“embedding坐标改变”与“信息本身丢失”。

### Probe-A：Cross-view frozen probe

- 在 reference view 上训练一次；
- 同一个 frozen probe 跨 temporal view 使用；
- 测量 representation geometry / coordinate compatibility。

### Probe-B：Per-view refitted fixed-capacity probe

- 每个 view 在 Waymo train 上独立重拟合；
- 使用完全相同 target、capacity、regularization search space 和训练预算；
- val 规则固定；
- 测量“该 view 中信息是否仍存在”。

解释：

- Probe-A 降、Probe-B 不降：更接近坐标/geometry shift；
- Probe-A 与 Probe-B 都降：更支持 information retention loss；
- 两者都稳定：不支持实质 temporal loss。

## 8.9 Seed 稳定性

A/B/C 在 3407/3408/3409 全部运行核心 D0 指标。

核心 temporal mechanism 若用于后续 RBR-B design，原则上要求：

```text
direction_consistency >= 2/3 seeds
```

并报告 primary seed 与 replication seeds 的 effect size / CI。

## 8.10 D0 输出状态

必须输出：

```text
execution_status
hypothesis_results
module_summary
```

D0 至少预注册以下 hypothesis：

```text
D0_LENGTH_EFFECT
D0_POSITION_RETENTION_ASSOCIATION
D0_POOLING_EFFECT
```

每条 hypothesis 的 `SUPPORTED` 必须满足其对应设计的预冻结最小效应与稳定性要求：

1. 合同核验确认相关 temporal factor 在实现上真实存在；
2. 对应受控/准实验达到预冻结最小效应；其中只有 D0-C 可直接称严格正交；
3. 至少两种 frozen learned representations 或 A/B/C 至少 2/3 seed 方向稳定；
4. 证据在 R0_AUDIT_HOLDOUT 重现；若无 audit holdout，则 evidence level 只能是 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；
5. 不是单一 Stage7L post-hoc effect。

`module_summary` 可根据多条 hypothesis 组合为 `TEMPORAL_EFFECT_PRESENT / NO_MATERIAL_TEMPORAL_EFFECT / MIXED_TEMPORAL_EFFECT / TEMPORAL_AUDIT_INCONCLUSIVE`，但不得替代逐假设结果。

**D0 scientific outcome 不自动指定下一代必须采用 overlapping80 或某一 pooling。具体方案进入 RBR-B development ablation。**

---

# 9. D1 — Gen-1 Information & Geometry Audit

## 9.1 科学问题

old64/A/B/C 是否已经包含 ego13、lateral、interaction 等可迁移语义？64D 是否发生表示塌缩？

## 9.2 Probe 原则

Primary semantic probe：

- linear ridge / linear regression / logistic regression；
- 低容量；
- 只在允许的 R0_DEVELOPMENT training partition 拟合；
- hyperparameter 只在 development val 选择；
- R0_AUDIT_HOLDOUT 只做一次 frozen evaluation；
- nuPlan/Stage7L 若已解盲，只属于 frozen cross-domain development transfer；
- probe 训练数据、target、normalization、seed、SHA 全记录。

高容量 MLP 只能 Secondary，不得以其好结果单独证明“representation contains linearly usable information”。

## 9.3 Target families

### Known semantics

- ego13 各维；
- speed / accel / jerk；
- yaw rate / curvature / lateral acceleration；
- lane-change duration / oscillation（若定义与数据契约一致）；
- following distance / THW / closing；
- interaction variables。

### 不使用 future confirmation morphology label

R0 不使用 FUTURE_R4_CONFIRMATION_ROSTER label 训练 probe。

## 9.4 Information metrics

连续变量：

- held-out R²；
- MAE/NRMSE；
- Spearman correlation；
- calibration slope（若适用）。

离散变量：

- AUROC；
- balanced accuracy；
- macro-F1。

跨域：报告 Waymo→nuPlan transfer degradation。

## 9.5 Geometry diagnostics

对每个 representation / seed / split 报告：

- covariance eigen-spectrum；
- singular values；
- PCA cumulative explained variance；
- effective rank；
- participation ratio；
- dimension-wise variance；
- cosine-distance distribution；
- isotropy proxy。

不得将“effective rank 更高”直接解释为“behavior information 更多”。必须与 probe、leakage、BDD 联合解释。

## 9.6 D1 输出

```text
execution_status = COMPLETE / BLOCKED
hypothesis_results = {
  D1_KNOWN_SEMANTIC_INFORMATION_PRESENT,
  D1_CROSS_DOMAIN_SEMANTIC_TRANSFER,
  D1_GEOMETRY_DEGENERACY
}
module_summary =
  KNOWN_SEMANTICS_PRESENT
  KNOWN_SEMANTICS_WEAK_OR_ABSENT
  DOMAIN_TRANSFER_FAILURE
  GEOMETRY_DEGENERACY_WITHOUT_CLEAR_INFORMATION_LOSS
  MIXED
```

逐假设至少包括：

```text
D1_KNOWN_SEMANTIC_INFORMATION_PRESENT
D1_CROSS_DOMAIN_SEMANTIC_TRANSFER
D1_GEOMETRY_DEGENERACY
```

解释要求：

- `KNOWN_SEMANTICS_PRESENT` 不是“Gen-1成功”，只表示 frozen latent 中存在可读信息；
- `KNOWN_SEMANTICS_WEAK_OR_ABSENT` 需要 held-out evidence，不能由单一 probe failure 推断；
- `DOMAIN_TRANSFER_FAILURE` 要区分训练域中有信息与跨域读出失败。

---

# 10. D2 — Context / Response Leakage Audit

## 10.1 科学问题

64D 的差异对 ego response、traffic context、ego-context pairing / conditional association、scenario/log/map shortcut 中哪些因素敏感？

## 10.2 关键解释边界：Gen-1 ablation 可能是 OOD intervention

对于共享 83D single-GRU：

- ego 置零；
- neighbor 置零；
- context-only；
- mask 改变；

都可能构成训练时从未见过的输入。

因此 Gen-1 任何 ablation 结果只允许解释为：

```text
ABLATION_SENSITIVITY
```

不得直接写成：

```text
causal dependence on ego
causal dependence on context
context is useless
neighbor information is necessary
```

未来 RBR-C 若在训练时原生支持 branch ablation，才更适合做正式信息归因。

## 10.3 Ablation 定义必须记录的字段

每种 ablation 必须明确：

- ablation 发生在 raw space 还是 normalized space；
- zero 对应 physical zero、normalized zero、train mean 还是 train median；
- valid mask 如何处理；
- missingness channel 是否同步改变；
- track-id/slot-valid semantics 是否同步改变；
- derived features 是否在 ablation 前还是后计算；
- neighbor slot identity-switch validity 是否保持；
- ablation 后 embedding norm / Mahalanobis distance / nearest-neighbor distance 是否超出训练分布。

## 10.4 数据视图

可使用：

- `full`；
- `ego-ablated`；
- `neighbor-ablated`；
- `context-only`；
- `context-shuffle`。

若某视图无法在不破坏模型输入语义的情况下定义，标记：

```text
NOT_APPLICABLE_TO_ARCHITECTURE
```

不得用任意 hack 强行补齐。

## 10.5 Context-shuffle：Primary controlled diagnostic

`context-shuffle` 作为 D2 更强的主要 controlled diagnostic。

必须在预冻结 matching strata 内进行，例如按以下变量分层：

- scenario family；
- lane-change direction；
- initial speed bin；
- traffic density bin；
- neighbor availability pattern；
- route/road geometry proxy；
- temporal event phase（若适用）。

shuffle 必须：

- 保持 slot-valid / missingness 边际分布；
- 保持每个 semantic slot 的 presence rate；
- 不跨明显不同 task family 打乱；
- 使用固定 seed；
- 按 independent unit 避免同 log 内伪独立扩增。

## 10.6 Leakage variables

至少包括：

- scenario/log identity；
- map/location；
- route/road geometry proxy；
- initial speed bin；
- traffic density；
- lane-change direction；
- neighbor availability pattern；
- dataset source。

不要求 scenario prediction 完全随机，因为 behavior 与场景天然相关。

核心问题是：在 behavior/context matching 后，route/log/map 等变量是否仍以超过预冻结 reasonable baseline 的能力主导 representation geometry。

## 10.7 OOD 诊断指标

每种 ablation 必须至少报告：

- embedding L2 norm distribution shift；
- distance to training embedding centroid；
- PCA subspace reconstruction error 或等价 OOD proxy；
- nearest-neighbor distance to training embedding bank；
- feature-level normalized range violation rate。

若 ablation 的 OOD shift 超过预冻结上限，则 scientific interpretation 降级为：

```text
ABLATION_OOD_DOMINATED
```

## 10.8 D2 输出

```text
execution_status
hypothesis_results
module_summary
```

D2 至少预注册：

```text
D2_RESPONSE_SENSITIVITY
D2_CONTEXT_SENSITIVITY
D2_PAIRING_SENSITIVITY
D2_SHORTCUT_RISK
D2_ABLATION_OOD_RISK
```

可能的 `module_summary`（可由多个 hypothesis_result 综合）：

- `RESPONSE_SENSITIVE`
- `CONTEXT_SENSITIVE`
- `CONTEXT_SHORTCUT_RISK`
- `EGO_CONTEXT_PAIRING_SENSITIVITY`
- `ABLATION_OOD_DOMINATED`
- `MIXED`

Stage6S-v3 只可作为 development diagnosis，不形成 RBR-V2 prospective interaction claim。

在 Waymo 等自然驾驶数据中，`EGO_CONTEXT_PAIRING_SENSITIVITY` 仅表示 representation 对 ego trajectory 与 traffic context 的匹配关系存在可测敏感性，不证明“ego 因某辆车而响应”的因果机制。`INTERACTION_INCREMENT_SUPPORTED` 仅保留给受控 planner treatment 或 future prospective intervention。

---

# 11. D3 — Measurement Readout Audit

## 11.1 科学问题

当行为信息存在于 z64 时，当前 full64 MMD 是否因 nuisance variance、kernel geometry 或 task dilution 而无法有效检测？

## 11.2 三层对象必须分离

严格区分：

```text
Representation z64
Measurement readout P_task
BDD statistic / null calibration
```

任何 readout 提升都只证明 task-conditioned measurement 可能更有效，不能单独证明 64D 是更好的通用 representation。

## 11.3 Readout candidates

Primary BDD/readout candidate family：

1. `R_full64`：现有 full64 BDD；
2. `R_linear_task`：development-train 拟合的低秩线性 task readout；
3. `R_fixed_semantic`：由预定义 target family 训练的受限 readout；
4. `R_multikernel`：预定义 kernel family 的 multi-kernel BDD。

Secondary non-BDD mechanistic comparator：

- `R_event_stat`：预定义 event-level paired directional statistic。它用于解释/对照，不要求经过 z64，因此不参与“full64 vs projected64 哪个 readout 更好”的 Primary measurement 比较，也不用于证明 representation superiority。

禁止：

- 在 nuPlan/Stage7L outcomes 上选择 projection dimension；
- 在 64 维中逐维扫 p-value 后挑显著方向；
- 高容量 nonlinear readout作为 Primary；
- 在 null calibration set 上训练 readout；
- 先看 audit-holdout outcome 再改 kernel/bandwidth/rank。

## 11.4 Projection rank 合同

v1.0 前必须冻结：

- rank candidate set，例如 `{1,2,4,8,16}` 或更小集合；
- rank selection metric；
- selection dataset；
- tie-break rule；
- maximum rank；
- 是否允许 task-specific rank。

rank 只能在 R0_DEVELOPMENT 上选择。

R0_AUDIT_HOLDOUT 只评估一次选定 rank。

## 11.5 MMD kernel / bandwidth 合同

v1.0 必须明确：

- kernel type：如 RBF / Laplacian / predefined family；
- bandwidth estimation dataset；
- bandwidth estimation是否按 representation/readout 独立；
- bandwidth 是否可看 treatment labels；
- paired 与 unpaired 是否使用同一 kernel rule；
- kernel 参数冻结时点；
- null calibration asset 与 readout training asset 隔离规则。

推荐原则：bandwidth 从 treatment-label-blind development/reference data 估计，并在 audit holdout 前冻结。

## 11.6 Multikernel 与多重检验

若 `R_multikernel` 同时产生多个 candidate statistics：

- 必须在统计分析计划中声明它是：
  - 单一聚合 statistic，或
  - 多重 hypothesis family；
- 若为多重 family，必须预冻结 Holm/FDR/其他校正；
- 不允许事后选择最显著 kernel 作为唯一结果。

## 11.7 Null calibration 隔离

- paired same-scenario：representation/readout-specific pair-label swap/randomization null；
- unpaired release：representation/readout-specific A/A calibration null；
- readout training rows 不得与用于其 threshold/null tuning 的 validation calibration rows混用；
- audit holdout 不参与 readout训练或 bandwidth/rank 选择；
- 不跨 representation 比 raw MMD²；
- 不混 paired null 与 unpaired A/A null。

## 11.8 Projected vs full64 的统计单位与 CI

比较 `projected` 与 `full64` 时必须以同一 independent unit 构造 paired comparison，例如：

- same scenario pair；
- same log-cluster bootstrap replicate；
- same release resampling trial。

必须报告：

- paired effect difference；
- cluster-aware bootstrap CI；
- seed-wise consistency；
- null calibration/FPR change。

不得只比较两个点估计的 Z 大小。

## 11.9 主指标

- raw statistic（仅同 representation/readout 内解释）；
- null q95；
- statistic / null-q95 ratio；
- standardized Z；
- permutation/A/A p-value；
- detection；
- FPR；
- bidirectional minimum detection；
- sample-size curve；
- projected vs full64 paired effect + CI。

## 11.10 D3 输出

```text
execution_status
hypothesis_results
module_summary
```

D3 至少预注册：

```text
D3_FULL64_SIGNAL_DILUTION
D3_PROJECTED_READOUT_GAIN
D3_NULL_CALIBRATION_PRESERVED
```

其中 `D3_PROJECTED_READOUT_GAIN = SUPPORTED` 至少要求：

1. readout 在 development 上按预冻结规则选择；
2. 在 R0_AUDIT_HOLDOUT 上 relative-to-null sensitivity 改善达到最小效应；
3. FPR/null calibration 不恶化超过 gate；
4. A/B/C 至少 2/3 seed 方向一致，或跨两种 frozen learned representation 稳定；
5. 不依赖 Stage7L outcome-driven selection。

只有同时结合 frozen semantic information evidence，才能进一步判断 `D3_FULL64_SIGNAL_DILUTION` 是否被支持。若无 audit holdout，则 evidence level 只能标记 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。

---

# 12. D4 — Residual Benchmark Protocol

## 12.1 科学问题

能否建立一种可证伪 benchmark：在人工描述符和主要场景条件已达到预冻结等价标准后，仍存在**机制可确认**的时序/交互行为差异，从而测试 learned representation 的增量信息？

本协议不使用“可感知”作为当前主张。若未来论文需要 human-perceptual claim，必须增加独立盲态人工语义验证协议。

## 12.2 三个 residual family

### R-HLC — Hesitant Lane Change

- decisive：approach → commit → monotonic transition → settle；
- hesitant：approach → pause/retreat → re-approach → delayed recommit → settle。

### R-TSB — Two-stage Braking

- continuous braking；
- brake → release → second brake。

### R-IP — Interaction Probing

优先构造：

- context 相似；
- ego response policy 不同；
- 例如相似 target-lane rear closing pressure 下，wait-for-gap vs early gap acceptance。

## 12.3 三重资格门禁

每个 matched pair / matched set 必须同时满足：

1. **Descriptor equivalence**：`F_match` 在预冻结 equivalence margin 内；
2. **Context equivalence**：road/maneuver/speed/traffic/initial-state 等在预定义 matching 规则内；
3. **Mechanism difference**：`M_behavior` 明确确认 morphology/interaction difference。

不确定样本不得强制设为 hard negative。

## 12.4 Matching 与 Mechanism 必须分离

- `F_match`：用于控制人工描述符；
- `M_behavior`：用于确认残余机制。

若 `pre_commit_retreat_count` 被用于确认 hesitation mechanism，则不得同时把它作为“人工方法已经匹配”的核心 feature，再宣称 baseline 无法看见该机制。

## 12.5 Equivalence 不能由“无显著差异”证明

正式 equivalence 必须采用至少一种等价推断方式：

### 推荐 Primary

- TOST；或
- 双侧 CI 完全落入预冻结 equivalence interval。

### 辅助

- pairwise standardized tolerance；
- standardized mean difference；
- Mahalanobis distance；
- propensity/context matching diagnostics。

禁止使用：

```text
p > 0.05 therefore equivalent
```

作为等价证据。

## 12.6 Equivalence margin 来源

每个 `F_match` 的 margin 必须优先来自：

1. 物理意义 / measurement resolution；
2. 已知传感/计算误差；
3. 业务或行为容忍度；
4. 历史 reference variability。

Power analysis 只回答：

> 当前样本量是否足以检验该 margin？

Power analysis **不能决定什么叫“足够等价”**。

## 12.7 多个 F_match 的校正

当一个 family 同时要求多个 descriptor 等价：

- 必须预冻结 family-wise rule；
- 可采用 intersection-union 逻辑：所有关键 feature 的 CI 都落入各自 equivalence interval；
- 或预注册 multiplicity adjustment；
- 不能只挑通过的 feature 报告。

## 12.8 Cluster-aware inference

必须明确：

```text
independence_unit
bootstrap_cluster
matching_group
```

优先：

- episode/window 不作为独立单位；
- scenario pair 为 paired mechanism unit；
- log 作为 bootstrap cluster；
- 若 driver-level data，则 driver 为更高层 cluster。

## 12.9 “未见 morphology”三级定义

- `LABEL_UNSEEN`：训练未使用标签，但 SSL 可能见过相似轨迹；
- `SAMPLE_UNSEEN`：scenario/log disjoint；
- `MORPHOLOGY_FAMILY_UNSEEN`：整个机制 family / 参数区间从训练/开发资产隔离。

R4 至少应包含 `SAMPLE_UNSEEN`；强主张尽量要求至少一个 `MORPHOLOGY_FAMILY_UNSEEN` family。

## 12.10 人类语义验证：可选增强，不属于当前 R0 核心门禁

若后续要使用“hesitation 被人类感知/可辨认”等措辞，则另建 protocol，至少包括：

- blind pairwise or categorical human rating；
- rater number；
- randomization；
- confidence；
- inter-rater reliability，例如 Krippendorff’s α；
- disagreement handling；
- 与 `M_behavior` 的一致性分析。

在该协议完成前，论文只能使用：

```text
mechanism-confirmed
mechanistically distinct
```

而不能使用：

```text
human-perceptible
perceptually distinct
```

## 12.11 Natural morphology 与 controlled planner treatment 的确认边界

R0/R1 必须把两类 confirmation source 分开管理，禁止用同一套 post-hoc 样本筛选逻辑处理。

### A. Natural behavior / morphology benchmark

对于自然驾驶或自然发生的 morphology 数据：

- 在 mechanism / matching / equivalence 规则冻结后，允许使用机制标签确定 contrast；
- episode / pair 的纳入必须仅依据预注册的 `F_match`、context matching、`M_behavior`、runnability、independence-unit 与 completeness 规则；
- 不得依据任何 representation、probe、BDD、classification 或 detection outcome 进行保留/删除；
- 若 mechanism label 本身来自人工规则，必须与 Primary generic residual SSL 的训练监督隔离，并在论文中明确其为 benchmark qualification label，而非“模型自主发现”的证据。

### B. Controlled planner treatment benchmark

对于由 planner parameter / software treatment 主动生成的受控 confirmation：

1. **confirmation roster 必须在运行 treatment rollout 之前冻结**，只能使用 pre-treatment scenario/context、历史 runnability、预注册 eligibility 与 power 信息；
2. rollout 完成后，必须对**整个 frozen roster**执行 treatment mechanism gate；
3. 禁止因为某些 scenario 的 realized mechanism effect 较弱、方向不理想或不显著，而删除这些 scenario 后再对 survivor subset 进行 Primary embedding / BDD confirmation；
4. Primary confirmatory estimand 采用 **whole-frozen-roster / intention-to-evaluate** 原则；
5. mechanism-success subset 若预先定义，可作为 Secondary descriptive / sensitivity analysis，但不得替代 Primary whole-roster 结论，也不得用于重新选择 readout、threshold、projection 或模型；
6. rollout 技术失败、日志损坏或缺测必须按 rollout 前冻结的 failure/missingness policy 处理；不得把“机制效果弱”重新解释成技术失败；
7. mechanism gate 的作用是判断 treatment 是否被整体实现，以及解释 representation outcome；不是 post-hoc 形成更容易检测的 confirmation set。

因此 future controlled treatment 的最小时间顺序必须是：

```text
pre-treatment eligibility / roster freeze
        ↓
treatment rollout
        ↓
whole-roster mechanism gate
        ↓
whole-roster Primary representation / BDD confirmation
        ↓
pre-specified Secondary mechanism-success sensitivity (optional)
```

若该顺序被违反，则相应结果必须降级为：

```text
POST_HOC_DEVELOPMENT_EVIDENCE
```

不得作为 R4 prospective confirmation。

## 12.12 Outcome-blind roster selection contract

`R0_AUDIT_HOLDOUT`、`FUTURE_R4_RESERVED_POOL` 与 `FUTURE_R4_CONFIRMATION_ROSTER` 的纳入/排除只能依据：

- pre-treatment scenario/context；
- descriptor matching / equivalence；
- mechanism confirmation；
- runnability / completeness；
- independent-unit / class-balance requirements；
- 预注册 statistical power / non-degeneracy criteria。

禁止依据任何 old64/A/B/C/ego13/RBR 的 embedding、probe、BDD、AUC、detection 或 relative-to-null outcome 进行场景筛选。

Development benchmark 可以报告预声明 baseline 结果用于理解任务，但这些结果不得反向改变 audit/future roster。

## 12.13 D4 输出

```text
execution_status
hypothesis_results
module_summary
```

D4 至少逐 family 记录：

```text
D4_DESCRIPTOR_EQUIVALENCE_<FAMILY>
D4_MECHANISM_DIFFERENCE_<FAMILY>
D4_OUTCOME_BLIND_FEASIBILITY_<FAMILY>
```

`RESIDUAL_BENCHMARK_FEASIBLE` 至少要求两个 residual family 在 development 中满足：

- descriptor/context equivalence；
- stable mechanism difference；
- 足够 independent units / log diversity；
- 预先锁定的 `R0_AUDIT_HOLDOUT` 中存在可按同一 outcome-blind 规则评估的对应资产；
- 已锁定或可在正式 RBR training 前锁定 `FUTURE_R4_RESERVED_POOL`；
- benchmark 非退化：机制效应存在、类别比例可用、独立单位足够、runnability 可接受、预期统计功效达到预冻结最低要求。

禁止使用 old64/A/B/C/ego13/RBR 的 embedding、BDD、probe、classification 或 detection 结果决定 `R0_AUDIT_HOLDOUT` / `FUTURE_R4_RESERVED_POOL` / `FUTURE_R4_CONFIRMATION_ROSTER` 的保留与删除。development 阶段允许报告预声明 baseline 结果用于理解 benchmark，但不得将“模型表现难度适中”作为 audit/future roster 的选择条件。

若 interaction family 暂不成熟，可标 `RESIDUAL_BENCHMARK_PARTIALLY_FEASIBLE`，不阻塞 temporal-residual 主线，但不得掩盖 interaction 未成熟事实。

---

# 13. D5 — External Asset Feasibility Audit（非阻塞）

## 13.1 候选数据集

- DriveDNA；
- Person2Drive；
- StyleDrive。

## 13.2 每套资产审核项

- official source；
- license / research-use restriction；
- download/access requirement；
- driver identity；
- repeated route/condition；
- sampling rate；
- ego state；
- lane/map/context；
- neighbor tracks；
- maneuver/event annotations；
- missingness；
- dataset-source shortcut risk；
- canonical schema compatibility；
- estimated storage / preprocessing cost。

## 13.3 D5 结果

每套数据独立给：

- `ADOPT_PRIMARY_EXTERNAL_VALIDATION`
- `ADOPT_SECONDARY`
- `DEFER`
- `REJECT`
- `INCONCLUSIVE`

D5 原则上不阻塞 RBR core design。

---

# 14. R0 全局 anti-leakage / anti-cheating 规则

1. 已解盲 Stage6/Stage7/Stage7L 只用于 R0_DEVELOPMENT diagnosis / method development。
2. Waymo test 已在 Stage6V 解锁，降级为 historical held-out development evidence。
3. `R0_AUDIT_HOLDOUT` 必须在 threshold/margin/readout 关键选择冻结前锁定。
4. `FUTURE_R4_RESERVED_POOL` 必须在正式 RBR training 前锁定数据源/token pool/生成规则。
5. `FUTURE_R4_CONFIRMATION_ROSTER` 只能在 R1 规则稳定后，从 reserved pool 按 outcome-blind 规则形成。
6. `FUTURE_R4_CONFIRMATION_ROSTER` 不得用于 R0/R1/RBR-A/B/C development。
7. RBR-V2 最终 R4 必须使用新的 scenario/log-disjoint prospective roster。
8. R4 roster 在 model/readout/kernel/threshold/margin freeze 前不得读取 RBR outcomes。
9. 不允许因为 Stage7L 已知 failure 逐维搜索 B/C latent 后将最佳维度定义成 future Primary。
10. projection/readout 的训练、val选择、维度、regularization与SHA必须独立记录。
11. probe target 不得来自 FUTURE_R4_CONFIRMATION_ROSTER labels。
12. residual hard-negative 不确定样本保持 unlabeled。
13. external controls 如 steering/gas/brake 不进入 Primary universal model，除非后续独立 protocol 明确批准。
14. 所有 random split / matching / shuffle / bootstrap / permutation seeds 写入 manifest。
15. 任何新增统计口径必须先在 R0_DEVELOPMENT 冻结，再接触 R0_AUDIT_HOLDOUT。
16. 若无 R0_AUDIT_HOLDOUT，结论必须标 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。
17. 任何 protocol deviation 必须在首次发现时写入 `r0_protocol_deviation_log.csv`，不得事后静默修补。
18. audit/future benchmark roster 的纳入/排除只能依据 pre-treatment/context/matching/equivalence/mechanism/runnability/power 规则，禁止依据任何 representation 或 baseline outcome 进行筛选。

---

# 15. R0 Closure 与 Candidate-specific Training Authorization

R0 的“行政闭环”与 RBR-A/B/C 的“训练授权”是两个独立判断。`BLOCKED` 不能因为被记录下来就自动满足训练授权。

## 15.1 R0 closure deliverables

R0 必须对六个模块形成正式记录：

1. `TEMPORAL_DECISION`
2. `INFORMATION_DECISION`
3. `CONTEXT_LEAKAGE_DECISION`
4. `MEASUREMENT_DECISION`
5. `RESIDUAL_BENCHMARK_DECISION`
6. `EXTERNAL_ASSET_DECISION`

每个模块至少包含：

```text
execution_status
hypothesis_results
module_summary
primary_evidence
secondary_evidence
evidence_level
limitations
next_action
blocks_rbr_training
fallback_id
```

一个 `BLOCKED` 模块只有同时满足以下条件，才可视为“不阻塞 R0 行政闭环”：

```text
blocks_rbr_training = false
and
a pre-frozen bounded fallback exists
```

这不等于该模块科学问题已经解决。

## 15.2 Candidate-specific authorization matrix

| 模块状态 | RBR-A | RBR-B | RBR-C | 解释 |
|---|---|---|---|---|
| D1 `BLOCKED` | 禁止 | 禁止 | 禁止 | information-retention 核心前提未知 |
| D4 `BLOCKED` | 禁止正式训练 | 禁止正式训练 | 禁止正式训练 | 无可用 qualification benchmark，无法防止 outcome-driven development |
| D0 `BLOCKED` | 可在其他核心门禁满足时讨论/授权 | temporal 方案不得锁定，原则上不授权正式 RBR-B | 不因 D0 单独决定 | bounded fallback 只能支持 A 类 objective repair |
| D2 `BLOCKED` | 可按其他门禁判断 | 可按其他门禁判断 | 禁止正式 RBR-C | conditional interaction 未获得可解释开发依据 |
| D3 `BLOCKED` | encoder/objective 训练可另行判断 | encoder temporal 训练可另行判断，但 task readout 不得设为 Primary | 同左 | measurement primary 方案未解决 |
| D5 `BLOCKED/DEFER` | 不阻塞 | 不阻塞 | 不阻塞 | external audit 非核心门禁 |

## 15.3 解除 `RBR_TRAINING_NOT_AUTHORIZED` 的最低共同条件

任何正式 candidate training 前至少要求：

- R0 protocol v1.0 已冻结；
- R0 data-tier split 与 `FUTURE_R4_RESERVED_POOL` 已冻结；
- D1 与 D4 均非 blocking `BLOCKED`；
- R0 final report 已生成；
- candidate-specific matrix 对该 candidate 显示 `AUTHORIZED`；
- R1 residual benchmark development 范围与 future reserved/confirmation 隔离方案明确；
- future reserved pool / confirmation roster 未被当前 R0/RBR development outputs 污染。

在这些条件满足前：

```text
RBR_TRAINING_NOT_AUTHORIZED
```

持续有效。


## 15.4 Machine-readable candidate training authorization

R0 finalization 不得只依赖 authorization matrix 的人工阅读。最终必须生成机器可审计文件：

```text
r0_training_authorization_manifest.json
```

顶层必须直接输出：

```text
RBR_A_TRAINING_AUTHORIZATION
RBR_B_TRAINING_AUTHORIZATION
RBR_C_TRAINING_AUTHORIZATION
```

取值仅允许：

```text
AUTHORIZED
NOT_AUTHORIZED
CONDITIONALLY_AUTHORIZED_WITH_FROZEN_FALLBACK
```

每个 candidate 的 authorization record 至少绑定：

```text
protocol_sha
decision_table_sha
asset_inventory_sha
contract_inventory_sha
split_manifest_sha
sap_sha
target_definition_sha
fallback_id
authorization_rationale
authorized_at_utc
authorized_by_protocol_version
```

其中：

- `AUTHORIZED`：所有该 candidate 的 blocking gates 已满足；
- `NOT_AUTHORIZED`：至少一个 blocking gate 未满足或关键资产/协议未冻结；
- `CONDITIONALLY_AUTHORIZED_WITH_FROZEN_FALLBACK`：仅当对应 blocking condition 已在 v1.0 中预先定义 bounded fallback，且 fallback 本身不会利用 audit/future outcome 时允许。

任何 authorization record 若缺少上述 SHA 绑定，自动视为：

```text
NOT_AUTHORIZED
```

授权 manifest 的 SHA 自身必须被写入 `r0_protocol_frozen.json`。

# 16. 计划输出目录与复现文件

统一目录建议：

```text
outputs/r0_representation_measurement_audit_v1/
```

## 16.1 必须产出

```text
r0_asset_inventory.csv
r0_split_manifest.csv
r0_target_definition.json
r0_statistical_analysis_plan.json
r0_environment.json
r0_command_ledger.jsonl
r0_protocol_deviation_log.csv
r0_protocol_frozen.json
r0_training_authorization_manifest.json

r0_temporal_contract_audit.csv
r0_temporal_orthogonal_experiment_metrics.csv
r0_temporal_content_window_descriptive_metrics.csv
r0_semantic_probe_metrics.csv
r0_cross_domain_probe_metrics.csv
r0_latent_geometry_metrics.csv
r0_context_leakage_probe_metrics.csv
r0_context_ablation_metrics.csv
r0_context_ablation_ood_metrics.csv
r0_context_shuffle_metrics.csv
r0_measurement_readout_metrics.csv
r0_measurement_null_calibration.csv
r0_kernel_bandwidth_audit.csv
r0_projection_rank_selection.csv
r0_residual_candidate_inventory.csv
r0_residual_matching_quality.csv
r0_residual_equivalence_tests.csv
r0_external_asset_audit.csv
r0_decision_table.csv
r0_manifest.json
r0_final_report_zh.md
```

## 16.2 附加图

```text
fig_r0_temporal_length_effect.*
fig_r0_temporal_position_retention.*
fig_r0_temporal_pooling.*
fig_r0_temporal_content_views.*
fig_r0_probe_transfer.*
fig_r0_latent_spectrum.*
fig_r0_context_response_controls.*
fig_r0_ablation_ood.*
fig_r0_readout_vs_full64.*
fig_r0_residual_matching.*
fig_r0_residual_equivalence.*
```

---

# 17. r0_decision_table.csv 建议字段

```text
decision_id
module
hypothesis_id
question
execution_status
hypothesis_result
module_summary
evidence_level
primary_evidence
secondary_evidence
minimum_effect_rule
null_or_reference
independence_unit
group_split_unit
bootstrap_cluster
permutation_unit
seed_consistency
limitations
next_action
blocks_rbr_training
reviewed_by
protocol_version
code_sha
asset_manifest_sha
split_manifest_sha
sap_sha
```

机器表要求：每个 `hypothesis_id` 单独一行；`module_summary` 可在同模块各行重复或另存 module summary 表，但不得只保留模块总结果。

---

# 18. R0 Statistical Analysis Plan 必须冻结的合同

`r0_statistical_analysis_plan.json` 至少包括：

## 18.1 共通字段

- alpha；
- confidence level；
- multiple-testing family；
- multiplicity correction；
- independence unit；
- group split unit；
- bootstrap cluster；
- permutation unit；
- bootstrap repeats；
- permutation repeats；
- fixed seeds；
- missing-value policy；
- outlier policy；
- effect-size definition；
- CI method；
- evidence-level rule。

## 18.2 D0

- orthogonal experiment definitions；
- length matching rule；
- position bins；
- pooling set；
- Probe-A / Probe-B contract；
- temporal minimum effect threshold；
- seed-consistency rule。

## 18.3 D1

- semantic target list；
- probe model family；
- regularization grid；
- primary metric by target；
- minimum interpretable threshold；
- cross-domain degradation metric。

## 18.4 D2

- ablation value definition；
- mask/missingness contract；
- OOD metric；
- OOD acceptable boundary；
- context-shuffle matching strata；
- leakage acceptable margin。

## 18.5 D3

- kernel family；
- bandwidth estimation rule；
- bandwidth dataset；
- projection rank candidate set；
- rank selection metric；
- maximum rank；
- multikernel testing family；
- null calibration data isolation；
- projected/full64 paired comparison unit；
- calibration/FPR gate。

## 18.6 D4

- F_match list；
- per-feature equivalence margin；
- margin rationale；
- TOST/CI rule；
- multi-feature equivalence rule；
- matching method；
- mechanism definition；
- minimum independent units/logs/scenarios；
- cluster-aware bootstrap rule。

---

# 19. R0 Protocol v1.0 前必须补齐的数值或操作参数

v0.3 仍不伪造以下数值；正式 v1.0 冻结前必须在 R0_DEVELOPMENT 上完成方法开发与 power/variance estimation，并冻结：

- D0 temporal minimum effect threshold；
- D0 early/middle/late position boundaries；
- probe ridge/regularization candidate grid；
- D1 semantic target minimum interpretable thresholds；
- D2 leakage acceptable margin；
- D2 ablation OOD boundary；
- D3 projection rank candidate set与 maximum rank；
- D3 kernel/bandwidth estimation rule；
- D3 calibration/FPR gate；
- D4 各 `F_match` equivalence margins；
- D4 minimum independent units/logs/scenarios；
- multiple-testing family与校正方案；
- bootstrap/permutation重复次数；
- fixed seeds。

## 19.1 关于 R4 release monitoring noninferiority margin `δ_NI`

R0 **只冻结以下内容**：

- `δ_NI` 的数学定义；
- 它比较的性能量；
- 估计方法；
- selection principle；
- CI method；
- 允许参考的 historical/development evidence；
- 明确禁止使用 FUTURE_R4_CONFIRMATION_ROSTER outcome 选择 `δ_NI`。

R0 **不冻结最终数值**。

最终 `δ_NI` 数值应在：

1. RBR architecture/readout 已定；
2. R4 sample design 已定；
3. R4 statistical power 已评估；
4. FUTURE_R4_CONFIRMATION_ROSTER 未解盲；

之后，于 R4 authorization 前正式冻结。

---

# 20. R0 之后的决策树

## 若 D1 = KNOWN_SEMANTICS_WEAK_OR_ABSENT

优先推进 RBR-A：objective / information-retention repair。

## 若 D1 = KNOWN_SEMANTICS_PRESENT，且 D3 = TASK_SIGNAL_DILUTION_SUPPORTED

优先推进受限 task readout 与 RBR-B temporal/measurement design；不得把 encoder failure 当成既定结论。

## 若 D0 支持 temporal effect

RBR-B 必须纳入至少两种受控 temporal solution 对比；R0 不提前指定 winner。

## 若 D2 module_summary = CONTEXT_SHORTCUT_RISK

RBR-C 必须采用 context-response separation，并把 native branch ablation / shuffle gate 列为必要资格门禁。

## 若 D2 module_summary = ABLATION_OOD_DOMINATED

不得据 Gen-1 zero-ablation 做强机制结论；RBR-C 设计需原生支持信息源隔离。

## 若 D4 module_summary = RESIDUAL_BENCHMARK_FEASIBLE

进入 R1 benchmark development；保留既有 `R0_AUDIT_HOLDOUT` 的已完成角色，不再新建 R0 holdout。按照 R0 阶段已锁定的 `FUTURE_R4_RESERVED_POOL`，在 R1 mechanism/matching/runnability 规则稳定后，prospectively 形成 `FUTURE_R4_CONFIRMATION_ROSTER`，且形成过程不得读取 RBR outcome。

## 若 D4 module_summary = RESIDUAL_BENCHMARK_NOT_FEASIBLE

优先补充受控 synthetic/planner residual treatment 或采用通过 D5 审计的 external asset；不得降低 matching/mechanism/equivalence 标准以强行通过。

---

# 21. RBR-V2 后续候选（R0阶段仅保留定义，不授权训练）

### RBR-A — Objective Repair

以 Generation-1 backbone 为基础，测试 generic residual SSL + semantic retention + balanced objectives 是否足够。

### RBR-B — Temporal Representation

RBR-A 基础加入由 D0 支持的 multi-scale / event-aware / pooling temporal design 与低容量 task readout。

### RBR-C — Conditional Interaction

RBR-B 基础加入 context encoder、response encoder及可检验 conditional coupling；训练时必须原生支持 full / ego-only / context-only / neighbor-ablated / shuffle 等控制，尽量减少 Gen-1 式 OOD ablation 的解释问题。

关键消融：

- shared64 + soft readout vs hard4×16；
- generic residual SSL vs + mechanism-guided hard negative。

---

# 22. RBR-V2 最终 Dominance 逻辑

R0 只冻结门禁类别、定义和未来统计设计原则，不冻结最终数值门槛。

只有在 FUTURE_R4_CONFIRMATION_ROSTER 中同时满足主要能力门禁，才允许写：

> `RBR-64 demonstrates superior overall behavior representation capability over ego13.`

门禁类别：

1. Known semantic information retention；
2. Known-treatment measurement noninferiority vs ego13；
3. Temporal residual superiority vs ego13 + extended handcrafted；
4. Interaction incremental value；
5. Release monitoring noninferiority vs B；
6. Leakage control。

R0 不要求 RBR 在每个物理单项指标上的 Z 都严格大于 ego13。

任何 task-conditioned semantic projection 的 BDD 提升，只能说明：

> task-conditioned measurement improved detectability under the frozen representation.

不能单独推出：

> the underlying 64D representation is universally superior.

---

# 23. Operational Freeze v1.0 的授权条件

v0.4 完成后，下一步仍不是训练模型，而是形成真正可执行的 **R0 Operational Freeze v1.0**。

必须完成：

1. 本地核对全部 `TO_BE_VERIFIED_IN_R0` 实现合同；
2. 建立 A/B/C 三 seed 的 asset inventory + SHA；
3. 冻结 R0_DEVELOPMENT / R0_AUDIT_HOLDOUT；
4. 锁定 FUTURE_R4_RESERVED_POOL 的数据源/token pool/生成规则，并明确 R0_AUDIT_HOLDOUT 是否存在；
5. 完成 development power/variance estimation；
6. 填写第19节数值/操作参数；
7. 冻结 D0-A controlled/descriptive classification、D0-B matched quasi-experimental design 与 D0-C orthogonal pooling design；
8. 冻结 D2 OOD-ablation contract 与 context-shuffle strata；
9. 冻结 D3 kernel/bandwidth/rank/readout/null calibration contract；
10. 冻结 D4 equivalence margin rationale + TOST/CI + cluster inference；
11. 固定 `r0_statistical_analysis_plan.json`；
12. 固定 `r0_split_manifest.csv`；
13. 固定 `r0_target_definition.json`；
14. 固定 environment / command ledger / deviation-log schema；
15. 生成并校验 `r0_training_authorization_manifest.json`，初始状态在 R0 audit 完成前必须为 `NOT_AUTHORIZED`；
16. 生成 `r0_protocol_frozen.json`，并写入 authorization manifest SHA；
17. 才允许正式执行 R0 audits。

在第15步完成以前：

```text
RBR_TRAINING_NOT_AUTHORIZED
```

保持有效。

---

# 24. v0.4 参数化准备状态

本协议当前建议状态：

```text
RBR_DIRECTION_FROZEN
R0_SCIENTIFIC_SCOPE_FROZEN
R0_PROTOCOL_V0_2_METHOD_REVIEW_PASSED
R0_PROTOCOL_V0_3_METHOD_AND_OPERATIONAL_REVIEW_PASSED
R0_PROTOCOL_V0_4_PARAMETERIZATION_PREP_DRAFT
R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
RBR_TRAINING_NOT_AUTHORIZED
```

v0.4 继承 v0.3 已通过的 method/operational review，并在此基础上补齐 natural-vs-controlled confirmation selection 与机器可审计训练授权输出。此前 v0.3 已消除的主要操作歧义包括：

- 单值模块科学结果无法表达多假设并存；
- `BLOCKED` 与训练授权混淆；
- R0 audit holdout 与 future R4 roster 的时间顺序冲突；
- D0-B 的准实验被过度称为正交；
- natural-data pairing sensitivity 被过度解释为 coupling/interaction；
- non-BDD event statistic 与 representation readout 层级混杂；
- benchmark difficulty 可能被模型 outcome 反向筛选。

仍需在 v1.0 前补齐的核心不是新的科学方向，而是：

1. 实际资产与 checkpoint 合同核验；
2. R0_AUDIT_HOLDOUT 可用性确认；
3. 数值门槛与统计参数冻结；
4. 完整 SAP / split / target / environment / command / deviation 复现链。

完成本地合同核验、audit holdout/reserved pool锁定、数值参数与机器可读SAP填写后，R0 才可进入 v1.0 一次性冻结执行阶段。


# 25. v0.4 变更记录

相对 v0.3，本版只做冻结前操作边界增强，不改变科学主线：

1. 增加 natural morphology 与 controlled planner treatment 的不同 confirmation selection / mechanism-gate 规则；
2. 受控 treatment 引入 whole-frozen-roster / intention-to-evaluate Primary 原则，禁止按 realized mechanism 强弱 post-hoc 筛选 confirmation survivors；
3. 新增 `r0_training_authorization_manifest.json`，直接输出 RBR-A/B/C authorization，并绑定 protocol/decision/assets/split/SAP/target/fallback SHA；
4. 明确 authorization manifest 缺少必要 SHA 时自动 `NOT_AUTHORIZED`；
5. v0.4 仍为 parameterization-prep draft，不能简单重命名为 v1.0。
