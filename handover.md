# E2E-Evaluation 博士研究项目权威交接

> **状态：`CURRENT_RESEARCH_HANDOVER_UPDATED_FOR_THESIS_CLOSURE`**  
> 更新时间：2026-08-19 13:36（Asia/Shanghai）
> 仓库：`forwardxp-021/E2E-Evaluation`  
> 分支：`20260611_stage7_conclusion`  
> 本次更新前基线：`c901fb53316b06791fc628cd8415f888bb8cba60`
> 本文件更新提交：运行 `git log -1 --format='%H %s' -- handover.md` 获取（Git提交无法在自身内容中稳定保存自己的最终SHA）  
> 当前阶段：核心研究证据、模型训练和BDD报告体系保持冻结；论文写作为主，同时开放一个prospective Stage7L pure-lateral controlled validation作为最终横向证据补充。
> 核心状态：`CORE_EVIDENCE_AND_MODELS_FROZEN`
> 唯一开放例外：`STAGE7L_PROSPECTIVE_LATERAL_VALIDATION_OPEN`

本文件是当前项目状态、科学结论、冻结协议、关键资产和后续工作的**总入口**。旧的
Windows→Mac迁移记录、Pittsburgh下载过程和Stage5/6/7早期研发流水已经降级到末尾的历史背景，
不得再把其中的旧实时状态当作当前任务。

---

## 0. 给下一个 conversation / Work session 的启动指令

### 0.1 推荐阅读顺序

1. `AGENTS.md`
2. `handover.md`
3. `docs/stage7l_pure_lateral_technical_feasibility_audit_zh.md`
4. `docs/stage7l_pure_lateral_technical_feasibility_audit_v1.json`
5. `docs/phd_thesis_research_closure_blueprint_zh.md`
6. `docs/stage6v_one_time_blind_evaluation_report_zh.md`
7. `outputs/stage6w_stage6s_v3_final_v1/stage6w_stage6s_v3_report_zh.md`
8. `docs/unified_bdd_evaluation_matrix_style_report_card_zh.md`
9. `configs/unified_bdd_reporting_schema_v2.json`
10. `configs/standardized_fixed_dimension_bdd_protocol_v2.json`
11. `README.md`
12. `QUICK_REFERENCE.md`

其中两份Stage7L-A审计文件是Stage7L当前技术状态的权威来源。

启动时先执行：

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
```

> Stage6、A/B/C checkpoint和BDD reporting schema均保持冻结；不得默认重新训练、重新选择场景、
> 修改统计门槛、继续扩展Stage6或补齐N/A行为维度。当前唯一授权的新实验方向是Stage7L
> pure-lateral controlled validation。除Stage7L外，不默认启动任何新训练、新模型或新Stage6实验；
> Stage7L不得反向修改旧冻结结论。新session应首先理解已有证据并推进论文写作。

工作树长期包含大量未跟踪实验输出和一个既有tracked数据文件修改。不要运行`git reset --hard`、
`git clean`或批量删除outputs；先区分用户资产与当前任务修改。

---

## 1. 一分钟项目摘要

### 1.1 论文当前定位

> **Task-conditioned trajectory-level behavior drift evaluation framework for closed-loop planning policies**

论文不再定位为“提出一个新的GRU embedding模型”。核心问题是：

> 如何判断两个E2E/planning policy版本的驾驶行为是否发生漂移、漂移发生在哪些行为维度，
> 并区分controlled same-scenario attribution与production-style unpaired release monitoring。

研究对象和方法包括：

- **trajectory-level**：以完整时间窗轨迹而非单帧指标描述行为；
- **planning policy behavior**：比较planner/software release的可观测闭环行为；
- **behavior drift**：判断Target相对Behavior Reference发生了哪些分布变化；
- **learned representation**：将83D ego-neighbor上下文编码为64D行为表示；
- **BDD**：在固定task、representation与null下估计行为分布差异；
- **paired / unpaired**：分别回答同场景归因和异场景release监控问题；
- **task-conditioned evaluation**：按纵向、跟车、变道、interaction等固定维度报告；
- **Style Report Card**：把统计显著性与semantic delta方向并列输出。

### 1.2 当前一句话结论

项目已经建立并验证一套task-conditioned trajectory-level behavior drift评估框架：它能在official
nuPlan closed-loop同场景实验中确认受控planner行为变化，并能在异场景release条件下用A/A标定监控
版本漂移；新训练的learned64显著增强了release-level检出，但没有通过全部联合门禁，也没有证明
interaction context具有独立增量价值。

### 1.3 当前最终模型决策

```text
NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE
```

B可以作为当前最简单、最强的learned release-level engineering candidate讨论，但不是
universal/final validated representation。当前研究状态为：

```text
RESEARCH_EXPERIMENTS_CAN_BE_FROZEN_FOR_THESIS_WRITING
```

这是Stage7L启动前形成的研究收口判断；Stage7L是一个受限、单独预注册的补充验证，不重新打开
Stage6模型研发。

### 1.4 当前唯一开放实验：Stage7L

**名称**：`Prospective Controlled Pure-Lateral Lane-Change Execution Benchmark`

**科学问题**：在相同scenario、相同lane-change intent、相同target lane、相同initial state以及固定
canonical longitudinal progress生成规则下，仅改变横向execution profile时，BDD是否能够可靠检测已知
pure-lateral behavior drift？

根据Stage7L-A技术审计，当前状态为：

```text
PURE_LATERAL_TREATMENT_IMPLEMENTATION_NOT_YET_CLEAN
```

当前还没有Stage7L development、confirmation roster、confirmation rollout、embedding、BDD或scientific
result。**No Stage7L scientific result exists yet.**

正式定义：**Pure-lateral means that the treatment parameterization affects only the lateral
trajectory-generation channel, while canonical longitudinal route progress, initial state, scenario,
source lane, target lane, trigger and all longitudinal controller parameters are held fixed.**

closed-loop realized behavior仍可能出现小量纵向副作用，因此未来必须设置longitudinal nuisance gate；
pure-lateral不意味着所有纵向指标在数学上完全为零差异。

现有Stage7 changing-lane slice属于`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`：它证明changing-lane
场景中的planner行为分布存在变化，但尚未证明一个已知、prospective、pure-lateral execution treatment
可以被BDD稳定检测。Stage7L专门补充这一evidence gap；当前论文不能写“BDD已经通过prospective
pure-lateral controlled confirmation”。

---

## 2. 正式研究贡献

论文贡献应写成框架贡献，而不是某个64D网络“全面胜出”：

1. **问题定义**：定义closed-loop planning policy的trajectory-level behavior/style drift问题，明确
   Behavior Reference、Target、task、representation和evaluation mode。
2. **统一行为表示接口**：构建共享的`[N,150,83]` ego-neighbor trajectory context与64D learned
   behavior representation，使Waymo训练域与nuPlan闭环验证域使用一致输入合同。
3. **双评估协议**：提出task-conditioned BDD，并严格区分same-scenario paired attribution与
   production-style unpaired release monitoring两个estimand。
4. **发布监控方法**：建立log-disjoint、A/A-calibrated unpaired drift monitoring，联合报告
   detection、A/A FPR、双方向稳定性和样本量依赖。
5. **标准化报告与能力边界**：冻结13维Standardized BDD Evaluation Matrix与两层Style Report Card，
   系统呈现正结果和负结果，证明单一representation不必在所有estimand上统一最优。

最终贡献不是“某个64D模型在所有任务上全面胜出”，而是完整的behavior drift evaluation framework、
可审计的统计协议，以及该框架经过系统验证后的适用边界。

---

## 3. BDD必须如何理解

### 3.1 三个不同概念

#### Behavior Drift Profile

回答：**Target相对Behavior Reference在哪里发生了行为变化？**

#### BDD Statistic

统一写作：

```text
BDD(Target | Reference, task, representation, evaluation_mode)
```

每个BDD必须同时绑定Behavior Reference、Null Reference、representation和task。当前实现用kernel
Maximum Mean Discrepancy估计BDD，报告数值通常为biased `MMD²`。

#### Representation Evaluation

回答：**old64/A/B/C/ego13中，哪个representation更可靠地检测某个已知behavior treatment？**

它评价测量器，不评价planner本身，也不直接给出“更激进/更保守”的方向。

### 3.2 三类Reference

- **Behavior Reference**：哪个planner/version/release作为比较起点；semantic delta固定为
  `Target − Behavior Reference`。
- **Null Reference**：paired使用该representation自己的pair-label-swap/randomization q95；
  unpaired使用该representation自己的A/A calibration q95。`BDD/null-q95=1.0×`只是统计背景线。
- **Representation Baseline**：old64历史baseline，只用于比较检测能力，不定义行为方向。

禁止再使用模糊术语“reference BDD”。

### 3.3 不能如何解释BDD

- raw MMD²受embedding尺度、kernel bandwidth、样本量和实验设计影响；不得跨representation直接排序。
- BDD显著表示“分布差异相对null异常”，不表示Target更安全、更好或更差。
- 行为方向必须来自`Δspeed`、`Δfront gap`、`Δfinite THW`、`ΔRMS accel`等semantic delta。
- paired null与unpaired A/A calibration回答不同问题，不能混用。
- 不存在已验证的、跨ODD/任务/representation通用OEM BDD报警阈值。

---

## 4. 当前冻结证据链

### 4.1 Stage7 M6.5：official nuPlan controlled confirmation

M6.5使用新的log/scenario-disjoint确认集，最终得到：

```text
310 complete scenario pairs
620 official closed-loop rollouts
overall MMD² = 0.0044693963
paired p ≈ 1e-5
5/5 pre-treatment tasks pass Holm correction
```

五个task为following interaction、lane-change scenario slice、stop-go control、high-motion dynamics和
dense/vulnerable interaction。该结果支持：

> official nuPlan closed-loop controlled same-scenario validation已经成立；Waymo训练的表示能够检出
> 新log/scenario上的planner-conditioned behavior shift。

45-pair Balanced50数据只保留为method-development历史，不再作为当前主要确认性证据。M6.6发现
lane-assignment fallback与embedding pair distance有关，因此M6.5保留`PASS_WITH_QUALITY_LIMITATIONS`；
post-treatment质量不能用于删样本或修改310-pair primary。

### 4.2 Stage6J/K：纯纵向controlled treatment与dose curve

Stage6J固定两个planner的横向参数相同，只改变纵向IDM参数；183个相同场景、156个log、366条
official rollout全部成功。实现运动学差异为：

```text
Δ mean speed ≈ +0.915 m/s
Δ RMS acceleration ≈ +0.182 m/s²
old64 overall dose100 Z_BDD ≈ 9.23
```

Stage6K已经完成，不是“正在运行”：25/50/75/100%四档realized kinematic gate均通过，四档overall
BDD经统一Holm均显著。本冻结协议内overall最小可检出**名义剂量**为25%，但这不是通用BDD或物理阈值；
task-level结果存在明显异质性。

Stage6J/K构成后续old64/A/B/C/ego13 paired representation evaluation的冻结基础。

### 4.3 Stage6L / Stage6M / Stage6P：representation与release结果

#### Stage6L paired representation ablation

- ego13在纯纵向controlled treatment中具有最高within-null标准化敏感度；
- old64仍能显著检测纵向变化；
- neighbor-zero结果证明更高Z不能自动解释为“包含更多interaction信息”；
- ego13强不等于neighbor/context无用，因为当前treatment大量直接作用于ego运动学。

#### Stage6M context balancing

coarse map/scenario-type context balancing没有证明是n=400误差的主要来源。不能据此写成
“scenario heterogeneity整体不重要”；Stage6W显示log/scenario异质性仍然占重要背景成分。

#### Stage6P unpaired release monitoring

在原800-pair / 489-log / 2400 split框架中，n=400 context-balanced结果为：

| Representation | A/B detection | A/A FPR | 双方向最小检出率 |
|---|---:|---:|---:|
| old64 | 66.5% | 5.0% | 62% |
| A | 90.5% | 3.0% | 90% |
| B | 100.0% | 5.0% | 100% |
| C | 99.5% | 6.5% | 99% |
| ego13 | 100.0% | 2.0% | 100% |

当前工程解释：

> B是当前最简单、最强的learned release-level engineering candidate。

同时必须写：

> B不是universal/final validated representation，也没有通过全部Waymo、paired与联合门禁。

### 4.4 Dynamic Interaction Builder v2与A/B/C训练

Dynamic Builder v2修复旧Stage5D builder的参考帧静态slot分配与整窗高有效率过滤，使用逐帧semantic
slot assignment、track-id时间序列、identity switch防跨agent导数以及strict lane topology。最终full51：

```text
source TFRecords: 51
scenarios: 24,872
windows: 168,700
train / val / test: 135,046 / 16,870 / 16,784
scenario split overlap: 0
intermittent-following train windows: 63,415
```

lead entry/exit、intermittent following、front identity switch和following/free-flow transition均得到恢复；
acceleration/jerk监督采用平滑、winsorization和train-only robust normalization，物理噪声明显改善。
Stage6O-v2数据门禁已通过；旧Stage6O-v1永久保持blocked历史记录。

A/B/C定义：

- **A — Dynamic-data only**：Dynamic-v2数据 + legacy single-GRU topology/objective；
- **B — Longitudinal recovery**：同single-GRU topology + clean longitudinal objectives/ranking/sampling；
- **C — Interaction-aware dual branch**：与B相同数据、loss、sampling和预算，encoder改为ego/context双分支。

三者均输入83D、输出64D；primary seed在解盲前固定为`3407`，3408/3409只用于seed stability，
不得用secondary seed更好的结果替换primary。Stage6U共9个正式任务已经9/9完成并锁定：

```text
LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK
```

### 4.5 Stage6V：一次性盲测

最终状态：

```text
FROZEN_STAGE6V_ONE_TIME_BLIND_EVALUATION_COMPLETE
NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE
```

#### Waymo test

- A primary longitudinal能力退化；
- B/C有正改善；
- A/B/C primary seed均满足综合非劣性，但均未通过完整预冻结Waymo primary门禁；
- secondary seed结果不得替代primary 3407。

#### Stage6J/K paired

| Representation | overall dose | task×dose | 冻结paired门禁 |
|---|---:|---:|---|
| old64 | 4/4 | 7/12 | 未通过完整门禁 |
| A | 4/4 | 7/12 | 未通过 |
| B | 3/4 | 2/12 | 未通过 |
| C | 3/4 | 2/12 | 未通过 |
| ego13 | 4/4 | 12/12 | 通过 |

因此B/C没有恢复controlled narrow longitudinal task的完整paired sensitivity。

#### Stage6P unpaired

B/C达到100%/99.5% n=400 detection，是新learned64最强正结果之一，且跨seed稳定。

#### Stage6S-v2

冻结80-pair roster只完成61对，19个token被nuPlan official `valid_scenes`边界排除。这是
**confirmation roster construction / runnability failure**，不是模型机制失败；机制和embedding均未解锁，
不能把Stage6S-v2写成interaction negative result，也不能用61个complete cases事后重定义confirmation。

### 4.6 Stage6W-A：paired/unpaired分离机制

在相同Stage6P pool和相同n=400下：

```text
old64 paired median Z ≈ 13.50
B paired median Z ≈ 28.29
C paired median Z ≈ 25.37
```

这证明历史Stage6J/K中B/C较弱，不是paired statistic天然压低B/C，而主要来自treatment、task、
scenario pool和estimand不同。

context-balanced unpaired signal decomposition：

```text
B standardized signal ≈ 2.59× old64
C standardized signal ≈ 2.64× old64
B增益中signal贡献 ≈ 86%
C增益中signal贡献 ≈ 93%
```

结论：B/C接近100%的unpaired检出主要由更强、更一致的planner signal驱动；null variance下降只是次要因素。

### 4.7 Stage6S-v3：prospective interaction confirmation

Stage6S-v3只修复v2已知的pre-treatment official-runnability遗漏，其余planner、metrics、mechanism gates、
bootstrap和representation endpoint保持冻结。最终：

```text
official rollout: 80/80 succeeded
Δ mean speed: +0.289 m/s
Δ RMS acceleration: +0.150 m/s²
Δ median front gap: -4.202 m
Δ median finite THW: -2.670 s
front-gap / finite-THW / closing-accel / following-accel: 4/4 gates pass
```

机制通过后才解锁representation：

```text
C full Z_BDD ≈ 28.95
C neighbor-zero Z_BDD ≈ 36.81
ΔZ (full - neighbor-zero) ≈ -7.85
log-cluster bootstrap 95% CI ≈ [-33.39, 29.22]
```

正式结论：

> **interaction mechanism positive confirmation + C incremental context negative evidence**：planner-level
> interaction mechanism confirmation为positive；负结果仅是C full-context相对C neighbor-zero没有证明
> 显著incremental interaction sensitivity。

不能把它扩大解释为“interaction context整体无价值”。这只是当前模型、当前数据、当前冻结interaction
treatment下没有获得C的独立增量证据；不能写成interaction失败、context无价值或C完全没有interaction能力。

### 4.8 Stage7L-A：pure-lateral technical feasibility audit

Stage7L-A是技术可行性审计，不是实验结果。审计确认当前PDM的横向path、leading-agent识别、IDM纵向推进、
proposal simulation/scoring和argmax相互耦合；修改`lateral_offsets`或对最终trajectory做warp，都不能构造
论文级、因果解释洁净的pure-lateral treatment。因此Stage7L-A停止于technical audit，没有运行新仿真、
训练、embedding或BDD，冻结状态为：

```text
PURE_LATERAL_TREATMENT_IMPLEMENTATION_NOT_YET_CLEAN
```

这不是横向BDD实验失败，而是技术洁净性审计拒绝了一个因果解释不充分的实现方案。

未来A2的推荐方向是external `PureLateralExecutionPlanner`：使用canonical route/Frenet progress
`s_route(t)`，固定source lane、target lane、direction与trigger，以quintic/minimum-jerk `d(s_route)`生成
five-dose lateral execution；old64/A/B/C/ego13 representation保持固定，并使用same-scenario paired null。
核心原则是dose只能进入lateral trajectory-generation channel，不能通过PDM总aggressiveness或
`lateral_offsets`间接改变整个planner。

最终实验逻辑为：

```text
Pure longitudinal treatment
  → Stage6J/K
  → controlled longitudinal BDD

Pure lateral execution treatment
  → Stage7L [prospective / not yet completed]
  → controlled lateral BDD

Interaction/headway treatment
  → Stage6S-v3
  → controlled interaction BDD

Unpaired release emulation
  → Stage6P/W
  → production release monitoring
```

Stage7L目前是planned evidence，不能与另外三条已完成证据混为一谈。

---

## 5. old64 / A / B / C / ego13如何定位

| ID | 定义 | 论文中的定位 | 不能声称 |
|---|---|---|---|
| old64 | Stage5D-balanced-v2历史64D checkpoint | 冻结Representation Baseline；能检出controlled shift，但release可靠性有限 | 不是当前唯一主模型；raw MMD²不能跨表示比较 |
| A | Dynamic-v2数据 + legacy topology/objective | 隔离“修数据”贡献；unpaired显著提升 | 未通过联合门禁 |
| B | Dynamic-v2 + single-GRU + longitudinal recovery objective | 最简单、最强的learned release-level工程候选 | 不是universal/final validated representation |
| C | Dynamic-v2 + ego/context dual branch | 检验dual-branch额外价值 | 不能称已验证interaction-aware主模型 |
| ego13 | ego kinematic 13D reference | controlled longitudinal sensitivity参考上界/诊断基线 | 不能称全局最佳style representation，不能证明context无价值 |

primary A/B/C均使用seed 3407；任何未来论文表格不得按test或nuPlan结果换seed、换epoch或改checkpoint。

---

## 6. 最终统一BDD报告体系

最终状态：

```text
FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN
```

控制定义：

```text
unified_bdd_reporting_schema_v2_final
standardized_fixed_dimension_bdd_protocol_v2_final_render_only
```

固定13维taxonomy覆盖overall、纵向、横向和interaction。报告固定为两层。

### 6.1 第一层：Behavior Drift / Style Report Card

当前`Primary Representation = B`。B只是用于测量Behavior Reference→Target漂移的representation，
不是被评价的planner/version。当前摘要：

| Behavior dimension | BDD/null-q95 | Z_BDD | 证据身份 |
|---|---:|---:|---|
| Longitudinal acceleration/deceleration | 2.74× | 10.33 | Stage6J/K confirmatory |
| Car-following | 1.72× | 5.25 | Stage6J/K confirmatory；60 scenario / 52 log |
| Lane-change scenario slice | 2.50× | 9.12 | `POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`；60场景 |
| Interaction | 7.39× † | 30.60 | Stage6S-v3 confirmatory；80 pair / 11 log |

`†`表示Closing response、Front-gap/THW interaction和Longitudinal following interaction共享同一个
parent task-level BDD，不是三次独立BDD检验。

其中Stage7 changing-lane slice仍只是post-hoc描述性证据，不得替代未来Stage7L的prospective
pure-lateral confirmation。

### 6.2 第二层：Representation Qualification Matrix

old64/A/B/C/ego13只能按各自null下的standardized sensitivity、detection/FPR、task coverage和门禁比较，
禁止跨representation比较raw MMD²。

ego13在多个controlled treatment中Z最高，但这些treatment大量直接作用于ego kinematics，因此不能把
ego13解释为全局最佳behavior representation。learned64的最强正结果仍包括production-style unpaired
release monitoring，representation能力必须按deployment/evaluation task解释。

### 6.3 当前N/A维度

- free-flow speed；
- lane keeping；
- lateral gap interaction。

N/A表示没有符合冻结协议的证据，不表示没有行为差异。不要为了填满N/A继续实验。

权威报告：

```text
outputs/final_standardized_bdd_style_report_card_v1/
  final_standardized_bdd_style_report_card_zh.md
  final_behavior_style_report_card.csv
  final_fixed_dimension_primary_matrix.csv
  final_representation_qualification_matrix.csv
  final_shared_parent_bdd_audit.csv
  final_standardized_bdd_reporting_manifest.json
```

---

## 7. 当前论文claim boundary

### 7.1 当前可以写什么

- task-conditioned trajectory-level behavior drift evaluation框架成立；
- controlled paired attribution与production unpaired monitoring是不同estimand，不要求单一representation统一最优；
- official nuPlan closed-loop新log/scenario confirmation确认了planner-conditioned behavior shift；
- Waymo训练的表示可以检出经运动学确认的典型纯纵向nuPlan behavior treatment；
- Dynamic Builder v2与新训练目标显著增强learned64的release-level unpaired detectability；
- Stage6P n=400中old64 66.5%，A/B/C 90.5%/100%/99.5%，A/A FPR受控；
- Stage6W证明B/C提升主要来自signal增强，而不是主要依赖null variance下降；
- standardized BDD matrix能够按纵向、跟车、变道场景slice、interaction等固定维度输出可读报告；
- negative results明确了paired、Waymo与interaction context增量的representation能力边界。
- 已完成pure-lateral technical feasibility audit，并明确现有PDM不适合直接构造因果洁净的
  pure-lateral treatment。

### 7.2 当前不能写什么

- 不能写C是已验证的interaction-aware论文主模型；
- 不能写A/B/C通过全部预冻结joint gates；
- 不能写64D全面优于ego13，或ego13是全局最佳representation；
- 不能写neighbor/context无价值；
- 不能写BDD代表安全性、质量或planner优劣；
- 不能写BDD越大planner越差；
- 不能跨representation直接比较raw MMD²；
- 不能把Stage7 post-hoc lane-change矩阵写成原预注册confirmation，或写成ego已确认执行换道；
- 不能写Stage6S-v2是模型interaction失败；
- 不能写Stage7L已经完成，或横向BDD已经得到prospective confirmation；
- 不能写现有PDM的`lateral_offsets`代表lane-change execution style；
- 不能用Stage7 changing-lane slice替代Stage7L prospective evidence；
- 不能因为未来Stage7L结果不好而重新训练B/C；
- 不能声称存在通用OEM BDD报警阈值；
- 不能声称已经完成真实整车厂版本验证或达到任意ODD下的单次release可靠性保证。

---

## 8. 关键冻结状态清单

| Item | Status |
|---|---|
| Stage7 controlled confirmation | Frozen complete：310 pairs / 620 rollouts |
| Stage6J/K longitudinal dose | Frozen complete |
| Dynamic Builder v2 | Frozen；Stage6O-v2 readiness passed |
| A/B/C training | 9/9 locked；primary seed 3407 |
| Stage6V blind evaluation | Complete |
| Stage6W paired/unpaired diagnostic | Complete |
| Stage6S-v2 | Frozen execution failure due to roster runnability omission |
| Stage6S-v3 interaction confirmation | Complete；80/80；mechanism passed；C increment failed |
| Stage7L pure-lateral validation | Prospective；Stage7L-A technical audit complete；implementation not yet clean |
| Stage7L scientific result | None yet |
| Stage7L-B development | Not authorized yet |
| Unified fixed-dimension BDD matrix | Complete |
| Final BDD reporting system | Frozen |
| Stage6V joint candidate decision | `NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE` |
| New model training | Not planned / not authorized |
| Thesis writing | **Current priority** |

---

## 9. Persistent Technical Invariants / Do Not Break

这些约束具有长期工程与科研价值，不因论文进入写作阶段而失效：

1. **Stage5D context合同**：共享context shape为`[N,150,83]`，learned embedding为`[N,64]`。
2. **固定五邻车语义**：front / left_front / left_rear / right_front / right_rear；Dynamic-v2逐帧分配，
   semantic correctness优先于track continuity。
3. **identity switch导数**：不同agent之间不得计算accel、yaw-rate、closing derivative；必须reset/invalidate。
4. **rollout validity mask**：smoothing、导数、事件、轨迹指标和物理诊断前必须消费`ego_seq_mask.npy`。
5. **non-contiguous scenario axis**：成功场景轴不能假定连续；必须读取alignment/index manifest。
6. **lane cache作用域**：局部LaneInfo cache至少绑定canonical map与original scenario index，不能只按map复用。
7. **strict Waymo lane topology**：Dynamic-v2 semantic slots依赖严格拓扑与可审计fallback。
8. **pre-treatment selection**：paired roster必须在任何planner outcome、embedding、BDD之前冻结。
9. **禁止post-treatment过滤**：realized quality只能做描述性诊断，不能用于选择或删除primary pair。
10. **null不可混用**：paired label-swap/randomization null与unpaired A/A calibration不能互换。
11. **raw MMD²不可跨representation排序**：跨表示只能比较各自null标准化统计或检测能力。
12. **primary seed不可事后更换**：A/B/C primary固定3407，secondary seed只评价稳定性。
13. **大资产保护**：large outputs、checkpoint、nuPlan DB、maps和Waymo数据不得随意删除或提交Git。
14. **不覆盖冻结资产**：新研究若启动，必须新建协议、版本、checkpoint和confirmation，不得覆盖Stage5D、
    Dynamic-v2、Stage6V/W/S-v3或最终BDD报告。

---

## 10. 环境与数据资产（2026-08-19实际核验）

### 10.1 硬件与Python环境

```text
machine: MacBook Air, Apple M5, 10 cores, 16 GB RAM
architecture: arm64
waymo_dev: E2E-Evaluation/waymo_dev/bin/python
Python: 3.10.20（训练ledger记录）
PyTorch: 2.5.1
MPS: available
nuPlan Python: /Users/liuqing/miniconda3/envs/nuplan/bin/python
nuPlan Python version: 3.9.19
```

外部仓库已实际核验：

```text
nuPlan devkit: e9241677997dd86bfc0bcd44817ab04fe631405b
tuPlan Garage: b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
```

### 10.2 Pittsburgh / nuPlan数据

Pittsburgh DB-only archive曾完整下载并用于解压；当前ZIP已从本机删除以释放空间，不能再写成“仍在下载”。
实际存在的可用资产为：

```text
nuplan/dataset/data/cache/train_pittsburgh/       1560 .db files
nuplan/dataset/data/cache/locked_pool_expanded_v1/ 1621 flat symlinks
```

因此一般分析/复现不需要重新下载ZIP。只有解压DB损坏或必须从archive重建时，才重新获取
`nuplan-v1.1_train_pittsburgh.zip`。不要把DB、maps或ZIP提交Git。

### 10.3 关键checkpoint

```text
old64:
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/
  context_gru_stage5d_balanced_v2/best_model.pt
SHA256 909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc

A primary seed 3407:
outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3407/best_model.pt
SHA256 353982753f208d27d677c6863a681997b8e28b728573a52fa407807f6fd0298d

B primary seed 3407:
outputs/stage6t_candidates_v1/candidate_B_single_gru_recovery/seed_3407/best_model.pt
SHA256 d8e0de6e74ee29076082aabef27a425b47678e1372c630e4f4a04106ff34265f

C primary seed 3407:
outputs/stage6t_candidates_v1/candidate_C_dual_branch/seed_3407/best_model.pt
SHA256 cc6bf3c427534f66f74904c8948bf427cfe9f1152bba4bca0e8342f3fa47433d
```

完整9-checkpoint ledger：

```text
outputs/stage6u_abc_formal_training_v1/checkpoint_lock/
  stage6u_formal_checkpoint_ledger.json
  stage6u_formal_checkpoint_ledger.csv
```

### 10.4 关键冻结结果路径

```text
Dynamic-v2 data:
outputs/stage6r_dynamic_full51_semantic_strict_v1/

Stage6O-v2 readiness:
outputs/stage6o_v2_dynamic_training_readiness_v1/

Stage6V blind evaluation:
outputs/stage6v_one_time_blind_evaluation_final_v1/

Stage6W + Stage6S-v3 final:
outputs/stage6w_stage6s_v3_final_v1/

Final BDD reporting system:
outputs/final_standardized_bdd_style_report_card_v1/
```

这些目录以及对应rollout/context/checkpoint是科研provenance的一部分。磁盘清理前必须先确认是否可由Git或其他
资产恢复；不要把“未被Git跟踪”误解为“可以删除”。

---

## 11. Git与provenance

```text
branch: 20260611_stage7_conclusion
baseline before this handover update: c901fb53316b06791fc628cd8415f888bb8cba60
remote: origin/20260611_stage7_conclusion
PR: #265, OPEN DRAFT, large historical development PR
```

PR #265用于当前长期研发分支归档，不应被当作一份小而独立的单实验PR。

本handover的权威重构版本已由`c901fb53316b06791fc628cd8415f888bb8cba60`正式纳入仓库。本次仅修正
Stage7L开放后的当前状态；提交时必须只stage `handover.md`，不得顺带提交大型outputs、日志、数据或
既有工作树修改。

关键provenance SHA：

```text
Dynamic-v2 content signature:
e760605cd8fb57d4dfee68b8044d2ad31ec71e7e7b2f544d039172d001053905

Stage6S-v3 freeze:
7105940bd822f02d643ed4f5cb9a8321b3827ca6117be289914057e3fe8a26c6

Final BDD protocol:
fac9f04d479185b1ef3548c08bc782d2a3114de8595da482a1f418e58f698762

Final BDD schema:
1c0325dc6e25bbeb40bbbc69c0b90504a792f19dfd3624c715e8d1d4a908d33d
```

---

## 12. 当前下一步：论文写作为主 + 一个受控Stage7L补充验证

### Track A — Thesis writing

1. Method；
2. Results；
3. figures/tables；
4. Discussion；
5. Limitations；
6. standardized BDD matrix整理。

### Track B — Stage7L

Stage7L-A2、B/B2 development与Stage7L-C prospective protocol/80-scenario roster freeze均已完成。Stage7L-C1/C2已在任何confirmation结果产生前完成最终protocol consistency amendment：roster仍为80场景、15 left/65 right、79 logs，SHA为`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`；Primary pair下限为76，`LAT.LANE_CHANGE`等于完整roster membership，`LAT.DYNAMICS`使用冻结pre-treatment official type mask；secondary family为排除唯一Primary后的39-test Holm family。

当前状态：

```text
STAGE7L_C2_TASK_POPULATION_CONSISTENCY_AMENDMENT_FROZEN
STAGE7L_C1_PROTOCOL_CONSISTENCY_AMENDMENT_FROZEN
STAGE7L_C_PROSPECTIVE_PROTOCOL_FROZEN
STAGE7L_C_CONFIRMATION_ROSTER_FROZEN
STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED
STAGE7L_D_ONE_TIME_CONFIRMATION_IN_PROGRESS
```

Stage7L-D已获单独授权。统一可恢复runner、attempt ledger、机制/nuisance/safety/canonical identity门禁工具已完成代码级验证；正式400格运行按冻结roster order×五档确定性执行，不允许replacement。当前仍禁止embedding、checkpoint、BDD/MMD和任何训练；只有机器gate全部通过才可解锁Stage7L-E，且不得自动执行Stage7L-E。
Stage7L不得重新打开Stage6模型训练，也不得为了让BDD更显著而调representation。
**新实验 ≠ 重新训练模型。**

除这一受限例外外，当前不要再做Stage6K、训练v3、扩Stage6S或补齐全部N/A。

Stage7L启动前的研究收口判断保留为：

```text
RESEARCH_EXPERIMENTS_CAN_BE_FROZEN_FOR_THESIS_WRITING
```

---

## 13. Historical Background / Archived Development History

以下内容只用于理解项目演进，不是当前执行入口：

- 项目最初在Windows 11 + WSL Ubuntu上开发，2026-08迁移到MacBook Air M5；
- 迁移阶段曾处理Waymo outputs、nuPlan mini/maps、Pittsburgh DB下载、Mac arm64环境兼容和绝对路径；
- Stage7 M1–M2修复non-contiguous scenario axis、rollout validity mask、lane cache和地图投影；
- M3的45-pair Balanced50达到最低开发规模，M4/M5/M6完成统计、representation mechanism与paired BDD方法开发；
- M6.4通过扩展Pittsburgh inventory完成锁定采集，M6.5形成310-pair正式确认；
- Stage6D–I建立公开数据上的unpaired release emulation；
- Stage6J/K完成纯纵向确认与剂量曲线；
- Stage6Q/R修复Dynamic Builder，Stage6T/U完成A/B/C协议、训练和checkpoint锁定；
- Stage6V完成一次性盲测；Stage6W解释paired/unpaired分离；Stage6S-v3完成prospective interaction确认；
- 最终工作从“模型开发”转向“task-conditioned behavior drift framework与claim boundary”。

下列旧状态已明确归档，不得恢复成当前任务：

- “Stage6K正在运行/等待完成”；
- “Pittsburgh仍在下载”；
- “M6.4尚未启动”；
- “Stage5D-only是唯一当前主模型”；
- “下一步应马上扩大Waymo或训练v3”；
- Windows旧机仍需继续迁移；
- 任何已经结束的后台下载、rollout或训练ETA。

若需要考古完整迁移过程，旧文件仍保留在`/Users/liuqing/Downloads/handover.md`；它不是当前权威状态。

---

## 14. 最后检查清单

新的session在采取任何写操作前，应能回答：

1. 论文主线是behavior drift evaluation framework，而不是新GRU模型吗？
2. Behavior Reference、Null Reference和Representation Baseline是否分开？
3. 当前结论来自paired还是unpaired estimand？
4. old64/A/B/C/ego13的角色是否清楚？
5. Stage6V为何没有候选通过联合门禁？
6. Stage6S-v2为何是runnability failure，而Stage6S-v3应写成interaction mechanism positive confirmation
   + C incremental context negative evidence？
7. 为什么ego13高Z不等于全局最佳representation？
8. 为什么B是release工程候选但不是最终主模型？
9. 哪些实验、checkpoint、场景和门槛绝对不能事后修改？
10. 下一步是否以论文写作为主，且没有默认重开冻结实验？
11. 当前唯一开放的新实验是否为Stage7L？
12. Stage7L-C2是否已作为最后一次pre-D amendment冻结，且Stage7L-D只运行planner-level confirmation、尚未读取representation？
13. 是否明确现有PDM不能直接提供clean pure-lateral treatment？
14. 是否区分Stage7 post-hoc lane-change slice与未来Stage7L prospective confirmation？
15. 是否明确Stage7L不能重新打开模型训练？

若以上任一问题不清楚，先回到本文件和第0节权威文档，不要启动训练、仿真或BDD重算。

`CURRENT_RESEARCH_HANDOVER_UPDATED_FOR_THESIS_CLOSURE`
