# Stage 7 M6：跨域 style-sensitive embedding 与配对 BDD 协议

## 1. 问题定义

Waymo 与 nuPlan 的数据构造和统计零假设不同：

- Waymo `pseudo_agg_vs_cons` 是从 human-driving test windows 中按行为代理选择两端的
  unpaired positive control；
- nuPlan M3-M6 是相同 scenario 的 assertive/conservative official rollout pairs，
  属于 matched simulation experiment；
- M5 显示 learned embedding 的 paired sign-flip 和 scenario-disjoint probe 显著，
  但 marginal MMD 不显著。主要机制是 scenario heterogeneity 与 estimand mismatch，
  不能据此认定 encoder 完全没有风格信息。

解决顺序固定为：

1. 使用与 matched experiment 对齐的 BDD 零假设；
2. 审计 scenario/planner/token、数据质量和 fallback 泄漏；
3. 在新 log/scenario-disjoint、selection config 独立冻结的数据上做锁定确认；
4. 只有锁定结果仍显示表示能力不足时，才启动跨域重训练消融。

Waymo-only 模型跨域应用到 nuPlan 是本研究的主要外部验证设定，不要求为了使用
nuPlan 验证集而先用 nuPlan 重训练。

## 2. M6A：scenario-conditioned paired BDD

M6A 不删除、不覆盖 Stage6 marginal BDD。它为 complete same-scenario pairs
增加两种设计匹配的统计量：

1. **original-space paired-label-swap BDD（primary）**
   - 在未变换的64维 embedding 上计算 biased single-RBF V-statistic MMD²；
   - bandwidth 是 pooled rows 的正有限 off-diagonal squared distance 精确中位数；
   - bandwidth 在 observed statistic 和所有 permutation 中保持固定；
   - permutation 只允许在每个 scenario pair 内交换 A/B label。
2. **scenario-residualized paired BDD（secondary）**
   - 对每对 embedding 计算 midpoint；
   - `r_A = z_A - midpoint`，`r_B = z_B - midpoint`；
   - 在 residual embedding 上使用同样的 paired-label-swap null。

Stage6 marginal BDD 回答“不控制 pairing 时，两组总体边际分布是否不同”；M6
paired BDD 回答“控制 scenario 后是否存在系统 planner effect”。二者不能互相替代。
Original 与 residual space 的尺度和 bandwidth 不同，不得直接比较 MMD² 大小。
Residualization 同时观察一对 A/B，仅是分析变换，不是可部署的单条日志 encoder。

## 3. M6.1 方法冻结结果

当前45对数据在 M3-M6 中已被反复观察，其角色固定为
`METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY`。100000 次 permutation 的结果为：

| analysis | role | space | MMD² | exceedance | plus-one p |
| --- | --- | --- | ---: | ---: | ---: |
| frozen Stage7 M4 marginal BDD | historical reference | original | 0.0142209 | — | 0.733267 |
| fixed-kernel pooled shuffle | control | original | 0.0141802 | 74086/100000 | 0.740863 |
| paired-label-swap BDD | primary | original | 0.0141802 | 175/100000 | 0.001760 |
| pair-midpoint residual BDD | secondary | residual | 0.0994187 | 0/100000 | ≤0.000010 |

M4 与 M6 pooled recheck 的轻微差异来自估计器配置：M4 使用历史 multi-kernel
配置，M6.1 冻结为 single-RBF、精确 pooled median bandwidth 的 biased
V-statistic；这不是数据或 embedding 变化。

Pair audit 结果：45/45 complete pairs，duplicate token、missing planner、
row metadata conflict、unequal within-pair valid horizon 和 non-finite embedding
均为0。Tier A 有40对，Tier A+B 有44对。质量敏感性分析中：

- Tier A primary：MMD²=`0.0164016`，p=`0.000440`；
- Tier A+B primary：MMD²=`0.0164385`，p=`0.000080`，
  两个预定义子集的 Holm-adjusted p 均小于0.001；
- embedding pair distance 与 max fallback rate、pair fallback-rate delta、
  max ambiguous rate 和 ambiguous-rate delta 的相关性均未通过 Holm 校正。

因此结果不是由单个低质量 pair 或已测 fallback 指标明显驱动，但这不等于证明
不存在任何未测混杂。当前结论是方法冻结完成，而不是独立确认完成。

## 4. Encoder 改进假设，而非已证实根因

Stage5D-balanced-v2 的33维弱监督 schema 不包含 `mean_speed`；M4 中稳定的 nuPlan
planner effect 包括 mean speed delta `+1.4277 m/s` 和 RMS acceleration delta
`+0.2562 m/s²`。M5 中 embedding distance 与 `|delta mean speed|` 相关，但与
acceleration 和 THW contrast 的相关性未通过 Holm correction。

这支持“纵向监督可能改善跨域 geometry”作为候选假设，但不能把缺少
`mean_speed` 宣称为 marginal BDD 不显著的根因。尤其是直接加入绝对速度可能让
模型学习道路等级、限速或拥堵，而非驾驶风格。后续表示学习必须优先使用场景归一化
或关系型目标，例如同场景相对速度/进度、speed-limit-normalized speed、相对前车的
closing/progress 和同任务内的 longitudinal response。

现有 Stage5D-balanced-v2 仍是主模型：它只由 Waymo human-driving 数据训练，
nuPlan 保持外部验证域。不得用当前45对 exploratory 数据微调主模型后，再把同一批
数据称作外部验证。

## 5. M6B：仅在触发条件满足时执行的跨域重训练消融

M6B 不是 M6.1 之后的默认下一步。只有新锁定集出现以下任一情况才触发：

- primary paired BDD 未复现，而物理行为差异和数据质量检查仍通过；
- task-conditioned 分析显示特定关键任务的 embedding sensitivity 系统不足；
- 预注册的 representation probe 未达到门槛。

M6B 必须创建新 checkpoint family，不覆盖 Stage5D-balanced-v2，也不替代
Waymo-only 主结果。拆分以 log、scenario token 和 planner config 为组完成，任何
同源组不得跨 train/validation/locked test。样本量由预注册 effect size 和 power
analysis 决定，不使用固定的120/40/80经验数作为通用充分条件。

保留 Stage5 group-weighted objectives，并把下列内容作为独立消融：

- 场景归一化的 kinematic metric loss；
- same-scenario continuous-delta/ranking loss；
- 冻结 Stage5D-balanced-v2 的 Waymo teacher consistency；
- 可选 domain adaptation，但不得默认消除真实 behavior-style signal。

Planner name/config 不得作为 encoder input；normalization 只在 training split
拟合；至少运行5个随机种子。绝对 `mean_speed`、`speed_p90` 或 progress 只能作为
对照消融，不能在未做 ODD/限速控制时直接解释为 style supervision。

## 6. 锁定确认与验收门槛

下一阶段先采集新的 log/scenario-disjoint pairs，保持冻结 planner treatment
参数不变，并在查看 planner labels
结果前冻结：

- primary：原始64维 embedding、single-RBF biased V-statistic MMD²、精确 pooled
  positive off-diagonal median bandwidth、within-pair label swap、100000
  permutations、plus-one p-value、`alpha=0.05`；
- secondary：pair-midpoint residual BDD；
- pair completeness、token/planner/row/horizon 一致性和 Tier A / Tier A+B
  quality sensitivity；
- task-conditioned BDD 与预处理定义；
- safety/performance metrics 单独报告，BDD 显著不等于更安全或更优。

锁定集必须与当前45对在 log、scenario token 和 planner config 上不相交。主结论
由 primary 原空间 paired BDD 决定；residual、probe、task slices 和多重质量子集
是预注册的 secondary/sensitivity analyses。

## 7. 当前数据状态

本机保留 Stage5 checkpoint、schema、标准化统计与历史评估产物，但 Waymo full51
manifest 中部分 shard paths 指向不可用的旧 macOS 路径。这不阻塞 Waymo-only
checkpoint 的 nuPlan 外部验证和 M6.1 方法冻结；只有 M6B 被锁定证据触发、需要
重新训练时，才必须先恢复 Waymo training shards。

M6.1 输出位于
`outputs/stage7_m6_1_paired_bdd_method_freeze_v1/`。

## 8. M6.2：锁定确认入口与预处理 task-conditioned BDD

M6.2 将“场景可比”落实为仿真前已存在的 `scenario_type` 分层。冻结任务族为
following interaction、lane change、stop-go control、high motion dynamics 和
dense or vulnerable interaction。由 rollout 结果计算的 lane-change、
hesitation、braking 等 bins 仍可用于定位和敏感性分析，但不得作为确认性 matching
或 task-selection 变量，否则可能对 treatment outcome 条件化。

新确认集必须满足：

- 与45对开发集的 `scenario_token` 和 `log_name` 交集均为0；
- assertive/conservative planner 的物理参数指纹与冻结 treatment 完全相同；
- 场景选择、仿真配置、分析脚本和 task mapping 在解盲前冻结；
- 80个 overall pairs 和每任务12对仅为运行/质量下限，不是功效充分性的证明；
- 必须另附 simulation-based power justification。

Task subset 不超过20对时枚举全部 `2^n` 个 within-pair assignments；更大 subset
使用100000次 Monte Carlo swaps。Learned embedding 的五个 task p-values 组成
一个 Holm family；interaction features 和 trajectory summary 只作为机制对照。

当前45对开发集中，五个 task 各8–9对，均未达到12对运行下限。Learned embedding
只有 `high_motion_dynamics` 通过 Holm correction（exact p=`0.00390625`，
Holm p=`0.01953125`）；其余四个 task 未显著。该结果只用于实现验证和新数据规划。

工具：`tools/stage7_m6_2_locked_task_bdd.py`。开发验证输出：
`outputs/stage7_m6_2_locked_task_bdd_development_v1/`。

## 9. M6.3：锁定确认集功效与采集配额

M6.3 不重新训练 encoder，而是用45对方法开发集建立 empirical-pilot generator。
生成器分别 bootstrap 中心化的 pair midpoint 与 pair-difference residual，并将
开发集平均 pair shift 乘以预先指定的 effect scale。每个模拟数据集复用 M6.1
冻结的 single-RBF biased MMD、pooled median bandwidth 和 within-pair label
swap；五个 task p-values 始终作为一个 Holm family。

主冻结灵敏度假设为 locked-domain effect 至少等于 development-pilot mean shift
的75%，目标 simultaneous power 为0.80。500次模拟/网格点和999次 planning
permutations 得到：

- overall 的纯功效选择为45个完整 pairs，但最终执行 M6.2 的80对运行/质量下限；
- 五个 task 必须各有60个完整 pairs，simultaneous power=`0.918`，Wilson 95% CI
  `[0.891,0.939]`；
- 按20%无效/失败率，预采每任务75对，若任务配额互斥则共375对；
- 最终 locked confirmation 使用100000次 swaps，而不是规划阶段的999次。

50% effect-scale 敏感性分析要求每任务160个完整 pairs，simultaneous
power=`0.936`，95% CI `[0.911,0.954]`；按20%损耗率为每任务200对、总计1000对。
它用于预算风险评估，不是主设计的事后替代。

主冻结文件
`outputs/stage7_m6_3_simulation_power_v1/m6_3_locked_power_justification.json`
通过 SHA256 绑定 M6.2 lock 和 M6.3 脚本。M6.2 locked mode 必须检查状态、hash、
五个 task mapping、overall quota、逐任务 quota 和 planner treatment fingerprints，
任一不一致即 fail closed。不得根据 locked effect size 提前停止采集、修改任务族
或降低配额。

这些数值是基于每任务仅8–9对 pilot 的设计假设，不是 post-hoc achieved power，
也不能证明新域一定复制开发效应。只有新 log/scenario-disjoint 数据完成盲化
确认后，才判断是否触发 Stage5D 新 checkpoint family 的条件式重训练。

## 10. M6.4：outcome-blind collection freeze

M6.4 将 M6.3 配额落实为仿真前 intake gate。选择器只允许读取 DB/log/token 和
nuPlan 原生 `scenario_type`；不得读取新 planner trajectory、embedding、BDD、
成功率或行为指标。开发集的45个 token 及其所有 log 整体排除，确保 locked set
同时满足 scenario-disjoint 和 log-disjoint。

主采集 roster 为每任务75个 gross scenarios；每任务另冻结15个 reserve。Primary
必须全部按固定顺序尝试。只有 primary 完整配对不足60时，才可按同任务 reserve
rank 补充，且停止条件只能是技术/质量完整性和冻结 complete-pair quota，不能是
已观察效应大小或显著性。Primary 和 reserve 合计每 log 最多2个场景，减少 M6.3
独立 pair power simulation 与同 log 聚类之间的偏差。

选择器 fail closed 检查：

- M6.2 lock、M6.3 power file、开发 metadata 及当前分析脚本 SHA256；
- Stage7C 当前 planner parameters、开发 metadata 和 M6.2 fingerprints 三方一致；
- frozen task mapping 完全一致；
- DB 文件存在、token/log 唯一、开发 token/log 零重叠；
- 同时命中多个 frozen scenario types 的 token 被排除，而不是任意择一；
- 全部 primary/reserve 配额满足后才输出 locked manifest。

2026-08-04 的本机 mini 预检未通过：63个 logs 中34个属于开发集，剩余29个
eligible logs，低于 `max_per_log=2` 下 primary 所需的至少188个。冻结 lane-change
候选在排除开发 logs 和歧义 token 后只有2个。因此状态为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，工具没有生成 locked manifest，
也不允许启动 rollout。该缺口只能通过增加并索引新的 nuPlan log DB 解决；不能用
修改 task mapping、复用开发 logs 或查看 planner outcomes 后重排候选来解决。
