# Stage7L-B Pure-Lateral Development 中文报告

## 一、结论

本阶段完成了独立的pure-lateral development，但**尚不具备进入Stage7L-C protocol freeze的条件**：

```text
DEVELOPMENT_ONLY_NOT_CONFIRMATORY
STAGE7L_B_DEVELOPMENT_NOT_READY_FOR_FREEZE
```

横向机制、代码级纯度、completion、off-road和纵向cleanliness均达到开发预期；唯一阻塞项是当前pre-treatment eligibility只检查初始时刻车道净空，未排除后续15 s交通轨迹与强制换道路径冲突。最终24个场景中有4个场景在五个dose下都发生责任碰撞。碰撞对dose完全不变，因此不是Sharp endpoint制造的差异，但17%的场景级碰撞率不适合作为prospective confirmation treatment。

本阶段没有建立confirmation roster，没有运行confirmation，没有读取或导出embedding，也没有计算BDD/MMD。

## 二、开发资源与provenance

- Stage7L-A2代码基线：`5208220a09ec3d33e55f6099697b555c59f2b218`。
- 最初pre-treatment roster：24 token / 24 log，6 left / 18 right。
- safety refinement新增2个替换token；Stage7L-B累计实际使用26个unique token，低于32个上限。
- 最终full-development roster：24 token / 24 log，6 left / 18 right。
- 最终roster SHA256：`4d8b12f923120260d186ae9bc7b35cc4ff47f7d5ecd67925bd907886c872bc04`。
- 最终maneuver manifest SHA256：`e2d1c1b14777226017779ac0190e2ad60146231b808882de84e467e58ce3673e`。
- 永久exclusion ledger SHA256：`2910e77970e32ad865bcb7b27ea8189d80ac5baf3c5d0a83dcecc622d47fe127`。
- background固定为`closed_loop_nonreactive_agents`；五档共享source/target/trigger、纵向配置和canonical `s_route(t)`。

最终24个场景覆盖：初速4.79–15.45 m/s、lane width 2.99–6.70 m、paired reference remaining 90.45–144.04 m；source/target curvature、左右方向和source-target geometry均有变化，不是单一直线空车道集合。

## 三、测试过的dose版本

共测试2套一维transition-length参数：

| 版本 | dose0/25/50/75/100（m） | 结论 |
|---|---|---|
| candidate v1 | 60 / 54 / 48 / 42 / 36 | 横向激励清晰，但短于54 m后off-road随dose增加；保留为开发历史 |
| safe v2 | 60 / 58.5 / 57 / 55.5 / 54 | 120/120 official run成功、无off-road；推荐继续审阅 |

调参只使用physical mechanism、safety、trajectory validity和longitudinal nuisance。没有使用任何representation结果。

## 四、横向dose-response

安全版五档的aggregate中位数如下：

| dose | Duration (s) | RMS lat accel (m/s²) | Peak lat accel (m/s²) | RMS yaw (rad/s) | Peak yaw (rad/s) | RMS lat jerk (m/s³) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.450 | 0.193 | 0.503 | 0.0261 | 0.0584 | 0.184 |
| 25 | 1.351 | 0.203 | 0.531 | 0.0272 | 0.0614 | 0.191 |
| 50 | 1.350 | 0.213 | 0.561 | 0.0286 | 0.0647 | 0.201 |
| 75 | 1.250 | 0.225 | 0.594 | 0.0300 | 0.0682 | 0.214 |
| 100 | 1.350 | 0.237 | 0.629 | 0.0315 | 0.0720 | 0.228 |

结论：

- RMS/peak lateral acceleration、RMS/peak yaw rate和RMS lateral jerk的aggregate顺序清晰。
- dose100−dose0的scenario-level方向一致率分别为100%、87.5%、100%、91.7%和100%。
- duration总体变短；dose100−dose0中位数为−0.300 s，方向一致率83.3%。dose75→dose100存在0.1 s采样和settling判定造成的局部反转，因此不应冻结“每档严格单调”的门禁。
- peak lateral jerk只有62.5%的endpoint方向一致率且存在离散平台，不建议作为primary metric。
- 24个场景均完成target-lane entry和settling，final target center offset保持小量级。

建议Stage7L-C优先冻结：`lane_change_duration_s`、`rms_lateral_accel_mps2`、`peak_yaw_rate_radps`。peak acceleration、RMS yaw和RMS jerk作为secondary mechanism evidence。

## 五、纵向nuisance

最强dose100相对dose0的p90/max absolute差异：

| 指标 | p90 | max |
|---|---:|---:|
| mean speed | 0.001009 m/s | 0.001086 m/s |
| RMS longitudinal accel | 0.000907 m/s² | 0.017632 m/s² |
| RMS longitudinal jerk | 0.000645 m/s³ | 0.020811 m/s³ |
| route progress | 0.022185 m | 0.031012 m |

这些扰动显著小于横向treatment尺度，没有系统性纵向分叉。建议Stage7L-C审阅以下有物理裕量、并非贴合开发最大值的候选门槛：

- `|Δ mean speed| ≤ 0.02 m/s`；
- `|Δ RMS longitudinal accel| ≤ 0.05 m/s²`；
- `|Δ RMS longitudinal jerk| ≤ 0.10 m/s³`；
- `|Δ route progress| ≤ 0.25 m`。

这些值在Stage7L-B中仅为proposal，尚未冻结。

## 六、validity与safety

| dose | Official success | Completion | Collision scenarios | Responsible collision scenarios | Off-road | Invalid/incomplete |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24/24 | 24/24 | 4/24 | 4/24 | 0/24 | 0/24 |
| 25 | 24/24 | 24/24 | 4/24 | 4/24 | 0/24 | 0/24 |
| 50 | 24/24 | 24/24 | 4/24 | 4/24 | 0/24 | 0/24 |
| 75 | 24/24 | 24/24 | 4/24 | 4/24 | 0/24 | 0/24 |
| 100 | 24/24 | 24/24 | 4/24 | 4/24 | 0/24 | 0/24 |

碰撞集中在相同4个token，且每个token五档均碰撞、事件数也不随dose变化。它证明dose treatment本身没有制造剂量依赖的失败，但也证明静态初始净空不足以定义未来confirmation的clean opportunity。

## 七、eligibility规则开发记录

旧规则使用初始target-lane gap ≥10 m。sanity v1中token `da3815fd973d5450`五档均碰撞，且短transition dose出现off-road；因此在完整24场景运行前统一修改为：

```text
initial target-lane object gap >= 15 m
scenario horizon = 15 s for tagged and untagged/default scenarios
paired source/target reference covers canonical progress through 15.4 s
```

规则统一排除了`da3815fd973d5450`和另一个低净空token，并用2个pre-treatment候选替换；所有用过的token仍永久排除confirmation。

完整开发进一步发现，需要新增但尚未实现/冻结的规则：

> 使用原始pre-treatment agent tracks，对未来15 s canonical source/target corridor做时间对齐的动态占用/安全净空审计，而不是只看初始帧距离。

该规则必须先在全inventory上重扫、验证它能排除4个已知冲突场景，并重新核算剩余供给；不能直接写成“删除这4个token”。

## 八、remaining inventory与Stage7L-C边界

按当前静态规则、排除全部26个开发token及其log后，剩余：

- 83 fresh eligible token；
- 67 fresh log；
- 15 left / 68 right；
- 所有83个token与development log-disjoint。

因此当前账面上能组成80-pair、development-log-disjoint roster，但只有3个token余量。由于新增动态交通净空规则尚未重扫，**不能声称Stage7L-C的80-pair供给已经最终保证**。

未来若供给仍≥80，建议pre-treatment分层约15 left + 65 right；正式比例、roster和SHA只能在Stage7L-C授权后冻结。

## 九、Stage7L-C候选协议（未冻结、未启动）

- Treatment：60 / 58.5 / 57 / 55.5 / 54 m，trigger progress 12 m，canonical longitudinal target speed 5 m/s、accel limit 1 m/s²，0.4 s planner horizon。
- Background：`closed_loop_nonreactive_agents`。
- Eligibility：native adjacency、route compatibility、reference覆盖15.4 s、official runnability、initial target gap≥15 m，并新增15 s动态source/target corridor clearance。
- Primary metrics：duration、RMS lateral acceleration、peak yaw rate。
- Nuisance proposal：0.02 m/s、0.05 m/s²、0.10 m/s³、0.25 m。
- Failure policy：runtime failure、invalid、incomplete、collision或off-road均不得post-treatment删除；mechanism/safety gate失败则停止后续representation evaluation。
- Sample size：优先80，不默认扩到100。

在动态交通规则和inventory rescan完成前，不建议冻结该协议，也不建立confirmation roster。

## 十、科学边界

Stage7L-B只证明安全版一维pure-lateral参数具有清晰横向机制、极低纵向副作用和良好off-road/completion表现。它没有计算BDD，不能写成“pure-lateral BDD confirmed”。当前阻塞是prospective eligibility/safety问题，不是representation或dose-response问题。
