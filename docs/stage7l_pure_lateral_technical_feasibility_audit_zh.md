# Stage7L-A：受控纯横向换道执行实验技术可行性审计

## 1. 审计结论

`PURE_LATERAL_TREATMENT_IMPLEMENTATION_NOT_YET_CLEAN`

当前仓库具备官方nuPlan仿真、同场景token对齐、83D context构建、old64/A/B/C/ego13锁定表示导出、representation-specific paired randomization null及最终BDD报告映射能力；但是，当前PDM-Closed没有独立的“同一换道意图/目标车道/纵向进度下，仅改变横向过渡时长或sharpness”的控制面。

现有PDM把横向路径、路径走廊内的前车搜索、IDM纵向推进、proposal仿真、评分和最终argmax联合在一起。直接修改`lateral_offsets`、复用现有assertive/conservative profile，或对最终轨迹做横向后处理，都会产生不可忽略的纵向污染或动力学不一致。因此本阶段停止于技术审计：**未实现planner、未建立roster、未运行development/confirmation、未读取任何新embedding/BDD。**

只有完成本文第8节的最小侵入实现和代码级洁净性门禁后，才允许把状态升级为`STAGE7L_PURE_LATERAL_TECHNICAL_AUDIT_PASS`并进入Stage7L-B。

## 2. 审计范围与不可变边界

- 工作分支：`20260611_stage7_conclusion`
- 审计基线commit：`c901fb53316b06791fc628cd8415f888bb8cba60`
- nuPlan devkit：`e9241677997dd86bfc0bcd44817ab04fe631405b`
- tuPlan Garage：`b51d5d04fac1bd4389653b9ab2ff73ea88f435a3`
- 未修改Stage6/Stage7冻结实验、A/B/C checkpoint、Stage6V/W/S-v3结果或最终BDD报告数值。
- 只进行了源代码、manifest、CSV/JSON库存和历史运行资源的只读审计。

## 3. 为什么现有PDM不能直接形成纯横向处置

### 3.1 `lateral_offsets`不是换道执行参数

`PDMClosedPlanner`只接收`lateral_offsets`作为中心线平行偏移。`AbstractPDMClosedPlanner._get_proposal_paths()`先按当前route lane构造一条Dijkstra centerline，再调用`parallel_discrete_path()`生成整条中心线的固定平移路径。它没有显式的：

- lane-change intent；
- source/target lane ID；
- trigger；
- transition duration；
- lane transition profile；
- settling phase。

所以`[-0.5, 0.5]`或`[-1.5, 1.5]`表示车道中心线附近的平行offset proposal，不等于从当前lane平滑进入相邻lane。

### 3.2 横向路径与纵向IDM不可分离

`PDMGenerator.generate_proposals()`对每条lateral path分别：

1. 建立driving corridor；
2. 查询该路径上的leading agent；
3. 推进对应IDM longitudinal state；
4. 把progress插值回该路径上的SE2状态。

随后`PDMSimulator`和`PDMScorer`对proposal联合评分，并以`argmax`选择最终轨迹。改变横向路径会改变leading-agent集合、纵向速度/加速度轨迹、碰撞/可行驶区评分及最终选中的longitudinal policy。因此不能把“只改lateral_offsets”解释为pure-lateral treatment。

### 3.3 不推荐最终轨迹横向warp

在`compute_planner_trajectory()`返回后再修改x/y会产生三个问题：

- 输出pose、heading、velocity、acceleration和steering之间可能不一致；
- perfect-tracking把修改后的状态反馈到下一仿真tick，两个dose的后续PDM输入与纵向结果仍会分叉；
- target lane、route合法性及交通走廊没有进入轨迹生成约束。

所以推荐注入点不是最终`InterpolatedTrajectory`之后，而是**共同scalar longitudinal progress生成之后、SE2 path插值之前**。

## 4. 对十个实施问题的明确回答

### 4.1 pure-lateral treatment在哪一层实现

未来应新增独立external Hydra planner：`PureLateralExecutionPlanner`。它可以复用PDM的route、observation、IDM和trajectory conversion工具，但不得修改冻结的tuPlan Garage源码，也不得继续使用PDM的“多横向path × 多纵向policy联合argmax”作为Stage7L treatment生成器。

推荐结构：

1. `CanonicalLongitudinalProgressGenerator`：所有dose共享，同一场景产生同一标量progress/time schedule；
2. `FrozenLaneChangeManeuver`：每个场景只在pre-treatment阶段确定一次source lane、target lane、方向和trigger；
3. `QuinticLateralPathGenerator`：只由dose改变transition length/duration；
4. 将共同progress投影到各dose的二维lane-change path，生成一致的pose和动态状态。

这是一个薄的实验planner adapter，而不是对现有PDM profile增加“总aggressiveness”参数。

### 4.2 如何保证same intent / same target lane

每个scenario在任何rollout前生成一条不可变`treatment_maneuver_manifest`，至少记录：

- scenario/log/DB token；
- initial ego state fingerprint；
- source lane ID；
- target adjacent lane ID；
- left/right direction；
- route roadblock IDs fingerprint；
- native adjacency证据；
- route compatibility证据；
- trigger定义；
-五档dose profile ID。

target lane必须使用nuPlan原生`Lane.adjacent_edges`并验证其roadblock/route及前向successor兼容性；不能只依赖当前`nuplan_lane_utils`的几何邻接fallback。manifest一旦生成，五个dose读取同一份source/target/trigger，不得在后续tick重新选择lane，也不得根据rollout结果重选方向。

### 4.3 如何尽量保持相同longitudinal progress

五档必须共享一个不依赖lateral dose的canonical longitudinal generator。建议在共同的虚拟参考走廊上计算leading-agent约束和IDM progress，再把同一`s(t)`映射到不同横向profile；不能让每个dose各自在自己的横向corridor内重新找前车并独立选择IDM proposal。

实现层至少要记录并验证：

- 每个dose的期望`s(t)`和time grid相同；
- 相同初始速度、目标速度、IDM参数、加减速上限；
- 不允许dose改变longitudinal配置；
- closed-loop后仍报告mean speed、RMS longitudinal acceleration、longitudinal jerk及route progress差异。

即使代码层共享progress，实际closed-loop仍可能因碰撞避免、地图约束或状态反馈产生副作用，因此“共同生成器”是必要条件，不是免除longitudinal nuisance gate的理由。

### 4.4 lane-change trigger如何冻结

推荐用**从scenario初始投影点起的固定route progress `s_trigger`**，而不是“看到gap后触发”“yaw达到阈值后触发”或“planner决定变道后触发”。固定progress与共同`s(t)`更适合保证各dose同一trigger。

具体数值只能在development中确定；confirmation freeze前必须同时冻结：

- `s_trigger`规则；
- trigger前最小直线/稳定距离；
- target lane剩余长度；
- transition结束后最小settling距离/时间；
- scene结束前安全余量。

这些规则只能使用initial state、route、map和scene horizon等pre-treatment信息。

### 4.5 dose如何实现

推荐在共同progress域使用minimum-jerk/quintic：

`q(u) = 10u^3 - 15u^4 + 6u^5`

`d(s) = d_source + (d_target - d_source) q(u)`，其中`u=clip((s-s_trigger)/L_dose, 0, 1)`。

`dose ∈ {0, 0.25, 0.50, 0.75, 1.0}`只控制`L_dose`或等价nominal transition duration；dose0最长、dose100最短。五档共享起终横向位置、方向、trigger、target lane、纵向生成器和约束。具体`L_gentle/L_sharp`不得在本审计阶段猜定，须由development的可运行性、lateral mechanism和longitudinal nuisance共同冻结。

### 4.6 哪些existing nuPlan scenes可以作为候选

可复用的pre-treatment原始来源为：

`outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh/all_scenario_tags.csv`

该库存包含1,621个DB、1,576个log、9,604,184条scenario-tag记录、5,386,575个unique scenario token。现有严格换道标签的只读计数为：

| 口径 | unique token | log |
|---|---:|---:|
| `changing_lane*`严格标签原始池 | 169 | 154 |
| 排除旧Stage7 lane-change 60 token | 109 | 101（token排除后） |
| 上述109中通过当前official scene-boundary规则 | 47 | 43 |
| 同时排除当前可识别Stage7P调参/探测token后 | 32 | 30 |

47个official-boundary候选按标签方向为34个left、13个right，但仍未通过native adjacency、route compatibility、forward length和scene horizon审计，不能称为Stage7L eligible roster。

结论：**严格`changing_lane*`标签池不足以同时支持24个development和至少80个confirmation。** Stage7L必须从更大的outcome-blind inventory建立真正的map-based “lane-change opportunity”库存，而不是要求expert标签本身就是changing-lane。任何场景只有通过初始ego lane、合法native adjacent lane、route compatibility、前向长度、horizon、地图和official runnability检查后才可称为eligible。

现有`stage7p_find_lane_change_candidates.py`的kinematic scan、event bin和rollout yaw/lateral displacement属于post-treatment或expert-outcome信息，不得用于confirmation roster筛选。

### 4.7 如何排除旧Stage7 changing-lane 60和调参场景

在Stage7L-B前先生成独立的`stage7l_prior_exclusion_ledger.csv`，至少合并：

1. `confirmation_scenario_ledger.csv`中`task=lane_change`的60个token；
2. 所有Stage7P实际运行过的`scenario_alignment.csv`中的target及actual nuPlan 16位token；
3. 后续Stage7L development全部scenario token；
4. development log集合；
5. 人工/自动调参使用过的任何额外token。

本次只读扫描在10个Stage7P simulation/alignment输出中识别到48个可直接作为nuPlan token的历史标识；它们尚未形成一份正式、去重、带来源的冻结ledger。confirmation freeze必须保存每条排除原因、来源文件和SHA，并要求development scenario零重叠、优先development log零重叠。

### 4.8 official runnability如何在freeze前检查

直接复用Stage6S-v3已验证方法：

1. 按DB分组调用nuPlan官方`get_scenarios_from_db(..., include_invalid_mission_goals=True, include_cameras=False)`；
2. 同时查询scene按name排序后的`row_num`和`scene_count`；
3. 要求`row_num >= 3 and row_num < scene_count - 1`与官方查询逐token一致；
4. roster候选必须100%为official query runnable；
5. 输出`official_runnability_audit.csv`并在roster manifest中锁定SHA。

Stage6S-v2的19/80失败说明只检查token存在不够；Stage6S-v3使用上述规则后80/80成功。本规则只证明official scene可构建，不替代Stage7L planner自身的unit/integration smoke。planner smoke必须只在development roster上运行，不能用confirmation treatment outcome决定是否入选。

### 4.9 development与confirmation需要多少rollout

五档dose中dose0由四个`doseX vs dose0`对照共享，所以每个scenario需要5条trajectory，而不是8条。

| 阶段 | scenario数 | 推荐dose数 | official rollout数 |
|---|---:|---:|---:|
| development端点快速检查 | 24 | 2（dose0/100） | 48 |
| development完整dose冻结（推荐） | 24 | 5 | 120 |
| confirmation最低规模 | 80 | 5 | 400 |
| confirmation目标规模 | 100 | 5 | 500 |

正式protocol应按120条development rollout完成五档调试，再一次性运行400或500条confirmation rollout。

### 4.10 时间、存储和失败风险

本机Stage6S-v3实测：80 scenario × 2 planner共160条rollout，批次46分钟，成功scenario平均34.456秒，即约17.23秒/rollout；rollout目录1.5 GiB，context目录54 MiB。按线性外推：

| 工作量 | 仿真中心估计 | rollout存储 | context估计 |
|---|---:|---:|---:|
| development 120 rollout | 约35分钟 | 约1.1 GiB | 约41 MiB |
| confirmation 400 rollout | 约1小时55分钟 | 约3.8 GiB | 约135 MiB |
| confirmation 500 rollout | 约2小时24分钟 | 约4.7 GiB | 约169 MiB |

考虑custom planner额外计算、Hydra启动、context构建、机制统计、100,000次paired null和失败重试，建议实际排期：development 1–2小时，80场景confirmation 3–4小时，100场景confirmation 4–5小时。当前磁盘可用约131 GiB；单次完整Stage7L预留8–12 GiB可覆盖结果、日志和有限重试。

主要风险按优先级为：

1. 当前最关键阻塞：横纵向联合PDM无法直接给出洁净处置；
2. 严格changing-lane标签库存不足80，必须建立更广的map opportunity inventory；
3. target lane虽相邻但不在route、前向长度不足或经过connector；
4. sharp dose导致offroad、碰撞、不可跟踪或transition未完成；
5. closed-loop状态反馈仍产生纵向分叉；
6. lane-change duration/settling/curvature需要真正的lane/Frenet语义，现有通用proxy不够；
7. external Hydra planner的import/config打包错误；
8. 若绕过official query，可能重演Stage6S-v2 scene-boundary失败。

## 5. 现有工具复用矩阵

| 资产 | 可复用性 | Stage7L边界 |
|---|---|---|
| `stage7c1_run_nuplan_simulation.py` | 可复用 | 已支持external Hydra planner、严格token/同场景对齐和official artifact审计；需注册五档planner config |
| `stage7p_build_scenario_inventory.py` | 可复用 | outcome-blind原始库存；当前字段不含ego lane、adjacency、route、length或runnability |
| `stage7p_find_lane_change_candidates.py` | 仅作seed/audit | DB标签可作候选来源；kinematic/event结果不得用于confirmation selection |
| `nuplan_lane_utils.py` | 部分复用 | 几何/path提取可复用；当前邻接fallback和基于rollout轨迹的空间查询不能直接作为pre-treatment target-lane证明 |
| `stage6s_v3_freeze_confirmation.py` | 高度可复用 | official query + boundary一致性和先freeze后run范式可直接迁移 |
| `build_nuplan_5neighbor_context_dataset.py` | 可复用 | 支持任意planner axis并输出83D；必须保持全部frozen roster行，不能按执行结果删行 |
| `stage6v_evaluate_stage6s_v2_representations.py` | 核心逻辑可复用 | checkpoint/scaler/ego13逻辑正确，但当前硬编码160行和旧planner名；需Stage7L通用适配器 |
| `stage6j_run_paired_bdd.py`及Stage6S paired-null | 可复用 | 100,000 pair swap、plus-one p、各representation独立bandwidth/null可迁移；每档dose需独立结果 |
| unified/final BDD schema | 可复用 | 结果完成后映射`LAT.LANE_CHANGE`和`LAT.DYNAMICS`；`INT.LATERAL_GAP`默认仍为N/A |

## 6. 机制指标仍缺少的实现

现有context/behavior-event工具包含yaw-rate、lateral-acceleration和curvature proxy，但没有一套面向source/target lane的、可冻结的Stage7L真实换道执行评估器。未来需新增Frenet/lane-aware mechanism工具，至少输出：

- source-lane离开、target-lane进入和settling的明确定义；
- lane-change duration与completion状态；
- RMS/peak lateral acceleration；
- RMS/peak yaw rate；
- lateral jerk；
- curvature/curvature-rate（仅在有效速度和稳定地图投影时）；
- target lane center offset与settling time；
- mean speed、RMS longitudinal acceleration、longitudinal jerk和route progress；
- collision、offroad、route failure、invalid/incomplete rollout。

机制工具不得把“executed successfully”作为保留BDD样本的条件。primary population始终是pre-treatment frozen opportunity roster。

## 7. 为什么本阶段不能判定technical PASS

以下关键证据目前不存在：

- 五档使用同一source/target/trigger manifest的代码；
- 横向dose只改变quintic transition length/duration的配置；
- 共同longitudinal progress随机/输入流的代码级一致性审计；
- 轨迹pose/velocity/acceleration/heading的动力学一致性测试；
- map-based opportunity inventory及至少24+80 fresh可运行场景供给证明；
- Stage7L专用lane/Frenet mechanism evaluator；
- external planner official one-scenario smoke。

在这些证据缺失时启动development，会把“PDM全局behavior差异”误当成“已知纯横向execution treatment”，不满足博士论文的因果解释边界。

## 8. 最小侵入实现方案

下一次单独授权的Stage7L-A2只应实现和smoke，不建立confirmation roster：

1. 新增external `PureLateralExecutionPlanner`及Hydra config，不改tuPlan Garage冻结源码；
2. 新增pre-treatment `Stage7L Lane-Change Opportunity Inventory`工具，使用scenario初始状态、native adjacency、route、lane length、horizon和official query；
3. 新增不可变maneuver manifest及五档dose config；
4. 新增lane/Frenet mechanism evaluator；
5. 泛化锁定checkpoint导出器和paired BDD wrapper，使其接受`N×5` planner axis，但不在A2读取正式BDD；
6. 只做synthetic/unit及1–2个development-only official smoke。

实现完成后必须同时通过：

- quintic起终位置、一阶/二阶边界导数和dose顺序单测；
- 五档manifest除dose profile外逐字段相同；
- 相同输入下五档canonical`s(t)`逐点一致；
- native source/target lane及route compatibility断言；
- finite/shape/time monotonic及SE2动态一致性；
- official smoke 5/5 rollout成功且严格same-token；
- 不读取任何embedding/BDD；
- 不触碰旧Stage6/Stage7冻结文件。

通过这些门禁后，才重新进行Stage7L-A验收；本次不自动进入Stage7L-B。

## 9. 审计可追溯性

机器可读审计清单：`docs/stage7l_pure_lateral_technical_feasibility_audit_v1.json`。

关键只读证据：

- PDM路径/纵向联合生成：tuPlan Garage `pdm_closed_planner.py`、`abstract_pdm_closed_planner.py`、`pdm_generator.py`；
- outcome-blind库存：`outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh/`；
- 旧Stage7 60场景：`outputs/stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv`；
- official runnability范式：`tools/stage6s_v3_freeze_confirmation.py`；
- 实测资源：`outputs/stage6s_v3_confirmation_batch_v1/batch_result.json`。

最终状态保持：

`PURE_LATERAL_TREATMENT_IMPLEMENTATION_NOT_YET_CLEAN`
