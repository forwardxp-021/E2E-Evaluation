# Stage7L-A2：Pure-Lateral Treatment 清洁实现与 Smoke 验证

## 1. 最终冻结结论

`STAGE7L_PURE_LATERAL_IMPLEMENTATION_CLEAN`

`STAGE7L_B_DEVELOPMENT_AUTHORIZED`

本结论只说明pure-lateral treatment的注入通道、动力学输出、official nuPlan接口、物理smoke和候选供给已经满足进入development的技术条件。它不表示BDD一定显著，也不表示任何representation更优。本阶段没有训练、正式development、confirmation roster/rollout、embedding导出或BDD计算；Stage7L-B尚未启动。

## 2. 十二个验收问题

### 2.1 pure-lateral treatment是否已经实现

是。新增独立external `PureLateralExecutionPlanner`，没有修改nuPlan或tuPlan Garage冻结源码。处置定义为：dose参数只进入横向quintic transition length，scenario、初始状态、source/target lane、direction、trigger、route、纵向生成器、目标速度、加速度上限、仿真horizon和background-agent模式全部固定。

### 2.2 canonical longitudinal reference是什么

canonical reference是冻结source-lane baseline centerline上的route progress。`CanonicalLongitudinalProgressGenerator`以official首帧速度、统一5.0 m/s目标速度和1.0 m/s²加速度上限生成共同`s_route(t)`；二维轨迹由同一route progress处的source/target baseline位置进行quintic横向融合。每条二维轨迹自身的arc length不参与纵向进度定义。

### 2.3 五档`s_route(t)`是否完全一致

是。dose0/25/50/75/100的planner audit各41个初始规划点逐点`np.array_equal`，并且五档的canonical generator SHA和dose-invariant maneuver SHA完全相同。official smoke summary中`s_route_pointwise_identical=true`。

### 2.4 dose具体只改变什么

只改变`L_transition`：dose0/25/50/75/100分别为60/54/48/42/36 m，状态均为`A2_SMOKE_ONLY_NOT_FROZEN_FOR_CONFIRMATION`。共同横向曲线为`q(u)=10u³−15u⁴+6u⁵`；起点、终点、lane width、trigger和settling target不随dose改变。

### 2.5 source/target/trigger是否五档完全一致

是。最终clean smoke五档共享同一maneuver manifest：scenario `2880228e6471586a`、同一official首帧指纹、source/target native adjacent lane、right方向、route fingerprint和`s_trigger=12 m`。planner运行中禁止重新选择lane、方向或trigger。

### 2.6 background-agent模式是什么

使用nuPlan official `closed_loop_nonreactive_agents`，observation/agent model为`nuplan.planning.simulation.observation.tracks_observation.TracksObservation`。背景车辆沿recorded/replay轨迹运行，不对ego处置作reactive response；五档共享配置SHA `9aad54d780d4c53612be8e397f51d8b1f6718b8323617f5eee6d49617c3b9f0c`。

### 2.7 trajectory state是否动力学一致

是。position先由共同`s_route(t)`与横向profile生成，heading、global velocity、acceleration、yaw rate、angular acceleration、curvature和steering再由同一position/time导数统一构建。单元测试验证finite、time monotonic、position/velocity/acceleration导数一致、heading jump和曲率合理；official smoke五档轨迹均valid且完成换道。

### 2.8 official 5-dose smoke是否全部成功

是。最终development-only token五档均official simulation成功，`official_success_count=5/5`、planner audit `5/5`、严格same-token alignment通过。五档均零at-fault collision且drivable-area compliant。两个A2 smoke token均已永久写入prior exclusion ledger，不得进入未来confirmation。

### 2.9 realized longitudinal nuisance有多大

以dose0为基准，五档最大绝对差异为：mean speed `0.005553 m/s`、RMS longitudinal acceleration `0.000187 m/s²`、RMS longitudinal jerk `0.002776 m/s³`、route progress `0.051174 m`。相对于约106.1 m的route progress，最大route差异约0.048%。横向峰值加速度则按dose严格递增：约0.405、0.548、0.763、1.011、1.273 m/s²。

### 2.10 map-based fresh opportunity inventory有多少token/log

最终pre-treatment inventory在排除历史及两个A2 smoke token后得到148个fresh eligible token、120个unique log；left=25、right=123，Las Vegas=3、Pittsburgh=145，覆盖8个source roadblock。selection只使用official初始状态、native adjacency、route、lane length、initial target-lane clearance和官方scenario映射边界；未使用expert是否换道、rollout结果、embedding或BDD。

### 2.11 排除历史场景后是否足够24+80

是。148≥104，且120个unique log≥104，因此未来可以做24 development + 80 confirmation的scenario零重叠及严格log-disjoint分配。方向明显偏right，未来冻结roster时必须报告并尽量平衡，但当前25个left候选足以覆盖24-scene development，不构成A2供给阻塞。preferred 150缓冲目标差2个，但它不是硬门禁。

### 2.12 是否可以进入Stage7L-B

可以单独授权进入Stage7L-B development，但本阶段停止，不自动创建development roster或运行正式development。

## 3. Official scenario初始帧边界修复

库存复核发现：nuPlan冻结的`nuplan_scenario_mapping`对有`scenario_tag`的token使用-3 s extraction offset，而未标注/default token从anchor本身开始。A2最终实现先按该官方规则解析真正的首个lidar token，再通过official EgoState query取得位置、heading、速度和时间，避免自行重建quaternion heading。该修复使inventory中的initial fingerprint与official simulation严格一致，并将最终fresh supply从错误口径的122修正为148。

## 4. Development-only physical smoke迭代记录

- token `6c262b4151415c9a`用于接口和低速初始smoke：official 5/5成功，但recorded background下发生碰撞，因此仅保留为失败的physical-cleanliness诊断并永久排除。
- token `2880228e6471586a`用于第二个也是最后一个A2 smoke：它暴露并验证了tagged scenario的-3 s初始帧边界。按physical feasibility将smoke-only transition length从48/42/36/30/24 m调整为60/54/48/42/36 m后，五档5/5成功、完成换道、零责任碰撞、全程drivable-area compliant，并产生严格有序的横向机制差异。
- 上述调整只查看runtime、lane/Frenet mechanism、collision/offroad和longitudinal nuisance；没有读取任何embedding/BDD，也没有按representation表现调参。

## 5. 关键输出与SHA256

| 资产 | SHA256 |
|---|---|
| smoke config | `17ca08c903b0fa13be6f05f282a7a9ac5639c9ff5aa922344868e4460c7de664` |
| final inventory CSV | `d3de7f2af7e7aac106f987cfbbdd69e95eb305c51554a097810efe26aacb8ead` |
| final inventory summary | `830098d32eec08cb70ee6e874a97cc90e764a94e2cf3482e90e18a9dc347d8ce` |
| clean smoke maneuver manifest | `9d823d5346ddabc7bf3708560d04888965fa493a8733688a7638e176f1810624` |
| official smoke summary | `6d070c5633a5c6b11f254dcd17292f5dfc3b4bd68f68896710618a9a149df1c0` |
| mechanism + safety summary | `24d7376608ab11323c846ecb2dbaab5d2ced3c51f712716c3adbea2d9fab6963` |

运行输出位于：

- `outputs/stage7l_a2_lane_change_opportunity_inventory_v1/`
- `outputs/stage7l_a2_official_smoke_v2_safe_final4/`
- `outputs/stage7l_a2_lateral_mechanism_v2_safe_final_with_safety/`

这些大型/运行时输出不提交Git；小型冻结证据汇总保存在`docs/stage7l_a2_clean_implementation_manifest_v1.json`，两个A2 smoke token同时固化在`docs/stage7l_a2_smoke_exclusion_ledger_v1.csv`。

## 6. 科学边界

A2证明的是`treatment injection channel`在代码层只改变横向profile，并且一个development-only physical smoke没有明显纵向分叉。它没有冻结未来confirmatory nuisance threshold，也没有证明正式库存中所有场景均会完成、无碰撞或保持相同realized longitudinal outcome。未来BDD population必须基于pre-treatment frozen opportunity roster，不能按换道完成或BDD结果删场景。
