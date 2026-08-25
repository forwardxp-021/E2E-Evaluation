# Stage7L-B2：动态预处理交通净空与库存扩展报告

## 一、最终结论

```text
STAGE7L_B2_DYNAMIC_CLEARANCE_COMPLETE
STAGE7L_C_PROTOCOL_FREEZE_RECOMMENDED
```

Stage7L-B2建立了一个只使用原始nuPlan map、route、官方初始状态和`lidar_box` replay tracks的动态净空审计。它不接收dose ID、不读取Stage7L planner轨迹或rollout结果，也不读取embedding/BDD/MMD。

在全部历史token排除、并与26个Stage7L-B实际开发log严格分离后，Pittsburgh Pool B仍有152个dynamic-clean token、94个unique log（19 left / 133 right）。这高于120个preferred token和80个unique log门槛，因此可以建议下一阶段进行**协议冻结审阅**；本阶段没有创建confirmation roster，也没有启动confirmation。

## 二、动态净空的定义

动态eligibility为：

```text
f(original nuPlan log, original replay agent tracks, map, route,
  frozen source/target lane, canonical route progress, fixed physical buffer)
```

它不是online lane-change policy。由于Stage7L的background为`closed_loop_nonreactive_agents`，本规则只在实验设计阶段利用原始日志未来15秒的replay traffic，构造一个减少interaction污染的controlled validation benchmark。

### Common lane-change corridor

- 使用与所有dose共同的`CanonicalLongitudinalProgressGenerator`：初速来自官方场景、target speed 5 m/s、accel limit 1 m/s²；时间范围15 s、步长0.1 s。
- `s < 12 m`时，包络仅覆盖source lane；
- `12 m ≤ s ≤ 12 + 60 + 10 m`时，包络为同一canonical station上的source-center至target-center完整横向strip，覆盖54–60 m全部合法transition profile；
- 后续包络仅覆盖target lane。

这个共同包络不按五个dose分别构造，因此不存在“只选择所有dose都不碰撞”的treatment-dependent过滤。

### 回放时间对齐与buffer

- 原始`lidar_box`按timestamp取样；在0.1 s canonical grid上做线性插值。
- 只允许相邻观测间隔≤0.25 s；不外推。整个scene无法覆盖15 s时拒绝为`INSUFFICIENT_TRACK_HORIZON`；单条短track则记录缺失插值，不臆造其未来位置。
- ego footprint固定为5.0 × 2.0 m；agent使用记录的oriented length/width。
- 在footprint投影外增加3.0 m longitudinal buffer与0.5 m lateral buffer。

这些是固定工程裕量，不是由development碰撞标签拟合出的阈值。

## 三、reason codes与静态规则

动态审计输出：`SOURCE_LANE_DYNAMIC_CONFLICT`、`TRANSITION_CORRIDOR_DYNAMIC_CONFLICT`、`TARGET_FRONT_DYNAMIC_CONFLICT`、`INSUFFICIENT_TRACK_HORIZON`、`MAP_PROJECTION_FAIL`等可解释原因。静态规则仍保留：native adjacency、route compatibility、official runnability、初始target gap≥15 m及source/target reference覆盖canonical progress through 15.4 s。

动态规则不是初始15 m gap的替代品，而是其时间维度补充。

## 四、对Stage7L-B development的回顾性解释

动态算法先固定，然后对24个已用development token进行同一套离线replay audit；碰撞标签只在事后用于解释性汇总，未用于调整buffer或规则。

| Development group | Count | Dynamic rejected |
|---|---:|---:|
| 五档固定责任碰撞场景 | 4 | 4 (100%) |
| 未碰撞场景 | 20 | 11 (55%) |

4个固定碰撞场景都被识别为`TRANSITION_CORRIDOR_DYNAMIC_CONFLICT`：

| Token | 首次冲突时间 | Replay actor | 原因 |
|---|---:|---|---|
| 5dfd9b207fba5c94 | 8.0 s | bfa0cf6387685a4f | transition corridor |
| 9d1244220148595c | 11.8 s | 87c9a1897278506e | transition corridor |
| bfa7583b4275539a | 4.1 s | db1e38a777c75cf6 | transition corridor |
| fbbe1a3968d15fb5 | 4.0 s | 84fc3f516e2c5184 | transition corridor |

因此没有发现这4个碰撞属于“换道包络之外”的collision。规则也没有追求20/20未碰撞场景全部保留：11个被拒绝的未碰撞development场景代表在原始replay下存在潜在traffic interaction，正是pure-lateral positive-control benchmark应主动避免的污染来源。

## 五、扩大Pittsburgh库存的结果

扫描只接受`us-pa-pittsburgh-hazelwood`，避免将输入中的Las Vegas记录混入最终池。

| 阶段 | 数量 |
|---|---:|
| 扫描DB | 1,621 |
| 扫描pre-treatment anchor | 34,782 |
| Pittsburgh静态eligible option/token | 327 / 327 |
| Dynamic-clean option/token | 165 / 165 |
| Pool A：scenario-disjoint token/log | 165 / 102 |
| Pool B：strict Stage7L-B development-log-disjoint token/log | 152 / 94 |
| Pool B left/right | 19 / 133 |

动态rejection中有150个transition conflict、7个source-lane conflict、5个target-front conflict。Pool B的152个候选全部满足official query/runnability、静态reference coverage和dynamic clearance；并排除了所有历史使用token，且其log与全部26个Stage7L-B development token所用log严格不重合。

现有pool足以支持未来80场景confirmation，同时仍保留72个token和14个unique-log以上的设计空间。left比例不追求50/50；19个left允许未来审阅时考虑约15 left + 65 right的pre-treatment分层，但本阶段不冻结比例或roster。

## 六、Stage7L-C候选协议（仍未冻结）

保持Stage7L-B已推荐的treatment：`60 / 58.5 / 57 / 55.5 / 54 m`，trigger 12 m，0.4 s planner horizon，`closed_loop_nonreactive_agents`。保留primary metrics：lane-change duration、RMS lateral acceleration、peak yaw rate；保留nuisance proposal：0.02 m/s、0.05 m/s²、0.10 m/s³和0.25 m。

Stage7L-C应将本B2的完整static + dynamic eligibility stack、failure policy、sample size及统计规则写入新的prospective protocol；任何runtime failure、invalid/incomplete、collision或off-road都不得在confirmation中作为post-treatment样本筛选条件。机制/安全门禁失败时，应停止后续representation evaluation。

## 七、科学边界

本结果只说明：在固定replay background下，可以以完全pre-treatment、dose-independent方式筛出一批足够宽裕的clean lane-change opportunities。它不证明任何BDD、representation或planner style结论；也不宣称真实部署系统可以预知未来15秒交通轨迹。
