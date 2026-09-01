# R1 B2.9-B 路线连续原生参考合同报告 v1

## 结论

新合同将 HLC 的单一原生车道引用替换为官方地图原生、成对且路线约束的连续走廊。V2.1、V2.2 与冻结机制均未修改；不存在外推、手工点、最近距离择路或结果驱动择路。

## 确定性规则

从冻结 source/target lane 与其在 route_roadblock_ids 中的唯一 occurrence 出发，枚举双方 native outgoing edge；只保留源分支在冻结路线后续唯一出现、方向连续且双方终点 lane 保持官方相邻关系的组合。候选不是恰好一个即 fail closed。连接处没有发现既有数值精度阈值，因此合同记录为 UNKNOWN；实现只接受 gap 精确为 0 的官方端点身份，不引入容差。

## Attempt 1 离线修复

34 行历史 realized trace 全部离线通过。iteration 33 不再出现 coverage failure；source 组件 `18524 → 20156`，总长 `169.411027 m`、requested max arc `145.922446 m`、余量 `23.488581 m`；target 组件 `18525 → 20157`，总长 `169.565122 m`、requested max arc `146.078881 m`、余量 `23.486241 m`。两处 join gap 均为 `0 m`。iterations 0...32 与旧 builder 输出 exact parity，34/34 current-ego state0 exact identity。

## 当前 12 个科学 HLC 身份

本项仅为 DIAGNOSTIC_ONLY；`12/12` 完成 0...79 双 arm 滚动覆盖，拓扑歧义 `0`。未创建或修改科学 roster。工程 canary 使用同一 nuPlan StepSimulationTimeController 的版本化 80-call 上限，使 full runner、controller、observation、metric 与 callback 在 iteration 79 后正常结束；没有为 Primary 窗口之后的地图终点发明非原生连接。

## 版本差异清单

- V2.1 与 V2.2 文件未修改；V2.2→V3 的 Primary 轨迹语义唯一变化是单 native lane reference 改为 route-continuous official-native reference。
- V3 的被动 trace writer 将 0...79 写入 Primary，>=80 仅写入独立 secondary diagnostic；该分流不读取或改变 planner trajectory。
- 工程 canary 专用 time controller 仅把 runner 结束点限定为 80 次 planner call；TwoStageController、observation、ego propagation、metric engine 与 callbacks 保持原绑定。
