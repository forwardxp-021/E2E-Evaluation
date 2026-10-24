# P1R-D 实际时钟及观察审计 v1

规范控制步长100000us；同一t0，80实际状态/79区间，两个arm物理时钟相同；81行原始metadata映射完整保存在各arm canonical_clock_mapping.json，terminal metadata不冒充第81帧科学结果。实际80帧逐帧证据见evidence/BASELINE_Actual_Observation_Audit_v1.json及TREATMENT对应文件，包括simulation_time_us、raw_log_time_us、raw_minus_sim_us、observation_index、lidar_token。

| arm | 实际帧 | 偏差范围 us | 晚/早/同刻帧数 | 审计 |
|---|---:|---:|---:|---|
| BASELINE | 80 | -330 / +323 | 32 / 47 / 1 | PASS |
| TREATMENT | 80 | -330 / +323 | 32 / 47 / 1 | PASS |

真实官方SimulationLog已只读反序列化：逐帧index、模拟ego时间、raw源时间/token对应；实际观测与原始同index源的actor标识/track token/timestamp/box center XY/heading、交通灯status/lane/timestamp逐值一致；actual/source actor均无future predictions。此指纹的字段范围明确，不宣称检查了所有actor属性。实际manifest绑定的产物无哈希改变。源码与装配检查共同确认未启future-waypoint观察通道；没有新增插值、换帧或查询更晚index来作决策。

`DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY`。

部分raw观察晚于模拟时刻，最大323us；最早−330us。时间相近不能推出严格因果，也不能证明碰撞/边界阈值附近安全标签完全不受偏差影响。当前C2纵向决策不消费这些动态观察，但官方安全计算会涉及其他对象。仅用于B-development，不是严格在线事件时因果验证、P2或量产端到端策略评价。没有用预检结论豁免实际序列审计。
