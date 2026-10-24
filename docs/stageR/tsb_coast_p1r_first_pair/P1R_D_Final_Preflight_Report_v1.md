# P1R-D 最终执行前检查 v1

状态：`PASS`。执行源码冻结于`73e96f7e6fa1c39bca0f2cd0f9f2947646024c93`。本文件及独立Owner授权的包含提交属于授权记录提交，不能冒称生产源码的新版本；完整逐文件SHA256绑定在授权及原C执行来源包中。

本轮Owner原指令见`evidence/Owner_Instruction_v1.txt`。独立新记录经过CONDITIONAL_PENDING_FINAL_PREFLIGHT→最终双臂预检→冻结credential校验后成为AUTHORIZED；旧PROPOSED_ONLY文件原样保留。只授权DEV-01 Baseline→Treatment各一次首次执行，Baseline任何技术/科学失败均不启动Treatment。DEV-02/03、retry、改C2及RBR/H/BDD/Primary/V/C均禁止。

## 最终15项检查

| 项 | 状态与证据 |
|---|---|
| source/config/data/runtime | PASS；1068源码/config文件，74仓库blob与73e96f7匹配，10数据来源，Python及全部依赖版本一致 |
| 初始身份 | PASS；SESSION 2021.09.17.14.49.23_veh-42，token 9fe700abf47057f0，两臂未舍入实际初态逐值相同 |
| precontext | PASS；`17c78f356097e5a8a7acc4dcb0c09a22df787df908417348c485f0f6586fd4b5` |
| route | PASS；`c65e8c8269b52a632bf87230412e0ce19bd35ef1dcaef43d6993eaf4fd47778d`；原生有限初始查询通过，不冒充未来完整episode验证 |
| 规范时钟 | PASS；同一CoastCanonicalTimeController，t0不变，100000us步长，80帧/79transition |
| 观察映射 | PASS；原81个metadata时间/token与C完全一致，同index未插值/换帧；DEV01范围−330～+323us |
| future通道 | PASS（源码/装配）；原TracksObservation默认不取future waypoint，C2不以动态观察反馈控制；实际运行后核查serializer对象 |
| coast区间 | PASS；[2.1,3.9)，18控制区间 |
| Baseline tracker | PASS；LQRTracker，不启coast |
| Treatment tracker/model/recorder | PASS；ZeroLongitudinalRequestTrackerV1包装官方LQR，官方KinematicBicycleModel，实际命令→model被动记录器绑定 |
| callback/safety/serializer/finalizer | PASS；正式官方链装配完整，尚未执行正式指标 |
| fresh目录/对象 | PASS；两个arm所有对象独立；科学输出目录尚未使用；guard对象丢弃，真实执行须重新构建 |
| D1/D2及配置 | PASS；实际analyzer逐项等于已批准合同；full cfg与C只允许output namespace和run-start experiment_uid变化，其余逐项一致 |
| 预算 | PASS；历史Coast attempts=0、CPU=0，新授权最多2次；全局3对/6首次/8累计/10800CPU秒，每arm1200秒；不重置 |
| DEV-02/03拒绝 | PASS；冻结credential明确拒绝代表run ID，完整白名单仅两个DEV01 arm |

所有guard调用为0，history及recorder行数为0。独立预检完整JSON见`evidence/Final_Pair_Preflight_v1.json`，SHA256 `86c7086ec28e7d84db59168cf8284b00848c9c04dd9f92660e0483de8cc74d4f`。这只是进入runner前条件成立，不预设真实科学PASS。

## 科学合同优先级

已通读本地v2.4原文；文件未进Git且仍是B1前17对/IR2的旧整合快照，记录`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，不覆盖原文，不以其过时状态推翻后续批准。S1R、当前D1/D2/D3及本轮Owner执行指令优先；旧近零core/漂移/正恢复仍计算但只作诊断，不重新作为资格门槛。

DEV01同index观察可能晚于模拟时刻323us、早330us；这是DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY。碰撞等阈值附近标签可能敏感；本次是B-development，不是P2、Primary或量产E2E因果评价。
