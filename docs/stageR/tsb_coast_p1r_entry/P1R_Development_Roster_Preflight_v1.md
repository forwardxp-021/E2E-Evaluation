# P1R 唯一三对Development候选与预检 v1

名单：`P1R_Development_Candidate_Roster_v1.csv`；六个spec：`evidence/Arm_Specs_v1.json`。状态全部 `CANDIDATE_PENDING_OWNER_CONTRACT_FREEZE`，不是已授权/已冻结执行名单。来源、数据库完整SHA256、静态shard SHA、代码与依赖见Manifest/Metadata_Selection。

## 固定outcome-blind顺序

仅使用既有静态census的 `STATIC_ELIGIBLE`、初速≥5m/s且E2_BENCHMARK_ENGINEERING的代表条目，合计54。排名 `SHA256("TSB_COAST_P1R_BDEV_C2_V1|session_id|representative_scenario_token|representative_log_id")`，按hash及session_id排序取前三；每SESSION一对，先Baseline后Treatment。仅核查元数据身份/初态/路由，不使用B1成功率、曲线、机制、F、safety、H/RBR结果。没有因时钟失败替换候选。

| pair | SESSION | scenario token | 读取初速m/s |
|---|---|---|---:|
| DEV-01 | 2021.09.17.14.49.23_veh-42 | 9fe700abf47057f0 | 14.151694 |
| DEV-02 | 2021.08.23.17.20.10_veh-45 | 2dc6aa6afd2d5eff | 6.006798 |
| DEV-03 | 2021.09.28.13.24.06_veh-44 | 1d5769ba8a5357a8 | 10.716628 |

同pair spec共享DB/scenario/session、initial timestamp、precontext hash和route hash；实际factory再用官方接口解析唯一scenario并核对precontext。不把不同token自动当成SESSION独立。

## 有限路由与实际配置

选择时用真正 `build_native_route_reference_v1_2` 对两臂初始common未来距离有限查询：DEV01 99.0093826m，DEV02 34.6647042m，DEV03 71.8723612m；均无extrapolation。Metadata_Selection初次输出的 `builder` 字段是返回值里的旧内层标签 `build_native_route_reference_v1_1`，**实际调用是源码可核对的v1.2入口**，不是切回旧builder。该字段差异在此披露，不静默覆盖初次证据。

这是初始有限查询，不是未来80步全部几何兼容证明；动态replan超出可用route时必须失败，不能外推救实验。未访问未来真实ego轨迹或执行planner。

真实Hydra对象6/6可构建，各arm有全新runner/planner/controller/tracker/model/recorder，无跨臂复用。resolved configs和binding原件位于 `outputs/stageR/tsb_coast_p1r_zero_run_clock_v1/<run_id>/`，其SHA在最终Manifest中绑定；工程输出不提交。回调/serializer和静态安全链通过，实际日志时钟6/6 BLOCKED，明细见Production_Clock_Preflight。

## 角色与暴露

仅B-development目的，既有E2工程暴露。此次新增的是元数据/配置审计，无新benchmark outcome、detector-development结果或Primary outcome暴露。未创建、筛选或解盲V/C名单。既有S1R允许按claim登记历史B复用；E2不能被包装成Claim-A未见SESSION泛化，亦不擅自宣布E2永久禁止Claim-B V。E3/E4的claim相关Primary V排除规则不变。U-only encoder selection、H30、SESSION cluster inference、原统计合同保持。

## 冻结首对还缺什么

D1代理机制新合同和D2压力测试角色待Owner批准；D3需要明确精确模拟时钟与日志观察时间如何对应，再做有限时钟接口修复和零运行核验。当前三对全有跨界控制步，不允许凭候选已列出就申请启动。没有科学attempt，旧P1预算不重置。准备阶段构造次数不是科学attempt；正式attempt仍0。
