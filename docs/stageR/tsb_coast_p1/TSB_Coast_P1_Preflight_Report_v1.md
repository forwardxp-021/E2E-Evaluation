# TSB-Coast P1 Preflight Report v1

最终状态：`TSB_COAST_P1_PREFLIGHT_BLOCKED`。日期2026-10-23（Asia/Shanghai）。基准HEAD：`8258a1bebdad980989821697670bb1173e942180`。

| Gate | 状态 | 证据 |
|---|---|---|
| Owner授权来源/C2唯一性/条件范围 | PASS | Owner_Instruction副本及Authorization；仅C2，无C1/C3/grid |
| 旧B1/S1/S1R/输出保护 | PASS | Protected_Baseline与最终hash审计 |
| 80帧/时间戳/边界/非法输入阻断 | PASS | 22项`test_tsb_coast_p1_measurement.py` |
| 理想机制辨识及恢复排除 | PASS，仅fixture | Analyzer实现及单刹、coast、拖曳、正恢复测试 |
| raw/median3、edge2/2机制稳定性 | PASS于规范及16个固定噪声条件 | resolution audit；不作真实轨迹稳定性主张 |
| 新F与旧公式一致性 | PASS，仅fixture | synthetic-state production-parity test |
| ±.01m/s误差下新F预算分辨率 | **BLOCKED** | mean_abs_accel误差.075～.0825>.02；恒速null=.10 |
| 可替代的实际生产误差上界证据 | INCONCLUSIVE | float无.01舍入≠实际总误差上界；未建立替代包络 |
| 新generator/planner/factory/固定平滑 | NOT_REACHED | 前置测量gate失败，未构造生产组件 |
| 三SESSION唯一名单和route静态预检 | NOT_REACHED | roster仅header，未选身份 |
| 运行时callback、serializer、官方安全 | NOT_REACHED | 没有runner构造或metrics |
| 持久化CPU/attempt执行监控及环境绑定 | NOT_REACHED | ledger为0且执行入口不存在；不能称已准入 |
| 科学执行激活 | BLOCKED | 无任何arm进程启动，无技术retry |

阻断是测量误差包络与F精度预算的兼容性，不是C2科学FAIL、基础设施事故或近零机制实际失败。全部授权首run仍未启动；后续实现和样本审计没有“默认通过”。本轮不靠测试PASS替代真实科学结果，也不把压力样例当真正nuPlan误差。

零运行实现轮次2/2；第一轮实现测量并完成19项测试，第二轮补齐浅波纹、短gap及半采样边界，最终22项。第二轮未修改analyzer或阈值；发现阻断后不继续实现生产组件。最终收尾只做文档、只读hash和Git，不新增科学版本。
