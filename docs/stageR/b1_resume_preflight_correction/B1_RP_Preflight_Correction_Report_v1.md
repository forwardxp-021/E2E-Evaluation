# B1-RP Preflight Correction Report v1

主状态：`B1_RP_PREFLIGHT_PASS_READY_FOR_OWNER_REVIEW`。

正式B1保持 `B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE`。本轮修复预检、验证真实静态装配；没有运行剩余六次，没有宣布B1 PASS。

## 修复与源码边界

旧gate误查共享callback列表；官方builder将有状态回调另行绑定到Simulation。新规则直接核验真实simulation与common_builder两个回调所有者、官方具体类型与方法绑定、engine/finalizer配置和输出路径、原recorder、serializer及六步生命周期。七个根因问题和源码位置见 B1_RP_Callback_Root_Cause_v1.md；正式顺序及静态证据边界见 B1_RP_Callback_Lifecycle_Contract_v1.md。

原resume文件只在preflight函数有4行差异。AST证明其余模块/函数不变；B1正式executor、canonical factory、planner/generator/controller、recorder、analyzer、安全adapter、schema、IR3 route代码均未修改。没有改变正式callback生命周期。新增工具只有零运行检查选项，不提供科学执行选项。

## 原顺序真实生产预检

| 顺序 | 原冻结执行项 | 实际零运行结果 |
|---|---|---|
| 1 | B1-TSB-18-TREATMENT / ATTEMPT_2 | PASS |
| 2 | B1-TSB-18-BASELINE / ATTEMPT_1 | PASS |
| 3 | B1-TSB-19-TREATMENT / ATTEMPT_1 | PASS |
| 4 | B1-TSB-19-BASELINE / ATTEMPT_1 | PASS |
| 5 | B1-TSB-20-BASELINE / ATTEMPT_1 | PASS |
| 6 | B1-TSB-20-TREATMENT / ATTEMPT_1 | PASS |

每项使用原roster/spec/scenario/session、precontext/route和未变的正式factory/配置。IR3 route全部COMPATIBLE，真实callback、静态收尾链、recorder、serializer、输出namespace和原预算合同全部通过。CSV以SHA绑定每份live-object证据和resolved配置，见zero_run_evidence/。

原配置源码字节保持一致，仅诊断输出namespace与自动时间标签不同；上轮在导出resolved cfg前已阻断，本轮不伪造旧resolved config哈希对比。这些attempt编号仅标识原计划检查项，没有创建实际ATTEMPT_2 execution manifest或任何科学执行记录。

## 正反测试与证据限制

16项matrix全部符合预期：4个正常条件PASS，12个错误条件BLOCK。故障来自真实正式factory的已装配对象；另从Hydra配置删除MetricFileCallback，经实际canonical factory/builder构建，验证真正缺失finalizer时仍阻断。没有使用手工理想对象替代正式装配。

源码冻结前定向回归51 passed、0 skipped、0 deselected；含8项新增真实factory关键测试。旧审计工具的回归在隔离副本中核对冻结报告重复性，不写回历史资产，不作为本阶段新科学判定。本阶段正式科学状态直接继承IR2，没有重新计算机制或生成历史官方安全指标。

全tools编译、临时依赖检查、diff whitespace检查通过。第三方deprecation/map cast warnings无失败。静态检查只建立 `STATIC_PREFLIGHT_VALIDATION=PASS`；`RUNTIME_METRIC_FINALIZATION=NOT_EXECUTED_NOT_PROVEN`，不保证未来指标生成、完整闭环或科学结果成功。

## 预算、历史与科学合同

HISTORICAL_ATTEMPTS=35；NEW_ATTEMPTS=0；CUMULATIVE_ATTEMPTS=35。原34 complete arms、17 complete pairs保持不变；18 treatment原ATTEMPT_1基础设施失败保留；另5个arm仍NOT_RUN。935个原B1/IR/IR2/IR3/B1-R文件（包括旧授权、旧failed preflight和两份预算）前后SHA相同。

20对/40臂、selection salt、TSB参数/timing、mechanism/measurement/F_match及阈值、安全、LOW_SPEED_ENDSTOP、初速/horizon、路由与precontext均保持原哈希。all-20成功标准、科学失败继续及基础设施失败停止规则均未变。

保护资产SHA256：`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## 授权与源码绑定

修复源码commit `5e5469b2a8a53140ed12de2b275cba0c7af8379f`。完整409个源码/config/test条目及前后哈希见Source Hash Transition与新草案；原24648个依赖及5个关键数据文件保持原实际绑定。

旧授权SHA `f4944458eb82501f9670f75afba96dff7f42144cbd99695538b4f709b7893669` 未变。实际旧worker入口在factory前返回 `HASH_MISMATCH:source_file_hashes:tools/b1_resume_execution.py`；factory/runner/budget均0。旧B1-R阻断没有重开或覆盖。

新草案显式 `AUTHORIZATION_STATUS=PROPOSED_ONLY; OWNER_APPROVAL=PENDING; EXECUTION=NOT_AUTHORIZED`，新增执行权限预算0，authorized_source_git_sha=null。执行validator拒绝PROPOSED_ONLY；它只是下一次Owner审核的输入清单，不是执行凭证。

## Zero-run和下一步

所有生产科学入口、指标计算和正式收尾tripwire计数均0；新诊断namespace无科学trace/telemetry、metric parquet/temp或simulation-log。没有TSB/HLC rollout、RBR训练/推理、H/BDD/MMD/z64、Primary、V/C。

没有未闭合的静态预检blocker；未来运行与科学结果仍未知。只建议 Scientific Owner review → B1-R2 Frozen Resume Authorization → 获新授权后执行原剩余六次。B1_RESUME_EXECUTION、RBR_TRAINING和PRIMARY_EVALUATION仍NOT_AUTHORIZED。本轮结束后停止。
