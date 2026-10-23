# TSB-Coast P1 Development协议 v1 — 测量预检阻断，未进入执行冻结

本文件记录Owner批准的不可事后调整约束和本次前置核查结果；文件名中的Frozen**不表示完整生产栈、样本和授权绑定已经冻结可执行**。

`OVERALL_STATUS=TSB_COAST_P1_PREFLIGHT_BLOCKED`；`EXECUTION_AUTHORIZATION_ACTIVATED=false`；`FIRST_SCIENTIFIC_ARM_STARTED=false`。

## 权限来源与前置顺序

Owner指令副本为`evidence/Owner_Instruction_v1.txt`，源附件及SHA见Authorization/Manifest。Owner授权唯一`TSB_COAST_C2_P1`、3对B-development/6次首次执行，明确以测量、实现及执行审计通过为前提。本包先核查测量，未通过即停止，未利用剩余实现轮次放宽判据或启动runner。技术retry并未自动授权；预算8次只是上限，额外每次须重新获Owner许可。

1. 测量实现/边界fixture及误差预算核查（已做，误差预算BLOCKED）。
2. 独立generator/planner binding、固定平滑、生产factory、route/callback/serializer、CPU执行监控和授权拒绝测试（NOT_REACHED）。
3. 确定性身份选择和唯一3SESSION roster、全部代码/config/runtime/data hash绑定、正式最终预检（NOT_REACHED）。
4. 每arm独立OS进程、六步生命周期、真实曲线/配对判定、第一科学或技术失败立即停止（NOT_RUN）。

第2–4项没有假称通过，也没有调用旧B1 factory来冒充新绑定。保护旧B1/S1/S1R优先；无需为了补齐实现组件而越过已失败的测量gate。

## 不变的C2及判据

Treatment参考：−1.305m/s²@[1.1,2.1)，0@[2.1,3.9)，−1.305@[3.9,4.9)；Baseline：−1.45@[2.1,3.9)。两臂冲量−2.61m/s、时间重心3s。不得改候选、SESSION专属参数、适用域或按实际结果挑平滑。本轮只用P0解析函数生成**非科学测量fixture**，没有实现或调用新的闭环generator/planner。

双制动、滑行及正恢复排除按本包Analyzer文档实施；四项F_match及额外剂量/峰平衡保持P0/Owner数值，见FMatch合同。代码、测量和参数均未在科学arm之后调整，因为实际科学arm=0。低阶因素消除与clean residual继续NOT_ESTABLISHED。

## 样本与预算合同

本轮名单文件仅有header，实际选定SESSION=0。测量gate先失败，不生成虚构身份或最终V/C名单。未来若重新审查通过，身份必须按预先写定、outcome-blind的元数据排名选择3个SESSION，初速≥5m/s、同pair precontext/route相同、静态地图/finite-route兼容、S1R exposure合规；不能用历史成功率选取。本次未冻结具体排名salt/身份，不能声称后续无需再冻结。

`PAIRS_MAX=3; INITIAL_ARMS_MAX=6; CUMULATIVE_ATTEMPTS_MAX=8; ZERO_RUN_IMPLEMENTATION_ITERATIONS_MAX=2; CPU_TIME_CEILING=10800s; PER_ARM_EXECUTION_CEILING=1200s`。本轮两轮零运行实现/边界测试，0个正式attempt；科学执行CPU=0，8次总上限未消耗，但没有额外retry许可。

若未来执行，CPU计量须从独立arm子进程启动（含生产构造、runner及收尾）累计user+system及后代CPU；父监督进程还须单调时钟20分钟超时、全局3CPU小时限制及持久化锁定ledger。启动前扣/登记attempt，不因异常或重启返还；中断无法核对CPU时应fail closed，不能重置为零。本次尚未实现/验证该执行监控，标NOT_REACHED，**不声称CPU gate已通过**。纯fixture和编译的准备开销与科学arm CPU分开，准备总CPU未完整计量；0只指未启动的科学执行。

## 停止原因及含义

P0明确要求±.01m/s速度噪声核查，并要求误差预算不能被测量误差占满。本次固定period4的±.01扰动，使同一参考输入的原始导数mean_abs_accel误差.075～.0825，大于.02；恒速null误差.10。这里采用的是**对P0指定误差包络进行保守准入核查**，并不宣称Owner已认定nuPlan实际误差为±.01，也不把纯噪声敏感性等同真实平台不可实现。

结果可逐字节重复，问题是误差稳健性而非代码随机性。源代码保留float速度且不按.01舍入，实际误差可能远小于该压力包络；但本轮没有建立可替代的生产误差上界，不能私自收窄既定核查范围来准入。最小下一步是Owner审查测量误差证据/原要求含义，维持.02及C2不变；不是自动新增rollout或把raw derivative换median3。

## 防火墙

旧B1仍20/20 PASS、40完整arm、41尝试。新数据角色仅B-development；本次无新SESSION outcome暴露。H30/RBR64、MMD/BDD、训练、Primary/V/C均0。TSB_COAST_P2、Primary修约、训练及V/C继续NOT_AUTHORIZED。v2.4原文未改，`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`。
