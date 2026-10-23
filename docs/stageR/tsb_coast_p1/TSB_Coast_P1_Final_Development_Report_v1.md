# TSB-Coast P1 Final Development Report v1

## A. Overall Verdict

`TSB_COAST_P1_PREFLIGHT_BLOCKED`。P1的条件执行授权未激活。不能报告DEVELOPMENT_PASS、SCIENTIFIC_FAIL或真实闭环不可行。

## B. 大白话

**这轮尚未看到实际“刹车—滑行—再刹车”，因为在开跑前就发现测量预算不能按现有要求通过。** 只给规范速度加上P0指定范围内的1厘米/秒误差，原始加速度绝对值均值就能变化超过固定匹配容差。不是网络不好、不是C2已经失败，也没有证据证明nuPlan实际误差就是这么大。

## C–E. 真实曲线、F_match和官方安全

实际pair/arm/attempt均0；两次减速峰、near-zero支持、速度漂移、正恢复、实际四项F、剂量/峰平衡、碰撞、drivable、low-speed全部NOT_RUN/NOT_EVALUATED。没有合成所谓真实PNG/SVG，没有用参考代替真实车辆，也没有重画旧B1冒充新样本。

测量实现覆盖两段时长/峰/剂量、gap/core、漂移及正恢复排除。固定16个误差条件的机制稳定性没有触发MEASUREMENT_UNSTABLE；阻断来自raw mean_abs_accel：同一C2参考输入误差.075～.0825m/s²，大于.02。独立恒速null从0到.10。细节及保守准入解释见FMatch合同；这些是非科学fixture。

## F. 预算

实施轮次2/2；科学attempt0，首run许可使用0/6，累计attempt0/8；科学arm CPU=0秒、全局3CPU小时尚未使用，无单arm运行。纯fixture/阅读/测试开销不冒充0，准备总CPU未完整计量，且正式CPU监控尚未实现，属于NOT_REACHED gate。本轮没有技术retry或剩余预算重置；未来不能把8上限解释为自动retry授权。

## G. 失败分类和未完成工作

没有真实科学失败，没有runner/route/callback基础设施失败。唯一实际阻断是测量分辨率审查；Actual production error bound未建立。生产generator/planner/factory、新授权运行时拒绝测试、静态路由、三SESSION冻结名单、预算监控未到达，不能省略这些门。

## H. 防火墙和历史资产

新科学暴露=0；只读已有源码/协议和hash、纯数组测试。H30、RBR64、MMD、BDD、比较、训练、Primary/V/C全部0。新SESSION=0；不将本次误差诊断用于encoder或H设计。旧B1始终20/20 PASS、40完整arm、41累计attempt；旧TSB、参数和S1/S1R均不修改。

`LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED; TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED`。原Primary的ΔBDD/.05/m20/π.50/δ*.10及H30不变，Coast未成为已批准替代干预；S3-P0阻断仍在。v2.4只保持原文/hash与SYNC_PENDING，不擅自覆盖。

## I–J. 代码、测试、来源与Git

新增一个纯测量工具及22项测试；新文件与Owner原文、误差JSON、保护基线、NOT_RUN表格由Manifest逐项绑定。已做Python编译、临时依赖检查、diff检查及只读真实历史资产hash核对。实际科学real-data smoke未运行，原因是Owner要求的预检门未通过，不是本地没有数据。最终commit/push/远端SHA在Work最终反馈中报告，Manifest自身由最终Git tree绑定。

## K. 只建议一个下一阶段

**Scientific Owner review → 零运行测量误差合同审查。** 明确P0 ±.01压力测试与实际生产误差上界的关系、.02 F预算能否在经证据支持的测量范围下得到认证。不是放宽阈值、换导数、换C2或开一对探索运行；若需改合同，必须前瞻授权。

完成后停止。TSB_COAST_P2、PRIMARY_PROTOCOL_AMENDMENT、RBR_TRAINING、V_EXECUTION、C_EXECUTION全部NOT_AUTHORIZED。
