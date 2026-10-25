# P2-Q1 — 最终执行预检 v1

Scientific Owner于2026-10-25明确批准TSB_DIRECT_COAST_P2_SCIENTIFIC_EXECUTION，以ad684593b9c62b88d5b435a42e2a9b04456241a9中Q0固定20名单和F2合同为条件。原始指令逐字保存在P2_Q1_Owner_Instruction_Original_v1.txt，SHA256为573645e5d1781aefa5ec3b432a7ce4728cb8abba5d5bcf64f34fe7e2fee327c1。

## 范围与冻结顺序

本轮只增加tools/tsb_direct_coast_p2_q1.py（授权/身份调度/持久化预算/进程监督/停止）和tools/tsb_direct_coast_p2_q1_report.py（只读结果汇总），以及故障注入测试。未更改F2、原官方源码、测量或阈值。

先冻结调度/汇总源码、测试、Owner原始指令、保护基线与零运行证据为source commit；然后Source Binding引用该commit和实际SHA256；独立正式Owner Authorization再引用Binding的SHA256，单独提交后才可启动。Authorization不会包含自身哈希或后续结果哈希；入口逐文件验证代码/数据/协议、HEAD中已提交Authorization与工作文件一致，避免循环哈希。后续结果Manifest在执行停止后生成。

资格机制、四F、额外剂量与峰值、官方安全和80状态/79推进全部调用原F2 lifecycle/analyzer，不复制科学判定。资格合同依F2_Owner_Protocol_v1.json的qualification部分；该历史文件关于DEV01、旧预算和P2禁止的历史授权范围不复用，新身份/预算由本轮独立Owner凭据覆盖。

冻结F2 audit_f2_observations内部最后一处证据输出固定指向历史F2目录。为保持其全部观察核验逻辑且不触碰历史文件，新调度子进程仅在该调用期间将模块OUT证据目的地指向本arm独立observation_review目录，finally恢复。没有改写F2源文件，没有更改clock、观察、测量、控制或模型行为。每arm目的地独立，旧F2证据只读保护。

## 最终零运行检查

详见evidence/P2_Q1_Final_Preflight_Evidence_v1.json和P2_Q1_Zero_Run_Tests_v1.json：原Q0 40/40装配预检、全部数据/配置/映射哈希再次核对；不重复40arm factory，不调用runner/model/controller/metric。原20 SESSION顺序、同pair身份/precontext/route/初态保持；Q0逐SESSION clock映射与原配置逐值绑定。完整原排名/候选/失败元数据证据继续保留。

10项零运行测试覆盖真实B1/C2/F2/Q0凭据拒绝（至少200组合）、并发重复claim、领取后中断、科学/技术失败停止、CPU不确定、输出碰撞、顺序/预算、pair失败阻断下一SESSION、全部20对才能成功，以及一次性调度故障注入。测试使用人工账本，不调用科学对象。

## 预算与监督

40个原run ID各一次；累计42仅天花板；technical retry=0；CPU总57600秒、wall1200秒/arm。先以锁和fsync持久化领取，再Popen独立进程。重复/乱序/半完成claim、现有state目录或输出、预算不确定均fail closed。没有resume/reset/retry命令。只读credential失败在claim之前拒绝；初始化后任何异常停止全局账本。

每arm监督wait4子进程及已回收后代CPU，与按PID+创建时间累计采样的活后代CPU最大值取较大者，保留计量记录。原配置worker=sequential、callbacks无并行，避免孤立子任务。若存在退出后仍存活后代、监控异常或计量不确定，则kill进程组、技术阻断、不得继续。进程异常即停止，无自动retry。

Baseline先运行，技术/机制/安全/真实观察和artifact完整后才Treatment；Treatment同样通过后只调用一次原analyze_completed_pair。首个arm或pair科学失败立即停止；技术失败立即停止；后续全员保留在固定20分母中NOT_RUN_AFTER_STOP。结果仅来自真实已启动arm；正式F_match不得在arm gate失败后执行。

## 科学边界

角色仅B_BENCHMARK_QUALIFICATION，原E1/E2保留，实际启动追加E2，不重标E5。旧B1 20/20、C2 FAIL、F1和F2 Development PASS保持。所有5个E5、旧B1及DEV01/02/03均不在新名单。无RBR/H30/BDD/MMD/Primary/V/C；不以资格表现选择encoder/loss/checkpoint、Primary样本或成功标准。

`ABSTRACT_COAST_PROXY_ONLY`。官方碰撞、TTC及观察以DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY解释。Q0全名单偏差范围−1014～+1443微秒，实际运行只报告实际样本自己的映射。20/20不等于总体通过概率100%、未见SESSION Claim-A或Primary统计功效证明。
