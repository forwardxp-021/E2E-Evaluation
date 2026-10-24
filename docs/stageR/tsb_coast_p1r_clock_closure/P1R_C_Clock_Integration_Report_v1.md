# P1R-C 时钟集成报告 v1

2026-10-24，基线8c6a3ad；分支20260825_stageR_new。新旧源码、配置、数据/运行环境SHA256见Execution_Source_Binding和Manifest，历史保护基线见evidence。唯一修改范围是新Coast时钟/factory/授权适配及其测试、证据和文档。

## 集成结果

`tools/tsb_coast_p1r_c_clock.py`及独立Hydra clock组为唯一模拟时间权威。`tools/tsb_coast_p1r_c_production.py` 在正式Hydra `build_simulations`过程中选择新clock，继承原C2 planner、原zero-request adapter、原官方LQR/TwoStage/model及原严格命令到model记录器。Baseline不启adapter；Treatment仅[2.1,3.9)解除纵向LQR/停车控制并输出严格零请求。

旧记录器可选time-controller参数只识别旧类名。新factory先精确校验新clock类型、anchor、setup/observer/controller场景身份及完整映射，再安装原记录器核心并明确绑定同一新clock。没有删除旧类型校验、修改旧类、伪装类名或允许任意时钟。model实际dt由权威current/next iteration传入，原记录器仍校验实际packet/输入state对象身份、固定步长、官方滤波方程及79次完整推进。

新增 `tools/tsb_coast_p1r_c_executor.py` 是默认关闭的未来首对入口，复用已有累计claim/parent ticket和六步官方metric/serializer/hash生命周期。新stage凭证、Owner明确执行批准、观察时间差披露确认、合同/source/config/data/preflight hash均须通过；旧凭证和PROPOSED_ONLY在claim/runner之前拒绝。没有用科学合同批准文件代替执行凭证。累计budget的审计快照在本目录，未来可变工作账本独立放在`outputs/stageR/tsb_coast_p1r_c_execution_state_v1/budget.json`，从该0-attempt快照继承；本轮未claim，未来授权执行不会覆盖历史报告快照。该入口只允许原DEV-01两个arm，Baseline通过后才能Treatment；任何失败停止，无自动重试/替换，首对结束停止，不能进入DEV-02/03。

## 真实对象零运行预检

固定原三个SESSION及原六spec，未选新候选。初始与验证阶段预检保留用于追溯；完整最终证据为 `evidence/production_verified/ThreePair_Preflight_v1.json`。它核对实际DB/map/shard hash、原81个日志时刻逐值不变、六个独立官方runner/planner/controller/tracker/model/recorder、相同paired precontext/route、真实callback/metric/serializer配置、finite-route初始查询及79步时钟语义。

生产对象在guard下构建，禁止runner run/initialize、Simulation initialize/propagate、正式planner、controller update、model propagate、metric compute/write及serializer/finalizer运行。guard计数全部0，新记录器行数和history均0。时间控制器next/reset仅用于纯索引接口检查，不推进真实SESSION状态。guard下构建的对象均丢弃，不可在解除guard后拿来执行；未来必须重新构建fresh arm。

## 官方模型及纯fixture

`tools/tsb_coast_p1r_c_fixture.py` 用有正负抖动的纯合成scenario metadata、规范clock以及真实官方controller/LQR/model进行隔离传播；Baseline/Treatment各80合成状态、79次model transition。它不读取真实SESSION，不进入runner，不是C2闭环科学结果。实际控制iteration全部由同一clock提供。

Treatment 18个零请求确实到model；输入负a不清零，按dt=.1、tau=.2的官方递推衰减；速度/位置持续前进；3.9秒恢复LQR。原adapter转向、边界跨步拒绝及命令检查测试继续通过。横向恒速预测仍是既有简化，不修改gain或增建车辆动力学。

另测planner 80次纯合成调用：每次未来80个时间点均从权威current time递增，first state是同一ego对象；前11个调用两臂整个future speed一致；每call只有一条current-ego trace。官方ego时间提取器读取合成state后仍得到同一canonical grid。没有执行正式callback/metric。

## D1/D2前瞻生效与边界

新的批准记录正式承接旧分层机制规格及四项F数值；旧提案、旧P1/P1R状态和压力反例永久保留。旧近零core、漂移、正恢复及响应稳定性诊断仍计算/保存，不能取代已批准的零命令+滤波异常gate。未改变两次真实制动/间隔/剂量、低速或官方安全门槛。

人工±.01测试是独立鲁棒性诊断，不自动代表生产误差；raw速度导数、整数时间处理、六位舍入和.02 caliper不变。若出现真实量化、随机误差或数值不稳定，必须重新阻断。映射只建立本抽象回放的可追溯性，不自动证明安全/机制/F合格。

## 历史与科学防火墙

旧B1保持20/20 PASS、40完成arm、41累计attempt、1技术retry、0科学失败；低阶nuisance消除及clean residual均NOT_ESTABLISHED。原P1两轮零运行实现不重置，本轮只用旧三对；全P1科学预算3pairs/6首次arm/8attempt/10800CPU秒不扩大。S1/S1R、H30、U-only及Primary合同不变，Coast未授权替换Primary。

本轮科学attempt、RBR训练/推理、H30、MMD/BDD、Primary/V/C均0。当前仅准备首对Owner审阅，执行凭证仍PROPOSED_ONLY。
