# P1R 回调、记录器及生命周期审计 v1

本审计只读取正式对象和源码，未调用runner/initialize/propagate/metric compute/serializer finish。SHA及依赖见Manifest。完整入口结论以 `evidence/Production_Clock_Preflight_v1.json` 为准：对象绑定通过，实际时钟BLOCKED。

## 记录器身份和可观察链

`CoastTransitionRecorderV1` 精确校验官方TwoStageController/KinematicBicycleModel类型；Baseline必须是未包装官方LQR，Treatment必须是既有适配器且inner为精确官方LQR，区间和锚点严格一致。任意tracker、重复安装、错误clock/形状均不能无条件进入。

先在实际tracker返回处保存**最终DynamicCarState对象**及输入EgoState，再在同一真实model实例的 `propagate_state` 校验对象身份、dt和时钟；只有调用原model、拿到有限新状态并满足方程后，才原子写入完整transition。不是仅凭adapter metadata推断model收到了零。

每行包含iteration/time/next_time、BRAKING/COAST/TRACKING、requested acceleration、steering、shadow及其性质、model input/output a、model filter tau、speed before/after、后向速度差a、displacement、command identity reached、coast reached、state propagation。正式完成须恰好79条、index0..78、无未完成pending；对应planner 80个唯一真实current-ego。

coast shadow严格执行同一零请求规则及恒速横向LQR，不调用原longitudinal/stopping，不生成未执行的反事实a。非coast保留官方实际路径的shadow，实际/影子命令一致容差仍1e−12。此数值校验不证明踏板、压力或轮端扭矩。

合成模型79条证据已持久化。真实候选的记录器目前0条；真实curve observability必须在将来合法首对中验证，不能预先写成已完成。

## 真回调及正式安全链

实际构建的runner包含官方MetricCallback/官方MetricsEngine（15个直接engine，corners为间接依赖工件，最终应有16个parquet）、SimulationLogCallback(msgpack.xz)；main callback链有官方MetricFileCallback。逐项核对producer/finalizer的metrics路径、serializer路径、sequential pool/futures为空、新history为空、runner持有的planner/simulation和controller/tracker/model/recorder同一实例，记录器方法绑定未被替换。

collision与drivable-area采用既有官方输出canonicalizer及 `r1_b2_8_r3_1_official_safety_adapter`，不自算/简化指标。未来要求每臂0 at-fault collision、drivable=true。只验证源码与绑定，本轮没有生成parquet、safety outcome或模拟serializer产物来冒充正式完成。

## 未来生命周期与失败保护

仅未来新增明确授权且所有gate通过后：独立子进程构造 → runner →79记录完整性 →保存runner report →官方main finalizer →16个parquet schema及官方安全adapter →官方serializer唯一非空工件 →manifest →所有必需工件hash验证 →新分层机制/低速/官方安全 →paired F与附加dose/peak平衡。每步缺失/失败立即停止，无自动重试、替换或继续下一个pair。

新授权必须绑定：本栈stage、Owner批准的新合同、**完全一致的analyzer规格**、名单hash、首对两个完整spec规范hash、源文件/config/官方依赖hash、完整PASS预检hash、实际DB/map完整文件hash和唯一累计budget ledger路径。旧B1/P1授权、待批准合同、当前BLOCKED时钟预检不能激活新栈。

父监督先持久化attempt claim并文件锁，再起独立进程；child要求与父PID相符、唯一ticket且原子消耗，直接child调用不能绕预算。记录总CPU、子孙CPU和wall time；上限10800CPU秒、单arm1200秒，异常fail closed。budget进程成功/失败分支用无仿真Python子进程做fixture验证，**并未启动生产child**。异常未完成claim不会返还attempt或允许自动retry。monitor采样间隔.25秒，CPU限额由进程RLIMIT和父监督共同执行；这是监督实现，不是已测完整runner资源消耗。

首对Baseline合格后才可启Treatment，pair完成/失败即锁止；DEV-02/03仍需后续阶段授权。全P1上限3pairs/6first arms/8attempts保持，额外技术retry仍需单独批准。旧两次零运行实现轮次不重置，P1R是本轮明确授权工程任务。

## 故障测试和证据边界

测试真实未初始化factory删去main finalizer、替换model传播绑定或controller内model时均拒绝；guard计数为0。单独故障fixture覆盖未观察命令、缺transition、时钟错位、滤波异常、旧授权、待批合同、预算耗尽/已停止/重复attempt和脱离父监督的child。

没有运行正式完整生命周期。因此“绑定PASS/故障拒绝PASS”与“正式metric/serializer/科学臂NOT_RUN”分别记载。实际日志时钟gate未通过，当前不得构建新的科学SESSION outcome。
