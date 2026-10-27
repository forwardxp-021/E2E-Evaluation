# D6-CUAA v1：具体任务标准化不配对AA/AB候选合同

**PROPOSED_ONLY / SCIENCE_EXECUTION_AUTHORIZED=false / CURRENT_PREFLIGHT=BLOCKED。**

这份合同给出可检查字段、固定抽样与拒绝逻辑，不激活D5或现行SAP修约，不提供科学runner入口。配套JSON由冻结计划和实际证据生成；源码`tools/rbr_v2_d6_contract.py`仅允许D角色和合成fixture，不允许V/C调用。

## 1. 观测与样本单位

一个SESSION只用原SDC/D2静态规则选择的原始LOG及首个适格anchor；同SESSION其它scenario/chunk不增加独立来源，不找替补。来源必须是原政策允许的官方train/val、身份完整、非C、非UNKNOWN；历史E3只可方法开发，不作新方法的独立确认。

初始时刻为冻结canonical episode t0，对应precontext里的initial_lidar_token/time_us；查询该lidar_pc关联的ego_pose及同token lidar_box/track/category。不得用scenario_tag的名称代替t0语义。初始XY若在metadata中舍入，先核对原记录的6位小数误差界，再用原始DB坐标；不改原anchor。

## 2. 固定任务C与q*

外生/重置条件：map=us-nv-las-vegas-strip；原SDC几何、完整性、offline原始交通适格；rear axle在恰一个LANE、无LANE_CONNECTOR、该lane恰一个outgoing edge；初速[5,15)m/s。道路拓扑、曲率和坐标是静态/前置任务描述，部署初速可能受版本过去行为影响，结论仅解释此重置离线实验。

初速两箱[5,10)、[10,15)；t0动态类别vehicle/pedestrian/bicycle任一中心距rear axle≤50m即busy，否则empty。位置/类别缺失为NOT_ASSESSABLE，不能按empty填补。50m仅为描述性任务半径。干预后mask、表示、报警、安全结果不作任务条件；天气/照明未知不推断。

四单元`5to10_empty / 5to10_busy / 10to15_empty / 10to15_busy`各1/4。在原151严格未来候选中按条件过滤，每个单元内SESSION均匀，作为当前共同有限来源总体规则；角色划分必须在新结果前独立冻结。**任一单元0时q*无法在本框完整实例化，禁止去掉单元或把其概率分给busy。** 当前future29、单元0/7/0/22，返回NOT_ASSESSABLE。

C仍未控制未来路口、汇入度、密度、相对速度、采集时段及不可观测环境；同城市/速度不保证条件行为分布相等。若改任何字段/分布，必须新候选版本，不沿用旧结论或偷偷复用阈值。

## 3. 真实null与数据角色

* D_fit：只用于版本化数值预处理，独立于cal/AA/AB；目前原3fit不覆盖q且稳定性未认证，不自动重拟合。
* 开发D：原D3/D4与D6方法暴露继续E3；现有同版本q Baseline仅6源，未达到最低draw要求。
* D_AA_final_cal：只用同一固定SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1 Baseline和同80帧/79推进官方协议。
* 独立AA：相同版本、相同q*与测量，同来源框事前随机角色分配；与cal及D_fit、AB SESSION互斥。需要不同SESSION，不需要反事实双臂。
* AB：同q*，原SDC Baseline reference，对target按固定随机化选择10 Treatment；科学资格仍需原机制、匹配、安全与完整分母。
* V/C：不执行、不转D；原10个待裁定E3不能在本合同自动进入独立final-cal。C25身份保护，q适用性UNKNOWN。

角色支持最小值：每池每单元≥10个不同SESSION。配额抽样每单元无放回取10个，前5 ref、后5 target，四单元共20/侧。role内、role间与原始来源别名必须核对；不共享任一SESSION。AA treatment_mask全False；AB恰10 True。两表示共享所有选择/权重/随机数。

分层配额AA的采样分布不同于无条件iid AA；外部cal与评价必须复制同一配额设计。跨来源同版本若未能辩护相同观测分布，只作transport stress，不自动计FPR。人工pair-swap不是本合同AA。

## 4. 数值和报警

RBR唯一epoch30、Canonical83及Waymo normalizer固定；H30原始30列固定；原二次MMD²与RBF定义不变，保留负值。固定配额下称“无偏两样本形式”，不虚称对任意总体量无偏。

数值认证要求：输入身份/单位/mask可比，算子oracle通过，scaler及kernel稳定性有独立证据。D6已证明微小Baseline时间分箱方差主要可由真实滤波尾巴解释，不能当作垃圾直接截掉。未认证全30列物理分辨率时不得定新的floor；原固定算法计算可重现不等于新的q下已认证可信管线。当前`NUMERICAL_PIPELINE_CERTIFIED=false`，必须拒绝正式运行。

未来如获批准且所有前提满足，AA外部cal1999个draw，seed2026091101，1900阶阈值、严格大于报警；独立AA/AB按原10000及MC误差规则。m20、π=.5、α=.05、δ*=.10均不改；没有评价池调阈值、核/feature/seed搜索、paired fallback或失败后补源。

## 5. 推断和接受条件

SESSION/更高可靠依赖组是cluster，所有臂/表示/缺失与失败同行。每角色每单元独立重采权重，释放时仍要求每单元10个不同原始ID；bootstrap副本不是新来源。不足支持的replicate不能静默丢弃，源域CI/coverage标NOT_ESTABLISHED。

原95%来源CI、同时FPR上界≤.075、点Δ≥.10且下界>0等科学门保持，不能以软件通过替代。开发必须在真实AA/AB支持与独立源域合成验证下检验δ*=.10、设计效应.20、coverage及power；目前这些门全部未建立，不计算Primary结论。

当前预检失败原因是结构性的：缺失q单元、来源数量、数值认证及推断认证，而不是runner暂未启动。No-Go时输出全名单和缺失原因，0arm后停止。

## 6. 运行与预算

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_static.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_numeric.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/rbr_v2_d6_finalize.py audit
```

这些命令只进行当前静态/解析诊断，不能启动正式AA、AB或车辆传播。`validate_pool`在源不足、未知单元、重复SESSION或V/C角色时抛错；`draw`只在合法D/fixture来源上生成身份，不运行模型。现有真实池不能通过。

本轮下载/新增来源/车辆执行/训练/Primary/V/C预算均0。将来若讨论裸pilot，至少三个互斥40源池且每池每单元10，共120个适格源；D_fit另计，独立验证不能用已暴露D替代。它不是覆盖或power认证，更不是自动批准240arm。原320规划在本q下严格缺291，空单元命中率为0，完整来源成本不能可靠外推。本合同不能以改变q来恢复执行。
