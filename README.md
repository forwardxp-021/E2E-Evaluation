# E2E-Evaluation

基于 Waymo 轨迹数据学习驾驶风格 embedding 的实验工程。

Stage6/Stage7、A/B/C checkpoint、一次性盲测和最终BDD报告体系均已冻结。Stage7L-A2已完成独立
external pure-lateral planner、pre-treatment map opportunity inventory、单元测试与development-only
official smoke；最终148个fresh token / 120个log，五档smoke 5/5成功，canonical route progress逐点一致，
且未读取embedding/BDD。当前冻结状态为`STAGE7L_PURE_LATERAL_IMPLEMENTATION_CLEAN`和
`STAGE7L_B_DEVELOPMENT_AUTHORIZED`，但尚未启动Stage7L-B。详细证据见
[`docs/stage7l_a2_pure_lateral_clean_implementation_zh.md`](docs/stage7l_a2_pure_lateral_clean_implementation_zh.md)；
Stage7L-A技术审计仍作为实施前历史记录保留。

## 项目目标

## 中文术语与指标解释清单

本节是本项目报告、图表和实验输出的统一中文词典。统计结论应以对应阶段的冻结配置和中文
报告为准；术语相同不代表不同数据集、不同checkpoint或不同实验设计下的数值可以直接比较。

### 一、如何阅读 BDD 结果表

Stage 6J/6K 报告中的典型表头如下：

| 表头 | 中文解释 | 正确读法 | 常见误解 |
|---|---|---|---|
| 名义剂量 | planner纵向参数从保守端点到激进端点的插值比例 | 25%表示六个IDM纵向参数走完保守→激进差值的25% | 不是“驾驶风格改变25%”，也不是现实车辆风格的统一单位 |
| BDD/MMD² | 两组embedding分布差异的观测统计量；当前Stage 6主要用带RBF核的biased MMD²实现 | 数值越大通常表示在当前数据和kernel下分布差异越大，但必须结合本次null分布解释 | 不能规定“达到0.5才有效”，也不能跨数据集直接比较裸值 |
| paired-null q95 | 在同场景pair内随机交换A/B标签后得到的零分布95%分位数 | observed BDD高于q95，表示超过了本次未校正的5%随机化参考线 | 不是通用阈值；也不能替代多重检验后的Holm p |
| BDD/q95 | `observed BDD ÷ paired-null q95` | 大于1表示观测值越过本档未校正q95；例如1.29表示是q95的1.29倍 | 不是“风格差异129%”，也不是模型准确率 |
| Z_BDD | `(observed BDD − null均值) ÷ null标准差` | 表示观测BDD距离本档随机化null均值多少个null标准差 | 这是null标准化诊断，不是默认服从正态分布的传统z检验，不能机械套用1.96 |
| raw p | 当前单项随机化检验未经多重校正的p值 | 越小表示null随机交换产生同样或更极端BDD的比例越低 | 单看raw p选剂量或选task会增加假阳性 |
| 四档Holm p | 对25/50/75/100%四个overall检验做Holm family-wise校正后的p值 | Stage 6K overall结论用该列；小于0.05才通过冻结统计门槛 | 不是把raw p简单乘4；表中的“4档”也不包括描述性0%原点 |

以Stage 6K的25%档为例：BDD=`0.001156`虽然绝对值不大，但paired-null q95=`0.000896`，
BDD/q95=`1.290`、Z_BDD=`3.649`、四档Holm p=`0.004290`，且实现运动学门禁通过，
因此在本冻结协议内判定为“检出”。这个结论不能转换成“0.001156是通用BDD阈值”。

### 二、模型、表征与数据术语

| 术语 | 中文解释与本项目中的含义 |
|---|---|
| behavior embedding | 行为嵌入/行为表征。编码器把一段轨迹及其交互上下文压缩成固定维度向量；向量不是人工可直接解释的单一风格分数。 |
| encoder | 编码器。把轨迹、邻车和派生特征映射为embedding的神经网络。 |
| checkpoint | 一次训练得到并保存的模型参数快照。本项目比较实验必须记录checkpoint路径和SHA-256；换checkpoint后BDD标尺可能改变。 |
| Waymo-only trained | 模型只用Waymo训练；nuPlan在当前主实验中是外部验证域，不参与该checkpoint训练。 |
| cross-domain | 跨域。训练域与验证域来自不同数据来源、传感/地图体系或生成机制，例如Waymo实车轨迹→nuPlan闭环仿真。 |
| domain shift | 域偏移。不同数据域在场景、轨迹尺度、交通参与者、地图或采样方式上的分布差异。 |
| trajectory-only baseline | 只使用自车轨迹、不使用邻车交互上下文的基线。 |
| interaction-aware | 交互感知。模型输入显式包含前车、相邻车等交通参与者信息。 |
| context | 上下文输入。本项目Stage5D通常是自车、5邻车slot、相对运动和lane关系组成的83维逐帧特征。 |
| 5-neighbor | 五邻车布局：front、left-front、left-rear、right-front、right-rear。空slot由mask/缺失规则处理。 |
| embedding dimension / 64D | embedding向量维度；64D表示每条rollout最终得到64个数，不表示64种已命名驾驶风格。 |
| context dimension / 83D | 输入上下文每个时间步的特征维度；83D与64D embedding不是一回事。 |
| feature | 人工计算或模型使用的输入/评价特征，例如速度、加速度、jerk、THW。 |
| style feature | 用来描述驾驶行为倾向的统计特征集合；它是风格代理量，不等于驾驶员身份或安全水平。 |
| learned representation | 由训练学习得到的表征，与人工定义的kinematic feature相区别。 |
| baseline | 比较基线。可以是旧模型、trajectory-only模型、显式运动学方法或保守planner；必须看具体上下文。 |
| ablation | 消融实验。固定其他条件，移除或替换某个输入/模块，以判断结果是否依赖它。 |
| seed | 随机种子。用于复现数据划分、bootstrap或permutation抽样；同seed不保证不同软件环境逐bit完全一致。 |
| split | 数据划分，例如train/validation/test，或release A/B集合。 |
| train / validation / test | 训练集用于拟合参数；验证集用于选模型/超参数；测试集用于最终评价。反复查看测试结果会使其退化为开发集。 |
| log-disjoint | 两个集合不共享采集log，降低同一次连续采集带来的依赖和泄漏。 |
| scenario-disjoint | 两个集合不共享scenario token；比只按行随机拆分更严格。 |

### 三、BDD、MMD与kernel术语

| 术语 | 中文解释与使用边界 |
|---|---|
| BDD | Behavior Distribution Difference，行为分布差异。本项目对“两个版本/策略的behavior embedding分布差多少”的总称；具体实现必须看报告。 |
| MMD | Maximum Mean Discrepancy，最大均值差异。使用kernel比较两组分布；理论上分布越不同，MMD通常越大。 |
| MMD² | MMD的平方。本项目Stage 6表格中的BDD通常实际报告MMD²，因此0.005是平方统计量，不是概率或百分比。 |
| MDD | 不是本项目当前冻结协议中的独立正式指标。历史文字若出现MDD，多数是BDD/MMD的口头混写或笔误；正式引用时应回到输出确认究竟是MMD²还是其他量。 |
| observed statistic | 用真实A/B标签计算出的观测统计量，与随机交换标签产生的null统计量相对。 |
| kernel | 核函数。把embedding间距离转换成相似度，决定MMD比较分布的尺度。 |
| RBF kernel | 径向基核/高斯核，形式为`exp(-distance²/(2×bandwidth²))`。距离越近，相似度越接近1。 |
| bandwidth | RBF核带宽。控制“多远算相似”；带宽变化会改变MMD²绝对量级，因此跨bandwidth裸BDD不可直接比较。 |
| pooled median bandwidth | 把A/B两组样本合并，取所有正的非对角欧氏距离中位数作为bandwidth；每个冻结比较独立确定后，在全部随机化中保持不变。 |
| biased MMD² / V-statistic | 包含kernel对角项的MMD²估计量。这里“biased”是统计估计形式，不表示实验有主观偏见。 |
| null distribution | 零分布。在“没有可区分的A/B标签效应”规则下反复随机化得到的BDD分布。 |
| q95 / q99 | null分布的95%/99%分位数；q95约有5%的null统计量位于其上方。 |
| paired-null | 保留scenario配对结构，只在每个pair内部交换A/B标签得到的null。 |
| unpaired-null | 不利用同场景pair，通常在两组间做整体标签重排；回答的是边际分布是否不同。 |
| permutation / randomization test | 置换/随机化检验。按照预先定义的可交换规则重排标签，并比较null统计量是否达到observed。 |
| within-pair label swap | 每个同场景pair独立决定是否交换A/B，是paired BDD的primary随机化方式。 |
| log-cluster label flip | 同一log内所有pair一起交换A/B，承认同一log内场景相关；Stage 6K中是补充敏感性，不替代primary。 |
| exceedance count | 随机化样本中，BDD达到或超过observed的次数。 |
| plus-one p | Monte Carlo p=`(exceedance+1)/(repetitions+1)`，避免有限随机化中报告p=0。0/100000对应约`9.9999e-6`。 |
| Monte Carlo resolution | 有限随机化能够报告的最小plus-one p；100000次时约为`1/100001`。 |
| Euclidean / L2 distance | 欧氏距离。可衡量两个embedding向量相距多远，但不是BDD本身。 |
| standardized BDD | 为降低采集构成差异而在共同支持域、固定参考分布或null尺度下标准化的BDD；必须查看具体标准化定义。 |
| raw BDD | 未相对null、共同支持域或其他参考量标准化的原始MMD²。 |
| universal BDD threshold | “通用BDD阈值”。当前证据明确不支持它；阈值必须针对checkpoint、数据域、样本量和运行设计重新标定。 |

### 四、统计检验与不确定性术语

| 术语 | 中文解释与使用边界 |
|---|---|
| p-value / p值 | 在冻结null和随机化规则成立时，得到当前或更极端统计量的概率量度；不是“原假设为真的概率”。 |
| raw p | 单个检验未经多重比较校正的p值。 |
| adjusted p | 经过Holm等多重比较方法调整后的p值；所属family必须同时说明。 |
| Holm correction | Holm逐步校正，用于控制一组检验的family-wise error rate；先按raw p排序，再逐步调整，不等同于每项机械乘以固定数。 |
| hypothesis family | 同时解释的一组假设。例如Stage 6K有4个overall剂量family和12个task×dose family。 |
| alpha / α | 预先设定的显著性水平，常用0.05；不是模型准确率。 |
| reject / not reject | reject表示在冻结检验下拒绝null；not reject表示证据不足，不等于已经证明两组完全相同。 |
| confidence interval / CI | 置信区间。表示按指定重采样/模型重复实验时估计量的不确定性范围；不是“真值有95%概率落在本次区间”。 |
| one-sided 95% lower bound | 单侧95%下界，只关心效应是否严格大于0；Stage 6K运动学门禁用于预期正方向的速度和RMS加速度。 |
| two-sided 95% CI | 双侧95%区间，同时描述正负两个方向的不确定性。 |
| bootstrap | 从观测单位中有放回重采样，估计统计量的不确定性。它不随机交换A/B标签，因此与permutation检验用途不同。 |
| cluster bootstrap | 以log等cluster为单位重采样，保留cluster内部相关性。 |
| Spearman rho / ρ | 基于秩的单调相关系数，范围[-1,1]；rho=1表示排序完全同向，不代表线性比例为1。 |
| task-adjusted rank residual | 先按task移除秩的组均值，再看质量指标与embedding距离的剩余相关；是描述性诊断，不是因果调整。 |
| effect size | 效应量，描述差异有多大。BDD、速度差、检测率差都可作为不同效应量，不能相互当作同一单位。 |
| statistical power | 真实存在指定效应时检验成功检出的概率；受效应大小、样本量、噪声和阈值影响。 |
| detection rate | A/B已知存在差异时，重复release trial超过阈值的比例；等于该实验定义下的经验检出率。 |
| false-negative rate / FNR | 假阴性率=`1−detection rate`。66.5%检出率对应33.5%假阴性率。 |
| false-positive rate / FPR | A/A本来没有版本差异时，仍超过阈值的比例。 |
| Wilson interval | 二项比例的Wilson置信区间；比简单正态近似更适合有限trial的检测率/FPR。 |
| calibration | 用独立A/A历史试验确定阈值或null参考，不在看到目标A/B结果后调阈值。 |
| sensitivity analysis | 敏感性分析。更换合理的依赖/质量处理规则，检查主结论是否稳健；不应事后替换预冻结primary。 |

### 五、实验设计与因果解释术语

| 术语 | 中文解释与本项目中的角色 |
|---|---|
| A/A | 同一软件/同一planner的两个独立样本集合比较，用于估计自然波动、FPR和阈值；不是期待BDD严格等于0。 |
| A/B | 两个不同软件版本或planner profile比较，用于评价是否能检出已知风格差异。 |
| paired A/B | A和B在相同scenario上各运行一次，能控制场景异质性，回答“同场景换版本是否改变行为”。 |
| unpaired A/B | 两版本来自不同log/路线/场景，贴近实际路试发布，但更容易被采集构成差异淹没或混杂。 |
| same-scenario pair | scenario token相同，只有planner/version不同的一对rollout。 |
| marginal BDD | 忽略pair，只比较两组总体边际embedding分布。场景差异较大时可能掩盖同场景版本效应。 |
| scenario-conditioned / paired BDD | 保留同场景对应关系，通过pair内label swap检验版本效应。 |
| overall | 合并冻结范围内全部符合条件的pair得到的总体分析。 |
| task-conditioned | 按预先定义的scenario type/task切片后分析，例如following或stop/go。 |
| primary | 预先指定、承载主要论文结论的分析。 |
| secondary | 预先指定的次要分析，通常需要自己的多重性控制；不能在primary失败后挑一个显著secondary冒充主结论。 |
| supplementary | 补充稳健性证据，例如log-cluster label flip；不替代primary estimand。 |
| exploratory | 探索性分析，用于发现问题或提出后续假设，不能包装成预先确认性结论。 |
| pre-treatment variable | 处置/版本运行前已经确定的变量，例如scenario type、log、路线；适合用于分层或匹配。 |
| post-treatment variable | 处置后才产生或可能受处置影响的变量，例如实现后的fallback、急刹、embedding距离；不得随意用于确认性删样本。 |
| treatment / intervention | 实验中有意改变的因素，例如planner的六个纵向IDM参数。 |
| confounding | 组间同时存在、会影响结果的其他差异；unpaired真实路试中路线、天气、交通密度等尤其重要。 |
| estimand | 统计分析真正要回答的目标量。paired BDD与unpaired release BDD的estimand不同。 |
| common support | A/B在预处理协变量上都有样本覆盖的可比区域；没有共同支持时不应强行比较。 |
| matching covariate | 用于匹配A/B采集构成的处置前变量，不能用版本运行后的结果变量。 |
| ESS | Effective Sample Size，有效样本量。重加权后权重越集中，ESS越小，名义样本数会高估信息量。 |
| release emulation | 用公开数据模拟两个软件版本在不同log/场景采集后进行比较；不是实际OEM发布验证。 |
| independent confirmation | 使用未参与方法开发、阈值选择或结果查看的新数据做确认。反复使用的数据不能继续称为独立确认集。 |
| leakage | 信息泄漏。相同token/log或测试结果进入训练、选模、阈值制定，会使性能过于乐观。 |

### 六、nuPlan仿真与场景术语

| 术语 | 中文解释与本项目中的含义 |
|---|---|
| planner | 自动驾驶规划器。根据当前状态和环境生成未来控制/轨迹；本项目比较的是planner行为，不直接比较安全优劣。 |
| PDM-Closed | nuPlan/tuPlan Garage的闭环PDM planner配置；自车后续状态会影响下一步规划。 |
| assertive | 本项目设置的相对激进纵向profile，通常目标速度更高、headway/min-gap更小、加速能力更强。 |
| conservative | 相对保守纵向profile，通常速度更低、headway/min-gap更大、加速度更温和。 |
| IDM | Intelligent Driver Model，智能驾驶员模型。PDM纵向策略使用的一组跟驰/速度参数；本项目只把它作为可控planner处置。 |
| nominal dose | 名义剂量。六个纵向IDM参数的插值比例，例如25/50/75/100%。 |
| realized dose | 实现剂量。rollout实际产生的速度、加速度、jerk、THW、gap等差异；不假定与名义剂量线性一致。 |
| kinematic gate | 运动学门禁。先确认planner参数确实在rollout中形成预期方向的行为差异，再允许解释embedding/BDD。 |
| rollout | planner在一个scenario中的完整仿真运行及其输出轨迹。 |
| official rollout | 由nuPlan官方仿真链路生成并通过token、planner、时间长度等审计的rollout。 |
| pseudo rollout | 非官方闭环输出或人为替代轨迹，只能用于接口/方法开发，不能冒充official simulation证据。 |
| open-loop | 只评价给定历史输入下的预测，不把planner输出反馈为下一时刻自车状态。 |
| closed-loop | planner输出会影响后续状态和观测，能反映控制行为累积，但仍是仿真。 |
| scenario | 一个有固定起始时间、地图、交通参与者和场景类型的仿真片段。 |
| scenario token | scenario的唯一标识；strict-token alignment用于确保A/B确实运行同一场景。 |
| log | 一段连续采集记录，通常包含多个scenario；同一log内场景可能相关。 |
| task | 按驾驶行为/场景类型定义的分析类别，不是后台计算任务。 |
| following | 跟车交互类场景，例如有前车、慢前车或邻近长车。 |
| stop/go | 红绿灯起停、拥堵静止、跟车停车等纵向控制场景。 |
| longitudinal high-motion | 高/中速度等纵向高运动工况；Stage 6J/6K排除了高横向加速度类型。 |
| lane-change | 数据标签为换道相关的场景slice；标签不自动证明本次planner rollout中的ego实际完成了换道。 |
| dense/vulnerable | 多车辆、行人/弱势交通参与者等密集交互场景集合。 |
| ODD | Operational Design Domain，设计运行域，包括道路、天气、速度、地区等适用条件。当前公开数据结论不能自动覆盖OEM全部ODD。 |

### 七、轨迹、交互与质量指标

| 术语 | 中文解释与单位 |
|---|---|
| speed | 速度，通常单位m/s。 |
| acceleration / accel | 加速度，通常单位m/s²；正负表示加速/减速方向。 |
| jerk | 加加速度，即加速度对时间的变化率，单位m/s³；绝对值/RMS常用于描述平顺性。 |
| yaw rate | 横摆角速度，单位rad/s，描述车辆航向变化快慢。 |
| RMS | Root Mean Square，均方根。先平方、求均值、再开方；对较大波动更敏感。 |
| p95 / p99 | 95%/99%分位数，用于描述尾部较大的加速度、jerk等，不等于最大值。 |
| median / p50 | 中位数，50%样本在其两侧，对极端值比均值稳健。 |
| IQR | 四分位距=`p75−p25`，描述中间50%数据的离散程度。 |
| THW | Time Headway，车头时距，常近似为前车距离/自车速度，单位秒；无有效前车时可能缺失。 |
| gap / front distance | 与前车的空间距离，单位米；必须同时关注前车有效帧比例。 |
| relative speed / v_rel | 自车与前车速度差；正负号定义必须查看具体特征实现。 |
| exposure rate | 某对象/条件在有效帧中出现的比例，例如有效前车覆盖率。 |
| lane-aware assignment | 结合地图lane、航向和拓扑关系，把交通参与者分配到五邻车slot。 |
| geometric fallback | lane投影不可用或不可靠时，退回基于自车局部几何位置分配邻车。fallback不是丢帧，但可能改变context语义。 |
| fallback rate | 有效帧中使用geometric fallback的比例；Stage 6K发现它与embedding pair距离正相关，因此必须作为测量限制。 |
| ambiguity rate | lane/context关系无法唯一确定或被标为模糊的帧比例。 |
| map/lane projection | 把车辆位置投影到地图lane中心线，并检查横向距离、航向和拓扑。 |
| valid frame / valid horizon | 轨迹和特征有效的时间步及总长度；A/B pair通常要求时间范围对齐。 |
| mask | 指示哪些时间步或slot有效的布尔数组，防止把填充值当作真实交通参与者。 |
| Tier A / Tier B | 按lane/context质量规则划分的样本等级；具体阈值看对应冻结配置。它们不是安全等级。 |

### 八、工程运行与证据冻结术语

| 术语 | 中文解释与操作含义 |
|---|---|
| freeze / 冻结 | 在读取目标结果前固定场景、模型、统计量、随机化次数、seed和判定规则，避免看结果后改口径。 |
| locked | 已锁定且带完整性校验的输入/清单；修改后SHA-256应变化并触发审计失败。 |
| manifest | 机器可读的清单文件，记录输入路径、数量、hash、配置、状态和输出。 |
| SHA-256 | 文件内容指纹。hash一致表示字节内容一致；它验证完整性，不证明科学结论正确。 |
| provenance | 可追溯信息，包括代码commit、工作树状态、输入hash、依赖版本、命令和随机seed。 |
| smoke test | 小规模真实链路冒烟测试，用于确认命令和接口能运行；不能替代全量统计结果。 |
| dry-run | 只检查配置、清单和将要执行的任务，不真正启动大规模仿真或写入核心结果。 |
| execute | 显式允许真实运行的开关；本项目的大规模任务通常默认dry-run以防误启动。 |
| resume | 断点续跑。重新审计已成功任务并跳过，只继续未完成任务。 |
| attempt | 某个任务的一次独立执行目录；重试应创建新attempt，保留旧日志用于追溯。 |
| SUCCEEDED / FAILED / PENDING | 成功/失败/待运行状态；只有全部目标任务成功且审计通过才称为完成。 |
| PASS / FAIL | 某项预先定义验证规则是否通过；PASS不自动等于论文所有主张成立。 |
| overwrite | 覆盖已有输出。冻结和权威结果默认禁止随意overwrite，除非明确验证目标和保留旧证据。 |
| issue-first | 先在GitHub Issue记录研究问题、冻结规则和结果，再实现/运行，形成可审计决策链。 |

如报告出现本清单没有收录的缩写，应在对应中文报告首次出现处写出英文全称、中文含义、
单位、方向和判定规则，并同步补充到本节，不能只依赖代码变量名推测。

## 当前状态（Stage 4）

- Stage 4G（comfort metric alignment）是当前最佳结果（current best）。
- Stage 4H（shuffled comfort target）sanity check 已通过。
- Stage 4I 负责最终结果固化与论文图表包生成，不引入新训练方法。
- Stage 5（interaction-aware design）为规划中的扩展方向：在 Stage 4G 轨迹基线上引入 lane-aware 5-neighbor 上下文。
- Stage 4G 仍是当前最佳 trajectory-only baseline，Stage 5 不替代 Stage 4G。
- Stage 5 详细设计见 `07_stage5_interaction_design.md`。
- 具体命令请见 `QUICK_REFERENCE.md`。


从自车与前车的对齐轨迹中构建 style feature，用 feature-guided soft contrastive 训练轨迹编码器，最后通过 UMAP、线性探针和邻域一致性验证 embedding 是否编码了行为风格信息。

## 近期代码更改总结 (2026-04)

### 1) 数据构建重构为滑窗 + 新 style 特征
- `build_dataset.py` 新增滑窗参数：`--window_len`、`--stride`。
- 每个 scenario 按窗口输出样本，不再只输出整段场景级样本。
- 新增 20 维 style 特征计算函数 `compute_style_features(...)`，并写出：
  - `output/feat_style_raw.npy`：原始特征（可能含 NaN）
  - `output/feat_style.npy`：NaN 填 0 后全局标准化特征（训练默认建议使用）
  - `output/feature_names_style.json`：特征名映射
- 保留兼容开关 `--save_legacy_features`，需要时可额外输出旧特征到 `feat_legacy.npy` 和兼容文件 `feat.npy`。

### 2) TFRecord 解析健壮性增强
- `build_dataset.py` 增加 `parse_scenario_from_record(...)`：
  - 支持“直接 Scenario proto”
  - 支持“tf.train.Example 中 bytes 特征包裹 Scenario”
- 增加 `_scenario_looks_valid(...)` 做解析后有效性检查，降低脏样本导致的崩溃风险。

### 3) 训练数据读取兼容修复（已解决真实报错）
- `dataset.py` 中 `TrajFeatureDataset` 对 `traj.npy` 进行统一 `float32` 转换。
- 修复问题：`numpy.object_` 轨迹样本在 `collate_variable_traj` 中无法转 tensor，报错
  - `TypeError: can't convert np.ndarray of type numpy.object_`
- 修复后已通过训练冒烟验证。

### 4) 导出脚本修复与增强（已解决真实报错）
- `export_embeddings.py` 删除了误粘贴的重复代码块，修复
  - `SyntaxError: 'return' outside function`
- `TrajOnlyDataset` 同步增加轨迹 `float32` 转换，避免导出阶段再次触发 object dtype 问题。
- 保留 `--checkpoint_path` 作为 `--checkpoint` 的兼容别名，并支持 `--split_path` 行数一致性校验。

### 5) 评估流程对齐新特征
- 推荐在评估中显式指定：
  - `--feat_path output/feat_style.npy`
  - `--feature_names_path output/feature_names_style.json`
- 已完成一轮 `train -> export -> evaluate` 全链路跑通。

## 新 style 特征 (20D)

顺序与 `feature_names_style.json` 一致：

1. `acc_abs_p95`
2. `acc_abs_p99`
3. `acc_rms`
4. `jerk_abs_p95`
5. `jerk_abs_p99`
6. `jerk_rms`
7. `yaw_rate_rms`
8. `yaw_rate_abs_p95`
9. `heading_change_total`
10. `speed_control_oscillation`
11. `cf_valid_frac`
12. `thw_p50`
13. `thw_p20`
14. `thw_iqr`
15. `v_rel_p50`
16. `closing_gain_kv`
17. `gap_gain_kd`
18. `desired_gap_d0`
19. `acc_sync_lag`
20. `acc_sync_corr`

## 快速使用

### 1) 构建数据

```bash
python build_dataset.py \
  --tfrecord_glob "/mnt/d/WMdata/*.tfrecord-*" \
  --output_dir output \
  --min_ego_speed 5.5 \
  --window_len 80 \
  --stride 20 \
  --min_points_cf 20 \
  --kd_min 1e-3 \
  --d0_min_gap 1.0 \
  --d0_max_gap 200.0 \
  --d0_log1p \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### desired_gap_d0 清洗说明（病态长尾防护）

- `desired_gap_d0` 来自拟合关系 `a_e ≈ kv*v_rel + kd*d + b` 的零加速度间距 `d0=-b/kd`。当 `kd` 过小或拟合病态时，`d0` 会数值爆炸并污染 feature 距离，导致 soft target 过平。
- 数据构建中已增加两层防护：
  - 拟合阶段：`abs(kd) < kd_min` 时不计算 `d0`（置为 NaN）。
  - 特征阶段：对窗口级 `d0` 做 sanitize（`<=d0_min_gap` 置 NaN、clip 到 `[d0_min_gap,d0_max_gap]`、可选 `log1p` 压缩长尾）。
- 相关参数：
  - `--kd_min`（默认 `1e-3`）
  - `--d0_min_gap`（默认 `1.0`）
  - `--d0_max_gap`（默认 `200.0`）
  - `--d0_log1p / --no-d0_log1p`（默认开启）

## 为什么需要工况门控（条件感知训练）

### 问题
直接在全体样本间做特征距离对比存在一个根本缺陷：**不同工况（速度/跟车距离/相对速度/跟车覆盖率）下的驾驶行为是不可比的**。例如：
- 高速行驶时的急刹与低速跟车时的轻刹，加速度指标天然不同，但不能据此判断它们风格相似或不同。
- 非跟车段（cf_valid_frac ≈ 0）没有 thw、kv、kd 等跟车特征，与跟车段的特征距离毫无意义（cf 维度都填 0，错误地认为"完全一样"）。

这导致评估中 `neighbor_consistency ratio_mean > 1`（embedding 邻居的特征差异反而比随机邻居更大），以及线性探针 Spearman 偏弱。

### 解决方案：工况门控（kNN 模式）
- **工况向量**：从 `traj.npy` 和 `front.npy` 计算每个样本的工况特征 `[speed_mean, dist_mean, vrel_mean, cf_valid_frac]`
- **kNN 门控**（`--cond_mode knn`，推荐）：对每个 anchor 选取工况距离最近的 `cond_k` 个样本（距离用鲁棒尺度 MAD/IQR/STD 归一化，无需手动调容差）。仅在候选数为 0 时触发最后兜底（回退全局）。
- **硬盒门控**（`--cond_mode hard_box`，保留向后兼容）：基于绝对容差盒子过滤，候选数不足时退回全局。

### 2) 训练（推荐：kNN 工况门控 + 混合 SupCon）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_cond_knn \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 0.2 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

> **说明**：`--feat_clip_value 3.0` 在特征归一化之后、距离计算之前，将标准化特征值裁剪到 [-3, 3]，可有效抑制 jerk/yaw 等长尾维度对距离计算的影响（推荐值 3.0；默认 0.0 表示不裁剪，与旧行为完全兼容）。

训练日志中新增诊断指标：
- `cond_cands`：每个 anchor 平均可用的工况兼容候选数（knn 模式下应接近 cond_k）
- `cond_fallback`：触发最后兜底（候选数=0）的 anchor 比例（knn 模式下应接近 0）
- `supcon`/`softkl`：hybrid 模式下两个分项的损失值

### 2b) 旧版硬盒门控训练（向后兼容）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_cond_hybrid \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode hard_box \
  --cond_speed_tol 2 --cond_dist_tol 5 --cond_vrel_tol 1 \
  --cond_cf_bucket_edges "0.2,0.6" --min_cond_candidates 8 \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 0.2 \
  --eval_every 10 --skip_val_clustering
```

### 2c) 不使用工况门控的基础训练（向后兼容）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_masked_ls_k1_a3 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --eval_every 10
```

### 3) 导出全量 embedding

```bash
python export_embeddings.py \
  --traj_path output/traj.npy \
  --split_path output/split.npy \
  --checkpoint output/run_cond_knn/best_model.pth \
  --output_path output/run_cond_knn/embeddings_all.npy
```

### 4) 评估（含工况感知邻域一致性）

```bash
python evaluate_embedding.py \
  --embeddings_path output/run_cond_knn/embeddings_all.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --eval_split test \
  --feature_names_path output/feature_names_style.json \
  --analysis_dir output/run_cond_knn/analysis_best \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --plot_first_k 20 --k_neighbors 10 \
  --umap_neighbors 30 --umap_min_dist 0.1 \
  --seed 42 --kmeans_clusters 8
```

工况感知评估额外生成 `neighbor_results_cond.csv`：
- `ratio_mean`：在工况兼容候选集内的随机基线比较（更公平）
- `mean_cond_candidates`：每个 anchor 在工况内的平均候选数
- `frac_fallback`：因候选数不足而退回全局随机基线的比例

## rel_kinematics 输入模式

### 动机

原始 `raw_xyv` 模式直接将 ego 轨迹的 `[x, y, vx, vy]` 送入 GRU，缺少显式的相对运动信息。对于 jerk、yaw_rate、thw 等风格特征，它们本质上是**差分**和**相对量**，如果在输入层就提供这些归纳偏置，模型更容易学习出可分离的风格维度。

`rel_kinematics` 模式从对齐的 ego/front 窗口计算 12 维逐帧特征后再送入 GRU，可改善 jerk/yaw/thw 等维度的邻域一致性。

### 12 维特征说明（dt = 0.1 s，Waymo 10 Hz）

| 索引 | 名称 | 公式 |
|------|------|------|
| 0 | `ego_v` | √(vx²+vy²) |
| 1 | `front_v` | √(front_vx²+front_vy²) |
| 2 | `v_rel` | ego_v − front_v |
| 3 | `dx` | front_x − ego_x |
| 4 | `dy` | front_y − ego_y |
| 5 | `dist` | √(dx²+dy²) |
| 6 | `closing_rate` | diff(dist) / dt（t=0 置 0） |
| 7 | `ego_a` | diff(ego_v) / dt（t=0 置 0） |
| 8 | `front_a` | diff(front_v) / dt（t=0 置 0） |
| 9 | `ego_heading` | atan2(vy, vx) |
| 10 | `ego_yaw_rate` | wrap(diff(ego_heading)) / dt（t=0 置 0） |
| 11 | `thw` | dist / max(ego_v, ε) |

填充帧被置零（用 `lengths` mask）。角度差通过 `wrap_angle` 归约到 `[-π, π]`。

### 示例训练命令（rel_kinematics）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

启动时会打印：
```
Input mode: rel_kinematics (12-dim) | dt=0.1s | front loaded: <N> windows
```

### 示例导出命令（rel_kinematics）

```bash
python export_embeddings.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --split_path output/split.npy \
  --checkpoint output/run_relkin_knn/best_model.pth \
  --output_path output/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

> **注意**：`--input_mode` 和 `--dt` 必须与训练时使用的值一致，否则模型结构不匹配会导致权重加载失败。

## 关键输出文件

- `output/traj.npy`: 自车滑窗轨迹
- `output/front.npy`: 前车滑窗轨迹
- `output/feat_style_raw.npy`: 原始 style 特征
- `output/feat_style.npy`: 训练用标准化 style 特征
- `output/feature_names_style.json`: 特征名
- `output/split.npy`: train/val/test
- `output/<run>/best_model.pth`: 最优模型
- `output/<run>/embeddings_all.npy`: 全量 embedding
- `output/<run>/analysis_*/neighbor_results.csv`: 全局邻域一致性
- `output/<run>/analysis_*/neighbor_results_cond.csv`: 工况内邻域一致性（条件感知评估）
- `output/<run>/analysis_*`: 评估结果图表与 CSV

---

## 合成策略 Rollout：在无真实 E2E 推理代码时验证 Embedding 区分能力

### 背景与目标

在没有多个 E2E 模型 rollout 数据时，可以利用已有的 Waymo log-replay 窗口生成若干条规则控制策略（conservative / aggressive / lateral_stable）的合成轨迹，用于验证 style embedding 是否能够区分不同策略的驾驶行为。

### 关键文件

| 脚本 | 功能 |
|------|------|
| `generate_policy_rollouts.py` | 从已有 traj/front 窗口为每个策略生成模拟自车轨迹 |
| `compute_style_features.py` | 对已有 traj/front npy 计算 20D style 特征（不依赖 TFRecord） |
| `evaluate_policy_separation.py` | 用 embedding 做策略分类 + Recall@K 检索评估 |
| `evaluate_policy_separation_aligned.py` | Source-aligned 策略区分评估（按 source_index 分组，控制场景分布） |

### 三种合成策略说明

| 策略 | THW 目标 | 最大加速度 | Jerk 限制 | 横向稳定性 |
|------|----------|------------|-----------|------------|
| `conservative` | 2.5 s | ±1.5/3.0 m/s² | 0.5 m/s²/step | 强（yaw_rate_clip=0.05 rad/step） |
| `aggressive` | 1.0 s | ±3.5/5.0 m/s² | 2.0 m/s²/step | 弱（yaw_rate_clip=0.20 rad/step） |
| `lateral_stable` | 1.8 s | ±2.0/3.5 m/s² | 0.8 m/s²/step | 极强（yaw_rate_clip=0.01 rad/step, heading_smooth_alpha=0.7） |

### 完整工作流（复制可用命令）

#### a) 生成合成策略 Rollout

```bash
python generate_policy_rollouts.py \
  --src_traj_path  output/traj.npy \
  --src_front_path output/front.npy \
  --src_split_path output/split.npy \
  --src_meta_path  output/meta.npy \
  --output_dir     output_policy_rollouts \
  --dt 0.1 \
  --policies "conservative,aggressive,lateral_stable" \
  --seed 42
```

输出文件：
- `output_policy_rollouts/traj.npy` — 模拟自车轨迹（N_policies × N_src 个窗口）
- `output_policy_rollouts/front.npy` — 原始前车轨迹（直接复制）
- `output_policy_rollouts/policy_id.npy` — 策略标签（int）
- `output_policy_rollouts/scenario_id.npy` — 场景 ID（来自 meta 或自动生成）
- `output_policy_rollouts/source_index.npy` — 回溯原始窗口索引
- `output_policy_rollouts/split.npy` — train/val/test 分割
- `output_policy_rollouts/policy_names.json` — 策略 id→名称映射

#### b) 计算 Style 特征

```bash
python compute_style_features.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --output_dir output_policy_rollouts
```

#### c) 训练 Embedding

```bash
python train_embedding.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --feat_path  output_policy_rollouts/feat_style.npy \
  --feat_raw_path output_policy_rollouts/feat_style_raw.npy \
  --split_path output_policy_rollouts/split.npy \
  --output_dir output_policy_rollouts/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

#### d) 导出全量 Embedding

```bash
python export_embeddings.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --split_path output_policy_rollouts/split.npy \
  --checkpoint output_policy_rollouts/run_relkin_knn/best_model.pth \
  --output_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

#### e) 评估策略区分能力

```bash
python evaluate_policy_separation.py \
  --embeddings_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path  output_policy_rollouts/policy_id.npy \
  --split_path      output_policy_rollouts/split.npy \
  --policy_names_path output_policy_rollouts/policy_names.json \
  --eval_split test \
  --k_neighbors 10 \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_policy \
  --seed 42
```

输出：
- `policy_separation_summary.json` — 分类准确率、Macro-F1、Recall@K（汇总）
- `policy_retrieval.csv` — 每个测试样本的 Recall@K

### 注意事项

- 合成轨迹基于简单纵向控制器 + yaw-rate 限幅，**不追踪车道**，仅保持与原始轨迹方向大致对齐。
- `front.npy` 中的前车轨迹保持不变（外生 log-replay，不响应 ego）。
- 下游 `train_embedding.py` / `export_embeddings.py` / `evaluate_embedding.py` 均无需修改，直接指向新 output_dir 即可。
- 若 `--src_split_path` 未提供，脚本会按 scenario_id MD5 哈希自动分配 train/val/test（与 `build_dataset.py` 一致）。

---

## Synthetic policy rollouts: source-aligned evaluation

### Background

The standard `evaluate_policy_separation.py` measures global classification and
retrieval performance but does not control for *scenario distribution*: if different
policies happen to be evaluated on systematically different scenarios, the metric may
reflect difficulty differences rather than policy style differences.

`evaluate_policy_separation_aligned.py` addresses this by grouping samples by their
`source_index` — each source window was rolled out under every policy exactly once, so
within a source group the only variable is the policy.  All four computations below are
therefore scenario-controlled.

### Computations

| Step | Description | Output |
|------|-------------|--------|
| (a) Coverage validation | Checks each `(source_index, policy_id)` pair appears exactly once; reports missing / duplicate counts | summary JSON |
| (b) Within-source pairwise distances | Euclidean + cosine distance between every policy pair within each source group | `policy_pairwise_dist.csv` |
| (c) Centroid classification accuracy | Nearest-centroid prediction using train-split centroids; evaluated per source group on eval split | summary JSON |
| (d) Within-source retrieval applicability + margin | Check whether within-source same-policy NN retrieval is well-defined; report mean/median within-source distance margin | summary JSON |

### Copy-pastable commands (using default paths)

#### Step 1 — Generate rollouts

```bash
python generate_policy_rollouts.py \
  --src_traj_path  output/traj.npy \
  --src_front_path output/front.npy \
  --src_split_path output/split.npy \
  --src_meta_path  output/meta.npy \
  --output_dir     output_policy_rollouts \
  --dt 0.1 \
  --policies "conservative,aggressive,lateral_stable" \
  --seed 42
```

#### Step 2 — Compute style features

```bash
python compute_style_features.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --output_dir output_policy_rollouts
```

#### Step 3 — Train embedding

```bash
python train_embedding.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --feat_path  output_policy_rollouts/feat_style.npy \
  --feat_raw_path output_policy_rollouts/feat_style_raw.npy \
  --split_path output_policy_rollouts/split.npy \
  --output_dir output_policy_rollouts/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

#### Step 4 — Export embeddings

```bash
python export_embeddings.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --split_path output_policy_rollouts/split.npy \
  --checkpoint output_policy_rollouts/run_relkin_knn/best_model.pth \
  --output_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

#### Step 5 — Run standard policy separation evaluation

```bash
python evaluate_policy_separation.py \
  --embeddings_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path  output_policy_rollouts/policy_id.npy \
  --split_path      output_policy_rollouts/split.npy \
  --policy_names_path output_policy_rollouts/policy_names.json \
  --eval_split test \
  --k_neighbors 10 \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_policy \
  --seed 42
```

#### Step 6 — Run source-aligned evaluation

```bash
python evaluate_policy_separation_aligned.py \
  --embeddings_path   output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path    output_policy_rollouts/policy_id.npy \
  --source_index_path output_policy_rollouts/source_index.npy \
  --split_path        output_policy_rollouts/split.npy \
  --eval_split test \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_aligned \
  --seed 42
```

### Outputs

| File | Description |
|------|-------------|
| `policy_separation_aligned_summary.json` | Coverage stats, centroid accuracy, pairwise distance stats (mean/median), within-source retrieval applicability and margin |
| `policy_pairwise_dist.csv` | Per-source-group pairwise (Euclidean + cosine) distances for each policy pair |

### Notes

- `source_index.npy` is generated automatically by `generate_policy_rollouts.py`
  alongside the other output files.
- Samples where a source group does not have all policies present (e.g. at split
  boundaries) are excluded from computations (b) and (d); the coverage report in step
  (a) will list any such gaps.
- Centroids for classification (step c) are always estimated from the **train** split,
  regardless of `--eval_split`.
- If each source has only one sample per policy (common aligned setup), within-source
  same-policy nearest-neighbour retrieval is undefined; summary JSON will report
  `retrieval_applicable=false` and set NN hit-rate/chance to `null` instead of a
  misleading numeric 0.0.

---

## Embedding interpretability demo: retrieval + trajectory replay

`tools/embedding_retrieval_demo.py` provides the most intuitive way to visually
verify that embeddings cluster/separate driving styles into different regions.  Given a
query window it:

1. Retrieves the Top-K most-similar windows in embedding space.
2. Overlays ego + front trajectories (aligned to the query's initial position and heading).
3. Plots time-series style signals: speed, acceleration, jerk, and a curvature proxy.

### Prerequisites

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

No additional dependencies beyond the standard project requirements.

### Minimal command examples

#### Global retrieval (query against all items in the selected split)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --topk 5 \
    --mode global \
    --split_filter test
```

#### Within-source retrieval (only other rows sharing the same source meta-key)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --mode within-source
```

#### Select query by scenario ID (instead of array index)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_scenario_id "my_scenario_id" \
    --query_start 10 \
    --topk 5 \
    --mode global
```

#### Exclude same-scenario neighbours (prevent trivial retrieval)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --topk 5 \
    --mode global \
    --exclude_same_scenario
```

#### Self-contained smoke test (no data files needed)

```bash
python tools/embedding_retrieval_demo.py --smoke_test
```

### Outputs

All files are written to `outputs/<run_id>/` (configurable with `--output_dir` and
`--run_id`).

| File | Description |
|------|-------------|
| `retrieval_table.csv` | Top-K results with index, meta fields, distance, and excluded flag |
| `traj_overlay.png` | Ego + front trajectories overlaid in aligned coordinates |
| `timeseries.png` | Speed / accel / jerk / curvature-proxy time series for query and Top-K |
| `summary.json` | Run parameters: mode, distance, topk, exclusions, data paths |

### Explanation of plots and what to look for

**`traj_overlay.png`** — Both ego and front trajectories are translated so the query
starts at the origin and rotated so the query's initial velocity vector points along +x.
This makes cross-scenario overlays comparable.  If the embedding is meaningful you
should see that Top-K retrieved trajectories follow a similar *shape* to the query (e.g.
similar following distance, similar lateral deviation).

**`timeseries.png`** — Shows four derived style signals sampled at `--dt` seconds per
step:
- **speed** — `sqrt(vx² + vy²)`
- **accel** — finite-difference of speed
- **jerk** — finite-difference of accel
- **curvature proxy** — `yaw_rate / max(speed, ε)`, where yaw_rate is estimated from
  heading differences.  This is an *approximation*; label it accordingly in any paper.

For a well-trained embedding the retrieved trajectories should show **similar profiles**
to the query across all four signals, especially in the features the embedding was
trained on (THW, jerk, lateral yaw-rate, etc.).

### Within-source limitation (no explicit policy_id)

The base dataset (`build_dataset.py`) stores meta as
`(scenario_id, start, window_len, front_id)` with **no** explicit `policy_id` field.
In `within-source` mode the script groups all rows sharing the same meta-key tuple and
plots all of them against the query.  When data was produced by `generate_policy_rollouts.py`
there will be exactly `n_policies` rows per meta-key (one per policy), and you can
inspect their relative ordering by distance to verify separability.  If you need
precise per-policy labels use the `policy_id.npy` output of `generate_policy_rollouts.py`
and the aligned evaluator (`evaluate_policy_separation_aligned.py`).

### Running the smoke / unit tests

```bash
python scripts/smoke_test_retrieval_demo.py
```

### PR2 interpretability demo (`tools/embedding_interpretability_demo.py`)

For PR2-style interpretability (same-source triplet + global retrieval cards), use:

```bash
python tools/embedding_interpretability_demo.py \
  --data_dir output_policy_rollouts \
  --out_dir outputs/embedding_demo/case_000 \
  --embedding feat_style \
  --split test \
  --mode both \
  --projection both \
  --case_selection best_human_readable \
  --topk 5 \
  --source_key_fields scenario_id,start,window_len,front_id \
  --auto_select_valid_source
```

If `front_id` changes across policy rollouts, relax grouping with:

```bash
--source_key_fields scenario_id,start,window_len
```

The demo requires multi-policy rollout rows (typically 3 rows per source key).  
Check `summary.json -> diagnostics` to verify:
- row counts before/after split,
- source-group size histograms,
- policy-id availability/source/counts,
- core array shapes (`embedding/meta/traj/front/split`).

When `policy_id` is unavailable, hit@k for same-policy retrieval is intentionally set to `null` and a warning explains that nearest-neighbour visualization is still possible but same-policy verification is not.

#### Embedding interpretability demo outputs (for paper/presentation)

- `summary.json`: includes `policy_mapping`, `case_selection`, within-source distances, retrieval hit@k, and diagnostics.
- `embedding_2d_projection.png` / `embedding_2d_projection.csv`: PCA projection (visualization only; lossy).
- `embedding_2d_projection_umap.png` / `.csv`: produced when `--projection umap|both` and `umap-learn` is available.
- `embedding_distance_matrix.png` / `.csv`: within-source embedding distances with per-cell numeric annotation and policy labels.
- `within_source_triplet.png`, `within_source_style_signals.png`, `within_source_style_fingerprint_kinematic.png`, `within_source_style_fingerprint_dynamics.png`, `within_source_style_fingerprint_normalized.png`, `within_source_style_fingerprint.csv`: same-source policy contrast and style statistics.
- `global_retrieval_cards.png`, `global_retrieval_style_signals.png`, `retrieval_table.csv`, `style_fingerprint.csv`: global nearest-neighbor interpretability and style fingerprints.
- `interpretability_report.md`: auto-generated textual summary from summary/CSV outputs.

Interpretation guidance:
- PCA/UMAP are visualization-only; benchmark conclusions should rely on aligned metrics and high-dimensional embedding distances.
- Lack of perfectly separated 2D clusters does not invalidate high-dimensional separation.
- Metadata (`policy_id`, `policy_name`, `source_index`) is required for policy-level same-source contrast and same-policy hit@k verification.

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

新增脚本：`tools/run_lateral_stable_ablation.py`，用于批量运行 `lateral_stable` 参数消融（生成 + population 评估 + 汇总 + 推荐 + 报告 + 图表）。

### Debug 命令
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable
```

### Full 命令
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### 仅预览配置（不执行）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_plan \
  --dry_run
```

输出包括：
- `ablation_summary.csv` / `ablation_summary.json`
- `ablation_recommendation.json`
- `ablation_report.md`
- 汇总图：`ablation_*.png`
- 每个 config 独立目录下的 `rollouts/` 与 `population_eval/`

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

**Motivation**: Experiment 1 showed p2/lateral_stable is recognizable but remains too close to conservative, so p2 is not consistently an independent third style.

**Script**: `tools/run_lateral_stable_ablation.py`

**Required inputs**:
- `--source_data_dir` containing generator-compatible `traj.npy` and `front.npy` (optional but recommended `split.npy`, `meta.npy`).
- Existing tools: `generate_policy_rollouts.py`, `tools/evaluate_policy_population.py`.

**Debug command (max_sources=100)**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

**Dry-run command**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --dry_run
```

**Full command**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

**Output directory structure**:
- `base_output_dir/<config_name>/rollouts/`
- `base_output_dir/<config_name>/population_eval/`
- `base_output_dir/ablation_summary.csv`
- `base_output_dir/ablation_summary.json`
- `base_output_dir/ablation_recommendation.json`
- `base_output_dir/ablation_report.md`
- `base_output_dir/ablation_p2_separation_margin.png`
- `base_output_dir/ablation_p2_farthest_rate.png`
- `base_output_dir/ablation_pairwise_distances.png`
- `base_output_dir/ablation_retrieval_classification.png`
- `base_output_dir/ablation_p2_style_metrics.png`
- `base_output_dir/ablation_tradeoff_plot.png`

**How to interpret core metrics**:
- `p2_farthest_rate`: higher is better.
- `mean_p2_separation_margin > 0`: p2 is farther from both p0/p1 than p0-p1 are from each other.
- Lower `p2_rms_yaw_rate_proxy_mean` indicates stronger lateral stability.
- Lower `p2_rms_jerk_mean` indicates smoother longitudinal comfort.
- Retrieval + centroid metrics quantify policy discriminability.

**Known limitations**:
- Synthetic policies (no human labels).
- Replayed front vehicle (not full multi-agent closed loop).
- No sensor rendering/perception stack.


## Experiment 2 Ablation（必须产出 base_output_dir 聚合文件）

### 推荐命令（可直接复制）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### 期望输出结构
```text
outputs/ablation_debug/
  ablation_summary.csv
  ablation_summary.json
  ablation_recommendation.json
  ablation_report.md
  ablation_p2_separation_margin.png
  ablation_p2_farthest_rate.png
  ablation_pairwise_distances.png
  ablation_retrieval_classification.png
  ablation_p2_style_metrics.png
  ablation_tradeoff_plot.png

  baseline_current/
    rollouts/
    population_eval/
      population_summary.json

  no_lateral_smoothing/
    rollouts/
    population_eval/
      population_summary.json

  lateral_only/
    rollouts/
    population_eval/
      population_summary.json

  comfort_only/
    rollouts/
    population_eval/
      population_summary.json

  full_strong_lateral_stable/
    rollouts/
    population_eval/
      population_summary.json
```

> 完成标准：`ablation_summary.csv` 与 `ablation_report.md` 必须存在于 `base_output_dir` 根目录。

## Experiment 2B: Local Fine-Grained Sweep Around full_strong_lateral_stable

### 1) Motivation
Broad ablation selected `full_strong_lateral_stable` as best overall, but `mean_p2_separation_margin` remained negative. Experiment 2B performs a focused local sweep around that center to improve p2 independence while preserving comfort/stability.

### 2) Script usage
Use the existing script with `--config_set local_fine`:
- `tools/run_lateral_stable_ablation.py`
- optional `--config_file configs/lateral_stable_local_sweep.json`

### 3) Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### 4) Full command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_full \
  --config_set local_fine \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### 5) Output files
- `local_sweep_summary.csv` / `local_sweep_summary.json`
- `local_sweep_recommendation.json`
- `local_sweep_report.md`
- `local_sweep_integrity_report.json`
- `local_sweep_rollout_sanity.csv`
- plots: `local_sweep_*.png`

### 6) How to interpret results
Primary targets:
- increase `p2_farthest_rate`
- improve `mean_p2_separation_margin` toward 0 / positive
- keep `p2_rms_jerk_mean` below baseline_current
- keep `p2_rms_yaw_rate_proxy_mean` low
- preserve `centroid_accuracy_p2` and retrieval metrics.
If margin stays negative, report: **"p2 independence improved but remains incomplete."**

### 7) Broad ablation vs local sweep
- Broad ablation: mechanism-level comparison across distinct config families.
- Local sweep: fine-grained perturbation around `full_strong_lateral_stable` (yaw clip / heading smoothing / THW / jerk interactions).

### 8) Limitations
- Synthetic policy rollouts only.
- No public-data human validation yet.
- Replayed-front setup, not full closed-loop multi-agent evaluation.

### Dry run
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --dry_run
```

## Experiment 2C: recommended_lateral_stable_v2 Final Comparison

### 1) Experiment 2B result
Local fine sweep selected `recommended_lateral_stable_v2` (`yaw_008_jerk_020`) as the best current p2-oriented configuration.

### 2) Recommended lateral_stable v2 parameters
- `heading_smooth_alpha = 0.75`
- `yaw_rate_clip = 0.008`
- `thw_target = 1.70`
- `jerk_limit = 0.200`
- `a_max = 1.275`
- `a_min = -2.52`

### 3) Final comparison command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2 \
  --config_set final_compare \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### 4) Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2_debug \
  --config_set final_compare \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### 5) Expected outputs
- `final_config_comparison_summary.csv`
- `final_config_comparison_summary.json`
- `final_config_comparison_report.md`
- `final_config_p2_separation.png`
- `final_config_margin.png`
- `final_config_classification_retrieval.png`
- `final_config_style_metrics.png`
- `final_config_tradeoff.png`
- `ablation_integrity_report.json`

### 6) How to interpret
- `p2_farthest_rate` higher is better.
- `mean_p2_separation_margin` closer to or above 0 is better.
- `centroid_accuracy_p2` measures p2 recognizability.
- `p2_rms_jerk` lower means smoother longitudinal behavior.
- `p2_rms_yaw_rate_proxy` lower means stronger lateral stability.
- Negative `mean_p2_separation_margin` means p2 is not yet fully independent.

### 7) Limitations
- Synthetic policy rollout only.
- Replayed front vehicle.
- No real human driver labels yet.
- No sensor rendering / perception stack.
- PCA / UMAP are visualization only.


## Phase 4A: Public Human Trajectory External Validation Scaffold

Purpose: validate whether embedding structure transfers beyond synthetic generator artifacts using trajectory-level weak-label evaluation.

### Unified input format
`traj.npy`, optional `front.npy`, `meta.npy`, `split.npy`, `feat_style.npy`, optional `feat_style_raw.npy`, optional `feature_names_style.json`, optional `embeddings.npy`.

### Pseudo-label assignment
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir <HUMAN_DATA_DIR> \
  --out_dir outputs/vehicledata_validation/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1
```

### Evaluation
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval \
  --embedding_path <OPTIONAL_EMBEDDING_PATH> \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

Baselines-only mode:
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

### Outputs
Pseudo-label outputs include summary/report/distribution files. Evaluation outputs include `human_validation_summary.json`, `human_validation_report.md`, `baseline_comparison_summary.csv`, retrieval/classification/correlation/cluster artifacts and figures.

### Interpretation and limitations
Pseudo labels are rule-based weak labels (not ground truth) for external validation only. Label-defining features can leak into classification metrics, so retrieval, cluster fingerprints, and baseline comparisons must be interpreted jointly.

### Smoke tests
Both scripts support `--smoke_test` and generate synthetic arrays locally without external dataset downloads.

## Phase 4A Validation Integrity Updates (2026-05-08)
- Added `--allow_skip_learned` to skip learned embedding only with explicit opt-in.
- Default retrieval mode is now `--retrieval_mode strict` with exclusions for same scenario/agent/track/source and temporal neighbors.
- Added retrieval chance and lift metrics in `baseline_comparison_summary.csv`.
- Expected plots include baseline classification/retrieval/style-correlation bars and representation PCA fallback plot.
- Cluster outputs are split into `cluster_size_distribution.*` and style fingerprint heatmap/csv outputs.

## Embedding alignment requirement（阶段4A关键约束）

- `traj.npy` / `meta.npy` / `feat_style.npy` / `pseudo_label.npy` 都是**按行对齐**的样本级数组。
- learned embedding 在 `tools/evaluate_vehicledata_validation.py` 中必须满足 `embedding.shape[0] == N_samples`。
- source-level embedding（例如每个 `source_index` 一行）**不能**默认用于 policy-level / pseudo-label evaluation。
- 仅在显式传入 `--allow_source_level_embedding_expansion` 时，才允许按 `source_index` 展开，且该结果仅用于 debug，不可作为 policy-level 有效结论。

`data1` 的已知情况：

- `traj` 行数 = `33471`
- `embeddings` 行数 = `11157`
- `33471 = 11157 x 3`，对应每个 source 的 3 个 rollout（p0/p1/p2）

这说明 `data1/embeddings.npy` 是 source-level，不是 row-level。评估 learned baseline 前必须先再生成 row-level embedding。

建议命令（当前为 TODO 占位，`tools/export_row_level_embeddings.py` 尚未实现）：

```bash
python tools/export_row_level_embeddings.py \
  --data_dir data1 \
  --model_ckpt <CHECKPOINT> \
  --out_path data1/embeddings_row_level.npy
```


## 阶段 4B：Waymo 真实人类轨迹数据提取

### 命令
```bash
python tools/build_waymo_human_trajectory_dataset.py \
  --waymo_dir <WAYMO_TFRECORD_DIR> \
  --out_dir outputs/waymo_human_v1 \
  --window_len 80 \
  --stride 20 \
  --min_speed 1.0 \
  --max_files 5 \
  --max_scenarios 200 \
  --max_agents_per_scenario 64 \
  --split_by_scenario \
  --overwrite

python tools/build_waymo_human_trajectory_dataset.py \
  --out_dir outputs/waymo_human_smoke \
  --smoke_test \
  --overwrite
```

后续 Stage 4C：
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir outputs/waymo_human_v1 \
  --out_dir outputs/waymo_human_v1/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1 \
  --dataset_type human_public

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1 \
  --label_dir outputs/waymo_human_v1/pseudo_labels \
  --out_dir outputs/waymo_human_v1/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

### 期望行为
- 从原始 Waymo 场景中提取真实 human vehicle agent 的 observed trajectory window。
- 不调用 synthetic policy generator。
- 不生成 p0/p1/p2。
- 不生成 policy_id / policy_name。
- 输出统一格式 npy 文件。
- 每一行对应一个真实 human agent trajectory window。
- split 按 scenario_id hash 分配，避免同一 scenario 泄漏到不同 split。
- 自动计算 style features 和标准化特征。
- 自动生成 build_summary.json 和 build_report.md。

### 通过标准
- out_dir 下生成 traj.npy / front.npy / meta.npy / split.npy / feat_style.npy / feat_style_raw.npy / feature_names_style.json。
- meta.npy 中 dataset_type = human_public。
- meta.npy 中不包含 policy_id / policy_name。
- len(traj) == len(front) == len(meta) == len(split) == feat_style.shape[0]。
- build_summary.json 中 n_windows_kept > 0。
- split_counts 中 train/val/test 至少有一个非空，正式运行应三者都有数据。
- front_found_rate 被记录。
- feature_names_style.json 与 feat_style 的列数一致。
- smoke_test 可以不依赖真实 Waymo 数据运行成功。


## 阶段 4D：训练并导出 Waymo human row-level learned embedding

### 1. 命令
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --batch_size 1024 \
  --device cuda \
  --traj_nan_mode interpolate \
  --max_traj_nan_ratio 0.2 \
  --overwrite
```

### 2. 期望行为
- Waymo human `traj.npy` 可能包含 NaN，因为观测轨迹可能部分无效。
- `export_human_row_embeddings.py` 必须复用训练脚本相同的轨迹清洗与局部归一化逻辑。
- 导出的 embedding 必须与 `traj.npy` 行对齐（row-aligned）。
- 若 `normalize_local` 产生非有限值（NaN/Inf），必须立即失败，禁止保存坏 embedding。
- 若 checkpoint 训练过程中出现 NaN loss，禁止导出，必须先修复并重训。

### 3. 通过标准
- 控制台输出 raw/sanitized 的 NaN/Inf 统计。
- `embedding_export_summary.json` 与 `embedding_export_debug.json` 成功生成。
- `embeddings_row_level.npy` 全量 finite，且 `shape[0] == len(traj.npy)`。
- `row_aligned = true`（官方 Stage 4D 默认不允许 drop）。


## 阶段 4E：jerk/comfort-aware learned embedding 训练

### 命令
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode jerk_comfort \
  --device cuda \
  --seed 42 \
  --overwrite

python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

### 期望行为
- 使用 train split 训练。
- 不使用 pseudo labels 训练。
- 提高 jerk/comfort 特征在 soft contrastive distance 中的权重。
- 导出 row-aligned embedding。
- 在 test split 上与 Stage 4D v1 和 baselines 对比。

### 通过标准
- train_loss / val_loss finite。
- embeddings_row_level_jerk_comfort.npy shape = [168191, 64]。
- learned_embedding_evaluated=true。
- 若权重生效，rms_jerk_delta 相关性优于 Stage 4D v1。
- retrieval/classification 不低于 random。
- report 与 paper tables 均生成。

## 阶段 4F：comfort-aware auxiliary regression（当前主线）

Stage 4D learned embedding 在 jerk 相关性上偏弱；Stage 4E 的 jerk/comfort feature weighting 没有改善 jerk correlation，且分类/检索略有退化。因此 Stage 4F 不再仅靠特征权重，而是在 embedding 上增加显式 comfort auxiliary regression 监督（rms_accel/rms_jerk/max_abs_accel/max_abs_jerk/mean_thw/min_thw），目标是提升 jerk/comfort 敏感性，同时保持 learned embedding 的判别能力。训练依然不使用 pseudo labels。

Stage 4F 评估分两部分，缺一不可：
1. auxiliary head prediction quality（`tools/evaluate_aux_predictions.py`，检查 MAE/RMSE/Spearman，确认 head 真的学到 comfort 目标）；
2. embedding retrieval/classification/style-distance correlation（`tools/evaluate_vehicledata_validation.py` 等，下游几何泛化能力）。

## Stage 4G（当前进行中）：comfort metric alignment

- 当前 active experiment 为 **Stage 4G**。
- Stage 4F 结论是：auxiliary regression 证明 jerk/comfort 信息在 embedding 中可解码，但 embedding 的欧氏距离几何仍未与 jerk 差异对齐。
- Stage 4G 在 Stage 4F 基础上增加 pairwise metric alignment：直接对齐 `embedding distance matrix` 与 `comfort feature distance matrix`。
- 目标是不仅“可预测 jerk”，还要让 embedding 几何本身对 jerk/comfort 更敏感，并提升 `spearman_rms_jerk_delta`。


## Stage 4D/4E/4F/4G 结论更新（当前主结果）

当前主方法为 **Stage 4G: comfort metric alignment**。

- Stage 4D：建立了可用的 learned behavior embedding，但 jerk 敏感性较弱。
- Stage 4E：仅做 jerk/comfort 特征重加权，未有效提升 jerk-sensitive 几何。
- Stage 4F：辅助回归证明 jerk/comfort 在 embedding 中“可解码”，但 embedding 距离几何本身仍未对齐 comfort。
- Stage 4G：直接约束 embedding pairwise distance 对齐 comfort metric pairwise distance，显著提升 jerk/comfort-sensitive 检索，同时保持分类/检索不塌缩。

> 重要说明：Stage 4G 不是“纯无监督发现”，而是 **metric-aligned behavior embedding**（通过 comfort metric 对 embedding geometry 施加显式结构约束）。


## Stage 5A 数据构建

仓库已新增 Stage 5A lane-aware 5-neighbor context 数据构建脚本：`tools/build_waymo_5neighbor_context_dataset.py`，用于在训练前验证交互上下文输入质量。设计说明见 `07_stage5_interaction_design.md`。

- Stage 5A-v2 focuses on true lane-aware neighbor assignment.

## Stage 7 M4（当前 nuPlan 闭环主结果）

Stage 7 已完成45个 same-scenario、assertive-vs-conservative PDM closed-loop
planner pairs，并通过 scenario/token/msgpack、地图投影、validity mask 和
lane-context Tier sensitivity 审计。

M4 正式统计结果：

- assertive mean speed delta：`+1.4277 m/s`，95% paired bootstrap CI
  `[1.0106,1.8723]`，paired dz=`0.948`；
- assertive RMS acceleration delta：`+0.2562 m/s²`，95% CI
  `[0.1701,0.3416]`，paired dz=`0.862`；
- Full embedding BDD：MMD²=`0.0142209`，1000-permutation
  `p=0.733267`；
- Tier A / Tier A+B BDD 同样不显著。

当前结论是：PDM assertive 参数在 official same-scenario rollout 中稳定改变
trajectory-level speed/acceleration，但现有 behavior encoder 的总体与
task-conditioned embedding distribution BDD 未检测到显著漂移。

M4 是对已观察 M3 exploratory result 的 retrospective formalization，不是独立
预注册确认实验。THW 仅有35个 finite available-case pairs，均值CI跨零；task
detector 仍存在 overlap 和 proxy 语义限制。完整结果见
`outputs/stage7_m4_statistical_evidence_v1/`。

M5 进一步解释 paired trajectory evidence 与 marginal BDD 的差异：learned
embedding 的同场景 sign-flip `p=0.0001`，scenario-grouped planner probe
ROC-AUC=`0.638`、pair-swap `p=0.00699`，说明embedding确实包含可推广的planner
信息；但 marginal MMD `p=0.733`。interaction features和trajectory summary也呈现
paired/probe显著而marginal MMD不显著。结论是跨场景异质性与estimand差异是主要
机制之一，不能把BDD不显著简化为embedding完全没有行为信息。完整结果见
`outputs/stage7_m5_representation_mechanism_v1/`。

M6 使用相同 embedding 和相同45个 scenario pairs，把 MMD 的 permutation null
与 matched simulation 设计对齐。original-space MMD²=`0.0141802` 在 pooled
shuffle 下 `p=0.737126`，在 within-scenario pair swap 下 `p=0.002300`；
pair-midpoint residual BDD 为 MMD²=`0.0994187`、`p=0.000100`。这证明现有
embedding 已编码稳定的同场景 planner shift，先前的主要矛盾是 marginal estimand
丢弃了 pairing，而不是模型完全失效。

M6.1 将 primary estimator、100000次 permutation、exceedance count、输入与
checkpoint hash、pair audit 和 quality sensitivity 正式冻结。Primary
exceedance=`175/100000`、plus-one `p=0.001760`；45/45 pairs 完整，Tier A 40对
和 Tier A+B 44对的 sensitivity 经 Holm correction 后仍显著，已测
fallback/ambiguous-rate 相关性经 Holm correction 后均不显著。

M6 保留 M4 marginal BDD，不把 paired BDD 用于异源实路日志。当前45对是方法开发集，
不是独立确认集；`mean_speed` supervision 只是候选改进假设，绝对速度还可能泄漏
ODD/限速。Waymo-only Stage5D-balanced-v2 继续作为主跨域模型，下一步是新
log/scenario-disjoint 且 selection config 独立冻结的锁定确认，而不是默认联合
重训练；两套 planner treatment 参数本身必须保持一致。M6.1 结果见
`outputs/stage7_m6_1_paired_bdd_method_freeze_v1/`，协议见
`docs/stage7_cross_domain_style_sensitive_training_protocol.md`。

M6.2 进一步冻结了新确认集入口和 pre-treatment task-conditioned paired BDD。
新数据必须与当前45对在 log/scenario 上不相交，同时保持两套 planner 参数指纹
完全一致。当前五个 task 各有8–9对，全部低于12对运行下限；learned embedding
只有 high-motion dynamics 在开发集上通过 Holm correction
（exact p=`0.00390625`，Holm p=`0.01953125`）。这说明总体 paired effect 存在
task heterogeneity，不能宣称所有任务都有同等明显差异。结果见
`outputs/stage7_m6_2_locked_task_bdd_development_v1/`。

M6.3 已把锁定确认集的样本规模变成机器可执行的 simulation-based power
justification。主设计假设锁定域效应至少保留开发 pilot 均值差的75%，要求五个
冻结任务各60个完整 pairs；按20%损耗率应采各75对，共375对，五任务 simultaneous
Holm-corrected power=`0.918`（95% CI `[0.891,0.939]`）。Overall complete pairs
另受 M6.2 质量下限约束，至少80对。若按50%效应做保守预算，需求上升到每任务
160个完整 pairs、总计1000个 gross pairs。M6.2 locked mode 会核验 power 文件
hash、task mapping、配额和 planner fingerprints，不满足即拒绝运行。M6.3 仍是
开发 pilot 驱动的采集规划，不是 achieved power，也没有触发重新训练。主设计见
`outputs/stage7_m6_3_simulation_power_v1/`。

M6.4 已实现 outcome-blind collection preflight：候选选择只使用 nuPlan
`scenario_type` 元数据，并强制开发 token/log 零重叠、歧义标签排除、DB 文件存在、
每 log 最多2个场景、planner fingerprint 及 M6.2/M6.3 SHA256 链一致。只有五任务
各75个 primary 和15个 reserve 均满足时才会生成 locked collection manifest。

当前本机只有63个 mini logs，其中34个已进入开发集；剩余 inventory 的冻结
lane-change 候选仅2个。因此预检状态为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，未生成 locked manifest、未启动
rollout。下一步必须增加新的 nuPlan logs 并重建 scenario-tag inventory，而不是
复用开发 log、放宽任务定义或提前重训练。审计见
`outputs/stage7_m6_4_locked_collection_preflight_v1/`。

M6.4A 已新增 first-class multi-DB inventory builder
`tools/stage7p_build_scenario_inventory.py`（GitHub Issue #236）。工具只读取一个或
多个 nuPlan SQLite DB root 中的 `scenario_tag`、`lidar_pc`、`scene` 和 `log`，
流式生成 M6.4 兼容的 `all_scenario_tags.csv`，同时输出 DB SHA-256 输入清单、
summary、报告和可选的扁平相对符号链接 DB pool。它不会读取 planner outcome、
trajectory、embedding 或 BDD，也不会自动运行 M6.4 preflight 或启动 rollout。
Mac mini-only smoke 扫描64个 DB、63个 logs 和892204个原始 tag rows；按
token/type/log/DB 去重后输出821831行，移除70373个重复 tag，token-location
冲突为0。重建后的 M6.4 task-capacity CSV 与历史版本逐字节相同，状态仍为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，未生成 locked manifest。在扩展
DB 到位前，mini-only 容量不足结论保持不变。

M6.4A Pittsburgh 扩展已于2026-08-07完成。官方 DB-only ZIP 精确大小为
`30620248893` bytes，CRC 测试通过；安全解压得到1560个 SQLite DB（51.90 GiB）。
其中3个 DB 与 mini 同名且 SHA-256 完全一致，因此 expanded 输入使用1560个
Pittsburgh DB 加61个非重叠 mini DB，共1621个 DB、1576个 logs。最终 inventory
包含9604184行、5386575个 unique scenario tokens，token-location conflict为0。

expanded M6.4 preflight 状态为 `FROZEN_BEFORE_LOCKED_ROLLOUTS`，五个冻结任务均
选满75个 primary 和15个 reserve；375个 primary来自306个 logs，primary+reserve
共覆盖350个 logs，每log最多2个。development token/log overlap均为0，planner
fingerprints保持不变，已生成锁定 manifest 和375行 Stage7C primary context。
M6.4B Mac readiness 与首个 locked scenario 双 planner smoke 已于2026-08-07通过。
tuPlan Garage 固定在 `b51d5d04fac1bd4389653b9ab2ff73ea88f435a3`，nuPlan devkit 固定在
`e9241677997dd86bfc0bcd44817ab04fe631405b`；PDM readiness 为
`ready_for_pdm_smoke`，两套 planner fingerprints、Stage7C tool hash 和 primary /
reserve CSV hash 均与 locked manifest 一致。首个 primary token
`6b5a9da8c0b353b9` 的 assertive / conservative official commands 均成功，输出
shape=`(1,2,149,8)`、298个有效 timesteps、missing pair=0，same-log 与 strict token
alignment 均通过，且未生成 pseudo rollout。该 smoke 只运行了2/750个 primary
rollouts；其余748个仍未启动。

M6.4B 批处理执行层已由 GitHub Issue #237 固化为
`tools/stage7_m6_4b_run_locked_rollouts.py`。它在每次启动前复核 locked manifest、
primary/reserve CSV、Stage7C hash、planner fingerprints、450个 DB 文件和
nuPlan/tuPlan commits；默认只生成 dry-run plan，真实运行还要求 `--execute` 和
primary manifest hash 双重确认。每个场景使用独立 one-row context、绝对输出路径和
append-only attempt/event 审计；`--resume` 只跳过重新验证为完整 PASS 的 pair，失败
不会覆盖旧 attempt。Reserve 仅生成 `PROPOSED_NOT_APPROVED_NOT_EXECUTED` 提案，工具
本身禁止自动执行。真实 batch smoke 已将 order 1 记录为 `SUCCEEDED`，原样 resume
未新增 event 或 attempt；当前权威 batch state 为1/375场景成功、374 pending、0
failed、0 reserve proposals，其余748个 rollout仍未启动。最终 v2 batch manifest
还冻结 batch tool SHA-256
`ef0026b3cc20942846035ac23d0d16d616a3d7dd6675e9a0f9c2612871d7fb06` 和 command
timeout，后续 runner 代码或 timeout 变化会使 resume fail closed。

随后按冻结顺序完成 order 2–6 技术 canary，五个场景覆盖 lane-change、stop-go、
high-motion、dense/vulnerable 和第二个 following；全部双 planner pair PASS，未产生
失败或 reserve 提案。连同 order 1，当前 batch state 为6/375场景成功、369 pending。
六个场景端到端耗时为30.70–41.05秒，均值35.48秒、中位数35.20秒；order 2–6
连续批次的实际 wall time 为176.64秒，即35.33秒/场景。按该 wall rate 外推，原始
374个 pending 场景约需3小时40分；canary后剩余369个约需3小时37分。按观测最慢值
外推上界约4小时16分，正式运行应预留4.5–5小时并防止Mac休眠。

M6.4B 全量冻结批次已于2026-08-07完成375/375个 primary 场景。原始结果为283
`SUCCEEDED`、92 `FAILED_REVIEW_REQUIRED`；任务成功数依次为 following=60、
lane-change=50、stop-go=67、high-motion=43、dense/vulnerable=63。375个场景的
attempt 端到端均值为27.54秒、中位数30.72秒。失败后的 M6.4C 审计只读取冻结
collection、SQLite scene 结构和技术状态，不读取 embedding、BDD、effect size、
trajectory metric 或 planner outcome。审计确认90条失败 token 位于 nuPlan 官方
scene 查询排除的边界位置，另2条 scene 有效但 token 会被 Hydra/OmegaConf 解析成
整数或科学计数法，必须以保留引号的字符串重试。

M6.4C 由 GitHub Issue #238 固化为独立审计与恢复流程：
`tools/stage7_m6_4c_audit_locked_recovery.py` 生成不可覆盖的技术审计和冻结配额方案，
`tools/stage7_m6_4c_run_locked_recovery.py` 只执行显式选择的 quoted-primary 或 frozen
reserve 动作；M6.4B batch runner 和 Stage7C 源文件保持原 SHA-256 不变。权威审计位于
`outputs/stage7_m6_4c_locked_recovery_audit_v2/`：primary 分类为283成功、90无效
scene position、2 quoted-token retry；75 reserve 中58条技术可运行、17条无效。

两条 quoted primary retry 均通过双 planner、trajectory completeness、same-log 和
strict-token alignment，耗时37.26秒与30.09秒。随后按冻结 task-rank 执行 lane-change
10条和 high-motion 10条 reserve，20/20全部通过，均值33.00秒、中位数32.55秒。
恢复后可用完整 pairs 总数为305：following=60、lane-change=60、stop-go=67、
high-motion=55、dense/vulnerable=63。除 high-motion 外均达到每任务60对的冻结要求；
high-motion 仍缺5对，而且冻结 primary+reserve 已耗尽。不得从集合外静默补样；下一步
必须先提交 outcome-blind supplemental protocol amendment，再冻结新增候选及哈希。

M6.4D outcome-blind high-motion supplement 已于2026-08-08完成（GitHub Issue #239）。
新工具从原 eligible inventory 中排除 development 及原375 primary/75 reserve 的全部
token/log，使用独立 salt 和 nuPlan 官方 scene-position 技术门冻结5个 primary 与5个
reserve。补充集合与既有数据的 token/log overlap 均为0；固定顺序前16个候选中，
4个无效 scene position、2个重复 log 被排除，最终10个候选全部技术可运行。

5个 supplemental primary 已全部运行成功，双 planner、trajectory completeness、
same-log 与 strict-token alignment 均通过；均值31.14秒/场，0失败，因此5个 supplement
reserve 保持未执行。最终有效完整 pairs 总数为310：following=60、lane-change=60、
stop-go=67、high-motion=60、dense/vulnerable=63，五任务均达到预冻结的每任务60对
要求。M6.4D 仅补齐 collection/technical completeness；下一步应按冻结的 M6.2/M6.3
统计协议生成确认性 task-conditioned paired BDD，不得再改变 estimand 或显著性规则。

M6.5 locked confirmation 已于2026-08-08完成（GitHub Issue #240）。新工具先把
M6.4B/C/D 的283+2+20+5个成功场景固化为310-pair ledger，并逐场景重跑 Stage7C
audit；确认集相对45-pair development set 的 token/log overlap 均为0。分析锁在读取
确认 embedding 前固定 M6.1/M6.2 工具、100000次 within-pair swaps、plus-one p、
exact pooled median bandwidth、Holm families、质量阈值、checkpoint 和全部入口 hash。

Mac context build 必须把本地 `tuplan_garage` 加入 `PYTHONPATH`。一次缺少该路径的
预检被识别为 neighbor 全空并隔离，未进入统计；修正后310场 context 在23分56秒完成，
620 rows、83D schema 通过，front/side neighbor coverage 非零。总体原始64D embedding
primary 为 MMD²=0.004469、0/100000 null exceedances、plus-one p=9.9999e-6。五个
pre-treatment task 的 learned-embedding BDD 全部通过 Holm（adjusted p 从0.00030到
0.01820），满足新的 log/scenario-disjoint confirmation。

质量结论须和主结果同时保留：Tier A=58、Tier A+B=135，两个原始-embedding
sensitivity 均通过 Holm（0.0182），但 Tier A residual p=0.126；全局 lane fallback
为10.59%，且 fallback 与 embedding pair distance 强相关。因此结果证明冻结 planner
treatment 下存在可检测的 behavior-distribution difference，不证明安全性或 planner
优越性，也不能把全部差异归因于不受 lane-context quality 影响的纯 mechanism。

M6.6 paper evidence package 已于2026-08-08完成（GitHub Issue #241）。新工具
`tools/stage7_m6_6_build_confirmation_evidence.py` 在生成任何产物前重新验证 M6.5 lock
及结果输入的全部 SHA-256、310 pairs/620 rows、development disjointness、power targets、
五任务 Holm 结果和58/135质量计数。它不重算确认性 p 值，而是把锁定结果、planner
treatment、mechanism controls、质量敏感性、运动学对比整理成10张 CSV/Markdown 表、
6张 PNG/PDF 图、双语 manuscript text、JSON summary 和完整 provenance。

探索性质量归因使用固定 seed 的10000次 bootstrap，并同时报告总体任务分层、五个
任务内及 task-adjusted rank residual 关联。最大 paired fallback 与 embedding distance
总体 rho=`0.5088`（95% CI `[0.4086,0.6035]`），任务调整后 rho=`0.4499`
（95% CI `[0.3842,0.5719]`）。这些量均为 post-treatment descriptive evidence，不能
作为因果调整；最终状态为 `PASS_WITH_QUALITY_LIMITATIONS`。详见
`docs/stage7_m6_6_confirmation_evidence_protocol.md` 和
`outputs/stage7_m6_6_confirmation_evidence_v1/`。

## Stage 6D：异源实路软件版本风格比较

Stage 6D（GitHub Issue #242）增加 `tools/stage6d_unpaired_version_bdd.py`，用于比较
无法同场景配对的两个软件版本路试集合。工具同时输出实际采集构成下的 raw BDD 和
共同支持域、equal-group pooled reference 标准化后的 BDD，并提供 support fraction、
ESS、最大权重、covariate balance 与 log/route/day/vehicle cluster bootstrap 区间。

设计 JSON 强制 matching covariates 和 task slices 为 pre-treatment；急刹、迟疑、
变道结果等软件行为 outcome 不能用于匹配。共同支持或有效样本不足时工具返回
`NOT_COMPARABLE_INSUFFICIENT_COMMON_SUPPORT`。结果是观察性风格漂移证据，不是
因果、安全或版本优劣结论；生产阈值仍需独立的同版本 A/A 历史窗口标定。

nuPlan 310-pair balanced embedding 已完成接口冒烟：20/20 common-support cells、两组
ESS ratio=1.0，raw 与 standardized overall MMD² 均为 `0.0044865829`。该结果只验证
接口和标准化实现，不替代真实异源路试验证。完整协议见
`docs/stage6_unpaired_style_drift_protocol.md`。

## Stage 6E：公开 field-release emulation

Stage 6E（GitHub Issue #243）新增 `tools/stage6e_calibrate_unpaired_release.py`。工具将
310个 paired nuPlan 场景按257个 logs 拆成完全 log/token-disjoint 的伪发布集合，以
同 planner A/A 的200个 calibration trials 冻结95% empirical threshold，再用独立
seed streams 的200个 A/A 和200个双方向 A/B trials 估计误报率和检出率。

总体 standardized BDD threshold=`0.00994295`；A/A holdout 为7/200、3.5%误报
（Wilson 95% CI `[1.7%,7.0%]`），A/B 为70/200、35.0%检出
（`[28.7%,41.8%]`）。两区间分离且 detection/false-positive ratio=10，但单次发布
敏感性仍有限。任务诊断里 lane-change 最强（53.5% vs 6.0%），stop/go 没有检测增益。

因此当前证据支持“公开闭环基准中，不配对版本仍有可检测信号”，但不支持“已经能够
稳定判定每次实路版本发布”。任务行没有多重性控制；公司数据到位后仍须重新 A/A 标定。

## Stage 6F：样本量功效曲线与公开数据充分性

Stage 6F（GitHub Issue #244）新增 `tools/stage6f_unpaired_power_curve.py`，对每版本
40/60/80/100/125/150个场景分别运行600次 log/token-disjoint pseudo releases，并为
每个 n 单独重标定 A/A threshold。总体 A/B detection 依次为7.0%、10.5%、12.0%、
11.5%、17.0%、35.0%；有限场景池和 Monte Carlo 波动使观测曲线不要求严格单调。

n=150 时 detection Wilson 95% CI=`[28.7%,41.8%]`，A/A false-positive=7.0%
（`[4.2%,11.4%]`），未达到 detection 下界≥80%且 false-positive 上界≤5%的冻结门，
状态为 `TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`。不允许在150以上做伪精确外推。

要增加下一个200/250/300/400场景/版本档位，唯一场景池至少需400/500/600/800条，
即相对当前310条至少新增90/190/290/490条；这只是实验集合规模，不保证达到80%功效。

## Stage 6G–6I：800场景扩展与论文结论冻结

Stage 6G（Issue #245）按outcome-blind冻结顺序完成新增490/490场景的两套official planner
rollout；与原310场景合并后，Stage 6H（Issue #246）得到800个完整pair、1600条64D
embedding和489个log clusters。全部2400个release splits均精确达到200/250/300/400目标
样本量，且版本间log和scenario overlap为0。

扩展曲线在400场景/版本时得到A/B detection=66.5%（Wilson 95% `[59.7%,72.7%]`）和
A/A false-positive=5.0%（`[2.7%,9.0%]`）。Stage 6I（Issue #247）进一步冻结可靠性
分解：四个观测档位的A/B与A/A Wilson区间均分离，支持“已知软件风格差异在异log、
异场景公开release emulation中仍可检测”；但400场景/版本的false-negative仍为33.5%，
所以不支持80%单次发布可靠性或通用BDD阈值。

论文可主张跨域表征和公开异场景版本差异检测的可行性，但不能声称已完成真实整车厂
软件版本验证、量产可靠报警或因果归因。公司数据可用后必须重新执行同版本A/A标定和
已知A/B版本验证。详见`docs/stage6_unpaired_style_drift_protocol.md`及
`outputs/stage6i_reliability_evidence_v1/`。

## Stage 6J：Waymo模型检出nuPlan纯纵向风格差异

Stage 6J（Issues #249–#251）针对“原PDM A/B同时改变横向参数，不能证明纯纵向检出”
的问题，冻结了横向参数完全相同、只改变speed/headway/gap/accel/decel等六项纵向参数
的两套PDM closed planners。主集合排除lane-change、dense/vulnerable和
high-lateral-acceleration，仅保留following=60、stop/go=67、longitudinal
high-motion=56，共183个same-scenario pairs、366条official rollouts、156个logs。

183/183场景运行和重新审计全部通过。Stage5D 5邻车context在Mac上耗时14分37秒，
输出366×150×83。读取本批结果前冻结的运动学门禁也通过：平均速度A-B=0.9147 m/s，
log-cluster bootstrap 95% CI `[0.7578,1.0784]`；RMS加速度A-B=0.1816 m/s²，
95% CI `[0.1456,0.2175]`。这说明处置本身确实产生了足够强的纯纵向差异。

使用原Waymo Stage5/6的83D context GRU checkpoint导出366×64 embedding后，预冻结
paired BDD总体MMD²=`0.00500090`，100000次pair内label swap中0次达到observed，
plus-one p=`9.9999e-6`。三个pre-treatment task经Holm校正后也全部reject：following
MMD²=`0.01706723`、Holm p=`0.00129999`；stop/go MMD²=`0.00537483`、Holm
p=`0.03300967`；longitudinal high-motion MMD²=`0.01358617`、Holm
p=`0.0000299997`。

因此当前证据直接支持窄论文主张：Waymo训练的模型可以检出nuPlan闭环中人为设置且经
运动学验证的典型纯纵向风格差异。MMD²的绝对数值依赖embedding尺度与kernel bandwidth，
不能因约0.005就单独判为“太小”；可检出性应以预冻结随机化null为准。该结果仍不等价于
异log/异场景单次release的高可靠性、通用BDD阈值或真实整车厂版本验证。

## Stage 6K：纯纵向风格最小可检出剂量

Stage 6K（Issue #252）在Stage 6J相同183场景、156 logs和同一Waymo checkpoint上增加
25%、50%、75%三档纵向IDM参数插值。549/549个scene-dose任务和1098/1098条official
rollout全部成功。新增embedding/BDD读取前冻结四档overall Holm、12项task×dose Holm、
实现运动学门禁、同log整体翻转及lane-quality post-treatment敏感性。

25/50/75/100%四档的平均速度差依次为0.255/0.446/0.637/0.915 m/s，RMS加速度差依次为
0.036/0.077/0.128/0.182 m/s²；两个门禁指标的单侧log-cluster 95%下界四档均大于0。
overall BDD依次为0.001156/0.001600/0.003322/0.005001，四档Holm p分别为0.004290、
0.000040、0.000040、0.000040。因此本冻结协议内最小可检出名义剂量为25%；该档
BDD/null-q95=1.290、Z_BDD=3.649。同log整体翻转四档也全部显著。

task结果并非一致：25%只有longitudinal-high-motion通过12项Holm，following到75%才通过，
stop/go在50%和75%通过但100%校正后未通过。lane-quality诊断还显示fallback rate与
embedding pair距离中等正相关，所以当前结果支持“受控同场景纯纵向差异可检出”的窄主张，
但不能声称25%所有task均可靠、存在通用BDD阈值或全部信号都不受context assignment质量
影响。中文总报告见`outputs/stage6k_final_conclusion_v1/stage6k_final_report_zh.md`。

### Stage 6K context 完整性修正（2026-08-11）

后续 Stage 6L 审计发现，原 Stage 6K `dose50/dose75` context 因 Mac 运行环境缺少完整
nuPlan/tuPlan `PYTHONPATH`，366/366行均发生`msgpack_timestep_mismatch samples=0`，
`neighbor_seq.npy`有效邻车覆盖为零。旧构建器把全局零覆盖当成低覆盖跳过项，因此仍错误
标记validation PASS。

旧 dose50/75 的完整64D embedding/BDD和THW/gap结果不再作为权威完整-context证据；
rollout ego速度、加速度和jerk仍有效。旧目录不删除、不覆盖。修复版写入
`outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/`，三档邻车槽位帧覆盖率为
17.14%/17.44%/17.37%，并新增构建阶段和Stage6L freeze阶段的非零覆盖fail-closed门禁。

修复后完整64D overall BDD在25/50/75/100%分别为0.001156/0.002025/0.003598/0.005001，
四档Holm均通过，25% overall最小检出剂量结论不变。原
`outputs/stage6l_context_representation_ablation_results_v1/`已标记superseded；权威结果为v2。

## Stage 6L：Context-quality representation 消融

Stage 6L（Issue #253）在修复版相同183场景剂量曲线上比较完整learned64、同checkpoint
邻车置零64D、显式ego运动学13D和手工交互+轨迹46D。不同表示各自使用独立bandwidth/null，
禁止跨表示比较raw MMD²。

完整64D、邻车置零、ego13D和手工46D的median overall Z_BDD分别为7.539、11.066、
21.082、5.384；12个task×dose Holm通过数为7、11、12、2；最小overall检出剂量分别为
25%、25%、25%、50%。因此当前完整64D没有相对ego-only control增加纯纵向敏感性，邻车
context在本受控问题上更像稀释ego信号。该结论不表示interaction对其他驾驶风格无用。

fallback与pair L2的正关联在邻车置零后仍存在但系数减弱，说明它不能全部归因于邻车通道。
预冻结决策为准备独立context-v2协议和新训练协议，但GO不授权覆盖checkpoint或结果导向调参。
权威结果位于`outputs/stage6l_context_representation_ablation_results_v2_runtime_repaired/`。

## Stage 6M：四种异场景 release BDD 方法

Stage 6M（Issue #254）复用Stage6H 800 pairs、489 logs和既有log/token-disjoint release
splits，在聚合结果前冻结四种release statistic：raw marginal、task-conditioned、
context-balanced、task+context-balanced。每种方法与每个200/250/300/400样本量均使用独立
A/A calibration threshold；平衡只使用pre-treatment map_name、scenario_type和冻结task。

n=400时四方法A/B detection为63.0%/65.0%/66.5%/64.5%，A/A FPR为4.5%/5.5%/5.0%/
6.0%。context-balanced相对raw为+3.5个百分点，但同split配对McNemar exact p=0.2478，
不支持稳定提升。协变量审计重建28800行；2个task scope-trial因支持度不足不可比，其余
加权后map/scenario-type最大类别比例差为1.22e-15，证明平衡实现正确。

所以当前约33.5%假阴性不能主要归因于已测量场景构成不平衡。主要瓶颈是representation
sensitivity，scenario heterogeneity次之，A/A calibration仍是部署必要条件。权威结果位于
`outputs/stage6m_context_balanced_unpaired_bdd_results_v1/`。

## Stage 6N：新 checkpoint Go/No-Go

Stage 6L预冻结规则已触发`GO_PREPARE_SEPARATELY_VERSIONED_TRAINING_PROTOCOL`，Stage 6M又
显示场景平衡没有解决unpaired检出瓶颈。因此Issue #255只负责准备独立新checkpoint协议：
扩大Waymo纵向覆盖，加入longitudinal contrastive/ranking、speed/accel/jerk/THW/gap
auxiliary objectives与context dropout/quality mask。旧checkpoint SHA-256
`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`保持冻结baseline。

GO不是立即训练或覆盖授权。新模型在任何结论前必须重新通过Waymo validation、Stage6J/6K
paired dose curve、Stage6M unpaired A/A calibration、A/B detection及FPR/FNR tradeoff。
完整中文主报告见`docs/stage6n_context_balanced_retraining_decision.md`。

## Stage 6O：纵向敏感 64D 新模型训练前冻结

Stage 6O（Issue #256）已经冻结独立新模型协议，但尚未启动训练。新接口继续保持83D context
输入和64D embedding输出，内部方向固定为16D ego-longitudinal专用子空间与48D
context/fusion子空间，保留邻车信息并加入mask-aware context dropout。损失、采样、seed、
预算、checkpoint命名、Waymo非劣性和nuPlan替换门槛均在结果出现前固定；raw BDD不是训练目标。

现有Waymo full51真实审计通过35个shard、164871窗口、24426个scenario的shape、finite、
SHA-256和scenario级防泄漏检查。但train split的front coverage只有free-flow 96649条和
sustained-following 35349条，intermittent-following为0，未达到冻结门槛。因此当前状态仍为
`FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`，不能直接训练或降低门槛。后续Stage6Q已证明
原始full51含54829个动态intermittent proxy窗口，问题来自正式builder的首帧固定front与
整窗>=0.8有效率规则；当前优先版本化修builder并重建数据，不扩大Waymo。

详细中文协议见`docs/stage6o_longitudinal_representation_training_protocol.md`，冻结产物位于
`outputs/stage6o_longitudinal_training_protocol_freeze_v1/`。

## Stage 6P/6Q：Unpaired representation reliability 与 Waymo raw interaction audit

Stage6P（Issue #257）原样复用Stage6H的800 pair、489 log和2400个release split，分别对
full64、ego13、handcrafted46与neighbor-zero64 diagnostic在n=200/250/300/400独立做A/A
calibration、A/A holdout FPR和A/B detection。禁止跨representation比较raw MMD²。

ego13在四个样本量的A/B detection均为100%，FPR为2.0%/4.0%/3.5%/1.5%；full64对应检出率
31.5%/30.0%/26.0%/63.5%，FPR为7.5%/4.5%/2.0%/4.5%。n=400同release配对比较中，
ego13-only=73、full64-only=0，McNemar exact p=2.12e-22。结论是ego13的unpaired reliability
明确优于当前full64，但这不表示neighbor/context无用。

Stage6Q（Issue #258）读取51个原始Waymo TFRecord、24872个scenario。3m动态lead proxy在
182837个raw合格窗口中发现54829个`intermittent<0.8`，2m/4m敏感性仍为53448/51109，远高于
冻结门槛5000。根因是正式builder仅在参考帧固定一次front，随后要求该track整窗有效率>=0.8，
从结构上过滤动态entry/exit。下一步是修builder、重建新版本数据并重跑Stage6O，不扩大Waymo、
不启动新checkpoint训练。中文总报告见
`docs/stage6p_stage6q_representation_unpaired_and_raw_audit.md`。

## Stage 6R/6S：动态交互数据与interaction-dominant benchmark

Stage6R（Issue #259）已确认旧Waymo builder并非只固定front，而是在每个80帧窗口开始时对
front、left_front、left_rear、right_front、right_rear五个semantic slot统一只分配一次track。
Dynamic Interaction Builder v2改为逐帧semantic assignment，显式写出slot valid mask、track-id
时间序列、identity-switch mask和derivative-valid mask；identity切换处禁止跨agent计算accel与
yaw-rate。ego有效率门槛与neighbor逐帧validity分离。新版纵向监督固定为5帧median平滑速度，
随后计算accel/jerk，再使用全体train split的q01/q99 winsorize和median/IQR normalization。
旧full51、旧33D监督和Stage6O v1均保持不变。

首次3-file pilot的自动重建曾被错误标记为人工语义通过；真实查看overview后发现交叉口附近
横穿车道会被误作`left_front`。根因是lane解析丢失Waymo neighbor relation的局部index范围。
该pilot及其full51授权已标记`SUPERSEDED`，首轮full51构建已中止。修复版保留
`self/neighbor start/end index`，并强制`lane_aware_only`、禁止几何fallback与几何相邻车道猜测；
缺少可信拓扑时slot必须为空。修复版已通过自动统计、原始TFRecord拓扑重建和独立视觉检查；
随后完成51个TFRecord、24872个scenario、168700窗口的strict full51重建。

Stage6O-v2全部数据门禁通过：train intermittent=63415（冻结门槛5000）、split重叠=0、
finite/shape/跨identity导数违规=0，五槽switch rate为1.29%–2.64%。新纵向监督同窗口RMS口径下
accel median由旧2.72降至1.48 m/s²，jerk median/q90由旧42.82/100.80降至15.51/28.47 m/s³。
因此Waymo数据侧已可准备Interaction-aware v2；这不等于已授权训练，也不等于五槽获得全量人工真值。

Stage6S（Issue #260）冻结24个same-scenario interaction pair。两个PDM planner的desired-speed、
accel/decel和lateral参数完全相同，只允许time headway与minimum gap不同。分析只看realized
trajectory、THW、front gap、closing及following acceleration response，保持embedding/BDD盲态；
若预冻结门禁失败，记录为PDM limitation，不按结果回调planner。详细设计、命令与训练授权边界见
`docs/stage6r_stage6s_dynamic_builder_and_interaction_benchmark.md`。实际结果为
`PDM_INTERACTION_BENCHMARK_LIMITATION`：平均速度差满足“小”，但front-gap未通过，只有一个预冻结
interaction指标通过。因此数据侧已准备、确认性benchmark侧仍未准备，当前未启动新checkpoint。

## Stage 6S-v2：扩大库存的interaction benchmark与独立confirmation冻结

Stage6S-v2（Issue #261）不再复用Stage6S-v1的24个场景，也不局限于Stage6J的183场景。
它从扩大的Pittsburgh DB中仅用pre-treatment信息审计15779个候选，得到301个eligible场景、
19个日志；随后冻结24个development pair并完成48条official rollout。两个planner的speed
schedule、accel/decel和lateral参数完全相同，只允许headway 0.8/2.2 s与minimum gap 0.5/2.5 m
不同。

Development中短headway减长headway的median `Δ mean speed=+0.259 m/s`、
`Δ RMS accel=+0.225 m/s²`，保持在小差异门槛内；`Δ front gap=-4.284 m`与
`Δ finite THW=-2.660 s`分别有91.7%和100% pair方向一致，满足预冻结的“至少两项interaction
mechanism通过”规则。THW严格限制为有限的`0 < THW < 20 s`，不含999/sentinel/cap。

机制通过后已冻结80-pair、15-log confirmation roster；它与development的log/token重叠均为0，
与Stage6S-v1 token重叠也为0。confirmation筛选未读取planner outcome、embedding或BDD/MMD，
尚未启动rollout、训练或新模型评估。至此数据与benchmark两侧均已具备准备Interaction-aware v2
训练的条件，但仍需单独授权才能启动。中文报告见
`docs/stage6s_v2_interaction_benchmark_confirmation_report_zh.md`。

## Stage 6T：A/B/C训练与盲测协议冻结

Stage6T（Issue #262）在第一个新checkpoint出现前冻结三个可归因candidate。A使用Dynamic v2数据、
旧single-GRU与旧objective，用于数据修复主导对照；B保持single-GRU但加入clean longitudinal
supervision、纵向sampling/ranking和mask-aware dropout；C与B使用完全相同的数据、loss、采样、
dropout、seed和预算，仅改为参数量匹配的ego16+context48双分支。三者均保持83D输入、64D输出，
不训练ego-only最终模型。没有额外A0时，old64→A不得严格写成纯数据版本因果效应。

冻结审计发现六个Dynamic v2 part的33D `interaction_feat_style.npy`各自使用局部train统计，不能直接
混合训练。Stage6T因此禁止A/B/C读取该数组，改为从`interaction_feat_style_raw.npy`用全体135046条
train rows拟合一次global mean/std，并原样应用到train/val/test；旧shard不被改写。这一规则在
任何新模型结果之前冻结。

36/36 shard SHA、168700行shape/finite、scenario防泄漏、Stage6O-v1 blocked状态、Stage6O-v2门禁和
Stage6S-v2 80-pair盲态全部通过。当前状态是
`FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING`：可实现和review统一trainer，但训练、
Waymo test、nuPlan正式盲测与confirmation rollout仍全部未授权，实际checkpoint为0/9。

C成功必须同时通过Waymo纵向提升/整体非劣性、Stage6J/K paired dose、Stage6P n=400 unpaired和
Stage6S-v2 interaction增量门禁；C不自动优于B，也不要求击败ego13。跨representation比较raw MMD²
继续禁止，C full-context相对neighbor-zero只使用各自null标准化Z差及log-cluster bootstrap。
完整中文协议见`docs/stage6t_training_evaluation_protocol_zh.md`。

## Stage 6U：Unified A/B/C Trainer实现冻结

Stage6U（Issue #263）用一套trainer按candidate配置切换A/B/C，没有复制三套训练逻辑。A/B/C均为83D
输入、64D输出；encoder参数量106560/106560/105616。B/C共用候选无关random plan，同seed下的样本顺序、
batch、sampling weights、ranking pair、dropout mask、augmentation seed、optimizer schedule和budget逐项
SHA一致，唯一主要差异是encoder topology。

Trainer只接受Dynamic v2的train/val，只读取raw33并应用Stage6T冻结global train标准化；test split、
part-local标准化数组、nuPlan/BDD/MMD和Stage6S-v2 confirmation均fail closed。Synthetic和真实Waymo
train/val subset的A/B/C forward/backward、finite loss、64D、save/load与精确resume全部通过；正式checkpoint
仍为0/9。

Implementation freeze状态为`FROZEN_READY_FOR_ABC_FORMAL_TRAINING`，但
`formal_training_authorized=false`。正式CLI必须取得一个独立、绑定最终implementation freeze SHA的授权
manifest才能运行。本机MPS初步估计A/B/C单seed最大30 epoch约1.0/1.9/3.5小时，9任务串行建议准备
22–27小时，正式首个epoch后需更新ETA。中文报告见
`docs/stage6u_unified_abc_trainer_implementation_zh.md`。

## Stage 6U正式训练与Stage 6V一次性盲测

Stage6U正式训练已按A→B→C、每个candidate按3407→3408→3409在单MPS上串行完成。9/9任务均只用
Waymo train优化、Waymo val选best checkpoint，primary seed在结果出现前固定为3407。checkpoint ledger
状态为`LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK`，9个best checkpoint及其SHA均已锁定。

Stage6V在读取任何test/nuPlan结果前创建一次性盲测授权，绑定Stage6T协议、Stage6U implementation freeze、
formal authorization、checkpoint ledger、9个best checkpoint和Stage6S-v2 roster。授权明确写入
`evaluation results cannot trigger retraining or protocol changes`；本轮没有换seed、换epoch、改loss、改架构、
改benchmark或训练返工。

Waymo Dynamic-v2 test上，A/B/C primary seed的longitudinal delta分别为-0.0232/+0.0248/+0.0159；三者均
通过following/lateral/behavior/retrieval综合非劣性，但均未达到冻结的primary longitudinal完整门禁。B-3409
虽通过全部Waymo门禁，但只能作为seed稳定性结果，不能替代预先固定的3407。

Stage6J/K的183个paired场景、四剂量盲测中，ego13以4/4 overall、12/12 task×dose、median Z_BDD=21.115
唯一通过完整门禁。learned64中A最好，为4/4、7/12、median Z=8.630；B/C均为3/4、2/12，三者都未通过
冻结paired门禁。因此B/C没有在该paired benchmark中恢复old64丢失的完整纵向敏感性。

Stage6P的800 pair、489 log、2400 split非配对发布结果则明显改善：n=400 context-balanced detection从
old64的66.5%提升到A/B/C的90.5%/100%/99.5%，FPR为3.0%/5.0%/6.5%，双方向最小检出率为
90%/100%/99%。A/B/C均通过冻结unpaired门禁；C三个seed均为99.5% detection，B三个seed均为100%，
说明release-level提升跨seed稳定。各representation独立A/A标定，未跨representation比较raw MMD²。

Stage6S-v2冻结80-pair roster实际完成61对，19对在原token两次运行中均被nuPlan官方`valid_scenes`规则排除。
根因是confirmation pre-treatment inventory未复用官方的scene-rank边界条件，而不是模型或planner失败。由于
roster已经冻结，禁止用61个成功子集事后重定义confirmation，也禁止换场景。因此trajectory mechanism未评估，
interaction embedding/BDD未读取，C full-context相对neighbor-zero的增量interaction证据为“不可判定”，不能写成
“没有增量”。

按预冻结联合规则，A/B/C均不能成为最终论文主模型。可以写入论文的正结果是：Dynamic-v2训练显著并稳定改善了
受控纵向版本差异的unpaired release检出；必须同时披露Waymo primary/paired门禁未通过以及confirmation roster
执行失败。old64继续作为冻结历史baseline，ego13继续作为纵向敏感性参考上界。完整中文报告见
`docs/stage6v_one_time_blind_evaluation_report_zh.md`，机器可审计结果位于
`outputs/stage6v_one_time_blind_evaluation_final_v1/`。

## Stage 6W-A / Stage 6S-v3：paired-unpaired机制与prospective interaction确认

Stage6W-A（Issue #266）在冻结的Stage6P 800-pair pool上，把paired与unpaired都固定为n=400，避免把
Stage6J/K的183场景与Stage6P的400场景规模差异误写成representation机制。同池结果中old64/B/C的paired
median Z分别为13.502/28.295/25.368，说明B/C并不存在固有的“paired不敏感”。历史Stage6J/K较弱主要来自
窄纵向dose/task benchmark与Stage6P广义assertive/conservative treatment、场景池和estimand不同。

B/C的release shift方向一致性为0.925/0.927，高于old64的0.815；planner signal energy fraction也由old64的
1.62%提高到3.97%/3.80%，但log heterogeneity仍占主导。context-balanced口径下B/C相对old64的标准化signal
为2.586×/2.643×，null noise为0.856×/0.927×；按log-Z增益分解，signal贡献85.9%/92.8%。因此接近100%的
unpaired检出主要由更强、更一致的planner signal驱动，null方差下降只是次要贡献。各representation只使用自身
bandwidth/null标准化，未跨representation比较raw MMD²。

Stage6S-v2永久保留为`confirmation execution failure due to roster runnability omission`。Stage6S-v3只在
pre-treatment roster冻结前新增nuPlan官方`valid_scenes`边界。v2的80个token中官方查询恰好返回61个，与实际
61成功/19失败逐场景完全一致。排除v1、v2 development和v2全部80个confirmation token后有162个候选，官方
可运行120个；新roster冻结80个、11个log，80/80 official pair全部成功。因合格库存只剩v2 confirmation使用过的
日志，v3无法做到与v2 log-disjoint，但token完全disjoint、选择不读outcome，统计使用log-cluster bootstrap。

v3机制门禁通过：median `Δ mean speed=+0.289 m/s`、`Δ RMS accel=+0.150 m/s²`，而`Δ front gap=-4.202 m`、
`Δ finite THW=-2.670 s`；front-gap、finite-THW、closing accel和following accel四项均通过。机制通过后才解锁
representation。C full-context与C neighbor-zero的Z分别为28.955/36.807；预冻结增量端点`ΔZ=-7.852`，
log-cluster bootstrap 95% CI为`[-33.393, 29.219]`，没有证明C具有增量interaction信息。

论文可以按“强unpaired release正结果 + paired/Waymo/interaction增量负结果”收口，但不能宣称C已成为验证通过的
interaction-aware主模型；Stage6V的`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`保持不变。若必须继续
追求interaction-aware主模型，当前负结果构成训练v3的科学理由，但须在训练前扩展并冻结全新、100% runnable的
confirmation；当前库存仅余40个未使用runnable候选，不足60-pair最低规模。本阶段未训练或写入任何新checkpoint。
中文总报告和审计manifest位于`outputs/stage6w_stage6s_v3_final_v1/`。

## 博士论文研究收口与写作冻结

自2026-08-14起停止模型训练与Stage6扩展，研究主线正式冻结为：

`Task-conditioned trajectory-level behavior drift evaluation for closed-loop planning policies`

论文不再以“提出新GRU模型”或“A/B/C谁胜出”为主线，而按paired attribution、unpaired release monitoring、
representation mechanism和interaction confirmation四个科学问题组织。核心正结果是n=400 context-balanced
release detection从old64的66.5%提升到A/B/C的90.5%/100%/99.5%，且Stage6W-A证明B/C提升主要由signal
增强驱动；必须同时保留Waymo primary、纯纵向paired和C context增量的负结果。

联合模型决策仍为`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。B只定位为当前最简单、最强的
release-level learned engineering candidate，不是universal/final validated representation。当前没有为支撑收窄后
论文核心claim而必须补做的实验，状态为`RESEARCH_EXPERIMENTS_CAN_BE_FROZEN_FOR_THESIS_WRITING`。

中文权威蓝图见[`docs/phd_thesis_research_closure_blueprint_zh.md`](docs/phd_thesis_research_closure_blueprint_zh.md)。

## 统一BDD Evaluation Matrix与Style Report Card

后续全部BDD报告统一使用`unified_bdd_reporting_schema_v1`，严格分离Behavior Drift Profile、BDD Statistic和
Representation Evaluation。任何结果必须显式给出Reference、Target、task、paired/unpaired、representation和
null/calibration；行为方向只能由Target−Reference semantic delta解释，禁止用BDD大小直接命名“激进/保守”。

固定报告包含13个行为维度，覆盖overall、纵向、横向和interaction。无样本或缺少冻结结果的维度保留为N/A并写明
reason code。业务结论使用表A Behavior Profile，表示能力使用表B Representation Scorecard；禁止跨representation
比较raw MMD²。中文规范见
[`docs/unified_bdd_evaluation_matrix_style_report_card_zh.md`](docs/unified_bdd_evaluation_matrix_style_report_card_zh.md)，
机器schema与历史task mapping位于`configs/unified_bdd_reporting_schema_v1.json`和
`configs/unified_bdd_stage_task_mapping_v1.csv`。冻结状态为`UNIFIED_BDD_REPORTING_SCHEMA_FROZEN`。

已完成的A/B/C训练后比较试验已按该规范重新输出为只读报告：
[`outputs/unified_bdd_posttraining_report_v1/unified_bdd_posttraining_report_zh.md`](outputs/unified_bdd_posttraining_report_v1/unified_bdd_posttraining_report_zh.md)。
其中表A报告固定13维的Reference→Target行为变化，表B独立比较old64/A/B/C/ego13的检测能力；不会把表示能力误写成行为方向，
也不会跨representation比较raw MMD²。可复跑命令见`QUICK_REFERENCE.md`。

## 固定维度BDD标准化矩阵

在原报告schema之上，项目已冻结`standardized_fixed_dimension_bdd_protocol_v1`，将已有冻结的
Stage6J/K、Stage6S-v3与Stage7资产组织为同一张`behavior × representation`BDD考试卷。每个结果明确拆开
Behavior Reference、Null Reference和old64 capability baseline；主矩阵固定输出13个维度和old64/A/B/C/ego13五列，
并保留raw MMD²、null q95、BDD/null-q95 ratio、Z_BDD、p、semantic delta/CI与evidence status。raw MMD²仍禁止跨representation排序。

Stage6J/K与Stage6S-v3继续是继承的确认性结果。为补齐Stage7 lane-change/lateral维度，工具仅在已有310对官方
assertive/conservative rollout上以锁定primary checkpoint重新导出表示；全部明确标记为
`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`，不修改Stage6V预注册结论。完整中文报告见
[`outputs/standardized_fixed_dimension_bdd_matrix_v1/standardized_fixed_dimension_bdd_evaluation_report_zh.md`](outputs/standardized_fixed_dimension_bdd_matrix_v1/standardized_fixed_dimension_bdd_evaluation_report_zh.md)，
当前状态为`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`。

## Final Standardized BDD Style Report Card

固定维度矩阵完成后，最终报告体系只对既有冻结CSV/JSON做一次排版冻结，不再导出embedding或重算BDD。控制定义升级为
`unified_bdd_reporting_schema_v2_final`与`standardized_fixed_dimension_bdd_protocol_v2_final_render_only`；v1文件和
原始输出永久保留为历史证据，不覆盖。

最终报告固定为两层：第一页`Behavior Drift / Style Report Card`只回答Target相对Behavior Reference发生了什么，
并在顶部独立声明Behavior Reference、Target、Evaluation mode、`Primary Representation = B`和Null Reference；
B只是测量行为漂移的representation，不是被评价的planner/version。第二页`Representation Qualification Matrix`
比较old64/A/B/C/ego13的固定treatment标准化敏感度、Stage6P detection/FPR及各类门禁。

原`Best capability`字段永久替换为`Highest standardized sensitivity on this treatment`（中文：
`该Treatment下最高标准化检测敏感度`）。它只描述特定已知treatment下相对各表示自身null的敏感度，不代表完整性、
通用性或全局最优。ego13的高敏感度不能解释为neighbor/context无价值；learned64的主要强正结果仍是production-style
unpaired release monitoring。

Stage6S-v3的逼近响应、front-gap/THW和纵向跟车交互三行统一带`†`，机器审计按每个representation保留同一个
`parent_bdd_result_id`并只计一次独立BDD检验。最终中文报告见
[`outputs/final_standardized_bdd_style_report_card_v1/final_standardized_bdd_style_report_card_zh.md`](outputs/final_standardized_bdd_style_report_card_v1/final_standardized_bdd_style_report_card_zh.md)，
最终状态为`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`。

## Stage7L-B Pure-Lateral Development

Stage7L-B已完成24场景×5档的official development：120/120运行成功、五档canonical `s_route(t)`逐点一致、
24/24各档均完成换道且无off-road。开发共使用26个unique token，最终roster为24 token / 24 log、6 left / 18 right；
共测试两套transition-length参数，安全版建议值为`60/58.5/57/55.5/54 m`。

安全版的RMS/peak lateral acceleration、yaw和RMS lateral jerk呈清晰有序变化，dose100相对dose0的duration中位差为
`-0.300 s`；最强dose的纵向副作用max仅为mean speed `0.001086 m/s`、RMS accel `0.017632 m/s²`、
RMS jerk `0.020811 m/s³`、route progress `0.031012 m`。

当前结论仍为`STAGE7L_B_DEVELOPMENT_NOT_READY_FOR_FREEZE`：4个场景在所有五档均发生相同责任碰撞，说明碰撞不由
Sharp剂量驱动，但当前只看初始帧的traffic-clearance规则不足。按静态规则尚余83 token / 67 log（15 left / 68 right），
必须先新增15 s pre-treatment动态走廊净空审计并重扫供给，才可人工审阅Stage7L-C。未建立confirmation roster，未读取
embedding，未计算BDD/MMD。中文报告见
[`docs/stage7l_b_pure_lateral_development_report_zh.md`](docs/stage7l_b_pure_lateral_development_report_zh.md)。

## Stage7L-B2 Dynamic Pre-treatment Traffic Clearance

Stage7L-B2已完成：在不读取任何Stage7L rollout、dose、embedding或BDD的条件下，以原始nuPlan replay tracks建立15 s、
time-aligned、dose-independent的common lane-change envelope。它用ego/agent footprint加3.0 m纵向和0.5 m横向固定buffer，
对24个development场景解释了4/4固定碰撞场景；同时没有为了保留未碰撞场景调节buffer。

扩大扫描全部1,621个Pittsburgh DB后，静态eligible 327个token，dynamic-clean为165个；排除全部历史token并与26个
Stage7L-B development log严格分离后，Pool B仍有152 token / 94 log（19 left / 133 right），且official runnability为100%。
状态升级为`STAGE7L_B2_DYNAMIC_CLEARANCE_COMPLETE`与`STAGE7L_C_PROTOCOL_FREEZE_RECOMMENDED`。这只是允许人工审阅
Stage7L-C协议，尚未建立confirmation roster或运行confirmation。详见
[`docs/stage7l_b2_dynamic_clearance_inventory_report_zh.md`](docs/stage7l_b2_dynamic_clearance_inventory_report_zh.md)。

## Stage7L-C Prospective Confirmation Freeze

Stage7L-C冻结80个Pittsburgh、pre-treatment dynamic-clean confirmation场景及其不可变maneuver manifest：15 left + 65 right，来自严格与Stage7L-B development log分离的Pool B。selection使用固定seed=620271和仅含几何/traffic的确定性分层规则；不读取任何rollout、embedding、BDD或MMD。冻结后只授权Stage7L-D进行`80×5=400`条planner-level rollout，并且仍必须先通过mechanism/safety gate才可解锁representation。

完整科学协议见[`docs/stage7l_c_prospective_confirmation_protocol_zh.md`](docs/stage7l_c_prospective_confirmation_protocol_zh.md)。

## Stage7L-C1 Protocol Consistency Amendment

Stage7L-D启动前完成了纯协议修订：`N_design=80`继续作为execution/safety/missingness人口；每个BDD contrast使用冻结80场景中全部完整dose0+doseX pair，Primary要求`N_pair(dose100)≥76`。secondary family现固定为old64/A/B/C/ego13×4 doses×2独立task views的40格减去唯一Primary格，即单一39-test Holm family；B保留完整dose curve，Primary固定标记`PRIMARY_NOT_PART_OF_SECONDARY_HOLM_FAMILY`。semantic CI仅作展示，使用log-cluster bootstrap 10,000次、seed 620272。

roster仍为80场景、15 left/65 right、79 logs，SHA256仍为`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`；没有运行Stage7L-D、rollout、embedding、BDD/MMD或训练。详见[`docs/stage7l_c1_protocol_consistency_amendment_zh.md`](docs/stage7l_c1_protocol_consistency_amendment_zh.md)。

## Stage7L-C2 Task-Population Consistency Amendment

C2将`LAT.LANE_CHANGE`统一定义为完整80场景prospective roster membership，使Primary与理论矩阵对应格共享同一个task population和cell-definition SHA；`LAT.DYNAMICS`仅按冻结Pool B的pre-treatment `official_scenario_types_json`生成mixed-proxy mask。当前重放为80/80与38/80，两mask不同。

理论矩阵仍为40格并只排除一次Primary，secondary Holm family固定39格。不可计算secondary cell不得删除，固定以`NOT_COMPUTABLE_PRE_FROZEN_TASK_POPULATION`和raw p=1进入Holm；可计算小样本cell正常计算并标记`LOW_N_SECONDARY_DIAGNOSTIC`。C2是Stage7L-D前最后一次protocol consistency amendment；该段描述的是D启动前冻结状态，后续D结果见下一节。详见[`docs/stage7l_c2_task_population_consistency_amendment_zh.md`](docs/stage7l_c2_task_population_consistency_amendment_zh.md)。

## Stage7L-D One-Time Planner-Level Confirmation

Stage7L-D已完成并冻结为`STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED`。统一runner在第一条official rollout前精确验证protocol SHA `f5a8b2df...`、roster SHA `90ec9b42...`、80场景/15 left/65 right/79 logs、development零重叠、80/80 runnable/dynamic-clear/static-eligible，并预先建立固定400格计划账本。最终400/400 official rollout成功、80/80场景五剂量完整、replacement=0；各dose均80/80成功。

本阶段只处理official planner trajectory、mechanism、longitudinal nuisance、safety/validity和canonical identity。dose100−dose0下，换道时长median Δ=−0.200160 s（88.75%同向）、RMS横向加速度Δ=+0.055832 m/s²（100%）、峰值横摆角速度Δ=+0.014404 rad/s（96.25%），三项机制均PASS；四项纵向nuisance均PASS。80场景scenario-level保守安全口径下official success/completion=100%/100%，off-road=2.5%，责任碰撞=1.25%，安全门禁PASS；canonical identity为80/80、mismatch=0。

全部planner-level gate通过，仅解锁`STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`，未自动执行Stage7L-E。全过程没有读取checkpoint/embedding或计算BDD/MMD。中文结果见[`docs/stage7l_d_one_time_planner_confirmation_report_zh.md`](docs/stage7l_d_one_time_planner_confirmation_report_zh.md)，机器化小型manifest见[`docs/stage7l_d_confirmation_manifest_v1.json`](docs/stage7l_d_confirmation_manifest_v1.json)。

首轮执行在任何有效trajectory产生前暴露出冻结maneuver manifest少4个planner接口字段的代码不可执行问题。按C2允许的pre-outcome例外，runner使用单独runtime adapter补齐既有冻结常量，不修改源manifest、roster、treatment、planner或gate；所有失败attempt原样保留并按实现代次审计。
