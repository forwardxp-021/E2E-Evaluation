# TSB-SR Primary Implications v1

状态：`READ_ONLY_CONSTRUCT_REVIEW_NO_PRIMARY_RESULT`。以S1.1已批准Scope/H/SAP和S1R批准记录为准；名称含Draft或模板PENDING不意味着可以重新设计已批准合同。

## 1. 正向恢复会不会给人工特征明显信号？

**会提供合理、显性的候选信号，但本轮没有测定H-BDD检出能力。** 20对均出现约+1.03～+1.18m/s²正峰和约1秒正加速度段，同时Baseline在相应期间仍明显降速。

| 既有表示 | 可能看见的信号 | 本轮能否据此断言报警 |
|---|---|---|
| F_match四项 | 全窗mean speed、end−start speed、path length、mean abs acceleration中的残余差 | 不能；caliper允许差值，不等于检测功效 |
| ego13/F0_project | speed分布/净变速、加速度RMS/abs/q95、jerk RMS/q95、路程 | 不能；这里只解释现有定义，不计算新的ego13或报警 |
| H30 | 上述ego13，加上8个固定秒级signed-a bins；固定lag ACF；制动质量时间质心/spread | 有直接表达正/负时序与两次减速分布的能力；仍须未来冻结公平BDD验证 |
| O | phase count、release fraction、second peak ratio | 正控制确认既定机制，不是H，也不能作为训练标签或Primary比较替代 |
| RBR64 | 将来从U-only学习的共享时序表示 | 尚未训练，当前没有与H的任何结果 |

H的时间分箱尤其可能保留Baseline连续减速与Treatment中段提速的符号差异；ACF和braking-mass摘要可能表达分离的减速质量。这是强挑战者应保留的正当能力。所谓“捷径风险”是**论文把这些显性信号误称为人工不可见新信息的解释风险**，不是H使用这些字段违规。

## 2. F_match全通过为什么仍有显著形态差异？

四项是全窗少量聚合量，且是有宽度的caliper，不是逐时刻相等或分布匹配。相近净速度变化可以由一次连续减速或“减速—回升—再减速”产生；全窗mean abs acceleration也不编码所有符号和时间位置。

所以F_match≠ego13≠H，F_match pass不约束所有a/jerk峰、分箱、ACF或braking centroid，也不排除四项本身仍有系统残余。已有S1开发审计曾报告四项有同向残余，这是已暴露DEV证据，不能伪装成这20对的新统计；本轮未重算F_match、更没有修改原caliper。

## 3. 当前TSB是否仍适合作为受控检测基准？

**可以，前提是Owner接受收紧后的construct与主张。** 它能提供来源清楚、原资格满足、两arm共同context的固定时序变化，适合回答“U-only学习表示在此冻结干预下，能否在同误报/同预算下比强H提高报警概率”。基准不必满足H完全看不见，才能有比较价值。

需要承认三个限制：干预是已知且开发过的；正向恢复是显性物理信号；闭环变化不是把所有低阶因素严格保持相等后的纯时间顺序置换。当前证据也不能证明H弱或强到何种报警率。若未来H已接近检出上限，增益空间可能小，但不能据本轮曲线认定Primary注定失败、改δ*或削弱H。

S3-P0已识别的输入与U独立性阻断继续有效。语义审查认可Route A不关闭训练Gate，更不授权训练或Primary。

## 4. 未来结果各自能支持什么？

| 未来合规结果 | 可支持 | 不可支持 |
|---|---|---|
| RBR满足原Primary success | 冻结z64-BDD在这一controlled brake–positive-recovery–brake、适用域/采样/统计合同下，相对固定强H具有检测效用增益 | 一般未知行为发现；独立于所有人工信息；自然coast双制动；真实驾驶员意图/安全退化；胜出因果来自某一特定latent结构 |
| 有效分析显示无预注册有用增益/FAIL | 该运行点和固定H下，未建立所要求的RBR增益；保留Level1、强人工基线及阴性证据 | RBR没有学到行为；学习表示普遍无效；任意其他行为/数据下都不能增益 |
| FPR、功效、源簇支持或区间不足 | 按原协议INCONCLUSIVE，说明证据限制 | 把不确定强行归为科学FAIL或SUCCESS，换指标/seed/样本救结果 |

不新增实验不能把相对增益归因于“正恢复”“两段间隔”或“人工遗漏信息”中的某一因果成分。即使结果阳性，当前能证明的仍是**特定受控时序机制的检测效用**，不是一般未知行为发现能力。

## 5. 保持原Primary合同

ΔBDD=P(alarm_RBR)−P(alarm_H)，α=.05，m=20独立logs/arm，π=.50，δ*=.10；原success为有效资格/FPR下L>0且点估计≥.10，原FAIL/INCONCLUSIVE规则保持。H仍是固定30维强对照，ego13是Secondary，O是机制控制，z64直接作为RBR输入。不得用语义Head预测替换z64。

U-only encoder选择；D只用于已批准development/calibration/power；E3/E4相关污染排除；SESSION聚类、共同draw与公平scaler/kernel预算、无survivor和freeze后不adapt均不变。B1的20对不是最终V，也不提供完整独立calibration/AA/AB容量。未查看未来V/C的表示或比较结果。

## 6. 暴露登记和禁止路径

本轮读取的是已冻结20个B1 SESSION的benchmark轨迹，属于既有E2语义/机制审查；没有新scientific rollout、H特征开发或Primary比较结果。本轮没有更改H/BDD方法、训练目标、V成员或停止规则。Manifest记录所读trace与决策用途；不重写旧exposure ledger。

若Owner选择未来Route B，其设计将受到本次B1观察启发，必须登记这种development influence；不能因为换名、换session或重跑，就声称设计从未接触行为结果。新资格验证和相应最终评估身份独立性必须前瞻设计，不能依照谁让H难检出来挑参数或基准。
