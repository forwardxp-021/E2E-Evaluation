# S3-P0 Primary Firewall Check v1

状态：`SCIENTIFIC_BOUNDARY_PRESERVED_FUTURE_EXECUTION_NOT_READY`。本轮没有V/C数据构建、筛选、推理、解盲、calibration或Primary。下文容量只对已冻结元数据作算术审计，不产生新科学样本身份。

## 合同优先级

S1.1的Scope、H和SAP已批准科学设计；标题含Draft不意味着可重新选择。S1R Owner Approval激活amendment，即使各模板仍写PENDING，也不能据此恢复旧的绝对fresh身份禁令。S2R早期preflight的工程0认证已由后续engineering closure supersede；B1按R2最终报告正式闭合。工作区v2.4是未跟踪的整合草案，部分状态过时；不覆盖任何正式合同。本轮记录`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，不把该行政事项当重新设计Primary的理由。

## 角色与用途

| 角色/用途 | 允许 | 禁止与独立性 |
|---|---|---|
| U | 架构、loss、normalizer、checkpoint、seed规则及训练验证 | 不能由nuPlan B/V/C或detector比较选encoder；当前Scenario-only分组不满足新训练资格 |
| B | TSB机制/安全/匹配/平台；B1为此角色 | B1不等于RBR效果；不是自动V成员；不得将B1科学表现转成网络选择信号 |
| D（功能名） | 现行UBVC下development/calibration/power用途的显式子集，不是额外永久角色 | D_fit、D_AA_final_calibration、最终V_AA、V_AB按批准合同分离；允许的历史B→V Claim B复用不能破坏同一最终校准/评估的独立性 |
| V | PRIMARY_FREEZE_POINT之后的冻结受控比较 | tested component对应E3/E4排除，UNKNOWN fail closed，完整分母/无survivor；U/D/内部子集源簇关系必须可检查 |
| C | 预先绑定低暴露稳健性，优先E5 | 不把E1/E2改名untouched；不得用C选checkpoint或救Primary；低容量如实不确定 |

原S1 Q/D/E中“所有历史B/Q都不可触碰”的绝对身份规则受S1R覆盖；测量资格、最终calibration与evaluation隔离、SESSION依赖、no outcome adaptation保留。本轮没有替Owner放宽内部必要独立性；未来若同SESSION跨功能子集，必须先证明符合批准的独立性/聚类合同，不能靠不同log ID自动认证。

## exposure最新元数据叠加

基于S2R248会话census与最新`B1_R2_Exposure_Ledger_Update_v1.csv`，后者20行均保留，18/19/20三会话从S2R E1快照追加E2；不是改写原ledger。18先前失败attempt也保留，R2的prior字段以S2快照为参照，不意味着失败暴露被擦除。

| 类别 | S2R快照 | 叠加最新B1后 |
|---|---:|---:|
| E0 | 1 | 1 |
| E1 | 149 | 146 |
| E2 | 62 | 65 |
| E3 | 31 | 31 |
| E4 | 0 | 0 |
| E5 | 5 | 5 |

B1是TSB benchmark E2，未发生当前RBR-H detector development或Primary-comparison outcome暴露。已有31 E3为方法开发暴露（包括影响现行paired/MMD边界的历史方法），对相应Primary仍排除；E4=0为当前记录状态，不是未来永久保证。nuPlan结果不参与本提案的训练目标选择；本轮U lineage/物理fixture读取属于数据合同审计。

静态合格214中E0=1/E1=144/E2=65/E5=4；Claim B静态上限仍214，Claim A unseen-session静态上限由152降至149（E2不合适）；strict C仍4。没有把任何一行标成最终V/C成员。来源hash及叠加结果见`evidence/B1_And_Exposure_Inheritance_v1.json`。

## Primary原样继承

`ΔBDD=P(alarm_RBR)−P(alarm_H)`；nominal FPR=.05；m=20独立logs/arm；π=.50（target10 treatment+10 baseline）；δ*=.10。它是synthetic moderate-drift运行点，不代表真实车队比例。有效资格+双方FPR gate后，success为95%区间L>0且点估计Δ≥.10；之后fail判U<.10，否则INCONCLUSIVE；不改成L>.10。

两边各自D_fit baseline mean/SD和同一数学median-distance带宽规则；相同样本、共享draw/target mask/source weights、alarm iff statistic严格大于原threshold，常数列scale1，不删H列；每边1 scaler+1 bandwidth，零监督核/投影搜索。

最终D_AA calibration 1999 draws，rank1900；V_AA与V_AB各10000 draws，按SAP预定数值SE条件最多40000（只增加Monte Carlo精度，不增source n）；2000外层cluster resamples传播calibration不确定性，继承SAP固定seed。SESSION作为最终source cluster，所有arms/feature/failure共同移动；不能把同SESSION副本或重复draw当独立样本，也不能只用简单binomial/McNemar宣称科学区间。FPR双方同时单侧95%上界≤.075（每边.025 tail）；区间/coverage不足则不确定，不重调threshold。尚无本轮coverage验证或power结果。

## 容量及paired fallback核查

B1只有20独立SESSION的20对，不足以构造一轮reference20+不同target20的unpaired draw，更不提供独立的calibration、AA evaluation、AB evaluation及C。不能用40 arms冒充40独立source logs。

冻结SAP的最小unpaired规划envelope：80 calibration +80 AA-eval +160 AB-eval=320 distinct source logs（还不含U/D_fit/工程reservation）。现有静态合格214是SESSION数，并未绑定最终内部角色、独立log计数或功效；在保守SESSION-disjoint内部划分下214<320，**不能认证该envelope**。1,564 canonical logs来自248 SESSION，不能据此宣称1,564个独立源；若最终以更细log分配并保留SESSION依赖，需要批准合同内的独立支持/coverage证据，不可在此静默放宽。C4也不能当m20 Primary来源或已充分power的confirmatory study。

paired fallback已预注册，不重新请Owner在看结果后选设计：在V身份构建前，若census或D-only power/coverage在批准预算内不能支持unpaired，执行原fallback决策；仍m20 PAIRS、π.50、同Δ和success规则，但用pair-swap null重新标定，主张仅受控paired attribution，不是operational baseline FPR。最小paired规划40/40/80=160独立pair logs有理论数量空间，但扣除角色reservation与源簇依赖、再验证power/coverage之前**不算已满足**。两种都不满足则Level1 closure，不缩m/增π/选赢的seed或扩V。

本轮没有选design、分配最终N、运行D功效或生成V/C名单。必要的未来动作是修复U→独立授权训练→按原D-only规则做capacity/power/coverage与身份绑定→PRIMARY_FREEZE_POINT→另行V/C授权；不能跳步。

## 冻结点缺项

未来checkpoint、input schema/normalizer、H代码/BDD配置、numerical calibration、角色身份hash、最终N/预算、依赖环境和coverage实现尚未完整绑定。现阶段只确认防火墙规则一致，不声称未来Primary执行已ready。主指标、kernel、样本量、成功规则与H不准事后改变；不以本次发现的输入问题重新挑选Primary。
