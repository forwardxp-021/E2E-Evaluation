# TSB-Coast P0 — Primary影响与科学防火墙 v1

状态：`CONSTRUCT_DIRECTION_SELECTED_PRIMARY_AMENDMENT_PENDING`。Owner已决定下一步研究自然语义更近的Coast构念；**未批准替换现有Primary干预、改统计参数或运行训练/评估**。

## 是否进入主线

建议把Coast作为下一步唯一受审行为构念，理由是科学问题语义一致，不是预计RBR会赢。旧正恢复TSB保留为有效历史工程/方法证据，不冒充Coast验证。新构念需完成有限P1和独立P2，并经单独Primary修约及冻结点，才可成为未来主检验干预。若不成立，论文收缩，不能悄悄回到Owner不接受的构念来宣传自然两脚制动。

## 强H、RBR及O

H仍为已批准30维强人工挑战者：ego13、冻结F的mean_abs_accel、8个固定秒级signed-a bins、3固定lag ACF、2 braking-mass时序摘要、2 validity及1 cadence。Coast仍有明确负—近零—负时序，H这些合法能力都应保留；不得删特征或改分箱起点让其看不见。需在修约审查中确认同一H30的输入/适用时域仍有效，**不是重新开H搜索或批准弱化**。F_match四项不能替代H，旧O的恢复fraction不适合新机制；新Coast O只能作为机制正控制，不能成为训练标签或Primary comparator。

Raw33历史人工表征、ego13/F0_project Secondary、F_match配对约束、H30唯一Primary手工对照、O机制正控制、RBR共享z64的角色保持。不能把语义head预测替代z64，也不因为Coast形态更贴近日常语义就断言其具备人工独立残差信息。

U-only encoder selection完全保持：Waymo U决定输入、架构/loss、超参数、seed/validation/checkpoint；禁止以旧B1、新Coast mechanism/treatment、安全/F_match、H比较或BDD结果选网络。S3-P0的输入83维、U原片段独立性和训练协议阻断仍未解除。新基准不是训练数据补充，更不授权训练。

## 数据角色与暴露

S1R Approval Record已激活U/B/V/C与E0～E5规则，即使原模板仍有PENDING字样也不能重新解释成未批准。D是development/calibration/power功能子集，必须显式绑定D_fit、D_AA_final_calibration与最终V_AA/V_AB的独立性。

旧B1机制/安全曲线已经影响Coast构念、正恢复排除、控制器风险和候选gap设计，属于**已披露的benchmark-development influence（E2）**；不能称本提案机制盲、独立于旧结果或新设计untouched。它没有产生新的H/RBR比较暴露，不凭此新增E3/E4；若未来用detector结果选H/核/样本量等，则按具体用途登记E3/E4，不能靠换名清除。P0新增事件见`evidence/Development_Exposure_Record_v1.json`；旧ledger不改写，无新SESSION身份选择。

建议P2资格SESSION与P1及旧20对分离，防止设计检验复用；它不自动变成最终V/C，旧B1也不是新20对。S1R允许Claim-B下合规B→V复用的原则依然有效，但不能据此免除最终calibration/evaluation独立性、无survivor和SESSION推断。Claim-A不能把已工程暴露SESSION说成unseen；C优先E5，UNKNOWN fail closed；相关E3/E4仍排除，不重置历史E3标记。

## 统计合同不自动改变

ΔBDD=P(alarm_RBR)−P(alarm_H)，nominal FPR=.05，m=20 independent logs/release arm，π=.50，δ*=.10。保留相同源样本与draw/报警定义/预算、各自scaler和同数学带宽规则、既定FPR有效性与SESSION cluster-aware区间、success L>0且点估计≥.10、既定FAIL/INCONCLUSIVE和no post-freeze adaptation。

新构念会改变可能的报警效应及discordance，旧B1的PASS和旧TSB开发效应不能作为Coast功效证据；需**在另行授权的D-only阶段**重检full-pipeline power/coverage与容量，不得假设RBR增益或改δ*/m/π。P0没有做power或任何MMD/BDD计算。

S3-P0冻结容量审计仍只是静态信息：Claim-B候选214 SESSION，最小unpaired规划80cal+80AA+160AB=320独立来源，保守SESSION-disjoint分配不足；paired最小40/40/80=160仅有理论数量空间，扣除新P1/P2/角色reservation、源簇与coverage后仍未认证。C低暴露静态4也不满足m20解释。新角色与适用初速可能进一步缩小容量，本轮没有生成、筛选或查看未来V/C名单和outcome。

沿用原预注册fallback：V构建前若unpaired容量/power在预算内不可支持，按既定paired规则重新标定并收缩paired attribution主张；两种都不足则Level1 closure，不减m、不增加π、不按结果扩样本。改换干预需Owner独立修约，统计设置不因本包自动更新。

## 本轮零执行证据边界

本包只读冻结报告/SR描述及源码；不重新读取未来评价表示/结果、不运行H向量、MMD、calibration、model inference或runner。所有计数为0：SIMULATION、RUNNER_RUN、NEW_TSB_ROLLOUT、HLC_ROLLOUT、RBR_TRAINING、H_BDD、RBR_BDD、PRIMARY_EVALUATION、V_EXECUTION、C_EXECUTION。Manifest绑定本轮来源和执行记录；计数是本轮调用记录，不宣称系统级审计能证明所有外部进程行为。
