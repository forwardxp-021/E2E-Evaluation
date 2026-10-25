# TSB-DirectCoast P2-Q1 — Final Qualification Report v1

日期：2026-10-25（Asia/Shanghai）。角色：`B_BENCHMARK_QUALIFICATION`。

## A. Overall Verdict

**`P2_QUALIFICATION_FAILED`**。原计划分母永久保持20个独立SESSION。前6对完整通过；第7对Baseline出现官方安全科学失败后立即全局停止。不能报告20/20或以成功子集代替完整名单。

## B. 大白话结论

这次没有全部通过。直接制动和中间零纵向请求已经在实际执行的样本上形成了预定行为，前6对还同时满足匹配与安全。但第7个SESSION的Baseline虽然制动合格，官方指标记录1次责任碰撞且可行驶区域不合规。因此按事前规则必须判整个P2资格失败，不能继续跑剩余样本寻找成功率，也不能删掉这对。

这说明当前固定干预在这份完整名单上的联合资格没有成立；它不证明所有DirectCoast机制都不可行，也不是RBR行为表示或检测能力的失败。本轮没有分析或修改失败的具体因果机制。

## C. 固定分母、完成与停止

| 项目 | 实际数量 |
|---|---:|
| 原冻结SESSION / pair | 20 |
| 已启动pair | 7 |
| 双arm完整、正式完成pair判定 | 6 |
| 成功pair | 6 |
| 科学失败pair | 1 |
| 基础设施阻断pair | 0 |
| 完全未运行pair | 13 |
| 已启动arm | 13 |
| 技术完整arm | 13 |
| 未运行arm | 27 |

DIRECTCOAST-P2-07-TREATMENT和P2-08至P2-20共27arm均NOT_RUN_AFTER_STOP，原身份与顺序保留。第7对只运行Baseline，不能算一对完整执行。所有科学数值只对应实际运行；原20pair/40arm矩阵和80行四F状态表保留完整分母。未运行的CPU为实际消耗0，科学指标留空，不补曲线或统计量。禁止将6/6已完成通过或6/7已启动通过解释为20 SESSION总体通过率估计。

## D. 双制动机制

13个已运行arm的机制均通过，包含7个Baseline各一段及6个Treatment各两段。以下是已存冻结analyzer结果的描述汇总，没有重算资格。

| 实际量 | 7个Baseline | 6个Treatment共12脚 |
|---|---:|---:|
| 合格支持段 | 2.2～4.0秒 | 1.3～2.2秒；4.1～5.0秒 |
| 每段支持时长 | 1.8秒 | 0.9秒 |
| 峰值减速度范围（m/s²） | 1.448773486～1.448773833 | 1.276556750～1.276751414 |
| 门槛内剂量范围（m/s） | 2.352338195～2.352398187 | 1.005415463～1.006225339 |

Treatment有效分离1.9秒。全部满足事前峰值≥0.8、支持≥0.4秒、剂量≥0.35及分离≥0.7秒。13arm最低实际速度为2.817546448m/s；冻结analyzer无低速端停、reference floor或机制原因。合格支持段剂量不等于全窗负加速度剂量，也不等于输入请求积分。

## E. 零请求与官方模型

6个已执行Treatment全部18/18中段请求精确为0，共108个零请求。实际同一请求包到达原官方KinematicBicycleModel，所有状态继续推进；全体实际transition的冻结模型滤波方程残差为0，无正向请求、纵向LQR或停车控制介入，横向LQR持续工作。

中段首个更新的模型实际加速度约−0.854934～−0.854830m/s²，末次约−0.000867723～−0.000867618m/s²：这是原模型滤波尾巴，不是请求非零，也不要求实际加速度在释放瞬间跳到0。未运行的Treatment没有模型响应证据。

能力仍为`ABSTRACT_COAST_PROXY_ONLY`，不宣称现实零踏板或零轮端扭矩。13arm共1040个真实current-ego状态、1027个controller transition和1040个planner记录；各arm完整官方metrics、serializer、simulation log、finalizer及观察审计保留。

## F. 正式四F与额外平衡

仅前6对双臂技术、机制、安全和观察完整后，调用一次原analyze_completed_pair作正式判定。全部通过。表中为冻结六位结果的绝对差分布；全精度描述符另存Unrounded CSV，仅作透明性导出，不改变正式判定。

| 指标 | 6对最小差 | 最大差 | 原容差 | 最小剩余余量 |
|---|---:|---:|---:|---:|
| mean_speed（m/s） | 0 | 0 | 0.10 | 0.10 |
| end_minus_start_speed（m/s） | 0.000001 | 0.000002 | 0.15 | 0.149998 |
| path_length（m） | 0 | 0.000001 | 0.80 | 0.799999 |
| mean_abs_accel（m/s²） | 0 | 0.000439 | 0.02 | 0.019561 |

额外全窗负剂量差最大0.002107693113m/s（限0.15，最小余量0.147892306887）；Treatment每脚与Baseline峰值差最大0.172216736247m/s²（限0.20，最小余量0.027783263753）。

第7对Baseline安全失败，Treatment未启动，正式F_match及额外pair平衡均为`NOT_EVALUATED_ARM_GATE_FAILED`，未调用pair analyzer，也没有将单臂描述符冒充配对FAIL。其余13对NOT_RUN。

## G. 官方安全与时钟限制

12/13已运行arm的官方安全通过。失败arm的官方parquet原值为at-fault collisions=1、drivable_area_compliance=false。没有缺失指标填0，也没有关闭安全门槛。全部13arm真实80帧serialized history与本SESSION原始同index tracked objects、traffic lights、lidar token逐帧一致，无插值、换帧或未来prediction对象。

实际已运行7 SESSION的raw-minus-sim偏差范围−1014～+580微秒；Q0完整20名单的−1014～+1443微秒仍是预检元数据范围，不将未运行部分当成科学结果。每个SESSION使用自己冻结的完整映射。

`DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY`：安全、碰撞及TTC仅是该离线回放及官方模型下的结论，不是严格事件时间在线因果或真实道路安全结论；阈值附近敏感性限制保持。

## H. 首个科学失败

- Pair / arm：`DIRECTCOAST-P2-07-BASELINE`。
- SESSION：`2021.09.27.17.52.47_veh-44`。
- LOG：`2021.09.27.17.52.47_veh-44_01131_01267`。
- Scenario：`da7a9065299e5cc2`。
- 技术完整：PASS；实际观察映射：PASS；一段制动机制：PASS。
- 首个不合格资格项：官方安全门，责任碰撞计数1且drivable-area=false。两项在同次资格检查不满足，本轮没有推断二者的时间或因果先后。
- Baseline制动峰值1.448773832840839m/s²、支持1.8秒、剂量2.3523981865633368m/s。
- 子进程正常完成证据收尾，以科学失败返回码20退出；全局账本stopped=true。无技术失败、自动重试、参数修改、样本替换或失败后科学执行。

失败官方parquet的精确副本、science review、safety adapter、completion、manifest及actual observation audit见evidence；完整原始记录仍在独立科学输出目录。指标说明“安全不合格”，本轮没有进一步推断碰撞为何发生或哪个控制环节应被修改。

## I. CPU、attempt与账本

13个首次attempt，13个独立子进程，全新runner/planner/tracker/model/callback；CPU累计133.310757秒（含实际子进程及可计量后代），计量确定。最大单arm墙钟12.808866666秒；未触碰57600秒CPU或1200秒wall上限。CPU不含本轮前期文档/零运行准备，科学子进程内来源哈希检查计入消耗。

预算天花板42与本轮40首次arm授权严格区分；技术retry授权0、实际retry0。持久账本位于outputs/stageR/tsb_direct_coast_p2_execution_state_v1/budget.json，完成副本见P2_Q1_Execution_Ledger_v1.json；不得删除、重建或重置。剩余预算不是继续执行许可。

## J. Exposure与Primary防火墙

按实际启动追加13个E2事件，覆盖7个SESSION（6个原E1转为E2工程暴露、1个原E2追加事件），保留原等级历史。未运行13SESSION不记科学暴露；第7SESSION仅Baseline暴露，不伪造Treatment结果。未访问E5保护样本，不把工程暴露重标E5。

本轮是B benchmark资格，不是未见SESSION Claim-A、端到端策略版本效果或Primary功效证明。没有RBR训练/推理、H30结果比较、BDD/MMD、Primary/V/C、参数搜索或F3。U-only模型选择、H30、S1R与Primary合同不变；不得用本次结果选择encoder/loss/checkpoint/seed或筛选有利未来V/C。

旧B1 20/20 PASS、LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED、TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED、旧C2 FAIL、F1及F2 Development PASS全部保持。

## K. 测试、来源与历史保护

10项零运行测试PASS，含200组真实旧凭据拒绝、并发重复claim、中断、CPU不确定、输出碰撞、失败停止与固定20分母；system/scientific Python编译及check_no_tmp_dependencies通过。调度源码8535de5caf76b525d715425b213395b218082ae4、独立授权f18c38709e88ab698a40b8aa5c10b86bbd3c501c均在首次attempt之前提交；1075个F2源码/配置、46个来源、原名单/每SESSION映射及授权链逐hash绑定。F2科学源码零修改，原PROPOSED_ONLY凭据永久不变。

原始科学namespace为outputs/stageR/tsb_direct_coast_p2_qualification_v1，448个原始文件保留本地并逐项SHA256索引；可审查的实际13arm报告/安全/完整性证据、6个正式pair review和失败原始parquet精确副本提交到本轮docs/evidence。阶段性manifest引用执行前版本的关系可重建，不以最终manifest哈希回填历史审计记录。最终完整性与保护复核见Final Verification与Manifest。已有用户改动保持。

## L. 唯一下一阶段建议

**Scientific Owner仅审查第7对Baseline的首次官方安全科学失败，并据此作后续研究路线决定。** 当前P2资格失败永久保留；本建议不授权重试、继续名单、修改参数/控制器、安全重判、F3、RBR训练或Primary。等待Owner新指令，本轮停止。
