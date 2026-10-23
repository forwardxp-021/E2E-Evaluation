# TSB-Coast P0 — Owner审查与下一阶段备忘录 v1

## A. Overall Verdict

`TSB_COAST_P0_KINEMATICALLY_FEASIBLE_CLOSED_LOOP_UNVERIFIED`。

自然语义目标已明确、无正恢复下等冲量/等时间重心的运动学空间存在；当前平台的真实滑行、测量误差、F_match及官方安全仍未验证。不能写“已实现/已资格PASS/Primary已替换”。

## B. 大白话

可以设计出“刹一下—速度大致平稳—再刹一下”的目标，而且不用中间主动提速来凑净降速。但nuPlan控制器会提前看约1秒，可能把滑行吃掉；是否真的做得出来只能由另行批准的小规模闭环验证回答。代价建议锁在一个C2版本、3对Development、最多8次含技术retry尝试、≤3 CPU小时和2个工作日，而不是重新开无期限平台工程。

## C–G. 设计提要

- 近似滑行拟定义为实际|a|≤.2m/s²连续≥.5s；允许小漂移，排除gap持续明显正a。两脚各有峰、持续时间和降速剂量，不能由噪声构成；不声称踏板动作。完整规格及测量不确定性见Mechanism Proposal。
- 理想C2：两次−1.305×1s，中间0×1.8s；新Baseline −1.45×1.8s，以同一制动时间重心对齐，两臂名义总降速2.61m/s。这是新版本的参考设计，不改旧参数。
- 旧F_match四项定义仍有用；旧calipers不能自动沿用，新误差预算和额外剂量/峰平衡gate都需审查。全窗匹配不消除H30的合法时序信号。
- 最小实现为新generator/parameter contract/planner/analyzer/薄执行binding；route、官方runner、recorder、安全adapter原理与原件复用，保持六步收尾。旧代码和授权不修改、不移用。

## H. Primary

统计合同α=.05、m20、π=.5、δ*=.1、ΔBDD、强H30、U-only selection和SESSION推断保持。更换被检验构念需独立修约；D-only功效/容量须重新核查但本轮不运行。旧B1启发新设计的E2 development exposure已披露；没有新detector/Primary暴露。U输入、来源独立性、训练协议阻断仍在。

## I–J. 边界和路线文件

建议P1科学候选仅C2一次冻结版本；科学FAIL停止、不替换/重跑；最多2个P1前零运行实现迭代、最多8个科学attempt（均尚未授权）。P2建议独立20SESSION/40arms、最多42attempts，仅为下一次单独审查的预算设想。v2.4原文未改，本包另列逐节修订，保留所有S1/S1R合同。

## K–L. 执行和来源

所有科学执行计数均0；只做源码/历史报告阅读、解析纯fixture及hash审计。9项解析回归通过不代表机制classifier已测试或新基准已验证。Manifest记录Owner原指令、来源/依赖、原B1工件与保护资产、测试记录和新输出hash；最终Git提交绑定Manifest自身。

## 尚需Owner决定（只列新事项）

1. 接受本P0操作性定义、测量/匹配误差预算作为**P1前冻结依据**，还是指出一个具体科学修改点？本包阈值未获批准，不能通过实施者调宽来达标。
2. 在同一份后续P1授权中，是否批准C2、最多2次零运行实现迭代、3个B-development SESSION/最多8attempts/≤3CPU小时/2工作日，以及技术retry和科学失败立即停止规则？身份、配置、实现hash和测量fixture必须在实际runner前另行绑定。

路线B选择本身已由本轮指令决定，不再请Owner重新选A/B。**P2资格授权、最终Primary干预修约、训练和V/C授权均留待后续阶段各自决策，本次不捆绑征求许可。**

## M. 只推荐一个下一阶段

**Scientific Owner审查本P0包并决定是否授权有界P1 Development。** 在该决定前停止，不开Coast实现迭代、仿真、训练或Primary。
