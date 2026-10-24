# F1 — Scientific Owner基准路线收口备忘录 v1

## A. Overall Verdict

`F1_C2_FAILURE_ATTRIBUTED_JOINT_FEASIBILITY_NOT_ESTABLISHED`。

**唯一首选建议：路线一，关闭当前Coast基准开发。** `COAST_PROGRAM_CLOSURE=RECOMMENDED_PENDING_OWNER_DECISION`。固定C2旧失败及停止状态已被Owner接受，不待重议；本建议不能自动解释为Owner已批准关闭整个Coast方向或已批准训练/Primary。

## B. 大白话

参考里每脚1.305 m/s的降速安排，经过“每步以当前速度重建参考+约1秒预瞄”的LQR后，每脚只产生约0.6525 m/s的负请求；短请求再被model滤波摊平，所以每脚峰值和支持段剂量都不足。零请求释放成功，不能补回总制动量差，也不能代替机制资格。

## C. 请求与实际传递

模型全窗负a积分Baseline2.49158、Treatment1.30602 m/s，与请求层2.49334/1.30769接近。滤波主要改变短脉冲峰形和时间，不是整体约一半差距的主要来源。逐层来源、严格求和恒等式及未观测的归因边界见Controller Review。原测量算子没有被替换，原review没有重跑。

## D. 真实整体运动的事后差异

`POST_HOC_DESCRIPTIVE_ONLY_NOT_FMATCH_QUALIFICATION`：按已公开六位描述符相减，T−B为均速+0.827941 m/s、首尾速度变化+1.185519 m/s、路径+6.504970 m、mean_abs_accel−0.148200 m/s²；绝对差分别为原预算的8.27941、7.90346、8.13121、7.41倍。不是新联合F_match正式FAIL，旧四个差值空白字段保留。

离散实际路径差约6.50496943米：对称代数分解中，净降速差项5.83570534米、时间重心项0.66926409米（约89.7%/10.3%）。这是选择明确的恒等式会计，不是两个因素的独立干预因果百分比。只调整时序而忽略总剂量，无法解释本次主要差距。

## E. 三目标联合可行性

理想运动学中不矛盾；固定C2在当前SESSION联合未实现；未经验证的新版本闭环联合可行性仍NOT_ESTABLISHED。只补到最低单脚阈值不能保证实际等冲量、等时重心、逐峰平衡和四F。未选新参数、未搜索、未建议改model或controller以追求PASS。

## F. 对核心博士问题的影响

没有RBR/H结果，故不构成表示学习方法失败。自然滑行双制动不是总体BDD比较问题逻辑上的唯一机制；关闭Coast后仍可修复Input83、U来源独立性和U-only训练入口。旧B1保留有限历史主张，不自动替代Primary；正式干预适用范围和Primary执行仍需单独闭合。

## G. 两条路线成本、收益与风险

| 项 | 路线一：关闭Coast开发（首选） | 路线二：另立有限新机制设计（本轮不推荐） |
|---|---|---|
| 价值 | 保留零请求能力、真实失败、既有B1及治理证据；将工作移回表示学习必要条件 | 可能提高自然滑行构念贴合度，但不保证公平匹配或RBR增益 |
| 科学依据 | 现有数据已足够归因并揭示匹配缺口；缺少重开联合可行性依据 | 只有运动学非矛盾；尚无可信的新闭环联合验证依据 |
| 资源 | 新Coast仿真预算消耗0；下一步仅独立最小S3修复授权 | 新协议/命名、门控与形态设计、测量、有限开发、独立资格与数据暴露审查；当前不能可靠估时/报成功率 |
| 统计独立性 | 保留已暴露DEV01身份，不扩大样本影响 | 新版本不能把DEV01当独立资格，必须明确设计启发及额外独立性代价 |
| 主要风险 | 自然Coast基准贡献不能声称完成；Primary机制适用性不会自动解决 | 再陷入调参工程，峰值/剂量/时序/四F联合修补并诱发选择偏差 |
| 对论文进度 | 直接处理83输入/U污染/训练规格等必要阻断 | 在表示学习尚未具备入口时继续延迟核心工作 |
| 执行权限 | 本备忘录只建议关闭，不自动批准任何后续执行 | 无命名候选、无新幅值、无预算授权、无仿真；未来若Owner坚持，须独立命名/合同/严格预算与停止规则 |

不因账面还有6次全局attempt余量就推荐花掉它们。剩余上限不是执行权限，也不是科研收益的证据。没有可信联合匹配依据时不重开参数迭代。

## H. Owner唯一推荐下一阶段

**确认关闭当前Coast基准开发 → 将下一工作授权限定为S3-P0输入合同、U独立性及零训练入口的最小阻断修复。**

这是一个研究范围收口决定。已经批准的H30、统计参数、旧B1结果不重新提请选择；具体训练、Primary干预选择、V/C均保持未授权。若Owner不接受本首选建议，路线二必须另立前瞻性设计授权，不能从F1直接进入任何新attempt。

## I. 执行防火墙与停止

本轮NEW_SCIENTIFIC_SIMULATION、NEW_C2_ATTEMPTS、PARAMETER_TUNING、CONTROLLER_MODIFICATION、H30_EVALUATION、RBR_TRAINING、MMD_BDD、PRIMARY_V_C_EXECUTION全部0。只读一次F1描述性分析；测试仅静态数学fixture与现有输出一致性，不调用原科学判定。旧ledger保持stopped=true、attempts2、科学CPU15.017957642秒，无重置。

本轮7项要求文件齐全，另提供逐窗积分CSV、80帧速度差CSV、数值/测试/保护证据。原始输入/合同和未提交工作区均由前后SHA256保护；只提交F1文件、只读工具/测试及QUICK_REFERENCE本轮段。准确最终Git commit和远端核对在交付回复中报告，避免自引用commit SHA。

完成后停止，不启动Coast新版本、训练或Primary。

## 来源与版本绑定

真实实验source commit：`73e96f7e6fa1c39bca0f2cd0f9f2947646024c93`；授权记录commit：`d3e5e75375440854ebddd554747b6042774bb460`；冻结结果commit：`b038ccc4dd19cedbb6c9e345c84d74e5fcba21d2`。

数值来自本目录`evidence/F1_Numeric_Analysis_v1.json`、三张F1描述性CSV及其绑定的P1R-D raw/telemetry/四组曲线CSV。源码、合同、输入和本轮输出实际SHA256见`F1_Manifest_v1.json`，代码行证据索引见`evidence/Source_Evidence_Index_v1.json`。读取了既存判定，未调用原始arm/pair analyzer、model、tracker或runner；任何新差值均为`POST_HOC_DESCRIPTIVE_ONLY_NOT_FMATCH_QUALIFICATION`。旧结果、账本和资格字段均不修改。
