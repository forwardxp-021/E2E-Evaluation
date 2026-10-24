# F1 — 博士论文价值、历史主张与独立主线 v1

## 已建立的支撑与尚未建立的主张

【已验证的方法学支撑】显式零纵向请求模式、官方model连续推进、实际命令identity记录、滤波残余核验、规范模拟时钟/原观察映射、正式安全/serializer及失败后停止的可审计链。它们支撑构念与实现分层验证；不是“新车辆动力学”“真实零踏板控制”或学习表示优于人工特征的创新证据。单个接口实现本身不足以构成博士论文全部创新。

【C2失败范围】失败针对一个固定受控干预在该SESSION下的资格和实际匹配，不是RBR-64表示学习方法的反证。本次没有训练/推理RBR，也没有H30/BDD结果，不能推断encoder学不到、RBR必输或必胜，更不能据此改架构、loss、seed、checkpoint或挑语义目标。

【旧B1有限主张】20/20 pairs PASS、40完整arms、41attempts/1技术retry/0科学failure永久有效；它证明冻结“制动—正向加速度恢复—再制动”满足其原机制、四F和安全合同，可作为历史受控验证与方法学证据。它不证明自然松刹滑行、不证明低阶nuisance全部消除、不证明纯残差任务、未知行为发现或RBR>H。

Owner在P0已明确不接受旧正恢复机制冒充自然两脚制动，且Primary干预修约未自动批准。F1不能沿用更早TSB-SR路线A建议覆盖后续Owner决定。关闭Coast也不自动把旧B1恢复成已获批Primary。

## 自然滑行双制动是否必须是唯一Primary？

从既定总体科学问题看，它不是逻辑上的唯一机制：问题围绕固定FPR/预算下学习表示相对强人工H的效用。自然滑行只是一个具体行为构念，做得更自然也不自动增加表示学习创新性。

但目前不能跳过干预适用性：**关闭Coast不会使Primary就绪**。未来采用何种受控机制、可支持什么主张，仍需Owner在任何相关比较前明确科学范围与相应前瞻性合同；本轮不选择替代机制，不创建新SESSION或V/C名单，不修订Primary参数。若Owner认为自然滑行不可替代，则Primary机制选择继续未闭合，也不能把这个需求变成无限开发授权。

## 若停止Coast，仍可推进和保留的研究贡献

1. 可追溯的行为表示/公平BDD问题定义、强H30挑战者和允许阴性结果的可证伪设计。
2. 输入物理语义与跨数据源合同审查、来源依赖/重叠审查及可复现的版本化数据治理。
3. 既有受控闭环平台的有限证据、资格与安全分层、零请求代理能力边界、负面结果与停止规则。
4. 未来在数据/协议入口闭合且独立授权后，U-only共享z64与语义/通用时序表示学习的研究；实际优越性和外推贡献仍待合规实验，不能预先宣称已完成。

无需为保留创新性把失败隐藏，或证明所有人工特征看不出差异。H30不能削弱，四F不是完整H。

## 独立工作如何继续：继承S3-P0已有阻断，不在F1实施

按已冻结`S3_P0_Training_Entry_Report_v1.md`，当前仍有Input83、U独立性和训练协议三类阻断，6个mandatory Gate BLOCKED。该状态不是此次C2失败产生的：

- Input83：time/slot轴、closing语义、时间/角度/anchor、缺失与mask/normalizer一致性；先审查canonical合同，再独立授权版本化适配和黄金样例，不覆盖旧特征。
- U来源/独立性：既有审计确认跨train/test重复ego轨迹；Scenario互斥不等于raw segment互斥。恢复来源依赖、去重/分组及新split证据，旧数据不自动重划分；RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED继续保留。
- U-only训练入口：完善可执行数据白名单、train-only scaler、encoder选择、梯度连通/导出接口与零训练验收；具体loss、超参和预算仍需Owner冻结。禁止把Coast/B1/H/Primary表现变成模型选择依据。

这些是下一阶段的最小授权范围建议，不在F1读取新U数据、修改pipeline、执行训练或选择checkpoint。全部Gate满足后才可申请独立训练授权。

## 统计与数据防火墙

Primary ΔBDD、nominal FPR0.05、m20、π0.50、δ*0.10及正式SAP推断/成功规则不重开；H30、U-only选择、U/B/V/C、E0–E5、E3/E4排除、SESSION推断和freeze后不适配保持。DEV01已有B-development暴露，本轮只再分析同一已公开结果，不新增独立SESSION或Primary-comparison暴露。开发样本不能被重新包装成新方案独立资格证据。

v2.4原件是未提交、B1快照过时的整合草案，保持`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，不覆盖。未来整合仅需准确更新B1正式闭合、C2失败/零请求实现、Coast路线决定与S3阻断；不能借文字整合替换已批准S1R/Primary合同或生成新授权。

## 来源与版本绑定

真实实验source commit：`73e96f7e6fa1c39bca0f2cd0f9f2947646024c93`；授权记录commit：`d3e5e75375440854ebddd554747b6042774bb460`；冻结结果commit：`b038ccc4dd19cedbb6c9e345c84d74e5fcba21d2`。

数值来自本目录`evidence/F1_Numeric_Analysis_v1.json`、三张F1描述性CSV及其绑定的P1R-D raw/telemetry/四组曲线CSV。源码、合同、输入和本轮输出实际SHA256见`F1_Manifest_v1.json`，代码行证据索引见`evidence/Source_Evidence_Index_v1.json`。读取了既存判定，未调用原始arm/pair analyzer、model、tracker或runner；任何新差值均为`POST_HOC_DESCRIPTIVE_ONLY_NOT_FMATCH_QUALIFICATION`。旧结果、账本和资格字段均不修改。
