# S3-P0 — RBR-V2 Training Entry & Data Contract Audit v1

日期2026-10-23；repository `forwardxp-021/E2E-Evaluation`；branch `20260825_stageR_new`；只读基准HEAD `9e781d596371781d00c326c57dc686b79b9596fe`。

## A. Overall Verdict

主状态：`S3_P0_BLOCKED_INPUT_CONTRACT`。

同时保留：`S3_P0_BLOCKED_U_DATA_INDEPENDENCE`、`S3_P0_BLOCKED_TRAINING_PROTOCOL`。

`S3_P0_READY_FOR_RBR_TRAINING_OWNER_REVIEW` **未达到**。12项强制Gate中6 PASS、6 BLOCKED。PASS项表示对应审计/建议材料满足该项，不代表新的训练或科学执行授权。

`RBR_TRAINING=NOT_AUTHORIZED; PRIMARY_EVALUATION=NOT_AUTHORIZED; V_EXECUTION=NOT_AUTHORIZED; C_EXECUTION=NOT_AUTHORIZED`。

## B. 大白话说明论文进度

B1已经把“可控地制造并测量TSB行为变化”的20对试题正式建好，现转入“第二代模型到底学什么、用什么数据、怎样公平考试”的准备阶段。B1成功不说明RBR已经训练，更不说明RBR胜H。

现在还不能训练：输入有真实的轴顺序/物理含义问题，U数据存在跨train/test重复ego轨迹，未来U版本、mask/标准化及训练选择规则也没有完整实现绑定。先修试题输入和训练/验证边界；再由Owner批准U-only训练；最后走独立calibration/power/coverage、Primary freeze及V/C授权。不能把B1漂亮的机制结果变成调网络的依据。

## C. Waymo / nuPlan Input Audit

已找到static Waymo、Dynamic v2 Waymo、nuPlan三个实际入口、历史old64配置与dataset/encoder。当前原则83=8+5×15、80帧/10Hz并没有被生产实现完整兑现。

- 组装将[5,T,15]直接reshape(T,75)，混淆time/slot；独立marker oracle有5970/6000格不符，现存36分片共72行实物抽样印证。
- 两源共用closing公式对同速恒间距返回8m/s逼近、TTC2.5s，是物理错误而非跨源数值差。
- 旧Waymo邻车yaw是absolute，nuPlan/Dynamic为relative；旧static与逐帧slot/identity政策不同。
- 时间固定.1/median-dt不一致、epoch秒float32量化、center/rear-axle anchor、完整速度/heading重建、missing/mask与normalizer合同不一致。
- 83中没有jerk；Raw33另有lateral acceleration漏dt、duration/count proxy离散化问题，不能直接当物理监督真值。

14类物理fixture，三条实际processing API共42次调用，另有独立物理oracle；小坐标float32容差atol2e-4、rtol2e-5。fixture通过运行不意味着科学合同PASS。详见Input83 Contract、83行Parity CSV和Golden JSON；没有修改原生成器、数组或旧模型。

## D. U Data Independence

现存Dynamic36分片168,700窗口、24,548 retained Scenario；train135,046 / val16,870 / test16,784，Scenario跨split=0。原始51 TFRecord全部hash/metadata索引24,872 Scenario，全部retained可追溯；manifest processed数与retained数已区分。旧static35分片meta均缺失，保留其原历史声明，不能新认证。

已证实一个train/test跨Scenario/cross-file的80帧ego world轨迹重复（80帧全valid、移动约19.9864m）；完整context不同。详细来源与hash见U Lineage报告与raw duplicate evidence。未证明全库只有一个问题，也未证明全部原始片段互斥：`RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED`。未来必须source dependency group去重/分组后重冻结新U，本轮未执行。

## E. RBR-V2 Training Architecture

已批准原则：GRU/temporal encoder→shared z64→训练时Semantic+Generic Temporal Heads，梯度均回encoder，Primary直接用z64。具体建议为GRU128、canonical13物理回归与masked temporal reconstruction、λrepr=λsem=1、AdamW lr.001/batch128、U_val固定loss选最早最优checkpoint、patience5/max30epoch；Primary seed3407，3408/3409只作重复性，禁止择优。

proposal包含mask、train-only scaler、random streams、分组、early-stop、预算、失败/resume与复现规则；全部具体项`PROPOSED_NOT_OWNER_FROZEN`。不继承旧人工距离loss，不使用TSB/O/H/BDD/Primary信息选模型；没有残差相减独立性主张。可执行新trainer/config/U版本仍缺，保持训练协议阻断。

## F. 13 / 33 / H / RBR Roles

Raw33为第一代人工特征；ego13/F0_project为常规摘要及Secondary；F_match为四项匹配描述；H为已批准30维强Primary挑战者（ego13+generic temporal features）；O为机制阳性控制；RBR64为共享学习表示。新建议canonical13监督与历史ego13的mask/时间公式不同，单独命名，不回改H。不会因H能检测TSB而删特征。语义评价标签重用只能说明已知语义保持，不能自证新信息。

## G. Scientific Firewall

依据S1.1、已批准S1R及后续S2R/B1 R2。B1最新ledger20行保留；3个原E1会话追加E2，248会话快照rollforward为E0=1/E1=146/E2=65/E3=31/E4=0/E5=5；Claim B static上限214不变、Claim A static149、strict C4。B1是benchmark exposure，不是当前detector development或Primary比较暴露。

U-only encoder selection、E3/E4 exclusion、SESSION inference、完整分母、no survivor及freeze后禁止adaptation保留。D仅用于批准的development/calibration/power；最终D_fit/calibration/AA-eval/AB-eval内部源身份隔离仍须绑定。m20/π.50/α.05/δ*.10、ΔBDD及成功/FPR规则不重开。

B1的20对不足unpaired reference20+distinct target20；214 static SESSION不自动满足最小unpaired planning envelope320及独立内部角色。paired fallback160规划也还需reservation/power/coverage，不能本轮擅自选择。详见Primary Firewall。无最终V/C roster或任何未来保留数据结果读取。

## H. Training Entry Gates

| Gate | 状态 | 主要依据 |
|---|---|---|
| G1 B1资格 | PASS | B1 R2正式20/20报告及继承hash |
| G2 U来源划分 | BLOCKED | raw record可追溯但segment/global time缺失；旧35meta不可用 |
| G3 overlap/leakage | BLOCKED | train/test重复ego已确认，未完成补救与部分重叠排除 |
| G4 Input83 | BLOCKED | 轴顺序、物理公式/时间/anchor/mask差异；proposal未实施 |
| G5 可冻结结构目标 | PASS | 完整单候选proposal，尚未Owner批准 |
| G6 U-only executable selection | BLOCKED | 无新U、完整trainer/config及可测试选择防火墙 |
| G7 无TSB/Primary监督 | PASS | U物理canonical13与generic reconstruction建议 |
| G8 z64-BDD接口 | BLOCKED | canonical/mask/normalizer严格版本export未实现 |
| G9 H/Primary角色 | PASS | H30、SAP、S1R原合同保持；未来执行未认证 |
| G10 资源预算复现计划 | PASS | CPU环境可查，3seed/90epoch/24设备小时为待批准上限，墙钟未测 |
| G11 source/config/data绑定 | BLOCKED | 本轮审计hash完整，但未来新数据/管线/normalizer不存在 |
| G12 Owner事项 | PASS | O1–O4明确，不重开已批准Primary |

逐项证据与代码入口见Training Entry Gates CSV，全部mandatory=true；没有以N/A规避必要Gate。

## I. Owner Decisions Required

只含新增未冻结项：O1 canonical83/适配科学变更；O2 新U来源依赖分组/去重与历史范围处置；O3 具体网络、语义/通用loss和U-only选择规格；O4 资源/固定seed重复性预算及最终独立训练授权。已批准Primary/H/S1R不重新提请选择。详见Owner Decision Items。

## J. Execution Restrictions

本轮NEW_NUPLAN_SIMULATION、NEW_TSB_ROLLOUT、HLC_ROLLOUT、RBR_TRAINING、RBR_CHECKPOINT_SELECTION、NEW_RBR_H_COMPARISON、NEW_BDD_EVALUATION、PRIMARY_EVALUATION、V_EXECUTION、C_EXECUTION全部0。只调用特征processing API处理确定性fixture、读取U metadata/数组和原始记录；未构造神经模型、未forward/backward/optimizer、未加载checkpoint权重、未创建科学runner。

B1 frozen结论直接继承：20/20 pairs PASS；40/40 arms COMPLETE；cumulative attempts41、technical retry1、scientific failures0。未重新仿真、重评或选择前20对。`LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED`；`TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED`。不据此声称RBR>H。

## K. Tests / Provenance / Protected Assets

新增一个只读audit工具、一个fixture测试文件；所有science生产源码未改。4项回归覆盖独立轴顺序oracle、14-case实际生产接口、wire reader截断拒绝与fixture确定性；Python编译、禁止临时依赖检查、真实U全元数据/精确hash/原始重复候选回查。完整实际命令和结果见Validation Record，不能把“测试PASS”混成“训练Gate PASS”。

S3 Manifest绑定报告、fixture、data lineage、raw索引和源码/config/runtime SHA；U manifest绑定全部已读当前数组与51原始文件SHA。历史ledger的重新核对仅认证字节相同，不把已读数组的数据合同标为科学正确。基线快照保护2185文件，含B1 roster/manifest/outcome、S1R批准及S1 SAP；保护CSV SHA始终为`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

本地v2.4完整读取并hash；文件为`RBR-64 博士研究总体方案 v2.4.md`，未跟踪、整合草案且B1快照过时；记录`ROADMAP_V2_4_REPOSITORY_SYNC_PENDING`，未替用户擅自提交或回写它。实际最终合同以S1R批准记录和B1 R2等更具体正式文件为准。

## L. Git State

起点HEAD=9e781d596371781d00c326c57dc686b79b9596fe。原有未提交Stage5文档、QUICK_REFERENCE既有内容、保护CSV及历史untracked outputs保留（基线listing有165,987条，含当时刚新增的audit入口；listing不是文件内容完整性证明）；本轮只提交S3目录、新audit/test和QUICK_REFERENCE新增S3段。没有reset/clean/add-all。最终包由实际Git commit/tree绑定，commit/push及远端核对结果在Work完成反馈给出；manifest不自包含其SHA，避免循环hash。

## M. Recommended Next Stage

仅进行最小阻断修复：Owner审查canonical83、U依赖分组方案与具体训练规格→按独立修复范围生成新版本适配/新U分组manifest/零训练trainer契约验收→重新核对G1–G12。原TSB/B1不扩展、不新仿真。入口全部满足后另行申请U-only训练授权；本轮到此停止。
