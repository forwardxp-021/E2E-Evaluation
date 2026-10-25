# TSB_DIRECT_COAST_F2 — 简明科学结论 v1

日期2026-10-25；分支20260825_stageR_new。

## 结论

**TSB_DIRECT_COAST_F2_DEVELOPMENT_PASS**。

在固定已暴露DEV01 SESSION、同一新直接请求适配器与官方车辆模型下，首次闭环同时实现了：Baseline一次合格制动、Treatment两次合格制动、中间严格零纵向请求、原四项F_match及保留额外balance合格、官方安全合格。两个arm各一次，没有修改参数或重试。

这证明该固定模型/控制版本/SESSION下三个要求能够兼容，推翻不了旧C2的真实FAIL，也不是多SESSION可靠性、独立P2资格、真实轮端零扭矩或RBR优于H的证据。Scientific Owner本轮已明确不接受F1关闭全部Coast的建议；F1既有归因与历史建议原件保留，本轮新证据单独记录。

## 实际制动机制

资格算子维持median3真实速度→physical-time gradient(edge2)，支持a≤−.6、峰≥.8、支持≥.4秒、剂量≥.35，domain[1.1,5.5]；分离≥.7秒。没有以参考请求或模型body-x状态替换资格代理。

| arm/阶段 | 实际有效支持区间 | 支持秒 | 减速度峰值 m/s² | 段内剂量 m/s |
|---|---|---:|---:|---:|
| Baseline | 2.2–4.0 | 1.8 | 1.448773569 | 2.352354268 |
| Treatment第一 | 1.3–2.2 | .9 | 1.276679412 | 1.005925782 |
| Treatment第二 | 4.1–5.0 | .9 | 1.276730438 | 1.006138070 |

Treatment有效段间隔1.9秒；两臂arm_pass=true、reasons为空。原nearzero、漂移、正恢复诊断字段保存在原review，不重新作为失效的旧资格门槛。

## 零请求、模型尾巴与横向控制

Treatment[2.1,3.9)18个最终纵向请求全部精确0，同一packet确实到达官方model，状态每步推进。模型a(2.1)=−1.282343529，按a_next=(2/3)a_before衰减到a(3.9)=−0.000867684 m/s²；最大残差约1.39e−17。速度13.128274815→12.871979646 m/s，中段仍自然下降0.256295169 m/s，不能把“零请求”写成“实际加速度立刻0”。3.9秒恢复第二段固定−1.305请求。

两臂全程最终请求仅0/冻结负值，无正请求；纵向LQR、停车及base.track_trajectory实际调用0，横向LQR各79调用。官方model、实际速度/加速度状态均未改写，模型含初态和滤波记忆。起始实际a略正来自原始ego初态，随后在0请求下衰减，不是补偿驱动。原始起始speed模长14.151693647 m/s，body vx14.151094524 m/s亦未被强行对齐。

命令冲量两臂−2.61 m/s；实际净速度变化约−2.58489 m/s，受共同初始加速度、滤波端点和speed模长/body-vx口径影响，不能声称真实净变化必须等于命令积分。

## 正式四项F_match及额外平衡

Baseline完整技术/科学/观察审计PASS后才启动Treatment；两臂完整PASS后调用已冻结pair analyzer。本表是正式F2 Development配对结果，不是F1的事后描述性表。

| 描述符 | Baseline | Treatment | 冻结六位绝对差 | 原caliper | 结果 |
|---|---:|---:|---:|---:|---|
| mean_speed m/s | 12.626162 | 12.626162 | .000000 | .10 | PASS |
| end_minus_start_speed m/s | −2.584888 | −2.584886 | .000002 | .15 | PASS |
| path_length m | 99.852555 | 99.852554 | .000001 | .80 | PASS |
| mean_abs_accel m/s² | .330027 | .329935 | .000092 | .02 | PASS |

F使用raw速度、真实规范时间、原gradient edge2/六位舍入/epsilon1e−12；未改caliper。额外全窗负a剂量差.000439539 m/s≤.15；两个T峰与B峰差.172094157/.172043131 m/s²≤.20，全部PASS。匹配通过不等于所有低阶nuisance消除，更不代表H30无法发现合法时序信号。

## 官方安全、生命周期及实际观察

两臂各16项正式nuPlan parquet存在，六步生命周期全部完成，官方serializer/metric finalizer完整；所有artifact hash验证通过。正式adapter读取at-fault collision计数各0、drivable_area=true；这是冻结安全合同PASS，不冒称所有类型碰撞均0。最低速度B11.566806136、T11.566807447 m/s，无低速端停或reference floor。

两臂各80个current-ego/79transition/80planner记录、实际serialized history观察审计PASS，actor标识/时间/位置航向和traffic-light同index一致，无额外future predictions。原raw时间/token未插值/改帧；偏差−330～+323us。保留DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY及安全边界标签可能敏感的限制，不宣称量产在线因果评价。

## 曲线及原始证据

- F2_Reference_Final_Request_Model_v1：两panel分别显示当前planner参考、最终直接请求和模型实际加速度，PNG/SVG/CSV。
- F2_Actual_Speed_Comparison_v1：两臂实际速度，PNG/SVG/CSV。

两图已视觉检查。阴影表示Treatment请求阶段：绿色[2.1,3.9)是Treatment零请求窗，而Baseline在同一时间制动。命令归属区间起点，model输出归属next_time；末帧无新命令、CSV留空。图只使用实际闭环数据，不用隔离/参考数据冒充实际轨迹。

原始完整产物保留在outputs/stageR/tsb_direct_coast_f2_dev01_v1/，含实际controller/planner/current-ego、official metrics、simulation log、stdout、manifest及pair review；逐文件SHA256见F2_Manifest_v1.json。实际观察审计内的execution_manifest_sha256指审计开始前版本；最终manifest随后只增加audit artifact条目以避免循环哈希。两个审计前版本已从唯一差异精确重建、与记录SHA一致，保存在evidence/F2_*_PreObservation_Execution_Manifest_v1.json，原始最终文件未改。本地raw保留且hash绑定，不冒称raw已上传GitHub。

## 预算、版本与科学防火墙

F2新独立账本2次首次attempt，CPU15.477921秒；Baseline wall8.924807458秒、CPU7.809161秒；Treatment wall8.156705083秒、CPU7.668760秒。均远低于每arm1200wall秒及累计3600CPU秒；stopped=true、automatic_retry=false。本轮授权两次已用完，不因CPU余量获得额外执行权限。旧C2 ledger仍stopped=true、2attempt/15.017957642CPU秒，未复用/重置。

真实执行源码commit afe038a10c6d1a40e1919351f3846b69d69489a8；独立授权记录commit ce86b5d（完整SHA见Manifest）；源码/配置/身份/资格/依赖/数据库地图的绑定在执行前完成。执行时resolved config与预检仅允许输出namespace及run-start timestamp不同，fresh独立OS进程、planner/tracker/model/recorder/callback均重新创建。

5项接口测试、固定初态158次隔离官方model预检、零运行双臂生产预检、授权白名单/错误run ID拒绝、全tools编译及no_tmp检查通过。隔离model更新与真实2attempt分别记账；不得说全轮仿真/model更新为0。旧C2、B1 20/20 PASS、S1/S1R、Primary统计与H30/U-only规则只读保护，前后保护证据见evidence/F2_Final_Verification_v1.json。

DEV02/03、P2、RBR训练/推理、H30/RBR/BDD/MMD比较、Primary/V/C均0。最终结果提交仅本轮文件；用户已有未提交Stage5/QUICK_REFERENCE/CSV及历史outputs保留。最终commit与push远端SHA核对在交付回复报告，Manifest自身由包含提交绑定，避免自引用。

## 停止点

**TSB_DIRECT_COAST_F2_DEVELOPMENT_PASS → Scientific Owner决定是否值得另行开展独立资格验证。**

本轮到此停止，不自动做P2、DEV02/03、新F3、调参或训练；本次DEV01为已暴露工程Development，不作为未来独立资格样本。
