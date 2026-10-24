# P1R-D 首个固定开发对最终报告 v1

## A. Overall Verdict

`P1R_D_FIRST_PAIR_SCIENTIFIC_FAIL`。两臂真实执行完成；失败位于Treatment arm机制，不是基础设施。source commit 73e96f7e6fa1c39bca0f2cd0f9f2947646024c93；独立Owner授权记录commit d3e5e75375440854ebddd554747b6042774bb460。只执行DEV01固定SESSION/token的一次Baseline和一次Treatment。

## B. 大白话

这次在实际nuPlan闭环里看到了“先减速—停用纵向跟踪并向模型发0请求—继续运动—恢复跟踪后再次减速”。零请求适配器起效，残余减速自然按官方模型衰减，没有强行把加速度或速度置0。但两次减速都偏弱，未达到事前冻结的峰值/剂量要求，因此还没有一个通过资格验证的TSB-Coast双制动基准。不能把实现现象写成科学PASS。

## C. Baseline

单合格段1.8–3.6秒，支持1.8秒，峰值1.365850222 m/s²、剂量2.050995315 m/s。arm PASS；全程官方LQR。详见Baseline_Arm_Result。

## D. Treatment

候选1.3–1.7和4.1–4.5秒，各支持0.4秒；峰值0.756895400/0.756575901 m/s²、剂量0.289452099/0.289305139 m/s，分别低于0.8/0.35。合格段0，TWO_BRAKE_PHASES_NOT_REALIZED。

[2.1,3.9)秒18个最终请求均严格0且实际到达模型；每步继续推进。模型a从−0.329421634衰减至−0.000222900 m/s²，a_next=(2/3)a_before最大残差2.78e−17。速度13.590260703→13.524420956 m/s。3.9秒恢复纵向跟踪，请求−1.168298181 m/s²。旧nearzero/drift/positive-recovery只作诊断，qualified gap为null，固定窗口诊断完整另列，不替代门槛。当前能力仍为ABSTRACT_COAST_PROXY_ONLY。

## E. 真实曲线

四组均输出中文PNG/SVG/原始绘图CSV：Actual_Acceleration_Curves、Reference_Command_Realized、Realized_Speed_Comparison、Coast_Window_Detail（完整文件名前缀P1R_D、后缀v1）。第一组分开显示模型body-x实际a与冻结资格代理；第二组分开显示planner当前区间参考、最终请求、下一状态对应的模型a、raw速度梯度。阴影为参考窗，不是合格段。第四组只显示1.8–4.2秒和真实控制模式恢复。

模型输出归属next_time，命令归属current interval，最后7.9秒没有新命令；绘图阶梯只延伸至区间末端，CSV最后状态命令留空。所有曲线来源真实current-ego/controller/planner记录，未使用理想轨迹补全。Plot_Source_Provenance绑定输入与工具SHA256，四组PNG均已视觉检查。

## F. F_match

Treatment arm失败，按Owner第十节停止于arm gate；联合F_match及剂量/峰值平衡均NOT_EVALUATED_ARM_GATE_FAILED。CSV列出四项实际单臂描述符及原caliper，配对差值留空。未执行pair analyzer或把单臂描述偷换为配对资格。原raw梯度/六位舍入/caliper未改。

人工±0.01 m/s压力反例继续保留在P1/P1R原证据中；它是独立鲁棒性诊断，非已经证明的生产误差界。本轮不重跑压力网格、不改导数或放宽0.02 m/s²。

## G. Official Safety

两臂各16项正式parquet完整，正式adapter从no_ego_at_fault_collisions和drivable_area读取；at-fault碰撞计数各0、drivable合规各true，arm safety均PASS。这不等同所有类型碰撞0。最低速度Baseline11.685947555、Treatment12.871499758 m/s；无低速端停，无reference floor。

## H. Clock Mapping

100000us规范时钟；两臂实际80帧同index raw观察审计PASS，偏差−330～+323us，完整逐帧token与时间保留。DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY；晚到raw观察和安全阈值敏感性限制保留。该控制器不以动态观察作纵向反馈，不把结果解释成在线因果验证。

## I. Budget

new attempts=2，cumulative Coast attempts=2，scientific CPU=15.017957642秒。Baseline CPU/wall=7.551957642/8.849630416秒；Treatment=7.466/8.753956083秒。历史Coast scientific attempts0、实施迭代2保留；未重置账本。operational stopped=true，no automatic retry。全局8次上限还余6次、10800秒上限还余10784.982042358秒，均不构成执行授权；本轮两个合法attempt已用尽，后续合法新attempt为0。

## J. Scientific Integrity

没有替换样本/参数/阈值/测量算法。DEV02/03、retry、P2、RBR训练/推理、H/BDD/MMD、Primary/V/C均0。未接触Primary结果；本次仅B-development benchmark exposure，新增1 SESSION/2 arms，非detector-development比较或Primary-comparison exposure；局部exposure记录单列，不改旧全局冻结账本。

旧B1仍20/20 pairs PASS、40/40 arms complete、41 attempts、1 technical retry、0 scientific failures。LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED；TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED。S1R/H30/U-only/SAP/U-B-V-C/E0–E5不变。原v2.4文件保持未提交快照，不覆盖，ROADMAP_V2_4_REPOSITORY_SYNC_PENDING；旧P1状态不追溯改写。

## K. Tests / Provenance / Protected Assets

独立最终双臂零运行预检PASS后才激活授权并先行提交；source/config/data/runtime全绑定；正式生命周期六项均完成，两臂逐项manifest artifact hashes通过。实际serialized history各80帧观察audit PASS。只读图导出校验80状态/79区间覆盖、model状态时间归属、trace/model速度一致及冻结资格代理与既存review逐值一致。Python全tools编译、科学Python3.9新增工具编译、no_tmp检查PASS。详细证据见evidence/Verification_Result_v1.json和Manifest。

受保护3625文件、冻结source/config1068文件、10数据来源hash均保持；原受保护CSV SHA256仍e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8。既有Stage5、QUICK_REFERENCE用户增量、原CSV和历史未跟踪产物保留。

## L. Git State

生产源码和授权记录commit在本报告开头明确区分。最终结果提交仅新增本轮文档、只读结果工具及QUICK_REFERENCE本轮段落，推送后以远端branch SHA等于本地HEAD验收；准确最终commit/push状态在交付回复中报告，避免本报告自引用commit哈希。未跟踪raw产物在本地保留且由Manifest逐文件SHA256绑定，不以Git已提交文档冒称raw已上传GitHub。

## M. Recommended Next Stage

`Scientific Owner review → C2 scientific feasibility and dissertation scope decision`。

唯一建议：Owner审阅这对真实曲线及失败，决定是否值得另行授权一个最小、前瞻性的C2科学可行性修订任务。当前停止；不自动提出/试跑新幅值，不创建候选或SESSION，不扩大平台工程，不修改Primary、不训练模型。
