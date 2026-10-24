# P1R-C 最终入口报告 v1

## A. Overall Verdict

`P1R_C_READY_FOR_FIRST_PAIR_OWNER_REVIEW`。

该状态仅表示可以审阅第一次Development科学执行申请，**不是科学执行授权或C2闭环合格**。G07/G08/G10已闭合；G18仍PROPOSED_ONLY/NOT_AUTHORIZED，是执行前必须取得的新Owner决定。代码/config/data/合同实际SHA256见Execution_Source_Binding和Manifest，所有数值均在首个科学attempt之前形成。

## B. 大白话

前一轮“真实日志时间有微小抖动，不能精确切进/切出滑行”的入口问题已解决。现在新Coast执行链有统一的精确10Hz模拟时钟，原三对的六个真实生产配置全部通过零运行预检，可以准备第一次真实闭环实验的Owner授权。

原日志观察没有被改时间或插值，仍按原index取用；有些观察时标最多晚于模拟时刻0.402毫秒，也有最多早1.038毫秒的观察，已逐帧披露。这是受控离线同index回放，不是严格在线传感器时间同步。正式C2能否形成双制动、F_match与安全是否通过，仍须获批后的首对实验验证。

## C. Scientific Amendments

D1/D2正式前瞻批准见 `P1R_C_Owner_Approved_Scientific_Amendment_v1.json`：抽象零请求滑行、完整模型响应异常检查；旧近零core/漂移/正恢复保留为诊断；四项F及.02、raw导数和舍入比较不变；±.01成为独立鲁棒性诊断。旧P1阻断和原反例原封不动保留。真实生产量化/随机误差/不稳定一旦出现，仍须重新阻断。

D3仅限新Coast工程。规范时间=t0+i×100000，不改t0/C2/旧R1/B1/官方模型；合法映射要求原index身份不变、唯一最近帧单元、全部偏差及未来时标风险披露。该数学index检查不是物理安全误差门槛；未为当前三对调选毫秒阈值。

## D–E. Canonical Clock / Log Observation Mapping

新clock由正式Hydra构造，Simulation/Setup同一实例；controller接收其current/next iteration，model按该dt推进，planner/recorder核对同一物理时间。81个接口索引按官方结束语义产生80帧、79transition；coast边界为21/39，控制区间18步。

完整raw/canonical/token/index表有486行，包含六臂各81个metadata索引；index80不是额外科学状态。三对δ范围分别[-330,+323]、[-1038,+93]、[-6,+402]微秒，旧最大偏差330/1038/402保留；每对两臂的映射、precontext及route完全相同。

原官方TracksObservation和traffic接口仍按原index/token查询；默认不提取/插值未来agent waypoints。此C2 planner不以邻车/灯状态反馈控制ego，因而这些观察时差不通过该路径影响C2控制。安全/TTC等仍把同index观察与canonical ego state组合，不能证明阈值边缘的标签相对严格同步回放不变。详情及条件影响式见 `P1R_C_Log_Observation_Time_Mapping_v1.md`；新授权草案要求明确承认该披露。

## F. Controller and Model

真实官方model、LQR和TwoStage源码/参数未变。规范clock驱动的两种纯合成隔离fixture各80状态、79实际model transition，Treatment18个零请求到达model，负a保留并按2/3逐步衰减，速度/位置继续变化，3.9秒恢复LQR。转向和原跨边界拒绝测试继续保留。它们不是正式SESSION或C2闭环PASS。

## G. F_match

物理时间来自规范模拟状态，80帧/79间隔的raw速度梯度、六位舍入和四项caliper不变。纯解析fixture和epoch平移验证完全一致；旧±.01的16条件反例重新复现并保留为独立诊断，不称实际生产噪声。尚无实际C2 F结果，不能预先保证.02通过。

## H. Production Gates

逐项文件 `P1R_C_Production_Entry_Gates_v1.csv`。G01–G17在其各自“历史保护/模型fixture/装配/元数据/批准/预算/绑定”证据范围内PASS；不能将装配PASS解释为真实科学结果PASS。

G07：三对六配置新clock及原观察映射PASS。G08：D1批准PASS。G10：D2批准PASS。G18：**PROPOSED_ONLY / NOT_AUTHORIZED，尚不可执行**；该项不是通过“Owner审阅准备”必须预先激活的权限。旧clock、旧授权及单独科学修约文件在新执行入口均被拒绝。

## I. Budget and Scientific Firewall

科学attempt=0，正式runner/SESSION状态推进/metric/serializer完成均0；RBR训练/推理、H30评价、MMD/BDD、Primary/V/C均0。原3对、6首次arm、8累计attempt、10800CPU秒及每arm1200秒上限不变，无自动retry。预算审计快照继承旧0-attempt账本；未来可变工作账本独立存放，避免覆写历史审计快照。

旧B1仍20/20 PASS、40arms、41attempts；LOW_ORDER_NUISANCE_ELIMINATED与TSB_CLEAN_RESIDUAL_TASK仍NOT_ESTABLISHED。S1/S1R、H30、U-only及原Primary统计合同不变，Coast未获准替换Primary。

## J. Tests / Hashes / Git

96项测试通过：本轮25项，既有P1R 28项、adapter 21项、测量22项；598条依赖/官方弃用警告保留日志。新文件在nuplan Python3.9编译通过；全tools在系统Python3.13编译通过；no-tmp检查通过。原全仓Python3.9的旧报告工具语法兼容性问题未修改、不伪报通过。

实际运行了同一三对的真实无执行factory检查和纯合成真实官方模型测试；本地数据存在，没有以无数据跳过。最终六对象guard调用全0，history/recorder行数全0。所有3485受保护文件、原roster/结果/科学合同及指定CSV SHA256不变。工程修复/复核证据均保留，guard构建对象已丢弃，未来必须fresh构建。

Git只提交本轮新clock/factory/entry/tests/config/报告及QUICK_REFERENCE本轮章节。原Stage5、QUICK_REFERENCE和CSV工作区修改及其他历史未跟踪文件保留。交付commit为包含本报告的提交；最终反馈报告实际commit及远端一致性，不使用自引用commit hash。

## K. Recommended Next Stage

`Scientific Owner review → First Development Pair Execution Authorization`。

只申请原DEV-01 Baseline与Treatment各一次，绑定本包科学修约、source/config/data/preflight及披露条件；不启动DEV-02/03、不自动retry。新的科学执行凭证仍PROPOSED_ONLY。本轮完成后停止，不自动运行科学闭环、P2、RBR或Primary。
