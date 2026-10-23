# S3-P0 Semantic Target Definitions v1

状态：`U_ONLY_PHYSICAL_TARGET_PROPOSAL_NOT_FROZEN`。选择依据是驾驶运动物理意义与既有ego13摘要先例；不取决于TSB两阶段形态、B1任何表现或RBR/H输赢。Human Semantic Head训练时通过z64给encoder梯度，不是post-hoc probe；其预测不作为Primary输入。

## 候选13项与计算合同

canonical center-world轨迹，统一网格/valid mask按Input83建议合同。令v=hypot(world vx,vy)，θ为wrapped世界heading；a_i=(v_i−v_{i−1})/(t_i−t_{i−1})，ω_i=wrap(θ_i−θ_{i−1})/dt，仅相邻有效且同一ego时成立；j_i=(a_i−a_{i−1})/dt要求三个连续有效原点。无效导数不纳入平均（不将reset0当真值）。无平滑、无事件对齐。各目标至少2个有效支持点，不足target mask=false，全部保留row状态；训练集某目标从无支持则阻断协议。

| # | target | 单位 | 精确定义 |
|---|---|---|---|
| 1 | mean_speed | m/s | 有效帧v算术均值 |
| 2 | std_speed | m/s | 有效帧v population SD，ddof0 |
| 3 | q95_speed | m/s | 有效v NumPy linear q=.95 |
| 4 | end_minus_start_speed | m/s | 仅首末网格帧均valid时v79−v0，否则invalid |
| 5 | rms_accel | m/s² | 有效a平方均值开根 |
| 6 | mean_abs_accel | m/s² | 有效a绝对值均值 |
| 7 | q95_abs_accel | m/s² | 有效abs(a) linear q=.95 |
| 8 | rms_jerk | m/s³ | 有效j平方均值开根 |
| 9 | q95_abs_jerk | m/s³ | 有效abs(j) linear q=.95 |
| 10 | rms_yaw_rate | rad/s | 有效ω平方均值开根 |
| 11 | mean_abs_yaw_rate | rad/s | 有效abs(ω)均值 |
| 12 | total_abs_heading_change | rad | 相邻有效角增量绝对值总和；任一网格ego无效则整项invalid，避免把不完整路程当完整总量 |
| 13 | path_length | m | 相邻中心位置欧式增量总和；任一ego无效则整项invalid |

这些13项是**canonical13语义监督建议**，不是重定义或替换已冻结ego13/F0_project。传统ego13会压缩valid帧后固定dt=.1、prepend0，原实现保持不动；本建议physical-time/mask规则不同，需独立target schema版本，不得共享名称却悄悄换公式。所有标准化仅U_train拟合；各目标同权SmoothL1，具体见Architecture Proposal。

## 与Raw33及未来通用时序结构的关系

Raw33包含纵向comfort、following、横向动力学、侧隙及behavior proxy。上表a/jerk/yaw与Raw33的明确运动学部分有概念交集；speed/path/heading与ego13有先例。它不是Raw33前13列：Raw33前13还包含THW、gap和closing，列序不同。

本轮建议不将Raw33的lane-change left/right整数拆分、transition duration、遗漏dt的lateral acceleration、错误closing衍生的following/yield/assertiveness proxy直接作为物理真值；这些保留为历史定义。未来若增加邻车语义任务，需先修复距离/closing、mask、ID和anchor，单独注册新target，不在这次把有问题的标签全集强塞Head。

固定低阶摘要不是完整时序；先后顺序、局部运动变化、交互时序与更细粒度形态由通用masked reconstruction head从U轨迹学习，**不是**使用TSB的phase count/release/second peak监督。是否学到对BDD有用结构只能由后续冻结评估回答。

## 自我验证与标签重用

对训练过的同一语义摘要做检索/probe，只能说明语义保持，不能当新信息发现的独立证据；Raw33若同时定义训练target和评估相似度，会形成已知语义重用，论文必须明示。Primary未来是独立角色数据上的报警概率差，仍不能因物理摘要被监督就预设RBR增益。H的30维结构固定、包含ego13及通用时间结构，不因和监督重合而删弱。

不使用TSB labels/treatment、O、B1 outcomes、H comparison、BDD detection、Primary results；不使用ego13成对距离或人工几何对齐loss。选择seed/checkpoint只依U_val预注册总loss，不依nuPlan任何结果。具体target、valid支持与mask政策需Owner冻结和零训练工具实现后才能训练。
