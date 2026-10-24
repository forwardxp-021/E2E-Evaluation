# P1R-C 日志观察时间映射 v1

证据范围：原三个B-development候选的只读metadata及无执行生产对象；完整每index表为 `evidence/Log_Time_Mapping_All_Indices_v1.csv`，权威JSON为 `evidence/production_verified/ThreePair_Preflight_v1.json`。所有source/config/data/contract/hash见Execution_Source_Binding和Manifest。本轮没有读取未来V/C名单或产生新科学outcome。

## 三对完整偏差摘要

定义δ=原日志时间−规范模拟时间，单位微秒。下表统计真正使用的80个index，index80额外元数据也完整保留于机器表；Baseline/Treatment逐index映射完全相同。

| pair | 最小δ | 最大δ | 最大绝对δ | δ>0帧数 | δ<0帧数 | 原日志首末跨度us | 新模拟跨度us |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV-01 | −330 | +323 | 330 | 32 | 47 | 7900286 | 7900000 |
| DEV-02 | −1038 | +93 | 1038 | 9 | 70 | 7900040 | 7900000 |
| DEV-03 | −6 | +402 | 402 | 75 | 4 | 7900179 | 7900000 |

每对index0的δ为0，解释了正负帧数合计79。最大日志时标超前于模拟时刻为402微秒（0.402ms）；最大日志陈旧时间为1038微秒（1.038ms）。原P1R报告330/1038/402微秒全部保留，81个原时间点与旧证据逐值完全一致，没有把原source timestamp舍入成规范时间。

## 观察映射的实际接口

官方 `TracksObservation.get_observation()` 直接调用同一原scenario的 `get_tracked_objects_at_iteration(current_iteration)`；update仅将current_iteration设为next_iteration.index。它不按新时间戳查询另一个frame，也不插值actor。真实factory核对observer/clock/controller/setup共享同一scenario，observer初始index为0。

NuPlanScenario按 `_lidarpc_tokens[index]` 抽取该frame对象；此调用默认 `future_trajectory_sampling=None`，不会进入可选的未来waypoint提取/插值分支。traffic light同样按原index/token查询并保留原timestamp。纯sentinel scenario上的官方TracksObservation测试证明0..79恰好同index、同对象读取，没有错位、漏帧、重排或访问index80。真实SESSION仅查询metadata时间/token，不调用其观察推进或runner。

原past history preload仍是原日志历史，未重写；当前C2 planner及LQR只用当前自车和已生成参考，未用past buffer的raw时间计算本轮F。原mission goal/route和官方progress指标的expert空间route保持既有离线语义；新clock没有新增对未来actor轨迹的读取。

## 因果性披露与控制影响

“没有选择未来index”与“没有使用时标略晚的同index观察”不是一回事。机器字段 `future_frame_selected=false` 仅表示没有换成i+1或之后的frame；`strict_event_time_causal_replay=false`明确保留后一限制。不能将同index规则包装成严格在线因果回放。

该版本C2 planner在其实际dispatch链中只用current ego、静态route/map及固定参数，忽略观察/traffic light来控制车辆；nonreactive TracksObservation也不控制ego。因此，这些细小同index观察时差不会通过邻车/灯状态反馈进入C2纵向控制。合成planner的80次完整未来参考测试无需这些对象也通过。这个结论不能迁移给会消费观察的未来学习式planner。

## callback、metric及serializer的时间关系

官方Simulation将同一权威iteration、对应model EgoState、原index观察组成SimulationHistorySample。模型状态与iteration的时间严格一致；观察metadata仍是原日志时间，区别由mapping表可追溯。MetricCallback把该history及原scenario交给同一官方MetricsEngine；serializer存同样history和原scenario；它们不另建ego clock。

所选ego acceleration/jerk/yaw、lane-change、speed-limit、direction等指标从history的ego时间取canonical时间；TTC及collision共用同一history的ego时间和同index观察，drivable-area主要使用静态地图及ego几何。progress-along-expert-route继续使用原expert空间route，未将expert日志时间伪改成模拟时间。源码文件及实际配置的metric集合全部绑定SHA。这里是装配/源码/纯时间提取器检查，**未计算正式metric或serializer结果**。

## 影响边界及不能证明的事项

- 新模拟80帧首末为7.9秒，原日志这80帧的首末跨度分别多286、40、179微秒。所有frame身份和初始t0不变；初始precontext/route hash与旧spec一致，两臂使用完全相同的映射。
- ego按固定dt运行是前瞻批准的新栈定义，不声称与旧抖动dt逐轨迹数值相同；旧B1不重算、不重新解释为Coast结果。
- F只用canonical ego state的raw速度/位置和实际canonical timestamp。它不从raw日志观察时间求ego导数，因此映射δ不是直接加入速度的±.01噪声。动力学所形成的真实F差值仍须首对测量，不能由fixture预判。
- 若邻车在这段时间满足速度上界V、加速度上界A，单个观察位置的时差影响可写为 `V|δ| + Aδ²/2`，相对几何还要计ego项。这里**没有已批准的全actor V/A上界或事件间距**，不把该式变成未经验证的厘米级全局误差保证。
- 在碰撞/车道/阈值边界附近，微小时间差也可能改变离散标签；本轮不能证明官方安全标签相对精确同步回放不变。官方规则不放宽，首对仍按同一披露的回放合同计算并报告，不能假称已有安全PASS。

本轮建立的是可审计、无错帧、同index且对两臂公平的受控离线回放映射，**不是同时刻传感器真值或真实在线系统验证**。这一限制已写入新执行授权草案，必须在Owner审查时保留。
