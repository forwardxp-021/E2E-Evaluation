# S3-P0 Waymo / nuPlan Input83 Contract v1

状态：`AUDIT_COMPLETE_CONTRACT_BLOCKED`。以下 canonical 规格为 **PROPOSED_NOT_APPROVED_NOT_IMPLEMENTED**，不改变第一代数组、模型、H或冻结科学合同。83行逐字段现状/建议/数值误差/证据见同目录 CSV；索引从0开始。

## 权威入口与历史实际合同

- Waymo 第一代：`tools/build_waymo_5neighbor_context_dataset.py::main/process_one_scenario`，TFRecord Scenario→vehicle tracks `[x,y,vx,vy,heading,valid]`；固定窗口80、stride20、dt=.1；原始时间戳没有传入特征函数。按Scenario ID先决定split；清洗/过滤再产生窗口；没有source-group去重。
- Waymo Dynamic v2：`tools/build_waymo_dynamic_interaction_dataset_v2.py::process_one_scenario`，逐帧语义槽位、身份切换导数置0并侧存derivative mask；生产模式lane_aware_only。它沿用同一83维组装函数，不能因为叫v2就推定问题已修。
- nuPlan：`tools/stage7c1_run_nuplan_simulation.py::_ego_state_to_row/build_simulated_seq`→`tools/stage7d_extract_neighbors_from_nuplan.py::parse_neighbor_world_tracks`→`tools/build_nuplan_5neighbor_context_dataset.py::build_nuplan_ego_features_8d/build_row_context`→`tools/stage7e_embed_stage6_dataset.py::embed_context`。R1的1秒precontext adapter用于机制/身份，不是80帧神经输入生成器。
- 历史已训练old64：`.../context_gru_stage5d_balanced_v2/training_config.json`与`training_summary.json`；83→GRU128→MLP128→64，seed42，batch64，20epoch，lr=.001。实际summary记录CPU。`tools/train_context_behavior_embedding.py`直接把raw float32 context送GRU；未拟合输入83的normalizer；不把Dataset返回的context_mask送encoder。33维监督目标按源码/merge报告记录为独立train-only标准化（旧35分片缺失，未重算其fit统计），**不是83维输入标准化**。历史辅助回归与人工特征距离loss均存在；这些距离目标不能直接继承为当前RBR-V2目标。
- 旧manifest的35个分片meta本机均缺失；有配置/summary/checkpoint文件可hash，未加载权重或推理。不能证明旧数组全部与当前源码逐字节一致。现存Dynamic36分片每片前2行共72行的83数组与slot-major reshape精确相符，独立正确布局至少部分不同；完整源文件hash已绑定。

## 已证实的差异及阻断

1. **时间/槽位混排**：输入邻车是`[5,80,15]`，当前`neighbor.reshape(80,75)`没有transpose。输出第t帧、宣称槽位s、通道k实际来自`neighbor[floor((5t+s)/80),(5t+s)%80,k]`。75列均不能称为当时五个语义邻车；只有k在15通道内部保持。独立marker oracle有5970/6000格不符。两个错误实现的数值相等不能算物理合同通过。
2. **closing共用物理缺陷**：实际`v_ego_window_x - relative_vx`，相对速度本来已减ego。8m/s同速、20m恒定间距返回closing=8m/s、TTC=2.5s；故不能当物理逼近速度。THW使用欧式距离和窗口固定轴速度，也只是特定proxy。
3. **yaw-rate语义**：旧Waymo取邻车绝对航向导数；Dynamic v2与nuPlan取neighbor−ego相对航向导数。共同转弯.04rad/s案例旧/nuPlan误差约.04rad/s，不能靠浮点容差放行。
4. **时间差分**：Waymo固定.1；nuPlan取有效时间戳差的median，但仍对相邻数组帧用同一dt。非均匀采样的加速度差约.06m/s²。原始51文件的Scenario都有91个时间点但端点并不完全相同；固定10Hz未校验。nuPlan先把seq转float32，epoch级秒时间可全部量化成一个值（fixture80→1个时间值），继而静默fallback .1。`_time_seconds`缺时间时还能返回index当秒。
5. **坐标/速度**：Waymo是box center、独立world vx/vy；nuPlan优先rear axle位置，ego vx/vy由speed和yaw重建，丢失侧滑分量；.8m/s侧向速度fixture误差.8。不能把rear axle与center位置自动视为同一点。
6. **缺失/边界/槽位**：旧Waymo一窗固定一个actor，Dynamic/nuPlan逐帧换槽；导数在不同身份和空隙处reset。Waymo允许至20%无效、自车用插值后值且未清零；nuPlan自车无效帧清零，不做相同修复。Waymo sanitize只按nonfinite插值，valid=false但有限值仍可参与；航向直接在wrapped角上插值可能跨±π错误。旧Waymo邻车可对修复值差分后才隐藏无效帧；Dynamic/nuPlan间断reset。reset=0不代表物理导数为0，mask必须使用。
7. **motion discretization**：83内部没有jerk或真实车道动作类别，jerk是后续二次差分；Raw33的`rms_lateral_accel`漏除dt（.5m/s²fixture约.04969而应约.49687），`lane_change_duration_mean_proxy`是transition indicator均值×dt，不是时长；左右count是整数拆分，不是方向识别；|local y|>2m二值阈值在弯道也会触发。历史保留，不拿这些proxy当V2真实标签。

## canonical input83 v1 建议（等待Owner冻结）

统一记录source schema/hash、车辆**几何中心**world pose与完整world velocity，SI单位，右手坐标x前/y左，heading弧度CCW，包裹[-π,π)。nuPlan需版本化适配从官方center/center_velocity提取，缺失fail closed；不再靠speed/yaw猜全速度。ego位置与速度以第一有效中心pose定义固定窗口轴；邻车rel位置/rel速度随每帧ego航向旋转。R(h)在本合同定义为world→local矩阵[[cos h,sin h],[-sin h,cos h]]，即旋转−h；两种frame刻意区分。旧Waymo选首个valid且speed>1e-3的reference，nuPlan选首个valid pose，静止起步fixture另证实原点不同；canonical统一首个valid center。

80点采样t=0,.1,…,7.9秒（8秒窗口标签对应80 samples，首末跨度7.9秒）。原始时间float64或整数microseconds，先减起点再转float32；无重复/逆序。未来适配统一到目标网格：只在原始相邻有效点内插值，线性位置/速度与unwrap后heading，原始跨度≤.15秒且两端有效才标valid；不外推，不能跨大缺口。`.15`是本轮提出的测量规则，未获批准，须独立误差fixture审核；不得把它当B1/H的新cadence cutoff。

一阶后向差分使用实际相邻有效物理dt；首点及无效/身份切换后的第一点导数填0并derivative_mask=false；同一actor相邻有效才计算。radial closing的distance分母epsilon=1e-6m，TTC closing阈值1e-3m/s、THW forward speed阈值1e-3m/s；不满足条件用无效mask，cap均999s。jerk只作为监督目标由有效连续a再次差分，无额外平滑；无mask的0禁止解释为静止。neighbor yaw_rate明确取**绝对world heading**导数；relative yaw rate如有需要属于另一个命名字段，不静默替换。本建议保留83总维度、5槽位、15通道及delta_x/y冗余，组装必须`transpose(1,0,2).reshape(T,75)`。

邻车每帧调用现有lane-aware semantic assignment，5槽固定front/left_front/left_rear/right_front/right_rear，复用源码阈值与排序；输入map/adjacency provenance必须绑定，缺图槽无效，不允许训练lane_aware_only而推理静默退到geometric。身份连续性以source actor token维护，缺token不得用对象类型冒充身份。具体跨地图几何一致性尚需新adapter的转弯/交叉口fixtures，当前直线三车道fixture不证明所有地图场景一致。

Mask侧存：ego_valid[T]、neighbor_present[T,5]、channel_valid[T,83]、derivative_valid[T,6]、actor_id[T,5]。无效数值填0，TTC/THW非定义采用999 sentinel并对应channel mask=false。连续值只用U_train的有效值拟合population mean/std（std≤1e-6用1）；角度除π、valid位原样；先normalize再把无效填0，不能让999进入统计。相同U统计用于train/val/test/nuPlan，禁止nuPlan refit。D_fit的z64/H scaler是另一个后续BDD步骤。

本轮无新adapter、无重导出数据；所以以上proposal存在并不关闭输入阻断。历史可比性只保留为原合同下结果；需之后独立授权才可用新版本检查对历史估计的影响，不回写旧分数、不用新结果选V2。

## 83维逐字段 canonical 定义

CSV另列当前Waymo/nuPlan公式、实际错误存储映射和每字段fixture最大误差。这里为建议合同，非已实现声明。

| index | field | SI unit | frame | canonical proposal |
|---|---|---|---|---|
| 0 | ego_x | m | window-fixed local | R(h0)(p_center-p0) |
| 1 | ego_y | m | window-fixed local | same as ego x, y component |
| 2 | ego_vx | m/s | window-fixed local | R(h0)world_v_x |
| 3 | ego_vy | m/s | window-fixed local | R(h0)world_v_y |
| 4 | ego_heading | rad | relative to first valid heading | wrap(world_heading-h0) |
| 5 | ego_speed | m/s | world speed norm | hypot(vx,vy) |
| 6 | ego_accel | m/s^2 | scalar speed derivative | backward diff(speed)/actual dt, derivative-valid mask |
| 7 | ego_yaw_rate | rad/s | world heading derivative | wrap(backward diff(world_heading))/actual dt, derivative-valid mask |
| 8 | front_valid | 1 | presence | 1 valid assigned neighbor; explicit channel mask |
| 9 | front_rel_x | m | instantaneous ego local | same formula with both vehicle centers |
| 10 | front_rel_y | m | instantaneous ego local | same, y component |
| 11 | front_rel_vx | m/s | instantaneous ego local | same formula from full world vectors |
| 12 | front_rel_vy | m/s | instantaneous ego local | same formula from full world vectors |
| 13 | front_distance | m | Euclidean separation | same |
| 14 | front_delta_x | m | instantaneous ego local | duplicate rel_x retained and disclosed |
| 15 | front_delta_y | m | instantaneous ego local | duplicate rel_y retained and disclosed |
| 16 | front_closing | m/s | mixed velocity bases | radial closing=-dot(rel_p,rel_v)/max(distance,epsilon) |
| 17 | front_ttc | s | proxy time | min(distance/radial_closing,999) if >1e-3; else sentinel+invalid |
| 18 | front_thw | s | proxy time | min(rel_x/ego_instantaneous_forward_speed,999) for positive gap and speed; else invalid |
| 19 | front_speed | m/s | world speed norm | hypot(neighbor world vx,vy), missing invalid |
| 20 | front_accel | m/s^2 | speed derivative | same world actor adjacent valid time derivative; reset+mask false |
| 21 | front_heading_rel | rad | instantaneous ego relative | same |
| 22 | front_yaw_rate | rad/s | ABSOLUTE in legacy; RELATIVE in dynamic/nuPlan | ABSOLUTE world heading derivative, actual dt + continuity mask |
| 23 | left_front_valid | 1 | presence | 1 valid assigned neighbor; explicit channel mask |
| 24 | left_front_rel_x | m | instantaneous ego local | same formula with both vehicle centers |
| 25 | left_front_rel_y | m | instantaneous ego local | same, y component |
| 26 | left_front_rel_vx | m/s | instantaneous ego local | same formula from full world vectors |
| 27 | left_front_rel_vy | m/s | instantaneous ego local | same formula from full world vectors |
| 28 | left_front_distance | m | Euclidean separation | same |
| 29 | left_front_delta_x | m | instantaneous ego local | duplicate rel_x retained and disclosed |
| 30 | left_front_delta_y | m | instantaneous ego local | duplicate rel_y retained and disclosed |
| 31 | left_front_closing | m/s | mixed velocity bases | radial closing=-dot(rel_p,rel_v)/max(distance,epsilon) |
| 32 | left_front_ttc | s | proxy time | min(distance/radial_closing,999) if >1e-3; else sentinel+invalid |
| 33 | left_front_thw | s | proxy time | min(rel_x/ego_instantaneous_forward_speed,999) for positive gap and speed; else invalid |
| 34 | left_front_speed | m/s | world speed norm | hypot(neighbor world vx,vy), missing invalid |
| 35 | left_front_accel | m/s^2 | speed derivative | same world actor adjacent valid time derivative; reset+mask false |
| 36 | left_front_heading_rel | rad | instantaneous ego relative | same |
| 37 | left_front_yaw_rate | rad/s | ABSOLUTE in legacy; RELATIVE in dynamic/nuPlan | ABSOLUTE world heading derivative, actual dt + continuity mask |
| 38 | left_rear_valid | 1 | presence | 1 valid assigned neighbor; explicit channel mask |
| 39 | left_rear_rel_x | m | instantaneous ego local | same formula with both vehicle centers |
| 40 | left_rear_rel_y | m | instantaneous ego local | same, y component |
| 41 | left_rear_rel_vx | m/s | instantaneous ego local | same formula from full world vectors |
| 42 | left_rear_rel_vy | m/s | instantaneous ego local | same formula from full world vectors |
| 43 | left_rear_distance | m | Euclidean separation | same |
| 44 | left_rear_delta_x | m | instantaneous ego local | duplicate rel_x retained and disclosed |
| 45 | left_rear_delta_y | m | instantaneous ego local | duplicate rel_y retained and disclosed |
| 46 | left_rear_closing | m/s | mixed velocity bases | radial closing=-dot(rel_p,rel_v)/max(distance,epsilon) |
| 47 | left_rear_ttc | s | proxy time | min(distance/radial_closing,999) if >1e-3; else sentinel+invalid |
| 48 | left_rear_thw | s | proxy time | min(rel_x/ego_instantaneous_forward_speed,999) for positive gap and speed; else invalid |
| 49 | left_rear_speed | m/s | world speed norm | hypot(neighbor world vx,vy), missing invalid |
| 50 | left_rear_accel | m/s^2 | speed derivative | same world actor adjacent valid time derivative; reset+mask false |
| 51 | left_rear_heading_rel | rad | instantaneous ego relative | same |
| 52 | left_rear_yaw_rate | rad/s | ABSOLUTE in legacy; RELATIVE in dynamic/nuPlan | ABSOLUTE world heading derivative, actual dt + continuity mask |
| 53 | right_front_valid | 1 | presence | 1 valid assigned neighbor; explicit channel mask |
| 54 | right_front_rel_x | m | instantaneous ego local | same formula with both vehicle centers |
| 55 | right_front_rel_y | m | instantaneous ego local | same, y component |
| 56 | right_front_rel_vx | m/s | instantaneous ego local | same formula from full world vectors |
| 57 | right_front_rel_vy | m/s | instantaneous ego local | same formula from full world vectors |
| 58 | right_front_distance | m | Euclidean separation | same |
| 59 | right_front_delta_x | m | instantaneous ego local | duplicate rel_x retained and disclosed |
| 60 | right_front_delta_y | m | instantaneous ego local | duplicate rel_y retained and disclosed |
| 61 | right_front_closing | m/s | mixed velocity bases | radial closing=-dot(rel_p,rel_v)/max(distance,epsilon) |
| 62 | right_front_ttc | s | proxy time | min(distance/radial_closing,999) if >1e-3; else sentinel+invalid |
| 63 | right_front_thw | s | proxy time | min(rel_x/ego_instantaneous_forward_speed,999) for positive gap and speed; else invalid |
| 64 | right_front_speed | m/s | world speed norm | hypot(neighbor world vx,vy), missing invalid |
| 65 | right_front_accel | m/s^2 | speed derivative | same world actor adjacent valid time derivative; reset+mask false |
| 66 | right_front_heading_rel | rad | instantaneous ego relative | same |
| 67 | right_front_yaw_rate | rad/s | ABSOLUTE in legacy; RELATIVE in dynamic/nuPlan | ABSOLUTE world heading derivative, actual dt + continuity mask |
| 68 | right_rear_valid | 1 | presence | 1 valid assigned neighbor; explicit channel mask |
| 69 | right_rear_rel_x | m | instantaneous ego local | same formula with both vehicle centers |
| 70 | right_rear_rel_y | m | instantaneous ego local | same, y component |
| 71 | right_rear_rel_vx | m/s | instantaneous ego local | same formula from full world vectors |
| 72 | right_rear_rel_vy | m/s | instantaneous ego local | same formula from full world vectors |
| 73 | right_rear_distance | m | Euclidean separation | same |
| 74 | right_rear_delta_x | m | instantaneous ego local | duplicate rel_x retained and disclosed |
| 75 | right_rear_delta_y | m | instantaneous ego local | duplicate rel_y retained and disclosed |
| 76 | right_rear_closing | m/s | mixed velocity bases | radial closing=-dot(rel_p,rel_v)/max(distance,epsilon) |
| 77 | right_rear_ttc | s | proxy time | min(distance/radial_closing,999) if >1e-3; else sentinel+invalid |
| 78 | right_rear_thw | s | proxy time | min(rel_x/ego_instantaneous_forward_speed,999) for positive gap and speed; else invalid |
| 79 | right_rear_speed | m/s | world speed norm | hypot(neighbor world vx,vy), missing invalid |
| 80 | right_rear_accel | m/s^2 | speed derivative | same world actor adjacent valid time derivative; reset+mask false |
| 81 | right_rear_heading_rel | rad | instantaneous ego relative | same |
| 82 | right_rear_yaw_rate | rad/s | ABSOLUTE in legacy; RELATIVE in dynamic/nuPlan | ABSOLUTE world heading derivative, actual dt + continuity mask |

## 验证范围

14种确定性运动fixture：匀速、加速、转弯、角度环绕、不规则时间、侧滑、换槽、邻车缺失、自车无效、真实lane-aware函数的三直车道、epoch时间精度、静止起步、加速邻车缺失、跨π邻车缺失；legacy Waymo、Dynamic Waymo、nuPlan实际接口共42次调用。atol=2e-4 SI、rtol=2e-5只覆盖小坐标float32算术误差，不容忍语义错误。时戳未按规则记录、布局混排和oracle物理错误独立判阻断，即使allclose。没有科学样本推理。
