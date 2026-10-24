# TSB-Coast C0 — 实际控制与动力学接口审计 v1

日期：2026-10-24（Asia/Shanghai）。能力判定：**`ABSTRACT_COAST_PROXY_ONLY`**。这是接口/模型的可表达能力判定，不是现有LQR已具有coast模式、也不是新的闭环资格PASS。

当前生产栈不能证明“松开驱动与制动，让真实车辆按被动力自然滑行”；它可以表达零纵向加速度请求及官方滤波后的简化响应。若Owner接受这个明确简化的代理，未来可另行授权版本化控制门与记录绑定。**当前原样LQR没有此控制门，本轮没有实现它。**

## 1. 实际生产身份和证据范围

仓库基准 `1d7b8a7c6121c6ca1d0119a7e1d784386c70392a`；外部nuPlan-devkit HEAD `e9241677997dd86bfc0bcd44817ab04fe631405b`。devkit已有`requirements_torch.txt`未提交修改，本轮保护不动；不能仅凭HEAD宣称整个依赖checkout干净，逐文件SHA为准。

读取旧B1-R2六份**实际执行时保存的**`resolved_config.json`，六份均绑定TwoStageController→LQRTracker→KinematicBicycleModel；dt=.1s、N=10、q_long=10、r_long=1。motion_model配置没有覆盖`accel_time_constant`，实际构造默认值来自源码`.2s`；不是把yaml缺省猜成零。现行`tools/s2r_production_executor.py::build_official_tsb_arm`仍导入相同官方类并安装原被动recorder。没有新建runner、初始化模型或重新运行历史arm。 另将官方LQR、TwoStageController、KinematicBicycleModel当前SHA与B1-R2原授权的dependency_file_hashes逐一比对，三者完全一致，证据见`evidence/Historical_Production_Source_Binding_v1.json`。

完整路径、符号行号、SHA、默认参数和六份配置绑定见`evidence/Source_And_Algebra_Audit_v1.json`及Manifest。结论只适用于这一实际本地版本/生产绑定，不概括所有nuPlan模型或所有仿真器。

## 2. 控制链逐层核对

| 层 | 实际入口 | 源码直接支持 |
|---|---|---|
| Planner | `tools/r2_b_controller_aware_planner_v1.py::_tsb_states, compute_planner_trajectory` | a_ref按时间积分为未来speed/route位置，再构造InterpolatedTrajectory；它不是踏板/扭矩命令。即便参考EgoState acceleration字段写0，速度/pose序列仍可隐含加减速 |
| Tracker | `nuplan/.../tracker/lqr.py::track_trajectory` | 生成纵向`accel_cmd`和`tire_steering_rate`，装入DynamicCarState；输入不是纵向jerk、驱动扭矩或制动扭矩 |
| 预瞄 | `LQRTracker._compute_reference_velocity_and_curvature_profile` | 从pose拟合velocity/curvature，于当前时间+h取参考速度，h=Ndt=1s；`jerk_penalty`是拟合惩罚，不是执行器jerk接口 |
| Controller | `TwoStageController.update_state` | 原样把tracker返回dynamic_state传给motion_model.propagate_state；没有驱动/制动分配器、coast mode或驱动制动互锁 |
| Motion model | `KinematicBicycleModel._update_commands, get_state_dot, propagate_state` | 对目标a做一阶离散低通，更新rear-axle车体纵向速度，Euler积分平面位置/航向；无纵向受力平衡 |
| State | `DynamicCarState.__init__`、`EgoState.build_from_rear_axle` | 保存速度、加速度、角速度/角加速度、转向及时间等运动学量；没有独立驱动力、制动力、踏板、轮端扭矩、档位或回收状态 |
| Vehicle parameters | `VehicleParameters.__init__` | 几何尺寸、轴距等；当前纵向模型无质量/阻力/坡度的受力参数输入 |
| Recorder | `tools/r2_bj_b0_2_passive_actual_lqr_recorder.py::install` | 包装tracker返回，记录actual_acceleration_command、steering_rate和独立shadow；**位置在运动模型滤波之前**，不观测轮端动作 |

## 3. 必须回答的八个问题

1. **当前输入是什么？** 期望rear-axle纵向加速度（m/s²）和转向率（rad/s）。加速度变化率是滤波产生的结果，非独立控制输入。
2. **独立驱动和制动执行器？** 此生产链没有。正/负a请求不能分别鉴定驱动扭矩和制动扭矩。
3. **空气阻力/滚阻/坡度/发动机制动/回收？** 此模型的状态导数与propagate完整路径没有这些纵向力项，也没有相应能量/动力总成状态。地图有几何或场景数据库有其他字段，不表示当前模型将其转成坡度力；当前平面SE2路径没有这个通路。
4. **a输入0代表什么？** 若指a_ref=0，只表示参考速度平台；若指tracker送给模型的u=0，表示抽象期望加速度请求归零，滤波后实际模型加速度仍可能非零。二者都不是可观测的零踏板/零轮端扭矩。
5. **不改官方模型能否真自由滑行？** 不能表示/核验完整物理语义。官方接口允许输入u=0，可构造简化的零命令代理，不必修改车辆模型；这还需要新的显式控制模式绑定，而不是恢复旧P1授权。
6. **约1秒预瞄是否会主动控制？** 会存在该可能。滑行期间若未来第二脚进入horizon，LQR可提前负控制；若未来拟合速度高于当前速度，也可输出正控制。a_ref平台不保证u=0。
7. **现有recorder能证明什么？** 对未经后处理的tracker输出可证明逐次实际返回的净a请求，前提是完整79次记录及绑定审计；不能证明两种轮端力都零，更不能证明未来新控制门之后真正送入模型的命令。
8. **仅净a命令足够证明真自由滑行吗？** 不足。相同净加速度可能来自同时驱动/制动或驱动力补偿阻力；该模型甚至没有这些可分解通道，不能由零和倒推出每一项为零。

## 4. 能力、物理推断、不可验证三类

**源码事实**：加速度接口、无独立力/扭矩通道、无被动阻力方程、官方低通与Euler更新、当前LQR无coast模式、recorder位置。

**解析推断**：零命令保留加速度滤波记忆；平路无阻力且残余状态已消退时会恒速；LQR速度误差可产生正/负请求。推断方程见Dynamics文档及纯代数证据，不是运行车辆模型得到的结果。

**目前不能验证**：真实驱动和制动均关闭、踏板状态、滑行时的空气/轮胎/坡度/动力总成效应、自然驾驶员动作或C2闭环是否成形。不能把本轮B级能力判定升级为A级或论文真实自由滑行实证。

本轮不改官方controller/model、旧recorder或任何冻结B1/P0/P1。来源、依赖和全部输出SHA统一绑定本目录Manifest。
