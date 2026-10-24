# TSB-Coast C0 — 动力学与控制可行性 v1

证据范围：源码解析、旧resolved config、闭式代数。**没有运行track_trajectory、update_state、propagate_state或任何runner，也没有生成车辆响应时间序列。** 本文数值仅重述已用参数，不提出新制动幅值、时长或容差。

## 1. 实际官方更新方程

`TwoStageController.update_state`取u_k=tracker返回的rear_axle_acceleration_2d.x；`KinematicBicycleModel._update_commands`执行：

`alpha=dt/(dt+tau); a_next=(1−alpha)·a_current+alpha·u_k`。

随后`get_state_dot/propagate_state`用该滤波a更新车体纵向速度：`v_next=v_current+dt·a_next`。平面位置用旧纵向速度和heading作Euler更新；steering另有滤波，lateral velocity在该模型设0。位置步长、状态a和由speed做中心差分的测量a并非同一计算定义，不能直接混作一个无误差量。

实际默认dt=.1s、tau=.2s，alpha=1/3、记忆比例r=2/3。这是模型构造默认/旧执行配置，不是C0新调参。模型没有空气、滚动、坡度或动力总成阻力项；`.2`速度floor来自旧planner参考构造、LQR另有stopping分支，不能把这些当成运动模型自然阻力。

## 2. 零请求为什么仍会减速

当u_k=0且初始模型加速度a_0≠0，闭式关系：

`a_n=r^n·a_0; Δv_n=dt·a_0·r·(1−r^n)/(1−r)`。

因此第一步仍保留2/3的旧a；若a_0为负，零请求后仍有一段减速尾巴。就旧1.8s长度所含18次.1s更新作**归一化代数核查**，残余比例约.00067664，Δv/a_0约.19986467s；无限尾部极限为tau·a_0。没有选择a_0的实际幅值，也不由这些数字确定新coast时长。

该尾巴是抽象加速度滤波记忆，不是测得的刹车压力释放、空气阻力或发动机制动。若a_0=0且u始终0，当前模型保持纵向速度，因其根本没有被动力；如果a_0>0，零请求也可短暂继续增速。不能把所有正速度漂移都归为新的主动驱动。

## 3. a_ref=0与LQR输出不等价

Planner平台速度目标经pose拟合/预瞄进入LQR。纵向一维子问题由源码可得：

`u = q·h/(r_LQR+q·h²) × (v_ref(t+h)−v_current)`。

现行h=1s、q=10、r_LQR=1，因此系数10/11。单位正/负速度误差分别产生正/负请求，误差为0才返回0（在普通LQR分支）；这只是归一化函数解析，不是新控制实验。

第二脚未来参考进入约1s预瞄时，即使当前中段a_ref=0也可能提前制动。若参考固定为释放开始速度而实际受残余/被动力降速，速度跟踪可能正向补偿；若旧planner逐次从current speed重建平台，在理想无拟合偏差/无未来制动时可得到零误差，但它仍不是强制coast模式，不能保证最终u=0。pose拟合和停止分支均必须审计。

## 4. 真正物理滑行需要的方程及缺项

物理推断：对已声明前向运动和符号，`m·dv/dt=F_drive−F_brake+F_passive(v,grade,powertrain_state)`。真正无主动纵向作用要求分别约束F_drive/F_brake，随后由被动力自然决定a。F_passive的具体构成、回收/发动机拖曳分类、参数和误差必须前瞻批准，本轮不给新数值。

恒速不能鉴别分解：主动驱动力可正好抵消阻力；零净命令也不能排除两个非零执行器相互抵消。现有模型既没有m/各力变量，也没有可独立观测的这些分量，无法由EgoState重建。其抽象输入空间足以输入u=0，但不足以证明物理层执行器都关断。

## 5. 最小未来代理路径与代价

建议仅在Owner接受代理后考虑一个独立版本的longitudinal command gate及post-gate记录，不改官方motion model、不重置滤波、不给控速回正。当前官方LQR/recorder保持原样；本轮不编码或安装gate。

必须在将来审查：lateral LQR使用按原accel_cmd预测的velocity_profile，若最终u被gate归零，其横向预测与实际输入可能不一致；不能宣称只改一个数就没有路线/安全代价。旧recorder对tracker类名/身份有安装检查，外包tracker也可能不符合旧验证，不能为绕过它而削弱冻结检查。需新版本绑定与独立零运行测试，量限定在这两个接口，不扩为新车辆平台。

若必须满足真实轮端物理语义，则最小技术类别为有独立纵向执行器和被动力分解的模型/数据接口；这会增加模型验证、参数证据、兼容性和统计再绑定成本，不在C0自动实施。

纯测试验证配置/AST与代数恒等式、非法输入阻断，不验证真实车辆传递函数。全部SHA及可重复输出见本目录Manifest。
