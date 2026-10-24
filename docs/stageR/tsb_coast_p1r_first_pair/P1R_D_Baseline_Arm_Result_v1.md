# P1R-D Baseline 真实闭环结果 v1

`P1R-C2-DEV-01-BASELINE`：技术完整，arm科学PASS。原始结果目录：`outputs/stageR/tsb_coast_p1r_d_first_pair_v1/P1R-C2-DEV-01-BASELINE/`。来源、完整SHA256见本轮Manifest；生产源码73e96f7、独立授权提交d3e5e75。本文件只读描述，不重算或替换冻结结论。

官方production独立OS进程执行一次，fresh planner、LQR、model、callbacks和recorder。80个current-ego状态、79个控制区间、80个planner记录；全六项生命周期完成，16项官方parquet与正式serializer均存在。Baseline全程原版LQR，没有COAST模式。实际80帧观察审计PASS（标识、时间戳、位置/航向和灯态同index比较；不含future predictions）。

冻结median3(speed)+物理时间gradient测量得到恰好一段有效制动：1.8–3.6秒，支持时长1.8秒，减速度峰值1.365850222 m/s²，门槛内负加速度剂量2.050995315 m/s。峰值≥0.80、支持≥0.40、剂量≥0.35，arm reasons为空。基线参考制动窗为[2.1,3.9)，它与实际合格段的时间并不相同：官方LQR预瞄产生提前响应，不能拿参考窗代替实际段。

实际起始/末尾速度14.151693647/11.686627195 m/s，最低11.685947555 m/s；无低速端停、无reference floor。所有区间的model expected/actual差为0。官方指标为at-fault碰撞计数0、drivable-area合规true，冻结arm safety PASS。这里没有宣称所有类型碰撞均为0。

科学CPU 7.551957642秒，wall 8.849630416秒。账本保留原状态COMPLETE_REVIEW_REQUIRED；后续实际观察审计通过且arm_science_review.json为arm_pass=true，不追溯改写原账本。
