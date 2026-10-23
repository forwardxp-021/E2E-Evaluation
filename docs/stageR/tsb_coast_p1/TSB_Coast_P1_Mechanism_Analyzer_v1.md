# TSB-Coast P1 机制测量实现 v1

代码：`tools/tsb_coast_p1_measurement.py`；版本`TSB_COAST_P1_MEASUREMENT_v1`。只接受数组，不能构建或执行runner。实现是Development测量合同，不是P2资格合同；当前执行准入被F_match分辨率阻断。

## 输入、时间与导数

必须80速度值和80整数微秒时间戳，即79个间隔；先整数减首时间戳，再float秒，防止epoch截断。速度有限、时间严格递增，Δt对.1s最大偏差≤1微秒，末时刻对7.9s偏差≤1微秒；否则抛MEASUREMENT_INVALID。无跨洞压缩、插补或planned替代。

主测量为edge-replicated median3(speed)和`np.gradient(..., physical_time, edge_order=2)`。raw derivative、median3+edge_order1为固定稳定性诊断，不能择优替换主测量。速度模的切向变化只是纵向行为代理，不是踏板观测；没有真正current-ego trace输入时不能称实测。

## 逐步算法及边界

先测全80帧，找连续a≤−.60支持；与固定[1.1,5.5]s域相交的段须满足：采样末−首支持≥.40s、峰幅值≥.80m/s²、支持内梯形负a积分≥.35m/s。恰好Baseline1段/Treatment2段；不合格短波纹不合并救PASS。若一条合格支持实际延续到域外/数据外，标PHASE_EDGE_CENSORED；起点恰落1.1但相邻已观测点不在该支持时，不误作未知延续。

Treatment两段gap使用第一支持末至第二支持首，≥.70s；在严格内部找|a|≤.20连续支持，取最长、并列取最早，支持≥.50s。raw速度漂移容差为`.20×core支持秒+.05m/s`，无需正恢复。所有duration使用时间戳末−首，绝不用N×.1；当前gate不插值扩充时长。

完整gap（含两端支持点）检查a>.20连续支持≥.30s、∫max(a,0)dt>.10m/s、任一点a>.40，任一即POSITIVE_RECOVERY_EXCEEDED。剂量以实际采样时间上的梯形积分计算非负part，积分规则在本次零运行实现中明确；不以积分采样数代替秒，不以负正抵消掩盖恢复。

全窗低速<1m/s持续≥.50s、出现非正速度或明确floor触发均LOW_SPEED_ENDSTOP。真正controller floor触发需要未来生产telemetry提供，本次数组工具不能凭速度正常就证明没有触发。raw/median3或edge诊断的合格段数不同、峰差>.20、段/core支持差>.20s或正恢复排除结论反转，稳定性返回MEASUREMENT_UNSTABLE。

## 已做纯函数验证

22项测试覆盖理想单制动/双制动/真coast、−.1/−.2拖曳、近零边界内外、+.2/+.401/+1.4恢复排除、sample-support与Ndt区别、epoch平移、±1微秒量化、缺帧/重复/非有限/异常cadence/非整数时间、低速及floor标记、旧四项F公式一致性、±.01速度噪声反例和禁止runner/detector调用。

解析C2的PASS只是检验实现可识别规范输入，**没有实际闭环C2 PASS**。固定噪声16个条件的机制raw/median3稳定性诊断没有触发不稳定，阻断来自F的raw mean_abs_accel误差预算；不可混写成近零加速度带已被实际噪声吞没。没有实现生产planner/factory/CPU监督或route正式预检，它们是NOT_REACHED。
