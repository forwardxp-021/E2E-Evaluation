# TSB-Coast P1 F_match合同与测量误差审查 v1

`CONTRACT_STATUS=DEVELOPMENT_VALUES_RETAINED_EXECUTION_BLOCKED`。独立测量函数为`tsb_coast_p1_measurement.f_descriptors`，与旧`trajectory_descriptors_timestamp_aware`在规范解析states上逐项完全一致；不是H30特征提取或BDD。

| 描述符 | 定义 | 固定绝对差上限 |
|---|---|---:|
| mean_speed | 80个raw速度算术均值 | .10m/s |
| end_minus_start_speed | raw末−首 | .15m/s |
| path_length | rear-axle XY欧氏步长和 | .80m |
| mean_abs_accel | raw速度physical-time gradient(edge_order2)绝对值均值 | .02m/s² |

保留六位小数描述符及≤limit+1e−12比较语义，未改旧caliper。额外平衡参照P0：两臂全窗实际负a积分差≤.15m/s、Baseline峰与Treatment每峰差≤.20m/s²；这两个附加项保持为Development审查值，没有因本轮结果调整。实际pre/post状态、低速和官方安全是独立检查。本轮未执行pair，全部NOT_EVALUATED。

## 决定性的测量反例

P0要求速度噪声预算±.01m/s。固定误差序列e=[+.01,+.01,−.01,−.01]重复，dt=.1s。在恒速规范输入上，中心导数误差为 `(e[i+1]−e[i−1])/.2=±.10m/s²`；二阶边界公式在此序列亦有同样幅值。因此mean_abs_accel从0变成.10，达到匹配预算.02的5倍。mean_speed仍10，净速度误差仅−.02，XY保持不变，说明这是速度导数分辨率问题而非总位移或模型检测差异。

对两条C2规范参考分别施加四个固定phase×两种sign（16个条件，**非16独立科学样本**）：mean_abs_accel误差.075～.0825，16条件都超过.02；其余三项没有超限。干净两臂F完全一致，含mean_abs_accel=.32625；连续时间解析值.330380与80点离散定义不同，不能用前者替代。全部数据和规则见`evidence/Measurement_Resolution_Audit_iteration1_v1.json`。

这些是array perturbation fixtures，位置保持不变以隔离速度测量误差，**不是新的物理轨迹、模型响应、nuPlan误差估计或C2结果**。压力序列在运行前固定，不是看真实结果挑出的噪声。它证明指定误差包络下预算不稳健，不证明nuPlan实际必然具有该误差，也不证明所有物理Coast都不匹配。

## 实际生产精度与不能越过的缺口

已读旧planner `_payload` 和`realized_current_ego_row`：speed取`float(ego.dynamic_car_state.speed)`，时间为整数微秒，JSON写出未按.01速度舍入。该来源不支持“实际速度噪声就是±.01”的断言；数值序列很可能更精确。但float保存精度本身不能证明相对目标物理量的总误差、导数误差和动态离散误差上界。

本次实现者将P0“误差预算不被占满”理解为在已要求的±.01压力包络内也应能准入，而非只需字节重复。这个保守解释及反例一起交Owner审查；不是私自宣布新的传感器噪声模型。当前没有获批的替代生产误差上界，因此不收窄噪声、不换median3 F、不扩容差、不自动跑一对“看看”。最小修复是零运行测量误差合同审查，明确实际source/error bound与P0压力测试的准入角色。

若Owner随后依据独立证据确认更小误差包络，必须前瞻记录理由并重新测试/绑定；不得用C2/H/RBR结果倒推。若需要修改F定义或阈值，则必须单独科学修约，本轮未实施。
