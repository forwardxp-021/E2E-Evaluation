# TSB-Coast P0 — F_match适用性与新版本提案 v1

状态：`NEW_FMATCH_VERSION_REQUIRED_PROPOSAL_ONLY`。旧四项有科学意义，可以沿用**测量含义及明确版本的计算公式**；旧数值阈值没有被本轮自动批准。

## 逐项审查

实际旧入口：`tools/r1_closed_loop_benchmark_v2_1.py::trajectory_descriptors_timestamp_aware`；caliper入口：`tools/r1_prospective_generator_contract_v2.py::TSB_PRIMARY_F_MATCH_CALIPERS`。

| 描述符 | 旧定义/容差 | 对Coast的意义 | 风险、建议 |
|---|---|---|---|
| mean_speed | 80点raw speed算术均值；.708203939m/s | 限制整体速度水平 | 保留定义；解析连续均速不能替代采样均值。新建议≤.10m/s |
| end_minus_start_speed | raw首尾速度差；.978755681m/s | 限制整体净降速 | 保留；旧容差可容纳较大剂量差，新建议≤.15m/s |
| path_length | rear-axle平面步长和；5.38423459m | 限制总运动量及重心偏移 | 保留；直线解析∫v仅是可行性参考，新建议≤.80m |
| mean_abs_accel | raw speed physical-time gradient(edge_order=2)的绝对值均值；.11777666m/s² | 限制制动/恢复总变差 | 保留公式，与median3机制测量分离；新建议≤.02m/s² |

这些旧小数是已冻结development-informed calipers。此次所读合同/源码**没有提供可独立转移到Coast的物理等价或测量误差论证**；不能因精确小数看起来严谨就当通用物理常数，更不能把历史PASS用来批准新基准。旧阈值原样保留于历史B1。

## 新数值的可审查依据与缺口

拟名 `TSB_COAST_FMATCH_v1`，上述四项均为**待审工程/物理误差预算，不是已经验证的等价界限**。净降速.15约为目标2.61的5.75%；均速.10在7.9s对应约.79m路程，与.80m路径预算一致；平均绝对加速度.02×7.9≈.158m/s约为总降速6.1%。均速/路径相关是预期，不当作四个独立控制变量。

建议额外公开但不能悄悄塞进旧四项：两臂实际负加速度积分差≤.15m/s、Baseline峰与Treatment两峰各自的幅值差≤.20m/s²，作为**新资格的剂量/峰值平衡附加gate提案**。理由：净速度差可能因正负抵消而掩盖总制动差；峰值相差.2已达到near-zero带的一个尺度，避免主要靠更强的一脚形成差异。C1理想同峰，C2理想峰差.145，因此不是按未来H效果定门槛；实际动态是否满足未知。新门槛需要P1实测量误差支持及Owner批准，缺任一项不能称新F_match已冻结。

P1前用确定性边界/量化fixture确认这些误差预算不会被测量误差占满；P1只看已批准B-development物理/安全信号。若无法满足，不按差值样本分布重取q95或不断放宽阈值；在限定候选/预算内停下报告不可行。P2前一次性冻结公式、rounding（建议沿用6位及1e−12比较）、所有阈值、附加gate、missingness/分母和完整实现hash。

## 与机制兼容及不可声称的消除

等I、等M且a≤0的解析例子证明四项目标不与“无正恢复”内在冲突，不是实际F_match证明。jerk、RMS、分箱加速度、ACF与制动时间分布仍可有系统差别；H30本来就有这些合法能力。新四项及附加gate也不能保证所有低阶因素消失、只剩纯时间次序因果差异。若整体速度/剂量仍主导，不能直接进入公平检测解释。

禁止以H低检出、RBR高检出或MMD距离挑参数、容差、场景；禁止计算本轮新的H/RBR/BDD。`F_match≠ego13≠H30`；两个NOT_ESTABLISHED继续保留。
