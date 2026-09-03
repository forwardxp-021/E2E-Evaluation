# R2-BJ-A HLC 形态解析可行性报告 v1

## 结论

最终状态为 `R2_BJ_A_OFFLINE_ARCHITECTURE_NOT_READY`。V4 将 treatment 改为绝对时间 C2 的 advance → hold → retreat → recommit，并取消会在 1.1 秒直接进入已运行 phase 的正 lag 偏移。按冻结最大车道分离 5.148466 m 计算，新的 intrinsic morphology 峰值横向加速度为 2.814720 m/s²，低于冻结 6.0 m/s²；这只证明 intrinsic 项，不等于 composite trajectory 全包络通过。

## 分项归因

审计分别保留 morphology intrinsic、online stitching、native road curvature、target capture 和 composite final trajectory。隔离的 straight/same-lane stitching 10 个 case 全部通过，最大横向加速度为 0.318816 m/s²，terminal target-frame offset 最大为 4.524e-16 m。完整 `_states` 共执行 3296 个主包络 case，1160 个通过、2136 个失败。原始 source-universe 笛卡尔边界包含速度 17.246181 m/s 与曲率 0.082281 1/m 的组合，仅 native 曲率项即对应 24.472901 m/s²，因此 composite 失败不能归因于 capture，也不能靠改 morphology 消除。

## 边界连续性

1.1 秒处 baseline 与 treatment 的 P/V/A 差均为零；没有采用 `tau=t-T_DIVERGE+lag` 的正时移。另以 t=1.0 common plan 的 state1 作为 t=1.1 treatment plan 的 state0，位置误差为 0 m，单独完成跨轮 common→treatment 检查。

## 治理处置

由于 mandatory envelope 存在失败，R2-BJ-B readiness request 被扣留。未选择 roster，未请求 simulation 授权；runner.run、工程 simulation、科学 simulation 和 TSB simulation 均为 0。R2-C、confirmatory smoke 与 RBR 均未开始。
