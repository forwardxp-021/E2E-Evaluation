# R2-BH HLC Target-Capture Development Report v1

## 结论

`R2_BH_DEVELOPMENT_NOT_CONVERGED`。V1 constant re-anchor diagnosis 经 5/5 合成 offset case 支持。V2 已建立固定 absolute-time target-center command attractor，但 3 轮 fresh DEV-ARCH closed-loop 结果没有实现 frozen mechanism 或 endpoint，因此没有冻结 HLC V2 candidate，也没有组合完整 G_R2 candidate。

## 架构证据

V2 将 behavior morphology 与 target capture 分离。state0 精确保持 current ego；state1+ 的 native target-frame lateral 与 heading residual 以 C2 quintic 权重衰减。capture start/end 固定在 episode absolute time，不随 replanning 重启。每轮 16/16 arm 在 capture end 的 state1 residual command 均为 0；scientific realized p(t) 定义未改变。

## 三轮结果

- Round 0：mechanism 0/8、endpoint 0/8、F_match 8/8、engineering 8/8、safety 4/8。
- Round 1：mechanism 0/8、endpoint 0/8、F_match 7/8、engineering 8/8、safety 4/8。
- Round 2：mechanism 0/8、endpoint 0/8、F_match 8/8、engineering 8/8、safety 4/8。

最终 treatment retreat count margin 全部为 -1，commit latency 与 monotonic margin 均因 measurement not OK 而不可评估。最终 endpoint gate：offset 0/8、heading 8/8、lateral velocity 8/8、route progress 7/8。treatment terminal offset |m| 的 min/p25/median/p75/max 为 `4.094948/4.433190/4.658622/5.511358/7.883494`。

## 防火墙与停止规则

8 个 DEV-ARCH identities 与 historical/R1/R2-A/R2-B 重叠为 0，全部永久 engineering-only。R2-B HLC 未重跑；TSB 新仿真为 0；科学阈值与 scenario-specific rule 均未改变。严格停止于 3 轮，不执行 Round 4，不选择 R2-C identity，不启动 confirmatory smoke 或 RBR。
