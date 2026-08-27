# R1 Official nuPlan Replay Contract v0.2（失败诊断）

## 结论

本文件不是 `v1.0` 通过合同。首次受约束的官方 runtime validation 在第一条已 claim 的运行
`R-HLC__25944935eadb52f1__RUN_A` 停止，因此 `BACKGROUND_REPLAY_DETERMINISM = NOT_VERIFIED`，
`OFFICIAL_REPLAY = NOT_READY`。

## 已绑定但未获验证的环境

- `MASTER_SEED=2026082701`；DB fingerprint：`5b53ad42497fe6926c73936970658a3717d1c2cc51077812d5284a57fd242489`。
- map fingerprint：`a85e17eba18e5fdd65148705844b8f189bb4d4373a1d82805e1f8ffd4ae8afb3`。
- HLC V2 contract SHA：`72414d1accf656704fadf255f002de815989c2f17908fdbf6bcb36b2352dc142`；TSB contract SHA：`24068de671a884114660d303283dcdd5e50a8454af0a0ff5f64468d2ac715722`。
- runtime-validation roster SHA：`fc5c52a15eef9f71adb6f279e99bb4a0a6312fdc6013671c75550703c2759ac6`。

## 失败与边界

官方 runner log 显示 planner 缺少 nuPlan 所需的 `compute_trajectory` 入口。runner 虽以进程返回码 0
结束，但该 scenario 被标记为 failed，未产生 planner trace、collision metric 或 drivable-area metric。
因此不能把空输出视为重复一致，也不能比较 15 个冻结类别。

本次只 claim 了 1 个 `OFFICIAL_CLOSED_LOOP_RUN`，随后 fail-closed；未执行 RUN_B、其他三场景、treatment、
48-call smoke 或第三次重跑。该问题是执行器接口实现缺陷，不是修改 mechanism threshold、替代 primary metric
或形成科学 outcome 结论，故不记录为 scientific protocol deviation。

## 后续条件

修复后的执行器没有被用于本合同中的任何官方运行。若要再进行 runtime validation，必须由 scientific owner
明确给出新的单独授权和新的绑定执行记录；本合同不会自动授权 48-call technical smoke。
