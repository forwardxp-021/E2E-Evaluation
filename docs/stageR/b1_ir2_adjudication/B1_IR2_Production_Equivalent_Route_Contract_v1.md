# B1-IR2 Production-Equivalent Route Contract v1

状态：`INITIAL_QUERY_EQUIVALENCE_ESTABLISHED; FULL_EPISODE_EQUIVALENCE_NOT_ESTABLISHED_FOR_LATENT_DEFECT_CASES`。

本合同取代 B1-IR v1 对全 topology 检查结果的解释。旧 artifact 不回写。没有修改 TSB、planner/controller 行为、trajectory generation 或生产 route builder。

## Identity

`ROUTE_IDENTITY_VALID`：manifest 与冻结 arm spec 的 map name、ordered roadblock IDs 所计算的 route hash 一致，且 pair 两 arm 的 frozen route identity 相同。它不包含运行兼容性结论。

## Production query

唯一生产 route 实现为 `tools/r1_closed_loop_benchmark_v2_1.py::build_native_route_reference_v1_1`。有限请求距离直接调用生产 planner 已有的 `_tsb_required_distance`，配置直接读冻结 round-0 TSB JSON。

初始时刻 nominal=0，生产使用 common BASELINE profile；current ego 由冻结 precontext 初始 pose/speed/time 提供。checker 使用同一 map API、ordered route、current state、required distance 调用同一函数。因此对该已知初始查询可证明相同输入/函数/结果，不产生新的 trajectory。

## Risk and classification

1. 初始有限查询失败：`INCOMPATIBLE`，阻止 runner。
2. 初始查询成功且全 topology walk 未发现缺失 successor：`COMPATIBLE`（只认证该静态检查范围，不保证未来 ego 在 lane/route 上的所有状态）。
3. 初始查询成功但全 walk 发现 tail successor 缺失：`AMBIGUOUS`，另记 `INITIAL_REQUIREMENT_PASS_WITH_LATENT_RUNTIME_RISK`；不是初始查询失败，也不是必然 episode 失败的证明。
4. 无法解释的解析/环境错误：`TECHNICAL_ERROR`。

全 walk 的无限请求只作为诊断：它不是 production planner 的输入，不能冒称 production-equivalent finite horizon。checker 保留风险是为了记录未证明的未来条件。当前 executor 接受的状态仍只有 COMPATIBLE，因此 B1-18/20 不能进入 runner；这部分阻塞未被本轮解除。

## Retrospective validation

对已执行 arm，每个记录的 call-entry current state 和 nominal iteration time 都重新交给同一个 route 函数；时间遵循生产 `_absolute_episode_clock` 的 iteration×0.1 定义。回放只构造 route reference，不调用 compute_trajectory、compute_planner_trajectory、track_trajectory、update_state、propagate_state 或 runner。

这能验证已经发生的有限查询：06/10/14 各两 arm 480 次通过，18 treatment 第 54 个 call-entry（iteration 53）复现失败。它不是对 B1-20 的未来轨迹预测。

## Resume authority

01–17：完整科学结果保留，NO_RERUN。18：attempt 1 保留；同 route/stack 的 attempt 2 当前不可执行，先做基础设施闭合。19：通过静态范围检查，仅在 Owner 授权后可首次执行。20：ROUTE_AMBIGUOUS，继续阻止 entry，不替换原 SESSION。

Owner 后续可要求不改变科学逻辑的基础设施修复/静态证明，或正式规定 frozen denominator 内技术未完成项如何进入最终判定；本合同不替 Owner 作最终科学决定，也不授权 prospective amendment。分母永远保持原 20 对，直到明确的新授权文件另作规定。
