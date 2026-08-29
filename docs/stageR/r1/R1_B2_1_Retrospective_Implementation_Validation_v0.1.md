# R1 B2.1 Retrospective Implementation-only Validation v0.1

状态：`DIAGNOSTIC_NOT_NEW_SMOKE_EVIDENCE`。只读取旧 B2.1 trace/roster/map，不修改历史合同或结果，不声称 amended benchmark 已通过。

## 结果

- corrected context ingestion：48/48 run 的 warmup current observation 均含真实 dynamic actors，48/48 均可观察到至少一个跨 8/10 帧稳定 track ID；证明强制全 ABSENT 不成立。完整 semantic canonicalization 仍需 runtime official map query，旧 trace 本身不足以回填 Primary。
- temporal 0..0.9：0/48 run 的前十个 current-ego physical timestamp 满足 exact 100000 μs cadence；观测 0→9 span 为 0.899445–0.900746 s。因此旧 B2.1 对新 Primary temporal grid 全部 `NOT_EVALUABLE_TEMPORAL_GRID`，不得重标/重采样。
- native route builder：在旧 24 identities/map 上，按 8 s 所需 forward coverage 可构造 23/24；HLC 12/12、TSB 11/12。剩余 TSB identity `f464a2a451d85356` 为 `CURRENT_EGO_HAS_NO_NATIVE_ROUTE_EDGE`，属于 applicability diagnosis。
- current-ego anchor：新实现 synthetic tests 对 position/heading/speed/timestamp 达成 exact construction identity；旧 TSB discontinuity 会被定义性消除，但这是 implementation counterfactual，不是新 smoke evidence。
- HLC native coverage：按旧 B2.1 frozen source/target reference，7/12 满足 native 8 s coverage，5/12 分类为 `NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION`。

所有数字仅用于验证 amendment 是否针对已识别实现缺陷，不用于拟合 threshold、筛选新 roster 或改变 generator 参数。
