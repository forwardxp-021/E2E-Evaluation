# R1 技术烟雾执行计数校正 v1.1

状态：`TECHNICAL_SMOKE_CAP_VIOLATION_RECORDED_NO_RERUN`。

此为实现/执行计数校正，不更改 R1 context/mechanism contract、候选参数、F_match caliper 或任何 threshold。

## 发现

`R1_Technical_Smoke_Report_v1.md` 与其 v1 manifest 以 unique logical arm 计为 48：每 family 为 6 个 scenario 的 baseline+3 treatments，即 24，两个 family 共 48。

但 v1 execution loop 在每一个 candidate pair 内重新构造同一 baseline trajectory。因此实际 trajectory-core construction calls 为：

- R-HLC：6 x (3 次 baseline 重建 + 3 次 treatment)=36；
- R-TSB：6 x (3 次 baseline 重建 + 3 次 treatment)=36；
- 合计：72，而非硬上限 48。

没有 external nuPlan rollout、background replay、embedding、BDD、probe、checkpoint 或 RBR 读取；重复的 baseline 为确定性、未持久化的 core construction。但用户指定的计数上限仍未满足，不能把 v1 当作合规 technical smoke。

## 处置

- v1 artifacts 完整保留，新增 v1.1 manifest/report/CSV 作为审计校正；不静默覆盖历史。
- 执行器已改为每 scenario 只构造一次 baseline，未来在获得新授权时才会符合 `6 x (baseline+3)=24` / family、总 48 的上限。
- 本阶段不重跑、不扩展 smoke budget、不创建正式 roster。现有结果只保留为 `NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`，不得据此推荐 generator 或解锁后续工作。
