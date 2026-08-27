# R1 技术烟雾报告 v1.1（执行计数校正）

结论：`NONCOMPLIANT_EXECUTION_DIAGNOSTIC_ONLY`，不是合规 technical smoke。

v1 的 logical arm 设计是 48（两个 family 各 24），但初版实现为每 candidate 重建 shared baseline，实际有 72 次 trajectory-core construction calls，超过批准上限。虽然无 external runtime rollout、无写入 trajectory tensor、无 representation/BDD/probe/checkpoint/RBR 读取，这仍是必须如实记录的 implementation accounting defect。

因此 v1 中的机制、F_match 与运动学数字只可作为确定性 core diagnostic：HLC 机制 gate 在 NOMINAL/STRONG 为 6/6，但 frozen heading F_match 和运动学完整性均 0/6；TSB F_match 和运动学完整性为 6/6，但机制 gate 全为 0/6。没有 candidate 可标为 `RECOMMENDED_AFTER_TECHNICAL_SMOKE`。

已修正 future executor 的 baseline reuse，使其在获得新的授权后才会按严格 48 构造。当前不重跑、不增补、不创建正式 roster，正式 readiness 保持 `NOT_READY`。
