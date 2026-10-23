# B1-R Final Qualification Report v1

正式判定：`B1_R_PREFLIGHT_BLOCKED`。

原20独立 SESSION、40 unique arms 保持冻结，01–17 原机制/F_match 直接继承，安全和 full-contract 继承 IR2；不重新计算或运行。历史18 treatment ATTEMPT_1 的 infrastructure failure 永久保留。IR3 静态 route gate 只是工程准入。

Historical attempts = 35；new attempts = 0；cumulative = 35 ≤ 41；unique complete arms = 34/40；complete pairs = 17/20。

| 项目 | 原始分母 | 可评价样本 | 描述性 Wilson 95%（分母20） |
|---|---:|---:|---|
| technical_completeness | 17/20 | 17/20 | [0.639581, 0.947631] |
| measurement_validity | 17/20 | 17/17 | [0.639581, 0.947631] |
| baseline_one_phase | 17/20 | 17/17 | [0.639581, 0.947631] |
| treatment_two_phase | 17/20 | 17/17 | [0.639581, 0.947631] |
| joint_mechanism | 17/20 | 17/17 | [0.639581, 0.947631] |
| F_match | 17/20 | 17/17 | [0.639581, 0.947631] |
| official_safety | 17/20 | 17/17 | [0.639581, 0.947631] |
| no_LOW_SPEED_ENDSTOP | 17/20 | 17/17 | [0.639581, 0.947631] |
| full_contract_qualification | 17/20 | 17/17 | [0.639581, 0.947631] |

LOW_SPEED_ENDSTOP observed events = 0/20；不可评价样本不算无 endstop。所有 full-contract 20/20 才通过；Wilson 不改变成功标准。

- `B1-TSB-18`：TECHNICAL_INCOMPLETE；mechanism=NOT_AVAILABLE；F_match=NOT_AVAILABLE；safety=NOT_AVAILABLE；NOT_RUN_AFTER_STOP_OR_PREFLIGHT
- `B1-TSB-19`：TECHNICAL_INCOMPLETE；mechanism=NOT_AVAILABLE；F_match=NOT_AVAILABLE；safety=NOT_AVAILABLE；NOT_RUN_AFTER_STOP_OR_PREFLIGHT
- `B1-TSB-20`：TECHNICAL_INCOMPLETE；mechanism=NOT_AVAILABLE；F_match=NOT_AVAILABLE；safety=NOT_AVAILABLE；NOT_RUN_AFTER_STOP_OR_PREFLIGHT

Infrastructure stop: `None`。

## 来源和身份

Owner 授权：`B1_R_Owner_Conditional_Authorization_v1.json`，SHA256 `f4944458eb82501f9670f75afba96dff7f42144cbd99695538b4f709b7893669`；执行源码 commit `d360d86f751a93218c31757d3d7d9151aa8d5279`。授权列出所有 tracked tools/config、外部官方实现与环境文件、DB/map、科学合同和全部历史产物的真实 SHA256。原 roster/specs/pair bindings SHA：`fdc44c7a126892b9dfcaa0b44fce56eeab42b8649f8c98caf355fc63e40b229b` / `06889559956943f94c5be6031db0b0fb974c81225e3cb04a2bfcf6858b568ee7` / `220a310184e750e9779b0df4dd29303deada2786c53f4b3ad01314921e1708b6`。

每个实际 arm 独立 OS 进程、planner/controller/tracker/callback/recorder/random state；复用唯一 B1 正式 runner site，官方 MetricFileCallback 收尾，官方 serializer、16 metric parquets、原 safety adapter、六步 lifecycle 与 artifact hash 全部验证。独立 attempt namespace 和账本不改写原35预算。原始失败或新失败均不被科学成功覆盖。

新 raw root：`/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/stageR/b1_resume_execution_v1`；每次 execution manifest、resolved config、trace、recorder、metric、serializer 原样保留，所有文件 hash 见本阶段 manifest；日志和执行 manifest 归档随报告提交。历史来源：原 B1 → IR offline metrics → IR2 safety/full-contract → IR3 native route repair → 本轮有限授权。

## 暴露和科学边界

Exposure Update 按批准的 S1R 记录 SESSION × claim × observed × decision influence。本轮机制/F_match/安全暴露属于 B benchmark exposure，不自动等于 Primary 污染；没有调整 detector/encoder/H/BDD/roster/阈值，也未安排未来 V/C。

LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED。
TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED。
RBR/H/BDD/MMD/z64/Primary/V/C 均未执行，训练或未来执行不获授权。本阶段不能证明 RBR 优于 H、所有传统特征失效、未见场景普遍100%成功或真实量产有效性。

## 验证与下一步

Postrun audit `PASS`；历史 910 文件 SHA 保持一致；受保护 CSV SHA `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

PASS 时只进入 Scientific Owner review → S3 / RBR-V2 training protocol readiness；FAIL 时只评审 benchmark 科学适用性和论文主张；INCOMPLETE 时只报告基础设施、预算和合法可选路径。不得额外 attempt，不自动启动训练。完成后停止。

## 本轮实际停止点

正式 preflight 在首个零运行工厂构建后，以 `RECORDER_CALLBACK_NOT_BOUND` 阻断。原因是本轮新增 gate 检查了可合法为空的共享 callback 列表，而非实际 simulation 的 callback graph；这是预检实现错误，不能解释为 TSB 科学失败或确认的生产回调缺失。未运行6次中的任何一次，原35次预算未变化。仅前7项完整记录为PASS，其余未完成。按照Owner指令，没有自动修复或再次预检。新运行的生命周期/80行trace/79条transition均无可报告；上文这些要求是执行合同，并非本轮已产生的验证结果。

原17对仍保持机制、F_match、安全与full-contract PASS。20对中其余3对没有新的完整数据；17/20只是原始分母下已确认通过的数量，不能解释为完整样本成功率或把未运行3对判为科学FAIL。后续仅建议Owner审查本预检阻断，再决定是否另行授权工程修复和新的版本化预检。详细责任、源码依据和合法路径见 B1_R_Preflight_Blocker_Diagnosis_v1.md。
