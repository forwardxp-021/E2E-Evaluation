# B1-R2 Final Qualification Report v1

正式判定：`B1_TSB_BENCHMARK_QUALIFICATION_PASS`。

Owner指令SHA `0090dab4e7ade26e397b644cd493e8b36d91cd9474adc7764a2157fe13f7ab33`；授权SHA `9189d1ca1f8b4ec2c22cc52771936127fb2bfdca724af34796aa283db9a1caeb`。Executable source base `498aa18fd1e22ad5edd7cfd7b6c3352fd012c7f4`；authorization record commit `6b09bffd79adc774ed47e1e51adf4ec748cfa556`。元数据提交改变HEAD不改变已冻结源码基准；逐文件hash/Git blob是权威绑定。

最终真实零运行预检：6/6 PASS（若verdict为BLOCKED则未完成全部准入），预检状态 `B1_R2_PREFLIGHT_PASS`。实际执行严格按18T2、18B1、19T1、19B1、20B1、20T1顺序，每arm独立OS进程，复用唯一正式B1 runner/六步生命周期、RP实际回调检查和官方安全adapter。没有修改TSB/参数/阈值。

Historical attempts=35；new attempts=6；cumulative=41≤41；unique complete arms=40/40；complete pairs=20/20。原18 Treatment ATTEMPT_1 infrastructure failure永久保留，没有ATTEMPT_3或其他第二次尝试。

| 项目 | 固定分母 | 可评价分母 | 描述性Wilson95（分母20） |
|---|---|---|---|
| technical_completeness | 20/20 | 20/20 | [0.838875, 1.000000] |
| measurement_validity | 20/20 | 20/20 | [0.838875, 1.000000] |
| baseline_one_phase | 20/20 | 20/20 | [0.838875, 1.000000] |
| treatment_two_phase | 20/20 | 20/20 | [0.838875, 1.000000] |
| joint_mechanism | 20/20 | 20/20 | [0.838875, 1.000000] |
| F_match | 20/20 | 20/20 | [0.838875, 1.000000] |
| official_safety | 20/20 | 20/20 | [0.838875, 1.000000] |
| no_LOW_SPEED_ENDSTOP | 20/20 | 20/20 | [0.838875, 1.000000] |
| full_contract_qualification | 20/20 | 20/20 | [0.838875, 1.000000] |

LOW_SPEED_ENDSTOP observed=0/20。未完成/不可评价成员不当作无endstop或科学PASS。只有ALL_20_PAIRS_PASS_FROZEN_JOINT_QUALIFICATION才通过，Wilson不改变标准。

- B1-TSB-18: PASS; mechanism=PASS; F_match=F_MATCH_PASS; safety=PASS; technical=; scientific=
- B1-TSB-19: PASS; mechanism=PASS; F_match=F_MATCH_PASS; safety=PASS; technical=; scientific=
- B1-TSB-20: PASS; mechanism=PASS; F_match=F_MATCH_PASS; safety=PASS; technical=; scientific=

Infrastructure stop: `None`。科学失败和基础设施失败分别记录；未完成时不声称20对已完整评价。原01–17机制/F_match来自原冻结报告，official safety/full-contract来自IR2，直接继承而非重新判定。历史B1→IR→IR2→IR3→R→RP→R2完整来源均保留。

所有输入身份、配置、原始trace、passive recorder、官方parquet、官方simulation-log、独立execution manifests和日志位于 `/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/stageR/b1_resume_execution_r2_v1` 与本目录归档；manifest绑定全部实际文件hash。技术缺失不在本轮自行恢复、不修改代码后继续。

Final hash audit `PASS`；保护CSV SHA `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。科学源码、依赖、关键数据和历史文件按新授权逐项核对。

Exposure按SESSION×claim×observed×decision influence记录。B benchmark机制/安全暴露不自动成为RBR-vs-H Primary污染；没有改变未来V/C成员、encoder、H、kernel或Primary规则。

LOW_ORDER_NUISANCE_ELIMINATED=NOT_ESTABLISHED；TSB_CLEAN_RESIDUAL_TASK=NOT_ESTABLISHED。B1不能证明RBR>H、RBR能检测TSB、所有人工特征无效或真实量产监控有效。

RBR训练/推理、H、BDD/MMD/z64、Primary、V/C均未执行且未获授权。PASS时只建议Owner review→RBR-V2 training-entry audit→U-only training authorization decision；FAIL时Owner benchmark scientific review；INCOMPLETE时Owner review of technical blocker and remaining attempt authority。完成后停止，不开启额外实验或工程循环。

## 实际收尾审计

最终六项均ARM_COMPLETE，6个不同OS进程，每arm实际80行trace、80行planner telemetry、79条recorder transition；96份官方metric parquet和6份非空官方simulation-log全部保留且校验和匹配。完整20对：measurement、baseline一阶段、treatment两阶段、joint mechanism、F_match四项、official safety及full-contract均20/20；40个完整arm的collision均0、drivable-area均true。LOW_SPEED_ENDSTOP=0/20，无新增科学或技术失败。

同一53项回归测试执行后再次通过，0 skipped/deselected。原始日志逐字节保留；随报告提交的副本仅去除行尾空白，双方SHA及位置见Log Archive Provenance。无未完成B1执行项；RBR-V2训练入口审计和U-only训练批准仍是后续Owner决策，未自动开展。
