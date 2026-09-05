# E2E-Evaluation 项目快速参考

## StageR / R2-BK — Family scope 分叉与 TSB-only R2-C 设计冻结

### 1. 命令

以下命令只校验冻结 SHA、HLC 关闭处置、TSB candidate integrity 和 zero-run 治理边界；不会运行 simulator，也不会再次调用 B1.1 recovery analyzer：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r2_bk_verify_family_scope_freeze.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bk_tsb_r2c_design.py
```

### 2. 期望行为

- 只读核验 B1.1 recovery、HLC negative result、TSB candidate 与传递 component SHA；
- 保持 HLC/B0 identities 的冻结与暴露边界，禁止跨 family pooling；
- 报告 frozen source universe 未物化完整 TSB eligibility population，容量结论 fail-closed；
- 不选择 TSB R2-C roster、reserve 或 schedule，不改变参数、阈值和 protected CSV。

### 3. 通过标准

- HLC closure、identity disposition、family bifurcation 和 TSB candidate SHA closure 全部通过；
- 容量不得用 `1,425` 个结构性 log 上界冒充 eligible pool，状态明确为需 Owner 授权离线 materialization；
- `runner.run=0`、offline recovery invocation `=0`、simulation `=0`；
- `ROSTER_SELECTION=FALSE`、`R2_C_STARTED=FALSE`、`CONFIRMATORY_SMOKE_STARTED=FALSE`、`RBR_STARTED=FALSE`。

## StageR / R1 B2.8-R3.2 — Pair binding 与 48-run orchestrator 冻结

### 1. 命令

以下仅执行零运行构造检查，绝不传递 `--execute`：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_8_r3_2_execute_frozen_48run_smoke.py \
  --output docs/stageR/r1/r1_b2_8_r3_2_orchestrator_dry_run_v1.0.json
```

### 2. 期望行为

- 固定读取 roster v2.1、schedule v2.1 与 24 个 pair binding，按 run order 1…48 逐一构造 runner；
- HLC 使用冻结 clearance 与 native references；TSB 的 clearance 固定为 `null`；
- dry-run 在 `runner.run()` 前硬停止，不启动 simulation、不消费真实预算。

### 3. 通过标准

- pair binding 为 24/24，HLC/TSB 各 12/12；
- orchestrator 为 48/48 ready，49th claim 在 runner 前失败；
- 正式执行必须同时具有匹配 final manifest SHA 的 Owner authorization；任何 SHA、输出路径或技术基础设施错误均在后续 run 前停止。

## StageR / R1 B2.8-R3.1 — 最终执行 SHA 与 safety/evaluator 接线冻结

### 1. 命令

只在新的版本化结果文件尚不存在时，执行最终零运行回归；它会构造 48 个 `SimulationRunner`，但严格在任何 simulation start、step 或 rollout 前停止：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_8_r3_1_final_zero_run_regression.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_8_r3_launch_freeze.py \
  tests/test_r1_b2_8_r3_1_final_binding.py
```

### 2. 期望行为

- 复核 immutable roster v2.1 / schedule v2.1 的 SHA，对已冻结 48 run 完成精确 scenario resolution、完整 Hydra composition 与 `SimulationRunner` 构造；
- 复用历史冻结的 nuPlan collision/drivable-area canonicalizer，并用版本化 adapter 将实际 metric-engine Parquet 输出交给 V2.1 evaluator；
- Primary trace 只读取 `REALIZED_CURRENT_EGO` 的 iteration 0...79；任何 iteration >=80 仅保留为 secondary、non-primary raw trace；
- 不重新枚举、不重新选择身份、不启动 simulation、不消费预算、不训练或运行 RBR。

### 3. 通过标准

- final manifest 状态为 `FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION`；
- 48/48 exact resolution、48/48 Hydra 与 48/48 runner construction 均通过，49th dry claim 在 simulator start 前失败；
- safety adapter 与 post-run evaluator dispatcher 均已 SHA-bind，80/81/100 行 trace 的 primary 0...79 提取通过，primary window 内缺失、重复或时间不单调失败；
- `OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`，actual official runs 与 consumed budget 均为 0。

## StageR / R1 B2.4 — Final Prospective Contract Conformance Freeze

### 1. 命令

仅执行 prospective synthetic/readonly conformance tests；不会启动 rollout、enumeration、roster selection 或 RBR：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_closed_loop_benchmark_v2.py \
  tests/test_r1_closed_loop_context_adapter_v2.py \
  tests/test_r1_b2_4_adversarial_conformance.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 验证 timestamp-aware HLC/TSB calculators 在 exact 0.1 s grid 上与 frozen v1 完整输出一致，并接受微小 official timestamp jitter；
- 验证 context v2.1 的 iteration ordering、actual time audit、anchor hazard、velocity fail-closed 与 stable 8/10 slow lead；
- 验证 TSB curved/offset/repeated-route continuity、HLC ±179° geometry、state0/first-segment continuity、2.0 m/s floor 与 HLC common-envelope clearance；
- 不读取 representation、BDD、probe、checkpoint 或 RBR，不生成新 scenario/roster/rollout。

### 3. 通过标准

- 合并测试为 `28 passed`，其中 B2.4 adversarial suite 为 `15 passed`；
- `tools/check_no_tmp_dependencies.py` 输出 `OK`；
- SHA binding manifest 中所有 implementation/contract SHA 与工作区一致；
- 保持 `NEW_ROLLOUT=NOT_AUTHORIZED`、`R1_FORMAL_DEVELOPMENT_ROSTER=NOT_READY`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 B2.3 — Prospective closed-loop benchmark implementation amendment

### 1. 命令

本阶段只运行 synthetic/unit 与既有 B2.1 trace/map 的只读测试，不启动 planner rollout：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m py_compile \
  tools/r1_closed_loop_benchmark_v2.py \
  tools/r1_closed_loop_context_adapter_v2.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_closed_loop_benchmark_v2.py \
  tests/test_r1_closed_loop_context_adapter_v2.py
```

### 2. 期望行为

- context v2 只接受 condition-identical warmup iterations 0–9 和 exact 100000 μs cadence，并从 official observation/map/route/traffic-light 真实构造 slots、稳定 track IDs、gap、THW 与 hazard multi-hot；
- Primary measurement 只使用 actual realized current-ego iterations 0–79；planned-first 仅作 generator-intent secondary；
- 每次 replan 的 trajectory[0] 与 current ego position/heading/speed/timestamp exact identity，phase clock 为 absolute episode time；
- TSB 沿 native route realization，HLC 沿 native source/target geometry 且禁止 extrapolation；generator schedules、mechanism thresholds、F_match calipers 均不修改；
- 历史 B2.1 trace/map 只作 `DIAGNOSTIC_NOT_NEW_SMOKE_EVIDENCE`，不选择新 roster、不运行新 smoke。

### 3. 通过标准

- 两个测试文件全部 PASS；旧 48-run trace 均能读取真实 actors/stable IDs，但 0/48 满足新的 exact temporal grid；
- old-map 8 s native route 构造为 HLC 12/12、TSB 11/12，失败 identity 仅作 applicability diagnosis；
- TSB outcome-blind 解析与 0.001 m/s synthetic grid 均得到 proposed floor 2.0 m/s；
- 保持 `NEW_ROLLOUT=NOT_AUTHORIZED`、`R1_FORMAL_DEVELOPMENT_ROSTER=NOT_READY`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 B2.2 — B2.1 残差基准只读法证审计

### 1. 命令

仅分析既有 B2.1 48-run trace；该命令不会启动 planner rollout，也不会读取 representation、BDD、probe、checkpoint 或 RBR：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_audit_b2_1_residual_forensics_v1.py \
  --output-dir docs/stageR/r1
```

### 2. 期望行为

- 先核验 48/48 run、24/24 pair、0 技术失败、冻结 roster SHA 和 selector salt；
- 流式只读既有 trace，构造 planned-first 与前 80 个连续 iteration 的 realized-ego 诊断；
- 分开报告 raw pre-context identity 与 frozen canonical context semantic conformance；
- 生成 gate contingency、安全归因、时间锚点、replan continuity、HLC geometry、HLC/TSB mechanism forensic、中文诊断报告和 owner 决策单；
- 不覆盖任何 v1.1 历史文件，不产生新 rollout，不修改生成器或冻结门禁。

### 3. 通过标准

- 工具输出 `status=PASS`，并在结束时通过冻结输入 SHA before/after 一致性断言；
- B2.1 完整性为 48 run、24 pair、0 技术失败；roster SHA 为 `0617e79b9f51d8b2ae8ac76b110e1dbcfaa77dad200a73b405eb2d6a54675e52`，selector salt 为 `617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9`；
- 保持 `R1_RESIDUAL_BENCHMARK_ENABLEMENT=BENCHMARK_FAMILY_NOT_READY`、`NEW_ROLLOUT=NOT_AUTHORIZED`、`RBR_A/B/C=NOT_AUTHORIZED`。

## Stage 6K — 纯纵向风格剂量曲线（Issue #252）

### 1. 命令

冻结25%、50%、75%三档纵向处置和549个场景×剂量任务：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_freeze_longitudinal_dose_response.py \
  --design_json configs/stage6k_longitudinal_dose_response.json \
  --stage6j_locked_scenarios_csv outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_locked_scenarios.csv \
  --output_dir outputs/stage6k_longitudinal_dose_freeze_v1
```

先将下述runner命令作为dry-run执行；真实全量在末尾追加`--execute`、确认SHA和`--resume`，
并使用`caffeinate -dimsu`：

```bash
caffeinate -dimsu /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_run_longitudinal_dose_rollouts.py \
  --freeze_manifest outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_freeze_manifest.json \
  --locked_jobs_csv outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_locked_jobs.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage6k_longitudinal_dose_batch_v1 \
  --execute \
  --confirm_locked_jobs_sha256 4bbfa3adb23c5e3e090c3d5a66f636cb9400d059257c987709dda55056980b26 \
  --resume
```

全量结束后，先冻结解盲前统计补充，再构建三档统一view、context和实现运动学曲线：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_freeze_preanalysis_addendum.py \
  --design_json configs/stage6k_preanalysis_addendum.json \
  --rollout_freeze_manifest outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_freeze_manifest.json \
  --locked_jobs_csv outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_locked_jobs.csv \
  --batch_manifest outputs/stage6k_longitudinal_dose_batch_v1/batch_manifest.json \
  --batch_state outputs/stage6k_longitudinal_dose_batch_v1/batch_state.json \
  --batch_status_csv outputs/stage6k_longitudinal_dose_batch_v1/batch_scenario_status.csv \
  --output_dir outputs/stage6k_preanalysis_addendum_freeze_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_prepare_longitudinal_dose_views.py

# 三档分别运行；dose_label替换为dose25/dose50/dose75，planner名称同步替换。
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6k_longitudinal_dose_views_v1/dose25 \
  --output_dir outputs/stage6k_longitudinal_dose_context_v1/dose25 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --required_planners pdm_closed_assertive_longitudinal_dose25_v1 pdm_closed_conservative_longitudinal_v1 \
  --write_projection_debug --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_evaluate_realized_dose_curve.py \
  --addendum_manifest outputs/stage6k_preanalysis_addendum_freeze_v1/stage6k_preanalysis_addendum_manifest.json \
  --views_dir outputs/stage6k_longitudinal_dose_views_v1 \
  --contexts_dir outputs/stage6k_longitudinal_dose_context_v1 \
  --stage6j_kinematic_dir outputs/stage6j_pure_longitudinal_kinematic_gate_v1 \
  --output_dir outputs/stage6k_realized_longitudinal_dose_curve_v1
```

使用同一个Waymo checkpoint生成三档embedding后，运行四档统一BDD与质量敏感性：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_run_longitudinal_dose_bdd.py \
  --addendum_manifest outputs/stage6k_preanalysis_addendum_freeze_v1/stage6k_preanalysis_addendum_manifest.json \
  --realized_dose_summary outputs/stage6k_realized_longitudinal_dose_curve_v1/stage6k_realized_dose_summary.json \
  --new_embeddings_dir outputs/stage6k_longitudinal_dose_embeddings_v1 \
  --stage6j_embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --stage6j_bdd_dir outputs/stage6j_pure_longitudinal_paired_bdd_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage6k_longitudinal_dose_bdd_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6k_evaluate_lane_quality_sensitivity.py \
  --addendum_manifest outputs/stage6k_preanalysis_addendum_freeze_v1/stage6k_preanalysis_addendum_manifest.json \
  --new_contexts_dir outputs/stage6k_longitudinal_dose_context_v1 \
  --new_embeddings_dir outputs/stage6k_longitudinal_dose_embeddings_v1 \
  --stage6j_context_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --stage6j_embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --output_dir outputs/stage6k_lane_quality_sensitivity_v1
```

### 2. 期望行为

配置以Stage 6J保守profile为0%、激进profile为100%，六个纵向IDM参数线性插值得到
25%、50%、75%；所有profile固定`lateral_offsets=[-0.5,0.5]`。0%同profile下限和
100%端点复用Stage 6J，不新增仿真；新增三档各运行183个相同场景、每场景两个planner，
合计549个任务和1098条official rollout。冻结阶段不读取新增剂量的embedding、BDD或
effect size。runner逐任务隔离，支持`--resume`，并写出`batch_state.json`、
`batch_scenario_status.csv`、`batch_events.jsonl`。

解盲前补充将四个非零overall剂量作为一个Holm family，将4剂量×3 tasks作为一个12项
Holm family；最小剂量必须同时通过speed和RMS acceleration的单侧log-cluster 95%门禁、
以及overall Holm p<0.05。lane fallback/ambiguity只做post-treatment描述性敏感性，禁止据此
删样本、重加权或替代primary。

### 3. 通过标准

- 冻结manifest状态为`FROZEN_BEFORE_LONGITUDINAL_DOSE_ROLLOUTS`，插值审计为true；
- 任务数549、rollout数1098，三档各183个，固定场景SHA-256不变；
- 每档smoke均为`SUCCEEDED`，A/B各一条official success，same-log和strict-token均PASS；
- 全量最终549/549 `SUCCEEDED`、0 failed、0 pending；
- 三档view、context和embedding各183 pair/366行，context validation全部PASS；
- 用“运动学处置实现 + overall Holm p<0.05”定义最小可检出剂量，同时报告raw BDD、
  null q95、BDD/q95、Z_BDD；不得声称存在跨数据集通用raw BDD阈值。

实际结果：549/549任务、1098/1098 rollout成功。25/50/75/100%四档实现运动学门禁均PASS；
overall BDD依次为0.00115612、0.00159972、0.00332234、0.00500090，四档Holm p依次为
0.00428996、0.0000399996、0.0000399996、0.0000399996。因此本协议内最小可检出名义剂量为
25%；25%对应BDD/null q95=1.290、Z_BDD=3.649。同log整体翻转的四档Holm也全部显著。
task结果不可写成“25%所有任务均检出”：25%仅longitudinal-high-motion通过12项Holm；following
到75%才通过，stop/go从50%通过但100%在12项Holm后未通过，属于任务异质性诊断。

## Stage 6J — 纯纵向 PDM A/B 冻结与真实 smoke

### 1. 命令

先冻结设计和183个同场景配对，不启动仿真：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6j_freeze_pure_longitudinal_confirmation.py \
  --design_json configs/stage6j_pure_longitudinal_confirmation.json \
  --confirmation_ledger_csv outputs/stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --output_dir outputs/stage6j_pure_longitudinal_freeze_v1 \
  --overwrite
```

第一个冻结跟车场景的双planner真实smoke：

```bash
env \
  NUPLAN_DEVKIT_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  NUPLAN_DATA_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset \
  NUPLAN_MAPS_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  NUPLAN_MAP_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  NUPLAN_EXP_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/exp \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage6j_pure_longitudinal_freeze_v1/stage7c_context \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --output_dir outputs/stage6j_pure_longitudinal_smoke_1scene_v1 \
  --planners pdm_closed_assertive_longitudinal_v1 pdm_closed_conservative_longitudinal_v1 \
  --max_scenarios 1 --min_timesteps 2 \
  --require_same_scenario_alignment \
  --require_strict_nuplan_token_alignment \
  --allow_external_planner_name \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]' \
  --command_timeout_s 3600 \
  --nuplan_simulation_command_template '/Users/liuqing/miniconda3/envs/nuplan/bin/python /Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_builder.db_files=[/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1/2021.09.13.18.55.23_veh-45_02099_02822.db] scenario_filter=all_scenarios {scenario_hydra_overrides} worker=single_machine_thread_pool worker.max_workers=1 scenario_builder.max_workers=1 max_callback_workers=1 gpu=false experiment_name=stage6j_pure_longitudinal_smoke_v1 job_name=closed_loop_nonreactive_agents_stage7c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m py_compile \
  tools/stage6j_freeze_pure_longitudinal_confirmation.py \
  tools/stage7c1_run_nuplan_simulation.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/check_no_tmp_dependencies.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6j_pure_longitudinal_confirmation.py \
  tests/test_stage7c_external_planner.py
```

### 2. 期望行为

- 两个profile的`lateral_offsets`固定为相同的`[-0.5,0.5]`；
- 只允许speed fraction、fallback speed、min-gap、headway、accel和decel六项纵向参数不同；
- freeze只读取M6.5 confirmation ledger和DB存在性，不读取embedding、BDD或planner结果数组；
- 主分析选择following、stop/go和`high_magnitude_speed/medium_magnitude_speed`；
- 排除lane-change、dense/vulnerable和`high_lateral_acceleration`；
- 输出冻结manifest、183行场景清单、3行smoke清单、planner参数审计、Stage7C context和中文报告；
- smoke只运行1场景×2 planners，不启动366条全量rollout。

### 3. 通过标准

- `pure_longitudinal_treatment=true`、lateral difference=0、longitudinal difference=6；
- 183个pair分布为following=60、stop/go=67、longitudinal high-motion=56；
- distinct logs=156、duplicate token=0、missing DB=0；
- smoke official success=2/2、trajectory rows=298、shape=`(1,2,149,8)`；
- same-log alignment与strict-token alignment均PASS，`pseudo_rollout=false`；
- 全量366条rollout必须在可断点续跑批处理和显式execute确认后才能启动。

实际结果：freeze与单场景双planner smoke均PASS；smoke耗时约33秒。权威输出为
`outputs/stage6j_pure_longitudinal_freeze_v1/`和
`outputs/stage6j_pure_longitudinal_smoke_1scene_v1/`。

### 4. 183场景可断点续跑批处理（Issue #250）

dry-run：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6j_run_pure_longitudinal_rollouts.py \
  --freeze_manifest outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_locked_scenarios.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage6j_pure_longitudinal_batch_v1
```

真实执行在上述命令后追加：

```text
--execute
--confirm_locked_scenarios_sha256 90b35382b53d4fada7fd4237f1a3efb8595505406ba99a6e0e3f839d7c777036
--resume
```

Mac全量运行必须使用`caffeinate -dimsu`。状态文件为
`batch_state.json`、`batch_scenario_status.csv`和`batch_events.jsonl`；主日志为
`full_primary_run.log`。每个场景使用独立`rollouts/order_NNNN_token/attempt_NNN`，
成功场景在`--resume`时重新审计后跳过；失败场景默认停止重试，只有人工检查后显式
`--retry_failed`才创建新attempt。

批处理启动前会复核freeze/locked SHA-256、183行顺序、DB/log/token、两个planner
fingerprint、Stage7C hash以及nuPlan/tuPlan commit。默认dry-run，缺少`--execute`或
精确确认SHA-256时不得启动official simulation。

实际全量结果：183/183场景成功、366/366 official rollout、0失败、0 pending。

### 5. 统一视图复核与5邻车上下文（Issue #251）

#### 1. 命令

先重新审计并合并183个隔离输出：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6j_prepare_pure_longitudinal_view.py \
  --freeze_manifest outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6j_pure_longitudinal_freeze_v1/stage6j_locked_scenarios.csv \
  --batch_manifest outputs/stage6j_pure_longitudinal_batch_v1/batch_manifest.json \
  --batch_state outputs/stage6j_pure_longitudinal_batch_v1/batch_state.json \
  --batch_status_csv outputs/stage6j_pure_longitudinal_batch_v1/batch_scenario_status.csv \
  --output_dir outputs/stage6j_pure_longitudinal_view_v1
```

再构建Stage5D兼容的5邻车上下文：

```bash
caffeinate -dimsu env \
  PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6j_pure_longitudinal_view_v1 \
  --output_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --required_planners pdm_closed_assertive_longitudinal_v1 pdm_closed_conservative_longitudinal_v1 \
  --write_projection_debug --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6
```

#### 2. 期望行为

- 统一视图工具重新审计全部183对，不跳过失败场景，不改动原始rollout；
- official目录以symlink引用隔离输出，ego张量使用memmap合并；
- 上下文工具解析每条official msgpack，构建与Waymo Stage5D一致的ego、5邻车和interaction特征；
- 生成lane projection与strict-filter诊断；
- 两步均不读取embedding、BDD或effect size。

#### 3. 通过标准

- 183/183重新审计通过、366/366 official rollout、任务数60/56/67；
- 统一张量shape=`(183,2,150,8)`、strict token与same-log均PASS；
- context输出366行，planner/scenario对齐完整，`validation.pass=true`；
- Stage5D schema/formula validation通过，且无静默几何fallback；
- 只有context通过后才允许计算预先定义的纯纵向运动学门禁。

统一视图实际结果：183/183通过、0复核失败、54612个有效trajectory rows、156个独立log。

### 6. 预冻结纯纵向运动学门禁

#### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6j_evaluate_kinematic_gate.py \
  --config configs/stage6j_kinematic_gate.json \
  --context_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --view_dir outputs/stage6j_pure_longitudinal_view_v1 \
  --output_dir outputs/stage6j_pure_longitudinal_kinematic_gate_v1
```

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m py_compile \
  tools/stage6j_prepare_pure_longitudinal_view.py \
  tools/stage6j_evaluate_kinematic_gate.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6j_prepare_pure_longitudinal_view.py \
  tests/test_stage6j_evaluate_kinematic_gate.py
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/check_no_tmp_dependencies.py
```

#### 2. 期望行为

- 只读取统一view ledger、context metadata/ego/mask/neighbor和context validation；
- 对每个same-scenario pair计算assertive-conservative的speed、accel、jerk、yaw-rate、THW、front distance和front exposure差；
- 按156个`log_name` cluster做10000次bootstrap，输出总体与三个task的95% CI；
- 使用结果读取前冻结的速度与RMS加速度双指标门禁；
- 不读取embedding、BDD或effect size，门禁未通过时禁止进入embedding阶段。

#### 3. 通过标准

- 183个完整pair、156个独立log、任务数60/56/67，context `validation.pass=true`；
- `delta_mean_speed`的log-cluster bootstrap 95% CI下界≥0.5 m/s；
- `delta_rms_accel`的log-cluster bootstrap 95% CI下界≥0.1 m/s²；
- 两个主指标必须全部PASS，才设置`embedding_and_bdd_analysis_allowed=true`；
- 输出逐pair CSV、总体/分task对比CSV、门禁decision CSV、summary JSON和中文报告。

实际结果：运动学门禁PASS。平均速度A-B=`0.9147 m/s`，log-cluster 95% CI
`[0.7578,1.0784]`；RMS加速度A-B=`0.1816 m/s²`，95% CI
`[0.1456,0.2175]`。

### 7. Waymo embedding与纯纵向 paired BDD

#### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --device cpu

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7f_aggressive_conservative_paired_delta.py \
  --embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --context_dataset_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --stage7f_dir outputs/stage6j_pure_longitudinal_stage7f_v1 \
  --planner_a pdm_closed_assertive_longitudinal_v1 \
  --planner_b pdm_closed_conservative_longitudinal_v1 \
  --output_dir outputs/stage6j_pure_longitudinal_stage7f_v1/paired_delta_assertive_minus_conservative

caffeinate -dimsu /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6j_run_paired_bdd.py \
  --config configs/stage6j_paired_bdd_analysis.json \
  --embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --paired_delta_csv outputs/stage6j_pure_longitudinal_stage7f_v1/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --kinematic_gate_summary outputs/stage6j_pure_longitudinal_kinematic_gate_v1/stage6j_kinematic_gate_summary.json \
  --output_dir outputs/stage6j_pure_longitudinal_paired_bdd_v1
```

#### 2. 期望行为

- embedding固定使用原Waymo Stage5/6的83D context GRU checkpoint，输出366×64；
- BDD总体primary固定为single-RBF biased MMD²和100000次pair内label swap；
- following、stop/go、longitudinal high-motion为三个pre-treatment secondary task；
- task p值做Holm校正；报告以中文输出；
- 只解释受控同场景纯纵向benchmark intervention，不外推异场景release可靠性。

#### 3. 通过标准

- checkpoint SHA-256=`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`；
- embedding shape=`(366,64)`、全部finite、无83D padding；
- 183/183 pair完整且A/B有效时长相同；
- overall按预冻结alpha=0.05判定；三个task只用Holm p解释；
- summary和中文报告保留论文主张边界。

实际结果：overall BDD/MMD²=`0.00500090`，0/100000 null达到observed，plus-one
p=`9.9999e-6`。following=`0.01706723`、Holm p=`0.00129999`；stop/go=
`0.00537483`、Holm p=`0.03300967`；longitudinal high-motion=`0.01358617`、Holm
p=`0.0000299997`。总体和三个task均reject。

## Stage 6I — 冻结可靠性分解与论文主张审计

### 1. 命令

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6i_build_reliability_evidence.py \
  --stage6h_dir outputs/stage6h_nuplan_power_curve_800_v1 \
  --embedding_pool_summary outputs/stage6h_expanded_800_embedding_pool_v1/stage6h_embedding_pool_summary.json \
  --embedding_pool_metadata outputs/stage6h_expanded_800_embedding_pool_v1/metadata.csv \
  --kinematic_contrasts outputs/stage7_m6_6_confirmation_evidence_v1/table_m6_6_kinematic_contrasts.csv \
  --output_dir outputs/stage6i_reliability_evidence_v1
```

### 2. 期望行为

- 只读取Stage 6H summary、operating/detection/split/trial CSV、800-pair pool
  summary/metadata、冻结paired-oracle summary和M6.6运动学对比表；
- 不读取rollout、context或embedding数组，不重新计算BDD；
- 生成overall可靠性表、双方向诊断、task定义/分类/BDD大小、planner处置审计、论文主张
  矩阵、中文Markdown报告和PNG/PDF图；
- 保留原始非单调曲线和每档独立threshold，不做平滑或事后调参；
- 禁止外推400场景/版本以上的检出率或精确样本量。

### 3. 通过标准

- 输入必须为800 pairs / 1600 rows / 489 log clusters且Stage 6H状态完整；
- 2400 splits和14400 scope rows完整，所有split精确达到目标n且log/scenario overlap=0；
- 四档均报告A/A、A/B、Wilson 95% CI、false-negative rate和区间分离margin；
- 两个A/B方向分别报告，且明确只作diagnostic；
- 主张矩阵必须区分公开基准支持、公开release emulation支持、不支持及未评估；
- summary必须保持`frozen_sufficiency_gate_passed=false`和`no_extrapolation_above_observed_range=true`。

实际结果：输入审计PASS；四档A/A与A/B Wilson区间均分离。400场景/版本时A/B detection
为66.5%（59.7%–72.7%），A/A FPR为5.0%（2.7%–9.0%），false-negative rate为
33.5%（27.3%–40.3%）；两个A/B方向分别为62%和71%。公开异场景版本信号获得支持，
但80%单次发布可靠性、通用threshold和真实OEM验证均未获得支持。

术语统一为：BDD是研究量，MMD是统计方法，报告数值为MMD²；项目没有定义MDD，旧讨论
中的MDD按MMD笔误处理。五个task的同场景paired-oracle MMD²依次为0.02478050、
0.02878431、0.00523033、0.01445332、0.01379180；400/版本异场景检出率依次为
15.5%、46.0%、13.0%、63.0%、11.5%。`lane_change`仅按nuPlan原始
`scenario_type=changing_lane_to_left/right`切片，没有确认PDM控制自车实际完成变道；
且两个planner的`lateral_offsets`不同，因此当前A/B是纵向+横向混合处置，不能作为纯
纵向风格证据。

## Stage 6H — 800-pair embedding池与200/250/300/400扩展功效曲线

### 1. 命令

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6h_prepare_expanded_rollout_view.py \
  --freeze_manifest outputs/stage6g_expanded_release_pool_freeze_v1/stage6g_freeze_manifest.json \
  --primary_csv outputs/stage6g_expanded_release_pool_freeze_v1/stage6g_locked_primary.csv \
  --batch_status_csv outputs/stage6g_expanded_release_pool_run_v1/batch_scenario_status.csv \
  --existing_ledger_csv outputs/stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv \
  --output_dir outputs/stage6h_expanded_490_rollout_view_v1

env PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage \
  caffeinate -dimsu \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6h_expanded_490_rollout_view_v1 \
  --output_dir outputs/stage6h_expanded_490_context_v1 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --write_projection_debug --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage6h_expanded_490_context_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage6h_expanded_490_embeddings_v1 \
  --device cpu

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6h_merge_expanded_embedding_pool.py \
  --existing_embedding_dir outputs/stage7_m6_5_locked_confirmation_embeddings_v1 \
  --new_embedding_dir outputs/stage6h_expanded_490_embeddings_v1 \
  --output_dir outputs/stage6h_expanded_800_embedding_pool_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6f_unpaired_power_curve.py \
  --embedding_path outputs/stage6h_expanded_800_embedding_pool_v1/embedding.npy \
  --metadata_csv outputs/stage6h_expanded_800_embedding_pool_v1/metadata.csv \
  --config_json configs/stage6h_nuplan_power_curve_800.json \
  --paired_oracle_json outputs/stage7_m6_5_locked_confirmation_analysis_v1/m6_5_locked_confirmation_summary.json \
  --output_dir outputs/stage6h_nuplan_power_curve_800_v1
```

### 2. 期望行为

- 只对新增490场景构建context和embedding，复用原310个已审计embedding；
- 新旧数据必须使用同一个Waymo checkpoint、83D Stage5D schema和64D embedding；
- 合并输出必须为800 pairs / 1600 rows，重建连续`global_row`且每pair严格完整；
- 功效曲线在200/250/300/400每档运行600 trials，每档独立A/A标定；
- `sequential_full_log_pool_v1`保证最大档位仍以完整log构造两个不重叠版本集合；
- 不读取结果重选场景，不平滑曲线，不外推400/版本以上的样本量。

### 3. 通过标准

- 490-pair view的Stage7C re-audit为490/490 PASS；
- 新embedding为`[980,64]`且全部finite，checkpoint SHA与原310完全一致；
- 合并pool为`[1600,64]`、800/800 complete pairs、旧新token overlap=0；
- 2400/2400 trials的A/B log和token overlap均为0，实际样本量在目标±1；
- 四个overall threshold全部有效，并报告A/A FPR、A/B detection及Wilson 95% CI；
- 只有Wilson detection下界≥80%且FPR上界≤5%时才能声称达到冻结充分性门槛。

实际执行结果：490/490 rollout复审通过；新增context为`[980,150,83]`，新增embedding为
`[980,64]`；合并pool为800 pairs / 1600 rows / 489 log clusters；2400/2400 split均为
精确目标样本量且log、scenario overlap为0。overall结果为：

| 场景/版本 | A/A FPR（Wilson 95% CI） | A/B detection（Wilson 95% CI） |
| ---: | ---: | ---: |
| 200 | 8.0%（5.0%–12.6%） | 30.0%（24.1%–36.7%） |
| 250 | 6.5%（3.8%–10.8%） | 28.5%（22.7%–35.1%） |
| 300 | 3.5%（1.7%–7.0%） | 41.5%（34.9%–48.4%） |
| 400 | 5.0%（2.7%–9.0%） | 66.5%（59.7%–72.7%） |

最终状态为`TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`，不能声称达到80%检出目标，
也禁止从当前结果外推400/版本以上所需的精确样本量。

## Stage 6G — 扩展公开不配对发布池到最多800场景

### 1. 命令

冻结 outcome-blind 主集和技术预备集：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6g_freeze_expanded_release_pool.py \
  --config configs/stage6g_expanded_release_pool.json \
  --output_dir outputs/stage6g_expanded_release_pool_freeze_v1
```

runner dry-run：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6g_run_expanded_release_pool.py \
  --freeze_manifest outputs/stage6g_expanded_release_pool_freeze_v1/stage6g_freeze_manifest.json \
  --primary_csv outputs/stage6g_expanded_release_pool_freeze_v1/stage6g_locked_primary.csv \
  --reserve_csv outputs/stage6g_expanded_release_pool_freeze_v1/stage6g_locked_reserve.csv \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --nuplan_data_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset \
  --nuplan_exp_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/exp \
  --nuplan_devkit_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  --tuplan_garage_root /Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  --output_dir outputs/stage6g_expanded_release_pool_run_v1
```

真实执行时必须额外提供 `--execute`、当前 freeze manifest 文件 SHA-256 和 source
canonical manifest SHA-256；可用 `--max_actions 1` 做 smoke，并用同一输出目录续跑。

### 2. 期望行为

- 选择只读取 frozen `scenario_type`、token/log/DB identity 和技术可运行性，不读取
  embedding、BDD、effect size、轨迹指标或 planner outcome；
- 新增主集490个，配额122/11/115/122/120；与现有310合并后的任务目标为
  182/71/182/182/183；
- lane-change 只新增11个，因为其余39个原定义候选不满足官方 scene-position；禁止用
  不同语义标签补数；
- runner 持续写入 `batch_state.json`、`batch_scenario_status.csv`、
  `batch_events.jsonl` 和每个 attempt 的 driver log，进程中断后可续跑；
- reserve 必须提供主集 status CSV，且只覆盖同任务、reserve-eligible 的技术失败。

### 3. 通过标准

- `stage6g_freeze_manifest.json.status=FROZEN_BEFORE_STAGE6G_ROLLOUTS`；
- 主集490、预备100，token overlap 均为0，现有+主集+预备每 log 最大3；
- 590个冻结 token 的 `official_scene_position_valid=true`，所有 forbidden-input flag=false；
- dry-run 通过 CSV/tool hash、planner fingerprint、nuPlan/tuPlan commit 和路径校验；
- 真实场景只有在两套 planner 均成功、trajectory 非空、严格 log/token alignment 和
  pair tensor audit 全通过时才记为 `SUCCEEDED`；最终池规模只按成功主集/预备补充计算。

## 项目结构
```
E2E-Evaluation/
├── build_dataset.py       # 从 Waymo TFRecord 构建 traj/feat/meta/split
├── dataset.py             # TrajFeatureDataset + KNN pair 预计算
├── model.py               # TrajectoryEncoder (GRU + MLP head)
├── loss.py                # SoftContrastiveLoss + multi_positive_infonce
├── train_embedding.py     # 训练主脚本
├── export_embeddings.py   # 导出全量 embedding (.npy)
├── evaluate_embedding.py  # UMAP 可视化 + 线性探针 + 邻域一致性
├── data/                  # 数据目录（数据不提交）
└── README.md              # 项目说明
```

## 工作流

### 第1步: 构建数据集
```bash
python build_dataset.py \
    --tfrecord_glob "/mnt/d/WMdata/*.tfrecord-*" \
    --output_dir output \
    --min_ego_speed 5.5 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```
**输出**: `output/traj.npy`, `feat.npy`, `meta.npy`, `split.npy`, `summary.txt`, `summary.csv`  
**特征维度**: 20D（标准化后）

### 第2步: 模型训练
```bash
python train_embedding.py \
    --traj_path output/traj.npy \
    --feat_path output/feat.npy \
    --split_path output/split.npy \
    --output_dir output \
    --epochs 50
```
**损失**: `SoftContrastiveLoss`（特征引导软对比）  
**输出**: `output/best_model.pth`, `output/model_final.pth`

### 第3步: 导出全量 embedding
```bash
python export_embeddings.py \
    --traj_path output/traj.npy \
    --checkpoint output/best_model.pth \
    --output_path output/embeddings_all.npy
```
**输出**: `output/embeddings_all.npy`，shape `(N, 64)`

### 第4步: 评估分析
```bash
python evaluate_embedding.py \
    --embeddings_path output/embeddings_all.npy \
    --feat_path output/feat.npy \
    --split_path output/split.npy \
    --analysis_dir output/analysis
```
**输出**: UMAP 散点图、线性探针 R²/Spearman、邻域一致性分析


## Stage 7 — nuPlan official simulation and E2E validation command reference

详见主路线图：[`docs/stage7_nuplan_simulation_and_e2e_validation_roadmap.md`](docs/stage7_nuplan_simulation_and_e2e_validation_roadmap.md)。Stage 7 的原则是：Stage 7C 及之后必须使用 nuPlan 官方 simulation 输出，不允许写成 offline pseudo rollout 或 numpy trajectory rewriting。

### 0. 通用本地环境

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"
```

常用 nuPlan root 参数：

```text
--nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini
--nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
```

Stage 7B.4 当前主 context 目录：

```text
outputs/stage7b4_nuplan_context_merged
```

### 1. 当前状态

- Stage 7A nuPlan readiness: PASS
- Stage 7B.1 expert ego/object export: PASS
- Stage 7B.2 Stage6C-compatible dynamic converter: PASS
- Stage 6C smoke on nuPlan expert context: PASS
- Stage 7B.3 map/ODD feature extraction: PASS
- Stage 7B.4 dynamic + map/ODD merge/alignment: PASS；当前验证目录 `outputs/stage7b4_nuplan_context_merged/`
- Stage 7C.1 official nuPlan simulation smoke: PASS
- Stage 7C.1C exact scenario alignment / exact-token smoke: PASS_LOG_AND_NUPLAN_TOKEN_RERUN
- Stage 7C.2A simple_planner × 3 distinct logs: PASS；运行时必须使用 `--sample_distinct_log_names`
- Stage 7C.2B simple_planner × 5 distinct logs: PASS；输出 shape `[5, 1, 149, 8]`
- Stage 7C.2C IDM longitudinal-only multi-planner rollout: PASS
- Stage 7C.2C-0 native IDM default/conservative/comfort/aggressive smoke: PASS
- Stage 7C.2C-1 wrapper smoke: 1 log × 4 planners: PASS
- Stage 7C.2C-2 wrapper rollout: 5 logs × 4 planners: PASS；输出目录 `outputs/stage7c2c2_idm_longitudinal_5logs`
- Stage 7C.3 PDM lateral/interaction planner extension: TODO
- Stage 7D Stage6-compatible export: PASS
- Stage 7E Stage5D common-core context builder: implemented，requires lane-aware runtime validation
- Stage 7E direct context-dataset embedding: PASS for previous smoke，cleanup 后需要 rerun
- Stage 7F: NEXT

---

### Stage 7B.3 — Map/ODD feature extraction

#### Purpose

读取 Stage 7B.2 dynamic context dataset，并基于 nuPlan map API 生成与 dynamic window 行对齐的 map/ODD-lite features。该步骤只提取 map/ODD context，不训练、不 rollout、不合并 Stage 7B.4 特征。

#### Command

当前 repo 中实际脚本名为 `tools/build_nuplan_map_odd_features.py`：

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

python tools/build_nuplan_map_odd_features.py \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --input_dynamic_dir outputs/stage7A_nuplan/expert_context_dataset \
  --output_dir outputs/stage7b3_nuplan_map_odd \
  --split mini \
  --max_scenarios 50 \
  --radius_m 50.0 \
  --sample_stride 5 \
  --overwrite
```

小样本 smoke：

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

python tools/build_nuplan_map_odd_features.py \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --input_dynamic_dir outputs/stage7A_nuplan/expert_context_dataset \
  --output_dir outputs/stage7b3_nuplan_map_odd_smoke \
  --split mini \
  --max_scenarios 5 \
  --overwrite
```

#### Expected output files

```text
outputs/stage7b3_nuplan_map_odd/
├── map_odd_feat.npy
├── map_odd_meta.csv
├── map_odd_feature_schema.json
├── map_odd_report.md
└── warnings.json
```

#### Expected shape / key metrics

```text
map_odd_feat.npy: [23, 37] in latest verified mini run
map_odd_meta.csv rows: 23 in latest verified mini run
warnings: []
map_odd_status: PASS
map_odd_feat rows align with dynamic context rows
```

#### PASS criteria

- `map_odd_feat.npy` 是二维 `[N, F_map]`，并且所有值 finite。
- `map_odd_meta.csv` 行数等于处理的 Stage 7B.2 metadata 行数。
- `map_odd_feature_schema.json.feature_names` 长度等于 `map_odd_feat.npy.shape[1]`。
- `map_odd_report.md` 报告 alignment check 和 map candidate validation。
- `warnings.json` 是结构化 JSON；最新验证期望 `warnings: []`、`map_odd_status: PASS`。
- `map_name` 必须来自能被 nuPlan map API 初始化并完成真实 lane / lane_connector 查询验证的候选；弱候选只有通过真实查询后才可接受。

#### Common failure modes

- map root 路径错误：检查 `--nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps`。
- DB root 路径错误或 split 不匹配：检查 `--nuplan_db_root .../nuplan-v1.1/splits/mini` 与 `--split mini`。
- map candidate 无法通过真实查询验证：不要 silent fallback；应在 `warnings.json` 中记录候选失败原因。
- 行数不一致：先检查 Stage 7B.2 `metadata.csv` / shard manifest 是否与当前输入目录一致。

---

### Stage 7B.4 — Merge dynamic context and map/ODD

#### Purpose

合并 Stage 7B.2 dynamic context features 与 Stage 7B.3 map/ODD features，输出 Stage 7C simulation wrapper 使用的 merged context 目录。该步骤不运行 planner simulation、不运行 BDD、不训练。

#### Command

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

python tools/stage7b4_merge_dynamic_map_context.py \
  --dynamic_context_dir outputs/stage7A_nuplan/expert_context_dataset \
  --map_odd_dir outputs/stage7b3_nuplan_map_odd \
  --output_dir outputs/stage7b4_nuplan_context_merged \
  --overwrite
```

#### Expected output files

```text
outputs/stage7b4_nuplan_context_merged/
├── merged_context_feat.npy
├── merged_metadata.csv
├── merged_feature_schema.json
├── alignment_report.md
├── warnings.json
├── ego_seq.npy
├── neighbor_seq.npy
├── context_traj.npy
├── context_mask.npy
├── dynamic_feat_style.npy
└── map_odd_feat.npy
```

#### Expected shape / key metrics

Latest verified Stage 7B.4 result:

```text
merged_context_feat shape: [23, 70]
alignment_keys:
  - db_name
  - scene_token
  - sample_id
  - start_frame_index
  - end_frame_index
row_order_already_aligned: true
warnings: 0
status: PASS
```

Related context shapes:

```text
ego_seq.npy: [23, 80, 8]
neighbor_seq.npy: [23, 5, 80, 15]
context_traj.npy: [23, 80, 83]
context_mask.npy: [23, 80, 5]
dynamic_feat_style.npy: [23, 33]
map_odd_feat.npy: [23, 37]
merged_context_feat.npy: [23, 70]
```

#### PASS criteria

- `merged_context_feat.npy` 存在，shape 为 `[23, 70]`（当前 mini smoke），列数等于 dynamic 33 + map/ODD 37。
- `merged_metadata.csv` 行数等于 dynamic rows。
- `alignment_report.md` 显示 `status: PASS`，并列出所选强 alignment keys、候选 key sets、row order / reindexing 结果。
- `warnings.json` 中 warnings 数为 0，且包含 alignment / feature-name / finite validation。
- `merged_feature_schema.json` 包含真实 feature names，以及 `dynamic::` / `map_odd::` 前缀的 merged names 和 feature slices。
- 所有导出数组 `ego_seq.npy`、`neighbor_seq.npy`、`context_traj.npy`、`context_mask.npy`、`dynamic_feat_style.npy`、`map_odd_feat.npy`、`merged_context_feat.npy` 均无 NaN/Inf。

#### Common failure modes

- dynamic 与 map/ODD 行数不一致：重跑 Stage 7B.3，确认使用同一个 Stage 7B.2 dynamic context 输入。
- 强 key 不唯一或字段缺失：检查 `merged_metadata.csv` / `map_odd_meta.csv` 中 `db_name`、`scene_token`、`sample_id`、`start_frame_index`、`end_frame_index`。
- feature schema 长度不等于数组列数：不要生成 fallback feature names，应修复上游 schema。
- Stage 7B.4 `scene_token` 仅是 Stage 7B metadata token；它不保证等于 nuPlan `scenario_filter.scenario_tokens`。exact rerun 见 Stage 7C.1C。

---

### Stage 7C.1 — Official nuPlan simulation smoke

#### Purpose

验证官方 nuPlan simulation → `simulation_log/*.msgpack.xz` → parser → `simulated_ego_seq.npy` export 全链路。No pseudo rollout is allowed。

#### Command

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c1_nuplan_simulation_smoke \
  --planners simple_planner \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents planner=simple_planner scenario_builder=nuplan_mini scenario_filter=one_of_each_scenario_type scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7c1_smoke job_name=stage7c1_simple_planner output_dir={output_dir}' \
  --overwrite
```

#### Expected output files

```text
outputs/stage7c1_nuplan_simulation_smoke/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── scenario_alignment_report.md
├── scenario_alignment.json
├── scenario_alignment.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/
```

#### Expected shape / key metrics

Latest smoke PASS metrics:

```text
warnings: []
validation.pass: true
official_success_count: 1
trajectory_rows: 150
pseudo_rollout: false
uses_official_nuplan_simulation: true
simulated_ego_seq.npy shape: [1, 1, 150, 8]
simulated_ego_seq_mask.npy shape: [1, 1, 150]
valid_timestep_count: 150
msgpack_simulation_log_files_found: 1
msgpack_simulation_log_files_parsed: 1
msgpack_trajectory_rows_extracted: 150
required_pose_valid_ratio: 1.0
x/y/yaw non-sentinel ratios: 1.0 / 1.0 / 1.0
```

#### PASS criteria

- `warnings.json` has no fatal warnings。
- `validation.pass == true`。
- `official_success_count >= 1`。
- `uses_official_nuplan_simulation == true`。
- `pseudo_rollout == false`。
- `msgpack_simulation_log_files_parsed >= 1`。
- `trajectory_rows > 0`。
- `simulated_ego_seq.npy` 是四维 `[N, P, T, 8]`。
- `simulated_ego_seq_mask.npy` shape 等于 `[N, P, T]`。
- `required_pose_valid_ratio == 1.0`。

#### Common failure modes

- 不要在 `experiment_name` / `job_name` 中使用 raw `{scenario_id}`；raw scenario id 可能包含 `|`，shell mode 下会破坏命令。优先使用固定名称或 `{scenario_id_safe}`。
- nuPlan CLI 失败：检查环境变量、DB root、map root、planner 名称和 Hydra config。
- `msgpack.xz` 找不到或解析不到 trajectory：不能改用 pseudo rollout；应修复 official output 路径或 parser。
- required pose field 缺失：`x`、`y`、`yaw` 不允许 silent sentinel fallback。

---

### Stage 7C.1C — Exact scenario alignment / exact-token smoke

#### Purpose

验证 nuPlan 可以通过 exact `log_name + actual nuPlan scenario token` 重新运行目标 scenario。同时明确：Stage 7B.4 `scene_token` 不一定等于 nuPlan `scenario_filter.scenario_tokens`。

#### Important verified evidence

```text
Stage 7B.4 target_log_name:
2021.05.12.22.00.38_veh-35_01008_01518

Stage 7B.4 scene_token:
165060762e765a5a

Actual nuPlan scenario token discovered from runner_report/msgpack path:
000e00790bc45da7
```

#### Command

先用 native log-only command 发现 actual nuPlan token：

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python -m nuplan.planning.script.run_simulation \
  +simulation=closed_loop_nonreactive_agents \
  planner=simple_planner \
  scenario_builder=nuplan_mini \
  scenario_filter=all_scenarios \
  'scenario_filter.log_names=["2021.05.12.22.00.38_veh-35_01008_01518"]' \
  scenario_filter.scenario_tokens=null \
  scenario_filter.limit_total_scenarios=1 \
  worker=single_machine_thread_pool \
  experiment_name=stage7c1_exact_log_only_native \
  job_name=stage7c1_exact_log_only_simple_planner \
  output_dir=$NUPLAN_EXP_ROOT/stage7c1_exact_log_only_native
```

Expected discovery:

```text
runner_report.parquet:
  succeeded: True
  scenario_name: 000e00790bc45da7
  log_name: 2021.05.12.22.00.38_veh-35_01008_01518

simulation_log path:
  simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.22.00.38_veh-35_01008_01518/000e00790bc45da7/000e00790bc45da7.msgpack.xz
```

Wrapper exact-token smoke：

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c1_nuplan_simulation_exact_token_smoke \
  --planners simple_planner \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents planner=simple_planner scenario_builder=nuplan_mini scenario_filter=all_scenarios scenario_filter.log_names=["2021.05.12.22.00.38_veh-35_01008_01518"] scenario_filter.scenario_tokens=["000e00790bc45da7"] scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7c1_exact_token_smoke job_name=stage7c1_exact_token_simple_planner output_dir={output_dir}' \
  --overwrite
```

#### Expected output files

与 Stage 7C.1 smoke 相同，输出到：

```text
outputs/stage7c1_nuplan_simulation_exact_token_smoke/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── scenario_alignment_report.md
├── scenario_alignment.json
├── scenario_alignment.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/
```

#### Expected shape / key metrics

Latest exact-token wrapper PASS metrics:

```text
warnings: []
validation.pass: true
official_success_count: 1
trajectory_rows: 149
pseudo_rollout: false
uses_official_nuplan_simulation: true
same_scenario_alignment_required: false
smoke_pass: true
simulated_ego_seq.npy shape: [1, 1, 149, 8]
simulated_ego_seq_mask.npy shape: [1, 1, 149]
valid_timestep_count: 149
msgpack_simulation_log_files_found: 1
msgpack_simulation_log_files_parsed: 1
msgpack_trajectory_rows_extracted: 149
required_pose_valid_ratio: 1.0
x/y/yaw non-sentinel ratios: 1.0 / 1.0 / 1.0
```

Alignment semantics:

```text
same_log_alignment_passed: true
stage7b_scene_token_match: false
actual_nuplan_scenario_token_available: true
exact_nuplan_token_rerun_supported: true
alignment_status: PASS_LOG_AND_NUPLAN_TOKEN_RERUN
```

#### PASS criteria

- official command succeeds。
- `msgpack.xz` is parsed。
- target `log_name` equals actual `log_name`。
- actual nuPlan scenario token is available。
- exact rerun with `log_name + actual_nuPlan_scenario_token` succeeds。
- `pseudo_rollout == false`。

#### Common failure modes

- `No scenarios found to simulate`：通常是 `scenario_filter.log_names` 或 `scenario_filter.scenario_tokens` 不匹配；先用 log-only filtering 发现 actual token。
- 不要假设 Stage 7B.4 `scene_token == nuPlan scenario_filter.scenario_tokens`。exact rerun 必须使用从 `runner_report.parquet` 或 `simulation_log` path 提取的 `actual_nuPlan_scenario_token`。
- exact token command 中 quote/escape 错误：保持上面 bash block 的单引号/双引号格式。

---

### Stage 7C.2A — simple_planner × 3 distinct logs

#### Purpose

从 1 scenario × 1 planner 扩展到多个 distinct logs 的 official nuPlan simulation，并验证 multi-scenario tensor output `[N, P, T, C]`。

#### Important metadata fact

```text
Stage 7B.4 merged_metadata.csv:
rows: 23
unique db_name: 5

db_name distribution:
2021.05.12.22.00.38_veh-35_01008_01518.db    5
2021.05.12.22.28.35_veh-35_00620_01164.db    5
2021.05.12.23.36.44_veh-35_00152_00504.db    5
2021.05.12.23.36.44_veh-35_01133_01535.db    4
2021.05.12.23.36.44_veh-35_02035_02387.db    4
```

使用 `--max_scenarios 3` alone 会选择前 3 行，但它们都属于同一个 `db_name`。Stage 7C.2A 必须使用 `--sample_distinct_log_names`，先按 normalized log name（`db_name` 去掉 `.db`）去重，再选择每个 log 的第一行。

#### Command

下面命令已经修正 output_dir 路径，使用 `/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/...`，不要使用误写的 `/home/forwardxp/00_nuplan_E2E_evaluation/...`。

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c2a_simple_planner_3logs \
  --planners simple_planner \
  --sample_distinct_log_names \
  --max_scenarios 3 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents planner=simple_planner scenario_builder=nuplan_mini scenario_filter=all_scenarios scenario_filter.log_names=["{target_log_name}"] scenario_filter.scenario_tokens=null scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7c2a_simple_planner job_name=stage7c2a_simple_planner output_dir={output_dir}' \
  --overwrite
```

#### Expected output files

```text
outputs/stage7c2a_simple_planner_3logs/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── scenario_alignment_report.md
├── scenario_alignment.json
├── scenario_alignment.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/
```

#### Expected shape / key metrics

Expected sampling diagnostics:

```json
{
  "scenario_sampling": {
    "original_metadata_rows": 23,
    "unique_log_names": 5,
    "sample_distinct_log_names": true,
    "selected_metadata_rows": 3,
    "selected_sample_ids": ["sample_000000", "sample_000005", "sample_000010"],
    "selected_log_names": [
      "2021.05.12.22.00.38_veh-35_01008_01518",
      "2021.05.12.22.28.35_veh-35_00620_01164",
      "2021.05.12.23.36.44_veh-35_00152_00504"
    ]
  }
}
```

Expected tensor shape:

```text
simulated_ego_seq.npy shape: [3, 1, T, 8] or [N_success, 1, T, 8] with N_success >= 1
simulated_ego_seq_mask.npy shape: [3, 1, T] or [N_success, 1, T]
pseudo_rollout: false
uses_official_nuplan_simulation: true
```

#### PASS criteria

- warnings has no `nuplan_cli_failed`。
- `official_success_count >= 3` for the intended full 3-log pass。
- `msgpack_simulation_log_files_found >= 3`。
- `msgpack_simulation_log_files_parsed >= 3`。
- `simulated_ego_seq.npy` shape 为 `[3, 1, T, 8]`；若部分 log 因环境问题失败，最低 smoke 记录可为 `[N_success, 1, T, 8]` 且 `N_success >= 1`，但不能宣称 full 3-log PASS。
- `simulated_ego_seq_mask.npy` shape 为 `[3, 1, T]` 或 `[N_success, 1, T]`。
- `pseudo_rollout == false`。
- `uses_official_nuplan_simulation == true`。
- 每个 successful record 的 `same_log_alignment_passed == true`。

#### Diagnostic commands after run

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

cat outputs/stage7c2a_simple_planner_3logs/simulation_report.md
cat outputs/stage7c2a_simple_planner_3logs/warnings.json
cat outputs/stage7c2a_simple_planner_3logs/scenario_alignment_report.md

python - <<'PY'
import json
import numpy as np
from pathlib import Path

base = Path("outputs/stage7c2a_simple_planner_3logs")
seq = np.load(base / "simulated_ego_seq.npy")
mask = np.load(base / "simulated_ego_seq_mask.npy")
print("seq shape:", seq.shape)
print("mask shape:", mask.shape)
print("valid timesteps:", int(mask.sum()))
print("finite:", bool(np.isfinite(seq).all()))

schema = json.loads((base / "simulation_schema.json").read_text())
print("uses_official_nuplan_simulation:", schema.get("uses_official_nuplan_simulation"))
print("pseudo_rollout:", schema.get("pseudo_rollout"))
print("sample_distinct_log_names:", schema.get("sample_distinct_log_names"))
print("selected_log_names:", schema.get("selected_log_names"))

align = json.loads((base / "scenario_alignment.json").read_text())
for r in align.get("records", []):
    print(r.get("scenario_index"), r.get("planner_name"), r.get("target_log_name"), "->", r.get("actual_log_name"), r.get("actual_nuplan_scenario_token"), r.get("alignment_status"))
PY

find outputs/stage7c2a_simple_planner_3logs/official_nuplan_runs \
  -maxdepth 8 -type f | sort | grep -E "msgpack|runner_report|nuplan_cli|log.txt"
```

#### Common failure modes

- repeated same log in multi-scenario run：确认命令包含 `--sample_distinct_log_names`。
- `No scenarios found to simulate`：先用 Stage 7C.1C log-only filtering 发现 actual token；不要直接用 Stage 7B.4 `scene_token` 当 nuPlan token。
- output path 拼写错误：正确路径是 `/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c2a_simple_planner_3logs`。
- 部分 log 成功、部分 log 失败：可以记录 smoke evidence，但不要写成 full 3-log PASS，除非 `official_success_count >= 3`。
- pseudo rollout accidentally introduced：Stage 7C.2A 必须保持 `pseudo_rollout=false`。

---

### Stage 7C.2B — simple_planner × 5 distinct logs（PASS）

#### Purpose

在 Stage 7C.2A 的 3 个 distinct logs 通过后，扩展到 Stage 7B.4 mini context 中全部 5 个 distinct logs，验证 official nuPlan simulation → msgpack parser → `[N, P, T, C]` tensor export 在多 log 条件下稳定工作。该阶段仍然只使用 `simple_planner`，不引入 Stage 7D BDD，也不允许 pseudo rollout。

#### Command

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c2b_simple_planner_5logs \
  --planners simple_planner \
  --sample_distinct_log_names \
  --max_scenarios 5 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents planner=simple_planner scenario_builder=nuplan_mini scenario_filter=all_scenarios scenario_filter.log_names=["{target_log_name}"] scenario_filter.scenario_tokens=null scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7c2b_simple_planner job_name=stage7c2b_simple_planner output_dir={output_dir}' \
  --overwrite
```

#### Expected output files

```text
outputs/stage7c2b_simple_planner_5logs/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── scenario_alignment_report.md
├── scenario_alignment.json
├── scenario_alignment.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/
```

#### Expected shape / key metrics

Latest verified Stage 7C.2B PASS metrics:

```text
warnings: []
original_metadata_rows: 23
unique_log_names: 5
sample_distinct_log_names: true
selected_metadata_rows: 5
selected_sample_ids: sample_000000, sample_000005, sample_000010, sample_000015, sample_000019
validation.pass: true
official_success_count: 5
trajectory_rows: 745
pseudo_rollout: false
uses_official_nuplan_simulation: true
same_scenario_alignment_required: true
strict_nuplan_token_alignment_required: false
simulated_ego_seq.npy shape: [5, 1, 149, 8]
simulated_ego_seq_mask.npy shape: [5, 1, 149]
valid_timestep_count: 745
missing_pair_count: 0
msgpack_simulation_log_files_found: 5
msgpack_simulation_log_files_parsed: 5
msgpack_trajectory_rows_extracted: 745
alignment_pass_ratio: 1.0
same_log_alignment_passed: true
required_pose_valid_ratio: 1.0
x/y/yaw non-sentinel ratios: 1.0 / 1.0 / 1.0
min_timesteps_per_trajectory: 149
mean_timesteps_per_trajectory: 149.0
num_trajectories_with_too_few_steps: 0
num_trajectories_with_zero_motion: 0
```

Parsed official nuPlan artifacts:

```text
scenario_0/simple_planner/simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.22.00.38_veh-35_01008_01518/000e00790bc45da7/000e00790bc45da7.msgpack.xz
scenario_1/simple_planner/simulation_log/SimplePlanner/stationary_in_traffic/2021.05.12.22.28.35_veh-35_00620_01164/001f3d5282985bbb/001f3d5282985bbb.msgpack.xz
scenario_2/simple_planner/simulation_log/SimplePlanner/traversing_traffic_light_intersection/2021.05.12.23.36.44_veh-35_00152_00504/00015fc2840d5313/00015fc2840d5313.msgpack.xz
scenario_3/simple_planner/simulation_log/SimplePlanner/traversing_intersection/2021.05.12.23.36.44_veh-35_01133_01535/0004544fe3715b27/0004544fe3715b27.msgpack.xz
scenario_4/simple_planner/simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.23.36.44_veh-35_02035_02387/0004bf5585cf5f26/0004bf5585cf5f26.msgpack.xz
```

#### PASS criteria

- `warnings == []` 或没有 fatal warning。
- `validation.pass == true`。
- `official_success_count == 5`。
- `uses_official_nuplan_simulation == true`。
- `pseudo_rollout == false`。
- `sample_distinct_log_names == true`，并且 selected log names 覆盖 5 个 distinct logs。
- `same_scenario_alignment_required == true`，每条成功记录至少满足 same-log alignment。
- `strict_nuplan_token_alignment_required == false`；不要要求 Stage 7B.4 `scene_token` 等于 nuPlan `scenario_filter.scenario_tokens`。
- `msgpack_simulation_log_files_found == 5` 且 `msgpack_simulation_log_files_parsed == 5`。
- `simulated_ego_seq.npy` shape 为 `[5, 1, 149, 8]`。
- `simulated_ego_seq_mask.npy` shape 为 `[5, 1, 149]`。
- `required_pose_valid_ratio == 1.0`，`x/y/yaw` 非 sentinel 比例均为 `1.0`。

#### Common failure modes

- 忘记 `--sample_distinct_log_names`：会重复抽到同一个 log 的多行，不能作为 5 distinct logs PASS。
- 把 Stage 7B.4 `scene_token` 当成 nuPlan `scenario_filter.scenario_tokens`：这是错误假设。Stage 7B.4 `scene_token != nuPlan scenario_filter.scenario_tokens` 是已验证 caveat；exact rerun 必须先从 `runner_report.parquet` 或 `simulation_log` 路径发现 actual nuPlan token。
- `No scenarios found to simulate`：通常是 log name / token filter 不匹配；先回到 Stage 7C.1C 的 log-only discovery。
- `msgpack.xz` 找不到或解析不到 trajectory：不能引入 pseudo rollout，应修复 official output path 或 parser。
- 部分 log 失败：只能记录 partial smoke，不能宣称 Stage 7C.2B PASS。

---

### Stage 7C.2C — IDM longitudinal-only multi-planner rollout（PASS）

#### Purpose

从 `simple_planner × 5 distinct logs` 扩展到 `simple_planner + IDM longitudinal profiles` 的 multi-planner official rollout，形成 `[N, P, T, 8]` official trajectory tensor。Stage 7C.2C 只准备和运行 planner simulation；不要实现 Stage 7D BDD validation。

The native IDM profile smoke tests prove that official nuPlan simulation can run parameterized IDM profiles. However, these profiles are longitudinal-only and are not complete driving-style models.

IDM profiles are longitudinal-only rule-based positive controls. They should not be described as complete conservative / comfort / aggressive driving styles. They cover following, lead-brake response, queue approach, and partial longitudinal components of cut-in/yield conflicts. They do not cover lane-change willingness, lane-change sharpness, overtaking execution, hesitation, target-lane rear-gap pressure, or full courtesy/yield behavior.

We first validate whether BDD can detect controlled longitudinal behavior drift using parameterized IDM profiles in official nuPlan simulation. Lateral and interaction style dimensions will be evaluated later through PDM or another lane-change-capable planner/E2E policy.

#### Stage 7C.2C status

```text
Stage 7C.2B — simple_planner × 5 distinct logs: PASS
Stage 7C.2C-0 — native IDM default/conservative/comfort/aggressive smoke: PASS
Stage 7C.2C-1 — wrapper smoke: 1 log × 4 planners: PASS
Stage 7C.2C-2 — wrapper rollout: 5 logs × 4 planners: PASS
Stage 7C.3 — PDM lateral/interaction planner extension: TODO
```

Stage 7C.2C-0 verified native IDM result:

```text
stage7c2c0_idm_default_native: succeeded=True
stage7c2c0_idm_conservative_native: succeeded=True
stage7c2c0_idm_comfort_native: succeeded=True
stage7c2c0_idm_aggressive_native: succeeded=True
```

#### Planner config discovery

已确认 wrapper 应使用以下本机 nuPlan IDM Hydra override key：

```text
planner=idm_planner
planner.idm_planner.target_velocity
planner.idm_planner.min_gap_to_lead_agent
planner.idm_planner.headway_time
planner.idm_planner.accel_max
planner.idm_planner.decel_max
```

#### IDM profile definitions（已写入 wrapper）

```text
simple_planner:
  planner_type: simple_baseline
  policy_style: simple_baseline
  hydra_overrides: planner=simple_planner

idm_longitudinal_conservative:
  planner_type: idm_rule_based
  policy_style: longitudinal_conservative
  style_scope: longitudinal_only
  nuplan_planner_config: idm_planner
  hydra_overrides:
    planner=idm_planner
    planner.idm_planner.target_velocity=8.0
    planner.idm_planner.min_gap_to_lead_agent=2.0
    planner.idm_planner.headway_time=2.0
    planner.idm_planner.accel_max=0.8
    planner.idm_planner.decel_max=2.5
  supported_behavior_tasks: [following, lead_brake_response, queue_approach, cutin_response_partial, yield_conflict_partial]
  unsupported_behavior_tasks: [lane_change, overtake_execution, hesitation, target_lane_gap_acceptance, rear_pressure_lane_change]

idm_longitudinal_comfort:
  planner_type: idm_rule_based
  policy_style: longitudinal_comfort
  style_scope: longitudinal_only
  nuplan_planner_config: idm_planner
  hydra_overrides:
    planner=idm_planner
    planner.idm_planner.target_velocity=10.0
    planner.idm_planner.min_gap_to_lead_agent=1.5
    planner.idm_planner.headway_time=1.5
    planner.idm_planner.accel_max=1.0
    planner.idm_planner.decel_max=3.0
  supported_behavior_tasks: [following, lead_brake_response, queue_approach, cutin_response_partial, yield_conflict_partial]
  unsupported_behavior_tasks: [lane_change, overtake_execution, hesitation, target_lane_gap_acceptance, rear_pressure_lane_change]

idm_longitudinal_aggressive:
  planner_type: idm_rule_based
  policy_style: longitudinal_aggressive
  style_scope: longitudinal_only
  nuplan_planner_config: idm_planner
  hydra_overrides:
    planner=idm_planner
    planner.idm_planner.target_velocity=12.0
    planner.idm_planner.min_gap_to_lead_agent=0.5
    planner.idm_planner.headway_time=1.0
    planner.idm_planner.accel_max=1.5
    planner.idm_planner.decel_max=4.0
  supported_behavior_tasks: [following, lead_brake_response, queue_approach, cutin_response_partial, yield_conflict_partial]
  unsupported_behavior_tasks: [lane_change, overtake_execution, hesitation, target_lane_gap_acceptance, rear_pressure_lane_change]
```

旧别名 `idm_conservative`、`idm_comfort`、`idm_aggressive` 仅用于兼容，文档和默认示例必须使用显式 `idm_longitudinal_*` 名称。

#### Wrapper placeholder

wrapper command template 支持 `{planner_hydra_overrides}` placeholder；每个 planner 运行时自动展开为对应 Hydra override fragment，模板不得硬编码 `planner=simple_planner`。

Examples:

```text
simple_planner:
planner=simple_planner

idm_longitudinal_conservative:
planner=idm_planner planner.idm_planner.target_velocity=8.0 planner.idm_planner.min_gap_to_lead_agent=2.0 planner.idm_planner.headway_time=2.0 planner.idm_planner.accel_max=0.8 planner.idm_planner.decel_max=2.5
```

#### Stage 7C.2C-1 command（wrapper smoke: 1 log × 4 planners，已 PASS）

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7c2c1_idm_longitudinal_1log \
  --planners simple_planner idm_longitudinal_conservative idm_longitudinal_comfort idm_longitudinal_aggressive \
  --sample_distinct_log_names \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios scenario_filter.log_names=["{target_log_name}"] scenario_filter.scenario_tokens=null scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7c2c_idm_longitudinal job_name=stage7c2c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

Expected output:

```text
simulated_ego_seq.npy shape: [1, 4, T, 8]
simulated_ego_seq_mask.npy shape: [1, 4, T]
official_success_count: 4
msgpack_simulation_log_files_parsed: 4
pseudo_rollout: false
uses_official_nuplan_simulation: true
```

#### Stage 7C.2C-2 result（wrapper rollout: 5 logs × 4 planners，PASS）

输出目录：

```text
outputs/stage7c2c2_idm_longitudinal_5logs
```

Planner axis：

```text
0 simple_planner
1 idm_longitudinal_conservative
2 idm_longitudinal_comfort
3 idm_longitudinal_aggressive
```

最新 PASS metrics：

```text
warnings: []
official_success_count: 20
trajectory_rows: 2980
msgpack_simulation_log_files_found: 20
msgpack_simulation_log_files_parsed: 20
msgpack_trajectory_rows_extracted: 2980
simulated_ego_seq.npy shape: [5, 4, 149, 8]
simulated_ego_seq_mask.npy shape: [5, 4, 149]
valid_timestep_count: 2980
missing_pair_count: 0
pseudo_rollout: false
uses_official_nuplan_simulation: true
alignment_pass_ratio: 1.0
same_log_alignment_passed: true
strict_stage7b_scene_token_match: false
alignment_level: log_name_plus_actual_nuplan_token
```

通过含义：Stage 7C.2C-2 已经形成 official nuPlan simulation 生成的 `[5, 4, 149, 8]` multi-planner tensor，可作为 Stage 7D 的输入。IDM profiles 是 longitudinal-only rule-based positive controls for BDD validation，不是完整 conservative / comfort / aggressive driving-style models。

#### Expected output files and metadata

Stage 7C.2C wrapper outputs should keep the same schema as Stage 7C.2B. `simulated_planner_metadata.csv`, `warnings.json`, and `simulation_schema.json` should expose planner metadata fields:

```text
planner_name
planner_id
planner_class
planner_type
policy_style
style_scope
nuplan_planner_config
hydra_overrides
supported_behavior_tasks
unsupported_behavior_tasks
parameters_json
```

```text
outputs/stage7c2c_*/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── scenario_alignment_report.md
├── scenario_alignment.json
├── scenario_alignment.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/
```

#### Expected shape / key metrics

- Stage 7C.2C-1 wrapper smoke：期望 shape 为 `[1, 4, T, 8]`，其中 planner 维度对应 `simple_planner + idm_longitudinal_conservative + idm_longitudinal_comfort + idm_longitudinal_aggressive`。
- Stage 7C.2C-2 5-log rollout：期望 shape 为 `[5, 4, T, 8]`，其中每个 scenario-planner pair 都必须成功。

#### Stage 7C.3 — PDM lateral/interaction planner extension（TODO）

Stage 7C.3 暂不实现。后续通过 PDM 或其他 lane-change-capable planner / E2E policy，从 longitudinal style 扩展到 lateral、lane-change、overtaking、hesitation、rear-gap pressure、interaction/yielding style。

#### PASS criteria

- 所有成功记录均来自 official nuPlan simulation log，不允许 pseudo rollout。
- `uses_official_nuplan_simulation == true`。
- `pseudo_rollout == false`。
- same-log alignment 通过；strict Stage 7B token alignment 不作为必需条件。
- 每个 successful scenario-planner pair 至少有 `min_timesteps` 个有效 timestep。
- wrapper 输出的 `missing_pair_count == 0` 才能宣称 full multi-planner PASS；否则只能记录 partial smoke。

#### Common failure modes

- `planner=idm_planner` 找不到：检查 nuPlan devkit 安装、Hydra config search path、`nuplan-devkit` 是否在当前 Python 环境。
- IDM override key 写错：必须读取本机 `idm_planner.yaml`，不要猜参数名。
- wrapper profile 名称和 Hydra planner 名称混淆：`idm_longitudinal_conservative` 等是本项目 profile ID，nuPlan 原生 planner config 仍是 `planner=idm_planner`；`idm_conservative` 等旧名称仅是兼容 alias。
- 非 full pair 输出：如果 `[N, P]` 中有 scenario-planner pair 缺失，不能宣称 Stage 7C.2C full PASS。
- 不要为了补齐 planner 维度而生成 offline pseudo trajectory。

---

### Stage 7D — 完整 Stage 6-compatible 数据导出

#### Purpose

Stage 7D 的方向已修正：Stage 7D 不再是单独的最终 BDD pipeline，而是把 Stage 7C official nuPlan planner rollout 导出为完整 Stage 6-compatible sharded dataset。Stage 6 仍然是 canonical BDD / report-card / task-conditioned BDD 引擎；Stage 7E/F 后续复用 Stage 6 模块。`tools/stage7d_validate_official_planner_bdd.py` 只作为 smoke diagnostic，不是 canonical final BDD path。

#### Command

```bash
python tools/stage7d_export_stage6_compatible_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7d_stage6_dataset_official_planner_5logs \
  --required_planners simple_planner idm_longitudinal_conservative idm_longitudinal_comfort idm_longitudinal_aggressive \
  --overwrite
```

#### Expected output files

```text
outputs/stage7d_stage6_dataset_official_planner_5logs/
  shard_manifest.json
  feature_schema.json
  planner_policy_indices/
    simple_planner.npy
    idm_longitudinal_conservative.npy
    idm_longitudinal_comfort.npy
    idm_longitudinal_aggressive.npy
  shards/
    shard_000/
      ego_seq.npy
      neighbor_seq.npy
      neighbor_slot_ids.npy
      interaction_feat_style.npy
      metadata.csv
  stage7d_export_schema.json
  warnings.json
  export_report.md
```

#### Expected shape / key metrics

输入 tensor 使用：

```text
outputs/stage7c2c2_idm_longitudinal_5logs/simulated_ego_seq.npy: [5, 4, 149, 8]
outputs/stage7c2c2_idm_longitudinal_5logs/simulated_ego_seq_mask.npy: [5, 4, 149]
```

输出行语义为 one row = one scenario × one planner-controlled nuPlan ego rollout；当前 5 logs × 4 planners 必须导出 20 行，不能导出 5 logs × 4 planners × num_agents。Stage 5 / Stage 6 Waymo 预处理可以为了数据量把多个 road participants 展开为 ego-like samples，但 Stage 7 official nuPlan planner 数据不能这样做：IDM / PDM / ML Planner 只控制 nuPlan ego vehicle，background agents 必须保留为 neighbor context。`ego_seq.npy` channel 必须为 `[x, y, vx, vy, heading, speed, accel, yaw_rate]`。`neighbor_seq.npy` 和 `neighbor_slot_ids.npy` 是 mandatory，不允许作为 optional。

#### PASS criteria

- 只读取 official nuPlan simulation outputs，不运行 nuPlan simulation，不读取 pseudo rollout。
- `pseudo_rollout == false` 且 `uses_official_nuplan_simulation == true`。
- `ego_seq.npy`、`neighbor_seq.npy`、`neighbor_slot_ids.npy`、`interaction_feat_style.npy`、`metadata.csv`、`feature_schema.json`、`shard_manifest.json` 全部存在。
- `planner_policy_indices` 下四个 planner `.npy` 全部存在。
- `neighbor_seq.npy` 或 `neighbor_slot_ids.npy` 缺失时必须 fail，不能写成 `neighbor_seq_missing` non-fatal warning。
- `ego_seq.npy`、`neighbor_seq.npy`、`interaction_feat_style.npy`、`metadata.csv` 行数一致且等于 `N * P`，不得按 neighbor/background agent 数量扩行。
- `stage7d_export_schema.json` 记录 `row_semantics = "scenario_planner_controlled_ego_rollout"`、`ego_definition = "nuPlan planner-controlled ego vehicle only"`、`neighbor_definition = "background road participants used only as context"`、`multi_agent_ego_expansion = false`、`total_rows_expected = num_scenarios * num_planners`。
- `warnings.json.validation.total_rows == num_scenarios * num_planners`，且 `no_multi_agent_ego_expansion == true`、`neighbor_agents_used_as_context_only == true`。
- `feature_schema.json` 明确列出 interaction/style feature names and indices。

#### Common failure modes

- 只有 ego trajectory，没有 surrounding-agent neighbor context：Stage 7D 必须 fail。
- 把 `tools/stage7d_validate_official_planner_bdd.py` 的 smoke diagnostic 当成 final BDD：不允许。
- 重新实现 Stage 7 final BDD pipeline：不允许；后续 Stage 7E/F 必须复用 Stage 6 BDD/report-card/task-conditioned BDD 模块。

---

### Stage 7 common failure modes

1. `No scenarios found to simulate`
   - 通常是 `scenario_filter.log_names` 或 `scenario_filter.scenario_tokens` 不匹配。
   - 先运行 log-only filtering。
   - 从 `runner_report.parquet` 或 `simulation_log/<Planner>/<type>/<log_name>/<scenario_token>/<scenario_token>.msgpack.xz` 路径发现 actual nuPlan token。

2. `Stage7B scene_token mismatch`
   - 不一定是失败。
   - Stage 7B.4 `scene_token` may not equal nuPlan `scenario_filter.scenario_tokens`。
   - exact rerun 使用 `log_name + actual_nuPlan_scenario_token`。

3. `raw scenario_id breaks shell`
   - raw `scenario_id` 可能包含 `|`。
   - 使用 `{scenario_id_safe}` 或避免在 shell/path command fields 中使用 raw scenario id。
   - 默认优先让 wrapper 使用 `subprocess.run(argv, shell=False)`；只有确实需要 shell 语义时才使用 shell mode。

4. `Repeated same log in multi-scenario run`
   - `--max_scenarios 3` 默认只取 metadata 前 3 行，可能全部来自同一 log。
   - Stage 7C.2A 使用 `--sample_distinct_log_names`。

5. `Pseudo rollout accidentally introduced`
   - Stage 7C 必须只使用 official nuPlan simulation。
   - `simulation_schema.json` 必须记录 `uses_official_nuplan_simulation=true` 和 `pseudo_rollout=false`。
   - 如果 official CLI 或 parser 失败，应报告 FAIL diagnostics，不允许用 numpy interpolation 或 expert trajectory rewriting 伪造成功。


## 关键参数

### Stage 5B 训练性能/内存排查（GPU 利用率低）
- GPU 利用率低通常不是模型太慢，而是 **DataLoader / CPU / 内存瓶颈**。
- 优先使用 `mmap` 方式加载 shard（避免一次性将大 `.npy` 完整读入 RAM）。
- 建议先从 `batch_size=64` 开始稳定跑通，再逐步调大。
- 可先尝试：`--num_workers 2 --pin_memory` 提升主机到 GPU 的喂数效率。
- 如果系统 RAM 已经很高，优先保持 `--num_workers 0 --cache_shards 1`，降低并发加载压力。
- 训练日志出现 `Killed` 通常是 **系统 RAM OOM**，而不是 CUDA OOM。

### build_dataset.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tfrecord_glob` | `/mnt/d/WMdata/*.tfrecord-*` | TFRecord 文件匹配模式 |
| `--output_dir` | `output` | 输出目录 |
| `--min_ego_speed` | `5.5` | 最低自车速度过滤阈值 (m/s) |
| `--train_ratio` | `0.8` | 训练集比例 |
| `--val_ratio` | `0.1` | 验证集比例 |
| `--test_ratio` | `0.1` | 测试集比例 |
| `--limit_files` | `None` | 限制处理文件数（调试用） |

**特征维度**: 20D
- 相对速度 (3): 均值, 标差, 正向比例
- THW (3): 均值, 标差, 最小值
- Jerk (3): 均值, 标差, 95分位
- 相对加速度 (2): 均值, 标差
- 反应时间 (1)
- 横向 (3): 偏航率标差, 变道次数, 变道时长
- 速度归一化 (2): 均值, 标差
- 稳定性 (2): 速度标差, 加速度标差
- 自车速度均值 (1)

### train_embedding.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `64` | 批大小 |
| `--lr` | `1e-3` | 学习率 (AdamW) |
| `--hidden_dim` | `128` | GRU 隐层维度 |
| `--emb_dim` | `64` | embedding 输出维度 |
| `--temperature` | `0.1` | 对比损失温度 |
| `--eval_every` | `2` | 每 N 轮评估一次 |
| `--n_clusters` | `3` | KMeans 聚类数 |

### evaluate_embedding.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--k_neighbors` | `10` | 邻域一致性分析的邻居数 |
| `--umap_neighbors` | `30` | UMAP n_neighbors |
| `--umap_min_dist` | `0.1` | UMAP min_dist |
| `--umap_max_points` | `50000` | UMAP 最大样本数 |
| `--ridge_alpha` | `1.0` | 线性探针正则强度 |

## 数据维度总结

| 模块 | 输入 | 输出 |
|---|---|---|
| `build_dataset.py` | Waymo TFRecord | traj `(N,T,4)`, feat `(N,20)`, meta `(N,3)`, split `(N,)` |
| `TrajectoryEncoder` | `(B,T,4)` 轨迹 | `(B,64)` L2 归一化 embedding |
| `export_embeddings.py` | traj.npy + checkpoint | `embeddings_all.npy (N,64)` |
| `evaluate_embedding.py` | embeddings + feat + split | UMAP 图, 探针结果, 邻域分析 |

## 合成策略 Rollout（generate_policy_rollouts.py）

### lateral_stable 可分性调优
`lateral_stable` 策略被设计为**第三种独立风格**（"横向稳定 + 舒适但不保守"），与 `conservative`（大间距/低动态）和 `aggressive`（小间距/高动态）在 embedding 空间中形成明显区分。关键设计：
- **thw_target = 1.4 s**：处于 conservative (2.5 s) 和 aggressive (1.0 s) 之间，但纵向动态更软
- **jerk_limit = 0.35 m/s²/step**：比 conservative (0.5) 更软，避免纵向特征与之重合
- **yaw_rate_clip = 0.02 rad/step**：per-step heading delta clip，适度的横向约束，在 embedding 中保留横向信号
- **heading_smooth_alpha = 0.45**：EMA 平滑系数，对 desired heading 做指数移动平均
  - `0.0` = 不平滑（默认用于 conservative / aggressive），heading 直接跟随 source
  - 接近 `1.0` = 更强平滑 / 更慢更新（heading 变化极平缓，几乎不跟随 source 转向）
  - `0.45` = 中等平滑，与 conservative (0.0) 有明显差异，在 embedding 中形成独立横向风格

> **注意**：`heading_smooth_alpha` 越大，lateral_stable 的横向 heading 变化越缓慢，风格越"稳"；
> 但过大（>0.8）会导致轨迹偏移源路径，不推荐。

### generate_policy_rollouts.py 参数
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--src_traj_path` | `output/traj.npy` | 源轨迹文件 |
| `--src_front_path` | `output/front.npy` | 源前车轨迹文件 |
| `--src_split_path` | `None` | 源 split 文件（可选） |
| `--src_meta_path` | `None` | 源 meta 文件（可选） |
| `--output_dir` | `output_policy_rollouts` | 输出目录 |
| `--policies` | `conservative,aggressive,lateral_stable` | 要生成的策略列表 |
| `--dt` | `0.1` | 时间步长 (s) |
| `--seed` | `42` | 随机种子 |
| `--conservative_yaw_rate_clip` | `None` | 覆盖 conservative 的 yaw_rate_clip |
| `--aggressive_yaw_rate_clip` | `None` | 覆盖 aggressive 的 yaw_rate_clip |
| `--lateral_stable_yaw_rate_clip` | `None` | 覆盖 lateral_stable 的 yaw_rate_clip（默认 0.02） |
| `--heading_smooth_alpha` | `None` | 覆盖 lateral_stable 的 heading EMA 平滑系数（默认 0.45） |
| `--lateral_stable_thw_target` | `None` | 覆盖 lateral_stable 的 thw_target（默认 1.4 s） |
| `--lateral_stable_jerk_limit` | `None` | 覆盖 lateral_stable 的 jerk_limit（默认 0.35） |
| `--lateral_stable_a_max` | `None` | 覆盖 lateral_stable 的 a_max（默认 1.5 m/s²） |
| `--lateral_stable_a_min` | `None` | 覆盖 lateral_stable 的 a_min（默认 -2.8 m/s²） |

### 基本生成命令
```bash
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --src_split_path output/split.npy \
    --src_meta_path  output/meta.npy \
    --output_dir     output_policy_rollouts
```

### 参数扫描示例（无需修改代码）
```bash
# 调整 lateral_stable 纵向参数以提升可分性
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --output_dir     output_policy_rollouts_sweep1 \
    --lateral_stable_thw_target 1.2 \
    --lateral_stable_jerk_limit 0.25 \
    --lateral_stable_a_max 1.3 \
    --lateral_stable_a_min -2.5 \
    --lateral_stable_yaw_rate_clip 0.02 \
    --heading_smooth_alpha 0.45
```
生成后对比 `Per-policy active parameters` 摘要输出，确认参数已生效，并观察 `yaw_rate|abs|p95` 变化。

### 冒烟测试
```bash
python scripts/smoke_test_policy_rollouts.py
```
验证输出形状正确且 `lateral_stable` 的 `yaw_rate_p95` 与 `aggressive` 有显著差异。

## Aligned 评估工作流（evaluate_policy_separation_aligned.py）

### 最小复现命令（生成 → 训练 → aligned eval）

```bash
# Step 1: 生成 policy rollouts
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --src_split_path output/split.npy \
    --src_meta_path  output/meta.npy \
    --output_dir     output_policy_rollouts

# Step 2: 训练 embedding（以 policy_rollouts 数据为训练集）
python train_embedding.py \
    --traj_path  output_policy_rollouts/traj.npy \
    --feat_path  output_policy_rollouts/feat.npy \
    --split_path output_policy_rollouts/split.npy \
    --output_dir output_policy_rollouts/run_demo

# Step 3: 导出全量 embedding
python export_embeddings.py \
    --traj_path      output_policy_rollouts/traj.npy \
    --checkpoint     output_policy_rollouts/run_demo/best_model.pth \
    --output_path    output_policy_rollouts/run_demo/embeddings_all.npy

# Step 4: Aligned 评估
python evaluate_policy_separation_aligned.py \
    --embeddings_path   output_policy_rollouts/run_demo/embeddings_all.npy \
    --policy_id_path    output_policy_rollouts/policy_id.npy \
    --source_index_path output_policy_rollouts/source_index.npy \
    --split_path        output_policy_rollouts/split.npy \
    --eval_split        test \
    --analysis_dir      output_policy_rollouts/run_demo/analysis_aligned
```

### 如何用 aligned 指标验证 policy separation

`evaluate_policy_separation_aligned.py` 输出 `policy_separation_aligned_summary.json`，
关键指标解读如下：

#### (b) Within-source pairwise distances — 检查 p0_vs_p2 距离是否被拉开

```
"p0_vs_p2": {"euclidean_mean": ...}   ← lateral_stable (p2) vs conservative (p0)
"p0_vs_p1": {"euclidean_mean": ...}   ← conservative vs aggressive (应最大)
"p1_vs_p2": {"euclidean_mean": ...}   ← aggressive vs lateral_stable
```

**验证指标（generator 方向正确的信号）**：
- `p0_vs_p1` 应最大（保守 vs 激进，风格差异最大）
- `p0_vs_p2` < `p0_vs_p1` 但 > 0（p2 与 p0 有差异，说明 lateral_stable 已与 conservative 分离）
- `p0_vs_p2` 变化趋势：随着 `yaw_rate_clip` 降低或 `heading_smooth_alpha` 增大，该距离应增大

#### (c) Within-source centroid classification accuracy — 检查 policy_2 准确率

```
"centroid_classification": {
    "accuracy": ...            ← overall, 应远高于 chance (0.3333)
    "per_policy_accuracy": {
        "0": ...,              ← conservative
        "1": ...,              ← aggressive
        "2": ...               ← lateral_stable ← 重点观察
    }
}
```

**验证指标**：
- overall accuracy > 0.60（明显高于 chance=0.3333 = good）
- `policy_2` accuracy 提升是 lateral_stable 可分性改善的直接信号
- 若 `policy_2` 准确率提升但 `policy_0` 下降，说明 lateral_stable 在往 conservative 方向漂移

#### (d) Within-source retrieval applicability + margin

```
"within_source_retrieval": {
    "retrieval_mode": "within_source",
    "retrieval_applicable": false,
    "retrieval_reason": "... one sample per policy ...",
    "nearest_neighbor_hit_rate": null,
    "nearest_neighbor_chance": null,
    "mean_within_source_margin": ...      ← 应 > 0
}
```

**说明**：
- within-source 每个 source 通常只有 1 个样本/policy，因此“same-policy 最近邻命中率”在定义上可能无效。
- 当无有效 same-policy 正样本可检索时，summary.json 会明确记录 `retrieval_applicable=false`，避免误导性的 `0.0000`。
- 此时应重点看：`pairwise_distances`、`centroid_classification`、`mean_within_source_margin`。

### 冒烟测试（aligned retrieval 回归测试）
```bash
python scripts/smoke_test_aligned_retrieval.py
```
验证 coverage 的 missing/duplicate 统计、以及 within-source 检索在无正样本时会被标记为 N/A。



### Q: 如何修改特征维度?
A: 修改 `build_dataset.py` 中 `compute_features()` 函数的返回列表（当前 20D）

### Q: 如何修改 embedding 维度?
A: 通过 `train_embedding.py --emb_dim <N>` 指定，`export_embeddings.py --emb_dim <N>` 保持一致

### Q: 如何只处理少量文件调试?
A: 使用 `--limit_files 5` 参数限制读取文件数

### Q: 数据集划分如何保证可重复?
A: `assign_split()` 使用 `scenario_id` 的 MD5 哈希值确定性划分，无随机性

## 依赖库

详见 `requirements-cpu.txt`。主要依赖：
- `torch` (CPU 版)
- `tensorflow-cpu`
- `waymo-open-dataset`
- `scikit-learn`
- `umap-learn`
- `numpy`, `pandas`, `matplotlib`

## 最近重构 (2026-04-07)

✅ 重构 `build_dataset.py`：添加 CLI 参数、输出 meta/split、MD5 确定性划分、速度过滤  
✅ 新增 `dataset.py`：`TrajFeatureDataset` + 变长轨迹 collate + KNN pair 预计算  
✅ 新增 `model.py`：`TrajectoryEncoder`（GRU + MLP head，64D 输出）  
✅ 新增 `loss.py`：`SoftContrastiveLoss`（特征引导软对比）  
✅ 重构 `train_embedding.py`：适配新架构，支持 split 文件  
✅ 新增 `export_embeddings.py`：全量 embedding 导出  
✅ 新增 `evaluate_embedding.py`：UMAP + 线性探针 + 邻域一致性  
✅ 移除旧脚本：`generate_embeddings.py`, `visualize_umap.py`, `analyze_style_embedding.py`, `analysis/`, `scripts/`, `docs/`

## 最近更新 (2026-04-27)

✅ 修复 `evaluate_policy_separation_aligned.py`：within-source NN 检索在无正样本时改为显式 N/A  
　　→ 原因：within-source 每 policy 只有 1 个样本时，same-policy 最近邻定义无效  
　　→ 修复：summary.json 记录 `retrieval_mode/retrieval_applicable/retrieval_reason`，并将 hit_rate/chance 置 `null`  
✅ 新增 `scripts/smoke_test_aligned_retrieval.py`：防止 false-0.0 回归测试  
✅ 更新 `QUICK_REFERENCE.md`：  
　　→ 补充 `heading_smooth_alpha` 含义（0.0=不平滑，接近 1.0=更强平滑/更慢更新）  
　　→ 新增 aligned 评估工作流与 policy separation 验证指南（p0_vs_p2 距离、policy_2 centroid accuracy）
✅ 新增 `tools/embedding_retrieval_demo.py`：embedding 可解释性 demo（检索 + 轨迹回放）  
✅ 新增 `scripts/smoke_test_retrieval_demo.py`：检索 demo 单元/冒烟测试

## Embedding 可解释性 Demo（tools/embedding_retrieval_demo.py）

### 功能

给定一个 query 窗口（by `--query_index` 或 `--query_scenario_id`），脚本：
1. 在 embedding 空间检索 Top-K 最相似窗口（euclidean 或 cosine 距离）
2. 将 ego + front 轨迹在 query 初始坐标系下对齐叠加，输出 `traj_overlay.png`
3. 输出风格信号时序图（speed / accel / jerk / curvature proxy），输出 `timeseries.png`

### 检索模式

| 模式 | 说明 |
|------|------|
| `--mode global` | 在选定 split 的所有样本中检索 |
| `--mode within-source` | 仅检索与 query 共享相同 meta-key `(scenario_id, start, window_len, front_id)` 的其他行 |

> **within-source 限制**：基础数据集中没有显式 `policy_id` 字段。within-source 模式按
> meta-key 分组，把同组所有行（可能是不同 policy 的 rollout）全部绘出。若需要严格
> per-policy 标签，请用 `generate_policy_rollouts.py` 生成的 `policy_id.npy` 并搭配
> `evaluate_policy_separation_aligned.py`。

### 常用命令

```bash
# Global 检索（test split，默认 Top-5）
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 --topk 5 --mode global

# Within-source 检索
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 --mode within-source

# Smoke test（无需数据文件）
python tools/embedding_retrieval_demo.py --smoke_test

# 单元测试
python scripts/smoke_test_retrieval_demo.py
```

### 输出文件（outputs/<run_id>/）

| 文件 | 说明 |
|------|------|
| `retrieval_table.csv` | Top-K 结果（index、meta 字段、distance、excluded 标记） |
| `traj_overlay.png` | 对齐后的 ego + front 轨迹叠加图 |
| `timeseries.png` | speed / accel / jerk / curvature proxy 时序对比图 |
| `summary.json` | 运行参数（mode、distance、topk、数据路径等） |

## Embedding interpretability demo

新增脚本：`tools/embedding_interpretability_demo.py`，用于**可解释可视化**，不修改 benchmark 指标定义。

### 1) Same-source triplet demo
在同一 `source_key = scenario_id|start|window_len|front_id` 下，比较不同 policy 的轨迹与信号，展示受控条件下风格分离。

### 2) Global retrieval demo
给定 query，执行跨 source 的 Top-K 检索，输出卡片图与信号对比，观察 embedding 邻居是否呈现相近驾驶风格。

### 3) within-source 与 global 区别
- `within-source`：受控对比（同源窗口）
- `global`：跨源检索（风格相似性）

### 4) 风格信号定义（轨迹级）
- `speed = sqrt(vx^2 + vy^2)`
- `accel = d(speed)/dt`
- `jerk = d(accel)/dt`
- `yaw_rate_proxy = d(unwrap(atan2(vy,vx)))/dt`
- `curvature_proxy = yaw_rate_proxy / max(speed, eps)`
- 若有 `front.npy`：`gap` 与 `thw = gap/max(speed,eps)`

### 5) 限制说明
- `yaw_rate_proxy/curvature_proxy` 来自速度方向估计，是近似 proxy
- demo 需要多 policy rollout 数据（同 source 至少 3 条记录）才能稳定展示 p0/p1/p2 对比
- 若 `policy_id` 缺失，`summary.json` 会写明 `policy_id_source=unavailable`，并将 same-policy hit@k 置为 `null`
- 此时会给出强提醒：global retrieval 仅能展示最近邻，不能验证 same-policy style retrieval
- 跨 source 轨迹叠加仅用于风格参考，不代表同一场景几何对齐

### 运行示例
```bash
python tools/embedding_interpretability_demo.py \
  --data_dir output_policy_rollouts \
  --out_dir outputs/embedding_demo/case_000 \
  --embedding feat_style \
  --split test \
  --query_index 0 \
  --mode both \
  --projection both \
  --case_selection best_hit_at_k \
  --distance euclidean \
  --topk 5 \
  --source_key_fields scenario_id,start,window_len,front_id \
  --auto_select_valid_source \
  --exclude_same_source \
  --exclude_same_scenario
```

> 若 rollout 的 `front_id` 在不同 policy 下不一致，可改用：  
> `--source_key_fields scenario_id,start,window_len`

### summary.json 诊断字段（重点看）
- `diagnostics.n_total_rows` / `n_rows_after_split`
- `diagnostics.n_unique_source_keys_total` / `n_unique_source_keys_after_split`
- `diagnostics.source_group_size_histogram_total` / `..._after_split`
- `diagnostics.has_policy_id` / `policy_id_source` / `policy_id_counts`
- `diagnostics.split_array_shape` / `embedding_shape` / `meta_shape` / `traj_shape` / `front_shape`

这组字段可直接判断：
1) split 是否把每个 source 只保留成单条（导致 within-source 失效）  
2) 当前数据是否包含可用 `policy_id`（或至少可恢复）  
3) 是否满足“每 source ≥3 条”的可解释 triplet 前提

### Smoke test
```bash
python tools/embedding_interpretability_demo.py \
  --out_dir outputs/embedding_demo/smoke \
  --smoke_test
```

### 关键输出文件
- `summary.json`：含 `diagnostics`（group histogram / policy_id 可用性 / shape）
- `embedding_2d_projection.png` + `embedding_2d_projection.csv`：PCA 2D 投影（仅可视化；query 星标 + Top-K 红圈 + rank）
- `embedding_2d_projection_umap.png` + `.csv`：`--projection umap|both` 且安装 `umap-learn` 时输出（仅可视化）
- `embedding_distance_matrix.png` + `embedding_distance_matrix.csv`：同源 embedding 距离矩阵（图中含数值标注）
- `within_source_triplet.png` / `within_source_style_signals.png` / `within_source_style_fingerprint_kinematic.png` / `within_source_style_fingerprint_dynamics.png` / `within_source_style_fingerprint_normalized.png` / `within_source_style_fingerprint.csv`：同源 policy 对比与风格统计
- `global_retrieval_cards.png` / `global_retrieval_style_signals.png` / `retrieval_table.csv` / `style_fingerprint.csv`：跨源 Top-K 检索解释
- `interpretability_report.md`：自动文本报告（query、同源距离、Top-K、hit@1/hit@k、局限性）

解释建议：
- PCA/UMAP 是降维可视化，不能替代高维 embedding 距离与 aligned evaluator 指标。
- 2D 上不出现完美三团，并不意味着高维空间没有有效分离。
- policy-level 解释依赖 `policy_id/policy_name/source_index` 元数据完整性。

## Experiment 2: lateral_stable Ablation Sweep

### 一键运行（debug）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable
```

### 常用参数
- `--dry_run`：仅打印命令与生效参数，不执行。
- `--skip_generation`：只跑评估（复用已生成 rollouts）。
- `--skip_evaluation`：只生成 rollouts。
- `--embedding {feat_style,feat_style_raw,feat,feat_legacy}`
- `--split {train,val,test}`
- `--distance {euclidean,cosine}`
- `--topk INT`
- `--configs a,b,c`（按名称选择消融子集）

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

- **Purpose**: Test which lateral_stable controls improve p2 independence while keeping comfort/stability metrics acceptable.
- **Script**: `tools/run_lateral_stable_ablation.py`
- **Required inputs**: `--source_data_dir` with `traj.npy`, `front.npy` (plus `split.npy` / `meta.npy` if available).

### Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### Dry-run command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --dry_run
```

### Full command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### Main outputs
- `ablation_summary.csv`, `ablation_summary.json`
- `ablation_recommendation.json`
- `ablation_report.md`
- `ablation_p2_separation_margin.png`
- `ablation_p2_farthest_rate.png`
- `ablation_pairwise_distances.png`
- `ablation_retrieval_classification.png`
- `ablation_p2_style_metrics.png`
- `ablation_tradeoff_plot.png`
- per-config `population_eval/`

### Interpretation
- Higher `p2_farthest_rate` is better.
- `mean_p2_separation_margin > 0` means p2 is a stronger independent mode.
- Lower `p2_rms_yaw_rate_proxy_mean` means stronger lateral stability.
- Lower `p2_rms_jerk_mean` means smoother comfort.
- Retrieval/centroid metrics measure style discriminability.

### Limitations
- Synthetic policies (not human labels).
- Replayed front-vehicle setup (not full closed-loop multi-agent simulation).
- No sensor rendering/perception stack.


## Experiment 2 Ablation（必须产出 base_output_dir 聚合文件）

### 推荐命令（可直接复制）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### 期望输出结构
```text
outputs/ablation_debug/
  ablation_summary.csv
  ablation_summary.json
  ablation_recommendation.json
  ablation_report.md
  ablation_p2_separation_margin.png
  ablation_p2_farthest_rate.png
  ablation_pairwise_distances.png
  ablation_retrieval_classification.png
  ablation_p2_style_metrics.png
  ablation_tradeoff_plot.png

  baseline_current/
    rollouts/
    population_eval/
      population_summary.json

  no_lateral_smoothing/
    rollouts/
    population_eval/
      population_summary.json

  lateral_only/
    rollouts/
    population_eval/
      population_summary.json

  comfort_only/
    rollouts/
    population_eval/
      population_summary.json

  full_strong_lateral_stable/
    rollouts/
    population_eval/
      population_summary.json
```

> 完成标准：`ablation_summary.csv` 与 `ablation_report.md` 必须存在于 `base_output_dir` 根目录。

## Experiment 2B: Local Fine-Grained Sweep Around full_strong_lateral_stable

### Motivation
Run a focused local sweep around `full_strong_lateral_stable` to improve p2 separation while preserving comfort and lateral stability.

### Script usage
`python tools/run_lateral_stable_ablation.py --config_set local_fine ...`

### Dry run
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --dry_run
```

### Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Full command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_full \
  --config_set local_fine \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Output files
- `local_sweep_summary.csv`, `local_sweep_summary.json`
- `local_sweep_recommendation.json`, `local_sweep_report.md`
- `local_sweep_integrity_report.json`, `local_sweep_rollout_sanity.csv`
- `local_sweep_p2_separation_margin.png`, `local_sweep_p2_farthest_rate.png`
- `local_sweep_pairwise_distances.png`, `local_sweep_retrieval_classification.png`
- `local_sweep_p2_style_metrics.png`, `local_sweep_tradeoff_yaw_vs_margin.png`
- `local_sweep_tradeoff_jerk_vs_margin.png`, `local_sweep_delta_vs_center.png`

### Interpretation
Broad ablation compares families; local sweep tests nearby parameter perturbations around the best broad config. If separation margin remains negative, conclude: **p2 independence improved but remains incomplete**.

### Limitations
No public data validation yet.

## Experiment 2C: recommended_lateral_stable_v2 Final Comparison

### Experiment 2B result
Local fine sweep selected `recommended_lateral_stable_v2` (`yaw_008_jerk_020`).

### Recommended lateral_stable v2 parameters
- `heading_smooth_alpha = 0.75`
- `yaw_rate_clip = 0.008`
- `thw_target = 1.70`
- `jerk_limit = 0.200`
- `a_max = 1.275`
- `a_min = -2.52`

### Final comparison command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2 \
  --config_set final_compare \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2_debug \
  --config_set final_compare \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Expected outputs
- `final_config_comparison_summary.csv`
- `final_config_comparison_summary.json`
- `final_config_comparison_report.md`
- `final_config_p2_separation.png`
- `final_config_margin.png`
- `final_config_classification_retrieval.png`
- `final_config_style_metrics.png`
- `final_config_tradeoff.png`
- `ablation_integrity_report.json`

### How to interpret
- `p2_farthest_rate` higher is better.
- `mean_p2_separation_margin` closer to or above 0 is better.
- `centroid_accuracy_p2` measures p2 recognizability.
- `p2_rms_jerk` lower means smoother longitudinal behavior.
- `p2_rms_yaw_rate_proxy` lower means stronger lateral stability.
- negative `mean_p2_separation_margin` means p2 is not yet fully independent.

### Limitations
- Synthetic policy rollout only.
- Replayed front vehicle.
- No real human driver labels yet.
- No sensor rendering / perception stack.
- PCA / UMAP are visualization only.


## Phase 4A: Public Human Trajectory External Validation Scaffold

Purpose: validate whether embedding structure transfers beyond synthetic generator artifacts using trajectory-level weak-label evaluation.

### Unified input format
`traj.npy`, optional `front.npy`, `meta.npy`, `split.npy`, `feat_style.npy`, optional `feat_style_raw.npy`, optional `feature_names_style.json`, optional `embeddings.npy`.

### Pseudo-label assignment
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir <HUMAN_DATA_DIR> \
  --out_dir outputs/vehicledata_validation/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1
```

### Evaluation
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval \
  --embedding_path <OPTIONAL_EMBEDDING_PATH> \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

Baselines-only mode:
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

### Outputs
Pseudo-label outputs include summary/report/distribution files. Evaluation outputs include `human_validation_summary.json`, `human_validation_report.md`, `baseline_comparison_summary.csv`, retrieval/classification/correlation/cluster artifacts and figures.

### Interpretation and limitations
Pseudo labels are rule-based weak labels (not ground truth) for external validation only. Label-defining features can leak into classification metrics, so retrieval, cluster fingerprints, and baseline comparisons must be interpreted jointly.

### Smoke tests
Both scripts support `--smoke_test` and generate synthetic arrays locally without external dataset downloads.

- `evaluate_vehicledata_validation.py` new flags: `--allow_skip_learned`, `--retrieval_mode strict`, exclusion flags.
- Learned embedding mismatches now fail by default; optional skip records warnings and marks `learned_embedding_evaluated=false`.
- Retrieval outputs now include chance/lift metrics and strict anti-leakage behavior.
- Expected outputs: baseline_* plots, cluster_size_distribution.png, cluster_style_fingerprint.png/csv, cluster_label_distribution.csv.

## Embedding alignment requirement

- 评估阶段的 `traj/meta/feat_style/pseudo_label` 是 row-level 数组，learned embedding 必须同样 row-level。
- `embedding.shape[0]` 必须等于样本行数 `N`。
- source-level embedding 默认禁止自动扩展；仅可在 `--allow_source_level_embedding_expansion` 下用于调试，并会标记 `learned_embedding_valid_for_policy_eval=false`。

`data1` 提示：
- `traj` = 33471 rows
- `embeddings` = 11157 rows
- 11157x3=33471，表示 source-level + 3 rollout/policy，不是 row-level learned embedding。

TODO（脚本占位）：
```bash
python tools/export_row_level_embeddings.py \
  --data_dir data1 \
  --model_ckpt <CHECKPOINT> \
  --out_path data1/embeddings_row_level.npy
```


## 阶段 4B：Waymo 真实人类轨迹数据提取

### 命令
```bash
python tools/build_waymo_human_trajectory_dataset.py \
  --waymo_dir <WAYMO_TFRECORD_DIR> \
  --out_dir outputs/waymo_human_v1 \
  --window_len 80 \
  --stride 20 \
  --min_speed 1.0 \
  --max_files 5 \
  --max_scenarios 200 \
  --max_agents_per_scenario 64 \
  --split_by_scenario \
  --overwrite

python tools/build_waymo_human_trajectory_dataset.py \
  --out_dir outputs/waymo_human_smoke \
  --smoke_test \
  --overwrite
```

后续 Stage 4C：
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir outputs/waymo_human_v1 \
  --out_dir outputs/waymo_human_v1/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1 \
  --dataset_type human_public

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1 \
  --label_dir outputs/waymo_human_v1/pseudo_labels \
  --out_dir outputs/waymo_human_v1/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

### 期望行为
- 从原始 Waymo 场景中提取真实 human vehicle agent 的 observed trajectory window。
- 不调用 synthetic policy generator。
- 不生成 p0/p1/p2。
- 不生成 policy_id / policy_name。
- 输出统一格式 npy 文件。
- 每一行对应一个真实 human agent trajectory window。
- split 按 scenario_id hash 分配，避免同一 scenario 泄漏到不同 split。
- 自动计算 style features 和标准化特征。
- 自动生成 build_summary.json 和 build_report.md。

### 通过标准
- out_dir 下生成 traj.npy / front.npy / meta.npy / split.npy / feat_style.npy / feat_style_raw.npy / feature_names_style.json。
- meta.npy 中 dataset_type = human_public。
- meta.npy 中不包含 policy_id / policy_name。
- len(traj) == len(front) == len(meta) == len(split) == feat_style.shape[0]。
- build_summary.json 中 n_windows_kept > 0。
- split_counts 中 train/val/test 至少有一个非空，正式运行应三者都有数据。
- front_found_rate 被记录。
- feature_names_style.json 与 feat_style 的列数一致。
- smoke_test 可以不依赖真实 Waymo 数据运行成功。


## 阶段 4D：训练并导出 Waymo human row-level learned embedding

### 1. 命令
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --batch_size 1024 \
  --device cuda \
  --traj_nan_mode interpolate \
  --max_traj_nan_ratio 0.2 \
  --overwrite
```

### 2. 期望行为
- Waymo human `traj.npy` 可能包含 NaN，因为观测轨迹可能部分无效。
- `export_human_row_embeddings.py` 必须复用训练脚本相同的轨迹清洗与局部归一化逻辑。
- 导出的 embedding 必须与 `traj.npy` 行对齐（row-aligned）。
- 若 `normalize_local` 产生非有限值（NaN/Inf），必须立即失败，禁止保存坏 embedding。
- 若 checkpoint 训练过程中出现 NaN loss，禁止导出，必须先修复并重训。

### 3. 通过标准
- 控制台输出 raw/sanitized 的 NaN/Inf 统计。
- `embedding_export_summary.json` 与 `embedding_export_debug.json` 成功生成。
- `embeddings_row_level.npy` 全量 finite，且 `shape[0] == len(traj.npy)`。
- `row_aligned = true`（官方 Stage 4D 默认不允许 drop）。


## 阶段 4E：jerk/comfort-aware learned embedding 训练

### 命令
同 README 的三条命令（训练/导出/评估），并可追加：
```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --out_dir outputs/waymo_human_v1_full51/paper_tables
```

### 期望行为
- 训练保持 Stage 4D v1 可复现（uniform 默认）。
- jerk_comfort 模式重点提升舒适性相关差异建模。
- 输出可直接用于论文表格。

### 通过标准
- 评估摘要包含 learned_embedding_evaluated=true。
- style_distance_correlation.csv 含各指标 valid_pairs_* 列。
- human_validation_report.md 的 next steps 与 Stage 4D 已完成状态一致。

# Stage 4F：comfort-aware auxiliary regression

## 1. 命令

```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --device cuda \
  --seed 42 \
  --overwrite

python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/aux_prediction_metrics_test.json

python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_aux.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_aux.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca

python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e_stage4f
```

## 2. 期望行为
- 使用 train split 训练。
- 不使用 pseudo labels 训练。
- 在 soft contrastive loss 基础上增加 comfort auxiliary regression。
- evaluate_aux_predictions.py 是训练后、导出前的必做诊断，用于验证 auxiliary regression head 是否真的学到舒适性目标。
- evaluate_aux_predictions.py 与训练/导出共享同一套轨迹 NaN 清洗逻辑（sanitize + normalize_local），可处理 Waymo human traj.npy 的非有限值。
- 报告 rms_accel / rms_jerk / max_abs_accel / max_abs_jerk / mean_thw / min_thw 的 MAE / RMSE / Spearman。
- 该诊断独立于 embedding retrieval/classification 评估。
- 导出 row-aligned embedding。
- 在 test split 上评估 learned vs baselines。
- 与 Stage 4D / Stage 4E 对比。

## 3. 通过标准
- train_total_loss / val_total_loss finite。
- aux_loss finite。
- outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/aux_prediction_metrics_test.json 存在。
- aux_prediction_metrics_test.json 中 row_aligned=true。
- traj_nan_count_after_sanitize=0。
- aux_head_loaded=true。
- rms_jerk / max_abs_jerk 的 Spearman 为有限值（finite），或显式报告为 N/A（含原因与 warning）。
- 若 rms_jerk Spearman 近似 0，视为 Stage 4F 未学到 jerk（即使训练 loss 有限）。
- 若 aux prediction 指标良好但 embedding jerk correlation 未提升，记录为“aux head 学到但未转移到 embedding geometry”。
- embeddings_row_level_comfort_aux.npy shape = [168191, 64]。
- evaluation learned_embedding_evaluated=true。
- learned 的 classification/retrieval 明显高于 random。
- rms_jerk_delta correlation 相比 Stage 4D v1 有明显提升。
- 如果 jerk 未提升，报告中明确记录 Stage 4F 未达到目标。

# Stage 4G：comfort metric alignment

## 1. 命令

Training:

```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --comfort_metric_alignment \
  --metric_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --metric_loss_weight 0.1 \
  --metric_loss_type mse \
  --metric_distance euclidean \
  --device cuda \
  --seed 42 \
  --overwrite
```

Aux prediction diagnostic:

```bash
python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/aux_prediction_metrics_test.json
```

Export:

```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite
```

Evaluate:

```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

Compare Stage 4D / 4E / 4F / 4G:

```bash
python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
    stage4g_comfort_metric=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e_stage4f_stage4g
```

```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4g_comfort_metric
```

## 2. 期望行为

- Stage 4G 在 Stage 4F 的 auxiliary regression 基础上，进一步增加 comfort metric alignment loss。
- auxiliary regression 让 embedding 中可解码 jerk/comfort 信息。
- comfort metric alignment 直接约束 embedding pairwise distance 与 comfort feature pairwise distance 对齐。
- 训练仍然只使用 train split。
- 不使用 pseudo labels 训练。
- pseudo labels 仅用于 test split evaluation。
- 导出的 embeddings_row_level_comfort_metric.npy 必须与 traj.npy 行对齐。
- 评估重点不是只看分类和检索，还要重点看 rms_jerk_delta correlation 是否高于 Stage 4D / 4F。
- Stage 4G 是为了让 embedding geometry 更 jerk/comfort-sensitive。

## 3. 通过标准

- train_total_loss / val_total_loss finite。
- train_metric_loss / val_metric_loss finite。
- aux_loss finite。
- embeddings_row_level_comfort_metric.npy.shape[0] == len(traj.npy) == 168191。
- embedding 全部 finite，无 NaN/Inf。
- human_validation_summary.json 中 learned_embedding_evaluated=true。
- learned 的 classification / retrieval 明显高于 random。
- rms_jerk_delta correlation 相比 Stage 4D v1 的约 0.0697 有明显提升。
- 目标：
  - rms_jerk_delta >= 0.15：初步有效
  - rms_jerk_delta >= 0.20：明显改善
  - rms_jerk_delta >= 0.30：非常理想
- 如果 rms_jerk_delta 未提升，但 aux prediction 仍然很好，说明 metric alignment 权重或距离形式还需要调整。
- 如果 classification/retrieval 接近 random，说明 metric alignment 过强导致 embedding 崩塌。
- compare_stage4d_stage4e_stage4f_stage4g/comparison_summary.csv 生成。

# Stage 4H：4G sanity check — shuffled comfort metric target

## 1. 命令

Training:
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --comfort_metric_alignment \
  --metric_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --metric_loss_weight 0.1 \
  --metric_loss_type mse \
  --metric_distance euclidean \
  --metric_target_shuffle \
  --metric_target_shuffle_seed 42 \
  --device cuda \
  --seed 42 \
  --overwrite
```

Aux prediction diagnostic:
```bash
python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/aux_prediction_metrics_test.json
```

Export:
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_shuffled.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite
```

Evaluation:
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric_shuffled \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_shuffled.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

Comparison:
```bash
python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
    stage4g_comfort_metric=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
    stage4h_metric_shuffled=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric_shuffled \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_to_stage4h
```

## 2. 期望行为

- Stage 4H 是 sanity check，不是主方法。
- 它使用与 Stage 4G 相同的模型和训练流程。
- 唯一区别是 comfort metric alignment 的 target 被打乱。
- 如果 Stage 4G 的提升是真实的，Stage 4H 的 jerk correlation 应该明显低于 Stage 4G。
- auxiliary regression target 不打乱。
- pseudo labels 仍然不用于训练。
- 这个实验用于排除代码 bug、随机偶然、以及无意义 metric alignment 也能提升的可能性。

## 3. 通过标准

- 训练 loss finite。
- 导出 embedding row-aligned。
- evaluation learned_embedding_evaluated=true。
- compare_stage4d_to_stage4h/comparison_summary.csv 生成。
- Stage 4H 的 rms_jerk_delta 应明显低于 Stage 4G。
- 如果 Stage 4H 仍然接近 Stage 4G 的 rms_jerk_delta，需要警惕 metric alignment 实现或评估存在泄漏/bug。
- Stage 4G 仍应作为主结果，Stage 4H 仅作为 sanity check。

# Stage 4I：最终结果固化与论文图表包

## 1. 命令

```bash
python tools/generate_stage4_final_report.py \
  --out_dir outputs/waymo_human_v1_full51/stage4_final_report
```

```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4g_comfort_metric
```

## 2. 期望行为

- 汇总 Stage 4D/4E/4F/4G/4H 的结果。
- 生成最终 ablation 表格。
- 生成 Stage 4G learned vs baselines 表格。
- 生成 Stage 4G auxiliary prediction 表格。
- 生成 Stage 4H shuffled-target sanity check 表格。
- 生成论文可用的图和 Markdown 报告。
- 不启动新训练，不修改模型。

## 3. 通过标准

- stage4_final_report.md 存在且非空。
- table_stage4_ablation.md / .csv 存在。
- table_stage4g_learned_vs_baselines.md / .csv 存在。
- table_stage4g_aux_prediction.md / .csv 存在。
- Stage 4G auxiliary prediction 表格不能全是 NaN。
- table_stage4h_sanity_check.md / .csv 存在。
- table_stage4h_sanity_check.md 必须使用按指标定制的解释文本。
- 若 Stage 4H 的 yaw/curvature 高于 Stage 4G，报告不得错误宣称其“退化”。
- 报告必须明确：sanity check 的关键证据是 jerk collapse + retrieval/classification 回落。
- figure_stage4_style_correlation.png 存在。
- figure_stage4_jerk_delta.png 存在。
- stage4_final_numbers.json 存在。
- 报告明确写出 Stage 4G 是 current best。
- 报告明确写出 Stage 4H shuffled target 使 jerk improvement 消失。
- 报告明确写出限制：pseudo labels 是 weak labels，4G 是 metric-aligned embedding，不是纯无监督发现。


# Stage 5：interaction-aware input design

## 1. 命令

> 当前仅做设计评审，不新增训练命令。

```bash
python -m py_compile tools/train_human_behavior_embedding.py
grep -R "Stage 5" -n README.md QUICK_REFERENCE.md 07_stage5_interaction_design.md
```

## 2. 期望行为

- 本阶段是设计阶段，不启动训练。
- 明确 5-neighbor lane-aware 输入设计。
- 明确 weak supervision feature 分组。
- 明确 longitudinal / lateral / interaction 三个显性 head。
- 明确 flatten GRU 与 slot encoder 两种架构路线。
- 不覆盖 Stage 4G 结果。

## 3. 通过标准

- 07_stage5_interaction_design.md 存在。
- README.md 中能看到 Stage 5 的高层说明。
- QUICK_REFERENCE.md 中能看到 Stage 5 设计阶段说明。
- 文档明确说明 5 个 neighbor slot。
- 文档明确说明 heading 使用 raw heading 优先。
- 文档明确说明 longitudinal / lateral / interaction 三组 features。
- 文档明确说明三个 explicit heads。
- 文档明确说明 Version A flatten GRU 与 Version B slot encoder + attention。
- 文档明确说明本阶段不启动训练。


# Stage 5A：lane-aware 5-neighbor context 数据构建

## 1. 命令

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --out_dir outputs/waymo_5neighbor_context_smoke \
  --smoke_test \
  --overwrite
```

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite
```

## 2. 期望行为

- 从 Waymo 场景中提取 target vehicle 轨迹窗口。
- 为每个 target vehicle 分配 front / left_front / left_rear / right_front / right_rear 五个邻车 slot。
- 优先使用 lane-aware assignment。
- lane-aware 失败时 fallback 到 ego-centric geometric assignment。
- 输出 ego_seq / neighbor_seq / context_traj / context_mask / interaction features。
- 输出 slot coverage、fallback rate、heading fallback rate 等诊断信息。
- 不启动模型训练。
- 不覆盖 Stage 4G 输出。

## 3. 通过标准

- smoke test 能跑通。
- context_traj.npy / ego_seq.npy / neighbor_seq.npy / context_mask.npy 存在。
- interaction_feat_style.npy 和 interaction_feat_style_raw.npy 存在。
- build_summary.json 中包含 slot_valid_ratio、fallback_assignment_rate、heading_proxy_fallback_rate。
- neighbor_slot_valid_ratio.csv 存在。
- lane_assignment_debug.csv 存在。
- build_report.md 用中文说明数据规模、slot coverage、fallback 情况、限制。
- context_traj.npy 和 interaction_feat_style.npy 无 NaN/Inf。
- Stage 4 结果文件未被覆盖。

### Stage 5A 非有限值排查（补充）

若 Stage 5A 在非有限值断言/报错处失败：
- 检查 `nonfinite_debug_*.json`。
- 常见原因是：窗口整体 valid_ratio 达标，但内部仍包含 Waymo invalid 帧（`x/y/vx/vy/heading` 为 NaN）。
- 构建脚本应对该类帧执行插值/清洗（sanitize），并确保最终输出不含 NaN/Inf。

通过标准：
- `build_summary.json` 中 `trajectory_nan_count_after_sanitize = 0`。
- `context_traj.npy` 的 finite 检查为 `true`。
- `interaction_feat_style.npy` 的 finite 检查为 `true`。
- 成功构建时不应产生 `nonfinite_debug_*.json`。

# Stage 5A-v2：真正 lane-aware 5-neighbor assignment

## 1. 命令

Smoke test:

python tools/build_waymo_5neighbor_context_dataset.py \
  --out_dir outputs/waymo_5neighbor_context_laneaware_smoke \
  --smoke_test \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite

Small real data:

python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite

Geometric-only debug baseline:

python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_geometric_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode geometric_only \
  --overwrite

## 2. 期望行为

- 优先使用 Waymo map/lane 信息给 target vehicle 找 current lane、left lane、right lane。
- 根据 lane s 坐标分配 front / left_front / left_rear / right_front / right_rear。
- lane-aware 失败时才 fallback 到几何分配。
- 输出 lane projection / fallback / slot method 诊断。
- 不启动训练。

## 3. 通过标准

- smoke test 中 lane_assignment_success_rate > 0。
- real small data 中 lane_assignment_success_rate 不能为 0。
- fallback_assignment_rate 不能等于 1.0。
- build_summary.json 包含 lane projection 诊断字段。
- lane_assignment_debug.csv 包含 ego_lane_id / neighbor_lane_id / delta_s 等字段。
- 如果 fallback_assignment_rate > 0.5，需要暂停 Stage 5B 训练，先分析原因。
- context_traj.npy / interaction_feat_style.npy 无 NaN/Inf。
- Stage 4 结果不被覆盖。

## Stage 5A-v2 卡住排查（lane-aware，中文）

如果脚本看起来卡住：
- 先看进度条（TFRecord / scenario / target agents / windows）。
- 检查 `build_summary.json` 里的 `timing_seconds`，确认是否 `lane_projection` 占比过高。
- 检查 `lane_projection_avg_candidate_lanes` 是否过大。
- 降低 `--lane_topk_candidates`。
- 降低 `--lane_search_radius`。
- 临时切到 `--assignment_mode geometric_only` 做 baseline/debug。

### 小规模 lane-aware（保守投影限制）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --lane_search_radius 20 \
  --lane_topk_candidates 32 \
  --overwrite
```

### geometric-only 调试
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_geometric_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode geometric_only \
  --overwrite
```


## Stage 5A-v3（lane-aware + fallback）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --overwrite
```

## Stage 5A-v4（normal clean，保留 good + ambiguous_intersection）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --overwrite
```

## Stage 5A-v4（strict quality，仅保留 good）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_strict_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --drop_if_lane_context_ambiguous \
  --overwrite
```

期望行为（中文）：
- normal clean run 保留 good + ambiguous_intersection，丢弃 bad/fallback。
- strict run 仅保留 good。
- empty neighbor slots 不应自动触发 ambiguous_intersection。
- slot coverage 与 empty slot ratio 单独报告。

通过标准（中文）：clean run 完成；lane_context_quality_counts 合理；good_lane_context_rate 不应仅因 slot 为空而接近 0；empty_slot_ratio_by_slot 存在；assignment_method_counts_by_slot 每个 slot 总和等于 n_windows_kept；context_traj.npy 与 interaction_feat_style.npy 无 NaN/Inf。

### Stage 5A-v3 常见错误与排查（clean 模式）

常见错误：
1. `timing referenced before assignment`
   - 原因：`timing` 初始化太晚。
   - 修复：在 `main` 开始处（参数解析与 out_dir 创建后）立即初始化 `t_global` 和 `timing`。

2. `boolean index did not match`
   - 原因：clean filtering 后输出列表行数不一致（先 append 后过滤）。
   - 修复：先完成过滤判断，再原子化统一 append。

3. `CSV dict contains fields not in fieldnames`
   - 原因：debug row schema 不统一。
   - 修复：统一 `LANE_DEBUG_FIELDS`，写 CSV 时使用 `normalize_debug_row`。

4. `lane_assignment_success_rate > 1.0`（或 `current_lane_found_rate > 1.0`）
   - 原因：分子在 clean filtering 之前累计，分母用 `n_windows_kept`（clean filtering 之后），导致分母不一致。
   - 修复：summary 中拆分 pre-filter 与 kept 计数：
     - pre-filter：`lane_assignment_success_count_pre_filter`、`current_lane_found_count_pre_filter`、`left_lane_found_count_pre_filter`、`right_lane_found_count_pre_filter`。
     - kept：`lane_assignment_success_count_kept`、`current_lane_found_count_kept`、`left_lane_found_count_kept`、`right_lane_found_count_kept`、`fallback_assignment_count_kept`。
   - 主指标 rate 统一使用 kept 分母：`n_windows_kept`。
   - 额外输出 pre-filter rate：`lane_assignment_success_rate_pre_filter`、`current_lane_found_rate_pre_filter`。
   - 可选启用 `--strict_summary_validation`，当任一主指标 rate > 1.0 时直接报错。

通过标准：
- clean run 不报错。
- `split.npy / meta.npy / interaction_feat_style.npy / context_traj.npy` 行数一致。
- `assignment_method_counts_by_slot` 每个 slot 的总数等于 `n_windows_kept`。
- `lane_assignment_debug.csv` 可以正常写出。
- `build_summary.json` 包含 clean filtering drop counts。
- `lane_assignment_success_count_kept <= n_windows_kept` 且 `current_lane_found_count_kept <= n_windows_kept`。
- 主指标 rate（`lane_assignment_success_rate/current_lane_found_rate/left_lane_found_rate/right_lane_found_rate/fallback_assignment_rate`）均 `<=1.0`。

## Stage 5A full51 安全构建（流式分片，避免 OOM）

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51 \
  --max_files 51 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --drop_if_lane_context_ambiguous \
  --streaming \
  --output_shard_size 5000 \
  --overwrite
```

说明：
- 不要用非 streaming 模式跑 full51。
- 如果在 “Processing TFRecord files” 阶段被 killed，通常表示 scenario 被整体堆在内存里。
- 请使用 streaming 或 `--file_start/--file_end` 分段运行。

## Stage 5A-v5 故障排查（临时文件依赖）

如果出现 `FileNotFoundError: /tmp/old.py`：
- 说明代码错误依赖了 Codex 临时文件。
- 仓库代码不能依赖 `/tmp/old.py`。
- 需要运行 `python tools/check_no_tmp_dependencies.py`。
- 所有 helper function 必须在仓库源码内定义或从 `tools` 模块导入。

## Stage 5A-v5 故障排查（slot_method 接口不一致）

如果出现：
`AttributeError: 'SlotAssignResult' object has no attribute 'slot_method'`

原因：
build script 与 `lane_aware_assignment.py` 的接口不一致。

修复原则：
- 优先从 `assign.per_slot_debug` 推导每个 slot 的 `assignment_method`。
- 不要假设 `SlotAssignResult` 存在未定义字段。
- smoke test 必须覆盖 streaming mode。

通过标准：
- `python -m py_compile tools/build_waymo_5neighbor_context_dataset.py` 通过。
- `python tools/check_no_tmp_dependencies.py` 通过。
- smoke test 通过。
- full51 streaming 命令可启动，且不会出现 `/tmp/old.py` 的 FileNotFoundError。

## Stage 5A-v5 Streaming 排障（新增）

### 常见现象
- 只看到 `Processing TFRecord files: 0/51` 不动。

### 原因
- 只有外层文件进度条，没有内部 scenario 进度；
- 或第一个 TFRecord 内部处理耗时较长。

### 修复
- streaming 模式必须显示 scenario 级别进度或每 N 个 scenario 的 heartbeat；
- full51 前必须先跑 `max_scenarios=10` 和 `max_scenarios=50` 的 streaming debug 命令。

### 通过标准
- `max_scenarios=10` 能快速完成；
- `max_scenarios=50` 能生成 shard；
- `build_summary.json` 保留完整诊断字段；
- `row_index` 不重复；
- full51 不再一次性缓存所有 scenario；
- 不出现 `/tmp/old.py`；
- 不出现 `SlotAssignResult` 接口错误。

## Stage 5A：并行分片结果合并

命令（示例）：

```bash
python tools/merge_waymo_5neighbor_context_shards.py \
  --input_roots \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51 \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --recompute_global_standardization \
  --overwrite
```

预期行为：
- 以 manifest/统计信息方式合并 Stage 5A 并行分片输出；
- 默认不在 merged root 生成 `ego_seq.npy`、`neighbor_seq.npy`、`context_traj.npy` 等超大单体文件；
- 重新基于 train split 计算全局交互特征标准化，并回写每个 shard 的 `interaction_feat_style.npy`；
- 输出 `shard_manifest.json`、`build_summary.json`、`merged_build_summary.json`、`build_report.md`。

通过标准：
- `shard_manifest.json` 存在；
- `build_summary.json` 存在；
- `interaction_feature_standardization.json` 存在；
- merged `n_windows_kept` 等于四个输入分片之和；
- `fallback_assignment_rate` 仍为 0；
- `good_lane_context_rate` 仍为 1；
- global standardization 的 `train_count > 0`；
- 默认不创建 monolithic 大 `.npy` 文件。

# Stage 5A：重建 sharded summary

## 1. 命令

```bash
python tools/rebuild_waymo_5neighbor_context_summary.py \
  --data_root outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
  --overwrite

python tools/rebuild_waymo_5neighbor_context_summary.py \
  --data_root outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
  --validate_only
```

## 2. 期望行为

- 从 shards 反扫 split/meta/context_mask/debug csv。
- 重建 build_summary.json。
- 修复 split_counts 为空的问题。
- 不拼接大型 npy。
- 不重新生成数据。

## 3. 通过标准

- split_counts 不为空。
- sum(split_counts) == n_windows_kept。
- assignment_method_counts_by_slot 每个 slot 合计等于 n_windows_kept。
- nonfinite_output_detected = 0。
- build_report.md 显示 summary_rebuilt_from_shards=true。
- 不修改 Stage 4。

# Stage 5B：Flatten Context GRU 训练

## 1. 命令

### 1.1 Preflight（真实数据 smoke）
```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_smoke \
  --batch_size 64 \
  --epochs 1 \
  --max_train_samples 2048 \
  --max_val_samples 512 \
  --device cuda \
  --smoke_test_real_data \
  --overwrite
```

### 1.2 全量训练
```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 256 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_alignment \
  --metric_loss_weight 0.1 \
  --metric_loss_type huber \
  --metric_targets all \
  --device cuda \
  --seed 42 \
  --overwrite
```

如遇 CUDA OOM：将 `--batch_size` 降到 `128`。

### 1.3 导出 embedding（按 shard）
```bash
python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1/model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings \
  --batch_size 512 \
  --device cuda \
  --split all \
  --overwrite
```

## 2. 期望行为

- 从 `shard_manifest.json` 读取 Stage 5A 数据。
- 不拼接大型 npy。
- `context_traj.npy` 作为输入。
- `interaction_feat_style.npy` 作为弱监督。
- 训练 Flatten Context GRU。
- 导出 row-aligned sharded embedding。
- 不修改 Stage 4 / Stage 5A 数据构建逻辑。

## 3. 通过标准

- preflight 1 epoch 能跑通。
- Stage 5B smoke 通过标准：`train_loss` / `val_loss` 为有限值且 `model.pt` 存在。
- 单 epoch 时 `loss_curve.png` / `val_loss_curve.png` 看起来“空白”通常只是可视化尺度问题，不代表训练失败。
- `training_summary.json` 存在。
- `context_dim` 和 `feature_dim` 正确记录。
- full training 不 OOM。
- full training 建议开启 `--metric_alignment`。
- `embedding_manifest.json` 存在。
- exported embedding 总行数 = 164871。
- `nonfinite_embedding_detected = 0`。

# Exact Stage 5C evaluator command
```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

# Stage 6C v2 three-split validation complete

## 1. 命令

三个 final experiment 已完成，结果目录：

```text
outputs/stage6C_task_bdd/negative_control_random_v2_final
outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2_final
outputs/stage6C_task_bdd/scene_confounding_v2_final
```

三路汇总输出目录：

```text
outputs/stage6C_task_bdd/stage6c_v2_three_split_summary
```

重新生成三路汇总表：

```bash
python tools/stage6c_summarize_task_bdd_experiments.py \
  --experiment_dirs outputs/stage6C_task_bdd/negative_control_random_v2_final,outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2_final,outputs/stage6C_task_bdd/scene_confounding_v2_final \
  --experiment_names negative,pseudo,scene \
  --output_dir outputs/stage6C_task_bdd/stage6c_v2_three_split_summary \
  --overwrite
```

## 2. 期望行为

- 读取三个 final 目录下的 `task_bdd_summary.csv`、`task_style_delta.csv`、`warnings.json`。
- 生成：
  - `task_bdd_cross_experiment.csv`
  - `task_bdd_pivot.csv`
  - `task_bdd_delta_vs_negative.csv`
  - `top_style_delta_by_experiment.csv`
  - `stage6c_v2_cross_experiment_summary.md`
  - `summarizer_warnings.json`
- 不修改三个 final experiment 原始结果目录。

三路解释：

- `negative_control_random_v2_final`：BDD near zero，sanity check passed。
- `pseudo_agg_vs_cons_v2_final`：BDD significant，positive control passed。
- `scene_confounding_v2_final`：BDD significant，confounding diagnosis passed。

Reliability tier：

| tier | task_key | interpretation |
|---|---|---|
| primary | `task_following` | strong detector，优先用于主要结论 |
| primary | `task_lane_change` | strong detector，优先用于主要结论 |
| primary | `task_yield_conflict` | strong detector，优先用于主要结论 |
| primary | `task_hesitation` | strong detector，但语义解释为 hesitation-like / prolonged maneuver |
| auxiliary_proxy | `task_cutin_response` | proxy-only，辅助诊断 |
| auxiliary_proxy | `task_queue_approach` | proxy-dominant，辅助诊断 |
| auxiliary_proxy | `task_lead_brake_response` | proxy-dominant，辅助诊断 |
| auxiliary_proxy | `task_overtake_opportunity` | proxy / sample-limited，辅助诊断 |
| auxiliary_proxy | `task_overtake_executed` | sample-limited，skipped 不等于 no drift |

## 3. 通过标准

1. `outputs/stage6C_task_bdd/stage6c_v2_three_split_summary/task_bdd_pivot.csv` 包含 `negative`、`pseudo`、`scene` 三列。
2. negative-control BDD 应接近 0 且 non-significant。
3. pseudo positive-control BDD 应在 behavior-style tasks 上显著。
4. scene-confounding BDD 可显著，但 pattern 应与 pseudo 区分，用于 confounding diagnosis。
5. 主要结论优先使用 primary strong-detector tasks。
6. proxy-heavy / sample-limited tasks 必须标记为 auxiliary，不能当作主结论。

# Stage 7 — Empirical Same-Scenario Style Separability

> **Stage 7 总体目标澄清**：Stage 7 的核心不是把 Waymo 风格流程简单重跑到 nuPlan expert data 上，而是在 nuPlan simulation / rollout 中使用同一批 scenario，比较不同 policy / E2E model 版本的驾驶风格，并用 behavior embedding + task-conditioned BDD 验证其分布是否可分。
>
> **明确警告**：不要把 expert nuPlan data export 解读为 Stage 7 最终实验。expert export 只用于 schema discovery 和 converter validation；Stage 7C / 7D 才是核心 proof，Stage 7D 是主 empirical policy-style BDD validation。详见 `docs/stage7_master_plan_same_scenario_policy_bdd.md`。

## Stage 7 A-E 子阶段检查表

### 1. 命令

当前 master plan 文档：

```bash
cat docs/stage7_master_plan_same_scenario_policy_bdd.md
```

子阶段清单：

- 7A：mini readiness check，确认 nuPlan mini DB / map / SQLite schema 可读。
- 7B：expert export and converter validation，只用于导出 expert ego trajectory / nearby object context 并验证 context dataset 转换接口。
- 7C：conservative / aggressive rollout generation，在同一 scenario set `S` 上运行保守 / 激进 planner。
- 7D：policy A/B BDD report，将 A/B rollout 转为 context dataset，构建 behavior events，运行 task-conditioned BDD。
- 7E：scaling and real E2E replacement，扩大 scenario 数量，并在可用时替换为 learning-based planner checkpoints 或 company E2E model A/B rollout。

### 2. 期望行为

- Stage 7A / 7B 只产出基础设施证据：数据就绪、schema 理解、converter 接口可用。
- Stage 7C 产出同场景 policy A/B rollout：`scenario_list.csv`、conservative rollout、aggressive rollout、`rollout_manifest.json`。
- Stage 7D 产出真正的 empirical validation：同一 policy 内 random A/B negative control 应低 BDD；conservative vs aggressive 应在 primary tasks 上有更高 BDD。
- Stage 7E 保持相同 context dataset 和 BDD pipeline，只替换 rollout 来源或扩大 scenario 数量。

### 3. 通过标准

1. 文档和报告必须明确写清：Stage 7 是 same-scenario different-policy / E2E rollout BDD validation。
2. 不得把 expert trajectory export 写成 Stage 7 主结果；它只能作为 Stage 7B converter debug data。
3. Stage 7D 必须包含同 policy random split negative control 和 conservative vs aggressive policy-style comparison。
4. primary tasks 至少关注 `task_following`、`task_lane_change`、`task_yield_conflict`、`task_hesitation`。
5. `task_cutin_response`、`task_lead_brake_response`、`task_queue_approach`、`task_overtake_opportunity` 作为 auxiliary tasks 解释。
6. Stage 7C / 7D 是核心 proof；Stage 7A / 7B 是基础设施，不证明 policy style separability。

## 1. 命令

Stage 7 的目标是从 pseudo split 走向 empirical validation：

- Stage 6D matched-task pseudo split 仍然是 pseudo-label based。
- 下一步目标应是同一 scenario set、同一 driving task 下，不同 policy / model / driver 的真实 behavior embedding distribution 是否可分。
- Stage 6C `scene_confounding` 应视为 confounding-awareness diagnostic，不是主要 empirical proof。

推荐数据优先级：

1. company E2E A/B data
2. nuPlan closed-loop planner rollout
3. CARLA same-scenario rollout
4. human-driver public datasets as auxiliary validation

如果没有 company E2E A/B 数据，优先从 nuPlan 开始；它是自动驾驶 planning benchmark，支持 closed-loop planner rollout，使用真实场景，并允许 same-scenario policy comparison。

Stage 7 common rollout schema 必需字段：

```text
scenario_id
policy_id or driver_id
timestamp
ego_x
ego_y
ego_vx
ego_vy
ego_speed
ego_accel
ego_heading
ego_yaw_rate
```

可选 neighbor 字段：

```text
neighbor_id
neighbor_x
neighbor_y
neighbor_vx
neighbor_vy
neighbor_speed
neighbor_heading
neighbor_type
```

占位命令：构建 Stage 7 common rollout dataset。

```bash
python tools/stage7_convert_rollouts_to_context_dataset.py \
  --source_type nuplan \
  --input_rollout_dir <path> \
  --output_dir outputs/stage7/<experiment_name>/context_dataset \
  --overwrite
```

占位命令：构建 behavior events。

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7/<experiment_name>/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7/<experiment_name>/context_dataset/feature_schema.json \
  --output_dir outputs/stage7/<experiment_name>/behavior_events_v2 \
  --overwrite
```

占位命令：计算 task-conditioned BDD。

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest outputs/stage7/<experiment_name>/embeddings/embedding_manifest.json \
  --shard_manifest outputs/stage7/<experiment_name>/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7/<experiment_name>/context_dataset/feature_schema.json \
  --a_indices_path outputs/stage7/<experiment_name>/splits/policy_A_indices.npy \
  --b_indices_path outputs/stage7/<experiment_name>/splits/policy_B_indices.npy \
  --behavior_event_bins_path outputs/stage7/<experiment_name>/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path outputs/stage7/<experiment_name>/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage7/<experiment_name>/task_bdd_report \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --overwrite
```

说明：以上 Stage 7 命令是 placeholder，需先实现 source-specific converter，例如 `tools/stage7_convert_rollouts_to_context_dataset.py`。

## 2. 期望行为

- 输入同一 scenario set 下 policy / driver A 和 B 的 rollout。
- 转换为统一 sharded context dataset：
  - `ego_seq.npy`
  - `neighbor_seq.npy`
  - `metadata.csv` 或 `metadata.npy`
  - `shard_manifest.json`
  - `feature_schema.json`
- 复用 Stage 6C v2 的 behavior-event builder 和 task-conditioned BDD report。
- 在同一 driving task 内比较 A/B behavior embedding distribution。

## 3. 通过标准

1. A/B rollout 必须来自同一 scenario set 或严格 matched scenario family。
2. A/B 必须保留 `policy_id` 或 `driver_id`，并可追溯到 `scenario_id`。
3. Primary tasks 优先解释 `following`、`lane_change`、`yield_conflict`、`hesitation-like`。
4. `cutin`、`lead_brake`、`queue` 仍按 auxiliary / proxy-heavy 解释。
5. nuPlan 是无 company data 时的推荐 open-source next step。
6. Stage 7 结论应区别于 Stage 6C pseudo validation：Stage 7 目标是真实 policy / model / driver 的 empirical same-scenario style separability。

# Stage 7A — nuPlan same-scenario policy validation

## 1. 命令

Stage 7A 是 Stage 7 的轻量化第一步：用 nuPlan mini，在同一批真实规划场景上运行 conservative / aggressive 两个 planner variant，再复用现有 behavior embedding + Stage 6C BDD pipeline。

为什么继续往前走：

- Stage 6 pseudo splits 有用，但仍然是 pseudo。
- Stage 7A 使用 same-scenario rollout，是 empirical policy A/B validation。
- 这还不是完整 E2E model A/B，但比 pseudo split 更接近真实模型/策略差异验证。

硬件建议：

- MacBook Air M5 16GB：文档、轻量分析、代码编辑。
- Intel Ultra5 + 8GB GPU：nuPlan runtime 主机器。
- 推荐 Ubuntu 或 WSL2 Ubuntu。
- 推荐 Python 3.9，因为 nuPlan devkit 主要在 Ubuntu + Python 3.9 上测试。
- 使用 nuPlan mini，不跑 full dataset。
- 不训练大模型，不做 sensor-based E2E training。

导出 conservative planner rollouts：

```bash
python tools/stage7a_export_nuplan_rollouts.py \
  --nuplan_data_root ~/nuplan/dataset \
  --nuplan_maps_root ~/nuplan/dataset/maps \
  --nuplan_exp_root ~/nuplan/exp \
  --scenario_filter mini \
  --planner_variant conservative \
  --max_scenarios 20 \
  --output_dir outputs/stage7A_nuplan/conservative_rollouts \
  --overwrite
```

导出 aggressive planner rollouts：

```bash
python tools/stage7a_export_nuplan_rollouts.py \
  --nuplan_data_root ~/nuplan/dataset \
  --nuplan_maps_root ~/nuplan/dataset/maps \
  --nuplan_exp_root ~/nuplan/exp \
  --scenario_filter mini \
  --planner_variant aggressive \
  --max_scenarios 20 \
  --output_dir outputs/stage7A_nuplan/aggressive_rollouts \
  --overwrite
```

转换 rollouts 到 context dataset：

```bash
python tools/stage7a_convert_rollouts_to_context_dataset.py \
  --rollout_dir outputs/stage7A_nuplan \
  --output_dir outputs/stage7A_nuplan/context_dataset \
  --overwrite
```

后续复用 Stage 6C behavior events：

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7A_nuplan/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7A_nuplan/context_dataset/feature_schema.json \
  --output_dir outputs/stage7A_nuplan/behavior_events_v2 \
  --overwrite
```

后续复用 Stage 6C task-conditioned BDD：

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest outputs/stage7A_nuplan/embeddings/embedding_manifest.json \
  --shard_manifest outputs/stage7A_nuplan/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7A_nuplan/context_dataset/feature_schema.json \
  --a_indices_path outputs/stage7A_nuplan/splits/policy_A_indices.npy \
  --b_indices_path outputs/stage7A_nuplan/splits/policy_B_indices.npy \
  --behavior_event_bins_path outputs/stage7A_nuplan/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path outputs/stage7A_nuplan/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage7A_nuplan/task_bdd_report \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --overwrite
```

注意：当前 `stage7a_export_nuplan_rollouts.py` 和 `stage7a_convert_rollouts_to_context_dataset.py` 是 skeleton tools。它们会写 manifest / schema validation 信息，但不会伪造 rollout 或 context dataset。

## 2. 期望行为

- exporter 检查 `nuplan_data_root`、`nuplan_maps_root`、`nuplan_exp_root` 是否存在。
- exporter 如果当前 Python 环境没有 nuPlan devkit，会清楚提示需要在 Ubuntu / WSL2 + Python 3.9 中安装 nuPlan devkit。
- exporter 写出 `rollout_export_manifest.json` 和 `README.md`，说明 requested rollout export 和 TODO。
- converter 检查未来 rollout CSV / parquet 是否包含必需字段：
  - `scenario_id`
  - `policy_id`
  - `timestamp`
  - `ego_x`
  - `ego_y`
  - `ego_vx`
  - `ego_vy`
  - `ego_speed`
  - `ego_accel`
  - `ego_heading`
  - `ego_yaw_rate`
- converter 写出 `feature_schema.json`、`conversion_manifest.json` 和 `README.md`。
- converter 不会静默创建假的 `ego_seq.npy` / `neighbor_seq.npy`。

## 3. 通过标准

1. `python -m py_compile tools/stage7a_export_nuplan_rollouts.py tools/stage7a_convert_rollouts_to_context_dataset.py` passes。
2. exporter 在缺少 nuPlan devkit 时给出清晰错误信息，而不是 traceback。
3. converter 对 CSV / parquet schema 做显式检查，缺字段时写入 manifest 并返回错误状态。
4. 不修改 Stage 6C final result files。
5. Stage 7A 只使用 nuPlan mini 和 rule-based / configurable planner variants 起步，不进行大规模训练。

## Expected outputs
Training output dir:
- `model.pt`
- `best_model.pt`
- `training_config.json`
- `feature_group_config.json`
- `train_log.csv`
- `training_summary.json`

Embedding output dir:
- `embedding_manifest.json`
- `embeddings/` (shard-aligned outputs)
- optional merged `embeddings.npy`

Evaluation output dir:
- `evaluation_summary.json`
- `evaluation_report.md`
- `category_correlation_summary.csv`
- retrieval/correlation plots and CSVs

## Success criteria
- learned still beats `random/context_l2`.
- following_interaction mean correlation improves vs Stage 5B baseline.
- lateral_lane_dynamics advantage is preserved.
- global retrieval improves, or at least does not degrade significantly.

# Stage 5D 组加权训练（context GRU）

### 1. 命令
```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_loss_type huber \
  --style_loss_weight 1.0 \
  --aux_longitudinal_weight 0.5 \
  --aux_following_weight 1.5 \
  --aux_lateral_dynamics_weight 1.0 \
  --aux_lateral_gap_weight 1.0 \
  --aux_behavior_proxy_weight 0.5 \
  --metric_longitudinal_weight 0.5 \
  --metric_following_weight 2.0 \
  --metric_lateral_dynamics_weight 1.0 \
  --metric_lateral_gap_weight 1.0 \
  --metric_behavior_proxy_weight 0.5 \
  --device cuda \
  --seed 42 \
  --overwrite

python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1/best_model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings \
  --split all \
  --merge_embeddings

python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

### 2. 期望行为
- 训练脚本会读取既有 Stage 5 shard 数据与 `feature_schema.json`，按特征名解析分组索引；不会重建 Stage 5A 数据集。
- 训练输出目录会保存模型、最优模型、训练配置、分组配置、日志和 summary。
- 导出脚本会生成与 shard 行顺序对齐的 embedding 文件与 manifest。
- 评估脚本使用现有 Stage 5C evaluator，对新 embedding 进行 strict-schema paper-grade 评估。

### 3. 通过标准
- 训练命令可正常启动并产生 `best_model.pt` 与 `training_summary.json`。
- `feature_group_config.json` 中可看到按 feature name 解析出的 group indices。
- 导出命令产出 `embedding_manifest.json`，且 `nonfinite_embedding_detected=0`。
- 评估命令产出 `evaluation_summary.json` 与 `category_correlation_summary.csv`，可用于验证 following 是否提升且 lateral 优势是否保持。

## Stage 6A：非配对风格漂移评估

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --feature_path <interaction_feat_style.npy> \
  --feature_schema_path <feature_schema.json> \
  --split_path <split.npy> \
  --experiment_name neg_ctrl

python tools/stage6_compare_unpaired_style.py \
  --context_traj_path <context_traj.npy> \
  --feature_path <interaction_feat_style.npy> \
  --feature_schema_path <feature_schema.json> \
  --encoder_ckpt <stage5d_balanced_v2.pt> \
  --a_indices_path outputs/stage6A_splits/neg_ctrl/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/neg_ctrl/b_indices.npy \
  --output_dir outputs/stage6A_compare/neg_ctrl
```

## 2. 期望行为

读取 Stage 5 处理产物与 Stage 5D 编码器，输出 BDD、类别/特征/slice 漂移、top drift case、markdown report 与最小图表；不会触发新训练或下载新数据。

## 3. 通过标准

输出目录包含 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`scenario_slice_delta.csv`、`top_drift_cases.csv`、`style_report_card.md` 与 `plots/*.png`。

## Stage 6A 非配对风格漂移（Issue #114）


### Stage 6A（Issue #116）推荐：full51 分片清单模式


#### Stage 6A-1：Negative control

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

随机同分布拆分 A/B，作为“无真实风格漂移”的负对照，BDD 与 permutation p-value 不应提示显著漂移。

## 3. 通过标准

- 结果目录包含 split 与 compare 全套产物。
- BDD 接近 0 且 p-value 不显著（与历史负对照量级一致）。

#### Stage 6A-2：Pseudo style positive control

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode pseudo_style_aggressive_vs_conservative \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name pseudo_agg_vs_cons \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/pseudo_agg_vs_cons \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

A（conservative_like）与 B（aggressive_like）应产生更明显风格差异；BDD 应高于 negative control，类别/特征变化应可解释 B 更激进、舒适性更弱。

## 3. 通过标准

- BDD 明显高于负对照。
- `category_delta.csv` 与 `feature_delta.csv` 支撑“更激进/更不舒适”解释。

#### Stage 6A-3：Scene confounding control

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode scene_confounding_control \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name scene_confounding \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/scene_confounding \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

A（easy_scene_like）与 B（complex_scene_like）场景代理分布不同，BDD 可能抬升，报告应提示潜在 scenario/ODD confounding。

## 3. 通过标准

- `scenario_slice_delta.csv` 存在且可解释哪些场景代理切片驱动差异。
- warning 与 report card 提示“可能由场景分布差异驱动”。


## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
EMB_ROOT=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $DATA_ROOT/shard_manifest.json \
  --feature_schema_path $DATA_ROOT/feature_schema.json \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random

python tools/stage6_compare_unpaired_style.py \
  --source_shard_manifest $DATA_ROOT/shard_manifest.json \
  --embedding_manifest $EMB_ROOT/embedding_manifest.json \
  --feature_schema_path $DATA_ROOT/feature_schema.json \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --indices_are_test_relative \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random
```

## 2. 期望行为

- 默认按 Stage 5 full51 的 `shard_manifest.json` 读取分片特征与 split，不依赖根目录扁平 `.npy`。
- compare 默认按 Stage 5D-balanced-v2 的 `embedding_manifest.json` 读取分片 embedding。
- `--feature_path/--split_path` 与 `--embedding_path` 仅保留为 legacy fallback。

## 3. 通过标准

- split 目录产出 `a_indices.npy`、`b_indices.npy`、`split_summary.json`。
- compare 目录产出 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`stage6_warnings.json` 与图表。
- 日志包含“使用 shard_manifest + embedding_manifest 模式（Stage5 full51 推荐路径）”提示。

### 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
FEATURE_PATH=$DATA_ROOT/interaction_feat_style.npy
SCHEMA_PATH=$DATA_ROOT/feature_schema.json
SPLIT_PATH=$DATA_ROOT/split.npy
CONTEXT_PATH=$DATA_ROOT/context_traj.npy
EMBEDDING_PATH=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embeddings.npy
CKPT=$DATA_ROOT/context_gru_stage5d_group_weighted_v1/best_model.pt

ls -lh \
  $FEATURE_PATH \
  $SCHEMA_PATH \
  $SPLIT_PATH \
  $CONTEXT_PATH \
  $EMBEDDING_PATH \
  $CKPT
```

> 重要警告：如果 `context_traj.npy`、`interaction_feat_style.npy` 或 `split.npy` 在 full51 merged 目录下缺失，单体数组命令不可用。若已有与 feature 行对齐的 embedding，请优先使用 `--embedding_path` 模式。

```bash
# 1) negative_control_random
python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random

# 2) pseudo_style_aggressive_vs_conservative
python tools/stage6_build_ab_splits.py \
  --mode pseudo_style_aggressive_vs_conservative \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name pseudo_style_aggressive_vs_conservative

# 3) scene_confounding_control
python tools/stage6_build_ab_splits.py \
  --mode scene_confounding_control \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name scene_confounding_control
```

```bash
# A. embedding_path 模式（推荐）
python tools/stage6_compare_unpaired_style.py \
  --embedding_path $EMBEDDING_PATH \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --top_k 20

# B. context/encoder 模式（回退）
python tools/stage6_compare_unpaired_style.py \
  --context_traj_path $CONTEXT_PATH \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --encoder_ckpt $CKPT \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare_ctx/negative_control_random \
  --device cuda \
  --batch_size 256 \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --top_k 20
```

### 2. 期望行为
- split 脚本读取 `feature/split/schema`，生成 A/B 索引与 split summary。
- compare 脚本读取 A/B、embedding（或 context+ckpt）与 feature，输出 BDD、category/feature/slice/case 与报告卡。
- compare 脚本会写 `stage6_warnings.json`，提醒缺失特征、未标定 BDD、元数据缺失等风险。

### 3. 通过标准
- 三个 split 实验都能生成 `a_indices.npy`、`b_indices.npy`。
- compare 产物包含：
  - `bdd_summary.json`
  - `bdd_bootstrap_samples.csv`
  - `bdd_permutation_samples.csv`
  - `category_delta.csv`
  - `feature_delta.csv`
  - `scenario_slice_delta.csv`
  - `top_drift_cases.csv`
  - `stage6_warnings.json`
  - `style_report_card.md`
- `feature_delta.csv` 的 `permutation_p_value` 不应全是 1.0（除非数据本身极端巧合）。

## Stage 6A（推荐流程，Manifest 模式，仅此为当前建议）

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python -m py_compile \
  tools/stage6_build_ab_splits.py \
  tools/stage6_compare_unpaired_style.py \
  tools/stage6_generate_report_card.py

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

- 使用 `shard_manifest.json` + `embedding_manifest.json`，按分片顺序对齐读取特征和 embedding。
- A/B 索引为全局行号；top drift 不再构建完整 A×B 距离矩阵，避免 OOM。
- 即使无可用切片，也会输出仅含表头的 `scenario_slice_delta.csv`，并继续生成报告卡。

## 3. 通过标准

- 无需 `--embedding_path __unused_manifest_mode_guard_workaround__` 之类临时参数。
- 输出目录包含 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`scenario_slice_delta.csv`、`top_drift_cases.csv`、`stage6_warnings.json`、`style_report_card.md` 和核心图表。
- 上述命令可直接复制执行且不依赖根目录扁平 `interaction_feat_style.npy/split.npy/context_traj.npy`。

> LEGACY/DEPRECATED：任何依赖 `$DATA_ROOT/interaction_feat_style.npy`、`$DATA_ROOT/split.npy`、`$DATA_ROOT/context_traj.npy`、`context_gru_stage5d_group_weighted_v1*` 的 Stage 6A 命令都不是当前推荐流程。


## Stage 6A — Full Negative Control（Manifest 模式，当前标准流程）

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python -m py_compile   tools/stage6_build_ab_splits.py   tools/stage6_compare_unpaired_style.py   tools/stage6_generate_report_card.py

python tools/stage6_build_ab_splits.py   --mode negative_control_random   --shard_manifest $SHARD_MANIFEST   --feature_schema_path $FEATURE_SCHEMA   --eval_split test   --output_dir outputs/stage6A_splits   --experiment_name negative_control_random   --seed 42   --overwrite

python tools/stage6_compare_unpaired_style.py   --embedding_manifest $EMBEDDING_MANIFEST   --shard_manifest $SHARD_MANIFEST   --feature_schema_path $FEATURE_SCHEMA   --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy   --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy   --feature_groups_config configs/stage6_feature_groups.yaml   --output_dir outputs/stage6A_compare/negative_control_random   --num_bootstrap 50   --num_permutation 100   --max_mmd_samples 2000   --max_top_case_candidates 5000   --top_k 20   --seed 42   --overwrite
```

## 2. 期望行为

- `negative_control_random` 是在 `test` 集内做同分布随机切分（A/B）。
- 期望结果：BDD 很小、`p-value` 不显著、category/feature effect size 整体较小。
- 该实验用于验证 Stage 6A 不会把同分布样本误报为 style drift。
- 输出目录：`outputs/stage6A_compare/negative_control_random/`，包含：
  - `bdd_summary.json`
  - `category_delta.csv`
  - `feature_delta.csv`
  - `scenario_slice_delta.csv`
  - `top_drift_cases.csv`
  - `stage6_warnings.json`
  - `style_report_card.md`
  - `plots/`

## 3. 通过标准

- `BDD_MMD` 约 `0.0004`、`p-value` 约 `0.396`（数量级接近即可），表示整体分布漂移不显著。
- category/feature 的 effect size 应整体较小；若个别特征 p-value 显著但 effect size 很小，不应过度解读。
- `scenario_slice_delta.csv` 存在（即使仅表头也要有明确 warning）。

> 重要说明：
> - Stage 6A 当前推荐路径是 manifest 模式，不推荐根目录扁平 npy 作为主流程。
> - 不推荐命令中使用：`$DATA_ROOT/interaction_feat_style.npy`、`$DATA_ROOT/split.npy`、`$DATA_ROOT/context_traj.npy`、`context_gru_stage5d_group_weighted_v1`。
> - 以上仅作为 legacy/fallback。
> - 当前推荐 embedding 模型：`context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json`。
>
> 已知限制：scenario slicing 依赖 `feature_schema.json` 可用代理特征；若 speed/density proxy 缺失会产生 warnings。Stage 6B 需要更丰富 scene metadata/scene descriptor matching。

## Stage 6B — Baseline Comparison and Scenario-Controlled BDD

### 1. 命令
```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/negative_control_random \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/pseudo_agg_vs_cons \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/scene_confounding \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_scenario_balanced_bdd.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --output_dir outputs/stage6B_balanced/scene_confounding \
  --balance_keys lateral_activity_bin \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --min_bin_size 100 --seed 42 --overwrite

python tools/stage6b_scenario_balanced_bdd.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --output_dir outputs/stage6B_balanced/pseudo_agg_vs_cons \
  --balance_keys lateral_activity_bin \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --min_bin_size 100 --seed 42 --overwrite

python tools/stage6b_summarize_experiments.py \
  --experiment_roots \
  outputs/stage6A_compare/negative_control_random \
  outputs/stage6A_compare/pseudo_agg_vs_cons \
  outputs/stage6A_compare/scene_confounding \
  outputs/stage6B_baselines/negative_control_random \
  outputs/stage6B_baselines/pseudo_agg_vs_cons \
  outputs/stage6B_baselines/scene_confounding \
  outputs/stage6B_balanced/scene_confounding \
  outputs/stage6B_balanced/pseudo_agg_vs_cons \
  --output_dir outputs/stage6B_summary \
  --overwrite
```

### 2. 期望行为
- baseline 脚本读取 manifest 模式特征与 embedding，输出 embedding/feature/PCA-feature 三套 MMD 统计与特征效应量。
- balanced 脚本按 `lateral_activity_bin` 做 A/B bin 内配平，再对比 raw 与 balanced BDD。
- summarize 脚本汇总 Stage6A/6B 输出，生成统一校准表与图。

### 3. 通过标准
- 三个 baseline 实验均生成 `baseline_summary.json`、`baseline_mmd.csv`、`top_feature_effects.csv` 和图。
- 两个 balanced 实验均生成 `balanced_bdd_summary.json`，且 `bins_used` 非空。
- summary 生成 `stage6b_calibration_table.csv` 与三张汇总图。

## Stage 6B — ODD bins and Behavior-event BDD

### 1. 命令
```bash
# 推荐 Stage6A/6B 工作流（manifest 模式）
# 1) shard_manifest.json
# 2) feature_schema.json
# 3) context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --inspect_metadata

DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json
RAW_SCENARIO_DIR=<path_to_original_waymo_scenario_tfrecords>

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1 \
  --overwrite

python tools/stage6b_build_odd_bins.py \
  --map_odd_manifest $DATA_ROOT/map_odd_features_v1/map_odd_manifest.json \
  --shard_manifest $SHARD_MANIFEST \
  --output_dir $DATA_ROOT/map_odd_bins_v1 \
  --overwrite

python tools/stage6b_build_behavior_event_bins.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --output_dir $DATA_ROOT/behavior_event_bins_v1 \
  --overwrite
```

### 2. 期望行为
- ODD bins 用于外部场景公平性控制（map/static context）。
- 若 `map_match_valid` 低于阈值，构建脚本会失败；不会再输出默认全零伪特征。
- Behavior-event bins 用于定位漂移发生在哪些驾驶任务。
- `event_lateral_activity_bin` 仅用于行为报告，不作为主 ODD 控制变量。

### 3. 通过标准
- `map_odd_feat.npy` 与分片 `interaction_feat_style.npy` 行对齐。
- `stage6b_build_map_odd_features.py` 必须报告 metadata/raw scenario overlap，且 `map_match_valid` 非零。
- `odd_bins.csv` / `behavior_event_bins.csv` 必须包含 `global_row, shard_id, local_row`。
- ODD 平衡后可输出 `BDD_odd_balanced`，并与 `BDD_overall` 对比解释。


## Legacy 说明

旧的 root-level flat npy 命令仅保留兼容，不再作为 Stage6A/6B 主流程推荐。

## Stage 6B（map-derived ODD）安全流程（2026-05 更新）

### 1. 命令

```bash
python -m py_compile \
  tools/stage6b_build_map_odd_features.py \
  tools/stage6b_build_odd_bins.py \
  tools/stage6b_build_behavior_event_bins.py \
  tools/stage6b_bin_bdd_report.py \
  tools/stage6b_scenario_balanced_bdd.py

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1_debug \
  --max_scenarios 1000 \
  --inspect_metadata

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1_debug \
  --max_scenarios 1000 \
  --path_source raw_track \
  --min_match_rate 0.01 \
  --min_lane_match_rate 0.5 \
  --overwrite

python tools/stage6b_build_odd_bins.py \
  --map_odd_manifest $DATA_ROOT/map_odd_features_v1_debug/map_odd_manifest.json \
  --shard_manifest $SHARD_MANIFEST \
  --output_dir $DATA_ROOT/map_odd_bins_v1_debug \
  --min_map_match_rate 0.01 \
  --overwrite

head $DATA_ROOT/map_odd_features_v1_debug/map_odd_debug_samples.csv
```

### 2. 期望行为

- 先做语法检查，再做 metadata inspect。
- Stage6B 重计算脚本默认开启进度条；非交互式 CI/日志环境可加 `--no_progress` 关闭。
- 完整 map ODD 抽取会显示三个关键进度：`scan processed shards`、`scan raw tfrecords`、`compute ODD per shard/row`。
- inspect 阶段会检查 processed/raw scenario overlap、`start/window_len/target_agent_id` 字段可用性。
- 仅当 inspect 建议可运行时再执行 debug build。
- map ODD 轨迹默认来源是 `--path_source raw_track`（Waymo 原始 scenario tracks 全局坐标），**不再默认使用 context_traj**。
- `--path_source context` 仅用于排障回归，会打印强 warning（context 可能是 ego-centric/feature-space，和 map 全局坐标不对齐）。
- 输出 `map_odd_debug_samples.csv` 用于人工检查坐标范围、最近车道距离、邻近地图要素计数。
- map_odd manifest 分片输出目录使用全局唯一命名，避免不同 part 的同名 shard 覆盖。
- map ODD 分箱仅使用 `map_match_valid=1` 行计算分位点，`map_match_valid=0` 行统一标记 `unknown`。
- odd bins 构建前会强校验：feature/meta 路径唯一、每 shard 行数一致、总行数与 manifest 一致；不一致直接失败。
- 当 `odd_map_complexity_bin` 或 `odd_lane_count_bin` 全部是 `unknown` 时默认失败（除非显式 `--allow_degenerate_bins`）。
- behavior-event bins 保持独立报告层，不与 map ODD bins 混用。
- 旧版 flat-npy Stage6 命令视为 **LEGACY/DEPRECATED**，不建议继续作为主流程。

### 3. 通过标准

- `py_compile` 全部通过。
- inspect 输出包含 overlap 与 context 检查，recommendation 为可执行状态。
- map_odd 输出包含 `map_odd_warnings.json`，且 `map_match_valid_rate` 高、`local_lane_match_valid_rate` 不接近 0。
- `no_near_lane_rows` 不应接近总行数；`nearest_lane_distance` p50/p90/p95 有合理数值。
- `odd_map_complexity_bin` 与 `odd_lane_count_bin` 不能全部为 `unknown`。
- odd_bins 行数必须严格等于 map_odd_manifest `total_rows`。
- behavior_event_bin_report 明确声明 `event_lateral_activity_bin` 是 behavior-contaminated。


## Stage 6B Behavior-event BDD Decomposition

### pseudo_agg_vs_cons

| Behavior-event bin | Bin value | n_A | n_B | BDD_MMD | Interpretation |
|---|---:|---:|---:|---:|---|
| event_following_bin | following_proxy | 4917 | 4917 | 0.1661 | following style drift remains strong |
| event_cut_in_bin | cut_in_proxy | 2908 | 429 | 0.1538 | cut-in proxy drift exists but A/B count is imbalanced |
| event_cut_in_bin | no_cut_in_proxy | 2009 | 4488 | 0.1350 | drift also exists outside cut-in proxy |
| event_lane_change_bin | lane_change | 675 | 2756 | 0.2314 | strongest lane-change-related drift |
| event_lane_change_bin | no_lane_change | 4242 | 2161 | 0.1331 | lower drift without lane-change |
| event_yielding_bin | non_yielding_like | 1931 | 4454 | 0.1460 | yielding-related proxy shift |
| event_yielding_bin | yielding_like | 2986 | 463 | 0.1398 | yielding proxy drift but imbalanced |
| event_lateral_activity_bin | high | 487 | 2827 | 0.2403 | strongest high-lateral-activity drift |
| event_lateral_activity_bin | mid | 1393 | 1693 | 0.1860 | medium lateral activity drift |
| event_lateral_activity_bin | low | 3037 | 397 | 0.1277 | lowest lateral activity drift |

### scene_confounding

| Behavior-event bin | Bin value | n_A | n_B | BDD_MMD | Interpretation |
|---|---:|---:|---:|---:|---|
| event_following_bin | following_proxy | 4917 | 4917 | 0.1246 | following drift close to overall scene_confounding BDD |
| event_cut_in_bin | cut_in_proxy | 278 | 2576 | 0.1475 | cut-in proxy exposure is strongly imbalanced |
| event_cut_in_bin | no_cut_in_proxy | 4639 | 2341 | 0.2240 | strong drift outside cut-in proxy |
| event_lane_change_bin | lane_change | 602 | 2575 | 0.1936 | lane-change-related drift |
| event_lane_change_bin | no_lane_change | 4315 | 2342 | 0.1858 | drift also persists without lane-change |
| event_yielding_bin | non_yielding_like | 4524 | 2415 | 0.2169 | strong non-yielding-like drift |
| event_yielding_bin | yielding_like | 393 | 2502 | 0.1837 | yielding proxy exposure is imbalanced |
| event_lateral_activity_bin | high | 430 | 2621 | 0.1881 | high lateral activity drift |
| event_lateral_activity_bin | mid | 1375 | 1333 | 0.1399 | moderate drift |
| event_lateral_activity_bin | low | 3112 | 963 | 0.2931 | strongest drift in low lateral activity bin |

Notes:
- `event_low_speed_bin` and `event_high_speed_bin` are currently unknown/unavailable in both experiments.
- All reported p-values are significant at approximately `0.0099` in the current run.
- Behavior-event bins are diagnostic/reporting bins, not primary fairness-control bins.

## Main interpretation update

Behavior-event decomposition confirms that pseudo_agg_vs_cons drift is behaviorally interpretable. Its largest BDD values occur in high-lateral-activity and lane-change bins, where BDD reaches approximately 0.2403 and 0.2314. This is consistent with the pseudo aggressive-vs-conservative construction and shows that the behavior-event report layer can localize where style drift is expressed.

For scene_confounding, behavior-event decomposition shows a different pattern. The drift is not explained by static map ODD balancing, and the largest behavior-event BDD values appear in low lateral activity, no-cut-in proxy, non-yielding-like, and lane-change/no-lane-change bins. This suggests that the scene_confounding split captures dynamic interaction-exposure and behavior-proxy differences rather than pure static map ODD mismatch.

## Conceptual clarification

Stage 6 now distinguishes three layers:

1. Static Map ODD control
   - map complexity
   - lane-count context
   - curvature
   - used for fairness control

2. Dynamic interaction / behavior-exposure diagnostics
   - following
   - cut-in proxy
   - yielding proxy
   - lane-change
   - lateral activity
   - used for drift localization

3. Overall behavior distribution drift
   - raw BDD
   - ODD-balanced BDD
   - behavior-event BDD

Map ODD bins and behavior-event bins should not be confused. Static ODD bins test whether A/B faced similar road geometry. Behavior-event bins explain which driving tasks or interaction modes contain the drift.

## Added limitations

- low/high speed bins are unavailable in the current feature schema.
- cut-in and yielding bins are proxy definitions, not ground-truth event annotations.
- several behavior-event bins are highly A/B imbalanced, so they should be interpreted as localization signals rather than causal proof.
- dynamic interaction exposure matching should be developed in a later Stage 6C/6D.

## Stage 6C：Dynamic Interaction Exposure 与 Event-specific Style Diagnosis

Stage 6C 是 Stage 6A/6B 之后新增的动态交互诊断层。它不重写 Stage 6A，也不删除 Stage 6B 的 ODD / behavior-event 命令。核心区别是：`exposure_*` 表示动态交互暴露，可作为后续 matching/control 候选；`outcome_*` 表示行为结果/风格，主要用于报告和定位。

### 1. 命令

先设置当前 full51 数据路径：

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json
```

A. 构建动态交互暴露与行为结果 bins：

```bash
python tools/stage6c_build_dynamic_event_bins.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --output_dir $DATA_ROOT/dynamic_event_bins_v1 \
  --overwrite
```

B. 构建事件内 style metrics：

```bash
python tools/stage6c_build_event_style_metrics.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --dynamic_event_bins_path $DATA_ROOT/dynamic_event_bins_v1/dynamic_event_bins.csv \
  --output_dir $DATA_ROOT/event_style_metrics_v1 \
  --overwrite
```

C. 对 `scene_confounding` 生成 event style report（默认 `--event_scope all`，同时报告 exposure 与 outcome）：

```bash
python tools/stage6c_event_style_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --dynamic_event_bins_path $DATA_ROOT/dynamic_event_bins_v1/dynamic_event_bins.csv \
  --event_style_metrics_path $DATA_ROOT/event_style_metrics_v1/event_style_metrics.csv \
  --output_dir outputs/stage6C_event_style/scene_confounding \
  --event_scope all \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

D. 对 `pseudo_agg_vs_cons` 生成 event style report（exposure-only）：

```bash
python tools/stage6c_event_style_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --dynamic_event_bins_path $DATA_ROOT/dynamic_event_bins_v1/dynamic_event_bins.csv \
  --event_style_metrics_path $DATA_ROOT/event_style_metrics_v1/event_style_metrics.csv \
  --output_dir outputs/stage6C_event_style/pseudo_agg_vs_cons_exposure_only \
  --event_scope exposure \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```


E. 对 `pseudo_agg_vs_cons` 只报告 outcome bins，用于定位 lane-change / overtake / brake / hesitation / assertive / stop-go / lateral-unstable 等行为结果：

```bash
python tools/stage6c_event_style_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --dynamic_event_bins_path $DATA_ROOT/dynamic_event_bins_v1/dynamic_event_bins.csv \
  --event_style_metrics_path $DATA_ROOT/event_style_metrics_v1/event_style_metrics.csv \
  --output_dir outputs/stage6C_event_style/pseudo_agg_vs_cons_outcome_only \
  --event_scope outcome \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

说明：`--event_scope` 的默认值是 `all`。如果传入 `--event_keys exposure_following,outcome_hard_brake` 这类显式列表，则 `--event_keys` 会覆盖 `--event_scope`。

如只想做无进度条的日志运行，可在三个命令后追加：

```bash
--no_progress
```

### 2. 期望行为

- `stage6c_build_dynamic_event_bins.py` 会读取 `shard_manifest.json` 指向的每个 shard 下 `interaction_feat_style.npy`，并用 `feature_schema.json` 做 alias resolution；输出 row-aligned 的 `dynamic_event_bins.csv/.npy`、schema、report、warnings；同时会尝试从 `metadata.csv`、`meta.csv`、`meta.npy` 透传安全 metadata 字段。关键 exposure 条件缺失时会 fail-closed 输出 `unknown`，不会因为 `combine_and` 忽略 None 而放宽分箱。
- dynamic event bins 至少包含 `global_row`、`shard_id`、`local_row`、9 个 `exposure_*` bins、8 个 `outcome_*` bins、`event_quality_flag`、`available_feature_count`、`missing_feature_count`。
- `stage6c_build_event_style_metrics.py` 会读取同一批 shard 和 `dynamic_event_bins.csv`，输出 row-aligned 的 `event_style_metrics.csv/.npy`、schema、report、warnings。
- 如果某个代理特征缺失，bin 会写成 `unknown`，metric 会写成 `NaN`，并在 warnings JSON 中记录缺失 alias；脚本不会把缺失值静默填成 0。分箱正类过少、比例低于 0.05 / 高于 0.95、全 unknown 会写入 `dynamic_event_bin_warnings.json` 并在 schema/warnings 中标记 `event_validity=degenerate`；metric 的 `valid_rate < 0.01` 会写入 `event_style_metric_warnings.json`。
- `stage6c_event_style_report.py` 会读取 embedding manifest、Stage 6A A/B indices、dynamic bins 和 event style metrics，按 event bin 计算 event-level BDD、metric delta、top drift cases，并生成 markdown report card。默认跳过 `event_validity=degenerate` 的分箱；只有显式传入 `--include_degenerate_bins` 才会计算这些分箱，且报告不会为退化分箱生成自然语言结论。
- 该流程不会重新训练 Stage 5 embedding，不会重建 Stage 6A split，不会删除 Stage 6B 既有输出。

### 3. 通过标准

- `dynamic_event_bins.csv`、`event_style_metrics.csv` 的 `global_row` 必须从 0 开始并与 shard 顺序严格对齐。
- `dynamic_event_bin_warnings.json` 和 `event_style_metric_warnings.json` 必须清楚列出 resolved features 与 missing feature aliases。
- `exposure_*` 与 `outcome_*` 必须分开解释：exposure 可作为动态 matching/control 候选，outcome 只用于 report/localization。
- `event_bdd_summary.csv` 必须包含 `event_key,event_value,n_A,n_B,bdd_mmd,ci95_low,ci95_high,p_value,effect_size,interpretation,warnings`。
- `event_style_delta.csv` 必须包含 `event_key,event_value,metric_name,n_A_valid,n_B_valid,mean_A,mean_B,delta_B_minus_A,relative_delta_percent,direction_label,interpretation`。
- `top_event_drift_cases.csv` 必须包含 `global_row,source_group,event_key,event_value,embedding_distance_to_opposite_centroid,dominant_style_metrics,shard_id,local_row`，如果可用则包含 `scenario_id,target_agent_id,start,window_len,split`。
- `event_report_card.md` 必须说明 embedding BDD 是统一行为分布测量层，event-specific features 是语义解释层，二者互补而不是互相替代。
- `event_report_card.md` 必须分开展示 valid exposure bins、skipped small bins、skipped degenerate bins、outcome bins；如果请求的 bin 退化，顶部必须出现退化告警。
### Debug validation checklist

1. 检查 `dynamic_event_bin_warnings.json` / `dynamic_event_bin_schema.json`：除非实验设计明确预期，否则不应出现 all-positive 的 `exposure_*` bin；如果 `positive_ratio > 0.95` 或 `< 0.05`，必须标记为 `event_validity=degenerate`。
2. 检查 `event_style_metric_warnings.json` 的 `metric_valid_stats` 和 `score_scale_warnings`：任一 composite score 的 `abs(p99)>100` 或 `abs(p01)>100` 都必须触发 `score_scale_exploded`，正常 debug run 不应再出现 1e11 量级指标。
3. `outcome_stop_go` 在缺少 `stop_count` / `speed_oscillation` 等代理特征时可以保持 unavailable / all unknown；不要用 0 填充伪造 stop-go。
4. `event_report_card.md` 必须列出 skipped degenerate bins；退化 bin 不应出现在自然语言结论中。

- 最低语法检查需通过：

```bash
python -m py_compile \
  tools/stage6c_common.py \
  tools/stage6c_build_dynamic_event_bins.py \
  tools/stage6c_build_event_style_metrics.py \
  tools/stage6c_event_style_report.py

python -m py_compile \
  tools/stage6b_build_behavior_event_bins.py \
  tools/stage6b_bin_bdd_report.py
```

---

# Stage 6C v2：Behavior-event task slices

## 1. 命令

生成 v2 behavior-event bins 与 task 内解释指标：

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage5_context/shard_manifest.json \
  --feature_schema_path outputs/stage5_context/feature_schema.json \
  --output_dir outputs/stage6c_behavior_events_v2 \
  --overwrite
```

如果只想做无进度条的批处理运行：

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage5_context/shard_manifest.json \
  --feature_schema_path outputs/stage5_context/feature_schema.json \
  --output_dir outputs/stage6c_behavior_events_v2 \
  --overwrite \
  --no_progress
```

生成后，后续 task-conditioned BDD 应优先使用 `behavior_event_bins_v2.csv` 中的 primary task 正类切片，例如：

```text
following == positive
lane_change == positive
overtake == positive
cutin_response == positive
hesitation == positive
yield_conflict == positive
```

建议在以下三类 split 上分别运行 task-conditioned BDD：

```text
negative_control_random
pseudo_agg_vs_cons
scene_confounding_control
```

## 2. 期望行为

该命令会读取 `shard_manifest` 指向的 sharded dataset，并按 shard 顺序逐行处理 raw arrays。脚本优先使用每个 shard 内的：

- `ego_seq.npy`；
- `neighbor_seq.npy`；
- `neighbor_slot_ids.npy`（存在时用于一致性/诊断）；
- `meta.npy`；
- `interaction_feat_style.npy`（存在时用于一致性检查；v2 不只依赖 33 个 aggregate features）。

脚本会在输出目录生成：

- `behavior_event_bins_v2.csv`：每个 event detector 的 `positive` / `negative` / `unknown` 标签；
- `behavior_event_metrics_v2.csv`：每个 task 的 handcrafted style explanation metrics；
- `behavior_event_schema_v2.json`：taxonomy、阈值、array layout 假设、event diagnostics、metric diagnostics；
- `behavior_event_report_v2.md`：可读诊断报告；
- `behavior_event_warnings_v2.json`：缺失文件、metadata mismatch、完成状态等 warning。

脚本会保留行对齐字段：`global_row`、`shard_id`、`local_row`，并尽量透传 `scenario_id`、`target_agent_id`、`start`、`window_len`、`split`。脚本不会默认合并 `ego_seq.npy` / `neighbor_seq.npy` 等大数组；缺失或不可计算的 metric 会保留为 `NaN`，不会填 0。

v2 的解释逻辑是：event bin 是可比较驾驶任务切片，BDD 在 task 内计算；`behavior_event_metrics_v2.csv` 中的 THW、gap、decel、jerk、sharpness、yielding/assertiveness 等指标只用于解释 drift 方向，不作为主评价对象。

## 3. 通过标准

- `behavior_event_bins_v2.csv` 行数必须等于所有 shard 的 window 总数，且 `global_row` 从 0 连续递增。
- 每个 primary event 都必须只包含三种状态：`positive`、`negative`、`unknown`。
- `behavior_event_schema_v2.json` 必须包含每个 event 的 `positive_ratio`、`unknown_ratio`、`n_positive`、`n_negative` 与 `degenerate` 标记。
- 当某个 event 的 `positive_ratio < 0.01` 或 `positive_ratio > 0.95` 时，必须被标记为 `degenerate=true`，后续 BDD 报告不得把它当成稳定结论。
- `behavior_event_metrics_v2.csv` 不得用 0 伪造缺失指标；无法计算的 metric 应为 `NaN`。
- `behavior_event_schema_v2.json` 必须包含 metric diagnostics：`valid_count`、`valid_rate`、`p01`、`p50`、`p99`、`min`、`max`。
- 主报告应强调 exposure/task-conditioned BDD，而不是 outcome bins；handcrafted metrics 只解释 drift 方向。

# Stage 6C v2 — Task-conditioned behavior-event BDD

Stage 6C v2 的主目标是：在相同 driving task / behavior-event slice 内，用 learned behavior embedding 的 BDD 检测 A/B policy 或 model version 的 style distribution drift，再用 task-specific metrics 解释 drift 方向。旧版 Stage 6C 的 outcome bins（例如 hard_brake、late_brake）保留为 legacy / post-hoc diagnostic，不作为 v2 主报告对象。

## 1. 命令

### 1.1 编译检查

```bash
python -m py_compile \
  tools/stage6c_build_behavior_events_v2.py \
  tools/stage6c_task_conditioned_bdd_report.py
```

### 1.2 设置数据路径

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
TRAIN_OUT=$DATA_ROOT/context_gru_stage5d_balanced_v2
EMBEDDING_MANIFEST=$TRAIN_OUT/embeddings/embedding_manifest.json

```

### 1.3 构建 Stage 6C v2 task-conditioned behavior events

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --output_dir $DATA_ROOT/behavior_events_v2 \
  --overwrite \
  --no_progress
```

### 1.4 negative_control_random

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/negative_control_random_v2 \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite \
  --no_progress
```

### 1.5 pseudo_agg_vs_cons

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2 \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite \
  --no_progress
```

### 1.6 scene_confounding_control

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/scene_confounding_v2 \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite \
  --no_progress
```

### 1.7 生成 remaining Stage 6A final splits

negative control random split：

```bash
python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random \
  --seed 42 \
  --overwrite
```

scene confounding split：

```bash
python tools/stage6_build_ab_splits.py \
  --mode scene_confounding_control \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name scene_confounding \
  --seed 42 \
  --overwrite
```

说明：

- `negative_control_random` 只需要 eval split row 集合；如果本机没有原始 `interaction_feat_style.npy` shard，脚本会从 `$DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv` 的 `global_row/split` 字段生成 random split。
- `scene_confounding_control` 优先使用 feature shard 构造 scene complexity score；如果本机没有原始 `interaction_feat_style.npy` shard，脚本会 fallback 到 `$DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv` 的 lateral / interaction pressure / gap 类数值指标，并在 `split_summary.json` 写 warning。

### 1.8 negative_control_random_v2_final

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/negative_control_random_v2_final \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite \
  --no_progress
```

### 1.9 scene_confounding_v2_final

```bash
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/scene_confounding_v2_final \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --top_k 20 \
  --seed 42 \
  --overwrite \
  --no_progress
```

### 1.10 汇总三路 final experiment

```bash
python tools/stage6c_summarize_task_bdd_experiments.py \
  --experiment_dirs outputs/stage6C_task_bdd/negative_control_random_v2_final,outputs/stage6C_task_bdd/pseudo_agg_vs_cons_v2_final,outputs/stage6C_task_bdd/scene_confounding_v2_final \
  --experiment_names negative,pseudo,scene \
  --output_dir outputs/stage6C_task_bdd/stage6c_v2_cross_experiment_summary \
  --overwrite
```

## 2. 期望行为

- `stage6c_build_behavior_events_v2.py` 读取每个 shard 下可用的 `ego_seq.npy`、`neighbor_seq.npy`、`neighbor_slot_ids.npy`、`metadata.csv` / `meta.csv` / `meta.npy`、`interaction_feat_style.npy`，优先使用 raw sequences 构建 task-conditioned behavior events。
- 构建脚本输出：
  - `behavior_event_bins_v2.csv`
  - `behavior_event_metrics_v2.csv`
  - `behavior_event_schema_v2.json`
  - `behavior_event_report_v2.md`
  - `behavior_event_warnings_v2.json`
- `behavior_event_bins_v2.csv` 与 `behavior_event_metrics_v2.csv` 通过 `global_row` 逐行对齐，并尽量保留 `shard_id`、`local_row`、`scenario_id`、`target_agent_id`、`start`、`window_len`、`split`。
- 每个 task detector 输出 positive label / negative label / `unknown`，不会把缺失 raw signal 静默填成 negative。
- 不可用的 style metric 写为 `NaN`，不会静默填 0。
- 如果 `neighbor_seq.npy` 或 `neighbor_slot_ids.npy` 缺失，cut-in、yield conflict、lead brake、queue、overtake 等 detector 会记录 warning，并在必要时使用 conservative proxy 或输出 `unknown`。
- `stage6c_task_conditioned_bdd_report.py` 只在 positive task label 内计算 A/B embedding BDD，并输出：
  - `task_bdd_summary.csv`
  - `task_style_delta.csv`
  - `task_report_card.md`
  - `top_task_drift_cases.csv`
  - `warnings.json`
  - `plots/task_bdd_bar.png`
  - `plots/task_style_delta_bar.png`
- BDD 报告默认跳过 degenerate / all_unknown tasks；如需调试可加 `--include_degenerate_tasks`。

## 3. 通过标准

1. `py_compile` passes。
2. `behavior_event_bins_v2.csv` 行数等于 dataset row count。
3. `behavior_event_metrics_v2.csv` 行数等于 dataset row count。
4. `global_row` 唯一且 bins / metrics 逐行对齐。
5. metadata 在 shard 中存在时被保留到 v2 输出。
6. 除非 raw signals 不可用，否则重要 task 不应全部为 `unknown`。
7. degenerate tasks 被写入 `behavior_event_warnings_v2.json` / `warnings.json`，且 BDD report 默认跳过。
8. `negative_control_random` 不应出现系统性高 task BDD。
9. `pseudo_agg_vs_cons` 应在 style-relevant tasks 中出现有意义的 task-conditioned BDD。
10. `scene_confounding_control` 应揭示 dynamic task / exposure confounding patterns。

## 4. 三路 final experiment 解释 checklist

1. `negative_control_random` 应低且非系统性；如果它在多个 primary task 上也很高，需要先怀疑 split / confounding / metric leakage。
2. `pseudo_agg_vs_cons` 应在 behavior-style tasks 上出现稳定 BDD；它是 positive control，不是真实 model A/B 结论。
3. `scene_confounding` 可以出现 BDD，但 pattern 应与 pseudo 不同，用于诊断 scene / exposure confounding。
4. Primary conclusions 优先使用 strong detector tasks：`task_following`、`task_lane_change`、`task_yield_conflict`、`task_hesitation`。
5. Proxy-heavy tasks 标记为 auxiliary：`task_cutin_response`、`task_queue_approach`、`task_lead_brake_response`、`task_overtake_opportunity`、`task_overtake_executed`。
6. 如果 `task_overtake_executed` 因 sample size 被 skipped，不要解释为 no drift。

## 5. Stage 6C v2 调试前 validation checklist

1. `neighbor_slot_ids.npy` 可以成功加载；若该数组为 object dtype，构建脚本应在 `behavior_event_warnings_v2.json` / schema notes 中记录 `neighbor_slot_ids_loaded_with_pickle=true`。
2. TTC metrics 只使用 `neighbor_seq.npy` 的真实 TTC column；如果 shard 缺少 TTC column，则 `lead_brake_min_ttc_after_lead_brake`、`cutin_min_ttc` 等写为 `NaN`，并记录 `ttc_column_unavailable_metric_set_nan`，绝不能把 THW 误标为 TTC。
3. `behavior_event_bins_v2.csv` 必须包含每个 task 的 detector strength 列，例如 `task_following_strength`、`task_lead_brake_response_strength`、`task_queue_approach_strength`、`task_cutin_response_strength` 等。
4. 解释 cut-in / lead-brake / queue BDD 前，必须先检查 `behavior_event_warnings_v2.json` 中的 `cutin_true_slot_transition_not_implemented_using_gap_drop_proxy`、`lead_brake_selective_detector_enabled`、`queue_approach_uses_gap_thw_closing_proxy` 等 warning。
5. `task_bdd_summary.csv` 与 `task_report_card.md` 必须展示 `dominant_detector_strength` 和 `detector_strength_counts`；如果 dominant strength 是 `proxy` 或 `weak_proxy`，只能解释为 proxy detector 下的 task-conditioned BDD。

## Stage 6C v2 smoothing / clipping 与 detector strength 复核

## 1. 命令

重新构建 behavior-event v2（默认启用 5 帧平滑与物理裁剪）：

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --output_dir $DATA_ROOT/behavior_events_v2 \
  --smoothing_window 5 \
  --enable_signal_smoothing \
  --accel_min_cap -12 \
  --accel_max_cap 8 \
  --jerk_abs_cap 80 \
  --yaw_rate_abs_cap 2 \
  --lateral_accel_abs_cap 8 \
  --curvature_abs_cap 1 \
  --lateral_speed_abs_cap 5 \
  --heading_change_total_cap 8 \
  --ttc_valid_max_s 30 \
  --thw_valid_max_s 30 \
  --lane_change_lateral_range_m 2.5 \
  --lane_change_min_lateral_range_m 1.5 \
  --hesitation_sign_changes 8 \
  --hesitation_min_evidence_count 2 \
  --overwrite
```

运行 queue strong-only sensitivity check：

```bash
export EMBEDDING_MANIFEST=$TRAIN_OUT/embeddings/embedding_manifest.json
python tools/stage6c_task_conditioned_bdd_report.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --behavior_event_bins_path $DATA_ROOT/behavior_events_v2/behavior_event_bins_v2.csv \
  --behavior_event_metrics_path $DATA_ROOT/behavior_events_v2/behavior_event_metrics_v2.csv \
  --output_dir outputs/stage6C_task_bdd/pseudo_agg_vs_cons_queue_strong_only_v2 \
  --task_keys task_queue_approach \
  --detector_strength_filter strong \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --min_bin_size 100 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

- 构建脚本会先对 speed、accel、yaw_rate、lateral velocity 做平滑，再计算 jerk、lateral_accel、curvature 等 derivative-sensitive metrics。
- `behavior_event_metrics_v2.csv` 写入的是用于正式 Stage 6C v2 分析的 smoothed/clipped metrics；raw diagnostic 不进入下游 metric delta 主表。
- `behavior_event_schema_v2.json` 会记录 `raw_metric_diagnostics`、`clipped_metric_diagnostics`、`metric_quality_warnings`，用于检查原始 finite-difference 噪声是否超过物理范围。
- TTC/THW 会在加载后清理：`>=999`、`<=0`、超过 `--ttc_valid_max_s` / `--thw_valid_max_s` 的值写为 `NaN`，正式 metrics 和 diagnostic scores 不应再出现 999 哨兵值。
- `lc_max_lateral_speed` 使用裁剪后的 lateral speed；`lc_heading_change_total`、lane-change detector、hesitation detector 使用封顶后的 heading-change total，并在 raw/clipped diagnostics 中保留 `raw_max_lateral_speed`、`clipped_max_lateral_speed`、`raw_heading_change_total`、`clipped_heading_change_total`。
- lane-change detector 需要足够 lateral displacement；yaw_rate 或 heading_change 不能单独触发。若 `task_lane_change` 的 `positive_ratio > 0.40`，`behavior_event_warnings_v2.json` 和报告中会出现 `lane_change_detector_broad`。
- hesitation detector 需要 maneuver context 且至少两个 evidence components；默认 `--hesitation_sign_changes 8`、`--hesitation_min_evidence_count 2`。若 `task_hesitation` 的 `positive_ratio > 0.40`，会出现 `hesitation_detector_broad`。
- lead-brake detector 优先使用 front_speed 的持续减速度；front_speed 不可靠时才使用 closing-rate derivative proxy，并继续在 strength column 中区分 `strong` / `proxy`。
- following 与 yield_conflict 目前是最可靠的 strong detectors；cutin、overtake 以及相当一部分 lead/queue 仍是 proxy-based；lane_change 与 hesitation 只有在收紧后 positive_ratio 不 broad 时才建议作为稳定结论。
- `--detector_strength_filter strong` 会在 BDD 前只保留 positive rows 中 detector strength 为 `strong` 的样本，并在 `task_bdd_summary.csv` 中同时报告过滤前后的 `n_A/n_B`。

## 3. 通过标准

1. `behavior_event_bins_v2.csv` 仍包含全部 `task_*` 列和对应 `task_*_strength` 列。
2. `behavior_event_metrics_v2.csv` 仍通过 `global_row` 与 bins 文件逐行对齐。
3. `behavior_event_schema_v2.json` 中 final metric diagnostics 的 decel p99/max 不应超过约 12 m/s²，jerk 不应超过约 80 m/s³，yaw_rate 不应超过约 2 rad/s，lateral_accel 不应超过约 8 m/s²，curvature 不应超过约 1。
4. TTC/THW 相关 final metrics（如 `lead_brake_min_ttc_after_lead_brake`、`lead_brake_min_thw_after_lead_brake`、`cutin_min_ttc`、`following_mean_thw`）的 `max` 不应等于 999，也不应超过配置的有效上限。
5. `queue_distance_when_start_decel` 不应出现 final `metric_physical_range_warning`，因为它是距离指标，不是减速度指标。
6. 如果 raw diagnostics 超出物理范围，应出现 `raw_metric_physically_implausible` / `metric_physical_range_warning`；如果 final diagnostics 仍超范围，应先停止正式分析并检查数据或阈值。
7. `task_lane_change` 的 `positive_ratio` 理想上应降到 0.40 以下；若仍大于 0.40，必须在 `behavior_event_report_v2.md` / `behavior_event_warnings_v2.json` 中出现 `lane_change_detector_broad`。
8. `task_hesitation` 的 `positive_ratio` 理想上应降到 0.40 以下；若仍大于 0.40，必须在 `behavior_event_report_v2.md` / `behavior_event_warnings_v2.json` 中出现 `hesitation_detector_broad`。
9. `task_lead_brake_response` 的 positive rows 不应几乎等同于 `task_following`，并应检查 `task_lead_brake_response_strength` 的 strong/proxy 分布。
10. `task_bdd_summary.csv` 应包含 `bootstrap_mean`、`bootstrap_std`、`observed_in_bootstrap_ci`、`mmd_estimator_config`。
11. `task_report_card.md` 应展示 detector strength、过滤前后样本数、bootstrap CI 一致性，以及 metric quality warnings。
Stage5A summary by 刘庆
4 路并行命令
0-13
13-26
26-39
39-51

每个开一个终端：

python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
  --file_start 0 \
  --file_end 13 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --drop_if_lane_context_ambiguous \
  --streaming \
  --output_shard_size 5000 \
  --overwrite
  然后把 --file_start / --file_end / --out_dir 分别改成：

13 / 26 / outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26
26 / 39 / outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39
39 / 51 / outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51

****然后就是Merge这4个shards到一起
python tools/merge_waymo_5neighbor_context_shards.py \
 --input_roots \
 outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
 outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26 \
 outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39 \
 outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51 \
 --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
 --recompute_global_standardization \
 --overwrite

****然后训练
Stage 5D-balanced-v2

balanced-v2 相比 v1：

降低 following 权重
提高 lateral dynamics 权重
保留 following 强化，同时恢复 lateral

训练命令：

python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_loss_type huber \
  --style_loss_weight 1.0 \
  --aux_longitudinal_weight 0.5 \
  --aux_following_weight 1.2 \
  --aux_lateral_dynamics_weight 1.5 \
  --aux_lateral_gap_weight 1.0 \
  --aux_behavior_proxy_weight 0.5 \
  --metric_longitudinal_weight 0.5 \
  --metric_following_weight 1.5 \
  --metric_lateral_dynamics_weight 1.5 \
  --metric_lateral_gap_weight 1.0 \
  --metric_behavior_proxy_weight 0.5 \
  --device cuda \
  --seed 42 \
  --overwrite

注意：

Stage 5D 不再使用 --metric_alignment
而是使用 group-specific metric weights

balanced-v2 结果：

hit@1 = 0.213092
hit@5 = 0.526232
mean_same_label_fraction_at_5 = 0.189776
longitudinal_comfort = 0.171751
following_interaction = 0.501998
lateral_lane_dynamics = 0.245608
behavior_proxy = 0.322344

结论：

Stage 5D-balanced-v2 是当前 Stage 5 推荐模型；
它是目前最好的 learned trade-off 表示；
global retrieval 仍未超过 raw/pca feature，但在 following_interaction 和 behavior_proxy 上胜过 raw/pca，在 longitudinal 和 lateral 上接近 raw/pca。

## Stage 5D-balanced-v2 Commands

训练命令（Stage 5D 不使用 `--metric_alignment`）：

```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_loss_type huber \
  --style_loss_weight 1.0 \
  --aux_longitudinal_weight 0.5 \
  --aux_following_weight 1.2 \
  --aux_lateral_dynamics_weight 1.5 \
  --aux_lateral_gap_weight 1.0 \
  --aux_behavior_proxy_weight 0.5 \
  --metric_longitudinal_weight 0.5 \
  --metric_following_weight 1.5 \
  --metric_lateral_dynamics_weight 1.5 \
  --metric_lateral_gap_weight 1.0 \
  --metric_behavior_proxy_weight 0.5 \
  --device cuda \
  --seed 42 \
  --overwrite
```

导出命令：

```bash
export DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
export SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
export TRAIN_OUT=$DATA_ROOT/context_gru_stage5d_balanced_v2

python tools/export_context_row_embeddings.py \
  --shard_manifest $SHARD_MANIFEST \
  --checkpoint $TRAIN_OUT/model.pt \
  --out_dir $TRAIN_OUT/embeddings \
  --batch_size 512 \
  --split all \
  --device cpu \
  --overwrite
```

评估命令：

```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

## Stage 7A.0 — nuPlan mini readiness check

### 1. 命令

在 nuPlan Python 3.9 环境中运行轻量 readiness checker：

```bash
python tools/stage7a_check_nuplan_mini.py \
  --output_dir outputs/stage7A_nuplan/mini_check \
  --overwrite
```

如需显式指定数据路径，可补充 `--nuplan_data_root`、`--nuplan_maps_root` 或 `--mini_db_dir`。未显式提供 `--mini_db_dir` 时，脚本会依次查找：

1. `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini`
2. `$NUPLAN_DATA_ROOT/data/cache/mini`

### 2. 期望行为

该命令只做 Stage 7A.0 的轻量数据就绪检查：

- 扫描 nuPlan mini SQLite DB 和 maps root 下的 `map.gpkg` 文件。
- 统计 mini DB 数量、DB 打开状态、关键表是否存在以及关键表行数。
- 对 `scenario_tag`、`ego_pose`、`lidar_pc`、`lidar_box`、`track` 抽样导出 CSV。
- 写出 inventory CSV、sample CSVs、schema JSON、warnings JSON 和 Markdown report。
- BLOB / bytes 字段会转换为 hex 字符串后写入 CSV / JSON。
- 单个 DB 打开失败时记录 warning 并继续扫描其他 DB。
- maps root 中没有 `map.gpkg` 时写入 warning，但不崩溃。
- 输出目录已存在且未传 `--overwrite` 时抛出 `FileExistsError`，避免覆盖旧结果。
- 不调用 nuPlan planner / simulation API。
- 不生成 fake rollout data。
- 不修改 Stage 6C 结果文件，也不修改既有 BDD 逻辑。

期望生成：

- `mini_db_inventory.csv`
- `mini_scenario_tags_sample.csv`
- `sample_ego_pose_rows.csv`
- `sample_lidar_pc_rows.csv`
- `sample_lidar_box_rows.csv`
- `sample_track_rows.csv`
- `mini_schema_report.json`
- `warnings.json`
- `mini_check_report.md`

### 3. 通过标准

Stage 7A.0 passes if：

- `mini DB count > 0`。
- `map.gpkg count = 4`。
- `DB open failure count = 0`。
- key tables exist：`log`、`scene`、`scenario_tag`、`ego_pose`、`lidar_pc`、`lidar_box`、`track`、`category`、`traffic_light_status`。
- `mini_check_report.md`、`warnings.json`、`mini_db_inventory.csv` 均成功生成。

Next step：implement expert ego trajectory + nearby object context exporter。

## Stage 7B.1 — Export nuPlan expert ego trajectory and nearby object context

### 1. 命令

在已配置 `NUPLAN_DATA_ROOT` 的 nuPlan mini 环境中运行：

```bash
python tools/stage7a_export_nuplan_expert_context.py \
  --output_dir outputs/stage7A_nuplan/expert_context_export \
  --max_dbs 5 \
  --max_scenes_per_db 5 \
  --max_lidar_pcs_per_scene 200 \
  --num_neighbors 10 \
  --overwrite
```

如需显式指定 mini DB 目录，可补充：

```bash
python tools/stage7a_export_nuplan_expert_context.py \
  --nuplan_data_root /path/to/nuplan \
  --mini_db_dir /path/to/nuplan/nuplan-v1.1/splits/mini \
  --output_dir outputs/stage7A_nuplan/expert_context_export \
  --max_dbs 5 \
  --max_scenes_per_db 5 \
  --max_lidar_pcs_per_scene 200 \
  --num_neighbors 10 \
  --overwrite
```

未显式提供 `--mini_db_dir` 时，脚本会依次查找：

1. `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini`
2. `$NUPLAN_DATA_ROOT/data/cache/mini`

### 2. 期望行为

该命令只做 Stage 7B.1 的 expert / historical nuPlan 中间格式导出；它不是最终 Stage 7 policy-style validation，只是在 policy A/B rollout 之前验证 nuPlan SQLite → trajectory/context export：

- 直接读取选中的 nuPlan mini SQLite DB，不调用 nuPlan planner / simulation API。
- 从 `scene`、`lidar_pc`、`ego_pose`、`lidar_box`、`track`、`category` 等表中发现 schema，并尽量解析 token / timestamp / pose / object 关联列。
- 对每个已导出的 `lidar_pc` frame 写出 expert ego trajectory，并在可计算时补充 `ego_speed`、`ego_accel`、`ego_yaw_rate`。
- 对每个 frame 仅保留距离 ego 最近的 `--num_neighbors` 个 object context，并写出相对位置和距离排序。
- 如果缺少列、scene 无法关联到 lidar sequence、frame 缺少 ego pose、没有 object、timestamp 缺失或无法计算速度 / 加速度 / yaw rate，会写入 `warnings.json`，不会伪造数据。
- 输出目录已存在且未传 `--overwrite` 时抛出 `FileExistsError`，避免覆盖旧结果。
- 不运行 planner simulation。
- 不生成 fake rollout data。
- 不修改 Stage 6C result files，也不修改 BDD 逻辑。

期望生成：

- `expert_ego_trajectory.csv`
- `expert_nearby_objects.csv`
- `selected_scenes.csv`
- `warnings.json`
- `expert_context_export_report.md`

### 3. 通过标准

Stage 7B.1 passes if：

- `python -m py_compile tools/stage7a_export_nuplan_expert_context.py` 通过。
- 上面的导出命令成功结束并生成 5 个输出文件。
- `expert_ego_trajectory.csv` 至少包含 1 行数据（不含 header）。
- `expert_nearby_objects.csv` 至少包含 1 行数据（不含 header）。
- `selected_scenes.csv` 记录了实际导出的 DB / scene 和 row count。
- `warnings.json` 中没有会阻断 expert context inspection 的严重 schema / join 问题。
- 没有生成 planner rollout CSV / JSON，也没有生成 fake rollout data。
- Stage 6C 结果目录保持不变。

Interpretation：Stage 7B.1 用于验证在实现 policy A/B rollouts 之前，我们可以先从 nuPlan mini 导出 expert ego trajectory 和 surrounding object context；它本身不证明最终的 same-scenario policy-style separability。

Next step：Convert `expert_ego_trajectory.csv` and `expert_nearby_objects.csv` into our context dataset format：`ego_seq.npy`、`neighbor_seq.npy`、`metadata`、`shard_manifest.json`、`feature_schema.json`。

# Stage 7B.2 — nuPlan expert dynamic context dataset converter

## 1. 命令

把 Stage 7B.1 导出的 expert ego trajectory / nearby objects 转换成 Stage 6C-compatible dynamic context dataset（Waymo 5-neighbor slot layout）：

```bash
python tools/stage7b_convert_expert_context_to_dataset.py \
  --expert_ego_csv outputs/stage7A_nuplan/expert_context_export/expert_ego_trajectory.csv \
  --expert_objects_csv outputs/stage7A_nuplan/expert_context_export/expert_nearby_objects.csv \
  --selected_scenes_csv outputs/stage7A_nuplan/expert_context_export/selected_scenes.csv \
  --output_dir outputs/stage7A_nuplan/expert_context_dataset \
  --target_hz 10 \
  --window_sec 8 \
  --stride_sec 4 \
  --num_neighbors 10 \
  --overwrite
```

可选几何 slot 参数：

```bash
--front_lateral_tolerance 2.5 \
--side_lateral_threshold 2.0 \
--rear_tolerance 5.0 \
--ttc_cap 999.0 \
--thw_cap 999.0
```

Stage 6C smoke check：

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7A_nuplan/expert_context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7A_nuplan/expert_context_dataset/feature_schema.json \
  --output_dir outputs/stage7A_nuplan/expert_behavior_events_smoke \
  --dt 0.1 \
  --overwrite \
  --no_progress
```

## 2. 期望行为

- 读取 `expert_ego_trajectory.csv`、`expert_nearby_objects.csv`，可选读取 `selected_scenes.csv`。
- 按 `db_name` + `scene_token` 分组，按 `frame_index_in_scene` + `lidar_pc_timestamp` 排序。
- 根据时间戳估计 source dt / source_hz，并按 `target_hz` 做稳健下采样；默认把约 20Hz expert export 转为 10Hz。
- 生成固定窗口 dynamic context：默认 `window_sec=8`、`target_hz=10`，所以每个窗口 80 帧；默认 `stride_sec=4`，所以滑窗步长 40 帧。
- 输出改为 sharded Stage 6C layout：
  - `shard_manifest.json`
  - `feature_schema.json`
  - `conversion_report.md`
  - `warnings.json`
  - `shards/shard_000000/ego_seq.npy`
  - `shards/shard_000000/neighbor_seq.npy`
  - `shards/shard_000000/metadata.csv`
  - `shards/shard_000000/meta.npy`
  - `shards/shard_000000/split.npy`
  - `shards/shard_000000/neighbor_slot_ids.npy`
  - `shards/shard_000000/context_traj.npy`
  - `shards/shard_000000/context_mask.npy`
  - `shards/shard_000000/context_mask_window.npy`
  - `shards/shard_000000/interaction_feat_style_raw.npy`
  - `shards/shard_000000/interaction_feat_style.npy`
  - `shards/shard_000000/shard_summary.json`
- `ego_seq.npy` shape 为 `[N, 80, 8]`；特征顺序为 `x, y, vx, vy, heading, speed, accel, yaw_rate`。位置和速度使用窗口 reference frame 下的 local coordinates。
- `neighbor_seq.npy` shape 为 `[N, 5, 80, 15]`；slot 顺序为 `front, left_front, left_rear, right_front, right_rear`；特征顺序为 `valid, dx, dy, rvx, rvy, distance, local_x, local_y, closing_rate, ttc, thw, neighbor_speed, neighbor_accel, relative_heading, neighbor_yaw_rate`。
- `context_traj.npy` shape 为 `[N, 80, 83]`，按 Waymo 5-neighbor context builder 对齐：每帧拼接 `ego_seq[i]` 的 8 维特征和 `neighbor_seq[i]` 按时间展开后的 `5*15=75` 维邻车特征。
- `context_mask.npy` shape 为 `[N, 80, 5]`，由 `neighbor_seq[..., 0] > 0.5` 的 valid 标志转置到时间维在前得到。
- `context_mask_window.npy` shape 为 `[N, 5]`，表示每个窗口内每个 neighbor slot 是否至少出现过一次有效目标。
- `metadata.csv` 至少包含 Stage 6C 必需列：`scenario_id, target_agent_id, start, window_len, split`，并保留 expert/source、scene、frame、timestamp、map/ODD status、slot assignment mode 等列。
- `split` 根据 `scenario_id=db_name|scene_token` 做 deterministic hash split：train 80%、val 10%、test 10%。
- `shard_manifest.json` 必须包含 `dataset_type=nuplan_expert_context_stage6c_compatible` 和 `shard_paths=["shards/shard_000000"]`。
- Stage 7B.2 只使用 geometric slot assignment；`warnings.json` 和 `conversion_report.md` 会说明 map/lane-aware assignment 将在 Stage 7B.3/7B.4 改进。
- `conversion_report.md` 会显式记录 `context_traj`、`context_mask`、`context_mask_window` 的 shape，并说明：Stage 7B.2 dynamic outputs are aligned with the Waymo 5-neighbor context dataset layout except map/ODD features, which are reserved for Stage 7B.3.
- 本阶段只转换 dynamic context，不解析 map，不运行 planner simulation，不生成 fake rollout，不修改 Stage 6C result files，不修改 BDD 逻辑。
- `feature_schema.json` 会保留 Stage 6-style `map_odd_features_reserved`，供 Stage 7B.3 对齐，并记录 interaction feature schema note。

## 3. 通过标准

1. `python -m py_compile tools/stage7b_convert_expert_context_to_dataset.py` passes。
2. 上述转换命令可以成功运行，并生成 root-level `shard_manifest.json`、`feature_schema.json`、`conversion_report.md`、`warnings.json`。
3. `shards/shard_000000/ego_seq.npy` 存在，shape 为 `[N, 80, 8]`，且 `N > 0`。
4. `shards/shard_000000/neighbor_seq.npy` 存在，shape 为 `[N, 5, 80, 15]`，且第一维与 ego window 数一致。
5. `shards/shard_000000/context_traj.npy` 存在，shape 为 `[N, 80, 83]`，且第一维与 ego window 数一致。
6. `shards/shard_000000/context_mask.npy` 存在，shape 为 `[N, 80, 5]`；`context_mask_window.npy` 存在，shape 为 `[N, 5]`。
7. `shards/shard_000000/metadata.csv` 行数等于 `N`，且包含 `scenario_id, target_agent_id, start, window_len, split`。
8. `shards/shard_000000/split.npy` 存在，长度等于 `N`。
9. `shard_manifest.json` 包含 `shard_paths`、`map_odd_feat_path: null`、`map_feature_status: not_built`、`next_map_stage: Stage 7B.3 map/ODD feature builder`。
10. `feature_schema.json` 包含 canonical ego / neighbor / slot schema 以及 `map_odd_features_reserved`。
11. Stage 6C smoke 命令不会因为 array shape、missing shard paths、missing metadata 而失败；允许部分 detector 因 geometric slots 输出 proxy / weak_proxy warning。
12. 不生成 fake data，不修改 Stage 6C result files，不修改 BDD 逻辑。

# Stage 7B.3 — nuPlan map/ODD feature builder（历史小节，已更新）

> 当前 Stage 7B.3 已实现；最新 copy-paste 命令、输出文件、shape、PASS criteria 和 failure modes 以本文档上方 `Stage 7B.3 — Map/ODD feature extraction` 小节为准。保留本历史小节仅用于说明 Stage 7B.2 最初预留的接口已经落地。

## 1. 命令

```bash
python tools/build_nuplan_map_odd_features.py \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --input_dynamic_dir outputs/stage7A_nuplan/expert_context_dataset \
  --output_dir outputs/stage7b3_nuplan_map_odd \
  --split mini \
  --max_scenarios 50 \
  --radius_m 50.0 \
  --sample_stride 5 \
  --overwrite
```

## 2. 期望行为

- 读取 Stage 7B.2 dynamic context dataset。
- 使用 nuPlan map API 构建与 dynamic window 行对齐的 map/ODD-lite features。
- 输出 `map_odd_feat.npy`、`map_odd_meta.csv`、`map_odd_feature_schema.json`、`map_odd_report.md`、`warnings.json`。
- 不运行 planner simulation，不生成 pseudo rollout，不修改 Stage 7B.2 输出。

## 3. 通过标准

1. `map_odd_feat.npy` 是二维 `[N, F_map]`，latest verified mini run 为 `[23, 37]`。
2. `map_odd_meta.csv` 行数与 dynamic context rows 对齐，latest verified mini run 为 23 行。
3. `warnings.json` 为结构化 JSON，latest verified result 为 `warnings: []`、`map_odd_status: PASS`。
4. 所有 map/ODD features finite，且 feature schema 长度等于数组列数。

## Stage 7D：从 official nuPlan msgpack 提取 mandatory neighbor tensors

## 1. 命令

```bash
python tools/stage7d_extract_neighbors_from_nuplan.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --max_neighbors 16 \
  --overwrite
```

提取完成后，再运行 Stage 6-compatible exporter。

## 2. 期望行为

该命令读取 Stage 7C.2C-2 official nuPlan simulation 输出目录中的 `simulated_ego_seq.npy`、`simulated_ego_seq_mask.npy`、`scenario_planner_index.csv`、`simulated_planner_metadata.csv`、`simulation_schema.json`、`warnings.json`，并优先解析 `official_nuplan_runs/**/*.msgpack.xz` 中的 observations / tracked objects。脚本不会运行 nuPlan simulation，不会生成 pseudo rollout，不会把 background agents 展开成 ego rows。

输出写回同一个 `--sim_dir`：

- `stage7d_neighbor_seq.npy`：mandatory neighbor tensor，layout 为 `[rows, K, T, 9]`；
- `stage7d_neighbor_slot_ids.npy`：与 neighbor slot 对齐的 `[rows, K]` ID；
- `stage7d_neighbor_schema.json`：记录 row semantics、neighbor channels、official simulation 标记；
- `stage7d_neighbor_report.md`：中文/英文可读的提取报告；
- `stage7d_neighbor_warnings.json`：低覆盖率等 warning，不伪造 neighbor。

neighbor channel 顺序固定为：`rel_x, rel_y, rel_vx, rel_vy, distance, bearing, heading_rel, speed, valid`。其中 `rel_*` 必须相对每一条 planner-controlled simulated ego trajectory 重新计算；即使 closed-loop nonreactive agents 在同一 scenario 的不同 planner 下 world-coordinate background trajectories 相同，也必须针对不同 planner ego rollout 重新投影为 ego-centric neighbor features。

## 3. 通过标准

命令通过时必须满足：

- `simulation_schema.json` 中 `pseudo_rollout == false`，且 `uses_official_nuplan_simulation == true`；
- 输出行数等于 `num_scenarios * num_planners`，当前 5 logs × 4 planners 应为 20；
- `stage7d_neighbor_seq.npy` shape 为 `[20, K, T, 9]`，其中 `T` 与 `simulated_ego_seq.npy` 一致，最后一维必须等于 9；
- `stage7d_neighbor_slot_ids.npy` shape 为 `[20, K]`；
- valid flag 不能全为 0；
- `stage7d_neighbor_seq.npy` 不能包含 NaN 或 `+/-inf`；
- 低 neighbor coverage 只写入 warning，不能凭空 fabricate neighbors；
- row semantics 保持 one row = one scenario × one planner-controlled nuPlan ego rollout，不允许 multi-agent ego expansion。

## Stage 7D：完整 Stage 6-compatible planner 数据导出

## 1. 命令

```bash
python tools/stage7d_export_stage6_compatible_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7d_stage6_dataset_official_planner_5logs \
  --required_planners simple_planner idm_longitudinal_conservative idm_longitudinal_comfort idm_longitudinal_aggressive \
  --overwrite
```

## 2. 期望行为

该命令只读取 Stage 7C.2C-2 已生成的官方 nuPlan simulation 输出，不运行 nuPlan simulation，不做 pseudo rollout，也不计算最终 BDD。它的唯一目标是导出完整 Stage 6-compatible sharded dataset，让 Stage 7E/F 后续复用 Stage 6 的 BDD、report-card、task-conditioned BDD 模块。注意：Stage 5 / Stage 6 Waymo 预处理可为数据量使用 multi-agent ego expansion；Stage 7 nuPlan planner 数据禁止这样做，因为 official planner 只控制 nuPlan ego，其他 road participants 只能作为 mandatory neighbor context。

输出必须包含：

- `shards/shard_000/ego_seq.npy`：Stage 6-compatible ego layout `[x, y, vx, vy, heading, speed, accel, yaw_rate]`；
- `shards/shard_000/neighbor_seq.npy`：mandatory surrounding-agent context，严格 layout 为 `[rows, K, T, 9]`，9 个 channel 依次是 `rel_x, rel_y, rel_vx, rel_vy, distance, bearing, heading_rel, speed, valid`；
- `shards/shard_000/neighbor_slot_ids.npy`：mandatory neighbor slot id；
- `shards/shard_000/interaction_feat_style.npy`：longitudinal comfort + interaction/style features；
- `shards/shard_000/metadata.csv`：one row = one scenario × one planner-controlled nuPlan ego rollout；
- `feature_schema.json`：feature names and indices；
- `shard_manifest.json`：Stage 6-compatible shard manifest；
- `planner_policy_indices/*.npy`：每个 planner 的 global row index。

`tools/stage7d_validate_official_planner_bdd.py` 仅是 smoke diagnostic，不是 canonical final BDD path。

## 3. 通过标准

命令通过时必须满足：

- `simulation_schema.json` 中 `pseudo_rollout == false`；
- `simulation_schema.json` 中 `uses_official_nuplan_simulation == true`；
- 输出总行数等于 `N * P`，当前 5 logs × 4 planners 应为 20，不能按 num_agents / num_neighbors 扩展为更多 ego rows；
- `ego_seq.npy`、`neighbor_seq.npy`、`neighbor_slot_ids.npy`、`interaction_feat_style.npy`、`metadata.csv` 行对齐；
- `stage7d_export_schema.json` 明确记录 Stage 7 row semantics、nuPlan planner-controlled ego-only 定义、background agents as context、`multi_agent_ego_expansion=false`、`total_rows_expected=num_scenarios*num_planners`；
- `warnings.json.validation` 明确记录 `total_rows == num_scenarios * num_planners`、`no_multi_agent_ego_expansion == true`、`neighbor_agents_used_as_context_only == true`；
- `neighbor_seq.npy` 和 `neighbor_slot_ids.npy` 缺失必须 fail，不能作为 non-fatal warning；当前 Stage 7D exporter 明确要求上游先从 official msgpack observations 或 nuPlan scenario DB 提取 `stage7d_neighbor_seq.npy` / `stage7d_neighbor_slot_ids.npy`，并且 neighbor 必须相对每一条 planner-controlled ego rollout 重新计算；
- `feature_schema.json` / `stage7d_export_schema.json` 必须记录 `neighbor_layout=ego_centric_relative` 和上述 neighbor channels；`interaction_feat_style.npy` 对缺失 neighbor-derived TTC/THW/distance 使用 NaN，不写入 `inf`；
- `metadata.csv` 必须保留 `simulated_planner_metadata.csv` 中的 planner profile 字段（例如 IDM 的 `style_scope=longitudinal_only`、`policy_style=longitudinal_conservative/comfort/aggressive`、`nuplan_planner_config=idm_planner`），不能覆盖为 generic planner；
- `metadata.csv` 必须从 `scenario_planner_index.csv` 映射 `db_name -> log_name`（去掉 `.db`）、`scenario_id -> actual_nuplan_scenario_token`、`scene_token -> stage7b_scene_token`、`sample_id`、`scenario_type`；
- 四个 planner index 文件均存在：`simple_planner.npy`、`idm_longitudinal_conservative.npy`、`idm_longitudinal_comfort.npy`、`idm_longitudinal_aggressive.npy`；
- Stage 7D 不重新实现最终 BDD，后续 Stage 7E/F 使用 Stage 6 BDD/report-card/task-conditioned BDD。



## Stage 7E final lane-aware context build and embedding rerun（Stage5D CORE ego local-frame）

## 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --overwrite

python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --overwrite
```

如果需要强制检查某些 planner 必须存在，可以给 context build 增加可选参数，例如：

```bash
  --required_planners simple_planner idm_longitudinal_conservative idm_longitudinal_comfort idm_longitudinal_aggressive
```

默认不要求固定 IDM planner 名称，以支持后续 PDM / ML planner 扩展。

## 2. 期望行为

- `build_nuplan_5neighbor_context_dataset.py` 发现 `simulated_planner_metadata.csv` / `scenario_planner_index.csv` 中的 planner axis，不再依赖 Stage 7D 的 IDM-only `REQUIRED_PLANNERS`。
- nuPlan ego 8D 通过 `tools.stage5d_context_core.build_ego_features_8d(...)` 生成：先把 `simulated_ego_seq` 行适配为 `[x, y, vx, vy, heading, valid]`，再用第一帧有效 ego 位置和 heading 构造 deterministic local window frame。
- neighbor `rel_x/rel_y/rel_vx/rel_vy` 仍按每个 timestep 的 ego 当前 pose/heading 计算，这与原 Waymo Stage 5D builder 的 neighbor convention 一致。
- Stage 7E embedding 只读取 `context_traj.npy [N,T,83]`；不得重新引入 `--dataset_dir`、`context_layout` 或 Stage 7D top-K neighbor bridge。

## 3. 通过标准

- `warnings.json` 记录 `ego_local_frame_source == "tools.stage5d_context_core.build_ego_features_8d"`。
- `warnings.json` 记录 neighbor local-frame contract，说明 neighbor 相对量是 per-timestep ego-centric。
- `planner_policy_indices/*.npy` 对所有 observed planners 非空；只有传入 `--required_planners` 时才强制检查指定 planner 名称。
- embedding 输出的 `warnings.json.validation.context_layout_used == "stage5d_context_dataset_direct"`，且 `context_padded_to_checkpoint_dim == false`。

## Stage 7E/7F-IDM final thesis path：先构建 common-core context，再导出 embedding

## 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --overwrite

python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --overwrite
```


## 2. 期望行为

- 最终论文路径固定为 `build_nuplan_5neighbor_context_dataset.py -> context_traj.npy -> stage7e_embed_stage6_dataset.py --context_dataset_dir`。
- `build_nuplan_5neighbor_context_dataset.py` 从 Stage 7C official nuPlan simulation 与 official msgpack tracked objects 构造 Stage 5D-compatible `context_traj.npy [N,T,83]`，其中 `83 = ego 8 + 5 semantic neighbor slots × 15 channels`。
- 行语义固定为 `row = scenario × planner × planner-controlled nuPlan ego rollout`；background agents 只作为 context，不扩展为 ego rows。
- Stage 7E embedding 直接读取 `context_traj.npy`，检查 checkpoint 的 `context_dim == 83`，导出 `embedding.npy` / `embeddings/shard_000000/embeddings.npy`，并复制 `metadata.csv` 与 `planner_policy_indices/*.npy`。
- 旧的 Stage 7D top-K `neighbor_seq` reconstruction 路径已经从最终 Stage 7E 脚本移除，不能作为 thesis evidence；最终脚本只接受 `--context_dataset_dir`。

## 3. 通过标准

- context build 输出 `ego_seq.npy`、`context_traj.npy`、`interaction_feat_style.npy`、`metadata.csv`、`feature_schema.json`、`stage5d_context_schema.json`、`shard_manifest.json`、`planner_policy_indices/*.npy`、`warnings.json`、`context_build_report.md`、`slot_assignment_report.md`。
- `warnings.json.validation.stage5d_dim_matched == true`、`stage5d_slot_schema_matched == true`、`stage5d_slot_order_matched == true`、`context_traj_no_nonfinite == true`。
- nuPlan semantic slots 在多 scenario/planner rollout 中可能发生 tracked-object ID switch；此时 `accel/yaw_rate` finite difference parity 可报告为 `nonfatal_slot_switch_reset`，并在 `warnings.json` 中写入 `temporal_formula_nonfatal_slot_switch_reset`。只要 structural checks 以及 static / closing / TTC / delta_x / delta_y 公式通过，这属于预期诊断，不会使 `context_traj.npy [N,T,83]` 或 Stage 7E embedding 输入失效。
- lane-aware runtime diagnostics 必须包含 `lane_assignment_available`、`map_query_success`、`lane_info_count`、`fallback_assignment_used_rate`、`ego_lane_projection_success_rate`、`candidate_lane_projection_success_rate`。
- `assignment_mode == lane_aware_only` 时，如果 map query 失败、`lane_info_count == 0` 或 ego lane projection 不可用，脚本必须 fail loudly；不能 silent fallback。
- `assignment_mode == lane_aware_with_geometric_fallback` 时，如果 `fallback_assignment_used_rate` 很高，`warnings.json` 必须有 high fallback warning，并需要检查 `--nuplan_map_root`、`map_name` 解析和 projection 诊断。
- embedding 输出的 `warnings.json.validation.context_layout_used == "stage5d_context_dataset_direct"`、`checkpoint_context_dim_matches_final_context_dim == true`、`context_padded_to_checkpoint_dim == false`、`stage5d_schema_matched == true`。
- Stage 7F / Stage 6 BDD-report-card 输入为 Stage 7E embedding 与对齐的 metadata/features/indices，不直接把 Stage 7D raw `ego_seq.npy` / `neighbor_seq.npy` 当最终 embedding 表示。

## Stage 7E：Stage 5D 83维 context embedding 合同（旧 reconstruction 路径废弃）

### 1. 命令

最终命令必须使用：

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs \
  --overwrite
```

最终 Stage 7E 脚本已经移除旧 debug bridge；不再提供 `--dataset_dir` / `--context_layout` 参数。

### 2. 期望行为

Stage 7D 只负责导出 Stage 6-compatible evaluation dataset。最终 Stage 7E 不再从 Stage 7D 的 top-K neighbor tensor 推断 `front/left_front/left_rear/right_front/right_rear` 语义 slot，也不再用旧 proxy 公式把 top-K 邻车重标为 Stage 5D 83 维 context。

Stage 5D-compatible `context_traj.npy [N,T,83]` 必须由 `tools/build_nuplan_5neighbor_context_dataset.py` 构建，并通过 `--context_dataset_dir` 传给 embedding 脚本。

### 3. 通过标准

- 最终 Stage 7E parser 要求 `--context_dataset_dir`，不再提供 `--dataset_dir` / `--context_layout` debug bridge。
- `embedding_manifest.json` 记录 `does_not_rebuild_context_from_stage7d_neighbor_seq == true`。
- `warnings.json` 中 `context_layout_used == "stage5d_context_dataset_direct"` 且 `context_padded_to_checkpoint_dim == false`。

## Stage 7E：nuPlan Stage 5D-compatible 5-neighbor context dataset（推荐架构）

## 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite
```

## 2. 期望行为

- 输出 `context_traj.npy [N,T,83]`，直接作为 Stage 7E embedding 的 encoder 输入。
- `context_traj.npy` 不包含 map/lane/ODD channels；lane-aware 逻辑只影响 5 个 semantic neighbor slots 的选择。
- `warnings.json.validation` 显式记录 lane-aware runtime diagnostics：`lane_assignment_available`、`map_query_success`、`lane_info_count`、`fallback_assignment_used_rate`、`ego_lane_projection_success_rate`、`candidate_lane_projection_success_rate`。
- `lane_aware_only` 是严格验证模式：地图查询或投影失败必须报错。
- `lane_aware_with_geometric_fallback` 是推荐构建模式：允许 fallback，但 fallback rate 高时必须写 warning，不能把高 fallback 结果包装成纯 lane-aware thesis evidence。

## 3. 通过标准

- `context_traj.npy` shape 为 `[num_scenarios × num_planners, T, 83]`，且不包含 NaN/Inf。
- `stage5d_context_schema.json` 中 `neighbor_slots` 必须精确为 `front, left_front, left_rear, right_front, right_rear`。
- `warnings.json.validation.map_query_success == true` 且 `lane_info_count > 0` 才能说明实际使用了 nuPlan lane-aware map query。
- `slot_assignment_report.md` 必须报告 assignment mode、lane-aware success rate、geometric fallback rate、map query success、lane info count、ego/candidate projection success rate、各 slot coverage/empty ratio、lane context quality 和 rejection reason counts。

## Stage 7E：nuPlan Stage5D derived-channel parity 修正

## 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c_official_sim \
  --output_dir outputs/stage7e_nuplan_stage5d_context \
  --overwrite
```

## 2. 期望行为

该命令读取 Stage 7C official nuPlan simulation 输出与 official nuPlan msgpack tracked objects，构造 Stage 5D checkpoint-compatible 的 `context_traj.npy [N,T,83]`。邻车 15 维通道顺序保持为：`valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y, closing, ttc, thw, speed, accel, heading_rel, yaw_rate`。

本版本明确区分 `direct_from_state`、`derived_same_as_stage5` 与 `approximated`：`delta_x/delta_y` 是 `rel_x/rel_y` 的 Stage 5 同公式派生通道，不再标记为 proxy；`closing` 使用 Stage 5 公式 `ego_forward_speed - rel_vx`；`ttc/thw` 使用 Stage 5 cap 公式。脚本会输出 `slot_assignment_report.md`，报告每个 slot 的 ID switch count、switch rate 与平均连续片段长度；如果 slot ID 发生切换，`accel/yaw_rate` 不会跨不同 agent 做有限差分，并会在 schema / warnings 中标记为未完全 Stage5D-equivalent。

该命令不会改变 row 语义：row 仍然是 `scenario × planner × planner-controlled nuPlan ego rollout`；background agents 只作为 context，不做 multi-agent ego expansion；也不会修改 Stage 6 逻辑。

## 3. 通过标准

- `context_traj.npy` shape 为 `[num_scenarios × num_planners, T, 83]`，且不包含 NaN/Inf。
- `warnings.json.validation.stage5d_closing_formula_matched`、`stage5d_ttc_formula_matched`、`stage5d_delta_xy_formula_matched` 为 `true`。
- `warnings.json.validation.slot_id_switch_rate_by_slot` 存在，并且 `slot_assignment_report.md` 包含每个 slot 的 `slot_id_switch_count`、`slot_id_switch_rate`、`mean_continuous_segment_length`。
- `stage5d_context_schema.json` 对每个通道包含 `source_kind`、`formula`、`matched_waymo_stage5_formula`；`delta_x/delta_y` 为 `derived_same_as_stage5`，不是 proxy。
- 若 slot switch rate 非零，`accel/yaw_rate` 在 schema 中标记为 `approximated` / `approximated_or_not_stage5_matched`，且报告不声称完全等价。

## Stage 7E nuPlan lane-aware Stage 5D context

### 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c_nuplan_idm_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite
```

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs/embeddings \
  --overwrite
```

### 2. 期望行为

- Stage 5 原始 5-neighbor slot schema 是 `front, left_front, left_rear, right_front, right_rear`，不是带 `rear` 的旧 nuPlan proxy 顺序。
- nuPlan Stage 7E context builder 复用 Stage 5 的 `tools.lane_aware_assignment.assign_neighbors_lane_aware`；nuPlan 只改变数据来源和 row 语义，不改变 Stage 5D 输入契约。
- 输出 `context_traj.npy [N,T,83]`，其中 `83 = ego 8 + 5 semantic neighbor slots × 15 channels`。
- row 语义保持 `scenario × planner × planner-controlled nuPlan ego rollout`；background agents 只作为 context，不展开成 ego rows。
- 默认 `--assignment_mode lane_aware_with_geometric_fallback`：优先 lane-aware assignment，地图或投影不可用时才走 Stage 5 geometric fallback。
- Stage 7E embedding 直接读取 `context_traj.npy`，检查 checkpoint 的 `context_dim == 83`，不会从 Stage 7D distance-topK `neighbor_seq` 重建 context。

### 3. 通过标准

- `stage5d_context_schema.json` 中 `neighbor_slots` 必须精确为 `front, left_front, left_rear, right_front, right_rear`。
- `warnings.json.validation.stage5d_slot_schema_matched == true` 且 `stage5d_slot_order_matched == true`。
- `warnings.json.validation.row_semantics_correct == true`、`no_multi_agent_ego_expansion == true`、`background_agents_context_only == true`。
- `warnings.json.validation.context_traj_no_nonfinite == true`，且 `context_traj.npy` 最后一维为 83。
- `slot_assignment_report.md` 必须报告 assignment mode、lane-aware success rate、geometric fallback rate、各 slot coverage/empty ratio、lane context quality 和 rejection reason counts。

## Stage 5D / Stage 7E：共享 83 维 context core

### 1. 命令

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir data/waymo \
  --out_dir outputs/waymo_5neighbor_context_laneaware_smoke \
  --max_files 1 \
  --max_scenarios 2 \
  --overwrite
```

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --overwrite
```

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --overwrite
```

```bash
python -m pytest tests/test_stage5d_context_core.py -q
```

### 2. 期望行为

Waymo builder 和 nuPlan builder 都复用 `tools/stage5d_context_core.py` 中的 Stage 5D context 定义：`SLOT_NAMES`、ego 8 维通道、neighbor 15 维通道、`context_dim=83`、lane-aware slot assignment 入口、derived channel 公式、`context_traj` 拼接、schema 生成与 validation。nuPlan builder 只负责把 official nuPlan simulation / msgpack tracked objects 适配为标准 ego、candidate、lane 输入；不会把 background agents 扩展成新的 ego rows。

### 3. 通过标准

- `context_traj.npy` shape 为 `[N,T,83]`。
- 83 维顺序固定为 ego 8 维 + 5 个 semantic neighbor slots × 15 维。
- slot 顺序固定为 `front, left_front, left_rear, right_front, right_rear`。
- `warnings.json` 包含 `validation.pass`、`map_query_success`、`lane_info_count`、`lane_assignment_available`、`fallback_assignment_used_rate`、`ego_lane_projection_success_rate`、`candidate_lane_projection_success_rate`、`stage5d_core_reused=true`、`stage5d_slot_schema_matched=true`、`stage5d_slot_order_matched=true`、`stage5d_derived_formula_matched`、`stage5d_accel_yaw_rate_formula_matched`、`slot_id_switch_rate_by_slot`；如果 slot ID switch rate 非 0，则 temporal accel/yaw_rate parity 不能被标记为 true。
- `build_nuplan_5neighbor_context_dataset.py` 不再定义自己的 `SLOT_NAMES` 或 neighbor channel order。

## Stage 7E lane-aware map_name/location 传递修复

### 1. 命令

Stage 7C official simulation 输出会在 `scenario_planner_index.csv` 中保留 `map_name`、`location`、`log_name`、`scenario_token`、`scenario_type`：

```bash
python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /path/to/nuplan/dbs \
  --nuplan_map_root /path/to/nuplan/maps \
  --output_dir outputs/stage7c1_nuplan_simulation \
  --nuplan_simulation_command_template '<official nuPlan command>' \
  --overwrite
```

Stage 7E context builder 的地图名解析顺序是：行级 `map_name`、行级 `location`、显式 `--map_name`、可选 `--scenario_map_metadata_csv` 映射文件。

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c1_nuplan_simulation \
  --output_dir outputs/stage7e_context_stage5d \
  --nuplan_map_root /path/to/nuplan/maps \
  --assignment_mode lane_aware_with_geometric_fallback \
  --scenario_map_metadata_csv outputs/stage7c1_nuplan_simulation/scenario_planner_index.csv \
  --overwrite
```

如果 Stage 7C 历史输出缺少 `map_name/location`，可用显式 override 做 smoke 验证：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c1_nuplan_simulation \
  --output_dir outputs/stage7e_context_stage5d \
  --nuplan_map_root /path/to/nuplan/maps \
  --map_name us-nv-las-vegas-strip \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite
```

严格 thesis evidence 验证应使用：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c1_nuplan_simulation \
  --output_dir outputs/stage7e_context_stage5d_lane_only \
  --nuplan_map_root /path/to/nuplan/maps \
  --assignment_mode lane_aware_only \
  --overwrite
```

### 2. 期望行为

Stage 7C 从 `merged_metadata.csv` 读取场景行时，会把可用的 `map_name`、`location`、`log_name`、`scenario_token`、`scenario_type` 写入 `scenario_planner_index.csv`，供 Stage 7D/Stage 7E 继续使用。Stage 7E 不会重建 Stage 7D neighbor context，也不会修改 Stage 5D CORE 或 Stage 7E embedding；它只在构建 Stage 5D-compatible `context_traj.npy [N,T,83]` 时解析地图名并尝试 nuPlan lane query。

`assignment_mode=lane_aware_only` 下，如果任一行无法解析 `map_name` 或地图查询/投影不可用，构建会失败并在 `warnings.json` 中写入 error。`assignment_mode=lane_aware_with_geometric_fallback` 下，缺少 `map_name` 会明确 warning，允许几何 fallback，但报告不会把该结果描述为 lane-aware thesis evidence。

### 3. 通过标准

- `scenario_planner_index.csv` 包含 `map_name`、`location`、`log_name`、`scenario_token`、`scenario_type` 列；
- `warnings.json.validation.map_name_resolved_rate` 大于 0，理想为 1.0；
- `warnings.json.validation.map_names_used` 列出实际使用的地图名；
- `warnings.json.validation.map_query_success=true` 且 `lane_info_count > 0`；
- `warnings.json.validation.lane_assignment_available=true`；
- `slot_assignment_report.md` 和 `context_build_report.md` 同时报告 map_name 解析、map query、lane info、lane-aware success rate、geometric fallback rate；
- 严格 `lane_aware_only` 模式不允许因为缺少 `map_name` 而 silent fallback。

---

## Stage5D / Stage7E lane-aware assignment 对比诊断

## 1. 命令

```bash
python tools/export_waymo_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --output_dir outputs/waymo_laneaware_diagnostics_strict_v1 \
  --max_rows 5000 \
  --filtering_mode strict_filter_lane_aware_only \
  --diagnostic_source_note user_confirmed_stage5_command_used_lane_aware_only_plus_drop_if_filters \
  --overwrite

python tools/compare_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_laneaware_diagnostics_strict_v1 \
  --nuplan_dir outputs/<stage7e_nuplan_context_output> \
  --out_dir outputs/lane_aware_diagnostic_comparison \
  --max_rows 2000
```

## 2. 期望行为

该命令不会实现新的 Stage7 lane-aware 算法，也不会改写任何已有数据集。它只读取 Waymo Stage5/Stage5D 输出目录中的 `build_summary.json`、`neighbor_context_summary.json`、shard 里的 `neighbor_seq.npy` / `neighbor_slot_ids.npy`，以及 nuPlan Stage7E 输出目录中的 `warnings.json`、`assignment_debug.json`、`neighbor_seq.npy` / `neighbor_slot_ids.npy`，然后用同一组 Stage5D lane-aware assignment 诊断口径生成可比报告。

输出文件：

```text
outputs/lane_aware_diagnostic_comparison/lane_aware_diagnostic_comparison.json
outputs/lane_aware_diagnostic_comparison/lane_aware_diagnostic_comparison.md
```

报告会对比：

- `lane_assignment_available`
- `fallback_assignment_used_rate`
- `candidate_projection_success_rate`
- `adjacency_source_counts`
- `lane_context_quality` counts
- rejection reason counts
- slot coverage by slot
- slot switch rate by slot

诊断规则是：如果 nuPlan 在复用 `tools.lane_aware_assignment.assign_neighbors_lane_aware` 的前提下，比 Waymo 有明显更高 fallback rate 或明显更低 candidate projection success rate，则优先判定为 nuPlan LaneInfo adapter / map topology / adjacency / projection quality issue；否则判定为 generic Stage5 lane-aware limitation 或证据不足。

## 3. 通过标准

- 命令正常结束并生成 `.json` 与 `.md` 两个报告文件。
- 报告明确写出 Stage5D CORE / `tools.lane_aware_assignment.py` 是唯一 lane-aware assignment 实现，Stage7 nuPlan 只是 adapter。
- 报告包含上述 8 类可比指标。
- 如果结论指向 generic limitation，后续只能改 `tools/lane_aware_assignment.py` 或 Stage5D CORE，使 Waymo 与 nuPlan 同时受益。
- 如果结论指向 nuPlan-specific issue，后续只能改 `tools/nuplan_lane_utils.py` 或 nuPlan adapter，不允许在 Stage7 复制 lane-aware assignment 逻辑。

## Stage7E 车道感知诊断（nuPlan / Waymo 对比）

## 1. 命令

nuPlan 构建并输出投影调试：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_debug \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --debug_projection_sample_rows 20 \
  --overwrite
```

Waymo 侧导出同粒度诊断：

```bash
python tools/export_waymo_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --output_dir outputs/waymo_laneaware_diagnostics_v2 \
  --max_rows 5000 \
  --overwrite
```

跨数据集诊断对比：

```bash
python tools/compare_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_laneaware_diagnostics_v2 \
  --nuplan_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_debug \
  --out_dir outputs/stage7e_laneaware_diagnostic_compare_v3 \
  --max_rows 5000
```

## 2. 期望行为

- Stage7 不实现新的 lane-aware assignment 算法；Stage7 只把 nuPlan map / tracked objects 转成 Stage5D CORE 可用的 `LaneInfo` 和候选状态。
- nuPlan 输出 `nuplan_lane_projection_debug_summary.json`、`nuplan_lane_projection_debug_report.md`；加 `--write_projection_debug` 时额外输出有界采样的 `nuplan_lane_projection_debug.csv`。
- Waymo 诊断脚本只读取已有 Stage5D 输出和数组，不重新分配 slot，不修改 `tools/lane_aware_assignment.py`。
- 对比脚本会显式报告 Waymo / nuPlan 的 candidate projection 指标是否存在、fallback rate 是否可比、slot coverage 是 array-derived 还是 summary-derived。

## 3. 通过标准

- `warnings.json` / `context_build_report.md` / `slot_assignment_report.md` 能指向 nuPlan projection debug artifact。
- 如果 Waymo 缺少 fallback 和 candidate projection 可比指标，verdict 必须是 `inconclusive_missing_comparable_metrics`。
- 如果 Waymo 指标存在且 nuPlan projection success 明显更低或 fallback 明显更高，verdict 可为 `nuplan_adapter_or_map_projection_issue`。
- 如果两侧都低 projection / 高 fallback，verdict 可为 `generic_stage5_lane_aware_limitation_or_dataset_common_issue`。
- 如果 nuPlan 不明显更差且指标可比，verdict 可为 `no_clear_nuplan_adapter_issue`。

## Stage7E nuPlan LaneInfo topology 调试

## 1. 命令

在构建 nuPlan 5-neighbor Stage5D context 时开启投影调试：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_topology_debug \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --debug_projection_sample_rows 20 \
  --overwrite
```

## 2. 期望行为

- 脚本仍然复用 Stage5D 的 `tools.lane_aware_assignment.assign_neighbors_lane_aware`，不会新增 Stage7 专用 assignment 算法。
- `tools/nuplan_lane_utils.py` 会把 nuPlan lane / lane_connector map object 转换为 Stage5D `LaneInfo`，优先读取 nuPlan map object 直接提供的 left/right adjacency 与 incoming/outgoing topology。
- 如果 map object 没有直接提供 left/right adjacency，adapter 只在 `LaneInfo` 层做几何邻接补全，使补全结果继续进入既有 Stage5D assignment；这不是新的 slot assignment 逻辑。
- lane_connector 不会被强行补全左右邻接；如果可用，只记录 predecessor/successor 到 `entry_lane_ids` / `exit_lane_ids`，并在 topology 报告中说明 Stage5D 当前只把这些字段用于诊断。
- 输出目录会生成：
  - `nuplan_lane_topology_debug_summary.json`
  - `nuplan_lane_topology_debug_report.md`
  - `nuplan_lane_projection_debug_summary.json`
  - `nuplan_lane_projection_debug_report.md`
  - 有 unknown/wrong_lane 样本时生成 `nuplan_lane_relation_unknown_debug.csv`
  - 加 `--write_projection_debug` 时生成 `nuplan_lane_projection_debug.csv`

## 3. 通过标准

- topology summary 中 `lane_info_count` 大于 0，且分别报告 `lane_count`、`lane_connector_count`、左右邻接非空比例、predecessor/successor 非空比例、centerline 点数分布、lane length 分布。
- `lane_relation_unknown_breakdown` 至少按 `missing_adjacency`、`topology_disconnected`、`direction_mismatch`、`candidate_projection_failed`、`ego_projection_failed`、`lane_connector_unhandled`、`other` 这些类别解释 unknown relation。
- `nuplan_lane_relation_unknown_debug.csv` 中包含 ego lane / candidate lane 类型、是否在左右邻接、是否共享 predecessor/successor、lateral offset、heading diff、s difference 等字段，便于定位是 topology 缺失还是方向/连接关系问题。
- `tools/lane_aware_assignment.py` 不应出现本次改动；Stage5D core formulas、Stage7E embedding、Stage6 不应被修改。

## Stage7E nuPlan Stage5 风格严格 lane-aware 过滤诊断

## 1. 命令

默认 Stage7E 仍保留官方 rollout 行语义；只额外写严格过滤诊断：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/<stage7c_sim_output> \
  --output_dir outputs/<stage7e_nuplan_context_output> \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --overwrite
```

如确实需要把严格过滤后的数组另存为诊断数据集，再显式增加：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/<stage7c_sim_output> \
  --output_dir outputs/<stage7e_nuplan_context_output> \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --overwrite
```

严格过滤公平对比：

```bash
python tools/compare_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --nuplan_dir outputs/<stage7e_nuplan_context_output> \
  --out_dir outputs/laneaware_strict_filter_compare \
  --max_rows 5000
```

## 2. 期望行为

- Waymo Stage5 clean lane-aware 数据集历史上使用 `--assignment_mode lane_aware_only`，并配合 `--drop_if_no_lane_map`、`--drop_if_ego_lane_missing`、`--drop_if_lane_context_bad`、`--drop_if_lane_context_ambiguous` 过滤。
- Stage7 nuPlan 主数据集必须保留官方 `scenario × planner rollout` 行语义，因此默认使用 `lane_aware_with_geometric_fallback`，不会静默删除 planner rollout 行。
- `--write_strict_filter_diagnostic` 只写诊断文件；`--strict_filter_min_laneaware_ratio 1.0` 保持旧行为，`0.8` 用于模拟 Waymo `--min_valid_ratio 0.8` 的严格过滤思想：
  - `nuplan_laneaware_strict_filter_summary.json`
  - `nuplan_laneaware_strict_filter_report.md`
- 只有显式传入 `--write_strict_filtered_dataset` 时，才会额外写 `strict_filtered_dataset/` 下的过滤后数组和 metadata。
- Waymo 诊断导出会把显式传入的 `filtering_mode` 和 `diagnostic_source_note` 写入 json/md；不要在未确认过滤来源时把 Waymo fallback=0 当成 strict-filter 高置信证据。
- 对比脚本会识别 Waymo strict-filtered 与 nuPlan fallback-preserving 的过滤口径不一致，并把 verdict 降级为 `inconclusive_due_to_filtering_mismatch`；只有 Waymo strict-filtered 与 nuPlan strict-filter diagnostic 对比时，才使用公平严格过滤 verdict。

## 3. 通过标准

- 默认 Stage7E 输出仍满足一行等于一个 `scenario × planner-controlled rollout`。
- 严格过滤诊断报告包含 strict_filter_min_laneaware_ratio、original rows、rows kept、rows dropped、kept_row_rate、dropped_by_reason、kept rows per planner、scenario-planner alignment、每个 scenario 是否仍保留所有 planners、frame/row lane-aware availability、availability quantiles、kept rows slot sanity 与 slot coverage；如传入 ratio sweep，还会输出多阈值表。
- Waymo `fallback=0` 与 nuPlan `fallback=41.9%` 不再被直接解释为高置信 nuPlan adapter 问题，除非两侧过滤口径一致。
- Stage5D CORE / `tools.lane_aware_assignment.py` 仍是唯一 lane-aware assignment 实现；Stage7 不新增专用 assignment 算法。

## Stage7F full fallback-preserving 主报告与 strict-filter 敏感性

## 1. 命令

A. 先生成或确认 Stage7E fallback-preserving full-row embedding（主路径）：

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --overwrite
```

B. 运行 Stage7F full fallback-preserving 主报告：

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --output_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --mode full \
  --run_stage6_pairwise \
  --overwrite
```

C. 只有当 strict-filter ratio=0.8 已经通过 `--write_strict_filtered_dataset` 明确写出 embedding 输入时，才运行 strict-filter 敏感性报告：

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_strictdiag_ratio08/embeddings \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_strictdiag_ratio08/strict_filtered_dataset \
  --output_dir outputs/stage7f_idm_5logs_strict_ratio08_sensitivity \
  --mode strict_sensitivity \
  --strict_filter_min_laneaware_ratio 0.8 \
  --run_stage6_pairwise \
  --overwrite
```

如果 ratio=0.8 目前只有 `nuplan_laneaware_strict_filter_summary.json` / `.md` 诊断，而没有真实过滤后的 context arrays 和 embedding，则不要伪造 embedding；应先显式写出 strict-filter 数据集并再 embedding：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/<stage7c_sim_output> \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_strictdiag_ratio08 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --strict_filter_min_laneaware_ratio 0.8 \
  --overwrite

python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_strictdiag_ratio08/strict_filtered_dataset \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2_strictdiag_ratio08/embeddings \
  --overwrite
```

推荐执行顺序固定为：A. Stage7F full fallback-preserving 主报告；B. strict-filter ratio=0.8 clean-subset sensitivity；C. 之后才做 Stage5 parameter / lane-aware threshold sweep。

## 2. 期望行为

- `tools/stage7f_run_report_card.py` 是薄封装：验证 Stage7E embedding 输出、metadata、planner axis、scenario × planner 对齐关系，并在可用时调用既有 Stage6 `tools/stage6_compare_unpaired_style.py` 与 `tools/stage6_generate_report_card.py`。如果显式传入 `--context_diagnostics_json`，优先读取该 JSON；否则会从 resolved `context_dataset_dir` 依次自动查找 `warnings.json`、`assignment_debug.json`、`nuplan_laneaware_strict_filter_summary.json`，把 fallback / lane-aware / strict-filter 诊断字段写入 `stage7f_summary.json` 和 `stage7f_report.md`。
- full 模式要求所有 scenario 都有完整 planner 组合；不完整时直接报错，避免把 clean-subset 误当主评估数据集。
- strict_sensitivity 模式允许 scenario 被过滤掉或 planner 组合不完整，但报告会明确写出它不是主 planner-evaluation dataset。
- 输出目录包含 `stage7f_summary.json`、`stage7f_report.md`、`planner_indices/*.npy`；summary/report 中会记录 `context_diagnostics_source`，并在可用时展示 `fallback_rate` / `fallback_assignment_used_rate`、`lane_assignment_available_rate`、`map_name_resolved_rate`、`map_query_success`、`lane_info_count`、`rows_kept`、`kept_row_rate`、slot sanity / coverage 等诊断；启用 `--run_stage6_pairwise` 且 context feature 输入存在时，还会生成 `stage6_pairwise/<planner_a>_vs_<planner_b>/` 下的 Stage6 report-card / BDD 输出。
- Stage7F 不修改 Stage6 metric definitions，不新增 Stage7 专用 BDD metric，不修改 Stage5D CORE / `tools/lane_aware_assignment.py`。

## 3. 通过标准

- full 主报告中 `all_scenarios_have_all_planners=true`，`row_semantics` 为 `scenario × planner-controlled nuPlan ego rollout`，且 embedding rows 与 metadata rows 一致。
- full 主路径保持 fallback-preserving，Stage7E main path 仍是 primary planner-evaluation dataset；即使命令没有传 `--context_diagnostics_json`，只要 `context_dataset_dir/warnings.json` 存在，报告中的 fallback rate 不应显示为 unavailable。
- strict-filter 敏感性报告写出 `strict_filter_min_laneaware_ratio=0.8`、`rows_kept`、`kept_row_rate`（如诊断 JSON 提供）、`scenarios_with_all_planners`、`scenarios_missing_any_planner`、fallback rate 与 slot sanity（如诊断 JSON 提供），并包含“不是主评估数据集”的 warning。
- 若 strict-filter ratio=0.8 没有真实 embedding 输入，只能保留为 diagnostic-only，不能虚构 `embedding.npy`。

## Stage 7F — pairwise aggregation collector（Stage6 输出汇总）

### 1. 命令

推荐先运行 full fallback-preserving Stage7F 主报告，并让 Stage7F runner 自动复用 Stage6 pairwise 工具、随后自动生成 pairwise 汇总：

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2 \
  --output_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --mode full \
  --run_stage6_pairwise \
  --overwrite
```

如果 Stage6 pairwise 目录已经存在，也可以只运行轻量汇总器：

```bash
python tools/stage7f_collect_pairwise_summary.py \
  --stage7f_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --overwrite
```

### 2. 期望行为

- runner 会读取 Stage7E `embedding.npy` / `metadata.csv` / `embedding_manifest.json`，保持行语义为 `scenario × planner-controlled nuPlan ego rollout`。
- 加 `--run_stage6_pairwise` 时，runner 继续调用既有 Stage6 pairwise compare/report-card 工具；Stage6 完成后自动写出：
  - `outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_pairwise_summary.csv`
  - `outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_pairwise_summary.json`
  - `outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_pairwise_summary.md`
- 单独运行 collector 时，只扫描 `stage7f_dir/stage6_pairwise/*/` 下已有的 `bdd_summary.json`、`style_report_card.md`、`stage6_warnings.json` 以及可选 CSV；不会重新计算 BDD/MMD，不会修改 Stage6 metric 定义，不会修改 Stage5D CORE，也不会读取或修改 lane-aware assignment 逻辑。
- 可选的 `category_delta.csv`、`feature_delta.csv`、`top_drift_cases.csv`、`scenario_slice_delta.csv` 缺失时，汇总器不会报错；对应字段会写为 null / unavailable。
- 运行后重点查看：

```text
outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_report.md
outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_pairwise_summary.md
```

### 3. 通过标准

- `stage7f_pairwise_summary.csv/json/md` 均存在。
- Markdown 报告包含 source Stage7F directory、full fallback-preserving mode、row semantics、scenario/planner/row 数、fallback/map/lane diagnostics、按 `bdd_mmd2` 降序排序的 pairwise 表、top/lowest BDD pair、warnings summary 和 limitations。
- `bdd_rank_desc=1` 对应最大的 `bdd_mmd2`。
- 当前 5-log 结果必须解释为 exploratory：每个 planner pair 的 `n_A=n_B=5` 太小，permutation p-value 低功效，BDD 只表示分布漂移幅度而不表示方向；category/feature delta 只是解释层。
- full 主结果 fallback rate 约 41.9% 时，结论必须后续配合 strict-filter ratio=0.8 sensitivity 和 Stage5 lane-aware parameter sweep。

## Stage7 official nuPlan A/B pilot: aggressive vs conservative

本节是 **main-path** 的 20-scenario 快速真实实验流程，只比较两个 longitudinal IDM planner：`idm_longitudinal_aggressive` vs `idm_longitudinal_conservative`。它不运行 `simple_planner`，也不运行 `idm_longitudinal_comfort`。实验 id 固定为 `stage7_official_idm_ab_v1_20scenes`。

### 1. 命令

#### A. Stage7C official simulation（main-path，带进度 JSON）

输入路径：

- Stage7B.4 merged context: `/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged`
- nuPlan mini DB: `/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini`
- nuPlan maps: `/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps`

输出路径：

- simulation output: `/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7_official_idm_ab_v1_20scenes`
- progress artifact: `/home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7_official_idm_ab_v1_20scenes/stage7c_progress.json`

```bash
python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7_official_idm_ab_v1_20scenes \
  --planners idm_longitudinal_aggressive idm_longitudinal_conservative \
  --max_scenarios 20 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios scenario_filter.log_names=["{target_log_name}"] scenario_filter.scenario_tokens=null scenario_filter.limit_total_scenarios=1 worker=single_machine_thread_pool experiment_name=stage7_official_idm_ab_v1_20scenes job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

如需把进度文件写到自定义位置，可追加：

```bash
  --progress_json outputs/stage7_official_idm_ab_v1_20scenes/stage7c_progress.json
```

#### B. Stage7E context（main-path，lane-aware with geometric fallback）

输入路径：

- Stage7C simulation dir: `outputs/stage7_official_idm_ab_v1_20scenes`
- nuPlan map root: `$NUPLAN_MAPS_ROOT`

输出路径：

- `outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1`

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7_official_idm_ab_v1_20scenes \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --debug_projection_sample_rows 20 \
  --overwrite
```

#### C. Stage7E embedding（main-path，复用 Stage5D checkpoint / Stage6 dataset layout）

输入路径：

- context dataset: `outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1`
- checkpoint: `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt`

输出路径：

- `outputs/stage7e_idm_ab_embeddings_20scenes_laneaware_v1`

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_ab_embeddings_20scenes_laneaware_v1 \
  --overwrite
```

#### D. Stage7F full fallback-preserving A/B report（main-path，复用 Stage6 pairwise tools）

输入路径：

- embedding dir: `outputs/stage7e_idm_ab_embeddings_20scenes_laneaware_v1`
- context dataset: `outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1`

输出路径：

- `outputs/stage7f_idm_ab_20scenes_full_fallback_preserving_v1`

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_idm_ab_embeddings_20scenes_laneaware_v1 \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_ab_20scenes_laneaware_v1 \
  --output_dir outputs/stage7f_idm_ab_20scenes_full_fallback_preserving_v1 \
  --mode full \
  --run_stage6_pairwise \
  --overwrite
```

### 2. 期望行为

- Stage7C 只运行两个 planner，因此 `20 scenarios × 2 planners = 40` 个 scenario-planner tasks / rows；终端会在每个 task 前后打印 progress、elapsed、avg task time、ETA、成功/失败累计数，`stage7c_progress.json` 会在每个 task 完成后更新。
- Stage7C 输出保持 `pseudo_rollout=false` 与 `uses_official_nuplan_simulation=true`；一行仍然表示一个 `scenario × planner-controlled nuPlan ego rollout`，background agents 仅作为 context。
- Stage7E context 读取 Stage7C simulation output，生成 Stage5D-compatible 5-neighbor context；主路径保留 geometric fallback，不把 strict-filter clean subset 当作主结果。
- Stage7E embedding 直接读取 Stage7E context dataset，使用 `context_layout_used=stage5d_context_dataset_direct`，不会从 Stage7D neighbor sequence 重建 context。
- Stage7F full mode 验证每个 scenario 都有 aggressive 与 conservative 两个 planner；启用 `--run_stage6_pairwise` 时只生成一个 pairwise 输出：`idm_longitudinal_aggressive_vs_idm_longitudinal_conservative`，继续复用 Stage6 工具，不新增 Stage7 planner-behavior metric。

### 3. 通过标准

- Stage7C PASS 条件：`stage7c_progress.json` 中 `total_scenarios=20`、`total_planners=2`、`total_tasks=40`、`completed_tasks=40`；`simulation_schema.json` 中 `pseudo_rollout=false`、`uses_official_nuplan_simulation=true`；`scenario_planner_index.csv` 有 40 个 scenario-planner rows，且所有 scenario 都有两个 planner。
- Stage7E context PASS 条件：`context_traj.npy` rows = 40，context dim = 83，Stage5D schema matched，fallback rate 与 strict-filter diagnostics 均有报告。
- Stage7E embedding PASS 条件：`embedding.npy` shape = `[40, 64]`，manifest 中 `context_layout_used=stage5d_context_dataset_direct`，`does_not_rebuild_context_from_stage7d_neighbor_seq=true`。
- Stage7F PASS 条件：`stage7f_summary.json` 中 `num_scenarios=20`、`num_planners=2`、`total_rows=40`、`all_scenarios_have_all_planners=true`；`stage7f_pairwise_summary.json` 中 `num_pairs=1`，且唯一 pair 为 `idm_longitudinal_aggressive_vs_idm_longitudinal_conservative`。

### 4. 已知限制

- 该流程是 **20-scenario A/B pilot**，用于快速真实实验，不替代更大规模 statistical evaluation。
- Stage7C 的 official nuPlan command 依赖本机 nuPlan devkit、Hydra config、DB 和地图路径；如果这些路径不存在，不能声称完成真实数据验证。
- Stage7F 的 BDD / pairwise report 是 Stage6 metric reuse；它衡量 embedding distribution drift，不表示因果解释、驾驶安全结论或 planner 行为指标。
- strict-filter ratio sweep 是 **diagnostic / sensitivity**，主报告仍是 full fallback-preserving。

## Stage7F 20-scenario IDM aggressive/conservative 小 BDD 诊断

### 1. 命令

#### A. 在 Stage7E context 上构建 Stage6C behavior events

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7e_nuplan_5neighbor_context_idm_20scenes_laneaware_v1/shard_manifest.json \
  --feature_schema_path outputs/stage7e_nuplan_5neighbor_context_idm_20scenes_laneaware_v1/feature_schema.json \
  --output_dir outputs/stage7f_idm_20scenes_stage6c_behavior_events_v2 \
  --overwrite
```

#### B. 运行 aggressive vs conservative task-conditioned BDD

```bash
python tools/stage7f_run_task_conditioned_bdd.py \
  --embedding_dir outputs/stage7e_idm_embeddings_20scenes_laneaware_v1 \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_20scenes_laneaware_v1 \
  --stage7f_dir outputs/stage7f_idm_20scenes_full_fallback_preserving_v1 \
  --planner_a idm_longitudinal_aggressive \
  --planner_b idm_longitudinal_conservative \
  --output_dir outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_bdd_v1 \
  --task_keys task_following,task_lead_brake_response,task_queue_approach,task_lane_change,task_cutin_response,task_yield_conflict \
  --min_bin_size 2 \
  --num_bootstrap 100 \
  --num_permutation 200 \
  --overwrite
```

#### C. 运行 same-scenario paired delta

```bash
python tools/stage7f_aggressive_conservative_paired_delta.py \
  --embedding_dir outputs/stage7e_idm_embeddings_20scenes_laneaware_v1 \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_20scenes_laneaware_v1 \
  --stage7f_dir outputs/stage7f_idm_20scenes_full_fallback_preserving_v1 \
  --planner_a idm_longitudinal_aggressive \
  --planner_b idm_longitudinal_conservative \
  --output_dir outputs/stage7f_idm_20scenes_aggressive_vs_conservative_paired_delta_v1 \
  --overwrite
```

### 2. 期望行为

- behavior events 命令读取 Stage7E context dataset 的 `shard_manifest.json` 与 `feature_schema.json`，输出 `behavior_event_bins_v2.csv`、`behavior_event_metrics_v2.csv` 和 schema/warnings；它复用 Stage6C v2 task detector，不修改 Stage5D CORE、`tools/lane_aware_assignment.py` 或 Stage6 metric 定义。
- task-conditioned BDD wrapper 会解析 `embedding_manifest.json`、`shard_manifest.json`、`feature_schema.json` 和 `stage7f_dir/planner_indices/*.npy`，必要时调用 `tools/stage6c_build_behavior_events_v2.py`，然后调用 `tools/stage6c_task_conditioned_bdd_report.py`；它不重新实现 BDD/MMD，也不新增替代 Stage6 的 planner-behavior metric。
- task-conditioned BDD 输出 `task_report_card.md`、`task_bdd_summary.csv`、`task_style_delta.csv`、`top_task_drift_cases.csv`、`warnings.json`、`plots/task_bdd_bar.png`、`plots/task_style_delta_bar.png`、`stage7f_task_bdd_summary.json`、`stage7f_task_bdd_summary.md`。
- paired delta 命令按同一 `scenario_token` / `scenario_id` 对齐 aggressive 与 conservative，同一 scenario 必须同时存在两个 planner；它不会做 unpaired matching，遇到 duplicate scenario-planner pair 会直接失败。
- paired delta 输出 `paired_delta_by_scenario.csv`、`paired_delta_summary.json`、`paired_delta_report.md`、`paired_delta_bar.png`、`embedding_pair_distance_hist.png`，用于检查 nominal planner 参数差异是否产生 realized rollout 差异。`paired_delta_report.md` 不硬编码 IDM 参数定义；它会优先读取 Stage7E `metadata.csv` 的 `parameters_json`，并向上查找 `simulation_schema.json.planner_profiles` / `simulated_planner_metadata.csv`。如果这些真实参数不可用，报告只写 planner names、Delta convention 和 paired delta 摘要，不输出伪造参数定义。

### 3. 通过标准

- A/B planner index 文件存在：`stage7f_dir/planner_indices/idm_longitudinal_aggressive.npy` 与 `stage7f_dir/planner_indices/idm_longitudinal_conservative.npy`。
- paired scenarios 数量 `> 0`，且没有 duplicate scenario-planner pair。
- 如果真实 `parameters_json` 存在，`paired_delta_summary.json` 应记录 `planner_parameter_sources` 与 `planner_parameters_available`，`paired_delta_report.md` 应显示从 metadata / planner profiles 读取的参数；如果不存在，报告不得出现硬编码的 `IDM parameter definitions`。
- task-conditioned BDD 可以生成 summary；如果某些 task 的 `n_A` / `n_B` 低于 `--min_bin_size`，允许被 skip，但必须在 `warnings.json` / skipped tasks 中可见。
- following 与 yield_conflict 是更可靠的 detectors；lead_brake_response、queue_approach、cutin_response 可能是 proxy-based，需要结合 detector strength 和 low-n 提示解释。
- 20-scenario 结果只作为 exploratory diagnostic，不替代完整 Stage7F pairwise BDD；该流程的目的只是解释 aggressive/conservative overall BDD 为什么很小。

## Stage7F task overlap matrix diagnostic（aggressive vs conservative）

## 1. 命令

```bash
python tools/stage7f_task_overlap_matrix.py \
  --events_dir outputs/stage7f_idm_20scenes_stage6c_behavior_events_v2 \
  --stage7f_task_bdd_dir outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_bdd_v1 \
  --embedding_dir outputs/stage7e_idm_embeddings_20scenes_laneaware_v1 \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_20scenes_laneaware_v1 \
  --stage7f_dir outputs/stage7f_idm_20scenes_full_fallback_preserving_v1 \
  --planner_a idm_longitudinal_aggressive \
  --planner_b idm_longitudinal_conservative \
  --task_keys task_following,task_lead_brake_response,task_queue_approach,task_lane_change,task_cutin_response,task_yield_conflict \
  --output_dir outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_overlap_v1 \
  --overwrite
```

## 2. 期望行为

该命令读取 Stage6C v2 的 `behavior_event_metrics_v2.csv` / `behavior_event_bins_v2.csv`、Stage7E 的 `metadata.csv`、Stage7F 的 `planner_indices`，并复用已有 task-conditioned BDD 摘要（如存在）。它只统计不同 task 正类 row set 与 A/B paired scenario set 的 overlap count / Jaccard，不重新实现 BDD/MMD，不修改 task detector 逻辑，不改变 row semantics，也不修改 Stage5D CORE、`tools/lane_aware_assignment.py` 或 Stage6 metric 定义。

预期输出目录为：

- `outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_overlap_v1/task_overlap_report.md`
- `outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_overlap_v1/task_overlap_summary.json`
- `outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_overlap_v1/task_overlap_matrix_all.csv`
- `outputs/stage7f_idm_20scenes_aggressive_vs_conservative_task_overlap_v1/task_overlap_matrix_paired_scenarios.csv`
- 同时还会生成 `task_overlap_matrix_planner_a.csv` 与 `task_overlap_matrix_planner_b.csv`。

## 3. 通过标准

- `task_following` 和 `task_queue_approach` 的 positive counts 非空。
- overlap matrix 文件成功生成，至少包含 all-row 与 paired-scenario 两类矩阵。
- `task_overlap_report.md` 明确报告 following-vs-queue 的 overlap count、Jaccard、row set 是否 identical、paired scenario set 是否 identical。
- 若 following 和 queue 高度重叠或完全相同，报告中将其解释为一个 combined longitudinal interaction evidence cluster，不能作为彼此独立证据过度声称。
- 未修改 Stage5D CORE、`tools/lane_aware_assignment.py` 或 Stage6 metric definitions。

## Stage7P — PDM readiness and smoke preparation

## 1. 命令

先运行只读 readiness check，确认当前 `nuplan-devkit`、本仓库和可选额外搜索路径中是否存在 PDM planner / Hydra config / Python class：

```bash
python tools/stage7p_pdm_readiness_check.py \
  --repo_root /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation \
  --nuplan_devkit_root /home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit \
  --output_dir outputs/stage7p_pdm_readiness_check_v1 \
  --overwrite
```

如外部 PDM 实现已经安装在其他目录，可以追加一个或多个搜索根目录：

```bash
python tools/stage7p_pdm_readiness_check.py \
  --repo_root /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation \
  --nuplan_devkit_root /home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit \
  --extra_search_roots /path/to/external/pdm/repo \
  --output_dir outputs/stage7p_pdm_readiness_check_v1 \
  --overwrite
```

PDM smoke template（**不可直接运行**，仅当 readiness check 明确 `pdm_available=true` 后才可按报告替换占位符）：

```bash
python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir ... \
  --nuplan_db_root ... \
  --nuplan_map_root ... \
  --output_dir outputs/stage7p_pdm_smoke_1scene \
  --planners <confirmed_pdm_planner_name> \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --nuplan_simulation_command_template '<confirmed hydra command here>' \
  --allow_external_planner_name \
  --overwrite
```

## 2. 期望行为

- readiness check 会只读搜索 `nuplan_devkit_root`、`repo_root` 和 `--extra_search_roots` 中的 `*pdm*` / `*PDM*` 文件、包含 `PDM` 的 Python class、包含 `pdm` 的 Hydra/config 文本，以及 `nuplan/planning/script/config` 下的 planner config。
- readiness check 会安全尝试 import `nuplan` 和 Stage7C 已使用的 planner/simulation 模块，并仅在 module spec 存在时尝试导入候选 PDM 模块；导入失败会写入 diagnostics，不会让脚本崩溃。
- 输出目录会生成 `pdm_readiness_summary.json` 和 `pdm_readiness_report.md`；JSON 中必须包含 `pdm_available`、`pdm_config_candidates`、`pdm_module_candidates`、`pdm_class_candidates`、`available_planner_configs` 和 `required_next_action`。
- 如果当前环境没有 PDM，`required_next_action` 应为 `install_external_pdm_implementation`；如果只发现部分路径/模块，可能提示 `configure_external_planner_path`；只有确认 config/module/class 后才应进入 `ready_for_pdm_smoke`。
- 该流程不会安装包、不会 clone 外部仓库、不会修改环境，也不会假设 `planner=pdm_planner` 一定可用。
- Stage7C 仍是 adapter / runner / diagnostic layer；外部 planner 名称必须通过 `--allow_external_planner_name` 显式启用，并且真正的 Hydra planner override 以 `--nuplan_simulation_command_template` 中已确认的命令为准。

## 3. 通过标准

- `outputs/stage7p_pdm_readiness_check_v1/pdm_readiness_summary.json` 成功生成。
- `outputs/stage7p_pdm_readiness_check_v1/pdm_readiness_report.md` 明确说明 PDM 是否可用。
- 若 `pdm_available=false`，报告必须提示下一步安装或配置外部 PDM 实现，不能把 PDM smoke template 标记为可运行命令。
- 若 `pdm_available=true`，再根据 readiness report 中确认的 planner name / config / module 替换 `<confirmed_pdm_planner_name>` 与 `<confirmed hydra command here>`。
- 不要在 readiness report 确认前运行 PDM smoke template；不要直接假设 `planner=pdm_planner` 或任意 PDM Hydra override 可用。
- 不修改 Stage5D CORE、`tools/lane_aware_assignment.py` 或 Stage6 metric definitions。

## Stage7P — PDM closed planner smoke

## 1. 命令

在 readiness check 已经确认 `pdm_available=true`、`required_next_action=ready_for_pdm_smoke`，并且 tuplan_garage 已通过 `pip install -e .` 安装后，先运行 1 个场景的 `pdm_closed_planner` smoke。`pdm_open_planner` 和 `pdm_hybrid_planner` 需要 `checkpoint_path`，这里不要把它们写成可直接运行命令。

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DEVKIT_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit
export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir outputs/stage7p_pdm_closed_smoke_1scene \
  --planners pdm_closed_planner \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --allow_external_planner_name \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
  --nuplan_simulation_command_template 'python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7p_pdm_closed_smoke_1scene job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

PDM smoke 成功后，可继续把该 1-scene 输出转成 Stage5D 5-neighbor context：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7p_pdm_closed_smoke_1scene \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_pdm_closed_smoke_1scene \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root "$NUPLAN_MAPS_ROOT" \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --debug_projection_sample_rows 20 \
  --overwrite
```

再对 PDM smoke context 运行 Stage7E embedding：

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_pdm_closed_smoke_1scene \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_pdm_closed_embeddings_smoke_1scene \
  --overwrite
```

Stage7F pairwise 不需要在单 planner smoke 上运行；等 PDM 与 IDM/simple 形成 paired planner 输出后再运行 Stage7F。

## 2. 期望行为

- Stage7C 读取 `outputs/stage7b4_nuplan_context_merged/merged_metadata.csv` 中最多 1 个场景，并调用官方 nuPlan `run_simulation.py`。
- Stage7C 会在格式化命令后对 `$NUPLAN_DEVKIT_ROOT` 等环境变量执行 `os.path.expandvars()`，因此这里允许在 command template 中使用 `$NUPLAN_DEVKIT_ROOT`。
- `{scenario_hydra_overrides}` 会由 Stage7C 按目标场景元数据替换：优先 `scenario_filter.scenario_tokens=[token]`，没有 nuPlan scenario token 时使用 `scenario_filter.log_names=[target_log_name] scenario_filter.limit_total_scenarios=1`；不要再用 `scenario_filter=all_scenarios scenario_filter.scenario_tokens=null` 作为最终控制机制。
- `--allow_external_planner_name` 允许 Stage7C 把 `pdm_closed_planner` 当作外部 Hydra planner adapter 传给 `{planner_hydra_overrides}`，不要求它存在于 Stage7C 内置 IDM/simple profile 中。
- `--hydra_searchpath` 会追加为 `hydra.searchpath="..."` 形式的 Hydra override，使 Hydra 能找到 tuplan_garage 的 PDM planner config，同时不会强制影响标准 IDM/simple runs。
- 该命令仍保持 Stage7 row semantics：一行表示一个 `scenario × planner-controlled nuPlan ego rollout`。
- 输出目录继续写入 `simulation_report.md`、`warnings.json`、progress JSON、官方命令 log、解析后的 trajectory CSV/NumPy tensor。
- 后续 Stage5D context 构建命令读取 PDM smoke 的 Stage7C simulation 输出，使用 lane-aware with geometric fallback assignment，不修改 Stage5D CORE 或 `tools/lane_aware_assignment.py`。
- Stage7E embedding 命令读取 PDM smoke context 和既有 Stage5D checkpoint，只导出 embedding，不修改 Stage6 metric definitions。

## 3. 通过标准

- Stage7C PDM closed smoke 中 official command successes = 1。
- `warnings.json` / 官方命令 log 中 final official command 已展开 `$NUPLAN_DEVKIT_ROOT`，并包含由 `{scenario_hydra_overrides}` 注入的 token 或 log_name 场景约束。
- `same_scenario_alignment_required = true`，且 `scenario_alignment.passed = true`：actual nuPlan log_name 匹配 target_log_name，或 actual token 匹配 target token。
- `simulation_report.md` 中 `pseudo_rollout = false`。
- 至少找到并解析一个官方 nuPlan msgpack trajectory artifact。
- `simulated_ego_seq.npy` shape 为 `(1, 1, T, 8)`，且 `T >= 2`。
- missing scenario-planner pair count = 0。


## Stage7P — PDM closed variant smoke / pilot

## 1. 命令

### 1-scene variant smoke

该命令一次运行 3 个有标签的 PDM closed profile。注意：Stage7C 的 `--planners` 使用 variant label，但 `{planner_hydra_overrides}` 内部仍展开为 `planner=pdm_closed_planner` 加对应参数 override。

```bash
cd /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation

export NUPLAN_DEVKIT_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan-devkit
export NUPLAN_DATA_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset
export NUPLAN_MAPS_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/home/forwardxp/00_nuplan_E2E_eva/nuplan/exp
mkdir -p "$NUPLAN_EXP_ROOT"

python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir outputs/stage7p_pdm_closed_variant_smoke_1scene \
  --planners pdm_closed_default pdm_closed_conservative_v1 pdm_closed_assertive_v1 \
  --max_scenarios 1 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
  --nuplan_simulation_command_template 'python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7p_pdm_closed_variant_smoke_1scene job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

### 5-log variant pilot

该命令抽取最多 5 个 distinct log，并比较 PDM closed variants 与 simple baseline；如需要也可加入 `idm_longitudinal_conservative`。

```bash
python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir outputs/stage7p_pdm_closed_variant_pilot_5logs \
  --planners pdm_closed_default pdm_closed_conservative_v1 pdm_closed_assertive_v1 simple_planner idm_longitudinal_conservative \
  --sample_distinct_log_names \
  --max_scenarios 5 \
  --min_timesteps 2 \
  --require_same_scenario_alignment \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
  --nuplan_simulation_command_template 'python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7p_pdm_closed_variant_pilot_5logs job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --overwrite
```

如果只想运行 4 个 planner，把上面命令中的 `idm_longitudinal_conservative` 删除即可。

## 2. 期望行为

- `pdm_closed_default`、`pdm_closed_conservative_v1`、`pdm_closed_assertive_v1` 都通过同一个 Hydra config `planner=pdm_closed_planner` 启动；conservative/assertive 额外注入已验证存在的 `planner.pdm_closed_planner.*` 参数。
- Stage7C 输出中的 `planner_name`、`scenario_planner_index.csv`、`simulated_planner_metadata.csv`、`warnings.json.planner_api_discovery` 和 `job_name=stage7c_{planner_name_safe}` 保留 requested variant label，例如 `pdm_closed_assertive_v1`。
- `warnings.json.planner_api_discovery` 中应同时能看到 requested `planner_name=pdm_closed_conservative_v1`、`nuplan_planner_config=pdm_closed_planner` 和完整 `hydra_overrides`。
- `{scenario_hydra_overrides}` 继续负责 same-scenario / same-log 对齐；不要手工把 `scenario_filter=all_scenarios` 和 `scenario_filter.scenario_tokens=null` 混入这些 smoke 命令。
- 命令只产生 Stage7C simulation 输出，不会修改 Stage5D CORE、lane-aware assignment 或 Stage6 metrics。

## 3. 通过标准

- 1-scene smoke 的 expected scenario-planner pairs = `1 × 3 = 3`，official command successes = 3，`simulated_ego_seq.npy` 第一、二维为 `(1, 3, ...)`。
- 5-log pilot 使用 5 个 distinct log 时，若运行 5 个 planner，则 expected pairs = `5 × 5 = 25`；若删除 IDM 只运行 4 个 planner，则 expected pairs = `5 × 4 = 20`。
- `same_scenario_alignment_required = true`，且 `scenario_alignment.passed = true`。
- PDM variant 的 metadata `parameters_json` 包含对应 speed limit fraction、fallback target velocity、min gap、headway、accel/decel 和 lateral offset 参数。
- `official_nuplan_runs/scenario_*/pdm_closed_conservative_v1/` 等输出目录使用 variant label，而不是 base `pdm_closed_planner` 覆盖不同 variant。
- 如果 Hydra 找不到 `pdm_closed_planner`，优先检查 tuplan_garage 是否已经在 `/home/forwardxp/00_nuplan_E2E_eva/tuplan_garage` 执行 `pip install -e .`，以及 `--hydra_searchpath` 是否完整传入。
- `pdm_open_planner` 和 `pdm_hybrid_planner` 因需要 `checkpoint_path` 暂不作为可直接运行 smoke 命令。
- 首轮使用 `closed_loop_nonreactive_agents`，保持与 IDM Stage7 pipeline 一致。
- Stage7F pairwise 只在 PDM 与 IDM/simple 已生成 paired planner 输出后运行。

## Stage7P — PDM closed config parameter discovery

## 1. 命令

```bash
python tools/stage7p_pdm_config_parameter_report.py \
  --tuplan_garage_root /home/forwardxp/00_nuplan_E2E_eva/tuplan_garage \
  --planner_config_name pdm_closed_planner \
  --output_dir outputs/stage7p_pdm_closed_config_params_v1 \
  --overwrite
```

## 2. 期望行为

- 脚本只读解析 `tuplan_garage/.../config/simulation/planner/pdm_closed_planner.yaml`，并扫描相关 PDM Python class 的 `__init__` 签名。
- 输出 YAML key/value、`_target_` 路径、数值标量、数值列表、boolean flag、class 参数默认值，并按 route/path、speed/progress、lateral、proposal、scoring、comfort、safety、simulator、unknown 分组。
- `pdm_closed_variant_blueprint.md` 只提出候选 override group；除非参数能从 YAML key 或 class arg 中验证，否则标记为 `inferred_candidate` / `no safe override yet`，不会生成 PDM open/hybrid 或可运行 variant 命令。

## 3. 通过标准

- 输出目录包含：
  - `pdm_closed_parameter_report.md`
  - `pdm_closed_parameter_summary.json`
  - `pdm_closed_parameter_table.csv`
  - `pdm_closed_variant_blueprint.md`
- JSON/CSV 中能看到 `pdm_closed_planner.yaml` 的 numeric / boolean / list-valued 参数。
- blueprint 不声称 candidate variant 已最终可运行；未验证参数必须报告 `no safe override yet`。

---

# Stage 7C / 7E / 7P 当前修正版命令（2026-06-18）

## Stage 7C PDM closed smoke（scenario_hydra_overrides + expandvars 修正版）

### 1. 命令

```bash
python tools/stage7c_run_external_planner_simulation.py \
  --context_dir /home/forwardxp/00_nuplan_E2E_eva/E2E-Evaluation/outputs/stage7b4_nuplan_context_merged \
  --output_dir outputs/stage7c_pdm_closed_smoke_v2 \
  --planner_names pdm_closed \
  --planner_hydra_overrides '+planner=pdm_closed_planner' \
  --scenario_hydra_overrides 'scenario_filter.log_names=["{target_log_name}"] scenario_filter.limit_total_scenarios=1' \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7c_pdm_closed_smoke job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --nuplan_devkit_root '$NUPLAN_DEVKIT_ROOT' \
  --nuplan_data_root '$NUPLAN_DATA_ROOT' \
  --nuplan_map_root '$NUPLAN_MAP_ROOT' \
  --max_scenarios 1 \
  --timesteps 149 \
  --overwrite
```

### 2. 期望行为

- `$NUPLAN_DEVKIT_ROOT` / `$NUPLAN_DATA_ROOT` / `$NUPLAN_MAP_ROOT` 可以保留为环境变量形式，因为 Stage 7C runner 已实现并测试 `os.path.expandvars`。
- `{scenario_hydra_overrides}` 必须出现在 `--nuplan_simulation_command_template` 中，由 CLI 单独注入 log/scenario 过滤条件。
- 不要再使用 `scenario_filter.scenario_tokens=null`；当前 smoke 以 log/scenario hydra override 机制对齐官方 nuPlan simulation。
- 输出 official nuPlan simulation artifacts、`simulated_ego_seq.npy`、mask、metadata、alignment validation；不允许 pseudo rollout。

### 3. 通过标准

- `warnings.json.validation.pass == true`。
- `pseudo_rollout == false`。
- official command successes 至少为 1。
- `simulated_ego_seq.npy` 形状符合当前 smoke 期望，例如 `[1, 1, 149, 8]`。
- scenario alignment 中 `same_log_alignment_passed == true` 且 `alignment_pass_ratio == 1.0`。

## Stage 7E PDM context build（低覆盖 slot sanity 非致命）

### 1. 命令

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c_pdm_closed_smoke_v2 \
  --output_dir outputs/stage7e_pdm_closed_context_v2 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root '$NUPLAN_MAP_ROOT' \
  --slot_sanity_min_coverage 0.05 \
  --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 1.0 \
  --overwrite
```

### 2. 期望行为

- 读取 Stage 7C official PDM closed rollout artifacts 和 nuPlan msgpack，构建 Stage5D-compatible `context_traj.npy [N,T,83]`。
- Stage5D CORE、slot order、83-dim schema、formula parity、row semantics 仍然是硬约束。
- 单场景 smoke 中某个语义 slot（例如 `right_front`）没有对象或覆盖率低于 `--slot_sanity_min_coverage` 时，会写入 `slot_sanity_insufficient_coverage` warning，并在 report 中标记为 skipped；这只是诊断信息，不会让 context validation 失败。
- 如果某个 coverage 足够的 slot 方向中位数违反语义预期，仍然会让 validation 失败。

### 3. 通过标准

- `warnings.json.validation.pass == true`。
- `warnings.json.validation.context_dim == 83` 或 `stage5d_dim_matched == true`。
- `warnings.json.validation.context_traj_no_nonfinite == true`。
- `stage5d_core_reused == true`，slot schema/order matched。
- `stage5d_derived_formula_matched == true`。
- `context_build_report.md` 包含 slot coverage、evaluated slots、skipped low-coverage slots、failed sufficiently-covered slots。
- strict filter report 中 `slot_coverage_on_kept_rows` 仍保留，但低覆盖/缺失 slot 不代表 context 无效。

## Stage 7P PDM parameter discovery（YAML inline comment 解析修正版）

### 1. 命令

```bash
python tools/stage7p_pdm_config_parameter_report.py \
  --tuplan_garage_root /home/forwardxp/00_nuplan_E2E_eva/tuplan_garage \
  --planner_config_name pdm_closed_planner \
  --output_dir outputs/stage7p_pdm_closed_config_params_v1 \
  --overwrite
```

### 2. 期望行为

- 读取 `tuplan_garage/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml`，保留 source line number，并清理 inline comments 后再解析值。
- `num_poses`、`interval_length`、`speed_limit_fraction`、`fallback_target_velocity`、`min_gap_to_lead_agent`、`headway_time`、`accel_max`、`decel_max`、`lateral_offsets`、`map_radius` 等应分类为 clean numeric scalar/list/bool/string/target_path。
- 输出 `pdm_closed_parameter_report.md`、`pdm_closed_parameter_summary.json`、`pdm_closed_parameter_table.csv`、`pdm_closed_variant_blueprint.md`。
- variant blueprint 只对已验证存在于 YAML 的 `verified_config_key` 输出 concrete override candidates；PDM open/hybrid 不标记为 runnable，因为需要 `checkpoint_path`。

### 3. 通过标准

- inline comments 不应进入 parsed value。
- `lateral_offsets` 和 `speed_limit_fraction` 是 `numeric_list`。
- numeric scalar 是 `numeric_scalar`。
- concrete override candidate 行必须标记 `verified_config_key`，未知或未验证项只能标记为 `inferred_candidate` / `unsafe_unknown`，不能作为 runnable command。


## Stage7P lane-change candidate discovery（PDM lateral smoke 场景筛选）

## 1. 命令

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --output_dir outputs/stage7p_lane_change_candidates_v1 \
  --top_k 20
```

如果已经有 Stage7 behavior event detector 输出，也可以显式传入：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --behavior_events_dir outputs/stage7f_idm_20scenes_stage6c_behavior_events_v2 \
  --output_dir outputs/stage7p_lane_change_candidates_v1 \
  --top_k 20
```

如果需要启用轨迹驱动的 high-lateral-motion / lane-change 候选发现，可以扫描 nuPlan mini DB（或单个 `.db` 文件）：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /path/to/nuplan/mini \
  --nuplan_map_root /path/to/nuplan/maps \
  --enable_kinematic_scan \
  --max_scenarios_scan 50 \
  --min_lateral_displacement 2.0 \
  --min_heading_change 0.25 \
  --min_yaw_rate_proxy 0.05 \
  --output_dir outputs/stage7p_lane_change_candidates_kinematic \
  --top_k 20
```

## 2. 期望行为

- 脚本读取 `context_dir/merged_metadata.csv`（如不存在则尝试 `context_dir/metadata.csv`），不会读取或合并 `context_traj.npy`、`neighbor_seq.npy`、`ego_seq.npy` 等大数组。
- 优先搜索 `scenario_type` / `scenario_label` / `type` 中包含 `changing_lane`、`lane_change`、`high_lateral_acceleration`、`near_multiple_vehicles`、`cut_in`、`merge` 的场景。
- 同时扫描 scenario labels、log names、scenario ids 等文本字段中的 lane-change-like 关键词。
- 如果能找到 `behavior_event_bins_v2.csv` 且其中存在 `task_lane_change`，则把 `task_lane_change=1` 的 rows 作为额外候选信号；如果该文件不存在，脚本写入 warning 并继续，不会崩溃。
- 启用 `--enable_kinematic_scan` 时，脚本会在 `--nuplan_db_root` 下查找 SQLite `.db`，进行 schema discovery，并尝试从包含 `x/y/yaw` 或 `x/y/heading` 的 pose-like 表计算 expert ego 横向位移、heading change、yaw-rate proxy、max lateral speed proxy 和 candidate score。当前 fallback 只做候选发现和 schema 诊断，不修改 PDM、不修改 Stage5D、不生成 adjacent-lane proposal。
- kinematic candidate score 使用 `2.0 * abs_lateral_displacement + 5.0 * heading_change_abs + 2.0 * yaw_rate_proxy + text_match_bonus`；满足 `abs_lateral_displacement >= min_lateral_displacement` 或 `heading_change_abs >= min_heading_change` 或 `yaw_rate_proxy >= min_yaw_rate_proxy` 的场景会进入候选。
- 输出 `lane_change_candidate_report.md`、`lane_change_candidate_summary.json`、`lane_change_candidate_metadata.csv`。报告和 summary 会区分 `text_match_candidates`、`behavior_event_candidates`、`kinematic_candidates`、`final_selected_candidates`。
- 该步骤只做候选场景筛选，不修改 Stage5D CORE、`tools/lane_aware_assignment.py`、Stage6 metric definitions，也不改变 PDM planner 配置。

## 3. 通过标准

- `lane_change_candidate_summary.json` 存在，且记录 `metadata_rows`、`candidate_rows`、`top_k_written`、`text_match_candidates`、`behavior_event_candidates`、`kinematic_candidates`、`final_selected_candidates` 和 behavior-event detector 是否可用。
- `lane_change_candidate_metadata.csv` 存在，包含 `candidate_rank`、`metadata_index`、`candidate_source`、`candidate_score`、`match_score`、`match_sources`，最多输出 `top_k` 行；启用 kinematic scan 后还应包含 lateral displacement / heading change / yaw-rate proxy 等字段。
- `lane_change_candidate_report.md` 存在，并列出匹配规则、source counts、warnings 和 top candidates。
- 如果本地 metadata 中没有任何 lane-change-like 文本且没有 `task_lane_change=1`，脚本应正常输出空候选报告，而不是崩溃；报告必须说明 `candidate_rows=0` 是 metadata-only / optional-kinematic discovery 没有命中，不代表 PDM 没有 lane-change 能力。
- 如果 nuPlan DB schema 与 fallback 假设不匹配，脚本应在 summary 的 `kinematic_scan.schema_discovery` / `kinematic_scan.warnings` 中记录可用 tables/columns 和 warning，而不是崩溃。

## Stage7P mini DB scenario_tag lane-change candidate discovery（Stage7C context 输出）

## 1. 命令

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --write_stage7c_context_dir \
  --max_per_log 2 \
  --output_dir outputs/stage7p_lane_change_candidates_from_db_v1 \
  --top_k 20
```

可选限制扫描量：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --max_db_files 2 \
  --max_candidates_per_type 20 \
  --max_per_log 2 \
  --write_stage7c_context_dir \
  --output_dir outputs/stage7p_lane_change_candidates_from_db_v1 \
  --top_k 20
```

重新生成 PDM v2 strict lane-change Stage7C 候选 context 时，建议使用 `--prefer_exact_changing_lane`：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --prefer_exact_changing_lane \
  --max_per_log 2 \
  --write_stage7c_context_dir \
  --output_dir outputs/stage7p_lane_change_candidates_strict_v2 \
  --top_k 20
```

## 2. 期望行为

- 脚本保留原有 `merged_metadata.csv` 文本匹配逻辑，同时在启用 `--scan_db_scenario_tags` 且提供 `--nuplan_db_root` 时直接扫描 mini DB 的 `scenario_tag.type`。
- DB 扫描只遍历 `nuplan_db_root/*.db`，读取 `scenario_tag(token, lidar_pc_token, type, agent_track_token)`，并关联 `lidar_pc(token, scene_token, ego_pose_token)` 与 `log(logfile, token)`。
- 匹配的 scenario tag 类型包括 `changing_lane`、`lane_change`、`high_lateral_acceleration`、`near_multiple_vehicles`、`cut_in`、`merge`。
- 如果 SQLite token 是 BLOB，会转换为 hex string 写入 CSV/JSON，避免二进制 token 破坏输出格式。
- 写入 Stage7C context 时，`scenario_token` 来自 `scenario_tag.lidar_pc_token`，可直接作为 `scenario_filter.scenario_tokens=[...]` 使用；DB 原始 `lidar_pc.scene_token` 仅保留为 `db_scene_token`，不会覆盖 nuPlan scenario token。
- 同一 `scenario_tag.lidar_pc_token` 有多个 `scenario_tag.type` 时按 `scenario_token` 去重，并按 `changing_lane_to_left`、`changing_lane_to_right`、`changing_lane`、`high_lateral_acceleration`、`cut_in`、`merge`、`near_multiple_vehicles` 的优先级保留最严格类型。
- `--max_per_log` 默认是 `2`，用于避免一个 log 占满 `top_k`。启用 `--prefer_exact_changing_lane` 时，top_k selection 会优先选择 `changing_lane_to_left`、`changing_lane_to_right`、`changing_lane`；strict changing-lane 候选不足时，再补充 `high_lateral_acceleration`、`cut_in`、`merge`、`near_multiple_vehicles`。
- 标准输出仍写入 `lane_change_candidate_report.md`、`lane_change_candidate_summary.json`、`lane_change_candidate_metadata.csv`。
- 启用 `--write_stage7c_context_dir` 时，会额外写出 `stage7c_candidate_context/merged_metadata.csv`，至少包含非空 `log_name`、`scenario_token`、`scene_token`（兼容旧字段，值同 `scenario_token`）、`db_scene_token`、`scenario_type`、`source`、`db_file`，并按 `scenario_token` 去重，供 Stage7C 读取。
- 该命令只增强 lane-change candidate discovery；不修改 PDM、不修改 Stage5D、不修改 Stage6、不生成 v2 深层参数，也不做 adjacent-lane proposal。

## 3. 通过标准

- `lane_change_candidate_summary.json` 中应包含 `metadata_text_candidate_rows`、`behavior_event_candidate_rows`、`db_scenario_tag_candidate_rows`、`final_candidate_rows`、`scenario_type_counts`、`selected_scenario_type_counts`、`raw_db_scenario_tag_rows`、`unique_scenario_token_rows`、`selected_rows`、`selected_log_counts`、`duplicate_scenario_token_count_removed`、`strict_changing_lane_candidate_rows`、`selected_strict_changing_lane_rows`。
- 当 23-row Stage7B merged metadata 没有文本候选、但 mini DB 有 lane-change/lateral scenario tag 时，报告应明确写出 `metadata_text candidates: 0`、`db_scenario_tag candidates: N`，并说明原 Stage7B merged subset 不富含 lane-change，但 mini DB 包含候选 tag。
- `lane_change_candidate_metadata.csv` 的 DB 候选行应包含 `db_file`、`log_name`、`scenario_type`、`scenario_tag_token`、`scenario_token`、`lidar_pc_token`、`scene_token`、`ego_pose_token`、`source=db_scenario_tag`、`candidate_score`。
- 如果启用 `--write_stage7c_context_dir`，`stage7c_candidate_context/merged_metadata.csv` 必须存在，并包含 Stage7C 所需关键列；`log_name` 不允许为空，`scenario_token` 必须是 nuPlan scenario token namespace（`scenario_tag.lidar_pc_token`）。

重新跑 Stage7C PDM v1 strict lane-change 5scenes 时，使用重新生成的 candidate context：

```bash
python tools/stage7c1_run_nuplan_simulation.py \
  --context_dir outputs/stage7p_lane_change_candidates_strict_lane_change_v1/stage7c_candidate_context \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --output_dir outputs/stage7c_pdm_v1_strict_lane_change_5scenes \
  --planners pdm_closed_default,pdm_closed_conservative_v1,pdm_closed_assertive_v1 \
  --allow_external_planner_name \
  --hydra_searchpath '[pkg://tuplan_garage.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
  --nuplan_simulation_command_template 'python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py {planner_hydra_overrides} {scenario_hydra_overrides}' \
  --require_same_scenario_alignment \
  --require_strict_nuplan_token_alignment \
  --max_scenarios 5 \
  --overwrite
```

---

# Stage7P — strict verified lane-change candidate selector（2026-06-19）

## 1. 命令

DB-tag-only strict candidate generation（默认允许 actual type 为空的 strict DB tag 进入 selected，但必须标记为 DB-tag-only）：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --prefer_exact_changing_lane \
  --verify_actual_scenario_type \
  --actual_type_allowlist changing_lane,changing_lane_to_left,changing_lane_to_right \
  --allow_db_tag_when_actual_type_unverified \
  --max_per_log 2 \
  --write_stage7c_context_dir \
  --output_dir outputs/stage7p_lane_change_candidates_strict_db_tag_only_v3 \
  --top_k 20
```

verified-only candidate generation（只允许 actual type 已验证的 strict lane-change rows 进入 selected）：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --prefer_exact_changing_lane \
  --verify_actual_scenario_type \
  --require_actual_type_verified \
  --actual_type_allowlist changing_lane,changing_lane_to_left,changing_lane_to_right \
  --max_per_log 2 \
  --write_stage7c_context_dir \
  --output_dir outputs/stage7p_lane_change_candidates_strict_verified_only_v3 \
  --top_k 20
```

如果本地 DB 无法直接解析 actual type，verified-only 模式可以接入已人工/Stage7C 验证的 known-good cache 或 Stage7C alignment feedback：

```bash
python tools/stage7p_find_lane_change_candidates.py \
  --context_dir outputs/stage7b4_nuplan_context_merged \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/nuplan-v1.1/splits/mini \
  --scan_db_scenario_tags \
  --prefer_exact_changing_lane \
  --verify_actual_scenario_type \
  --require_actual_type_verified \
  --actual_type_cache outputs/stage7p_known_good_actual_type_cache.csv \
  --stage7c_alignment_feedback outputs/stage7c_pdm_lane_change/scenario_alignment.csv \
  --actual_type_allowlist changing_lane,changing_lane_to_left,changing_lane_to_right \
  --max_per_log 2 \
  --write_stage7c_context_dir \
  --output_dir outputs/stage7p_lane_change_candidates_strict_verified_cache_v4 \
  --top_k 20
```

2 scenes × 2 planners 的 Stage7C smoke 可以直接读取上一步写出的 `stage7c_candidate_context/merged_metadata.csv`：

```bash
python tools/stage7c_run_external_planner_simulation.py \
  --context_dir outputs/stage7p_lane_change_candidates_strict_verified_v2/stage7c_candidate_context \
  --output_dir outputs/stage7c_pdm_lane_change_strict_verified_2scenes_2planners_v1 \
  --planner_names pdm_closed,pdm_closed_conservative_v1 \
  --planner_hydra_overrides '+planner=pdm_closed_planner' \
  --scenario_hydra_overrides 'scenario_filter.scenario_tokens=["{target_scenario_token}"] scenario_filter.limit_total_scenarios=1' \
  --nuplan_simulation_command_template 'python -m nuplan.planning.script.run_simulation +simulation=closed_loop_nonreactive_agents {planner_hydra_overrides} scenario_builder=nuplan_mini scenario_filter=all_scenarios {scenario_hydra_overrides} worker=single_machine_thread_pool experiment_name=stage7c_pdm_lane_change_strict_verified_2scenes_2planners job_name=stage7c_{planner_name_safe} output_dir={output_dir}' \
  --nuplan_devkit_root '$NUPLAN_DEVKIT_ROOT' \
  --nuplan_data_root '$NUPLAN_DATA_ROOT' \
  --nuplan_map_root '$NUPLAN_MAP_ROOT' \
  --max_scenarios 2 \
  --timesteps 149 \
  --overwrite
```

## 2. 期望行为

- selector 仍先用 Stage7 metadata text / behavior-event / mini DB `scenario_tag.type` 生成候选池，但 DB `scenario_tag.type` 只作为候选标签，不等同于最终 official nuPlan scenario builder 会解析到的 `actual_scenario_type`。
- 开启 `--verify_actual_scenario_type` 后，verified actual type 属于 `changing_lane,changing_lane_to_left,changing_lane_to_right` 的候选会进入 verified strict lane-change selected set；验证顺序为输入 metadata、`--actual_type_cache` known-good cache、`--stage7c_alignment_feedback`、SQLite exact-token sidecar lookup。如果 actual-type lookup 失败或返回空值，但 DB `scenario_tag.type` 是 `changing_lane / changing_lane_to_left / changing_lane_to_right` 且 `--allow_db_tag_when_actual_type_unverified` 为 true（默认 true），则不会直接丢弃该 strict DB-tag 候选，而是写入 selected set 并标记 `actual_type_verified=false`、`selected_by_db_tag_only=true`、`actual_type_verification_error=<reason>`，report 中必须称为 DB-tag-only strict lane-change candidates，不能称为 verified strict。
- `--actual_type_cache` 支持 CSV/JSON，至少提供 `scenario_token`（或 `lidar_pc_token`）与 `actual_scenario_type`；usable cache row 会被视为 verified actual type，因此 `--require_actual_type_verified` 不再因为 SQLite lookup 为空而选不出候选。
- `--stage7c_alignment_feedback` 支持 Stage7C `scenario_alignment.csv/json`；如果反馈中的 `actual_scenario_type` 是 `traversing_pickup_dropoff` 等非 allowlist 类型，即使 DB tag 是 strict changing-lane，也必须进入 rejected 统计，不能混入 final selected rows。
- 如果没有开启 `--allow_fallback_lateral_types`，`high_lateral_acceleration` 不会补位；如果开启 fallback，则 fallback rows 会单独标记 `selected_as_fallback_lateral=true`，不会混入 strict lane-change 统计。
- `lane_change_candidate_metadata.csv` 与 `stage7c_candidate_context/merged_metadata.csv` 的 `scenario_token` 均使用可传给 `scenario_filter.scenario_tokens=[...]` 的 nuPlan token（即 DB `scenario_tag.lidar_pc_token`），`scene_token` 为 Stage7C 兼容字段并写成同一 token，原始 DB scene token 写入 `db_scene_token`。
- `stage7c_candidate_context/merged_metadata.csv` 只包含 final selected rows，并写入 `actual_scenario_type`、`actual_type_verified`、`selected_by_db_tag_only`、`actual_type_verification_error`。verified non-lane-change actual type（例如 `traversing_pickup_dropoff`）会被剔除；actual-type 未验证但 strict DB tag 命中的行会作为 DB-tag-only strict candidate 保留。若启用 `--require_actual_type_verified`，actual type 为空或未验证的 rows 不能进入 final selected rows。若 `final_selected_rows=0`，不写看似可用的 empty Stage7C context，并在 summary/report 中写明 insufficient candidates。

## 3. 通过标准

- `lane_change_candidate_summary.json` 包含 `strict_db_tag_candidate_rows`、`strict_actual_type_verified_rows`、`strict_actual_type_unverified_but_db_tag_selected_rows`、`strict_actual_type_rejected_rows`、`selected_db_tag_only_rows`、`selected_actual_type_verified_rows`、`selected_actual_type_empty_rows`、`actual_type_verification_failed_rows`、`final_selected_rows`、`insufficient_strict_changing_lane_warning`、`strict_db_tag_candidates_exist_but_none_selected`、`actual_type_cache`、`stage7c_alignment_feedback` 等字段；`selected_actual_scenario_type_counts` 不统计空字符串。
- 未开启 fallback 时，Stage7C context 中所有 selected rows 要么 verified `actual_scenario_type` 属于 `changing_lane / changing_lane_to_left / changing_lane_to_right`，要么 `actual_type_verified=false`、`selected_by_db_tag_only=true` 且 `scenario_type_db_tag` 属于 strict changing-lane 三类；若 verified actual type 明确是非 lane-change（例如 `traversing_pickup_dropoff`），必须被剔除。开启 `--require_actual_type_verified` 后，后一类 DB-tag-only rows 必须被剔除。
- 如果存在 strict DB-tag candidates 但 `final_selected_rows=0`，`lane_change_candidate_summary.json` 和 report 必须写出 `strict_db_tag_candidates_exist_but_none_selected=true` / insufficient warning，且不应把 empty `stage7c_candidate_context/merged_metadata.csv` 误判为 OK。
- `log_name` 非空，`scenario_token` 不重复，`scene_token == scenario_token`，`db_scene_token` 保留原始 DB scene token。
- Stage7C smoke 的 expected scenario-planner pairs = `2 × 2 = 4`，`warnings.json.validation.pass == true`，`scenario_alignment.passed == true`，且 official command successes 为 4。

## Stage7 Milestone 1 non-contiguous-axis repair and aligned rebuild

Stage7E 必须读取 `simulated_ego_seq_index.json`，不能假设成功场景轴连续。以下命令复用已有 17 个成功 official rollouts，不重新运行 Stage7C simulation：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7p_pdm_v1_balanced_20_stage7c_v1 \
  --output_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --debug_projection_sample_rows 20 \
  --overwrite

python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v2_aligned \
  --device cuda \
  --overwrite

python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v2_aligned \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --output_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/report_card \
  --mode full \
  --run_stage6_pairwise \
  --overwrite

python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned/shard_manifest.json \
  --feature_schema_path outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned/feature_schema.json \
  --output_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/behavior_events_v2 \
  --overwrite

python tools/stage7f_aggressive_conservative_paired_delta.py \
  --embedding_dir outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v2_aligned \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --stage7f_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/report_card \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --output_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/paired_delta_assertive_minus_conservative \
  --overwrite

python tools/stage7f_run_task_conditioned_bdd.py \
  --embedding_dir outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v2_aligned \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --stage7f_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/report_card \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --behavior_events_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/behavior_events_v2 \
  --task_keys task_following,task_lead_brake_response,task_queue_approach,task_lane_change,task_cutin_response,task_yield_conflict \
  --min_bin_size 2 \
  --output_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned/task_bdd_assertive_minus_conservative_v1 \
  --overwrite

python tools/audit_stage7_m1_data_credibility.py \
  --sim_dir outputs/stage7p_pdm_v1_balanced_20_stage7c_v1 \
  --context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --embedding_dir outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v2_aligned \
  --stage7f_dir outputs/stage7f_pdm_v1_balanced20_paired17_v2_aligned \
  --output_dir outputs/stage7_m1_pdm_data_quality_audit_v2_aligned \
  --overwrite
```

通过标准：

- metadata 原始 scenario axis 为 `0..11,14..18`，不是重新编号后的 `0..16`；
- 34/34 行通过 scenario/planner/token/msgpack 严格一致性检查；
- `warnings.json.validation.scenario_planner_token_alignment_strict=true`；
- `warnings.json.validation.msgpack_global_fallback_disabled=true`；
- `ego_seq_mask.npy` shape=`[34,150]` 且逐元素匹配 Stage7C `simulated_ego_seq_mask.npy`；
- `interaction_feat_style.npy` 只聚合 mask=true 的有效 rollout 帧；
- Stage6C `behavior_event_warnings_v2.json` 包含 `rollout_validity_mask_applied`，padding 帧不得进入导数、事件检测或 raw physical diagnostics；
- `yaw_rate/speed` 曲率仅在 `|speed| >= 0.5 m/s` 时定义；更低速度的曲率为 NaN，不使用分母下限制造大曲率；
- context shape=`[34,150,83]`，embedding shape=`[34,64]`；
- Stage7F full mode 保持 17 个完整 planner pair；
- Milestone 1 重审输出 `PASS_WITH_LIMITATIONS` 时，limitations 必须继续在论文结论中披露，不能把 high fallback、过小 strict subset 或 padding-frame physical warning 隐藏为 PASS。

## Stage 7 Milestone 2A：逐场景地图投影与 fallback 修复

### 1. 命令

复用已有 17 个成功 rollout，不重新执行 Stage7C simulation：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7p_pdm_v1_balanced_20_stage7c_v1 \
  --output_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v3_m2a \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/data/cache/mini \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --debug_projection_sample_rows 34 \
  --debug_projection_max_frames_per_row 30 \
  --debug_projection_max_candidates_per_frame 16 \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --overwrite

python tools/audit_stage7_m2a_lane_assignment.py \
  --baseline_context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v2_aligned \
  --repaired_context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v3_m2a \
  --output_dir outputs/stage7_m2a_lane_assignment_audit_v1 \
  --overwrite
```

embedding 和 Stage7F 使用 `v3_m2a` context 重建到：

```text
outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v3_m2a/
outputs/stage7f_pdm_v1_balanced20_paired17_v3_m2a/
outputs/stage7_m1_pdm_data_quality_audit_v3_m2a/
```

### 2. 期望行为

- lane cache 的作用域必须是 `(map_name, source scenario_index)`，不能把同一地图第一个场景的局部 lane 集合复用于其他场景。
- 同一场景的两个 planner 共同定义地图查询覆盖范围。
- 构建器优先从 nuPlan log DB 的 `log.map_version` 解析真实地图；`--map_name` 仅是最后兜底。
- `log.location=las_vegas` 等旧内部别名不能直接传给 map factory，必须使用 canonical `map_version=us-nv-las-vegas-strip`。
- 输出 `nuplan_lane_assignment_by_row.csv` 和 `nuplan_lane_assignment_diagnostics.json`，覆盖全部 34 行及所有有效帧。
- 投影失败必须区分 `lateral_distance_exceeded`、`heading_difference_exceeded`、`no_projectable_lane` 等原因。

### 3. 通过标准

- `lane_cache_scope=map_name_plus_source_scenario`；
- lane cache entries=`17`，map API cache entries=`4`；
- 34/34 行诊断完整，每个场景包含两个 planner；
- geometric fallback rate `<0.5`，且相对 v2 绝对下降至少 `0.3`；
- 实际结果：fallback `0.865104 → 0.016148`，相对减少 `98.13%`；
- ego lane projection success rate=`0.983852`；
- `lane_map_unavailable` fallback=`0`，剩余 82 帧均为 `heading_difference_exceeded`；
- strict-0.8：`20/34` 行、7 个完整 planner-paired 场景；
- Milestone 2A audit verdict=`PASS`。

## Stage 7 Milestone 2B：lane-context 质量分层、成对门控与扩容判定

### 1. 命令

使用与 Milestone 2A 相同的 17 个成功 rollout 重建带逐行质量统计的 context：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7p_pdm_v1_balanced_20_stage7c_v1 \
  --output_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/data/cache/mini \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --debug_projection_sample_rows 34 \
  --debug_projection_max_frames_per_row 30 \
  --debug_projection_max_candidates_per_frame 16 \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --overwrite

python tools/stage7_m2b_build_paired_quality_gate.py \
  --context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b \
  --output_dir outputs/stage7_m2b_lane_context_quality_v1 \
  --overwrite
```

Tier A 与 Tier A+B 的 BDD 敏感性分析使用同一份 M2A embedding：

```bash
python tools/stage6_compare_unpaired_style.py \
  --embedding_path outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v3_m2a/embedding.npy \
  --feature_path outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b/interaction_feat_style.npy \
  --feature_schema_path outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b/feature_schema.json \
  --a_indices_path outputs/stage7_m2b_lane_context_quality_v1/indices/tier_a_pdm_closed_assertive_v1.npy \
  --b_indices_path outputs/stage7_m2b_lane_context_quality_v1/indices/tier_a_pdm_closed_conservative_v1.npy \
  --output_dir outputs/stage7_m2b_lane_context_quality_v1/bdd_sensitivity/tier_a_assertive_vs_conservative \
  --num_bootstrap 50 --num_permutation 100 --min_slice_size 2 --top_k 20 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_path outputs/stage7e_pdm_v1_balanced20_paired17_embeddings_v3_m2a/embedding.npy \
  --feature_path outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b/interaction_feat_style.npy \
  --feature_schema_path outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b/feature_schema.json \
  --a_indices_path outputs/stage7_m2b_lane_context_quality_v1/indices/tier_b_inclusive_pdm_closed_assertive_v1.npy \
  --b_indices_path outputs/stage7_m2b_lane_context_quality_v1/indices/tier_b_inclusive_pdm_closed_conservative_v1.npy \
  --output_dir outputs/stage7_m2b_lane_context_quality_v1/bdd_sensitivity/tier_b_inclusive_assertive_vs_conservative \
  --num_bootstrap 50 --num_permutation 100 --min_slice_size 2 --top_k 20 \
  --overwrite

python tools/stage7_m2b_finalize_quality_analysis.py \
  --quality_dir outputs/stage7_m2b_lane_context_quality_v1 \
  --baseline_context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v3_m2a \
  --rebuilt_context_dir outputs/stage7e_pdm_v1_balanced20_paired17_context_v4_m2b \
  --full_bdd_summary outputs/stage7f_pdm_v1_balanced20_paired17_v3_m2a/report_card/stage6_pairwise/pdm_closed_assertive_v1_vs_pdm_closed_conservative_v1/bdd_summary.json \
  --output_dir outputs/stage7_m2b_final_audit_v1 \
  --overwrite
```

### 2. 期望行为

- 每个 planner row 按有效帧统计 fallback、ambiguity、bad 与 quality-eligible 比例。
- Tier A 默认要求 fallback 与 ambiguity 均不超过 `0.05` 且没有 bad frame；Tier B 上限均为 `0.20`；其他为 Tier C。
- pair tier 取同一 scenario 两个 planner 中较差的 tier，任何敏感性子集都必须同时保留或删除两个 planner row。
- 全部 17 对是论文主分析；Tier A 与 Tier A+B 仅用于敏感性分析，不能按 realized rollout quality 单独删除某个 planner row。
- 六个核心数组与 `v3_m2a` 逐字节一致时复用原 embedding，不重复编码。
- 候选级 `lane_relation_unknown` 是采样候选 lane 的关系诊断，不等同于 ego slot assignment fallback。

### 3. 通过标准

- row tiers=`A:31, B:2, C:1`，pair tiers=`A:15, B:1, C:1`；
- full / Tier A / Tier A+B pair 数为 `17 / 15 / 16`，各子集 planner 索引严格对称；
- 六个核心数组相对 `v3_m2a` 的 SHA-256 全部一致；
- full BDD：`MMD²=0.0292350, p=0.891089`；
- Tier A BDD：`MMD²=0.0365372, p=0.722772`；
- Tier A+B BDD：`MMD²=0.0326079, p=0.702970`；
- 三层结果均保持“小且不显著”的定性结论；
- final audit verdict=`PASS`，`scale_readiness=READY_TO_SCALE`。这表示数据管线可以扩容，不表示 17 对已经达到论文最终统计规模。

## Stage 7 Milestone 3：Balanced50 论文规模扩容

### 1. 命令

先冻结 50 个目标场景和 20 个仅用于技术失败替换的 reserve：

```bash
python tools/stage7_m3_select_balanced_scaleup.py \
  --inventory_csv outputs/stage7p_mini_scenario_inventory_v2/all_scenario_tags.csv \
  --seed_context outputs/stage7p_pdm_balanced_20_context_v1/stage7c_candidate_context/merged_metadata.csv \
  --prior_sim_dir outputs/stage7p_pdm_v1_balanced_20_stage7c_v1 \
  --output_dir outputs/stage7_m3_pdm_balanced50_selection_v1 \
  --overwrite
```

执行 50 scenarios × 2 planners 的 official nuPlan simulation：

```bash
bash scripts/run_stage7_m3_balanced50_simulation.sh
```

长任务运行时可查看：

```bash
python -m json.tool \
  outputs/stage7_m3_pdm_balanced50_stage7c_v1/stage7c_progress.json

tail -f outputs/stage7_m3_pdm_balanced50_stage7c_v1.run.log
```

### 2. 期望行为

- 选择集以 M2B 的 17 个成功 complete pairs 为冻结种子，再加入 33 个新候选。
- 历史技术失败 token `8b9c1329bd1855c9`、`20d049975d305f58`、
  `736373a7bc135d12` 不再进入主选择集。
- 冻结配额为 lane-change 8、following 10、stop-go/signal 10、dense
  interaction 8、lateral/turning 7、speed context 7。
- 主选择集在看到 M3 planner outcome 之前冻结；manifest SHA-256 必须为
  `a59b003ee517237d5a888e9774f939879ce812ac99d09a8f41e23c6d7e196313`。
- reserve 只能按 `reserve_rank` 替换被记录为 scenario extraction 等技术失败的
  场景，不能依据 planner 轨迹、embedding distance、BDD 或显著性选择替换对象。
- 两个 planner 必须对同一 scenario 同时进入或退出统计分析。
- simulation 脚本在运行前复核冻结 hash，防止长任务期间选择集被静默修改。

### 3. 通过标准

- selection verdict=`PASS`；
- selected scenarios=`50`，official rollout target=`100`；
- selected token 唯一，覆盖 37 个 log，每个 log 不超过 2 个场景；
- 六个 bucket 精确达到冻结配额；
- selected 与 reserve token 不重合；
- official simulation 至少得到 30 个、目标 50 个 complete planner pairs；
- 所有成功 pair 必须通过 scenario/planner/token/msgpack 严格一致性检查；
- 下游继续使用 full pairs 作为主分析，Tier A 和 Tier A+B 仅作为成对敏感性分析；
- `READY_TO_SCALE` 不预设 M3 BDD 必须显著，统计结果无论显著与否都必须报告。

完成 simulation 后的下游命令：

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7_m3_pdm_balanced50_stage7c_v1 \
  --output_dir outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/data/cache/mini \
  --map_name us-nv-las-vegas-strip \
  --write_projection_debug \
  --write_strict_filter_diagnostic \
  --write_strict_filtered_dataset \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --overwrite

python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3 \
  --device cuda \
  --overwrite

python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3 \
  --context_dataset_dir outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3 \
  --output_dir outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/report_card \
  --mode full \
  --run_stage6_pairwise \
  --overwrite

python tools/stage7_m3_final_audit.py \
  --selection_dir outputs/stage7_m3_pdm_balanced50_selection_v1 \
  --sim_dir outputs/stage7_m3_pdm_balanced50_stage7c_v1 \
  --context_dir outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3 \
  --embedding_dir outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3 \
  --stage7f_dir outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3 \
  --quality_dir outputs/stage7_m3_lane_context_quality_v1 \
  --output_dir outputs/stage7_m3_final_audit_v1 \
  --overwrite
```

M3 实际通过结果：

```text
selected scenarios: 50
complete paired scenarios: 45
successful official rollouts: 90
context: [90,150,83]
embedding: [90,64]
fallback rate: 0.00737156
ego projection success: 0.992628
Full / Tier A / Tier A+B pairs: 45 / 40 / 44
Full BDD: MMD²=0.0142209, p=0.742574
Tier A BDD: MMD²=0.0163792, p=0.683168
Tier A+B BDD: MMD²=0.0164485, p=0.673267
final verdict: PASS_WITH_LIMITATIONS
thesis scale: MINIMUM_USEFUL_SCALE_REACHED
```

必须保留的 M3 限制：

- `task_lead_brake_response` 只有 7 个 complete positive pairs，低于 10；
- following/queue paired-scenario Jaccard=`0.833`，不能当作相互独立证据；
- lead-brake 与 cut-in detector 含 proxy-dominant 语义。

## Stage 7 Milestone 4：正式统计推断与论文证据包

### 1. 命令

Full、Tier A 和 Tier A+B BDD 均使用 `1000 bootstrap + 1000 permutation`
重新计算，输出到：

```text
outputs/stage7_m4_bdd_robustness_v1/full/
outputs/stage7_m4_bdd_robustness_v1/tier_a/
outputs/stage7_m4_bdd_robustness_v1/tier_b_inclusive/
```

生成配对统计推断、Holm 校正和论文表图：

```bash
python tools/stage7_m4_build_statistical_evidence.py \
  --paired_delta_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --task_bdd_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/task_bdd_assertive_minus_conservative_v1/task_bdd_summary.csv \
  --m3_summary outputs/stage7_m3_final_audit_v1/milestone3_final_summary.json \
  --bdd_root outputs/stage7_m4_bdd_robustness_v1 \
  --output_dir outputs/stage7_m4_statistical_evidence_v1 \
  --bootstrap_repetitions 10000 \
  --seed 20260726 \
  --overwrite
```

### 2. 期望行为

- 冻结 M3 的45个 complete pairs，不重新选择场景或修改 planner 参数。
- 主统计 family 固定为 mean speed、RMS acceleration、mean THW 三项。
- 每项报告 paired mean delta、10000次 paired bootstrap CI、median、Hodges–Lehmann
  delta、paired Cohen's dz、rank-biserial、单侧 Wilcoxon 和 exact sign test。
- Wilcoxon 与 sign-test 分别在三项主端点内使用 Holm family-wise correction。
- 六个 task-conditioned BDD 作为单独 exploratory family 做 Holm correction。
- THW 没有 finite front-agent contrast 的场景不能填补，必须执行 available-case
  分析并报告缺失数。
- BDD bootstrap interval 只能解释为重采样变异，显著性以 permutation p-value 为准。

### 3. 通过标准

- paired rows=`45`；
- paired bootstrap=`10000`，固定 seed=`20260726`；
- 三组 BDD 均为 `1000 bootstrap + 1000 permutation`；
- Full/Tier A/Tier A+B 的 pair 数仍为 `45/40/44`；
- 主端点和 task family 均完成 Holm correction；
- 生成非空 CSV、Markdown、JSON 和两张论文图；
- verdict=`PASS_WITH_LIMITATIONS`；
- analysis status=`RETROSPECTIVE_FORMALIZATION_OF_M3_EXPLORATORY_RESULTS`，不能称为独立预注册确认实验。

M4 实际主端点结果：

| endpoint | n | mean delta (95% paired bootstrap CI) | paired dz | Wilcoxon Holm p | sign-test Holm p |
| --- | ---: | --- | ---: | ---: | ---: |
| mean speed | 45 | `+1.4277 [1.0106, 1.8723] m/s` | 0.948 | `1.71e-13` | `3.92e-12` |
| RMS acceleration | 45 | `+0.2562 [0.1701, 0.3416] m/s²` | 0.862 | `3.00e-7` | `7.88e-8` |
| mean THW | 35 | `-7.9987 [-31.1524, 12.3490] s` | 0.119 | `0.0177` | `0.00677` |

THW 的 median=`-2.3005 s`、Hodges–Lehmann=`-3.2037 s`，非参数方向检验通过，
但 mean CI 跨零、10对缺失且均值受极端值影响。论文中只能将其描述为
available-case robust location shift，不能写成稳定的平均THW下降。

高分辨率 BDD：

| dataset | pairs | MMD² | permutation p |
| --- | ---: | ---: | ---: |
| Full | 45 | 0.0142209 | 0.733267 |
| Tier A | 40 | 0.0163792 | 0.697303 |
| Tier A+B | 44 | 0.0164485 | 0.593407 |

六个 task BDD 的 Holm p-value 均为 `1.0`，没有 task-level distribution
significance。

## Stage 7 Milestone 5：paired-vs-marginal representation mechanism

### 1. 命令

```bash
python tools/stage7_m5_representation_mechanism_analysis.py \
  --embedding_path outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/embedding.npy \
  --interaction_feature_path outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3/interaction_feat_style.npy \
  --paired_delta_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --m4_summary outputs/stage7_m4_statistical_evidence_v1/milestone4_statistical_summary.json \
  --m4_full_bdd_summary outputs/stage7_m4_bdd_robustness_v1/full/bdd_summary.json \
  --output_dir outputs/stage7_m5_representation_mechanism_v1 \
  --probe_permutations 1000 \
  --sign_flip_repetitions 10000 \
  --mmd_permutations 1000 \
  --folds 5 \
  --seed 20260726 \
  --overwrite
```

### 2. 期望行为

- 固定使用 M4 的45个 complete pairs。
- 比较 learned embedding、33维 interaction features 和12维 trajectory summary。
- paired sign-flip 检查同场景 A-B 表示向量是否存在一致平均方向。
- grouped linear probe 使用 scenario-disjoint 5-fold GroupKFold；median imputation、
  scaling 和 classifier 都只能在每个 training fold 内拟合。
- probe null 通过每个 scenario pair 内随机交换 assertive/conservative label 构造，
  不能跨场景任意打乱。
- marginal MMD 继续回答“不使用scenario pairing时，两组边际分布是否不同”。
- 三种表示的 MMD² 因维度和kernel bandwidth不同，不能直接按数值大小排名；
  主要比较各自 permutation p-value 和 paired/probe 结果。
- paired MDE 是给定 n/alpha/power 的设计敏感度，不是 observed-effect post-hoc power，
  也不是 MMD 样本量保证。

### 3. 通过标准

- 三种表示覆盖相同45个 pairs；
- sign-flip=`10000`，probe pair-swap permutation=`1000`，MMD permutation=`1000`；
- grouped probe 每个测试scenario均不进入对应training fold；
- 所有probe预处理在training fold内拟合；
- learned embedding 同时报告 paired、grouped probe 和 marginal BDD；
- 生成representation table、distance-sensitivity table、MDE table、JSON、报告和图；
- verdict=`PASS_WITH_LIMITATIONS`；
- analysis status=`EXPLANATORY_POST_M4_MECHANISM_ANALYSIS`。

M5 实际结果：

| representation | paired concentration | sign-flip p | grouped ROC-AUC | pair-swap p | marginal MMD p |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned embedding | 0.326 | 0.000100 | 0.638 | 0.00699 | 0.733 |
| interaction features | 0.429 | 0.000100 | 0.773 | 0.000999 | 0.126 |
| trajectory summary | 0.464 | 0.000400 | 0.704 | 0.000999 | 0.123 |

Embedding distance 与 `|delta mean speed|` 的 Spearman rho=`0.454`，
Holm p=`0.00518`；与 acceleration/THW contrast 的相关性未通过 Holm 校正。

45对的 paired-t design sensitivity：

```text
one-sided alpha=0.05, power=0.80: minimum detectable dz=0.376
conservative alpha=0.05/3, power=0.80: minimum detectable dz=0.454
```

解释：三种表示都包含系统性的同场景 planner shift，且能够在scenario-disjoint
probe中提供planner信息；但三者的边际MMD均未显著。主要机制是paired analysis
控制了场景异质性，而marginal BDD丢弃了这种配对结构。不能把MMD不显著等同于
embedding完全没有行为信息。

## Stage 7 Milestone 6.1：paired BDD 方法冻结与质量审计

### 1. 命令

```bash
python tools/stage7_m6_scenario_conditioned_bdd.py \
  --embedding_path outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/embedding.npy \
  --embedding_manifest outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/embedding_manifest.json \
  --metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --paired_delta_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --marginal_bdd_summary outputs/stage7_m4_bdd_robustness_v1/full/bdd_summary.json \
  --row_quality_csv outputs/stage7_m3_lane_context_quality_v1/row_quality_tiers.csv \
  --pair_quality_csv outputs/stage7_m3_lane_context_quality_v1/paired_quality_gate.csv \
  --output_dir outputs/stage7_m6_1_paired_bdd_method_freeze_v1 \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --permutations 100000 \
  --seed 20260726 \
  --overwrite
```

### 2. 期望行为

- 严格检查 embedding、metadata、scenario token/index、planner、global row 和
  valid horizon 一致性；
- `paired_delta_csv` 必须覆盖所有 embedding rows，scenario 不得重复；
- primary 使用原始64维 embedding、single-RBF biased V-statistic MMD²、精确
  pooled positive off-diagonal median bandwidth 和 within-pair label swap；
- 报告 exceedance count、plus-one p-value 和 Monte Carlo resolution；
- residual BDD 为 secondary；M4 marginal BDD 与 fixed-kernel pooled shuffle
  作为历史参考和 control；
- 对 full、Tier A、Tier A+B 运行预定义质量敏感性分析，并检查 embedding pair
  distance 是否由 fallback/ambiguous rate 驱动；
- 写入输入、checkpoint、脚本 SHA256、git/runtime provenance 与 frozen spec；
- 不修改 Stage6 unpaired BDD，不修改 Stage5D checkpoint；
- 当前45对只能标记为方法开发集，不得标记为独立确认集。

### 3. 通过标准

- complete pairs=`45`，duplicate/missing/conflict/unequal-horizon/non-finite 均为0；
- original-space paired-label-swap BDD `p <= 0.01`；
- Tier A 与 Tier A+B primary sensitivity 的 Holm-adjusted `p <= 0.05`；
- fallback-distance correlations 在 Holm correction 后不显著；
- frozen marginal BDD 仍在结果中且未被覆盖；
- `mmd_magnitudes_across_spaces_not_ranked=true`；
- verdict=`PASS_WITH_LIMITATIONS`；
- analysis status=`DEVELOPMENT_SET_METHOD_FREEZE`；
- `method_freeze_ready_for_new_locked_set=true`。

M6.1 实际结果（100000 permutations）：

| analysis | role | MMD² | exceedance | plus-one p |
| --- | --- | ---: | ---: | ---: |
| frozen M4 marginal BDD | historical reference | 0.0142209 | — | 0.733267 |
| fixed-kernel pooled shuffle | control | 0.0141802 | 74086/100000 | 0.740863 |
| original-space paired-label-swap | primary | 0.0141802 | 175/100000 | 0.001760 |
| pair-midpoint residual | secondary | 0.0994187 | 0/100000 | ≤0.000010 |

Tier A 有40对，primary p=`0.000440`；Tier A+B 有44对，primary
p=`0.000080`，两个子集经 Holm 校正后仍显著。四个 fallback/ambiguous-rate
相关性经 Holm 校正均不显著。完整审计、frozen spec 和 provenance 位于
`outputs/stage7_m6_1_paired_bdd_method_freeze_v1/`。

注意：M4 与 M6 pooled statistic 的轻微差异来自历史 multi-kernel 与本次冻结
single-RBF 估计器配置差异。下一步是新 log/scenario-disjoint、selection config
独立冻结且 planner treatment 参数不变的锁定确认，
不是默认重训练。异源实路日志仍使用 Stage6 unpaired-first 协议。

## Stage 7 Milestone 6.2：锁定确认入口与 task-conditioned paired BDD

### 1. 命令

```bash
python tools/stage7_m6_2_locked_task_bdd.py \
  --metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --paired_delta_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --development_metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --m6_frozen_spec outputs/stage7_m6_1_paired_bdd_method_freeze_v1/m6_frozen_analysis_spec.json \
  --representation learned_embedding=outputs/stage7_m5_representation_mechanism_v1/representations/learned_embedding.npy \
  --representation interaction_features=outputs/stage7_m5_representation_mechanism_v1/representations/interaction_features.npy \
  --representation trajectory_summary=outputs/stage7_m5_representation_mechanism_v1/representations/trajectory_summary.npy \
  --output_dir outputs/stage7_m6_2_locked_task_bdd_development_v1 \
  --analysis_role development_validation \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --minimum_overall_pairs 80 \
  --minimum_task_pairs 12 \
  --task_monte_carlo_permutations 100000 \
  --seed 20260729 \
  --overwrite
```

未来新数据运行时改为 `--analysis_role locked_confirmation`，并额外提供
`--lock_manifest` 与 `--power_justification_file`。

### 2. 期望行为

- 用仿真前已知的 `scenario_type` 定义五个 task；
- 小于等于20对时枚举全部 `2^n` assignments，超过20对时运行100000次 swaps；
- learned embedding task family 使用 Holm correction；
- handcrafted representations 仅作为机制对照；
- 开发模式生成锁定规范和 power-justification 模板；
- 确认模式强制开发集与新数据 log/scenario 零重叠，并强制 planner 参数一致；
- 不训练或修改 Stage5D checkpoint。

### 3. 通过标准

- 45个开发 pairs 通过原 M6 alignment；
- task selection timing=`pre_treatment`；
- dataset role=`METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY`；
- 生成锁定规范和 power 模板；
- 未来确认集 log/scenario overlap 均为0、planner fingerprints 完全相同；
- 不在解盲后修改 task mapping、估计器或样本量。

开发集实际结果：五个 task 各8–9对，均低于12对运行下限。只有 high motion
dynamics 的 learned embedding 通过 Holm correction（exact p=`0.00390625`，
Holm p=`0.01953125`）；其余任务不显著，不能视为独立确认。

## Stage 7 Milestone 6.3：simulation-based power planning

### 1. 命令

```bash
python tools/stage7_m6_3_simulation_power_analysis.py \
  --embedding_path outputs/stage7_m5_representation_mechanism_v1/representations/learned_embedding.npy \
  --metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --paired_delta_csv outputs/stage7f_pdm_v1_balanced50_paired45_v1_m3/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --m6_2_lock_spec outputs/stage7_m6_2_locked_task_bdd_development_v1/m6_2_locked_confirmation_spec.json \
  --output_dir outputs/stage7_m6_3_simulation_power_v1 \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --candidate_pairs 12,20,30,45,60,80,120 \
  --effect_scales 0.5,0.75,1.0 \
  --target_effect_scale 0.75 \
  --target_power 0.80 \
  --attrition_rate 0.20 \
  --simulations 500 \
  --planning_permutations 999 \
  --blas_threads 1 \
  --alpha 0.05 \
  --seed 20260730 \
  --overwrite
```

### 2. 期望行为

- 以45对开发集为 pilot，对中心化 pair midpoint 和 pair-difference residual
  分别 bootstrap，再按冻结 effect scale 注入均值差；
- overall 使用冻结的 paired single-RBF biased MMD，五个 task 使用同一统计量并
  做 Holm correction；
- 报告每个网格点的 Monte Carlo power 和 Wilson 95% CI；
- 同时满足五个 task 的最小样本数决定 task quota，而不是逐 task 事后选择；
- 生成与 M6.2 lock SHA256 绑定的机器可读 power justification 和采集配额；
- M6.2 locked mode 对 overall/task 数量、task mapping、lock hash 和 planner
  fingerprints fail closed；
- 不训练或修改 Stage5D checkpoint。

### 3. 通过标准

- power justification status=`FROZEN_BEFORE_LOCKED_CONFIRMATION`；
- target effect scale=`0.75`、target power=`0.80`；
- 五个 task 的 simultaneous Holm-corrected power 不低于0.80；
- 每 task complete quota=`60`，20%损耗后 gross quota=`75`；
- overall complete pairs 不低于 M6.2 运行下限80；
- 新确认集仍须与开发集 log/scenario 零重叠，且 planner treatment 指纹不变；
- 不能把模拟功效写成 achieved power 或确认性结果。

实际主设计：每任务60个完整 pairs 时，五任务 simultaneous power=`0.918`，
Wilson 95% CI=`[0.891,0.939]`；按20%损耗率为每任务75个 gross pairs、总计
375个 gross pairs。Overall 的纯功效选择为45对，但由冻结运行/质量下限提升为
至少80个完整 pairs。

保守敏感性分析位于
`outputs/stage7_m6_3_half_effect_extension_v1/`：若锁定域真实均值差只有开发
pilot 的50%，需要每任务160个完整 pairs；其 simultaneous power=`0.936`
（95% CI `[0.911,0.954]`），对应每任务200个 gross pairs、总计1000个。
该结果是预算敏感性上界，不覆盖0.75主冻结设计。

## Stage 7 Milestone 6.4：锁定采集候选池预检

### 1. 命令

```bash
python tools/stage7_m6_4_freeze_locked_collection.py \
  --inventory_csv outputs/stage7p_mini_scenario_inventory_v2/all_scenario_tags.csv \
  --development_metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --m6_2_lock_spec outputs/stage7_m6_2_locked_task_bdd_development_v1/m6_2_locked_confirmation_spec.json \
  --power_justification_file outputs/stage7_m6_3_simulation_power_v1/m6_3_locked_power_justification.json \
  --nuplan_db_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/data/cache/mini \
  --output_dir outputs/stage7_m6_4_locked_collection_preflight_v1 \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --max_per_log 2 \
  --reserve_per_task 15 \
  --selection_salt stage7-m6.4-locked-v1 \
  --overwrite
```

### 2. 期望行为

- 只读取 rollout 前已经存在的 nuPlan `scenario_type` 和 DB/log/token 元数据；
- 严格排除开发集45个 scenario token 及其全部34个 log；
- 排除同时命中多个冻结 scenario type 的歧义 token 和缺失 DB 文件；
- 检查 M6.2 lock、M6.3 power justification、当前 M6.2/M6.3 脚本、开发
  metadata 及 planner treatment fingerprints 的 SHA256 链；
- 按固定 salt 稳定排序，限制 primary+reserve 每 log 最多2个场景；
- 准备每任务75个 primary gross scenarios 和15个 task-specific reserve；
- 只有全部配额满足时才生成 `m6_4_locked_collection_manifest.json` 和 Stage7C
  context；否则只输出容量审计并返回非零，不启动仿真；
- 不读取新 planner outcome、embedding 或 BDD，不重新训练模型。

### 3. 通过标准

- status=`FROZEN_BEFORE_LOCKED_ROLLOUTS`；
- 五任务 primary 均为75，reserve 均为15；
- development token/log overlap 均为0；
- candidate、primary、reserve token 唯一且互不重叠；
- planner fingerprints 与 M6.2 lock 完全相同；
- `--max_per_log 2` 下 primary 至少来自188个未使用 log；
- locked manifest 在任何新 planner outcome 产生之前冻结。

当前 mini inventory 实际结果为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，因此没有生成 locked manifest，也
没有启动 rollout。mini 共有63个 log，开发集使用34个；排除开发 log 后只有29个
eligible logs，而 primary 在每 log 最多2个的约束下至少需要188个。更关键的是，
冻结 lane-change 类型只有2个 eligible candidates，距离75个 primary 和15个 reserve
明显不足。必须扩展并重新索引新的 nuPlan log DB；不得放宽任务定义或复用开发 log
来消除该缺口。

## Stage 7 Milestone 6.4A：多 DB pre-treatment inventory 构建

### 1. 命令

仅使用现有 mini DB 重建可复现 inventory：

```bash
export NUPLAN_DATA_ROOT=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset

python tools/stage7p_build_scenario_inventory.py \
  --db_root "$NUPLAN_DATA_ROOT/data/cache/mini" \
  --flat_db_root "$NUPLAN_DATA_ROOT/data/cache/locked_pool_v1" \
  --output_dir outputs/stage7p_expanded_scenario_inventory_v1 \
  --overwrite
```

Pittsburgh 等扩展 DB 解压到独立目录后，重复传入 `--db_root`：

```bash
python tools/stage7p_build_scenario_inventory.py \
  --db_root "$NUPLAN_DATA_ROOT/data/cache/mini" \
  --db_root "$NUPLAN_DATA_ROOT/data/cache/train_pittsburgh" \
  --flat_db_root "$NUPLAN_DATA_ROOT/data/cache/locked_pool_v1" \
  --output_dir outputs/stage7p_expanded_scenario_inventory_v1 \
  --overwrite
```

测试：

```bash
python -m py_compile tools/stage7p_build_scenario_inventory.py
python tools/check_no_tmp_dependencies.py
python -m pytest -q \
  tests/test_stage7p_build_scenario_inventory.py \
  tests/test_stage7_m6_4_freeze_locked_collection.py
```

### 2. 期望行为

- 每个 `--db_root` 只扫描直接子目录中的 `*.db`，多 root 输入按稳定顺序处理；
- SQLite 以只读模式打开，读取 `scenario_tag -> lidar_pc -> scene -> log`；
- BLOB token 统一写成 lowercase hex；`scenario_token` 和兼容 `scene_token` 都使用
  `scenario_tag.lidar_pc_token`，原始 scene token 写入 `db_scene_token`；
- 同一 token 的多个 scenario types 保留为多行，由 M6.4 排除跨冻结 task 的歧义；
- 使用临时 SQLite staging 流式去重，避免把完整 inventory 常驻内存；
- DB basename 冲突、token 指向多个 log/DB、缺表、缺列、空 token 和断裂外键均
  fail closed；
- `--flat_db_root` 只创建相对符号链接，已有正确链接幂等复用，不覆盖普通文件或
  指向错误目标的链接；
- 生成 `all_scenario_tags.csv`、`scenario_inventory_inputs.csv`、
  `scenario_inventory_summary.json` 和 `scenario_inventory_report.md`；
- 输入清单记录 DB 路径、大小、mtime、SHA-256、原始 tag rows、去重后 rows 和 log；
- 工具只读取 pre-treatment SQLite metadata，不读取 planner outcome、trajectory、
  embedding 或 BDD，不启动 M6.4 preflight 和 rollout。

### 3. 通过标准

- 输出 CSV 列严格为 `db_file,log_name,scenario_token,scene_token,db_scene_token,scenario_type,scenario_tag_token`；
- summary 中 `schema_version=stage7p_scenario_inventory_v1`、
  `status=COMPLETE_PRETREATMENT_INVENTORY`、`outcome_blind=true`；
- mini 基准为64个 DB、约892204个 scenario-tag rows、63个 logs；
- 当前 Mac smoke 实际读取892204个原始 tag rows，按 token/type/log/DB 去重后输出
  821831行，移除70373个重复 tag，unique tokens=390186，冲突=0；
- flat DB pool 中每个 inventory `db_file` 均可在单层 root 下解析；
- fixture 和既有 M6.4 测试通过；
- mini-only inventory 重跑 M6.4 时仍应返回
  `BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，不得生成 locked manifest；
- 只有扩展 DB 后 M6.4 status 变为 `FROZEN_BEFORE_LOCKED_ROLLOUTS`，才可进入
  M6.4B official rollouts。

当前重建验证中，新旧 `m6_4_task_capacity.csv` 逐字节一致；冻结类型的 unique
tokens=`177313`、eligible candidates=`70995`、eligible logs=`29`，五个 task 的
capacity 均未改变。这说明 tag-level 去重没有改变 M6.4 的候选 estimand。

### 4. Pittsburgh expanded inventory 实际结果（2026-08-07）

下载与解压验收：

```text
ZIP bytes: 30620248893
ZIP entries: 1562
DB entries / extracted DB files: 1560 / 1560
uncompressed DB bytes: 55726387200 (51.90 GiB)
unsafe archive paths: 0
unzip CRC test: pass
invalid SQLite headers: 0
```

Pittsburgh 与 mini 有3个同名 DB，三组文件大小和 SHA-256 均完全一致。为保持
basename-conflict fail-closed 合同，expanded 输入使用 Pittsburgh 中的3个副本，
并建立 `mini_non_pittsburgh_v1` 相对链接 root，包含其余61个 mini DB。

expanded builder 命令：

```bash
python tools/stage7p_build_scenario_inventory.py \
  --db_root ../nuplan/dataset/data/cache/mini_non_pittsburgh_v1 \
  --db_root ../nuplan/dataset/data/cache/train_pittsburgh \
  --flat_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --output_dir outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh \
  --overwrite
```

实际 inventory：

```text
DB files: 1621
logs: 1576
source scenario_tag rows: 9695626
inventory rows: 9604184
unique scenario tokens: 5386575
duplicate rows removed: 91442
token-location conflicts: 0
inventory SHA-256: 3fc6c02647d4df48362e4d124f8b01443d904ad6f491d8a68ddc0871caa2f5ab
```

expanded M6.4 preflight：

```bash
python tools/stage7_m6_4_freeze_locked_collection.py \
  --inventory_csv outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh/all_scenario_tags.csv \
  --development_metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --m6_2_lock_spec outputs/stage7_m6_2_locked_task_bdd_development_v1/m6_2_locked_confirmation_spec.json \
  --power_justification_file outputs/stage7_m6_3_simulation_power_v1/m6_3_locked_power_justification.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --output_dir outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh \
  --planner_a pdm_closed_assertive_v1 \
  --planner_b pdm_closed_conservative_v1 \
  --max_per_log 2 \
  --reserve_per_task 15 \
  --selection_salt stage7-m6.4-locked-v1 \
  --overwrite
```

通过结果：

```text
status: FROZEN_BEFORE_LOCKED_ROLLOUTS
ready_to_launch_locked_rollouts: true
primary: 75 × 5 tasks = 375 scenarios / 750 rollouts
reserve: 15 × 5 tasks = 75 scenarios
primary distinct logs: 306
primary + reserve distinct logs: 350
max scenarios per log: 2
development token overlap: 0
development log overlap: 0
missing DB files: 0
Stage7C primary context rows: 375
primary manifest SHA-256: c825d87826b951bcdd6ed987195aeea25b02290eacca7cc6a2fc2b9e91ba8839
reserve manifest SHA-256: c6c148d6298a0c6b8cdccd083f363cded1335f41845ba802148967e3f5328904
```

该结果只冻结 M6.4B 的输入和执行顺序，不代表750个 rollouts 已经运行。进入仿真前
仍须核验 Mac `nuplan` 环境、tuPlan Garage commit、PDM readiness、地图变量和单场景
smoke；不得直接把全部750个任务投入未经验证的 Apple Silicon 环境。

## Stage 7 Milestone 6.4B：Mac PDM readiness 与首个 locked smoke

### 1. 已核验环境

```text
nuPlan devkit: e9241677997dd86bfc0bcd44817ab04fe631405b
tuPlan Garage: b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
Python: /Users/liuqing/miniconda3/envs/nuplan/bin/python (3.9.19)
PDM readiness: true / ready_for_pdm_smoke
```

Readiness 与参数报告：

- `outputs/stage7p_pdm_readiness_check_v2_mac/`
- `outputs/stage7p_pdm_closed_config_params_v2_mac/`

Mac 的 protobuf C extension 与旧 tensorboard 组合需要在 official command 环境中设置：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 2. 首个 locked smoke 结果

输入为
`outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/stage7c_primary_context`
的第一行，目标 token=`6b5a9da8c0b353b9`，只运行两个冻结 planner。输出：

```text
outputs/stage7_m6_4b_locked_smoke_1scene_mac_v1
PASS
official commands = 2 / 2
shape = (1, 2, 149, 8)
valid timesteps = 298
missing pairs = 0
same-log alignment = true
strict token alignment = true
pseudo rollout = false
```

实际命令模板必须同时满足：

- `{planner_hydra_overrides}` 保留冻结 assertive / conservative 参数；
- `scenario_builder.db_files=[<exact locked DB>]`，不扫描无关 DB；
- `scenario_filter=all_scenarios {scenario_hydra_overrides}`，先清除默认
  `one_continuous_log` 的固定 log，再注入 locked token；
- `worker.max_workers=1`、`scenario_builder.max_workers=1` 和 BLAS threads=1；
- 使用绝对 output path；批量运行时令 `job_name` 包含
  `closed_loop_nonreactive_agents`，避免 metric aggregator 的 challenge-name 警告；
- 保留 `--require_same_scenario_alignment`、
  `--require_strict_nuplan_token_alignment` 和 `--allow_external_planner_name`。

首个 token 同时具有 `near_long_vehicle` 与非冻结 `stationary` DB tags，serializer
目录显示 `stationary`；严格 token/log 对齐仍为 PASS，M6.4 task assignment 保持
outcome 前冻结的 `near_long_vehicle`，不得根据 smoke 结果改写。

本 smoke 只完成2/750个 primary rollouts。下一步先建立可审计的批处理、进度、断点
续跑、失败分类和 reserve 消耗流程，再启动剩余748个；禁止按中途 effect size
停止或修改 manifest / planner 参数。

## Stage 7 Milestone 6.4B：locked primary 批处理与断点续跑

### 1. 命令

先执行 dry-run。该命令验证完整375行 primary、75行 reserve 和450个 DB，但
`--max_scenarios 1` 只把 order 1 标记为本次候选，不运行仿真：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation

python tools/stage7_m6_4b_run_locked_rollouts.py \
  --manifest_path outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_collection_manifest.json \
  --primary_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage7_m6_4b_locked_batch_mac_v2 \
  --max_scenarios 1
```

真实执行 order 1，必须显式提供执行开关、primary canonical manifest hash 和
resume；以下命令可直接运行：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage7_m6_4b_run_locked_rollouts.py \
  --manifest_path outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_collection_manifest.json \
  --primary_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage7_m6_4b_locked_batch_mac_v2 \
  --max_scenarios 1 \
  --execute \
  --confirm_primary_manifest_sha256 c825d87826b951bcdd6ed987195aeea25b02290eacca7cc6a2fc2b9e91ba8839 \
  --resume
```

后续分段执行可使用 `--start_order`、`--end_order` 或 `--max_scenarios`。完全相同的
命令配合 `--resume` 会重新审计成功输出后跳过；不要删除 batch manifest 或 status
文件。只有人工检查失败原因并决定重试时，才增加 `--retry_failed`；它会创建新的
`attempt_NNN`，不会覆盖旧 attempt。

测试：

```bash
python -m py_compile tools/stage7_m6_4b_run_locked_rollouts.py
python tools/check_no_tmp_dependencies.py
python -m pytest -q tests/test_stage7_m6_4b_run_locked_rollouts.py
```

### 2. 期望行为

- 每次启动都重新核验 manifest、CSV/Stage7C hash、planner fingerprints、顺序、
  task rank、selection salt、DB 文件及两个外部 commit；
- 默认 dry-run，未显式提供 `--execute` 时不产生 rollout；
- 每个 scenario 生成独立 one-row context，并运行完整 assertive/conservative pair；
- official command 固定单 worker、BLAS threads=1、精确 DB 和 token、绝对 Stage7C
  output path；
- `batch_manifest.json` 额外冻结 batch tool SHA-256、command timeout 和执行环境，
  `batch_state.json` 原子更新，
  `batch_events.jsonl` 追加记录 attempt；
- `--resume` 只有在 pair completeness、trajectory、same-log、strict-token 等全部
  复核通过时才跳过；损坏/失败输出不会静默覆盖；
- `reserve_replacement_proposal.csv` 只生成技术/质量失败的 task-rank 顺序提案，
  所有行均为 `PROPOSED_NOT_APPROVED_NOT_EXECUTED`；工具不会运行 reserve；
- 不读取 embedding、BDD、effect size，不按观察到的 planner behavior 停止或换样本。

### 3. 通过标准

- full dry-run 显示375 primary、750 planned rollouts、374/375等状态与实际输出一致；
- frozen input audit 中三个 SHA-256、两个 planner fingerprints 和两个 commits 与
  locked 值完全一致；
- order 1 真实 smoke 为2/2 official successes、298 trajectory rows、strict alignment
  PASS；
- 原样 resume 输出 `SKIP`，event ledger 行数不变且没有 `attempt_002`；
- 当前 `batch_scenario_status.csv` 为1个 `SUCCEEDED`、374个 `PENDING`、0个失败；
- `batch.lock` 在正常退出后不存在；reserve proposal为空；
- order 2–375 未启动，剩余748个 rollout 保持 pending。

当前真实结果目录：`outputs/stage7_m6_4b_locked_batch_mac_v2/`。冻结 batch tool
SHA-256 为 `ef0026b3cc20942846035ac23d0d16d616a3d7dd6675e9a0f9c2612871d7fb06`。
nuPlan metric aggregator 仍会打印“no metric files found for aggregation”警告，但 per-scenario
metrics、runner report、msgpack 和轨迹导出均存在；该警告不改变 batch PASS。

### 4. Order 2–6 canary 实际耗时与全量估算

```text
order 1 following_interaction:          36.26 s
order 2 lane_change:                    32.34 s
order 3 stop_go_control:                38.39 s
order 4 high_motion_dynamics:           30.70 s
order 5 dense_or_vulnerable_interaction:34.13 s
order 6 following_interaction:          41.05 s

mean / median / sample SD: 35.48 / 35.20 / 3.86 s
order 2–6 actual wall time: 176.64 s
effective wall rate: 35.33 s/scenario
```

按实际连续 wall rate 估算：原始374个 pending场景约3小时40分；canary后剩余369个
约3小时37分。按当前最快/最慢场景外推，374个约3小时11分至4小时16分；正式全量
建议预留4.5–5小时。平均磁盘占用约13.66 MiB/scenario，剩余369个约需4.92 GiB。

当前 batch 状态为6 `SUCCEEDED`、369 `PENDING`、0 failures、0 reserve proposals。
全量运行前应使用 `caffeinate` 防止Mac休眠，并继续使用同一个 v2 output、相同 batch
tool hash 和 `--resume`；不要新建 manifest 或改变 timeout。

## Stage 7 Milestone 6.4C：locked technical audit 与恢复

M6.4B 全量375场景完成后，先运行 outcome-blind 技术审计。审计不会执行 rollout，
也不会读取 embedding、BDD、effect size、trajectory metric 或 planner outcome：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage7_m6_4c_audit_locked_recovery.py \
  --locked_manifest outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_collection_manifest.json \
  --primary_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv \
  --batch_status_csv outputs/stage7_m6_4b_locked_batch_mac_v2/batch_scenario_status.csv \
  --batch_manifest outputs/stage7_m6_4b_locked_batch_mac_v2/batch_manifest.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --output_dir outputs/stage7_m6_4c_locked_recovery_audit_v2
```

审计结果：283个 primary 已成功、90个落在 nuPlan 官方 scene position 边界外、2个
有效 token 需要 Hydra 字符串引号；58/75 reserve 技术可运行。`recovery_plan.csv`
仅提出22个冻结动作：2个 quoted primary retry、10个 lane-change reserve 和10个
high-motion reserve。输出目录必须不存在，工具拒绝覆盖旧审计。

恢复 runner 默认 dry-run；真实执行必须复述 `recovery_plan.csv` 的 SHA-256，并选择
单一 action。除 `--action` 与 `--output_dir` 外，两次执行使用相同冻结参数：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage7_m6_4c_run_locked_recovery.py \
  --audit_summary outputs/stage7_m6_4c_locked_recovery_audit_v2/m6_4c_recovery_audit_summary.json \
  --recovery_plan outputs/stage7_m6_4c_locked_recovery_audit_v2/recovery_plan.csv \
  --locked_manifest outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_collection_manifest.json \
  --primary_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv \
  --batch_status_csv outputs/stage7_m6_4b_locked_batch_mac_v2/batch_scenario_status.csv \
  --batch_manifest outputs/stage7_m6_4b_locked_batch_mac_v2/batch_manifest.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  --action RETRY_PRIMARY_QUOTED_TOKEN \
  --output_dir outputs/stage7_m6_4c_quoted_primary_recovery_mac_v1 \
  --execute \
  --confirm_recovery_plan_sha256 370da6919905cdacce616639cfc47407081120a7eacae8fe859fde7d3553d7cb
```

执行 reserve 时改为 `--action RUN_FROZEN_RESERVE` 和新 output directory。Runner
复核审计输入哈希、冻结 Stage7C/batch tool、planner fingerprints、两个外部 commits、
runtime 路径与 timeout；每场仍要求2/2 official success、完整 trajectory pair、
same-log 和 strict-token alignment。quoted retry 使用转义双引号，让引号在 Stage7C
的 `shlex.split` 后仍进入 Hydra argv，同时保留原始 `scenario_token` 用于身份校验。

真实结果：2/2 quoted retry 成功；20/20 frozen reserve 成功。恢复后完整 pairs 为：

```text
following_interaction:             60 / 60
lane_change:                       60 / 60
stop_go_control:                   67 / 60
high_motion_dynamics:              55 / 60
dense_or_vulnerable_interaction:   63 / 60
overall:                          305
```

high-motion 仍缺5对且冻结 reserve 已用完。禁止直接从集合外选择5条；必须先新增
outcome-blind supplemental protocol amendment，明确候选池、去重/零重叠规则、
固定 salt、追加配额及新 manifest/hash，再启动补充 rollout。

测试：

```bash
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
python -m pytest -q tests/test_stage7_m6_4c_audit_locked_recovery.py \
  tests/test_stage7_m6_4c_run_locked_recovery.py
```

## Stage 7 Milestone 6.4D：high-motion outcome-blind supplement

### 1. 命令

先冻结补充集合：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage7_m6_4d_freeze_high_motion_supplement.py \
  --locked_manifest outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_collection_manifest.json \
  --eligible_inventory outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_eligible_candidate_inventory.csv \
  --development_metadata_csv outputs/stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv \
  --primary_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4_locked_collection_preflight_v3_pittsburgh/m6_4_locked_reserve_collection.csv \
  --batch_status_csv outputs/stage7_m6_4b_locked_batch_mac_v2/batch_scenario_status.csv \
  --batch_manifest outputs/stage7_m6_4b_locked_batch_mac_v2/batch_manifest.json \
  --m6_4c_audit_summary outputs/stage7_m6_4c_locked_recovery_audit_v2/m6_4c_recovery_audit_summary.json \
  --quoted_recovery_state outputs/stage7_m6_4c_quoted_primary_recovery_mac_v1/recovery_state.json \
  --reserve_recovery_state outputs/stage7_m6_4c_frozen_reserve_recovery_mac_v1/recovery_state.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --output_dir outputs/stage7_m6_4d_high_motion_supplement_freeze_v1
```

Runner 默认 dry-run；真实执行命令为：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage7_m6_4d_run_locked_supplement.py \
  --supplement_manifest outputs/stage7_m6_4d_high_motion_supplement_freeze_v1/m6_4d_locked_supplement_manifest.json \
  --primary_csv outputs/stage7_m6_4d_high_motion_supplement_freeze_v1/m6_4d_locked_primary_collection.csv \
  --reserve_csv outputs/stage7_m6_4d_high_motion_supplement_freeze_v1/m6_4d_locked_reserve_collection.csv \
  --batch_manifest outputs/stage7_m6_4b_locked_batch_mac_v2/batch_manifest.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp \
  --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  --output_dir outputs/stage7_m6_4d_high_motion_supplement_primary_mac_v1 \
  --execute \
  --confirm_supplement_manifest_sha256 3dc11ab70c71479191bb4c789782e5ebe78dd7e43efdaec55651451b99041c2f \
  --confirm_source_manifest_sha256 e63634711345e590de8db038c44a0fbe890700cd197e4de01156f338481113bb
```

### 2. 期望行为

- Freeze 工具只读取 pre-treatment inventory、development/original collection identity、
  SQLite 技术结构和 M6.4B/M6.4C 技术状态；不读取 embedding、BDD、effect size、
  trajectory metric 或 planner outcome；
- 排除 development 和原450条集合的全部 token/log，补充集合内部每 log 最多1条；
- 使用固定 salt `stage7-m6.4d-high-motion-supplement-v1` 冻结5 primary + 5 reserve；
- Runner 默认不执行；真实执行前复核所有 hashes、commits、planner fingerprints、
  runtime path、timeout 和 SQLite technical runnability；
- Primary 必须全部按冻结顺序执行。只有 primary 有 documented technical failure 时，
  才允许 `--source reserve --primary_run_state ...`；否则 runner 拒绝 reserve；
- 每场运行完整 assertive/conservative pair，禁止按 effect size 中途停止。

### 3. 通过标准

- supplement 与 development/original collection 的 token/log overlap 均为0；
- 5 primary + 5 reserve 均通过 official scene-position preflight；
- dry-run 选择5条但不生成 rollout；
- 真实 primary 为5 `SUCCEEDED`、0 failures，2/2 official success、trajectory pair、
  same-log 和 strict-token alignment 全部通过；
- high-motion 完整 pairs 从55提升到60，五任务均达到冻结配额；
- reserve 未执行；Stage7C 和 M6.4B tool hashes 保持不变。

测试：

```bash
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
python -m pytest -q tests/test_stage7_m6_4d_freeze_high_motion_supplement.py \
  tests/test_stage7_m6_4d_run_locked_supplement.py
```

## Stage 7 Milestone 6.5：310-pair locked confirmation

### 1. 准备并冻结

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_5_prepare_locked_confirmation.py \
  --output_dir outputs/stage7_m6_5_locked_confirmation_view_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_5_run_locked_confirmation.py freeze \
  --output_dir outputs/stage7_m6_5_locked_analysis_freeze_v1
```

确认 view 固定283个 M6.4B primary successes、2个 quoted-primary recoveries、20个
frozen reserves 和5个 M6.4D supplement，合计310 pairs。Freeze 必须发生在确认
embedding/effect 被读取之前。

### 2. Mac context 注意事项

```bash
env PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage \
  caffeinate -dimsu \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7_m6_5_locked_confirmation_view_v1 \
  --output_dir outputs/stage7_m6_5_locked_confirmation_context_v1 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --write_projection_debug --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6
```

缺少 `PYTHONPATH` 会令 nuPlan pickle 因找不到 `tuplan_garage` 而退化为空 history；
不能只看83D shape，必须同时检查 neighbor slot coverage 非零。正确运行耗时23分56秒，
输出 `[620,150,83]`。

### 3. 锁定结果

- overall original 64D primary：MMD²=`0.0044693963`，0/100000 exceedances，
  plus-one p=`9.9999e-6`；
- five-task learned-embedding Holm p：following `0.00030`、lane `0.00036`、
  stop/go `0.01820`、high-motion `0.00042`、dense/vulnerable `0.00258`；
- Tier A=58、Tier A+B=135，original sensitivities Holm p 均为`0.0182`；
- Tier A residual p=`0.126249`，不显著；
- global fallback=`10.59%`，max-pair fallback 与 embedding distance 的 rho=`0.5088`。

解释边界：确认支持 planner-conditioned behavior distribution difference；不代表安全性、
planner superiority 或完全不受 lane-context quality 影响的纯 planner mechanism。

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage7_m6_5_locked_confirmation.py
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
```

## Stage 6P：Representation × Unpaired Release（Issue #257）

### 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6p_run_representation_unpaired_release.py \
  --config configs/stage6p_representation_unpaired_release.json \
  --embedding_pool outputs/stage6h_expanded_800_embedding_pool_v1 \
  --assignments outputs/stage6h_nuplan_power_curve_800_v1/power_curve_log_assignments.csv \
  --context_existing outputs/stage7_m6_5_locked_confirmation_context_v1 \
  --context_expanded outputs/stage6h_expanded_490_context_v1 \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --scaler outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired/scalers/handcrafted_reference_scalers.npz \
  --output_dir outputs/stage6p_representation_unpaired_release_v1 \
  --device cpu
```

### 期望行为

- 不调用nuPlan simulation、不训练checkpoint，原样复用800 pair、489 log、2400 release split；
- 四种representation逐trial使用完全相同的日志和场景；
- 每种representation×n独立计算bandwidth与A/A q95，只在匹配阈值下比较FPR/detection；
- 禁止跨representation比较raw MMD²；neighbor-zero64只作diagnostic。

### 通过标准

- 生成9600行representation-trial，四个样本量各有200 calibration、200 holdout A/A和200 A/B；
- ego13在n=400为FPR=1.5%、detection=100%，full64为4.5%/63.5%；
- 同release ego13-only/full64-only=`73/0`，McNemar exact p约`2.12e-22`；
- Stage6O v1与既有rollout、embedding输入未修改。

## Stage 6Q：Waymo full51 raw interaction coverage audit（Issue #258）

### 命令

```bash
waymo_dev/bin/python \
  tools/stage6q_audit_waymo_raw_interaction_coverage.py \
  --config configs/stage6q_waymo_raw_interaction_coverage_audit.json \
  --builder_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --stage6o_manifest outputs/stage6o_longitudinal_training_protocol_freeze_v1/stage6o_training_protocol_freeze_manifest.json \
  --output_dir outputs/stage6q_waymo_raw_interaction_coverage_v1
```

### 期望行为

- 直接读取原始Waymo TFRecord 00000–00050，不调用正式builder的neighbor-valid>=0.8筛选；
- 逐帧动态审计lead entry/exit、intermittent、identity switch和两种transition；
- 输出raw全部合格vehicle、正式前64 target sampling、正式builder retained三层漏斗；
- 2/3/4m几何敏感性全部报告，主规则为3m；不修改Stage6O v1、不降低5000门槛。

### 通过标准

- 51/51文件、24872 scenario全部完成且记录SHA-256；
- 3m raw intermittent<0.8=`54829`，2m/4m为`53448/51109`，全部大于5000；
- 根因判定为首帧固定front与整窗>=0.8有效率造成的builder结构性过滤；
- 决策为先修builder和重建版本，不扩大Waymo，不启动Interaction-aware v2训练。

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6p_stage6q_representation_and_raw_audit.py
python -m py_compile \
  tools/stage6p_run_representation_unpaired_release.py \
  tools/stage6q_audit_waymo_raw_interaction_coverage.py
python tools/check_no_tmp_dependencies.py
```

## Stage 6O：纵向敏感 64D 训练前冻结

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6o_freeze_longitudinal_training_protocol.py \
  --config configs/stage6o_longitudinal_representation_training_protocol.json \
  --out_dir outputs/stage6o_longitudinal_training_protocol_freeze_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6o_freeze_longitudinal_training_protocol.py
```

如果要重新生成同名冻结目录，必须显式增加`--overwrite`。这只覆盖Stage6O冻结报告，不修改
Waymo shard、基线checkpoint或任何Stage6J–6M证据。

### 2. 期望行为

- 核对配置、Waymo manifest/build/feature schema、基线checkpoint及Stage6L/6M证据hash；
- 顺序审计35个shard，不把大型context数组合并到内存；
- 检查83D context、5D mask、33D features、finite、split和meta逐行对齐；
- 检查scenario/scenario-agent跨split泄漏和冻结MD5分割算法；
- 统计速度、free/intermittent/sustained following、steady/dynamic和lateral strata；
- 写出`stage6o_waymo_data_audit.json`、freeze manifest和中文报告；
- 不导入trainer、不运行optimizer、不写`.pt`、不覆盖基线。

### 3. 通过标准

- 数据与证据hash全部匹配；
- 35 shards、164871 windows、train/val/test=`131998/16481/16392`；
- scenario和scenario-agent跨split重叠均为0；
- 所有必需数组shape正确且finite；
- 每个速度档、跟车档和运动状态达到配置中的预冻结最小覆盖；
- 所有覆盖通过时状态才可为`FROZEN_READY_FOR_IMPLEMENTATION_NOT_TRAINING`；
- 当前full51的intermittent-following为0，所以权威状态应为
  `FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING`，`training_authorized=false`；
- 阻塞时先扩展/重建Waymo数据，禁止降低门槛后直接训练。

## Stage 6L：修复版 context representation 消融

原Stage6K dose50/75 context为零邻车覆盖，旧Stage6L v1已作废。Mac重建必须同时加入三个
Python路径，并启用非零覆盖门禁：

```bash
env PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6k_longitudinal_dose_views_v1/dose50 \
  --output_dir outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired/dose50 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --required_planners pdm_closed_assertive_longitudinal_dose50_v1 pdm_closed_conservative_longitudinal_v1 \
  --write_projection_debug --write_strict_filter_diagnostic \
  --strict_filter_min_laneaware_ratio 0.8 \
  --strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6 \
  --require_nonzero_neighbor_coverage
```

25/50/75三档修复后，依次运行：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6l_freeze_context_representation_ablation.py \
  --design_json configs/stage6l_context_representation_ablation.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --stage6j_context_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --stage6j_embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --stage6k_contexts_dir outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired \
  --stage6k_embeddings_dir outputs/stage6k_longitudinal_dose_embeddings_v2_runtime_repaired \
  --stage6j_bdd_config configs/stage6j_paired_bdd_analysis.json \
  --output_dir outputs/stage6l_context_representation_ablation_freeze_v2_runtime_repaired

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6l_prepare_context_representation_ablation.py \
  --freeze_manifest outputs/stage6l_context_representation_ablation_freeze_v2_runtime_repaired/stage6l_context_representation_ablation_freeze_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --stage6j_context_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --stage6j_embedding_dir outputs/stage6j_pure_longitudinal_embeddings_v1 \
  --stage6k_contexts_dir outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired \
  --stage6k_embeddings_dir outputs/stage6k_longitudinal_dose_embeddings_v2_runtime_repaired \
  --output_dir outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired \
  --device cpu

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6l_run_context_representation_ablation.py \
  --freeze_manifest outputs/stage6l_context_representation_ablation_freeze_v2_runtime_repaired/stage6l_context_representation_ablation_freeze_manifest.json \
  --decision_addendum_manifest outputs/stage6l_preanalysis_decision_addendum_freeze_v2_runtime_repaired/stage6l_preanalysis_decision_addendum_manifest.json \
  --representation_dir outputs/stage6l_context_representation_ablation_representations_v2_runtime_repaired \
  --stage6j_context_dir outputs/stage6j_pure_longitudinal_context_v1 \
  --stage6k_contexts_dir outputs/stage6k_longitudinal_dose_context_v2_runtime_repaired \
  --stage6j_bdd_config configs/stage6j_paired_bdd_analysis.json \
  --output_dir outputs/stage6l_context_representation_ablation_results_v2_runtime_repaired
```

权威结论：A/B/C/D的task-dose Holm通过为7/11/12/2，median Z_BDD为
7.539/11.066/21.082/5.384。raw MMD²禁止跨表示比较。

## Stage 6M：Context-balanced unpaired BDD 四方法比较

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6m_freeze_context_balanced_unpaired_bdd.py \
  --design_json configs/stage6m_context_balanced_unpaired_bdd.json \
  --stage6h_config configs/stage6h_nuplan_power_curve_800.json \
  --embedding_pool_summary outputs/stage6h_expanded_800_embedding_pool_v1/stage6h_embedding_pool_summary.json \
  --embedding_pool_metadata outputs/stage6h_expanded_800_embedding_pool_v1/metadata.csv \
  --trial_bdd outputs/stage6h_nuplan_power_curve_800_v1/power_curve_trial_bdd.csv \
  --log_assignments outputs/stage6h_nuplan_power_curve_800_v1/power_curve_log_assignments.csv \
  --fixed_scope_bandwidths outputs/stage6h_nuplan_power_curve_800_v1/fixed_scope_bandwidths.csv \
  --output_dir outputs/stage6m_context_balanced_unpaired_bdd_freeze_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6m_run_context_balanced_unpaired_bdd.py \
  --freeze_manifest outputs/stage6m_context_balanced_unpaired_bdd_freeze_v1/stage6m_freeze_manifest.json \
  --trial_bdd outputs/stage6h_nuplan_power_curve_800_v1/power_curve_trial_bdd.csv \
  --output_dir outputs/stage6m_context_balanced_unpaired_bdd_results_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6m_audit_covariate_balance.py \
  --freeze_manifest outputs/stage6m_context_balanced_unpaired_bdd_freeze_v1/stage6m_freeze_manifest.json \
  --output_dir outputs/stage6m_context_balanced_unpaired_bdd_results_v1
```

n=400 raw/task/context/task+context detection为63.0%/65.0%/66.5%/64.5%，FPR为
4.5%/5.5%/5.0%/6.0%。context相对raw的配对McNemar p=0.2478，不支持稳定提升。

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6l_context_representation_ablation.py \
  tests/test_stage6m_context_balanced_unpaired_bdd.py \
  tests/test_stage5d_context_core.py \
  -k 'stage6l or stage6m or required_neighbor_coverage'
python tools/check_no_tmp_dependencies.py
```

## Stage 6E：公开 A/A 标定与 log-disjoint 版本发布模拟

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6e_calibrate_unpaired_release.py \
  --embedding_path outputs/stage7_m6_5_locked_confirmation_embeddings_v1/embedding.npy \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --config_json configs/stage6e_nuplan_release_emulation.json \
  --paired_oracle_json outputs/stage7_m6_5_locked_confirmation_analysis_v1/m6_5_locked_confirmation_summary.json \
  --output_dir outputs/stage6e_nuplan_release_emulation_v1
```

### 2. 期望行为

- 先确认310个 scenario 各有且只有 assertive/conservative 两行，并审计 pair 内 log、map、
  scenario type 一致；
- 以257个 logs 为不可拆分 cluster，运行600次近似 ODD-balanced pseudo releases；
- 每次 trial 的 A/B logs 和 scenario tokens overlap 必须为0；
- 200个同版本 A/A calibration trials 冻结每个 scope 的95% threshold；独立随机流的
  200个 A/A evaluation 估计误报率，200个双方向 A/B 估计检出率；
- overall 是 primary；task rates 是未做 multiplicity control 的 diagnostic；
- paired oracle 只作为参考读取，不重算、不修改，也不能把 paired p-value 当成 unpaired
  p-value；
- 输出完整 trial、log assignments、support audit、threshold、operating characteristics、
  JSON/provenance 和 Markdown report；输出目录已存在时拒绝覆盖。

### 3. 通过标准

- summary 状态为 `PASS_PUBLIC_FIELD_RELEASE_EMULATION`；
- 600/600 trials 的 log overlap 和 scenario overlap 都为0；
- overall A/A threshold=`0.00994295`，holdout false-positive=7/200=`3.5%`；
- overall A/B detection=70/200=`35.0%`，Wilson 95% CI 与 A/A 区间分离；
- conclusion 为 `AB_SEPARATED_FROM_AA_BUT_SINGLE_RELEASE_SENSITIVITY_LIMITED`，不得写成
  稳定量产报警能力；
- lane-change 诊断约53.5% detection，stop/go 不得声明有检测能力；
- 公司数据可用后重新标定，禁止直接迁移当前 absolute threshold。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6e_calibrate_unpaired_release.py
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
```

## Stage 6F：不配对 BDD 样本量功效曲线

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6f_unpaired_power_curve.py \
  --embedding_path outputs/stage7_m6_5_locked_confirmation_embeddings_v1/embedding.npy \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --config_json configs/stage6f_nuplan_power_curve.json \
  --paired_oracle_json outputs/stage7_m6_5_locked_confirmation_analysis_v1/m6_5_locked_confirmation_summary.json \
  --output_dir outputs/stage6f_nuplan_power_curve_v1
```

### 2. 期望行为

- 对40/60/80/100/125/150场景/版本分别运行600次伪发布，共3,600 trials；
- 每个样本量独立生成 A/A threshold、A/A holdout false-positive 和双方向 A/B
  detection，不能跨样本量复用 threshold；
- 完整 log 不可拆分，每次 A/B log 和 scenario-token overlap 都必须为0；
- 实际 n_A/n_B 因单 log 含1–2场景允许目标±1，并写入 split audit；
- 输出 overall primary 和未做 multiplicity control 的 task diagnostics；
- 生成 CSV、JSON、provenance、Markdown，以及 PNG/PDF 功效曲线；
- 不拟合或输出150场景/版本之外的精确 power extrapolation。

### 3. 通过标准

- execution status=`POWER_CURVE_COMPLETE`；
- 3,600/3,600 trials 的 log/token overlap=0，实际样本量全部在目标±1内；
- 六个 overall thresholds 全部通过；n=40 的 following/stop-go/dense task thresholds
  因有效 trials 不足必须标记为 insufficient；
- overall detection 曲线为7.0%、10.5%、12.0%、11.5%、17.0%、35.0%；
- n=150 detection=35.0%（Wilson `[28.7%,41.8%]`），A/A false-positive=7.0%
  （`[4.2%,11.4%]`）；
- sufficiency status=`TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS`；
- 不得报告达到80%所需的伪精确样本量；扩样后必须增加新的实证档位并重新标定阈值。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6f_unpaired_power_curve.py
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
```

## Stage 7 Milestone 6.6：paper evidence package

```bash
MPLCONFIGDIR=/tmp/mpl-m6-6 \
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage7_m6_6_build_confirmation_evidence.py \
  --analysis_dir outputs/stage7_m6_5_locked_confirmation_analysis_v1 \
  --analysis_lock outputs/stage7_m6_5_locked_analysis_freeze_v1/m6_5_confirmation_analysis_lock.json \
  --quality_summary outputs/stage7_m6_5_locked_confirmation_quality_v1/milestone2b_summary.json \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --paired_delta_csv outputs/stage7_m6_5_locked_confirmation_stage7f_v1/paired_delta_assertive_minus_conservative/paired_delta_by_scenario.csv \
  --output_dir outputs/stage7_m6_6_confirmation_evidence_v1
```

输出目录必须不存在；工具会先复核 M6.5 lock/summary 的所有输入 hashes 和完整性门。
默认固定 seed=`20260808`、bootstrap=`10000`。生成10张 CSV/Markdown 表、6张
PNG/PDF 图、summary、provenance、report 和中英文 manuscript results，不重新计算
任何确认性 p 值。

权威状态为 `PASS_WITH_QUALITY_LIMITATIONS`。总体最大 fallback 与 embedding distance
rho=`0.5088`（task-stratified 95% CI `[0.4086,0.6035]`）；task-adjusted rank-residual
rho=`0.4499`（95% CI `[0.3842,0.5719]`）。两者都是 post-treatment descriptive
association，不能解释为因果机制或 covariate adjustment。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage7_m6_6_build_confirmation_evidence.py
```

## Stage 6D：异源实路软件版本 BDD

### 1. 命令

先复制并按实路字段修改示例设计。matching covariates、task slices 必须是版本运行前
已确定或不受待比较软件影响的 pre-treatment 字段，cluster 建议使用独立采集单元
（优先 log / route-day / vehicle-day），不能把逐帧 row ID 当作 cluster。

```bash
cp configs/stage6d_unpaired_version_design.example.json \
  configs/stage6d_company_release_design.json

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6d_unpaired_version_bdd.py \
  --embedding_path /absolute/path/to/embedding.npy \
  --metadata_csv /absolute/path/to/version_metadata.csv \
  --design_json configs/stage6d_company_release_design.json \
  --bootstrap_repetitions 1000 \
  --max_mmd_samples 2000 \
  --seed 20260809 \
  --output_dir outputs/stage6d_company_release_v1
```

本地 nuPlan balanced interface smoke（只验证接口，不用于实路结论）：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6d_unpaired_version_bdd.py \
  --embedding_path outputs/stage7_m6_5_locked_confirmation_embeddings_v1/embedding.npy \
  --metadata_csv outputs/stage7_m6_5_locked_confirmation_embeddings_v1/metadata.csv \
  --design_json configs/stage6d_nuplan_interface_smoke_design.json \
  --bootstrap_repetitions 50 \
  --max_mmd_samples 2000 \
  --seed 20260809 \
  --output_dir outputs/stage6d_nuplan_interface_smoke_v1
```

### 2. 期望行为

- 输入 embedding 必须是与 metadata 逐行对齐的二维 `.npy`；输出目录必须不存在；
- categorical covariates 做 exact cells，continuous covariates 用 A/B 合并样本的冻结
  quantile edges 分箱；
- 两组重加权到 equal-group pooled common-support reference；
- 同时输出 raw observed-mixture BDD、standardized BDD、task-frequency shift 和
  task-conditioned BDD；
- cluster bootstrap 每次都按组重采 cluster，并重新计算共同支持和权重；
- timing 泄漏、无共同支持或 support / ESS / weight / cluster 门槛失败时 fail closed，
  状态为 `NOT_COMPARABLE_INSUFFICIENT_COMMON_SUPPORT`；
- 工具不输出 universal p-value，也不把 BDD 解释为安全性、性能优劣或因果效应。

### 3. 通过标准

- `stage6d_unpaired_version_summary.json` 状态为
  `PASS_DESCRIPTIVE_STANDARDIZED_VERSION_DRIFT`；
- A/B support fraction、ESS ratio、max weight ratio 和 cluster 数全部通过设计门槛；
- `common_support_cells.csv` 非空，`covariate_balance.csv` 不出现未解释的严重失衡；
- overall 与每个冻结 task 都有 raw / standardized MMD²、固定 bandwidth 和有效的
  cluster-bootstrap SE / 95% 区间；
- `task_frequency_shift.csv` 与 within-task BDD 分开，provenance 记录输入和设计 hash；
- nuPlan interface smoke 应为 A/B 各310行、20/20 common cells、ESS ratio=1.0，raw 与
  standardized overall MMD² 均约为 `0.0044865829`，bootstrap 样本共600行；
- 生产应用还必须完成独立同版本 A/A 历史窗口标定，否则只能报告 descriptive drift，
  不能设置正式报警阈值。

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6d_unpaired_version_bdd.py
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
```
# Stage 6R/6S Dynamic Interaction（2026-08-12）

> 重要更正：`stage6r_dynamic_builder_v2_pilot_file0/1/2`及原`pilot_decision_v1`
> 是未保留Waymo局部邻接index区间的pre-fix结果，已由视觉检查判定失效，不得用于启动full51。
> 已中断的`stage6r_dynamic_full51_part_*`也不得续跑或finalize。修复版必须使用新的
> `*_semantic_strict_multirelation_*`目录，并同时通过自动、拓扑重建、独立视觉三道门禁。

## Stage 6R 3-file pilot

```bash
waymo_dev/bin/python tools/build_waymo_dynamic_interaction_dataset_v2.py \
  --waymo_dir /Users/liuqing/Projects/01_E2E_QA_Code/training \
  --out_dir outputs/stage6r_dynamic_builder_v2_pilot_semantic_strict_multirelation_file0 \
  --file_start 0 --file_end 1 --max_agents_per_scenario 64 \
  --window_len 80 --stride 20 --dt 0.1 --min_valid_ratio 0.8 \
  --min_speed 1.0 --assignment_mode lane_aware_only \
  --output_shard_size 5000 --overwrite --progress_every 50
```

预期：生成动态track-id/mask/switch/derivative-mask和longitudinal v2 supervision；`dynamic_summary_validation_pass=true`。对file1/file2分别使用`--file_start 1/2 --file_end 2/3`。

```bash
waymo_dev/bin/python tools/stage6r_audit_dynamic_builder_pilot.py \
  --config configs/stage6r_waymo_dynamic_builder_v2.json \
  --legacy_root outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --dynamic_roots outputs/stage6r_dynamic_builder_v2_pilot_semantic_strict_multirelation_file{0,1,2} \
  --output_dir outputs/stage6r_dynamic_builder_v2_pilot_audit_semantic_strict_multirelation_v1 --overwrite
```

预期：自动门禁通过后状态为`AUTOMATED_PASS_PENDING_MANUAL_REVIEW`；必须复核20个case后才能启动full51。

```bash
waymo_dev/bin/python tools/stage6r_review_dynamic_pilot_cases.py \
  --cases_csv outputs/stage6r_dynamic_builder_v2_pilot_audit_semantic_strict_multirelation_v1/stage6r_manual_semantic_cases.csv \
  --waymo_dir /Users/liuqing/Projects/01_E2E_QA_Code/training \
  --file_start 0 --file_end 3 \
  --output_dir outputs/stage6r_dynamic_builder_v2_pilot_topology_semantic_strict_multirelation_v1 --overwrite
```

预期：20/20 case、每slot 4例通过原始TFRecord重建与lane-topology复核，track-id重建不一致为0；
状态只能是`TOPOLOGY_RECONSTRUCTION_PASS_PENDING_VISUAL_REVIEW`，不能把自动重建称为人工语义通过。
随后必须实际查看overview图，确认20例局部车道方向与slot含义正确，再显式运行
`stage6r_record_visual_semantic_review.py`记录图像SHA。

## Stage 6S rollout与机制门禁

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage6s_run_interaction_dominant_rollouts.py \
  --freeze_manifest outputs/stage6s_interaction_dominant_freeze_v1/stage6s_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6s_interaction_dominant_freeze_v1/stage6s_locked_scenarios.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage6s_interaction_dominant_batch_v1 --start_order 1 --end_order 999 \
  --execute --resume \
  --confirm_locked_scenarios_sha256 c07542c693d31d02327b2d16aabb05b68057da5f270818801049b41d41dc8130
```

预期：24/24 `SUCCEEDED`、0 failure、48条official rollout；未读取embedding/BDD。

机制报告预期状态以冻结门禁为准。本次结果为`PDM_INTERACTION_BENCHMARK_LIMITATION`：速度/加速度差满足“小”，但THW与front-gap两个预冻结interaction指标只有THW通过；不得在本批结果后调参并继续当作同一确认实验。

## Stage 6R full51 finalize 与 Stage 6O-v2

```bash
waymo_dev/bin/python tools/stage6r_finalize_dynamic_full51.py \
  --part_roots outputs/stage6r_dynamic_full51_semantic_strict_part_00_09 \
    outputs/stage6r_dynamic_full51_semantic_strict_part_09_18 outputs/stage6r_dynamic_full51_semantic_strict_part_18_27 \
    outputs/stage6r_dynamic_full51_semantic_strict_part_27_36 outputs/stage6r_dynamic_full51_semantic_strict_part_36_44 \
    outputs/stage6r_dynamic_full51_semantic_strict_part_44_51 \
  --waymo_dir /Users/liuqing/Projects/01_E2E_QA_Code/training \
  --pilot_decision outputs/stage6r_dynamic_builder_v2_pilot_decision_semantic_strict_v1/stage6r_pilot_decision.json \
  --expected_file_count 51 --output_dir outputs/stage6r_dynamic_full51_semantic_strict_v1 --overwrite

waymo_dev/bin/python tools/stage6o_v2_freeze_training_readiness.py \
  --config configs/stage6r_waymo_dynamic_builder_v2.json \
  --dynamic_full51_manifest outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json \
  --stage6o_v1_manifest outputs/stage6o_longitudinal_training_protocol_freeze_v1/stage6o_training_protocol_freeze_manifest.json \
  --expected_stage6o_v1_sha256 4175054bbcf38d604ff0bab5bda77233a066c475c5e19335b0d219f00f1d164e \
  --output_dir outputs/stage6o_v2_dynamic_training_readiness_v1 --overwrite
```

预期：finalize只使用全体train split重算q01/q99、median/IQR并写源TFRecord与shard SHA ledger；Stage6O-v2保持5000 intermittent门槛并强制验证旧Stage6O v1 SHA。即使v2通过，也只允许准备训练，不会启动checkpoint训练。

实际通过标准与结果：

- 51/51 TFRecord、24872 scenario、168700窗口、36 shard；train/val/test=`135046/16870/16784`。
- Stage6O-v2状态=`FROZEN_READY_FOR_INTERACTION_AWARE_V2_PREPARATION`，8项门禁全部为true。
- train intermittent=`63415 >= 5000`；scenario跨split重叠、nonfinite、shape和跨identity导数违规均为0。
- 五槽帧覆盖率=`28.73%/17.78%/16.77%/17.65%/17.19%`，switch rate=`1.29%/2.09%/2.48%/2.12%/2.64%`。
- 新longitudinal raw `|q99|=21.64/6.20/76.30`，normalized max abs=`4.74`；窗口RMS accel median=`1.48`，jerk median/q90=`15.51/28.47`。
- 旧Stage6O v1 SHA保持`4175054bbcf38d604ff0bab5bda77233a066c475c5e19335b0d219f00f1d164e`且永久BLOCKED；未训练、未扩大Waymo。

# Stage 6S-v2 interaction benchmark development与confirmation冻结（Issue #261）

## 1. 命令

Development机制评估前，nuPlan msgpack反序列化必须显式包含`tuplan_garage`：

```bash
env PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6s_v2_development_view_v1 \
  --output_dir outputs/stage6s_v2_development_context_v1 \
  --max_neighbors_for_context 5 \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --required_planners pdm_closed_interaction_short_headway_v2 pdm_closed_interaction_long_headway_v2 \
  --require_nonzero_neighbor_coverage --overwrite

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6s_v2_evaluate_development_mechanism.py \
  --config configs/stage6s_v2_interaction_benchmark.json \
  --freeze_manifest outputs/stage6s_v2_development_freeze_v1/stage6s_v2_development_freeze_manifest.json \
  --view_dir outputs/stage6s_v2_development_view_v1 \
  --context_dir outputs/stage6s_v2_development_context_v1 \
  --output_dir outputs/stage6s_v2_development_mechanism_v1 --overwrite

/Users/liuqing/miniconda3/envs/nuplan/bin/python \
  tools/stage6s_v2_freeze_confirmation.py \
  --config configs/stage6s_v2_interaction_benchmark.json \
  --inventory_summary outputs/stage6s_v2_pretreatment_interaction_inventory_v1/stage6s_v2_pretreatment_inventory_summary.json \
  --inventory_csv outputs/stage6s_v2_pretreatment_interaction_inventory_v1/stage6s_v2_pretreatment_interaction_inventory.csv \
  --development_manifest outputs/stage6s_v2_development_freeze_v1/stage6s_v2_development_freeze_manifest.json \
  --development_roster outputs/stage6s_v2_development_freeze_v1/stage6s_v2_development_roster.csv \
  --development_mechanism outputs/stage6s_v2_development_mechanism_v1/stage6s_v2_development_mechanism_summary.json \
  --stage6s_v1_roster outputs/stage6s_interaction_dominant_freeze_v1/stage6s_locked_scenarios.csv \
  --output_dir outputs/stage6s_v2_confirmation_freeze_v1 --overwrite
```

## 2. 期望行为

- context命令只从24个development pair的official msgpack恢复动态背景车辆；缺少`tuplan_garage`
  时必须由`--require_nonzero_neighbor_coverage`报错，不能静默接受五槽全零。
- mechanism命令只读取realized ego trajectory和semantic neighbor context，不读取embedding、
  BDD/MMD或confirmation roster；THW只保留有限的`0 < THW < 20 s`。
- confirmation命令只从pre-treatment eligible inventory排序，在development log/token与旧Stage6S-v1
  token排除后冻结80对；development outcome只作为是否允许冻结的门禁，不参与排序。
- confirmation统计固定为原机制门禁加按`log_name`聚类的10,000次bootstrap percentile 95%区间，
  seed为620261；本阶段只冻结方法，不计算confirmation outcome。
- 三条命令均不会训练checkpoint或运行正式新模型评估；confirmation freeze不会启动其rollout。

## 3. 通过标准

- development context `validation.pass=true`、front coverage非零且slot sanity通过；
- 24/24 complete pair、至少18对有有效front，`|Δ mean speed|<=1.0 m/s`、
  `|Δ RMS accel|<=0.75 m/s²`；四项interaction mechanism至少两项通过；
- 本次实际通过项为front gap与finite THW，机制状态为
  `DEVELOPMENT_MECHANISM_PASS_CONFIRMATION_FREEZE_ALLOWED`；
- confirmation为80 pair、60–100冻结范围内，development log overlap=0、scenario overlap=0、
  Stage6S-v1 token overlap=0；
- confirmation状态为`CONFIRMATION_ROSTER_FROZEN_NOT_RUN`，并明确记录outcome-blind、未运行rollout、
  未读取embedding/BDD、未训练checkpoint。

# Stage 6T A/B/C训练与盲测协议冻结（Issue #262）

## 1. 命令

```bash
waymo_dev/bin/python tools/stage6t_freeze_training_evaluation_protocol.py \
  --config configs/stage6t_training_evaluation_protocol.json \
  --output_dir outputs/stage6t_training_evaluation_protocol_freeze_v1 \
  --overwrite
```

协议测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6t_freeze_training_evaluation_protocol.py
```

## 2. 期望行为

- 校验Dynamic v2 manifest、36个shard冻结SHA、168700行shape/split和Stage6O-v2门禁；
- 校验Stage6O-v1继续BLOCKED、old64 SHA不变、Stage6S-v2的80-pair roster仍未运行且未解盲；
- 冻结A/B/C架构、采样、loss、seed、预算、checkpoint选择和四类成绩单；
- 检测六个part-local 33D标准化不同，禁止trainer使用`interaction_feat_style.npy`，从raw33生成全体
  train-only global mean/std，但不改写任何旧shard；
- 输出manifest、A/B/C差异CSV、global standardization、训练输入SHA ledger和中文报告；
- 不训练checkpoint，不读取Waymo test，不运行nuPlan/confirmation，不读取embedding、BDD或MMD。

## 3. 通过标准

- 状态为`FROZEN_READY_FOR_ABC_TRAINER_IMPLEMENTATION_NOT_TRAINING`；
- config/source/dataset/Stage6O-v2/Stage6O-v1/Stage6S-v2/environment八项validation均为true；
- 36 shards、168700 rows、train/val/test=`135046/16870/16784`，shape failure、raw33 nonfinite、
  scenario cross-split overlap和SHA mismatch均为0；
- 六个part-local standardization被识别并明确禁止用于Stage6T训练，全局raw33统计train_count=135046；
- A/B/C×3 seed计划checkpoint=9，但`training_authorized=false`、`checkpoint_training_launched=false`、
  实际candidate输出非空目录数=0；
- Stage6S-v2保持`CONFIRMATION_ROSTER_FROZEN_NOT_RUN`，confirmation rollout与embedding读取均为false。

# Stage 6U Unified A/B/C Trainer实现冻结（Issue #263）

## 1. 命令

运行synthetic与小规模Waymo train/val smoke：

```bash
waymo_dev/bin/python tools/stage6u_smoke_unified_abc_trainer.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --output_dir outputs/stage6u_unified_abc_trainer_smoke_v1 \
  --overwrite
```

Smoke全部通过后冻结implementation：

```bash
waymo_dev/bin/python tools/stage6u_freeze_trainer_implementation.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --smoke_dir outputs/stage6u_unified_abc_trainer_smoke_v1 \
  --output_dir outputs/stage6u_trainer_implementation_freeze_v1 \
  --overwrite
```

测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python -m pytest -q \
  tests/test_stage6u_unified_abc_trainer.py \
  tests/test_stage6t_freeze_training_evaluation_protocol.py
```

## 2. 期望行为

- 单一trainer按candidate配置构造A/B/C，全部输入83D、输出64D；
- B/C同seed生成相同sample/batch/pair/weight/dropout/augmentation/schedule/budget随机计划，并输出逐项SHA ledger；
- synthetic与Dynamic v2 train/val subset各运行A/B/C少量forward/backward；
- 只读取`interaction_feat_style_raw.npy`并应用Stage6T global33，不读取part-local标准化数组；
- checkpoint smoke验证save/load和epoch、batch cursor、optimizer、scheduler、Python/NumPy/Torch RNG与plan恢复；
- freeze记录trainer/config/data/standardization/smoke/fairness SHA、参数量、环境、ETA和0/9正式checkpoint；
- 不读取test、Stage6J/K/P、nuPlan、embedding、BDD/MMD或Stage6S-v2 confirmation，不启动正式训练。

Formal CLI虽然已实现完整epoch loop，但没有独立授权manifest时必须失败：

```bash
waymo_dev/bin/python tools/stage6u_unified_abc_trainer.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --candidate A --mode formal --seed 3407 \
  --output_dir outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3407
```

预期报错：缺少独立authorization manifest与implementation freeze SHA，不会创建正式输出。

## 3. 通过标准

- smoke状态=`PASS_UNIFIED_ABC_TRAINER_SMOKE_NO_FORMAL_TRAINING`，全部validation=true；
- A/B/C encoder参数量=`106560/106560/105616`，embedding shape均为64D，loss/gradient均finite；
- synthetic和Waymo subset的B/C全部11项公平随机流SHA相同；
- global33手工公式逐位一致、fit split=train、train_count=135046；
- resume连续/恢复loss序列与最终model state SHA完全一致；
- implementation状态=`FROZEN_READY_FOR_ABC_FORMAL_TRAINING`，全部validation=true；
- `formal_checkpoint_count=0`、`formal_training_authorized=false`、`formal_training_launched=false`；
- Waymo test、nuPlan、BDD/MMD和confirmation读取/运行标志全部为false。

# Stage 6U A/B/C正式训练授权、串行运行与每小时监控

## 1. 命令

Trainer代码有任何变化后，先重跑smoke和implementation freeze：

```bash
waymo_dev/bin/python tools/stage6u_smoke_unified_abc_trainer.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --output_dir outputs/stage6u_unified_abc_trainer_smoke_v2_preformal \
  --overwrite

waymo_dev/bin/python tools/stage6u_freeze_trainer_implementation.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --smoke_dir outputs/stage6u_unified_abc_trainer_smoke_v2_preformal \
  --output_dir outputs/stage6u_trainer_implementation_freeze_v2_preformal \
  --overwrite
```

用最终freeze生成一次性formal authorization：

```bash
waymo_dev/bin/python tools/stage6u_create_formal_authorization.py \
  --config configs/stage6u_unified_abc_trainer.json \
  --implementation_freeze_manifest \
    outputs/stage6u_trainer_implementation_freeze_v2_preformal/stage6u_trainer_implementation_freeze_manifest.json \
  --output_dir outputs/stage6u_formal_training_authorization_v1
```

在单MPS上串行启动9个任务；Mac终端可用`caffeinate`防止系统休眠：

```bash
caffeinate -dimsu waymo_dev/bin/python tools/stage6u_run_formal_abc_serial.py \
  --authorization_manifest \
    outputs/stage6u_formal_training_authorization_v1/stage6u_formal_training_authorization_manifest.json \
  --run_dir outputs/stage6u_abc_formal_training_v1
```

普通中断或重启后，从现有`resume_model.pt`继续：

```bash
caffeinate -dimsu waymo_dev/bin/python tools/stage6u_run_formal_abc_serial.py \
  --authorization_manifest \
    outputs/stage6u_formal_training_authorization_v1/stage6u_formal_training_authorization_manifest.json \
  --run_dir outputs/stage6u_abc_formal_training_v1 \
  --resume
```

只读查看当前进度与剩余时间：

```bash
waymo_dev/bin/python tools/stage6u_monitor_formal_training.py \
  --run_dir outputs/stage6u_abc_formal_training_v1
```

## 2. 期望行为

- Authorization绑定最终implementation freeze SHA、A/B/C、seeds 3407/3408/3409、A→B→C串行顺序和9个精确输出目录；
- orchestrator同一时间最多启动一个formal trainer，自动跳过已经完成且SHA绑定正确的任务；
- formal trainer只打开Dynamic v2 train/val，train优化、val选best与早停，不打开test；
- train/val每epoch显示tqdm；每100 steps写`progress.jsonl`并原子更新`resume_model.pt`；
- 任务完成后写`best_model.pt`、`last_model.pt`、`formal_training_summary.json`；
- 9/9完成后自动生成JSON/CSV checkpoint ledger与中文锁定报告，然后停止；
- 不运行Stage6J/K/P、nuPlan、embedding、BDD/MMD或Stage6S-v2 confirmation。

## 3. 通过标准

- 新implementation freeze状态=`FROZEN_READY_FOR_ABC_FORMAL_TRAINING`且全部validation=true；
- authorization状态=`AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING`且其implementation freeze SHA与文件实际SHA完全一致；
- 全程最多一个formal trainer进程，任务顺序固定为A3407/3408/3409、B3407/3408/3409、C3407/3408/3409；
- 每个任务`training_complete=true`，有best/last checkpoint、best epoch、Waymo val loss和完整resume history；
- 9个best checkpoint均计算并写入各自SHA；primary seed保持3407；
- ledger状态=`LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK`；
- test、Stage6J/K/P、nuPlan、BDD/MMD、confirmation标志全部为false，本阶段不自动解锁正式评估。

# Stage 6V一次性盲测与最终决策

## 1. 命令

一次性授权与前三类冻结评估分别使用：

```bash
waymo_dev/bin/python tools/stage6v_create_blind_evaluation_authorization.py \
  --output_dir outputs/stage6v_blind_evaluation_authorization_v1

waymo_dev/bin/python tools/stage6v_run_waymo_test.py \
  --output_dir outputs/stage6v_waymo_dynamic_v2_test_v1

waymo_dev/bin/python tools/stage6v_run_stage6jk_blind.py \
  --output_dir outputs/stage6v_stage6jk_paired_blind_v1

waymo_dev/bin/python tools/stage6v_run_stage6p_blind.py \
  --output_dir outputs/stage6v_stage6p_unpaired_blind_v1
```

Stage6S-v2必须先运行冻结roster的official rollout，再根据完整执行状态决定是否允许机制和representation分析。
本轮权威执行冻结与最终汇总命令为：

```bash
waymo_dev/bin/python tools/stage6v_finalize_stage6s_v2_confirmation_execution.py \
  --run_dir outputs/stage6v_stage6s_v2_confirmation_batch_v1 \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --freeze_manifest outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_roster.csv \
  --batch_manifest outputs/stage6v_stage6s_v2_confirmation_batch_v1/batch_manifest.json \
  --batch_state outputs/stage6v_stage6s_v2_confirmation_batch_v1/batch_state.json \
  --batch_status_csv outputs/stage6v_stage6s_v2_confirmation_batch_v1/batch_scenario_status.csv \
  --output_dir outputs/stage6v_stage6s_v2_confirmation_execution_freeze_v1

waymo_dev/bin/python tools/stage6v_finalize_blind_evaluation.py \
  --authorization outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json \
  --waymo_manifest outputs/stage6v_waymo_dynamic_v2_test_v1/stage6v_waymo_test_result_manifest.json \
  --waymo_decisions outputs/stage6v_waymo_dynamic_v2_test_v1/waymo_test_decisions.csv \
  --paired_manifest outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_result_manifest.json \
  --paired_decisions outputs/stage6v_stage6jk_paired_blind_v1/stage6v_stage6jk_decisions.csv \
  --unpaired_manifest outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_result_manifest.json \
  --unpaired_decisions outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_primary_decisions.csv \
  --unpaired_seed_stability outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_seed_stability_n400.csv \
  --confirmation_execution outputs/stage6v_stage6s_v2_confirmation_execution_freeze_v1/stage6s_v2_confirmation_execution_freeze.json \
  --output_dir outputs/stage6v_one_time_blind_evaluation_final_v1
```

## 2. 期望行为

- 先校验并绑定Stage6T/6U、checkpoint ledger、9个best checkpoint和Stage6S-v2 roster SHA；
- Waymo test只用primary 3407做确认结论，其余seed只做稳定性；
- Stage6J/K和Stage6P复用冻结rollout/split，不重跑原纵向simulation；
- Stage6S-v2只有80/80 official rollout完整且机制门禁通过后才能读取representation；
- 任何失败都不得触发换seed、换epoch、训练返工、benchmark替换或complete-case重定义；
- 输出独立manifest、CSV和中文报告，不跨representation比较raw MMD²。

## 3. 通过标准

- authorization包含`evaluation results cannot trigger retraining or protocol changes`且SHA匹配；
- Waymo、Stage6J/K、Stage6P结果状态分别冻结完成；
- confirmation若不完整，必须状态为`CONFIRMATION_EXECUTION_INCOMPLETE_STOP_NO_MECHANISM_OR_EMBEDDING`；
- confirmation失败时`embedding_or_bdd_read=false`且不生成post-hoc子集；
- 最终manifest按Stage6T联合门禁给出可审计模型决策，并保持训练/协议修改标志为false。

## 4. 本轮结果

Stage6U的A/B/C×3407/3408/3409共9个任务已全部完成并锁定，状态为
`LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK`。所有best epoch只由Waymo val选择，primary seed固定3407；
checkpoint ledger SHA为`e87c74527d3702de49bc68bebd47ebb485f3ced2a143cd5724cc3c12d59e7ab5`。

Stage6V盲测授权SHA为`c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd`，明确禁止用结果返工训练或协议。
Waymo primary的A/B/C longitudinal delta为-0.0232/+0.0248/+0.0159；综合非劣性均通过，但完整Waymo门禁均未通过。

Stage6J/K paired中ego13以4/4 overall、12/12 task×dose和median Z=21.115唯一通过完整门禁。A为4/4、7/12、
Z=8.630；B/C均为3/4、2/12，三者未通过。Stage6P n=400则明显改善：old64/A/B/C/ego13的context-balanced
detection为66.5%/90.5%/100%/99.5%/100%，FPR为5.0%/3.0%/5.0%/6.5%/2.0%；A/B/C均通过unpaired门禁。

Stage6S-v2的80个冻结scenario有61个成功、19个因nuPlan官方`valid_scenes` scene-rank边界规则失败；原token重试
仍完全复现。禁止事后替换roster或把61个成功项重新定义为confirmation，因此mechanism未评估、interaction
embedding/BDD未读取、C相对neighbor-zero增量不可判定。

最终状态为`FROZEN_STAGE6V_ONE_TIME_BLIND_EVALUATION_COMPLETE`，预冻结决策是
`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。正结果限于新64D显著改善unpaired release检出；Waymo/paired
门禁失败和confirmation执行失败必须作为限制或负结果同步披露。完整报告见
`docs/stage6v_one_time_blind_evaluation_report_zh.md`。

# Stage 6W-A paired/unpaired解释与Stage 6S-v3 prospective confirmation

## 1. 命令

Stage6W-A只复用冻结representation、800-pair pool和release splits：

```bash
waymo_dev/bin/python tools/stage6w_a_analyze_paired_unpaired_separation.py \
  --output_dir outputs/stage6w_a_paired_unpaired_mechanism_v1

waymo_dev/bin/python tools/stage6w_a_context_balanced_driver_addendum.py \
  --output_dir outputs/stage6w_a_context_balanced_driver_addendum_v2
```

Stage6S-v3先在任何rollout前冻结官方scene可运行性边界和80-pair roster：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage6s_v3_freeze_confirmation.py \
  --stage6s_v2_config configs/stage6s_v2_interaction_benchmark.json \
  --repair_config configs/stage6s_v3_interaction_confirmation_repair.json \
  --inventory_summary outputs/stage6s_v2_pretreatment_interaction_inventory_v1/stage6s_v2_pretreatment_inventory_summary.json \
  --inventory_csv outputs/stage6s_v2_pretreatment_interaction_inventory_v1/stage6s_v2_pretreatment_interaction_inventory.csv \
  --development_manifest outputs/stage6s_v2_development_freeze_v1/stage6s_v2_development_freeze_manifest.json \
  --development_roster outputs/stage6s_v2_development_freeze_v1/stage6s_v2_development_roster.csv \
  --development_mechanism outputs/stage6s_v2_development_mechanism_v1/stage6s_v2_development_mechanism_summary.json \
  --stage6s_v1_roster outputs/stage6s_interaction_dominant_freeze_v1/stage6s_locked_scenarios.csv \
  --stage6s_v2_confirmation_manifest outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_freeze_manifest.json \
  --stage6s_v2_confirmation_roster outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_roster.csv \
  --stage6s_v2_confirmation_design outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_frozen_design.json \
  --stage6s_v2_execution_failure outputs/stage6v_stage6s_v2_confirmation_execution_freeze_v1/stage6s_v2_confirmation_execution_freeze.json \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_devkit_root ../nuplan-devkit \
  --nuplan_scenario_query_source ../nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py \
  --output_dir outputs/stage6s_v3_confirmation_freeze_v1
```

official rollout使用冻结roster SHA
`47ad896c2afcb4c2a6272f8027eb50cc19bb7a3e6b06a64fbc10d7466400d5e7`：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage6s_v3_run_confirmation_rollouts.py \
  --freeze_manifest outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_roster.csv \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root ../nuplan/dataset/maps --nuplan_data_root ../nuplan/dataset \
  --nuplan_exp_root ../nuplan/exp --nuplan_devkit_root ../nuplan-devkit \
  --tuplan_garage_root ../tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  --expected_nuplan_commit e9241677997dd86bfc0bcd44817ab04fe631405b \
  --expected_tuplan_commit b51d5d04fac1bd4389653b9ab2ff73ea88f435a3 \
  --output_dir outputs/stage6s_v3_confirmation_batch_v1 --execute \
  --confirm_locked_scenarios_sha256 47ad896c2afcb4c2a6272f8027eb50cc19bb7a3e6b06a64fbc10d7466400d5e7
```

80/80成功后构建view/context并运行机制门禁：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage6s_v3_prepare_confirmation_view.py \
  --freeze_manifest outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_freeze_manifest.json \
  --locked_scenarios_csv outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_roster.csv \
  --batch_manifest outputs/stage6s_v3_confirmation_batch_v1/batch_manifest.json \
  --batch_state outputs/stage6s_v3_confirmation_batch_v1/batch_state.json \
  --batch_status_csv outputs/stage6s_v3_confirmation_batch_v1/batch_scenario_status.csv \
  --output_dir outputs/stage6s_v3_confirmation_view_v1 --overwrite

env PYTHONPATH=../nuplan-devkit:../tuplan_garage:. \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage6s_v3_confirmation_view_v1 \
  --output_dir outputs/stage6s_v3_confirmation_context_v1 \
  --max_neighbors_for_context 5 --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root ../nuplan/dataset/maps \
  --nuplan_db_root ../nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --required_planners pdm_closed_interaction_short_headway_v2 pdm_closed_interaction_long_headway_v2 \
  --require_nonzero_neighbor_coverage --overwrite

/Users/liuqing/miniconda3/envs/nuplan/bin/python tools/stage6s_v3_evaluate_confirmation_mechanism.py \
  --design outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_frozen_design.json \
  --freeze_manifest outputs/stage6s_v3_confirmation_freeze_v1/stage6s_v3_confirmation_freeze_manifest.json \
  --view_dir outputs/stage6s_v3_confirmation_view_v1 \
  --context_dir outputs/stage6s_v3_confirmation_context_v1 \
  --output_dir outputs/stage6s_v3_confirmation_mechanism_v1 --overwrite
```

只有机制状态为`STAGE6S_V3_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED`时运行：

```bash
waymo_dev/bin/python tools/stage6s_v3_evaluate_representations.py \
  --mechanism_summary outputs/stage6s_v3_confirmation_mechanism_v1/stage6s_v3_confirmation_mechanism_summary.json \
  --context_dir outputs/stage6s_v3_confirmation_context_v1 \
  --output_dir outputs/stage6s_v3_confirmation_representations_v1
```

## 2. 期望行为

- Stage6W-A在同一800-pair pool、同一n=400下比较paired/unpaired，并输出pair displacement、方向一致性、
  planner/log/scenario能量分解和raw/context-balanced signal-noise归因；不训练、不重跑nuPlan。
- Stage6S-v3冻结前用nuPlan官方`get_scenarios_from_db`验证100% runnability，排除v1、v2 development与
  v2全部80个confirmation token；v2失败记录不修改。
- rollout阶段只写official trajectory；机制阶段只读运动学与邻车context；机制失败时禁止运行representation。
- representation阶段比较old64/A/B/C/ego13/C-neighbor-zero，各自独立bandwidth/null；主端点只比较
  null-standardized ΔZ及log-cluster bootstrap，不比较raw MMD²。
- 全阶段不训练、不换checkpoint、不改seed/epoch/loss/architecture或既有benchmark。

## 3. 通过标准

- Stage6W-A：同池paired support均为n=400；analytic paired null与10000次swap验证相对误差可接受；
  B/C driver在raw和context-balanced口径均可审计。
- roster：80个token；v1/v2 development/v2 confirmation token重叠为0；selected official runnability=100%；
  freeze manifest SHA=`7105940bd822f02d643ed4f5cb9a8321b3827ca6117be289914057e3fe8a26c6`。
- rollout：`SUCCEEDED=80, FAILED=0, PENDING=0`，160条planner rollout、strict token和same-log审计通过。
- mechanism：mean-speed/RMS-accel控制门禁通过，至少两项interaction指标通过；实际四项全部通过。
- interaction增量：只有C-full减C-neighbor-zero的cluster bootstrap 95% CI下界>0才通过；实际
  `ΔZ=-7.852, CI=[-33.393,29.219]`，因此正式结果为不通过，不得解释成增量interaction证据。
- 最终状态=`FROZEN_STAGE6W_STAGE6S_V3_COMPLETE_NO_NEW_CHECKPOINT`，Stage6V联合模型结论不变。

## 统一BDD Evaluation Matrix与Style Report Card

### 1. 命令

冻结后的训练比较试验可直接生成统一报告（只读取锁定的CSV/JSON，不会重新跑BDD）：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation
python tools/build_unified_bdd_posttraining_report.py \
  --output-dir outputs/unified_bdd_posttraining_report_v1
```

若目标目录已存在且非空，工具会拒绝覆盖；请使用新的版本目录，不要覆盖已冻结报告。

所有后续BDD导出器和人工报告必须读取以下冻结定义：

```text
configs/unified_bdd_reporting_schema_v1.json
configs/unified_bdd_stage_task_mapping_v1.csv
docs/unified_bdd_evaluation_matrix_style_report_card_zh.md
```

不得为了填满统一矩阵自动启动训练、仿真、embedding导出或重算已冻结BDD。

### 2. 期望行为

- 固定输出13个behavior dimensions；缺失行保留N/A和reason code；
- 每个BDD行显式记录Reference、Target、task、paired/unpaired、representation和null/calibration；
- semantic delta统一为Target减Reference，并与BDD并列报告；
- 业务行为变化写入表A Behavior Profile；representation能力写入表B Evaluator Scorecard；
- Stage6J/K、Stage6P、Stage6S-v3、Stage6W和Stage7历史结果只做schema映射，不修改原统计值。
- 本命令生成`behavior_drift_profile.csv`（表A）、`representation_scorecard.csv`（表B）、
  `evidence_gap_matrix.csv`、中文总报告和带输入/输出SHA256的manifest；不会读取embedding、Waymo test或nuPlan rollout。

### 3. 通过标准

- 报告能直接回答Reference/Target、行为维度、显著性、semantic方向、最大差异task、可靠representation和paired/unpaired来源；
- BDD显著但缺少semantic delta时，Direction必须为N/A；
- 不使用overall semantic delta冒充task-specific方向；
- 不跨representation比较raw MMD²；
- 同一task-level BDD映射多个semantic维度时共享`parent_bdd_result_id`，不重复计作独立检验；
- 运行成功时输出状态必须为`FROZEN_UNIFIED_BDD_POSTTRAINING_REPORT_COMPLETE`，表A为13行、表B为5行；
- schema冻结状态保持`UNIFIED_BDD_REPORTING_SCHEMA_FROZEN`。

## 固定维度BDD标准化对比矩阵（冻结checkpoint后的描述性补齐）

### 1. 命令

先只读检查冻结资产、任务成员、old64与A/B/C checkpoint SHA，不导出embedding：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation
waymo_dev/bin/python tools/build_standardized_fixed_dimension_bdd_matrix.py \
  --preflight-only
```

在新的、尚不存在的输出目录中构建完整矩阵：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation
waymo_dev/bin/python tools/build_standardized_fixed_dimension_bdd_matrix.py \
  --output-dir outputs/standardized_fixed_dimension_bdd_matrix_v1
```

冻结协议位于：

```text
configs/standardized_fixed_dimension_bdd_protocol_v1.json
```

### 2. 期望行为

- 固定13个行为维度，以及old64/A/B/C/ego13五列；没有冻结有效证据的维度仍输出`N/A`，不删除行。
- 每一条BDD长表行分开记录三类Reference：Behavior Reference（Reference→Target）、该representation自己的Null Reference和old64 capability baseline。
- Stage6J/K完整保留`overall`、`following_interaction`、`longitudinal_high_motion`、`stop_go_control`及25/50/75/100% dose；Stage6S-v3保留相同80对的old64/A/B/C/ego13和C-neighbor-zero diagnostic。
- 使用既有Stage7 310对assertive/conservative rollout、固定pre-treatment task membership和primary seed 3407重新导出A/B/C/ego13 embedding；结果必须标记为`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`，不能替代Stage6V确认性端点。
- 写入`standardized_bdd_long.csv`、`fixed_dimension_primary_matrix.csv`、`representation_gate_scorecard.csv`、`evidence_gap_matrix.csv`、中文报告和带SHA256的manifest。
- 命令不会训练、重跑nuPlan、修改planner/checkpoint、选择新场景或改写Stage6V联合结论；也不会以raw MMD²跨representation排序。

### 3. 通过标准

- preflight返回`PREFLIGHT_PASS_STANDARDIZED_FIXED_DIMENSION_BDD_PROTOCOL`，并确认固定Stage7任务数量：following=60、lane_change=60、stop_go=67、high_motion=60、dense/vulnerable=63。
- 完整运行返回`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`；主矩阵为13行，审计长表包含Stage6J/K、Stage6S-v3和Stage7的逐representation行。
- 同一跟车工况在长表中为60个场景/52个log，并为每个representation同时给出raw MMD²、null q95、BDD/null-q95 ratio、Z_BDD、raw/Holm p及semantic delta。
- `LON.CAR_FOLLOWING`如果只绑定speed/accel语义，direction必须为`TARGET_MORE_ACTIVE_FOLLOWING`，不得写`CLOSER`。
- Stage6S-v3的front-gap/THW、closing与following子行共享同一个`parent_bdd_result_id`，不得计为多次独立BDD检验；C-neighbor-zero仅为diagnostic。
- Stage6P/Waymo/Stage6J/K/interaction/Stage6V门禁必须在`representation_gate_scorecard.csv`拆成五个明确字段，不得再合并为模糊的`frozen_gate_result`。

## 最终标准化BDD Style Report Card（只读排版冻结）

### 1. 命令

该命令只读取已经冻结的v1矩阵CSV/JSON，并生成最终两层报告；不会读取checkpoint、embedding或rollout，也不会重算BDD：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation
waymo_dev/bin/python tools/build_standardized_fixed_dimension_bdd_matrix.py \
  --finalize-existing-dir outputs/standardized_fixed_dimension_bdd_matrix_v1 \
  --final-output-dir outputs/final_standardized_bdd_style_report_card_v1
```

若输出目录已经存在，工具会拒绝覆盖。正式定义位于：

```text
configs/unified_bdd_reporting_schema_v2.json
configs/standardized_fixed_dimension_bdd_protocol_v2.json
```

### 2. 期望行为

- 校验既有`standardized_bdd_long.csv`、主矩阵、门禁表和evidence-gap表的冻结SHA256；任何源文件变化立即失败。
- 原13维和全部统计值原样保留；free-flow、lane-keeping、lateral-gap继续为N/A。
- 最终第一页使用`Primary Representation = B`回答Behavior Reference→Target的行为变化；第二页独立评价representation资格。
- 主矩阵列名为`该Treatment下最高标准化检测敏感度`，不出现`Best capability`。
- Stage6S-v3三条共享语义维度的所有表示单元格均带`†`；`final_shared_parent_bdd_audit.csv`按representation保留统一`parent_bdd_result_id`，三行只计一个独立检验。
- 只读取Stage6P已冻结n=400 detection/FPR以补充资格表；不修改Stage6V联合结论。

### 3. 通过标准

- 命令返回`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`与`statistics_recomputed=false`。
- `final_fixed_dimension_primary_matrix.csv`恰好13行，shared-parent维度恰好3行，N/A维度恰好3行。
- 跟车保持60 scenario / 52 log；变道保持60场景且为`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`；Stage6S-v3保持80 pair / 11 log。
- B的冻结摘要保持：跟车`1.72× / Z=5.25`、变道`2.50× / Z=9.12`、纵向`2.74× / Z=10.33`、interaction `7.39× / Z=30.60 †`。
- manifest明确`training_run=false`、`simulation_run=false`、`embedding_export_run=false`、`statistics_recomputed=false`。

## Stage7L-A2 Pure-Lateral清洁实现与Smoke（已冻结）

### 1. 命令

运行新增单元测试：

```bash
cd /Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_stage7l_pure_lateral_execution.py \
  tests/test_stage7l_opportunity_inventory.py
```

重建pre-treatment map opportunity inventory：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_build_lane_change_opportunity_inventory.py \
  --inventory_inputs outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh/scenario_inventory_inputs.csv \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --stage7_lane_change_roster outputs/stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv \
  --stage7p_root outputs \
  --output_dir outputs/stage7l_a2_lane_change_opportunity_inventory_v1 \
  --candidates_per_db 8 --stop_after_eligible 160
```

冻结一个明确标记为A2 smoke-only的maneuver后，可用`tools/stage7l_run_official_smoke.py`运行同token五档official simulation。最终机制及安全检查命令为：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_evaluate_lateral_mechanism.py \
  --trajectory_csv outputs/stage7l_a2_official_smoke_v2_safe_final4/stage7c_output/simulated_ego_trajectory.csv \
  --maneuver_manifest outputs/stage7l_a2_official_smoke_v2_safe/stage7l_a2_smoke_maneuver_manifest.json \
  --official_runs_root outputs/stage7l_a2_official_smoke_v2_safe_final4/stage7c_output/official_nuplan_runs \
  --output_dir outputs/stage7l_a2_lateral_mechanism_v2_safe_final_with_safety
```

### 2. 期望行为

- external planner在冻结source/target lane和共同canonical source-lane progress上生成五档轨迹；dose只改变60/54/48/42/36 m横向transition length。
- official simulation固定为`closed_loop_nonreactive_agents`；五档共享initial state、route、trigger、纵向目标速度/加速度约束和background配置。
- inventory会区分tagged token的官方-3 s extraction offset与untagged/default token，并冻结official真实首帧fingerprint。
- smoke token在任何rollout前写入prior exclusion ledger，未来不得进入confirmation。
- mechanism工具只读trajectory和official collision/drivable-area metric，不读取checkpoint、embedding或BDD。
- 本流程不建立Stage7L-B development roster，不运行formal development或confirmation。

### 3. 通过标准

- 单元测试通过quintic边界、dose顺序、manifest identity、canonical progress逐点一致、dynamic consistency和native map adjacency。
- official smoke必须5/5成功；五档maneuver SHA、canonical generator SHA一致，`s_route(t)`逐点完全相同。
- 五档轨迹均valid、完成换道、无责任碰撞、drivable-area compliant，且横向峰值加速度随dose严格递增。
- realized longitudinal nuisance不得出现明显分叉；最终最大绝对值为mean speed 0.005553 m/s、RMS accel 0.000187 m/s²、RMS jerk 0.002776 m/s³、route progress 0.051174 m。
- 排除旧Stage7/Stage7P及全部A2 smoke token后，fresh eligible必须≥104；最终为148 token / 120 log、left/right=25/123，严格log-disjoint 24+80分配可行。
- 通过后状态仅升级为`STAGE7L_PURE_LATERAL_IMPLEMENTATION_CLEAN`和`STAGE7L_B_DEVELOPMENT_AUTHORIZED`；不得自动进入Stage7L-B。

## Stage7L-B Pure-Lateral Development（当前已执行）

### 1. 最终安全版参数

```text
dose0/25/50/75/100 transition length = 60/58.5/57/55.5/54 m
trigger = 12 m
planner horizon = 0.4 s
scenario horizon = 15 s（tagged与untagged统一）
background = closed_loop_nonreactive_agents
```

参数通过`configs/stage7l_hydra/planner/stage7l_b2_pure_lateral_dose*.yaml`切换；只改变横向transition length。

### 2. Full-development运行

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_run_development.py \
  --maneuver_manifest outputs/stage7l_b_final_development_freeze_v1/final_development_maneuver_manifest.json \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --nuplan_data_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset \
  --nuplan_exp_root "$PWD/outputs" \
  --nuplan_devkit_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  --tuplan_garage_root /Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage \
  --stage7c_tool tools/stage7c1_run_nuplan_simulation.py \
  --python_executable /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  --planner_prefix stage7l_b2_pure_lateral \
  --output_dir outputs/stage7l_b_full_development_v1
```

### 3. 机制与dose-response分析

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_evaluate_lateral_mechanism.py \
  --trajectory_csv outputs/stage7l_b_full_development_v1/stage7c_output/simulated_ego_trajectory.csv \
  --maneuver_manifest outputs/stage7l_b_final_development_freeze_v1/final_development_maneuver_manifest.json \
  --official_runs_root outputs/stage7l_b_full_development_v1/stage7c_output/official_nuplan_runs \
  --output_dir outputs/stage7l_b_full_development_mechanism_v1

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_analyze_development_dose_response.py \
  --mechanism_metrics_csv outputs/stage7l_b_full_development_mechanism_v1/stage7l_a2_lateral_mechanism_metrics.csv \
  --roster_csv outputs/stage7l_b_final_development_freeze_v1/final_development_roster.csv \
  --run_summary outputs/stage7l_b_full_development_v1/stage7l_b_development_run_summary.json \
  --freeze_summary outputs/stage7l_b_final_development_freeze_v1/refined_development_roster_freeze_summary.json \
  --output_dir outputs/stage7l_b_development_analysis_v1
```

### 4. 当前门禁

- 120/120 official success、120/120 completion、0 off-road、canonical纵向一致性通过。
- 4个场景在五档均发生责任碰撞；不是dose-dependent，但使safety feasibility gate失败。
- 静态规则后剩余83 token / 67 log / 15 left / 68 right；动态15 s traffic-clearance规则尚未重扫。
- 状态：`STAGE7L_B_DEVELOPMENT_NOT_READY_FOR_FREEZE`；不得建立Stage7L-C roster或运行confirmation。
- 本流程禁止并未执行embedding、BDD、MMD或model training。

## Stage7L-B2 动态预处理交通净空与库存扩展（已完成）

### 1. 命令

对已有development roster进行纯replay动态审计：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_audit_dynamic_lane_change_clearance.py \
  --candidate_csv outputs/stage7l_b_final_development_freeze_v1/final_development_roster.csv \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --output_dir outputs/stage7l_b2_dynamic_clearance_development_audit_v1
```

扩大Pittsburgh inventory：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_expand_clean_opportunity_inventory.py \
  --inventory_inputs outputs/stage7p_expanded_scenario_inventory_v2_pittsburgh/scenario_inventory_inputs.csv \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --stage7_lane_change_roster outputs/stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv \
  --stage7p_root outputs \
  --additional_exclusion_ledger outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv \
  --candidates_per_db 24 \
  --required_map_name us-pa-pittsburgh-hazelwood \
  --output_dir outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh
```

### 2. 期望行为

- 只读取原始map、route、official initial state和`lidar_box` replay track；绝不读取planner rollout、dose ID、collision outcome、embedding、BDD或MMD。
- 使用15 s / 0.1 s共同canonical longitudinal schedule。触发前仅检查source lane；54–60 m family期间检查source→target共同strip；之后检查target lane。
- replay track按timestamp线性插值，最大允许间隔0.25 s；全局轨迹时域不足15 s返回`INSUFFICIENT_TRACK_HORIZON`。
- 输出Pool A（scenario-disjoint）和Pool B（与所有Stage7L-B development log严格分离），但不会选择、冻结或运行任何confirmation roster。

### 3. 通过标准

- 单元测试覆盖time alignment、远离包络、未来transition冲突、方向对称、buffer边界、missing track和dose independence。
- development audit以同一算法识别4/4固定collision case，且无token hardcode。
- Pittsburgh Pool B需≥120 dynamic-clean token、≥80 unique log、左右方向均有真实供给，并且official runnability为100%。
- 当前实测：Pool B为152 token / 94 log / 19 left / 133 right；满足上述门槛。
- 状态仅为`STAGE7L_B2_DYNAMIC_CLEARANCE_COMPLETE`和`STAGE7L_C_PROTOCOL_FREEZE_RECOMMENDED`，不得自动进入Stage7L-C。

## Stage7L-C 前瞻性 Protocol 与 80 场景 Confirmation Roster Freeze

### 1. 命令

冻结protocol、80场景roster与盲测授权；此命令只读取Pool B、ledger、map/DB官方scene query和冻结checkpoint SHA，不构建planner rollout：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_freeze_confirmation_roster.py \
  --protocol_config configs/stage7l_c_prospective_confirmation_protocol_v1.json \
  --pool_b outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh/pool_b_strict_development_log_disjoint_dynamic_clean.csv \
  --development_ledger outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --nuplan_devkit_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  --output_dir outputs/stage7l_c_confirmation_freeze_v1 \
  --authorization_manifest docs/stage7l_c_blind_confirmation_authorization_manifest_v1.json
```

随后验证冻结可重放性和全部门禁：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit:/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage:$PWD \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_validate_confirmation_freeze.py \
  --protocol_config configs/stage7l_c_prospective_confirmation_protocol_v1.json \
  --pool_b outputs/stage7l_b2_dynamic_clearance_expanded_inventory_v2_pittsburgh/pool_b_strict_development_log_disjoint_dynamic_clean.csv \
  --development_ledger outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv \
  --freeze_dir outputs/stage7l_c_confirmation_freeze_v1 \
  --authorization_manifest docs/stage7l_c_blind_confirmation_authorization_manifest_v1.json \
  --output_json outputs/stage7l_c_confirmation_freeze_v1/confirmation_freeze_validation.json
```

### 2. 期望行为

- 固定5档`60/58.5/57/55.5/54 m`、trigger 12 m、15 s horizon与non-reactive replay background。
- 只从B2 Pool B选取80个候选，固定`15 left + 65 right`。left供给只有14个log，因此一处预先记录的left log重用不可避免；right优先不重用log。
- 对选中的80个token再次执行official nuPlan scene query/boundary一致性审计，输出runnability audit、selection trace、geometry summary、roster、maneuver manifest和reserve inventory。
- 不运行Stage7L-D rollout，不导出embedding，不计算BDD/MMD，不产生representation结果。

### 3. 通过标准

- `N=80`、left/right=`15/65`、duplicate token=`0`；与所有历史token scenario-disjoint，且与26个Stage7L-B development log严格分离。
- `official runnable=80/80`、dynamic clearance=`80/80`、static eligibility=`80/80`、source/target/trigger manifest完整=`80/80`。
- 选择trace必须由`Pool B + config + seed=620271`重放；reserve明确不是运行失败后的replacement pool。
- 只有通过后才可记录`STAGE7L_C_PROSPECTIVE_PROTOCOL_FROZEN`、`STAGE7L_C_CONFIRMATION_ROSTER_FROZEN`和`STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED`；仍不得自动启动Stage7L-D。

## Stage7L-C1 Protocol Consistency Amendment验证

### 1. 命令

只读验证C1 protocol、盲测授权、原80场景roster和development-disjoint不变性：

```bash
python tools/stage7l_validate_c1_amendment.py
```

### 2. 期望行为

- 只读取protocol/manifest、原roster、development exclusion ledger和原freeze summary。
- 检查`N_design=80`与逐dose `N_pair`定义、Primary pair下限76、B完整dose curve、单一39-test Holm family及Primary排除标记。
- 检查roster SHA、80/15/65/79 logs、scenario/log overlap、dose/trigger/eligibility/gates/checkpoint/Primary科学定义均未改变。
- 不启动Stage7L-D，不读取或生成rollout、embedding、BDD/MMD，不训练模型。

### 3. 通过标准

- 输出状态`STAGE7L_C1_PROTOCOL_CONSISTENCY_AMENDMENT_FROZEN`。
- roster SHA仍为`90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9`，N/left/right/log=`80/15/65/79`，development scenario/log overlap均为0。
- minimum complete与Primary minimum pair均为76；secondary test count为39；Primary不进入secondary Holm。
- amended protocol与blind authorization SHA互相绑定，`stage7l_d=NOT_STARTED`。

## Stage7L-C2 Task-Population Consistency Amendment验证

### 1. 命令

只读重放pre-treatment task mask并验证最终C2机器协议：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_validate_c2_amendment.py
```

如需单独导出可审计mask，只允许使用冻结roster与pre-treatment Pool B：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_generate_pretreatment_task_masks.py \
  --output-csv /path/to/stage7l_c2_pretreatment_task_masks.csv
```

### 2. 期望行为

- `LAT.LANE_CHANGE`直接等于完整冻结roster membership；不读取expert/treatment outcome。
- `LAT.DYNAMICS`只根据`official_scenario_types_json`是否命中冻结high-motion标签生成。
- 检查Primary与理论矩阵对应格定义SHA相同、Primary只排除一次、40格理论矩阵与39格Holm family一致。
- 检查不可计算cell固定raw p=1且仍留在family，小样本可计算cell不新增门槛。
- 不运行Stage7L-D、planner rollout、embedding、BDD/MMD或训练。

### 3. 通过标准

- 输出`STAGE7L_C2_TASK_POPULATION_CONSISTENCY_AMENDMENT_FROZEN`。
- task mask重放为`LAT.LANE_CHANGE=80/80`、`LAT.DYNAMICS=38/80`且SHA匹配C2 manifest。
- roster仍为80/15/65/79 logs且SHA不变；dose、gates、failure policy、checkpoint和Primary统计规则均不变。
- `theoretical_cells=40`、`secondary_cells=39`、`stage7l_d_started=false`。

## Stage7L-D 一次性 Planner-Level Confirmation（已通过并停止）

### 1. 命令

第一条rollout前只做SHA/roster/环境验证并预建400格账本：

```bash
PYTHONPATH=../nuplan-devkit:../tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_run_confirmation.py \
  --prepare_only \
  --output_dir outputs/stage7l_d_one_time_confirmation_v1
```

正式执行或中断后确定性resume：

```bash
PYTHONPATH=../nuplan-devkit:../tuplan_garage:$PWD \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_run_confirmation.py \
  --output_dir outputs/stage7l_d_one_time_confirmation_v1
```

400格结束后，仅提取planner trajectory/official safety并运行冻结门禁：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_extract_confirmation_metrics.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/stage7l_evaluate_confirmation_gates.py
```

### 2. 期望行为

- preflight精确验证protocol/authorization/roster/planner/dose config SHA，以及80场景、15/65、79 logs、development零重叠和80/80 official runnable。
- 预先写入400个固定cell；运行顺序为roster collection order×dose0/25/50/75/100。每个attempt单独保留，成功结果不会被覆盖；只允许同cell基础设施重试，不允许replacement或结果性重跑。
- gate evaluator只读取official rollout、trajectory mechanism、nuisance、safety和canonical identity；绝不加载checkpoint/embedding，不计算BDD/MMD。
- safety denominator固定为全部80场景；五档均成功/完成才算scenario success/completion，任一档off-road或责任碰撞即计为该scenario发生。

### 3. 通过标准

- `planned_rollout_ledger.csv`恰好400行数据，`N_complete_all_five_doses>=76`。
- dose100−dose0：duration median<0且一致性≥70%；RMS lateral accel与peak yaw median>0且一致性各≥80%。
- mean speed、RMS longitudinal accel、RMS longitudinal jerk、route progress的median absolute与p90均不超过冻结门槛。
- 80场景口径official success/completion≥95%，off-road/责任碰撞≤5%，canonical longitudinal identity无mismatch。
- 全部门禁通过才写`STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`；否则写`...NOT_UNLOCKED`。本命令永远不自动执行Stage7L-E。

### 4. 冻结结果

- 80场景、400计划格；400/400 official rollout成功，80/80场景五剂量完整，各dose均80/80，replacement=0。
- dose100−dose0：duration `−0.200160 s / 88.75%`，RMS lateral accel `+0.055832 m/s² / 100%`，peak yaw `+0.014404 rad/s / 96.25%`；三项mechanism PASS。
- 四项longitudinal nuisance PASS；scenario-level official success/completion `100%/100%`，off-road `2.5%`，responsible collision `1.25%`，safety PASS。
- canonical identity `80/80`、mismatch `0`；总状态`STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED`。
- 仅解锁`STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED`；Stage7L-E尚未执行，embedding/checkpoint/BDD/MMD均未读取或计算。
- 详见`docs/stage7l_d_one_time_planner_confirmation_report_zh.md`和`docs/stage7l_d_confirmation_manifest_v1.json`。

## Stage7L-E E1 输入/推理/统计执行冻结（正式BDD未运行）

### 1. 命令

准备五档冻结Stage7C视图并重放C2 task mask：

```bash
waymo_dev/bin/python tools/stage7l_e_prepare_input_contract.py \
  --output-dir outputs/stage7l_e_prospective_bdd_v1
```

对`dose0/dose25/dose50/dose75/dose100`逐档复用既有Stage5D builder：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7l_e_prospective_bdd_v1/stage7c_views/<dose> \
  --output_dir outputs/stage7l_e_prospective_bdd_v1/contexts/<dose> \
  --nuplan_map_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps \
  --nuplan_db_root /Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data/cache/locked_pool_expanded_v1 \
  --require_nonzero_neighbor_coverage --slot_sanity_min_coverage 0.06
```

E1测试：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_stage7l_e_prospective_bdd.py \
  tests/test_stage7l_c2_task_masks.py
```

### 2. 期望行为

- 只读取冻结的400条Stage7L-D official rollout，不调用nuPlan simulation。
- 生成五档各80行、`T=150`、83D的context；149步只右补零且新增mask=false。
- task mask固定为80个`LAT.LANE_CHANGE`和38个`LAT.DYNAMICS`。
- 新增正式工具已经绑定old64/A/B/C/ego13、100,000次pair swap、plus-one p和固定39格Holm，但E1不读取checkpoint、不导出embedding、不计算BDD。

### 3. 通过标准

- `input_contract_audit.json`状态为`STAGE7L_E_INPUT_CONTRACT_VALIDATED`，五档shape均为`[80,150,83]`且finite。
- 五档builder structural validation均PASS，不删除collision/off-road场景。
- pytest通过；实现manifest状态为`FROZEN_READY_FOR_STAGE7L_E_PROSPECTIVE_BDD_EXECUTION_NOT_RUN`。
- E2单独执行前，`stage7l_e_final_decision.json`、正式embedding和BDD结果均不存在。

## Stage7L-E E2 正式BDD与机器结果冻结（已完成）

### 1. 命令

正式一次性推理与40格BDD：

```bash
waymo_dev/bin/python tools/stage7l_e_run_prospective_bdd.py \
  --prepared-dir outputs/stage7l_e_prospective_bdd_v1 \
  --context-root outputs/stage7l_e_prospective_bdd_v1/contexts \
  --output-dir outputs/stage7l_e_prospective_bdd_v1
```

只读审计并冻结机器结果：

```bash
waymo_dev/bin/python tools/stage7l_e_freeze_machine_results.py \
  --result-dir outputs/stage7l_e_prospective_bdd_v1
```

### 2. 期望行为

- 只读取E1冻结context、old64/A/B/C primary seed 3407 checkpoint及ego13 scaler；不调用nuPlan、不训练、不改checkpoint。
- 计算5 representations×4 nonzero doses×2固定task共40格；Primary固定为B-3407、dose100 vs dose0、`LAT.LANE_CHANGE`。
- null固定100,000次same-scenario within-pair label swap；Primary独立报告，剩余39格组成唯一secondary Holm family。
- machine freeze只审计E2已有结果并写小型JSON/CSV与中文摘要，不重算统计。

### 3. 通过标准与冻结结果

- 40格齐全；`LAT.LANE_CHANGE`每格80 pair，`LAT.DYNAMICS`每格38 pair；39个Holm检验、20个low-N diagnostic、0个不可计算格。
- Primary raw MMD²=`0.001075040606`、null q95=`0.002466807391`、ratio=`0.435802410`、`Z=-0.065036660`、raw p=`0.411905881`，状态必须为`STAGE7L_E_PRIMARY_BDD_FAILED`。
- dose100 lane-change中old64/A/B/C均未检出；ego13=`13.087068× / Z=40.201025 / p=9.9999e-06`。
- 总状态为`STAGE7L_E_MACHINE_RESULTS_FROZEN_READY_FOR_E3_REPORTING`；不得因Primary失败调整模型、task、roster、null或门槛。

## Stage7L-E E3 报告与论文证据整合（已完成）

### 1. 命令

```bash
waymo_dev/bin/python tools/stage7l_e_finalize_reporting.py
```

### 2. 期望行为

- 逐值继承E2冻结CSV/JSON，仅生成中文完整报告、最终manifest和13维Style Report Card addendum。
- `LAT.LANE_CHANGE`与`LAT.DYNAMICS`主展示采用Stage7L prospective dose100；旧Stage7 60场景post-hoc证据另表保留。
- 固定taxonomy、Stage6V联合结论、checkpoint、planner、scenario和所有统计值不变；不运行训练、仿真、embedding或BDD重算。
- 若默认输出目录已经存在，工具拒绝覆盖；复核时应给新的临时`--output-dir`并比较哈希。

### 3. 通过标准

- `stage7l_e_prospective_bdd_long.csv`为40行，整合主矩阵保持13维，历史Stage7 post-hoc lateral evidence非空且身份未改变。
- Primary仍精确等于E2的B-3407失败结果；`statistics_recomputed=false`、`stage6v_joint_conclusion_modified=false`。
- 最终状态同时为`STAGE7L_E_PROSPECTIVE_REPRESENTATION_EVALUATION_COMPLETE`与
  `STAGE7L_E_PROSPECTIVE_EVIDENCE_INTEGRATED_FOR_THESIS`。
- 权威输出：`docs/stage7l_e_prospective_representation_bdd_report_zh.md`、
  `docs/stage7l_e_prospective_bdd_manifest_v1.json`和
  `outputs/final_standardized_bdd_style_report_card_v2_stage7l/`。

## StageR / R1 Phase B0 合同兼容性审计

### 1. 命令

```bash
waymo_dev/bin/python tools/r1_phaseb0_compatibility_audit.py \
  --output /new/path/r1_phaseb0_compatibility_results_v0.1.json
waymo_dev/bin/python -m unittest tests.test_r1_phaseb0_compatibility -v
```

### 2. 期望行为

- 只读取冻结 HLC/TSB mechanism contract 与 treatment-independent raw-scale evidence。
- 只构造平行车道和分段加速度合成 fixture；不选择真实 scenario，不运行 planner/smoke，不读取 representation、BDD、probe、checkpoint 或 RBR。
- 第一条命令拒绝覆盖已存在的输出；第二条用 mock ID 验证 baseline reuse 与 48-call 构造前 hard cap。

### 3. 通过标准

- HLC synthetic witness 同时为 `HLC_MECHANISM_PAIR_PASS` 和 `F_MATCH_PASS`，兼容性记录为 `MARGINALLY_FEASIBLE`。
- 三个 `TSB_GEN_V2_OPTION_*` 均同时通过冻结 mechanism/F_match，且仍标记 `PROPOSED_NOT_FROZEN`。
- 每 family 精确 24 次、总计 48 次 core construction；第 49 次 claim 必须在构造前抛错，计数保持 48。
- unit tests 全部通过；不得产生任何真实 rollout、smoke metrics 或 roster。

## StageR / R1 Phase B1 科学修订与冻结准备

### 1. 只读/合成准备命令

以下工具只生成 HLC synthetic design、SQLite/map inventory 与 fresh source universe，不运行 nuPlan simulation 或 roster selection；正式输出已存在时会拒绝覆盖：

```bash
waymo_dev/bin/python tools/r1_phaseb1_freeze_preparation.py
waymo_dev/bin/python tools/export_r1_official_nuplan_db_inventory_csv.py
```

### 2. 48-call preflight 与静态检查

```bash
waymo_dev/bin/python -m unittest \
  tests.test_r1_phaseb0_compatibility \
  tests.test_r1_phaseb1_freeze_preparation -v
waymo_dev/bin/python -m py_compile \
  tools/r1_phaseb1_freeze_preparation.py \
  tools/export_r1_official_nuplan_db_inventory_csv.py \
  tools/stageR_execute_r1_technical_smoke.py
waymo_dev/bin/python tools/check_no_tmp_dependencies.py
```

### 3. 通过标准与边界

- HLC A/B/C 在 12-cell synthetic speed/lane-width 包络内均通过冻结 mechanism、新三项 Primary F_match 与既有 engineering limits，但必须保持 `PROPOSED_NOT_FROZEN`。
- inventory 必须为 1,624 个非零、SQLite read-only 可读且 map-compatible 的 DB；fresh source universe 只可为 `READY_FOR_OUTCOME_BLIND_SELECTION`，不得包含实际 roster。
- replay 固定 `MASTER_SEED=2026082701`；background determinism 未实证时，总状态保持 `VERSION_AMBIGUOUS`。
- executor 必须精确计划 48 次 core construction，逐次 pre-call claim；重复 baseline、未计划调用或第 49 次调用均在计数递增前 fail-closed。
- 本阶段禁止运行新 technical smoke、真实 planner rollout、representation/BDD/probe/RBR 读取或训练。

## StageR / R1 Phase B1.1 官方 runtime determinism 核验

### 1. 命令

以下命令先只读地以已冻结 salt 选择四个 runtime-only 场景；正式 roster 已存在时工具会拒绝覆盖。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_prepare_runtime_determinism_roster.py
```

以下命令是唯一获授权的官方执行入口：精确运行四个冻结场景各两次。它只能执行 HLC 的
`DECISIVE_MONOTONIC_LANE_CHANGE` baseline 和 TSB 的 `SINGLE_CONTINUOUS_BRAKING` baseline；不得加 treatment。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_run_runtime_determinism_validation.py
```

### 2. 期望行为

- selector 先读取 owner-bound salt、fresh source universe、old12 blacklist、历史 technical-smoke roster 与 R4 freeze；只使用 SQLite/地图/官方初始状态等 pre-treatment 信息，选择 2 个 R-HLC 与 2 个 R-TSB，四个 token/log 均唯一并永久隔离。
- executor 在每一次调用官方 nuPlan 前写入 `OFFICIAL_CLOSED_LOOP_RUN` claim；总 cap 固定为 8，第 9 次 pre-run claim 必须在启动 simulator 前拒绝。
- 每个 RUN_A/RUN_B 写入稳定的 history、raw/canonical context、traffic-light、background tracks、ego、planner-output 与官方 collision/drivable metric 的哈希摘要。不会读取 representation、BDD、probe、checkpoint 或 RBR，也不会输出科学 outcome 结论。
- 原始 trace、nuPlan logs 和 metrics 只保留在 `outputs/r1_runtime_determinism_validation_v1/`，不应提交；小型最终结果写入 `docs/stageR/r1/r1_runtime_determinism_result_v1.0.json`。

### 3. 通过标准

- roster 精确为 R-HLC=2、R-TSB=2、total=4、unique_logs=4，并标记 `RUNTIME_DETERMINISM_VALIDATION_ONLY` 及三个永久排除标签。
- 官方 run 数必须精确为 8；每对 15 类比较均为 exact canonical equality，浮点没有人为 tolerance；collision 与 off-road/drivable 官方 metric 均必须存在。
- 四对均一致才可将 `BACKGROUND_REPLAY_DETERMINISM` 设为 `VERIFIED_ON_BOUND_RUNTIME`、将 `OFFICIAL_REPLAY` 设为 `READY_FOR_TECHNICAL_SMOKE_REVIEW`。任一 technical failure 或不一致即 `NOT_VERIFIED/NOT_READY`，不得第三次重跑或换场景。
- 即使通过，也只满足 48-call technical smoke 的 owner-review 条件；48-call smoke 仍需独立 scientific-owner authorization。

## StageR / R1 Phase B1.2 V2 零预算接口预检

### 1. 命令

在申领任何 V2 官方闭环额度前，使用以下命令检查修复后的 planner 是否满足 nuPlan 1.2.2 的
`AbstractPlanner` 可调用接口。该命令只构造两个内存 mock 输入（R-HLC、R-TSB 各一个），不会读取
scenario DB、启动仿真或写入 V2 正式运行目录。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_runtime_determinism_v2_interface_preflight.py
```

### 2. 期望行为

- 比较 `name()`、`observation_type()`、`initialize()`、`compute_planner_trajectory()`、`compute_trajectory()` 和
  `generate_planner_report()` 的参数契约，并确认 planner 是 `AbstractPlanner` 子类。
- 对冻结 roster 的一个 HLC 和一个 TSB 条目完成单步内存调用，输出 trace/binding 的摘要哈希；不消耗
  `OFFICIAL_CLOSED_LOOP_RUN` 额度。
- 早期 v1.0/v1.1 预检均为零预算诊断；当前完整 V2 执行器 binding 对应的最终预检为
  `docs/stageR/r1/r1_runtime_determinism_v2_interface_preflight_v1.2.json`；
  `PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED` 才允许启动 V2 执行器。

### 3. 通过标准

- 全部接口参数契约精确匹配、两种 family mock 均产生非空 trajectory 与一个 planner report runtime sample。
- `official_closed_loop_runs_claimed=0`、`official_closed_loop_runs_started=0`、`budget_consumed=0`。

## StageR / R1 Phase B1.2 V2 官方 runtime determinism 核验

### 1. 命令

以下是绑定 V2 授权、最终零预算 preflight 与原四行冻结 roster 的唯一执行入口。它只会为每个
scenario 运行 `V2_RUN_A` 和 `V2_RUN_B`，总计精确 8 次；不得以任何理由替换 roster 或增加运行。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_run_runtime_determinism_validation_v2.py \
  --preflight docs/stageR/r1/r1_runtime_determinism_v2_interface_preflight_v1.2.json
```

### 2. 期望行为

- 每次 nuPlan 调用前将 `V2_CLAIMED_BEFORE_SIMULATION` 写入独立账本；第九次 pre-run claim 在 simulator
  启动前拒绝。V1 的历史技术失败不计入 V2 账本。
- 对四个 A/B pair 的 15 个冻结类别使用 exact canonical equality，不设浮点 tolerance；不一致时报告
  max-absolute diagnostic、first differing step 与 affected fields，但不把该 diagnostic 当作通过阈值。
- 原始 trace、logs、metrics 仅在 `outputs/r1_runtime_determinism_validation_v2/`，不提交；小型结果为
  `r1_runtime_determinism_result_v2.0.json`、比较 CSV 与 V2 ledger CSV。

### 3. 通过标准

- 8/8 完成、全部四个 pair 的 15 类比较精确相等、collision 与 off-road/drivable metric 均存在，且第九次
  pre-run claim 被拒绝，才是 `VERIFIED_ON_BOUND_RUNTIME / READY_FOR_TECHNICAL_SMOKE_REVIEW`。
- 任一技术失败、缺 artifact、A/B 不相等或预检 binding 不一致即立即 fail-closed，保持
  `NOT_VERIFIED / NOT_READY`；不得重跑、调参或转入 48-call smoke。

## StageR / R1 Phase B1.3 官方 Parquet metric canonicalization

### 1. 命令

先以合成 Parquet fixture 和 V2 已有 output 做零预算 parser preflight；该命令绝不启动 official simulation。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_official_metric_canonicalizer_preflight.py
```

单独解析一个已完成 official run 的两份冻结 metric 时，使用：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_official_metric_canonicalizer.py \
  --run-dir /path/to/official_run \
  --output /new/path/canonical_metric_payload.json
```

### 2. 期望行为

- 只接收恰好一个 `no_ego_at_fault_collisions.parquet` 与一个
  `drivable_area_compliance.parquet`，并按 Stage7L 已有字段语义输出小型 canonical JSON payload。
- Parquet file SHA 仅作为 provenance；V3 primary replay comparison 只比较 collision/drivable canonical payload
  的 exact JSON equality。
- preflight 覆盖两组有效 fixture 和 missing/duplicate/missing-column/empty-table 的 fail-closed fixture，并只把
  V2 既有 run 用于 path/schema/column compatibility，不解释其 metric 数值。

### 3. 通过标准

- `r1_official_metric_canonicalizer_preflight_v1.0.json` 为
  `PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED`，同时所有无效 fixture 都在预期位置拒绝。
- 每次 future official run 缺少、重复、不可读、空表、多 row、列缺失或类型不兼容时，parser 必须在 comparison
  前报 `TECHNICAL_FAILURE`；不得以 trace 或 Parquet container SHA 替代 canonical metric comparison。

## StageR / R1 Phase B1.3 V3 官方 runtime determinism 核验

### 1. 命令

仅在 canonical metric parser preflight 已通过且 V3 authorization binding 完整时，使用以下唯一入口。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_run_runtime_determinism_validation_v3.py
```

### 2. 期望行为

- 原四行 frozen roster 各执行 `V3_RUN_A/B`，总计精确 8 个 `OFFICIAL_CLOSED_LOOP_RUN`；每次 simulation 前写 `V3_CLAIMED_BEFORE_SIMULATION`，第九次在启动前拒绝。
- 仅运行 HLC decisive baseline 和 TSB single-continuous-braking baseline。每个 run 必须有 trace、binding、恰好一份两类官方 Parquet，并由 canonicalizer 成功解析；任一失败立即停止整个 V3。
- 4 个 A/B pair 逐项比较 15 个冻结类别；collision/drivable primary 比较 canonical semantic payload，浮点仍不设 tolerance。原始 outputs/Parquet 永不提交。

### 3. 通过标准

- 仅当 8/8 run 成功、4/4 pairs 的 15/15 类别精确相等、canonicalization 完整、且第九次 claim 被拒绝时，才能写 `VERIFIED_ON_BOUND_RUNTIME / READY_FOR_TECHNICAL_SMOKE_REVIEW`。
- 即使通过，48-call smoke 仍是 `PENDING_SEPARATE_SCIENTIFIC_OWNER_AUTHORIZATION`；本命令不选择 smoke roster、不执行 smoke、treatment 或 RBR training。

## StageR / R1 Phase B2 一次性 fresh 官方合规技术 smoke

### 1. 命令

先冻结前瞻 scope、继承的 selector salt 和 fresh 24-scenario roster。该命令只读 SQLite、官方初始状态、route 与 native map，不启动仿真；输出文件已存在时会拒绝覆盖。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_prepare_official_technical_smoke_roster.py
```

在任何 official run claim 前执行零额度 preflight：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_run_official_compliant_technical_smoke_v1.py --mode preflight
```

仅在独立的一次性 authorization 已生成、SHA binding 全部匹配时，以下命令才可执行；不得使用历史
`tools/stageR_execute_r1_technical_smoke.py`。

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_run_official_compliant_technical_smoke_v1.py --mode execute
```

### 2. 期望行为

- selector 必须继承 `MASTER_SEED=2026082701` 和既有 salt SHA，不得重新生成 salt；按
  `SHA256(salt_sha256|family|scenario_token|log_id)` 选择 12 个 R-HLC 与 12 个 R-TSB，且 24 token/log 均唯一。
- 每个 scenario 仅构造一个 frozen baseline 与一个 frozen treatment：HLC 为 Option-B，TSB 为 Option-A；旧
  MILD/NOMINAL/STRONG candidate 一律不可执行。
- executor 只使用 V3 已验证的 official nuPlan runtime 与 official Parquet canonicalizer，统一测量窗口为
  `np.arange(0.0, 8.0, 0.1)`，即 `[0.0,8.0)` 的 80 帧，不生成 81 帧。
- 每次 official simulation 前写入 claim，达到 48 次后第 49 次必须在 simulator 启动前被拒绝。任一 technical
  failure 立即停止；mechanism/F_match/endpoint/engineering/safety 失败则保留该 pair 并继续完整冻结日程。
- 原始 official trace、Parquet、logs 仅保留在 `outputs/r1_official_compliant_technical_smoke_v1/`，不得提交；小型
  roster、账本、pair/context/safety CSV、manifest 与中文报告写入 `docs/stageR/r1/`。

### 3. 通过标准

- preflight 只能输出 `PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED`、`0/48`，并验证 24-row roster、48-run schedule、
  `AbstractPlanner`、两 family 的 frozen arm、canonicalizer/context/mechanism/F_match/endpoint 调用链及第 49 次 fail-closed。
- R-HLC 只有 12/12 同时满足 technical、context identity、Option-B mechanism、三项 Primary F_match、Primary endpoint、
  既有 engineering 与官方 safety 才能进入 formal development roster review；heading total 仅 secondary。
- R-TSB 只有 12/12 同时满足 technical、context identity、Option-A mechanism、四项 Primary F_match 与官方 safety 才能进入 review。
- 不论结果如何，本阶段不执行 formal development rollout，RBR-A/B/C 始终 `NOT_AUTHORIZED`。

## StageR / R1 Phase B2 technical-failure fail-closed 归档

### 1. 命令

仅当原始 budget 账本明确显示“已 claim、但官方 simulator 命令尚未启动”的技术异常时，使用以下只读归档器生成
`NOT_EVALUABLE` 结果表和中文报告：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_finalize_official_technical_smoke_failure_v1.py
```

### 2. 期望行为

- 只读取 `outputs/r1_official_compliant_technical_smoke_v1/official_run_budget_v1.0.json`；不启动 nuPlan、不修改原始 claim
  账本、不替换 roster、不重跑或继续剩余日程。
- 仅接受 `claimed_count=1`、`CLAIMED_BEFORE_SIMULATION` 和 `CLAIMED_NOT_STARTED` 的 fail-closed 情形，并输出小型 ledger、
  空 pair/context/safety 表、family summary、manifest、中文报告和 readiness 更新。

### 3. 通过标准

- manifest 必须明确区分 budget claim 数与 simulator command start 数；没有 trace/Parquet 时，pair gate 与 safety 必须标为
  `NOT_EVALUABLE`，不得伪造 scientific pass/fail。
- 归档后仍不得修复后重跑、替换 identity、继续剩余额度、进入 formal development rollout 或训练 RBR。

## StageR / R1 Phase B2.1 官方技术 Smoke 环境装配恢复

### 1. 命令

先记录对原 B2 科学状态的语义修正，并在 `0` 个新 official claim 下，以 V3 已验证的五个显式运行时根路径完成绑定和完整执行路径预检：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_run_official_compliant_technical_smoke_v1_1.py --mode correct-b2-status
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_run_official_compliant_technical_smoke_v1_1.py --mode environment-binding
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_run_official_compliant_technical_smoke_v1_1.py --mode recovery-preflight
```

只有预检状态为 `PASS_COMPLETE_EXECUTION_PATH_NO_OFFICIAL_RUN_BUDGET_CONSUMED`、环境绑定为 `MATCHES_V3_BOUND_RUNTIME` 后，才生成一次性授权；授权 SHA 均匹配后才可执行：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_run_official_compliant_technical_smoke_v1_1.py --mode authorize
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_run_official_compliant_technical_smoke_v1_1.py --mode execute
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/r1_finalize_official_compliant_technical_smoke_v1_1.py
```

### 2. 期望行为

- B2.1 只修正 `stage7c_environment(args)` 的完整环境装配；默认五个 roots、Python 解释器、1200 秒 timeout 和单线程变量必须与通过 8/8 的 V3 一致。
- 运行时绑定记录环境变量、线程控制、Python/nuPlan 版本、DB/map fingerprint 和 master seed；任何不一致均 fail-closed，`0/48`。
- 预检会构造 HLC/TSB 的 baseline/treatment 四条真实 nuPlan 命令、检查 `run_simulation.py`、Hydra planner、`AbstractPlanner`、roster/config/environment 可实例化性、冻结 evaluator 链、48 条 schedule 与第 49 次 claim 拒绝，但不启动 simulator、不 claim run。
- 实际执行使用新的 `B2R1` run-ID namespace 和 `outputs/r1_official_compliant_technical_smoke_v1_1/`。历史 B2 的原始 claim/output 保持不动且不属于 B2.1 证据；raw trace、Parquet 和日志不提交。

### 3. 通过标准

- roster SHA256 必须为 `0617e79b9f51d8b2ae8ac76b110e1dbcfaa77dad200a73b405eb2d6a54675e52`，24 token/log 唯一，12 R-HLC + 12 R-TSB；不得重跑 selector 或改 salt。
- 新批次仅允许 48 个 fresh official closed-loop run；任一技术失败立即停止，不重试、不替换场景。科学或 generator gate 未通过则照实记录并继续冻结完整日程。
- R-HLC 与 R-TSB 各自只能在 `12/12` 所有冻结 required gates 通过时进入 formal review；无论结果如何均不启动 development rollout 或 RBR 训练。

## StageR / R1 Phase B2.5 零 Rollout 官方执行集成冻结

### 1. 命令

本阶段只执行语法检查、依赖检查和 fail-closed integration tests；禁止调用 `run_simulation.py`：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m py_compile \
  tools/r1_prospective_generator_contract_v2.py \
  tools/r1_closed_loop_benchmark_v2_1.py \
  tools/r1_hlc_dynamic_clearance_v1_1.py \
  tools/r1_official_map_query_bridge_v2_1.py \
  tools/r1_official_ego_vehicle_binding_v1.py \
  tools/r1_closed_loop_context_adapter_v2_1.py \
  tools/r1_official_technical_smoke_planner_v2.py \
  tools/r1_official_technical_smoke_evaluator_v2.py \
  tools/r1_b2_5_zero_rollout_preflight.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_5_official_execution_integration.py \
  tests/test_r1_b2_4_adversarial_conformance.py \
  tests/test_r1_closed_loop_benchmark_v2.py \
  tests/test_r1_closed_loop_context_adapter_v2.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- replay observation horizon 不完整时即使 actor tracks 为空也必须 `NOT_ELIGIBLE`；只有 global horizon complete 才能记为 `DYNAMIC_CLEAR_NO_ACTORS`。
- context adapter 直接调用 authoritative Stage5D assignment，固定 `lane_aware_only`，禁止 geometric fallback。
- future V2 planner/evaluator 依赖中不得导入历史 B2.1 planner；Primary 只读 realized current ego，planned trajectory 只能是 secondary intent。
- `launch_official_simulation()` 在 B2.5 必须硬失败；candidate/roster/run ledger 均保持 0。

### 3. 通过标准

- B2.5 adversarial tests、B2.4 regression、benchmark/context regression、语法与临时依赖检查全部通过。
- SHA manifest 必须在测试全部通过后生成，并绑定 map bridge、planner/evaluator V2、route builder v1.1、endpoint v1.1、clearance v1.1、construction parity 与 Stage5D parity。
- selector v0.6 只能到 `READY_FOR_SCIENTIFIC_OWNER_ENUMERATION_AUTHORIZATION`；`enumeration_authorized=false`、`new_rollout_authorized=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.6 最终官方调度与 HLC 测量符合性修复

### 1. 命令

本阶段只允许零 rollout dispatch/preflight 与回归测试，禁止调用 `run_simulation.py`：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m py_compile \
  tools/r1_hlc_measurement_conformance_v1.py \
  tools/r1_official_technical_smoke_planner_v2_1.py \
  tools/r1_official_technical_smoke_evaluator_v2_1.py \
  tools/r1_b2_6_official_dispatch_preflight.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_6_final_dispatch_conformance.py \
  tests/test_r1_b2_5_official_execution_integration.py \
  tests/test_r1_b2_4_adversarial_conformance.py \
  tests/test_r1_closed_loop_benchmark_v2.py \
  tests/test_r1_closed_loop_context_adapter_v2.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- Planner V2.1 必须从真实公开 `compute_trajectory(current_input)` 入口委托到 `compute_planner_trajectory`，并返回 `InterpolatedTrajectory`；只检查方法存在不算通过。
- episode phase 必须由 `current_input.iteration.index/time_us` 相对 iteration 0 确定，并记录 iteration、物理时间、nominal/physical elapsed 与 phase source；连续 replan 不得重启 HLC/TSB phase。
- HLC progress 必须由 source/target native projection 的局部跨车道向量读取，raw progress 留存且只有 frozen mechanism 输入可 clip；纵向-only 运动不得变成 lane-transition progress。
- HLC paired route progress 必须把两个 realized terminal 投影到同一个 frozen native route reference 后比较 route-s；禁止 path-length difference surrogate，1.5 m gate 不变。
- Evaluator V2.1 只能以 realized current ego 为 Primary；planned trajectory 仍为 secondary generator intent。所有 projection ambiguity fail closed。

### 3. 通过标准

- B2.6 至少 15 个新增对抗测试及既有 B2.5/B2.4/benchmark/context 回归全部通过；冻结合同 SHA 与 scientific numerics 不变。
- preflight 通过真实 nuPlan 1.2.2 `PlannerInput` 连续调用公开 dispatch，验证 state0 identity、absolute clock、construction parity、route builder 与 phase persistence，但不得启动 simulation。
- selector v0.7 只能记为 `READY_FOR_FINAL_SCIENTIFIC_OWNER_ENUMERATION_AUTHORIZATION`，同时保持 `actual_candidates_enumerated=0`、`actual_roster_selected=false`、`enumeration_authorized=false`、`new_rollout_authorized=false`。
- 本阶段完成后停止：`ENUMERATION=NOT_AUTHORIZED`、`NEW_ROLLOUT=NOT_AUTHORIZED`、`R1_FORMAL_DEVELOPMENT_ROSTER=NOT_READY`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.8-R2 官方启动控制面 fail-closed 核验

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_8_r2_official_launch_fail_closed_audit.py --write

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_8_r2_official_launch_fail_closed.py
```

### 2. 期望行为

- 仅使用已绑定的 nuPlan 1.2.2 官方 `get_scenarios_from_db` 路径核验每个冻结 `scenario_token + log_id`；不会调用 `run_simulation.py`、`SimulationRunner.run()`、`simulation.step()` 或 planner rollout。
- 为 48 个冻结 run 生成完整的未来启动参数与唯一输出路径；若任意官方场景解析不是恰好一个，立即 fail-closed，后续 Hydra composition 与 SimulationRunner construction 均不执行。
- 写入 owner approval record、launch manifest、execution binding manifest 与中文阻断报告；不会修改 roster、schedule、selector、阈值或任何实验产物。

### 3. 通过标准

- 只有 `48_OF_48` exact official scenario resolution 才能继续执行完整 Hydra composition 和 SimulationRunner construction。
- 任一 `0 match` 或 `>1 match` 必须保持 `simulation_started=false`、`official_runs=0`、`consumed_budget=0`，不得 replacement。

## StageR / R1 Phase B2.8-R3.3 最终授权递归 SHA 闭包

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_8_r3_2_pair_executor.py \
  tests/test_r1_b2_8_r3_3_recursive_authorization.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- R3.3 执行器沿用 R3.2 的 48-run 调度语义，只在授权前增加递归 SHA 闭包核验；本阶段不得使用 `--execute`。
- 授权记录必须绑定当前 R3.3 最终 manifest；该 manifest 的 R3.1 继承 manifest、其状态以及 authoritative runtime 组件逐项按 SHA 核验。
- R3.2 当前层的 roster、schedule、冻结 pair binding 与 runtime 组件继续逐项核验；任一缺失或 SHA 不符均在仿真开始前 fail-closed。

### 3. 通过标准

- 正确的 R3.1→R3.2→R3.3 继承链与 34 个 R3.1 runtime 组件全部闭包通过。
- inherited manifest SHA、继承组件、当前组件、roster、schedule、pair binding 和 owner manifest SHA 的负向测试全部 fail-closed。
- 始终保持 `actual_official_runs=0`、`consumed_budget=0`、`OFFICIAL_SMOKE_AUTHORIZED=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-A 首次 official smoke native coverage 离线取证

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_a_native_reference_coverage_forensic.py

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_9_a_native_reference_coverage_forensic.py
```

### 2. 期望行为

- 只读取冻结 roster、official native map、Attempt 1 的 34-row realized trace 与失败产物；不会调用 runner、simulation step、selector 或 RBR。
- 离线重建 iteration 0–33 的 source/target 7.9 秒 query envelope，并对 12 个冻结 HLC identity 计算 constant-speed nominal rolling coverage。
- 输出两个版本化 JSON；若目标已存在则 fail-closed，禁止覆盖。

### 3. 通过标准

- 离线 first invalid iteration 必须与真实 iteration 33 failure 对齐，并区分实际先抛出的 source 与同时越界的 active target。
- 12 个 identity 各包含 baseline/treatment iteration 0–79 envelope，且明确标记为 technical diagnostic，不作为 scientific outcome。
- Attempt 1 原始授权、stop record、trace 和 raw partial output 的 SHA 被记录且文件不移动、不删除、不覆盖；`simulation_executed=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-B 路线连续 HLC 工程 Canary

### 1. 命令

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_b_route_continuous_canary.py --prepare

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_b_route_continuous_canary.py --execute

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_b_route_continuous_canary.py --execute --retry-failed

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_b_route_continuous_canary.py --write-reports

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_9_b_route_continuous_canary.py
```

### 2. 期望行为

- `--prepare` 只从当前冻结 roster、Attempt 1 trace 与既有永久排除账本构造合同、3 个 canary 身份和离线审计；不扫描 source universe，不启动仿真。
- `--execute` 只允许 ledger 中永久标记为 `NON_SCIENTIFIC_ENGINEERING_ONLY` 的 3 个 HLC identity，使用 V3 planner、官方 nuPlan 1.2.2 runner、TwoStageController、observation、metric 与 callbacks；当前其余科学身份会在 runner 前被拒绝。
- route-continuous builder 只沿官方 native topology 连接 source/target 对；路线 occurrence、方向和相邻对应关系必须唯一，歧义、反向、self-intersection 或非零 native join gap 均 fail-closed。
- 工程 canary time controller 沿用官方 StepSimulationTimeController 的逐步语义，只将 canary horizon 版本化为恰好 planner iterations 0...79；不会为 Primary 窗口后的地图终点发明非原生连接。
- `--retry-failed` 保留旧 run ID、trace 与 raw output，并为允许的工程修复创建新 attempt/run ID；禁止覆盖或 append 历史 run。

### 3. 通过标准

- Attempt 1 iteration 33 离线 source/target 均有正覆盖余量，iterations 0...32 与 V2.2 exact parity，34/34 state0 exact identity。
- 当前 12 个冻结 HLC identity 的 baseline/treatment 0...79 滚动覆盖为 12/12，拓扑歧义为 0，且结果只标记 `DIAGNOSTIC_ONLY`。
- 最终至少 3 个独立 canary × baseline/treatment 共 6 个最新 attempt 全部达到 80 行 Primary、无 native coverage failure、无其他技术 failure，并完成 metric/callback。
- canary token/log 永久写入科学排除 ledger；科学 roster、阈值、F_match、安全定义均不变，`OFFICIAL_SMOKE_AUTHORIZED=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-C Primary80 与跨 Family 工程 Canary

### 1. 命令

以下命令只适用于已冻结、永久禁止科学使用的工程 canary 身份；不得用于当前 scientific identities：

```bash
/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_c_cross_family_canary.py prepare

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_c_cross_family_canary.py execute

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_c_cross_family_canary.py finalize

/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_9_c_primary80_cross_family.py \
  tests/test_r1_b2_9_b_route_continuous_canary.py
```

### 2. 期望行为

- `prepare` 冻结 Primary80 runtime contract，离线审计当前 12 个 HLC 身份与 3 个既有 HLC canary 的 source/target frozen-route progression；任何 target roadblock 不一致都在仿真前 fail-closed。
- `execute` 仅运行 roster 中 3 个 HLC 与 3 个 TSB 永久科学排除身份，每个 baseline/treatment 各一次；每个输出根和 trace path 必须全新，工具不提供自动 rerun。
- 科学 time-controller 继承 nuPlan 1.2.2 StepSimulationTimeController，仅将有效场景固定为 81 controller iterations，从而产生 planner calls 0...79；少于 81 iterations 显式 `NOT_EVALUABLE`。
- `finalize` 只读取实际 80-row trace、官方 metric Parquet、历史 pretreatment artifacts，并调用冻结 safety adapter 与 V2.1 evaluator dispatcher；scientific gate 结果不得用于调参或选身份。

### 3. 通过标准

- route progression invariant 为 PASS，target route-consistency violation 为 0；V2.3 与 V2.2 对所有接受输入的 native reference exact parity。
- fresh actual runs 为 12、reruns 为 0；HLC 与 TSB 均为 6/6 technical complete，12/12 trace 精确覆盖 0...79，secondary planner calls 为 0。
- 12/12 metric/callback 与 safety adapter structural complete；pair dispatcher HLC 3/3、TSB 3/3 完成。
- candidate manifest 状态只能是 `READY_FOR_SCIENTIFIC_SELECTOR_ROSTER_REBUILD_REVIEW`；不得创建 scientific roster，`OFFICIAL_SMOKE_AUTHORIZED=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-D Prospective Scientific Roster 最终冻结

### 1. 命令

以下命令只做冻结资产复核、零运行构造和结构测试。禁止添加 `--execute`：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_d_execute_frozen_48run_smoke.py \
  --output docs/stageR/r1/r1_b2_9_d_zero_run_final_construction_audit_v1.0.json

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_d_finalize_scientific_package.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_9_d_final_package.py
```

### 2. 期望行为

- selector v1.3 复用冻结 source universe、master seed、salt、rank/hash 和既有 eligibility gates，只增加永久排除、Primary80 与 HLC route-continuous eligibility；不读取 canary scientific outcome 选 identity。
- 新 roster 固定 24 个 identity（12 HLC、12 TSB），schedule 固定 48 个全新 `R1B29D-...` run ID，pair binding 在 simulation 前冻结 24/24。
- 零运行 executor 对 48 个 run 依次完成 exact scenario resolution、完整 Hydra compose、V3.1 planner、Primary80 controller 和 SimulationRunner 构造，然后在 `runner.run()` 前硬停止。
- finalizer 仅以合约有效的临时 80-row REALIZED trace 和真实 parquet 格式调用 24 个 pair dispatcher；临时文件随进程退出清理，不生成 scientific outcome 或 official output。
- HLC planner reference 固定为 `ROUTE_CONTINUOUS_V2_3`；measurement reference 仍为 `FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT`，不得改变 measurement numerics。

### 3. 通过标准

- effective exclusion 为 45，Attempt 1 identity 永久保留 `OFFICIAL_ATTEMPT_CONSUMED=true`；roster 为 HLC 12/12、TSB 12/12，且没有 excluded token/log。
- 48/48 exact resolution、48/48 exact V3.1 planner、48/48 exact Primary80 controller、48/48 runner construction 全部通过，controller iteration 数统一为 81。
- 24/24 pair binding pre-outcome complete，24/24 dispatcher structural invocation 通过；第 49 次 claim 在 runner 前拒绝。
- 完整传递 SHA 闭包通过，protected CSV SHA256 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。
- 最终状态只能是 `FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION`；仍保持 `runner.run()=0`、official runs=0、consumed budget=0、`OFFICIAL_SMOKE_AUTHORIZED=false`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-E Post-Run Callback Lifecycle 修复

### 1. 命令

以下命令只复核新版本 package，不执行 scientific simulation。禁止添加 `--execute`，且已消费的 exact-lifecycle canary 不得再次运行：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r1_b2_9_e_execute_frozen_48run_smoke.py \
  --output docs/stageR/r1/r1_b2_9_e_zero_run_final_construction_audit_v1.0.json

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b2_9_e_lifecycle_repair.py
```

`tools/r1_b2_9_e_finalize_package.py` 是一次性 versioned artifact finalizer；本阶段产物已生成后不得对同名输出覆盖运行。

### 2. 期望行为

- roster v3.0 和 24 个 scientific identities 保持原 SHA；selector 不运行，source universe 不扫描。
- schedule v3.1 与 v3.0 的 identity、family、arm、pair order、run order 完全一致，只把已消费的 B2.9-D run/pair references 机械版本化为 `R1B29E-...`。
- pair bindings v2.1 与 v2.0 的 context、hash、clearance、measurement/native route reference 和全部 scientific semantics 完全一致，只更新 run/pair references 与 package provenance。
- zero-run 对 48 个新 run 完成 exact scenario resolution、V3.1 planner、Primary80 controller、runner construction 与 pair lookup，然后在执行前停止；不调用 `runner.run` 或 `run_runners`，不生成 fake metric。
- 新 executor 的唯一 runtime 修复是经共享 `run_one_with_full_nuplan_lifecycle(...)` 调用 nuPlan `run_runners(...)`，从而完成 runner report、post-run main callbacks 与 metric parquet aggregation。
- B2.9-D 两个旧 attempts 和输出只作为历史取证；不得补 callback、补 parquet、补 evaluation，亦不得作为新 scientific pair input。

### 3. 通过标准

- exact-executor engineering canary：HLC 2/2、TSB 2/2 technical complete；4/4 Primary80 trace、metric parquet、runner report、安全适配 complete；2/2 pair dispatcher complete；simulation rerun 为 0。
- 48/48 zero-run construction PASS，`runner.run=0`、`run_runners=0`；24/24 synthetic structural dispatcher PASS。
- executor 源码没有直接 `SimulationRunner.run()` 调用；expected safety parquet 缺失时 shared helper 必须 fail-closed。
- roster SHA 保持 `efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6`，protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。
- callback transitive 与完整 SHA closure 均为 PASS；最终状态只能是 `FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_REAUTHORIZATION`。
- 本轮 official scientific simulation 为 0；`OFFICIAL_SMOKE_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`。

## StageR / R1 Phase B2.9-E 一次性 Official Scientific Smoke 结果

### 1. 已消费的一次性授权

授权 manifest、roster、schedule 与 pair binding 已按 Scientific Owner 的 ONCE 授权完成 48-run 正式执行。该授权和 48 次预算已经全部消费；**不得再次运行带 `--execute` 的命令**，不得重试、替换 identity 或修改阈值。

后续只允许读取并复核下列固化结果：

```text
docs/stageR/r1/r1_b2_9_e_official_smoke_run_ledger_v1.0.json
docs/stageR/r1/r1_b2_9_e_official_smoke_pair_gate_table_v1.0.json
docs/stageR/r1/r1_b2_9_e_official_smoke_family_summary_v1.0.json
docs/stageR/r1/r1_b2_9_e_official_smoke_raw_output_sha_manifest_v1.0.json
docs/stageR/r1/R1_B2_9_E_Official_Scientific_Smoke_Execution_Report_v1.md
docs/stageR/r1/R1_B2_9_E_Scientific_Owner_Result_Review_Request_v0.1.md
```

### 2. 固化结果

- 48/48 attempts 已 claim，48/48 `run_runners` 完成，48/48 technical complete；没有 retry、identity replacement 或 threshold change。
- 48/48 Primary trace 精确为 iterations 0...79，source 均为 `REALIZED_CURRENT_EGO`，secondary trace 为 0；48/48 runner report、metric lifecycle 与 safety adapter 输入完整。
- HLC：context 12/12、mechanism 0/12、F_match 12/12、endpoint 6/12、engineering 12/12、safety 11/12，因此 `HLC_FAMILY_SMOKE_READY=FAIL`。
- TSB：context 12/12、measurement applicability 0/12、mechanism 0/12、F_match 12/12、safety 11/12，因此 `TSB_FAMILY_SMOKE_READY=FAIL`。
- 两个 family 的 FAIL 都是冻结门规则下的科学结果，不是技术基础设施失败；完整 schedule 已执行完毕，不得据此补跑或换样本。

### 3. 下一步边界

- Scientific Owner 只读审阅 execution report、pair gate table、family summary 与 raw-output SHA manifest。
- `RBR_A/B/C=NOT_AUTHORIZED_PENDING_SCIENTIFIC_OWNER_RESULT_REVIEW`。
- 原始 simulation output 保留在本地证据目录，不提交 Git；Git 只固化小型账本、门结果、哈希清单、报告和授权记录。

## StageR / R1 Phase B3 Realized Mechanism Transfer Forensic

### 1. 只读复核命令

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r1_b3_realized_mechanism_forensic.py
```

不得再次运行 B2.9-E `--execute`；B3 不提供 simulation、canary、selector、generator tuning 或 RBR 入口。

### 2. 期望行为

- 只读加载 24 个 B2.9-E frozen evaluator 文件，并验证其与 committed pair gate table 24/24 一致；不重算或替代 scientific result。
- 24 个 official identities 全部新增标记为 outcome-exposed，只允许 R1 历史失败诊断，禁止用于 R2 development、calibration、model selection 或 confirmatory smoke。
- HLC 用冻结 native measurement 与 Option-B 函数解析 realized progress，并把 ideal generator 结果明确隔离为 analytical diagnostic。
- TSB 用冻结 timestamp-aware measurement 解析实际速度和加速度；one-step transfer 只读取 nuPlan 已序列化的 exact planner trajectory，与下一帧 realized ego 对照，不构造或运行 simulator。
- 所有 B2.9-E raw outputs 由既有 SHA manifest 做 1,080/1,080 只读闭包验证。

### 3. 固化结论

- HLC：baseline monotonic 恒为 1.0；treatment median 为 0.927766；realized delta median 为 -0.072234，未达到冻结 -0.10 gate。Retreat、latency 与 status 均为 12/12，唯一 mechanism failure 为 monotonic attenuation。
- HLC endpoint：6/12 PASS；5 个 failure 来自 treatment terminal lateral velocity，1 个来自 treatment terminal offset，heading 与 route-progress failure 均为 0。
- TSB：baseline/treatment 均为 12/12 `NO_BRAKE_PHASE`、0 个 `LOW_SPEED_ENDSTOP`；release window descriptive realization 为 12/12，dominant failure 是 brake-amplitude attenuation。
- Ideal TSB generator 可产生 baseline 1 phase 与 treatment 2 phases；该结果仅用于 generator-intent validity，不替代 realized failure。
- F_match：HLC 12/12、TSB 12/12，`HANDCRAFTED_NUISANCE_MATCHING=SUCCESSFUL`。
- 推荐 R2 repair family：`CONTROLLER_AWARE_TRAJECTORY_SHAPING + FEEDBACK_CALIBRATED_GENERATOR`，但只允许在 fresh、永久 engineering-only identities 上开发；threshold relaxation 不推荐。
- `R1_RESIDUAL_BENCHMARK_ENABLEMENT=FAILED_UNDER_FROZEN_R1_CONTRACT`；`RBR_FORMAL_TRAINING=NOT_AUTHORIZED`。

## StageR / R2 Phase A Controller Transfer Identification

### 1. 命令

R2-A 的 80 个有效 DEV design units 与 4 个技术重跑已经执行完毕。下面命令只做离线识别、闭包和测试；**不得再次运行 engineering executor**，也不得据此建立 confirmatory roster 或训练 RBR：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r2_a_analyze_controller_transfer.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r2_a_finalize_controller_transfer_freeze.py

PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_a_controller_transfer_identification.py
```

### 2. 期望行为

- 只读取 8 个 HLC 与 8 个 TSB 的永久 engineering-only DEV identities，以及 80 套有效的 Primary80 realized/planner/LQR telemetry；不读取 R1 official outcome 来调参。
- HLC 离线识别 commanded/realized retreat、monotonic effect、commit、tracking lag、settling 与 terminal lateral velocity。
- TSB 离线分解 generator→LQR 与 LQR→realized gain，并审计 absolute-time repeated replanning 下的 phase shortening、boundary migration、phase disappearance 和 release carryover。
- 生成小型 deterministic linear surrogate，并执行 leave-one-identity-out 描述性验证；该模型只用于 R2-B architecture，不是最终 generator。
- finalizer 把 16 个 DEV identities 追加到永久排除账本，并 SHA-bind 80 套有效 telemetry；它不构造 simulator，也不调用 `runner.run`。

### 3. 通过标准

- roster 为 HLC 8、TSB 8，和此前 69 个历史/R1 outcome-exposed identities 的重叠为 0；最终永久排除账本为 85 个 identities。
- 预冻结设计为 HLC 5×8、TSB 5×8，共 80 个有效运行；80/80 technical complete。只因技术故障产生 4 个 fresh-root 重跑，实际 engineering runs 为 84。
- 每个有效运行恰有 80 个 realized rows、80 个 planner telemetry rows、79 个 controller transitions；planner telemetry 覆盖 state0...state10，LQR control return value 可用。
- surrogate 采用 leave-one-identity-out；不使用复杂黑盒，不改变 scientific threshold，不冻结最终 R2 generator 参数。
- `R2_confirmatory_roster_selected=false`、`RBR_started=false`，protected CSV SHA 为 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase B Controller-Aware Generator Development

### 1. 命令

R2-B 已消费冻结的 DEV-CAL 运行计划并达到 4 轮 HLC 上限。**不得再次执行带 `--execute` 的校准命令**。只允许运行下列离线测试与只读核验：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_b_controller_aware_generator_development.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读验证 8 个 HLC、8 个 TSB DEV-CAL identities 的数据防火墙、永久 engineering-only 处置和全局参数化。
- 检查 HLC/TSB 在 1.1 秒之前两 arm 完全一致，generator 不允许 scenario token/log ID 参数查表。
- 读取 5 个已完成 round 结果与 80 个 DEV 工程 run 的 SHA provenance；不提交 raw simulation output。
- 不生成 R2-C identities，不运行 confirmatory smoke，不训练 RBR，也不修改任何科学阈值。

### 3. 通过标准

- DEV-CAL 与 R1 official、R2-A、既有黑名单重叠均为 0；16 个 identities 全部永久排除于后续 scientific use。
- HLC 严格停止在 4 轮：最终 mechanism 6/8、F_match 8/8、endpoint 0/8、engineering 8/8、safety 8/8。
- TSB 第 0 轮达到 measurement、one/two-phase mechanism、F_match、safety 全部 8/8。
- 因 HLC 未收敛，整体必须为 `R2_B_DEVELOPMENT_NOT_CONVERGED`；不得生成 `r2_b_selected_generator_parameters_v1.0.json`，不得进入 R2-C。
- protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BH HLC Target-Capture Architecture Development

### 1. 命令

R2-BH 已完成冻结的 3 轮 HLC DEV-ARCH 工程执行。**不得再次运行带 `--execute` 的命令**。只允许运行以下离线核验：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bh_hlc_target_capture_architecture.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读验证 V1 constant re-anchor invariant、V2 fixed absolute-time quintic target capture、三轮结果与数据防火墙。
- V2 每次 trajectory 的 state0 精确等于 current ego；state1+ residual command 在固定 capture end 归零，不随 replanning 重启 settling horizon。
- TSB family candidate 只做机械冻结，不重跑 TSB；raw DEV output 只以 SHA provenance 固化，不提交 Git。
- 不执行 Round 4，不选择 R2-C identities，不运行 confirmatory smoke，不训练 RBR。

### 3. 通过标准

- V1 合成 offset `0、±0.25、±0.50 m` 的 terminal residual 5/5 原样保留，支持 architecture diagnosis。
- 新 roster 为 8 个 fresh HLC DEV-ARCH identities，与 historical/R1/R2-A/R2-B 重叠为 0；最终永久排除账本为 109 个 identities。
- V2 planner command 在每轮 capture end 后 16/16 归零，但最终 realized mechanism 0/8、endpoint 0/8，因此状态必须为 `R2_BH_DEVELOPMENT_NOT_CONVERGED`。
- 最终 F_match 8/8、engineering 8/8、safety 4/8；不得据此降低 mechanism、endpoint 或 safety 定义。
- 不生成 HLC selected-parameter 或 complete G_R2 candidate manifest；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BI HLC Controller-Observable Kinematic Target-Capture

### 1. 命令

R2-BI 已在 Round 0 首个 treatment 的首次 arm divergence 触发冻结运动学可行性门并停止。**不得再次运行带 `--execute` 的工程执行命令**，不得补跑剩余 14 个 run 或启动 Round 1。只允许运行以下离线核验：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bi_hlc_kinematic_target_capture.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读验证 R2-BH exposure firewall、V2 controller-interface forensic、V3 morphology/capture 组合、25 个 zero-run entry cases、fresh DEV-KIN roster 和 Round 0 stop audit。
- state0 保持 current ego；state1+ 由最终 XY 唯一推导 heading/curvature；nonzero residual 对 exact frozen LQR shadow 产生方向正确的非零 steering response。
- Round 0 raw 只作为 SHA provenance：第一个 baseline 完成，第一个 treatment 在 absolute time 1.1 s fail-closed；不得把后续 callback 缺少 safety parquet 误记为根因。
- 不重跑 R2-B/R2-BH identities，不使用 R2-BH raw 做 V3 数值调参，不修改 scientific threshold，也不提交 raw simulation output。

### 3. 通过标准

- mandatory zero-run entry gates 为 25/25 PASS；fresh DEV-KIN roster 为 8，和 historical/R1/R2-A/R2-B/R2-BH 重叠为 0。
- 实际 HLC engineering `runner.run` 为 2：baseline 1 次 technical complete，treatment 1 次在首次 divergence 因 `7.391761 m/s² > 6.0 m/s²` 横向加速度门失败；technical rerun 为 0。
- treatment 失败的曲率、yaw-rate、state0→state1 连续性和 XY-heading consistency 均在冻结门内；不将单个 identity 外推为跨 identity 系统性结论。
- 状态必须为 `R2_BI_DEVELOPMENT_NOT_CONVERGED`；Round 1 不启动，不生成 selected HLC V3 parameters 或 complete G_R2 candidate。
- scientific simulation、TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-A HLC Morphology Feasibility Envelope

### 1. 命令

本阶段是离线、零仿真的可行性审计。结果文件已版本化，日常复核只运行测试；不得调用任何带 `--execute` 的工程命令：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_a_offline_morphology_feasibility.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读核验 8 个 R2-BI DEV-KIN identities 的永久防火墙、V4 全局 C2 morphology、滚动 stitching horizon、完整 planner `_states` 零运行包络和 exact frozen LQR shadow。
- V4 保留 V3 的 final XY → tangent heading → curvature、state0 continuity、controller-visible curvature 和 infeasible fail-closed；不使用 scenario/log/identity 参数查表。
- 审计把 morphology intrinsic、online stitching、native curvature、target capture 与 composite final trajectory 分开报告；不会把 composite failure 全部归因于 capture。
- 不构造 simulator、不调用 `runner.run`，不选择新 roster，不请求 BJ-B simulation authorization，不启动 TSB、R2-C、confirmatory smoke 或 RBR。

### 3. 通过标准

- 新 intrinsic morphology 在冻结最大 lane separation 下的峰值横向加速度不超过 `6.0 m/s²`，1.1 秒 common→treatment 边界满足 P/V/A C2，且无正 lag phase shift。
- expanded zero-run audit 覆盖 baseline/treatment、Primary80、所有 phase/capture 边界、直/左右曲线、左右目标车道、lane/speed/curvature/residual 边界；每个 case 都走完整 `_states`。
- 当前 raw source-universe 笛卡尔包络为 1160/3296 PASS，存在 2136 个冻结运动学门失败，因此最终必须为 `R2_BJ_A_OFFLINE_ARCHITECTURE_NOT_READY`，BJ-B request 为 `REQUEST_WITHHELD`。
- `runner.run=0`、engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-A2 HLC Joint-Support Applicability Envelope Audit

### 1. 命令

A2 的版本化结果已经生成。日常复核只运行静态测试；不要再次执行结果生成工具，也不得调用任何 simulator 或带 `--execute` 的命令：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_a2_joint_support_applicability.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读验证已提交、outcome-blind 选择的 HLC opportunity；每条记录把 token/log/anchor、pre-treatment speed、source/target reference、曲率、lane separation 和 provenance 保持为不可拆分联合单元。
- 同时保留 raw pointwise curvature 与预注册的 `0.25 m` 等弧长重采样、`5.0 m` 固定窗口 robust curvature；历史 `0.082281 1/m` 只作独立取证，不与其他 identity 的速度拼成主包络。
- 使用冻结 V4 `_states` 对 nominal/预注册速度裕量、`±0.25 m` residual、两 arm 和 Primary80 的所有 absolute replanning time 做分量审计；分别检查 native、morphology、stitching/capture、generated increment 与 composite。
- 不读取 realized outcome，不选择 roster，不修改 V4 或阈值，不运行 engineering/scientific/TSB simulation，不进入 BJ-B、R2-C、confirmatory smoke 或 RBR。

### 3. 通过标准

- 每个已物化 joint record 的 provenance closure 为 100%，数据防火墙为 `PASS_NO_OUTCOME_LEAKAGE`，`runner.run=0` 且 simulation count 为 0。
- BJ-A Cartesian 结果继续作为 adversarial stress appendix，不能单独决定 actual-domain readiness；generated increment 必须单独过门，禁止用 native 正负曲率抵消。
- 因冻结 eligibility 管线没有持久化全 source universe 的全部 eligibility-pass population，A2 必须为 `JOINT_SUPPORT_EXTRACTION_INCOMPLETE` 并 withholding Owner readiness，不得声称包络已闭合。
- native-only infeasible 记录不得自动排除；任何 curvature representation、generated increment 或 terminal settling 未闭合项都必须附加相应 fail-closed blocker。
- protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-A3 HLC Prospective Applicability Predicate

### 1. 命令

A3 的固定审计框和版本化结果已经生成，不得再次运行 `freeze-frame` 或 `evaluate`。日常只读复核使用：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_a3_prospective_applicability.py \
  tests/test_r2_bj_a_offline_morphology_feasibility.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- 只读核验在任何新候选 V4 结果前冻结的 hash seed、排序、256 条 frame、token/log 去重、永久排除和左右方向规则。
- 主速度严格使用 `max(official initial, pre-treatment 0–1.0 s max)`；裕量速度使用 `v_audit + max(0.5, 0.05*v_audit)`。anchor timestamp 只作 provenance，不参与选择或 eligibility。
- 对 47 条 A2 完整记录、10 条 A2 extraction failure 和固定 256 条 fresh audit-frame candidates 使用同一 V2.3/V4 predicate；全部为纯离线 `_states` 审计。
- raw 与 robust curvature 同时保留。历史 `0.082281 1/m` 继续作为 terminal short-segment gradient artifact 留在 adversarial appendix，不进入 actual joint support。
- 历史 BJ-A manifest 按其绑定 commit/tree 校验；当前活文档 `QUICK_REFERENCE.md` 由 A3 manifest 绑定。
- 不选择 BJ-B roster，不构造 simulator、不调用 `runner.run`，不运行 engineering/scientific/TSB simulation，也不启动 R2-C、confirmatory smoke 或 RBR。

### 3. 通过标准

- 固定 frame 必须 256/256 审完且不得在获得足够通过者时提前停止；不能把它称为完整 source-universe census。
- 当前结果为 17/256 通过同一 predicate，低于 32 条 readiness 要求；47 条历史完整记录在修正速度包络下为 46/47，因此 A3 必须保持 `JOINT_SUPPORT_EXTRACTION_INCOMPLETE`，不得进入 BJ-B。
- 10/10 历史 extraction failure 均获得统一技术处置，且不形成 outcome blacklist；所有 17 条通过者的 provenance、reference geometry、速度与组件 closure 为 100%。
- full V4 component stage 共覆盖 28 条、26,880 个离线 planner-state cases；17 条全门通过、11 条 generated increment/composite 不可行，禁止利用正负曲率抵消。
- `runner.run=0`、所有 simulation count 为 0、roster selected 为 false；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-A4 HLC Moving-Regime Applicability

### 1. 命令

A4 已在 frame 冻结阶段 fail-closed。不得重新运行 frame 生成或对 557 条不完整 frame 做部分 predicate；日常只读复核使用：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_a4_hlc_moving_regime_applicability.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- estimand 固定为 `MOVING_VEHICLE_HESITANT_LANE_CHANGE`，在 topology/curvature/V4 前执行 `v_audit >= 3.0 m/s`；主速度只取 official initial 与 0–1.0 s pre-treatment 最大速度。
- frame 固定目标为 768，使用新 salt/seed、SHA256 排序、全局 token/log 去重，并排除全部历史、A2、A3 与永久 exclusion；不读取 outcome。
- 构造采用对全部 canonical logs 的穷尽式“每 log 首个 hash-ranked 基础 HLC pass，再全局 hash 排序”，不受有限 rank-prefix 容量影响。
- A3 的 11 条 V4 failure 均保持原结果，新增 `LOW_SPEED_OUTSIDE_V4_APPLICABILITY` 处置；A2 token `3feb5f93f24e5b77` 保持当前 V2.3 topology 不适用，不新增 outcome blacklist。

### 3. 通过标准与当前处置

- 冻结 source universe 的 1,621 个 canonical logs、5,386,575 条 source rows 已穷尽；在严格全局 log 去重与基础 HLC 条件下仅能形成 557 条记录，距 768 缺 211。
- 因预注册 frame 无法完成，A4 状态为 `APPLICABLE_POOL_INSUFFICIENT`；A4 speed/topology/curvature/V4 predicate 评估数为 0，不把 557 条部分 frame 当作候选池。
- 未修改 V4、morphology/capture 参数、topology builder 或任何阈值；未选择 BJ-B roster。
- `runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-A5 Frozen 557-Log Finite-Frame Census

### 1. 命令

A5 的 557 条有限框架普查与版本化结果已生成，不得再次运行 census。日常只读复核使用：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_a5_finite_frame_applicability_census.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 期望行为

- A5 只使用 A4 在任何 predicate outcome 打开前冻结的 557 条记录；不重建 frame、不重扫 source universe、不 rerank、不 replacement、不提前停止。
- A4 的 `APPLICABLE_POOL_INSUFFICIENT` 保持为 768 条原 frame 目标不可构造的历史结论，不解释为 applicable pool 少于 32。
- moving-regime speed floor 保持 `v_audit >= 3.0 m/s`，严格复用 A4 predicate、V2.3 topology、raw/robust curvature taxonomy、V4 generator/planner 和全部阈值。
- 只有通过速度、topology/reference 与 curvature 前置门的记录进入完整 960-case `_states` component audit；禁止 native/generated 正负抵消。

### 3. 当前结果与治理

- `A4_FRAME_CAPACITY = 557`；`A5_CENSUS_EVALUATED = 557`；无提前停止。
- `A5_APPLICABLE_POOL = 34`；`A5_COMPONENT_STAGE_COUNT = 34`；共完成 32,640 个离线 planner-state cases。
- `A5_MOVING_REGIME_COMPONENT_FAILURES = 0`；native、generated increment、composite、continuity、terminal settling 均无失败，curvature 未定义类别为 0，passing provenance closure 为 100%。
- 状态为 `R2_BJ_A5_CENSUS_COMPLETE_READY_FOR_BJ_B_OWNER_REVIEW`，仅请求 Owner 审阅；`BJ_B_ROSTER_SELECTED = FALSE`。
- `runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-B0 HLC V4 Engineering Freeze

### 1. 命令

B0 已冻结 8 条 engineering-only roster 与 16-run intended schedule。不得重新生成 roster；日常只读复核使用：

```bash
PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_b0_hlc_v4_engineering_freeze.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 tools/check_no_tmp_dependencies.py
```

### 2. 冻结内容

- 唯一候选输入为 A5 的 34 条 `MOVING_REGIME_V4_APPLICABLE`；使用预注册 salt 和 NUL 分隔 SHA256 排名，选择字典序最小的满足精确方向、地图和速度带配额的 8 元子集。
- 冻结选择 rank tuple 为 `(1,2,3,4,5,6,7,9)`；8 条均为 `PERMANENT_ENGINEERING_ONLY`、尚未 outcome exposed；其余 26 条保持未选择且 outcome-unexposed，不建立 reserve/replacement 顺序。
- 8 个 pair、16 个 intended runs 按 selection rank、每 pair baseline 后 treatment 排序。两 arm 的 token/log、初始状态、route/reference、pre-treatment context、控制器、配置、seed 和 Primary80 完全共享；`t<1.1 s` trajectory construction exact equal。
- 未来 planner 每次调用检查 curvature、yaw-rate、lateral acceleration、state0/step/tangent/XY-heading continuity、terminal target residual、rolling stitching horizon、controller-visible steering 和 pre-divergence equality。架构失败将停止当前和剩余 schedule，禁止 replacement 或参数更新。

### 3. 零运行状态

- 16/16 full Hydra composition、exact scenario resolution、pair lookup、V4 planner construction、Primary80 controller 和 SimulationRunner construction 均通过。
- 当前授权门保持关闭：`BJ_B_ENGINEERING_SIMULATION_AUTHORIZED=false`、`CANARY_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RUNNER_RUN=0`。
- 下一次仅向 Owner 申请 selection rank 1 的一对 baseline→treatment canary（2 runs）；本阶段未执行该 canary。
- 未启动 R2-C、confirmatory smoke、TSB 或 RBR；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。

## StageR / R2 Phase BJ-B0.1 Canary Production Execution Path Closure

### 1. 命令

B0.1 正式授权门保持关闭。以下命令只显示零运行状态，不构造或启动 simulator：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r2_bj_b0_1_production_canary_launcher.py

PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_b0_1_production_execution_path.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/check_no_tmp_dependencies.py
```

不得使用 `--execute`，除非下一阶段 Scientific Owner 提供同时绑定 B0 component/schedule/pair-binding SHA、B0.1 execution manifest SHA、exact run IDs 和 budget 2 的独立授权记录。

### 2. 期望行为

- production launcher 在同一真实控制流中先核验 protected CSV、全部 B0/B0.1 SHA、授权、预算、exact `[1,2]` schedule 和 fresh output roots，再认领一次性授权与 run attempt，最后才可能到达唯一的 `runner.run()` 调用点。
- future Owner record 还必须绑定固定 output/control roots；attempt ledger 记录授权文件字节 SHA，不能通过更换 ledger 路径重复消费同一授权。
- baseline technical complete 是 treatment runner construction 的必要前提。任何 architecture 或 infrastructure failure 都停止当前及剩余 schedule，不重试、不替换 identity、不更新参数。
- B0.1 wrapper 正常路径原样返回 B0 planner 轨迹；捕获 architecture failure 时，在重新抛出前用同目录临时文件和原子 rename 写入独立 JSON failure audit。
- 当前关闭门、全部 mutation tests 和 mock-runner scheduler tests 均不启动 simulator。

### 3. 通过标准

- 正式 gate 为 `CANARY_AUTHORIZED=false`、`AUTHORIZED_RUN_ORDERS=[]`、`NEW_RUN_BUDGET=0`、`AUTHORIZATION_CONSUMED=false`。
- authorization false、budget 0、SHA/schedule/order/output collision 均产生 0 次 mock runner 调用。
- 临时内存授权下，成功路径严格调用 baseline、treatment 各一次；预算 `2→1→0`，第三次及重复消费均 fail-closed。
- baseline architecture/infrastructure failure 后 treatment 不启动；treatment failure 后不存在第三次调用；architecture 分类不会降级为 infrastructure。
- `RUNNER_RUN=0`，未执行 canary、R2-C、confirmatory smoke 或 RBR。

## StageR / R2 Phase BJ-B0.2 Passive Actual-LQR Telemetry and Canary Analysis Freeze

### 1. 命令

B0.2 正式授权门保持关闭。以下默认命令只读取关闭门并显示零运行状态，不构造或启动 simulator：

```bash
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/r2_bj_b0_2_production_launcher_adapter.py

PYTHONWARNINGS=ignore \
PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation:/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 -m pytest -q \
  tests/test_r2_bj_b0_1_production_execution_path.py \
  tests/test_r2_bj_b0_2_passive_lqr_and_preanalysis_freeze.py

PYTHONPATH=/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation \
  /Users/liuqing/miniconda3/envs/nuplan/bin/python3.9 \
  tools/check_no_tmp_dependencies.py
```

不得使用 `--execute`。只有 Scientific Owner 提供同时绑定 B0、B0.1、B0.2 manifests、exact `[1,2]` run slice 和 budget 2 的独立授权记录后，才能进入生产路径。

### 2. 期望行为

- B0.2 不修改 B0/B0.1/V4；在实际 runner 构造后验证 Primary80=81、`TwoStageController`、`LQRTracker`，在预算认领前完成可安装性验证，并在 B0.1 唯一 `runner.run()` 调用点之前安装被动 recorder。
- recorder 原样返回 actual LQR 结果对象，同时独立重算冻结 acceleration/steering shadow；未来每 arm 必须恰好写 79 行 controller-transition telemetry。
- 既有 80 行 `controller_visible_telemetry.jsonl` 明确仅是 `PLANNER_REFERENCE_STEERING`，不是 actual controller command。
- pair analyzer 已在 outcome 前冻结，自动读取两 arm 固定产物并执行 frozen mechanism、F-match、endpoint、engineering、safety、actual/shadow、target-offset 与 hard-jump 审计；不允许人工拼装，也不自动授权剩余 14 runs。

### 3. 通过标准

- 2/2 zero-run runner construction、Primary80、TwoStageController、LQRTracker 和 recorder installation 通过。
- mock mutation tests 覆盖同一返回对象、命令不变、79/80 cardinality、非有限值、方向不一致只记录、写入失败停止、错误 controller/tracker、baseline gating 与 architecture 分类优先。
- 正式 output/control roots 均不存在；`CANARY_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RUNNER_RUN=0`。
- 未启动 R2-C、confirmatory smoke 或 RBR；protected CSV SHA 保持 `e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`。
