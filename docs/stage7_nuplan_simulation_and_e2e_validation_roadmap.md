# Stage 7 — nuPlan Official Simulation Data Generation and Stage 6 Reuse Roadmap

## 1. Top-Level Purpose

Stage 7 的核心目的不是重新实现 BDD，也不是重新实现 report card。

Stage 7 的核心目的，是使用 **nuPlan official simulation** 生成可控的自动驾驶 planner / policy 行为数据，并将这些数据转换成 **Stage 6-compatible dataset**，然后复用 Stage 6 已经完成的：

- BDD / MMD 分布漂移计算；
- scenario-balanced BDD；
- task-conditioned behavior-event BDD；
- category / feature / slice explanation；
- report card；
- top drift cases；
- case-level interpretability。

因此，Stage 7 的定位是：

```text
Stage 6 = canonical BDD / report-card evaluation engine
Stage 7 = controllable nuPlan planner-generated data source
```

Stage 7 要解决的是 Stage 6 的核心数据限制：

```text
Stage 6 Waymo 数据：
  来自真实道路 logged trajectories；
  驾驶员未知；
  无法区分同一驾驶员 / 不同驾驶员；
  无法控制同一场景下的不同驾驶风格。

Stage 7 nuPlan 数据：
  来自 official nuPlan closed-loop simulation；
  可以在同一 scenario 下运行不同 planner / policy；
  可以构造可控的 conservative / comfort / aggressive planner profiles；
  可以生成替代 Waymo 的可控自动驾驶行为数据；
  然后喂给 Stage 6 做 BDD 和 report card。
```

Stage 7 所有 planner 数据都必须来自 official nuPlan simulation。禁止使用 pseudo rollout、numpy trajectory rewriting、offline trajectory interpolation 来冒充 closed-loop simulation。

---

## 2. Updated Compact Roadmap

| Stage | Purpose | Main data source | Main output | Status |
|---|---|---|---|---|
| 7A | nuPlan readiness | nuPlan DB / map / devkit | readiness evidence | PASS |
| 7B | nuPlan context construction | nuPlan logs / maps | ego + neighbor + context features | PASS |
| 7C | official planner data generation | nuPlan official simulation | planner rollout tensors + official logs | IN PROGRESS |
| 7D | Stage 6-compatible dataset export | Stage 7C official simulation outputs | full Stage 6-compatible shards | NEXT |
| 7E | embedding export / manifest construction | Stage 7D dataset + Stage 5 encoder | embedding manifest + policy indices | TODO |
| 7F | reuse Stage 6 BDD / report-card engine | Stage 7E embeddings + Stage 7D metadata | BDD, task-BDD, report cards | TODO |
| 7G | final Stage 7 thesis evidence | 7C / 7D / 7E / 7F | final planner-style validation evidence | TODO |

---

## 3. Stage 7A — nuPlan Readiness

**Definition:** Stage 7A = nuPlan environment / data / map / scenario / simulation API readiness.

**Purpose:**

- Check nuPlan DB access.
- Check map root access.
- Check scenario metadata.
- Check expert ego pose and object extraction.
- Discover available simulation APIs and planner classes.

**Current status:** PASS.

Stage 7A is an infrastructure-readiness stage. It does not by itself prove planner-induced or E2E-induced behavior drift.

---

## 4. Stage 7B — nuPlan Context Dataset Construction

**Definition:** Stage 7B = build a strict, row-aligned, auditable scenario context dataset for downstream nuPlan simulation and Stage 6-compatible dataset export.

**Sub-stages:**

- 7B.1 expert ego/object export.
- 7B.2 dynamic context conversion.
- 7B.3 map/ODD-lite feature builder.
- 7B.4 dynamic + map/ODD merge/alignment.

**Current status:** PASS.

Validated final directory:

```text
outputs/stage7b4_nuplan_context_merged/
```

Validated shapes:

```text
ego_seq.npy:                  [23, 80, 8]
neighbor_seq.npy:             [23, 5, 80, 15]
context_traj.npy:             [23, 80, 83]
context_mask.npy:             [23, 80, 5]
dynamic_feat_style.npy:       [23, 33]
map_odd_feat.npy:             [23, 37]
merged_context_feat.npy:      [23, 70]
```

Stage 7B is not itself simulation. It is the context and alignment foundation used to select, align, condition, and diagnose later nuPlan simulation and Stage 6-compatible dataset export.

---

## 5. Stage 7C — Official nuPlan Planner Data Generation

Stage 7C is the data-generation stage.

It should run multiple planner / policy variants in official nuPlan simulation on the same selected scenarios, then export official simulation trajectories and logs.

Stage 7C is divided into three planner families:

```text
Stage 7C.2 — IDM longitudinal-only planner profiles
Stage 7C.3 — PDM longitudinal + lateral planner profiles
Stage 7C.4 — ML planner longitudinal + lateral planner profiles
```

The common output of each planner family should be:

```text
simulated_ego_seq.npy
simulated_ego_seq_mask.npy
simulated_ego_seq_index.json
simulated_planner_metadata.csv
scenario_planner_index.csv
simulation_schema.json
warnings.json
official_nuplan_runs/**/*.msgpack.xz
```

The official simulation output tensor should follow:

```text
simulated_ego_seq.npy:       [N, P, T, C]
simulated_ego_seq_mask.npy:  [N, P, T]
```

where:

```text
N = number of scenarios
P = number of planner / policy variants
T = simulation timesteps
C = ego state channels
```

All Stage 7C planner data must satisfy:

```text
pseudo_rollout == false
uses_official_nuplan_simulation == true
official_success_count == N × P
missing_pair_count == 0
msgpack_simulation_log_files_parsed == N × P
```

---

## 6. Stage 7C.1 — Official Simulation Smoke and Alignment Foundation

**Current status:** PASS.

Validated facts:

- Stage 7C.1A official simulation smoke: PASS.
- Stage 7C.1B official msgpack trajectory export: PASS.
- Stage 7C.1C exact log + actual nuPlan scenario token wrapper smoke: PASS.
- Stage 7C.1C strict Stage7B scene_token == nuPlan scenario_token: NOT REQUIRED / mismatch observed.
- Stage 7C.2A simple_planner × 3 distinct logs: PASS.
- Stage 7C.2B simple_planner × 5 distinct logs: PASS.

Important alignment rule:

```text
Stage 7B.4 scene_token should be preserved as source metadata,
but it must not be assumed to equal nuPlan scenario_filter.scenario_tokens.
```

For nuPlan exact reruns, the verified key is:

```text
log_name + actual_nuPlan_scenario_token
```

Validated exact key:

```text
log_name = 2021.05.12.22.00.38_veh-35_01008_01518
Stage 7B.4 scene_token = 165060762e765a5a
actual_nuPlan_scenario_token = 000e00790bc45da7
```

Correct alignment interpretation:

```text
same_log_alignment_passed: true
strict_stage7b_scene_token_match: false
alignment_level: log_name_plus_actual_nuplan_token
alignment_status: PASS_LOG_AND_NUPLAN_TOKEN_RERUN
```

---

## 7. Stage 7C.2 — IDM Longitudinal-Only Planner Profiles

### Purpose

Use IDM Planner to generate three controlled **longitudinal-only** behavior profiles:

```text
idm_longitudinal_conservative
idm_longitudinal_comfort
idm_longitudinal_aggressive
```

These are rule-based positive controls for validating whether Stage 6 BDD/report card can detect controlled longitudinal behavior differences.

### Current Status

```text
Stage 7C.2C-0 native IDM default/conservative/comfort/aggressive smoke: PASS
Stage 7C.2C-1 wrapper smoke, 1 log × 4 planners: PASS
Stage 7C.2C-2 wrapper rollout, 5 logs × 4 planners: PASS
Stage 7C.2 IDM longitudinal-only multi-planner rollout: PASS
```

Validated output:

```text
outputs/stage7c2c2_idm_longitudinal_5logs/
```

Validated shape:

```text
simulated_ego_seq.npy:       [5, 4, 149, 8]
simulated_ego_seq_mask.npy:  [5, 4, 149]
```

Planner axis:

```text
0 simple_planner
1 idm_longitudinal_conservative
2 idm_longitudinal_comfort
3 idm_longitudinal_aggressive
```

Validated metrics:

```text
warnings: []
official_success_count: 20
trajectory_rows: 2980
msgpack_simulation_log_files_found: 20
msgpack_simulation_log_files_parsed: 20
msgpack_trajectory_rows_extracted: 2980
valid_timestep_count: 2980
missing_pair_count: 0
pseudo_rollout: false
uses_official_nuplan_simulation: true
alignment_pass_ratio: 1.0
same_log_alignment_passed: true
strict_stage7b_scene_token_match: false
alignment_level: log_name_plus_actual_nuplan_token
```

### Interpretation Guardrail

IDM profiles are **not full driving styles**.

They are only longitudinal rule-based positive controls. They are suitable for:

```text
following
lead_brake_response
queue_approach
cutin_response_partial_longitudinal
yield_conflict_partial_longitudinal
```

They are not sufficient for:

```text
lane_change willingness
lane_change sharpness
overtake execution
hesitation
target-lane rear-gap pressure
full courtesy / yielding behavior
```

Therefore Stage 7C.2 only validates whether Stage 6 can detect controlled longitudinal behavior differences from official nuPlan planner rollouts.

Backward-compatible aliases `idm_conservative`, `idm_comfort`, and `idm_aggressive` may remain in code metadata, but documentation should prefer `idm_longitudinal_conservative`, `idm_longitudinal_comfort`, and `idm_longitudinal_aggressive` so they are not confused with complete driving-style profiles.

---

## 8. Stage 7C.3 — PDM Longitudinal + Lateral Planner Profiles

### Purpose

Use PDM Planner or a PDM-compatible planner configuration to generate three **longitudinal + lateral** behavior profiles:

```text
pdm_conservative
pdm_comfort
pdm_aggressive
```

Unlike IDM, PDM should be used to cover behavior dimensions involving both longitudinal and lateral decisions.

### Required Style Coverage

PDM profiles should attempt to cover:

```text
longitudinal comfort
following distance
speed assertiveness
braking response
lane-change willingness
lane-change sharpness
overtake opportunity / execution
lateral stability
gap acceptance
interaction with adjacent-lane traffic
yield / courtesy proxy behavior where supported
```

### Expected Planner Profiles

The exact PDM parameters depend on the available PDM implementation, but the intended behavior profiles are:

```text
pdm_conservative:
  lower speed target
  larger following buffer
  larger lane-change gap requirement
  smoother acceleration / braking
  lower lane-change willingness
  stronger courtesy / yielding tendency if configurable

pdm_comfort:
  moderate speed target
  moderate following buffer
  smooth longitudinal response
  smooth lateral response
  balanced lane-change decision

pdm_aggressive:
  higher speed target
  smaller following buffer
  smaller lane-change gap requirement
  stronger acceleration allowance
  higher lane-change willingness
  more assertive overtaking / merging behavior if configurable
```

### Required Output

PDM should produce the same official simulation output structure as IDM:

```text
outputs/stage7c3_pdm_full_style_5logs/
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/**/*.msgpack.xz
```

Expected planner axis:

```text
simple_planner or expert reference
pdm_conservative
pdm_comfort
pdm_aggressive
```

### PASS Criteria

PDM Stage 7C.3 passes only if:

```text
pseudo_rollout == false
uses_official_nuplan_simulation == true
official_success_count == N × P
missing_pair_count == 0
msgpack_simulation_log_files_parsed == N × P
simulated_ego_seq.npy has shape [N, P, T, C]
planner metadata clearly records PDM profile parameters
```

### Interpretation

PDM is the first Stage 7 planner family intended to support fuller behavior-style coverage than IDM because it can include lateral and interaction-related decisions.

However, PDM results should still be described as planner-profile validation, not human-driver validation.

---

## 9. Stage 7C.4 — ML Planner Longitudinal + Lateral Profiles

### Purpose

Use nuPlan ML Planner, or a nuPlan-compatible learned planner, to generate three learned planner behavior profiles:

```text
ml_planner_conservative
ml_planner_comfort
ml_planner_aggressive
```

This stage is the bridge from rule-based planner validation to E2E / learned-policy validation.

### Possible ML Planner Sources

Possible sources include:

```text
nuPlan official ml_planner
a trained lightweight local neural planner
a checkpointed planner wrapped as nuPlan AbstractPlanner
a cloned ML planner with different checkpoints / cost weights / policy heads
```

### Required Style Coverage

ML planner profiles should include both longitudinal and lateral behavior:

```text
speed profile
following behavior
braking response
comfort / jerk
lane-change willingness
lateral smoothness
gap acceptance
overtake behavior
hesitation / assertiveness
yield / courtesy proxy behavior
```

### Expected ML Planner Profiles

The exact implementation depends on available model control knobs. Acceptable ways to create profiles include:

```text
different trained checkpoints
different planner heads
different cost weights
different sampling / scoring preferences
different target-speed / comfort / lateral-cost parameters
```

But the resulting profiles must be documented as planner/model variants, not as human personality labels.

### Required Output

```text
outputs/stage7c4_ml_planner_full_style_5logs/
├── simulated_ego_seq.npy
├── simulated_ego_seq_mask.npy
├── simulated_ego_seq_index.json
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── simulation_schema.json
├── simulation_report.md
├── warnings.json
└── official_nuplan_runs/**/*.msgpack.xz
```

Expected planner axis:

```text
simple_planner or expert reference
ml_planner_conservative
ml_planner_comfort
ml_planner_aggressive
```

### PASS Criteria

ML Planner Stage 7C.4 passes only if:

```text
official nuPlan simulation is used
pseudo_rollout == false
all scenario-planner pairs succeed
all msgpack logs are parsed
trajectory tensor has [N, P, T, C]
planner metadata records model/checkpoint/profile configuration
```

### Interpretation

ML planner profiles provide stronger evidence than IDM/PDM because they are closer to learned E2E-style behavior. However, final E2E conclusions require careful documentation of model source, input adapter, checkpoint identity, and simulation configuration.

---

## 10. Stage 7D — Export Full Stage 6-Compatible Dataset

### Definition

Stage 7D is not a new BDD implementation.

Stage 7D converts official Stage 7C planner rollout outputs into a full Stage 6-compatible sharded dataset.

### Purpose

Stage 7D exists so that Stage 6 can be reused without rewriting BDD/report-card logic.

For each Stage 7C planner family, Stage 7D should export:

```text
Stage 7C.2 IDM outputs     -> Stage 6-compatible IDM dataset
Stage 7C.3 PDM outputs     -> Stage 6-compatible PDM dataset
Stage 7C.4 ML outputs      -> Stage 6-compatible ML dataset
```

### Mandatory Outputs

Each Stage 7D export must include:

```text
shard_manifest.json
feature_schema.json
planner_policy_indices/*.npy
shards/shard_000/ego_seq.npy
shards/shard_000/neighbor_seq.npy
shards/shard_000/neighbor_slot_ids.npy
shards/shard_000/interaction_feat_style.npy
shards/shard_000/metadata.csv
stage7d_export_schema.json
warnings.json
export_report.md
```

These files are mandatory. `neighbor_seq.npy` and `neighbor_slot_ids.npy` must not be treated as optional for the thesis pipeline.

### Row Semantics

Each row corresponds to:

```text
one scenario × one planner-controlled nuPlan ego rollout
```

For current IDM output:

```text
5 logs × 4 planners = 20 rows
NOT 5 logs × 4 planners × num_agents
```

Stage 5 / Stage 6 Waymo preprocessing may expand many road participants into ego-like samples to increase behavior-embedding training data. Stage 7 must not use that expansion because official nuPlan IDM / PDM / ML planners control only the nuPlan ego vehicle. Background road participants remain neighbor context only.

For an input tensor `simulated_ego_seq.npy` with shape `[5, 4, 149, 8]`, the Stage 7D export must therefore produce:

```text
ego_seq.npy:                    [20, T, ego_dim]
neighbor_seq.npy:               [20, K, T, 9]
neighbor_slot_ids.npy:          [20, K]
interaction_feat_style.npy:     [20, F]
metadata.csv rows:              20
```

### Required Alignment

All of the following must align row-by-row:

```text
ego_seq.npy
neighbor_seq.npy
neighbor_slot_ids.npy
interaction_feat_style.npy
metadata.csv
planner_policy_indices/*.npy
```

### Required Stage 6-Compatible Structure

```text
outputs/stage7d_stage6_dataset_<planner_family>/
├── shard_manifest.json
├── feature_schema.json
├── planner_policy_indices/
│   ├── reference_or_simple.npy
│   ├── conservative.npy
│   ├── comfort.npy
│   └── aggressive.npy
├── shards/
│   └── shard_000/
│       ├── ego_seq.npy
│       ├── neighbor_seq.npy
│       ├── neighbor_slot_ids.npy
│       ├── interaction_feat_style.npy
│       └── metadata.csv
├── stage7d_export_schema.json
├── warnings.json
└── export_report.md
```

### PASS Criteria

Stage 7D passes only if:

```text
pseudo_rollout == false
uses_official_nuplan_simulation == true
ego_seq.npy exists
neighbor_seq.npy exists
neighbor_slot_ids.npy exists
interaction_feat_style.npy exists
metadata.csv exists
feature_schema.json exists
shard_manifest.json exists
planner_policy_indices exist for all planner profiles
total rows == N × P
no multi-agent ego expansion: background/neighbor agents do not create rows
all arrays and metadata have consistent row counts
stage7d_export_schema.json records row_semantics == "scenario_planner_controlled_ego_rollout"
stage7d_export_schema.json records ego_definition == "nuPlan planner-controlled ego vehicle only"
stage7d_export_schema.json records neighbor_definition == "background road participants used only as context"
stage7d_export_schema.json records multi_agent_ego_expansion == false
stage7d_export_schema.json records total_rows_expected == num_scenarios * num_planners
warnings.json validation.total_rows == num_scenarios * num_planners
warnings.json validation.no_multi_agent_ego_expansion == true
warnings.json validation.neighbor_agents_used_as_context_only == true
all planner profiles have non-empty index arrays
neighbor_layout == ego_centric_relative
neighbor_channels == [rel_x, rel_y, rel_vx, rel_vy, distance, bearing, heading_rel, speed, valid]
metadata.csv preserves planner profile fields from simulated_planner_metadata.csv
metadata.csv maps db_name/scenario_id/scene_token/sample_id/scenario_type from scenario_planner_index.csv
interaction_feat_style.npy contains no +/-inf; undefined neighbor-derived values use NaN
warnings.json records validation.pass == true
```

### Important Note on Neighbor Tracks

If the nuPlan simulation uses nonreactive / log-replay background traffic, world-coordinate neighbor trajectories may be identical across planner variants in the same scenario.

However, `neighbor_seq.npy` must still be recomputed relative to each planner’s simulated ego trajectory. Stage 7D export requires this as an upstream extraction contract: materialize `stage7d_neighbor_seq.npy` and `stage7d_neighbor_slot_ids.npy` from official Stage 7C msgpack observations or by reloading nuPlan scenarios with `log_name + actual_nuPlan_scenario_token` before running the exporter. The exporter may accept explicit `--neighbor_seq_path` / `--neighbor_slot_ids_path`, but it must fail rather than fabricate neighbors when these audited tensors are absent.

Therefore:

```text
same world neighbor tracks
different simulated ego trajectories
different ego-centric relative neighbor_seq
```

This is required for Stage 6C task-conditioned BDD. Required planners are configurable via `--required_planners`, so IDM is the default positive-control set but future PDM / ML families can use their own planner axis without code changes.

### Non-Canonical Stage 7D Diagnostic

`tools/stage7d_validate_official_planner_bdd.py` may remain as a smoke diagnostic for official planner tensor sanity checking, but it is not the canonical final BDD path.

Canonical BDD/report-card evaluation must happen through Stage 6 reuse in Stage 7F.

---

## 11. Stage 7E — Embedding Export / Manifest Construction

### Definition

Stage 7E applies the existing Stage 5 / Stage 6 embedding pipeline to Stage 7D exported datasets.

Stage 7E should not define a new embedding model unless explicitly needed. The default should be to reuse the existing trained behavior embedding encoder.

### Inputs

```text
Stage 7D Stage 6-compatible dataset
trained Stage 5 behavior encoder
feature_schema.json
shard_manifest.json
planner_policy_indices/*.npy
```

### Outputs

```text
embedding_manifest.json
embedding.npy or embedding shards
stage7e_embedding_report.md
stage7e_schema.json
warnings.json
```

### Purpose

Stage 7E makes planner-generated nuPlan data available to Stage 6 BDD scripts.

---

## 12. Stage 7F — Reuse Stage 6 BDD / Report Card Engine

### Definition

Stage 7F reuses Stage 6 modules directly.

It should not reimplement BDD, MMD, task-conditioned BDD, scenario-balanced BDD, or report card logic.

### Reused Stage 6 Modules

Stage 7F should reuse:

```text
tools/stage6_compare_unpaired_style.py
tools/stage6_generate_report_card.py
tools/stage6b_compare_baselines.py
tools/stage6b_scenario_balanced_bdd.py
tools/stage6c_build_behavior_events_v2.py
tools/stage6c_task_conditioned_bdd_report.py
```

### Comparisons

For each planner family, Stage 7F should run Stage 6 comparisons such as:

```text
IDM:
  idm_longitudinal_conservative vs idm_longitudinal_comfort
  idm_longitudinal_conservative vs idm_longitudinal_aggressive
  idm_longitudinal_comfort vs idm_longitudinal_aggressive

PDM:
  pdm_conservative vs pdm_comfort
  pdm_conservative vs pdm_aggressive
  pdm_comfort vs pdm_aggressive

ML Planner:
  ml_planner_conservative vs ml_planner_comfort
  ml_planner_conservative vs ml_planner_aggressive
  ml_planner_comfort vs ml_planner_aggressive
```

### Required Outputs

Each comparison should reuse Stage 6 output style:

```text
bdd_summary.json
category_delta.csv
feature_delta.csv
scenario_slice_delta.csv
task_conditioned_bdd.csv
top_drift_cases.csv
style_report_card.md
warnings.json
```

### Interpretation

Stage 7F is where BDD/report card conclusions are made.

Stage 7D only exports data. Stage 7E exports embeddings. Stage 7F runs the canonical Stage 6 evaluation engine.

---

## 13. Stage 7G — Final Stage 7 Thesis Evidence

Stage 7G consolidates all planner families into thesis-ready evidence.

### Final Evidence Structure

```text
1. IDM longitudinal-only validation
   Shows Stage 6 BDD/report card detects controlled longitudinal planner differences.

2. PDM longitudinal + lateral validation
   Shows Stage 6 task-conditioned BDD detects richer full-style planner differences.

3. ML Planner longitudinal + lateral validation
   Shows the method extends from rule-based planners to learned planner outputs.

4. Cross-planner-family analysis
   Compares whether BDD/report card behaves consistently across IDM, PDM, and ML planner data.

5. Limitations
   Documents nuPlan mini scale, nonreactive/reactive agent configuration, planner parameterization limits, and difference between planner profile labels and human driving styles.
```

### Final Claim Boundary

Stage 7 may claim:

```text
Using official nuPlan simulation, we generated controllable same-scenario planner behavior data and showed that the existing Stage 6 BDD/report-card engine can detect planner-induced behavior/style differences across longitudinal-only IDM, richer PDM, and learned ML planner profiles.
```

Stage 7 must not overclaim:

```text
IDM proves full driving style.
Planner labels equal human driver personality.
5-log mini results are a full benchmark.
Feature-only smoke metrics are equivalent to final BDD.
Pseudo rollout is acceptable.
```

---

## 14. Immediate Next Actions

### Next Engineering Step

Implement Stage 7D full Stage 6-compatible dataset export for the validated IDM output:

```text
Input:
outputs/stage7c2c2_idm_longitudinal_5logs/

Output:
outputs/stage7d_stage6_dataset_idm_longitudinal_5logs/
```

The Stage 7D export must include ego, neighbor, neighbor slot IDs, interaction features, metadata, feature schema, shard manifest, and planner policy indices.

### Next Documentation Step

`QUICK_REFERENCE.md` should be updated after the Stage 7D adapter implementation is finalized, with exact commands and PASS criteria for:

```text
Stage 7D: export Stage 6-compatible dataset
Stage 7E: build embeddings / manifest
Stage 7F: reuse Stage 6 BDD/report card scripts
```

## Stage 7E/7F-IDM smoke：Stage 6 BDD/report-card bridge validation

Stage 7E/7F-IDM smoke 复用既有 Stage 5 / Stage 6 embedding、BDD 和 report-card 逻辑，目标是验证 Stage 7D 导出的 IDM official nuPlan simulation 数据能够被 Stage 6 评估引擎直接消费。

### Stage 7E：embedding export

Stage 7E 输入 Stage 7D 的 Stage 6-compatible dataset：

```text
outputs/stage7d_stage6_dataset_idm_5logs/
```

运行命令示例：

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --dataset_dir outputs/stage7d_stage6_dataset_idm_5logs \
  --output_dir outputs/stage7e_idm_embeddings_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_embedding_model/model.pt \
  --max_neighbors 5 \
  --overwrite
```

输出：

```text
outputs/stage7e_idm_embeddings_5logs/
embedding.npy
embedding_manifest.json
metadata.csv
planner_policy_indices/
warnings.json
embedding_report.md
```

Stage 7E 不改变 Stage 7D row semantics：one row = one scenario × one planner-controlled nuPlan ego rollout。neighbor 只作为 context 输入既有 encoder，不做 multi-agent ego expansion。

### Stage 7F：reuse Stage 6 BDD/report-card

Stage 7F 使用 Stage 7E embedding 和 Stage 7D planner_policy_indices，调用既有 Stage 6 BDD/report-card 工具，不在 Stage 7 中重新实现 BDD。

运行命令示例：

```bash
python tools/stage7f_run_idm_stage6_bdd_report.py \
  --dataset_dir outputs/stage7d_stage6_dataset_idm_5logs \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs \
  --output_dir outputs/stage7f_idm_bdd_report_5logs \
  --overwrite
```

必跑对比：

1. `idm_longitudinal_conservative` vs `idm_longitudinal_comfort`
2. `idm_longitudinal_conservative` vs `idm_longitudinal_aggressive`
3. `idm_longitudinal_comfort` vs `idm_longitudinal_aggressive`

每个 comparison 子目录应至少生成：

```text
bdd_summary.json
style_report_card.md
feature_delta.csv
category_delta.csv
```

若后续 Stage 6C event/task-conditioned 输入也存在，可继续扩展为 task-conditioned report；当前 5-log smoke 的最低目标是 BDD/report-card 接口验证。

### Interpretation boundary

Stage 7E/7F-IDM smoke 验证端到端桥接：

```text
official nuPlan simulation → Stage 6-compatible data → embedding → BDD/report card
```

该 smoke 不替代后续 PDM 或 ML Planner 实验，不重新定义 row 语义，不训练 Stage 7 专用 embedding，也不在 Stage 7 中重新实现 BDD。5-log 输出只能作为 exploratory positive-control evidence；不要过度声称统计显著性。

### Stage 7E Stage 5D-compatible embedding input contract

Stage 7D is a data export stage: it exports a Stage 6-compatible evaluation dataset with one row per `scenario × planner-controlled ego rollout`; background agents remain context only and must not be expanded into ego rows.  Stage 6 BDD/report-card modules consume embedding vectors plus aligned feature/metadata artifacts; they do not consume raw `ego_seq.npy` / `neighbor_seq.npy` tensors directly as the final BDD representation.

The current recommended Stage 5 model is Stage 5D-balanced-v2 (`ContextFlattenGRUEncoder`).  Its training input was `context_traj.npy` built by `tools/build_waymo_5neighbor_context_dataset.py`, where each frame is:

```text
ego_seq[8] + 5 neighbor slots × 15 neighbor channels = 83 channels
```

The 83-D `context_traj.npy` does **not** include map/lane/ODD channels.  Lane-aware assignment affected which five neighbors were selected, but map/lane/ODD features are not appended to the encoder tensor.  `interaction_feat_style.npy` is used by Stage 5/6 evaluation and report-card metrics; it is not an input channel to `ContextFlattenGRUEncoder`.

Stage 7E must therefore build a Stage 5D checkpoint-compatible `[rows, T, 83]` `context_traj.npy` before embedding inference.  The thesis path is:

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --dataset_dir outputs/stage7d_stage6_dataset_idm_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs \
  --context_layout stage5d83 \
  --overwrite
```

`--context_layout pad_to_checkpoint_dim` is allowed only for smoke/interface validation.  It must not be used as final thesis evidence, because zero-padding an `ego_neighbor9` tensor does not recreate the Stage 5D 83-channel schema.  A valid Stage 7E thesis run writes `stage7e_context_schema.json` and `warnings.json` with:

```text
context_layout_used = stage5d83
context_padded_to_checkpoint_dim = false
stage5d_schema_matched = true
```

## Stage 7 corrected data architecture: nuPlan Stage 5D-compatible context builder

The corrected Stage 7 architecture cuts into the pipeline at the Stage 5 sample/context-building level, not after a Stage 7D neighbor export.  Official nuPlan simulation replaces Waymo data for the planner-controlled ego rollout only:

```text
Stage 7C official nuPlan simulation
  → Stage 7E nuPlan 5-neighbor context builder
  → Stage 5D best context encoder
  → tools/export_context_row_embeddings.py / Stage 7E thin wrapper
  → Stage 7F existing Stage 6 BDD/report-card scripts
```

Stage 7D remains a Stage 6-compatible evaluation dataset export.  Stage 7E is now the Stage 5D-compatible nuPlan context dataset plus embedding export.  Stage 7F reuses the existing Stage 6 BDD/report-card scripts.  Stage 7 must not reimplement Stage 6 BDD and must not train a Stage 7-only embedding model.

Row semantics are intentionally different from Waymo Stage 5 multi-agent ego expansion.  Waymo Stage 5 may use `row = scenario × agent × window`.  nuPlan Stage 7 uses:

```text
row = scenario × planner × planner-controlled nuPlan ego rollout
```

Background agents are context only and must not be expanded into ego rows.  For the current IDM 5-log smoke this gives `5 scenarios × 4 planners = 20 rows`.

The Stage 5D best model was trained on `context_traj.npy [N,T,83]` from `tools/build_waymo_5neighbor_context_dataset.py`.  The 83-D contract is:

```text
ego 8 channels + 5 semantic neighbor slots × 15 channels = 83
```

The ego channels are `ego_x, ego_y, ego_vx, ego_vy, ego_heading, ego_speed, ego_accel, ego_yaw_rate`.  The semantic neighbor slots are `front, rear, left_front, left_rear, right_front`; each uses `valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y, closing, ttc, thw, speed, accel, heading_rel, yaw_rate`.

`context_traj.npy` has no map/lane/ODD channels unless the original Stage 5 builder is changed; in the current Stage 5D contract these features are not appended to the encoder tensor.  `interaction_feat_style.npy` is for reports/evaluation and Stage 6 BDD/report-card metrics, not an encoder input channel.  Stage 6 consumes exported `embedding.npy`, aligned `interaction_feat_style.npy`, `metadata.csv`, and planner/policy A/B indices; it does not feed raw Stage 7D ego/neighbor tensors directly into BDD.

The new builder is:

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --max_neighbors_for_context 5 \
  --slot_assignment_method geometric_proxy \
  --same_lane_abs_y 1.8 \
  --adjacent_lane_min_abs_y 1.5 \
  --overwrite
```

It writes `ego_seq.npy`, `context_traj.npy`, `interaction_feat_style.npy`, `metadata.csv`, `feature_schema.json`, `stage5d_context_schema.json`, `shard_manifest.json`, `planner_policy_indices/*.npy`, `warnings.json`, `context_build_report.md`, and `slot_assignment_report.md` directly from official Stage 7C simulation artifacts and `official_nuplan_runs/**/*.msgpack.xz` tracked objects.

The initial slot assignment is `geometric_proxy`: same-lane candidates satisfy `abs(rel_y) <= same_lane_abs_y`; `front` and `rear` use nearest positive/negative `rel_x`; left/right slots use `rel_y` sign and nearest positive/negative `rel_x`.  This directly assigns semantic slots from candidate tracked objects relative to the planner-controlled ego.  It must not take a distance top-K tensor such as `neighbor_seq[:, :5]` and relabel those slots as `front/rear/left_front/left_rear/right_front`.  Because this is a geometric proxy rather than exact Waymo lane-aware assignment, reports and `warnings.json` must say so and only set `stage5d_slot_semantics_verified` when slot sanity checks pass.

Embedding export should use the direct context-dataset mode:

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs \
  --overwrite
```

In `--context_dataset_dir` mode, Stage 7E loads `context_traj.npy` directly, checks `checkpoint["context_dim"] == context_traj.shape[-1]`, exports `embedding.npy`, and copies `metadata.csv` plus `planner_policy_indices/*.npy`.  It does not rebuild context from Stage 7D `neighbor_seq`.
