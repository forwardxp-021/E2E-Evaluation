# Stage 7 — nuPlan Official Simulation Data Generation and Stage 5/6 Reuse Roadmap

## 0. Latest Architecture Decision

Stage 7 is no longer treated as a separate implementation of Stage 5 / Stage 6 logic.

The corrected architecture is:

```text
Stage 5D common context core = single source of truth
Waymo data source adapter     -> Stage 5D common core -> context_traj.npy [N,T,83]
nuPlan simulation adapter     -> Stage 5D common core -> context_traj.npy [N,T,83]
                                         ↓
                              Stage 5D best encoder
                                         ↓
                                  embedding.npy
                                         ↓
                         existing Stage 6 BDD/report-card engine
```

The most important rule:

```text
nuPlan must replace Waymo at the Stage 5 sample/context-building boundary,
not by reimplementing Stage 5D schema inside Stage 7 and not by post-hoc patching
Stage 7D top-K neighbor tensors into a fake 83-D context tensor.
```

Stage 7 may change the **data source** and **row semantics**, but it must not change the Stage 5D input contract or Stage 6 evaluation logic.

---

## 1. Top-Level Purpose

Stage 7 的核心目的不是重新实现 BDD，也不是重新实现 report card。

Stage 7 的核心目的，是使用 **nuPlan official simulation** 生成可控的 planner / policy 行为数据，并将这些数据接入既有 Stage 5D embedding 与 Stage 6 evaluation pipeline。

Stage 7 的定位是：

```text
Stage 5D = canonical behavior embedding input contract and trained encoder
Stage 6  = canonical BDD / report-card evaluation engine
Stage 7  = controllable nuPlan planner-generated data source and adapter
```

Stage 7 要解决 Stage 6 Waymo 数据的核心限制：

```text
Stage 6 Waymo 数据：
  来自真实道路 logged trajectories；
  驾驶员未知；
  无法控制同一场景下不同驾驶风格；
  Waymo Stage 5 可做 multi-agent ego expansion。

Stage 7 nuPlan 数据：
  来自 official nuPlan closed-loop simulation；
  可以在同一 scenario 下运行不同 planner / policy；
  可以构造可控 conservative / comfort / aggressive planner profiles；
  planner 只控制 nuPlan ego；
  background agents 只能作为 context，不能扩展为 ego rows。
```

Stage 7 所有 planner 数据都必须来自 official nuPlan simulation。禁止使用 pseudo rollout、numpy trajectory rewriting、offline trajectory interpolation 来冒充 closed-loop simulation。

---

## 2. Non-Negotiable Data-Architecture Guardrails

### 2.1 Stage 5D Core Is the Single Source of Truth

Stage 7 must not maintain a copied implementation of Stage 5D schema, slot names, channel order, derived formulas, or lane-aware assignment logic.

The following must come from a shared Stage 5D core or directly from existing Stage 5 modules:

```text
SLOT_NAMES
EGO_CHANNELS
NEIGHBOR_CHANNELS
CONTEXT_DIM = 83
lane-aware slot assignment
geometric fallback assignment
neighbor 15-D channel construction
closing / TTC / THW / accel / yaw-rate formulas
context_traj construction
schema generation
validation
```

Recommended common layer:

```text
tools/stage5d_context_core.py
```

or direct reuse of existing modules where appropriate:

```text
tools/lane_aware_assignment.py
tools/waymo_lane_utils.py
tools/build_waymo_5neighbor_context_dataset.py
```

Stage 7 nuPlan code should be an adapter that converts nuPlan official simulation artifacts into standardized ego / candidate / lane inputs, then calls the Stage 5D common core.

### 2.2 Waymo and nuPlan Differ in Row Semantics, Not in Context Contract

Waymo Stage 5 may use:

```text
row = scenario × agent × window
```

nuPlan Stage 7 must use:

```text
row = scenario × planner × planner-controlled nuPlan ego rollout
```

Background agents in nuPlan are context only:

```text
multi_agent_ego_expansion = false
neighbor_agents_used_as_context_only = true
```

For the current IDM 5-log smoke:

```text
5 scenarios × 4 planners = 20 rows
```

not:

```text
5 scenarios × 4 planners × num_agents
```

### 2.3 Exact Stage 5D Slot Schema

The original Stage 5 5-neighbor slot schema is:

```text
front
left_front
left_rear
right_front
right_rear
```

It is **not**:

```text
front
rear
left_front
left_rear
right_front
```

This slot order is part of the Stage 5D model input contract. Any Stage 7 output using the old `rear` schema is smoke-only / invalid for final thesis evidence.

### 2.4 Exact Stage 5D Context Contract

Stage 5D best model input:

```text
context_traj.npy [N, T, 83]
```

Dimensional contract:

```text
83 = ego 8 channels + 5 semantic neighbor slots × 15 channels
```

Ego 8 channels:

```text
ego_x
ego_y
ego_vx
ego_vy
ego_heading
ego_speed
ego_accel
ego_yaw_rate
```

Neighbor slots:

```text
front
left_front
left_rear
right_front
right_rear
```

Neighbor 15 channels per slot:

```text
valid
rel_x
rel_y
rel_vx
rel_vy
distance
delta_x
delta_y
closing
ttc
thw
speed
accel
heading_rel
yaw_rate
```

`context_traj.npy` does not include map / lane / ODD channels. Lane-aware information affects **which neighbors are selected into the five slots**, but map/lane/ODD features are not appended to the Stage 5D encoder tensor.

`interaction_feat_style.npy` is for feature explanation, report card, and Stage 6 evaluation. It is not an input channel to `ContextFlattenGRUEncoder`.

### 2.5 Derived Channels Must Match Stage 5 Formulas

Do not call every non-raw field a nuPlan-specific proxy. In Stage 5 Waymo, several fields are also derived from trajectories.

Required classification in schemas:

```text
direct_from_state
derived_same_as_stage5
approximated_or_not_stage5_matched
```

Expected formula parity:

```text
delta_x = rel_x
delta_y = rel_y
speed = hypot(vx, vy)
accel = finite difference of speed
yaw_rate = finite difference of heading
THW = distance / ego_speed with Stage 5 caps / eps
closing = Stage 5 closing formula
TTC = Stage 5 TTC formula based on closing
```

Validation must explicitly record:

```text
stage5d_derived_formula_matched
stage5d_closing_formula_matched
stage5d_ttc_formula_matched
stage5d_delta_xy_formula_matched
stage5d_accel_yaw_rate_formula_matched
slot_id_switch_rate_by_slot
```

If slot ID switching causes finite differences to be computed across different agents, `accel` and `yaw_rate` must not be overclaimed as exact Stage5D-equivalent.

---

## 3. Updated Compact Roadmap

| Stage | Purpose | Main data source | Main output | Status |
|---|---|---|---|---|
| 7A | nuPlan readiness | nuPlan DB / map / devkit | readiness evidence | PASS |
| 7B | pre-simulation nuPlan context / map foundation | nuPlan logs / maps | ego + neighbor + context features | PASS |
| 7C | official planner data generation | nuPlan official simulation | planner rollout tensors + official logs | PASS for IDM 5-log smoke |
| 7D | Stage 6-compatible evaluation dataset export | Stage 7C official simulation outputs | ego / neighbor / interaction / metadata / indices | PASS for IDM 5-log smoke |
| 7E-core | nuPlan adapter to Stage 5D context contract | Stage 7C official simulation + nuPlan map | context_traj.npy [N,T,83] | Stage5D common-core refactor implemented; lane-aware assignment path implemented; requires runtime validation with real --nuplan_map_root and fallback-rate diagnostics before final thesis evidence. |
| 7E-embed | embedding export | Stage 7E-core context + Stage 5D best model | embedding.npy + manifest | PASS for direct context-dataset smoke |
| 7F | reuse Stage 6 BDD / report-card engine | Stage 7E embeddings + Stage 7D metadata/features | BDD, task-BDD, report cards | NEXT |
| 7G | final Stage 7 thesis evidence | 7C / 7D / 7E / 7F | final planner-style validation evidence | TODO |

---

## 4. Stage 7A — nuPlan Readiness

Stage 7A = nuPlan environment / data / map / scenario / simulation API readiness.

Purpose:

- Check nuPlan DB access.
- Check map root access.
- Check scenario metadata.
- Check expert ego pose and object extraction.
- Discover available simulation APIs and planner classes.

Current status: PASS.

Stage 7A is an infrastructure-readiness stage. It does not by itself prove planner-induced or E2E-induced behavior drift.

---

## 5. Stage 7B — Pre-Simulation nuPlan Context Foundation

Stage 7B builds a strict, row-aligned, auditable scenario context dataset for downstream simulation selection, alignment, and diagnostics.

Sub-stages:

- 7B.1 expert ego/object export.
- 7B.2 dynamic context conversion.
- 7B.3 map/ODD-lite feature builder.
- 7B.4 dynamic + map/ODD merge/alignment.

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

Stage 7B is not itself official simulation. It is context and alignment foundation.

Important: Stage 7B may already have `context_traj.npy [N,T,83]`, but final planner-profile evidence must come from Stage 7C official simulation outputs, not from expert/log replay alone.

---

## 6. Stage 7C — Official nuPlan Planner Data Generation

Stage 7C runs planner / policy variants in official nuPlan simulation on the same selected scenarios.

Planner families:

```text
Stage 7C.2 — IDM longitudinal-only planner profiles
Stage 7C.3 — PDM longitudinal + lateral planner profiles
Stage 7C.4 — ML planner longitudinal + lateral planner profiles
```

Common output:

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

Tensor contract:

```text
simulated_ego_seq.npy:       [N, P, T, C]
simulated_ego_seq_mask.npy:  [N, P, T]
```

PASS criteria:

```text
pseudo_rollout == false
uses_official_nuplan_simulation == true
official_success_count == N × P
missing_pair_count == 0
msgpack_simulation_log_files_parsed == N × P
```

### 6.1 Stage 7C.1 — Official Simulation Smoke and Alignment Foundation

Current status: PASS.

Validated facts:

- Stage 7C.1A official simulation smoke: PASS.
- Stage 7C.1B official msgpack trajectory export: PASS.
- Stage 7C.1C exact log + actual nuPlan scenario token wrapper smoke: PASS.
- Stage 7C.2A simple_planner × 3 distinct logs: PASS.
- Stage 7C.2B simple_planner × 5 distinct logs: PASS.

Alignment rule:

```text
Stage 7B.4 scene_token should be preserved as source metadata,
but it must not be assumed to equal nuPlan scenario_filter.scenario_tokens.
```

For nuPlan exact reruns, the verified key is:

```text
log_name + actual_nuPlan_scenario_token
```

### 6.2 Stage 7C.2 — IDM Longitudinal-Only Planner Profiles

Purpose: generate controlled longitudinal positive controls:

```text
simple_planner
idm_longitudinal_conservative
idm_longitudinal_comfort
idm_longitudinal_aggressive
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

Validated status:

```text
Stage 7C.2 IDM longitudinal-only multi-planner rollout: PASS
warnings: []
official_success_count: 20
msgpack_simulation_log_files_parsed: 20
pseudo_rollout: false
uses_official_nuplan_simulation: true
```

Interpretation guardrail:

IDM profiles are longitudinal-only positive controls. They are suitable for following, lead-brake response, queue approach, and partial cut-in longitudinal response. They are not sufficient for full driving-style validation involving lane change, overtaking, lateral assertiveness, or full courtesy/yielding behavior.

### 6.3 Stage 7C.3 — PDM Longitudinal + Lateral Planner Profiles

PDM should cover richer behavior dimensions:

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

Expected planner axis:

```text
simple_planner or expert reference
pdm_conservative
pdm_comfort
pdm_aggressive
```

PDM should produce the same official simulation structure as IDM.

### 6.4 Stage 7C.4 — ML Planner Longitudinal + Lateral Profiles

ML planner profiles bridge from rule-based planner validation to learned-policy / E2E-style validation.

Possible sources:

```text
nuPlan official ml_planner
a trained local neural planner
a checkpointed planner wrapped as nuPlan AbstractPlanner
learned planner variants with different checkpoints / heads / cost weights
```

Expected planner axis:

```text
simple_planner or expert reference
ml_planner_conservative
ml_planner_comfort
ml_planner_aggressive
```

ML planner evidence requires careful documentation of model source, checkpoint identity, input adapter, and simulation configuration.

---

## 7. Stage 7D — Stage 6-Compatible Evaluation Dataset Export

Stage 7D is not a new BDD implementation.

Stage 7D converts official Stage 7C planner rollout outputs into a full Stage 6-compatible sharded evaluation dataset.

Stage 7D outputs are for Stage 6 report-card / BDD feature alignment, not for rebuilding the Stage 5D context tensor.

Mandatory outputs:

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

For IDM 5-log smoke, Stage 7D row semantics:

```text
one row = one scenario × one planner-controlled nuPlan ego rollout
rows = 5 scenarios × 4 planners = 20
```

Expected shapes:

```text
ego_seq.npy:                    [20, T, ego_dim]
neighbor_seq.npy:               [20, K, T, 9]
neighbor_slot_ids.npy:          [20, K]
interaction_feat_style.npy:     [20, F]
metadata.csv rows:              20
```

Stage 7D may use a distance/top-K neighbor tensor for Stage 6C diagnostics and interaction feature export. This top-K tensor must **not** be interpreted as the Stage 5D semantic 5-slot context input.

PASS criteria:

```text
pseudo_rollout == false
uses_official_nuplan_simulation == true
total rows == N × P
no multi-agent ego expansion
background agents context only
all arrays and metadata have consistent row counts
all planner profiles have non-empty index arrays
neighbor_layout == ego_centric_relative
metadata preserves planner profile fields
warnings.json validation.pass == true
```

Current IDM 5-log Stage 7D status: PASS.

---

## 8. Stage 7E-core — nuPlan Adapter to Stage 5D Context Contract

Stage 7E-core is the corrected Stage 5D-compatible context construction path.

It should build:

```text
context_traj.npy [N,T,83]
```

directly from official Stage 7C simulation artifacts and nuPlan map/lane data, through the Stage 5D common core.

### 8.1 Correct Input Boundary

Input:

```text
outputs/stage7c2c2_idm_longitudinal_5logs/
  simulated_ego_seq.npy
  simulated_ego_seq_mask.npy
  scenario_planner_index.csv
  simulated_planner_metadata.csv
  simulation_schema.json
  official_nuplan_runs/**/*.msgpack.xz
```

Output:

```text
outputs/stage7e_nuplan_5neighbor_context_idm_5logs/
  ego_seq.npy
  context_traj.npy
  interaction_feat_style.npy
  metadata.csv
  feature_schema.json
  stage5d_context_schema.json
  shard_manifest.json
  planner_policy_indices/*.npy
  warnings.json
  context_build_report.md
  slot_assignment_report.md
```

### 8.2 Correct Implementation Boundary

**Status (2026-06-17): Stage 7E final embedding cleanup is DONE.** The final embedding path is the clean `context_traj.npy -> --context_dataset_dir` path; legacy `--dataset_dir` / `context_layout` / top-K neighbor reconstruction remains retired.

**DONE architecture item:** nuPlan ego 8D is built through `tools.stage5d_context_core.build_ego_features_8d(...)` from a standard `[x, y, vx, vy, heading, valid]` track window. Stage 7E-core must not import Stage 7D `convert_ego`, because Stage 7D emits world-frame ego channels while Stage 5D CORE defines the local-window ego frame used by the Waymo builder.


`tools/build_nuplan_5neighbor_context_dataset.py` should not own Stage 5D schema constants or formulas.

It should only adapt nuPlan data into standardized structures:

```text
StandardEgoTrack
StandardNeighborTrack / candidate tracked objects
StandardLaneInfo / nuPlan-lane adapter
```

Then it should call the Stage 5D common core for:

```text
slot assignment
neighbor 15-D construction
ego 8-D construction
context_traj assembly
schema / validation
```

### 8.3 Lane-Aware Slot Assignment

Local-frame contract: ego 8D uses one deterministic local window frame (`origin = first valid ego xy`, `base_heading = first valid ego heading`, `dt = median valid simulated time_s delta or 0.1`). Neighbor `rel_x/rel_y/rel_vx/rel_vy` stay per-timestep ego-centric using the current ego world pose and current ego heading, matching the original Waymo Stage 5D builder behavior.


Preferred assignment mode:

```text
lane_aware_with_geometric_fallback
```

The nuPlan builder should reuse Stage 5 lane-aware logic:

```text
tools/lane_aware_assignment.py::assign_neighbors_lane_aware
```

The nuPlan-specific part is a map/lane adapter that converts nuPlan lane / lane_connector / baseline path objects into the LaneInfo-like structure expected by the Stage 5 assignment logic.

Geometric assignment is allowed only as fallback or smoke mode:

```text
geometric_only / geometric_proxy = smoke or fallback, not final preferred method
```

Validation must report actual lane-aware runtime availability, not just slot coverage:

```text
assignment_mode
lane_assignment_available
map_query_success
lane_info_count
fallback_assignment_used_rate
ego_lane_projection_success_rate
candidate_lane_projection_success_rate
slot_sanity_passed
slot_coverage_by_slot
stage5d_slot_schema_matched
stage5d_slot_order_matched
```

Runtime policy:

```text
assignment_mode == lane_aware_only:
  fail loudly if --nuplan_map_root is missing, map_name cannot be queried, lane_info_count == 0, or ego lane projection is unavailable.

assignment_mode == lane_aware_with_geometric_fallback:
  allow geometric fallback for map/projection gaps, but write a loud warning when fallback_assignment_used_rate is high; high fallback is not strong lane-aware thesis evidence.
```

### 8.4 Stage 7E-core PASS Criteria

```text
row_semantics_correct == true
no_multi_agent_ego_expansion == true
background_agents_context_only == true
context_traj.shape == [N,T,83]
stage5d_dim_matched == true
stage5d_channel_schema_matched == true
stage5d_slot_schema_matched == true
stage5d_slot_order_matched == true
stage5d_static_derived_formula_matched == true
stage5d_closing_formula_matched == true
stage5d_ttc_formula_matched == true
stage5d_delta_xy_formula_matched == true
stage5d_temporal_derived_formula_matched is reported conservatively
slot_id_switch_rate_by_slot is reported
context_traj_no_nonfinite == true
planner_indices_non_empty == true
stage5d_core_reused == true
```

Stage 7E no longer requires `stage5d_derived_formula_matched == true` globally as a PASS criterion. Static derived channels (`closing`, `ttc`, `delta_x`, `delta_y`) must match Stage5D CORE formulas exactly, while temporal accel/yaw-rate parity must be reported conservatively. When semantic slot IDs switch within a slot, accel/yaw_rate can be an approximation because the finite difference may span different physical agents; `slot_id_switch_rate_by_slot` must therefore be reported alongside temporal parity.

Current final Stage 7E thesis path:

```text
tools/build_nuplan_5neighbor_context_dataset.py
  -> context_traj.npy [N,T,83]
  -> tools/stage7e_embed_stage6_dataset.py --context_dataset_dir
  -> embedding.npy / embeddings/shard_000000/embeddings.npy
```

Removed legacy ambiguity: the final Stage 7E embedding script no longer exposes a `--dataset_dir` / layout debug bridge and cannot construct encoder context from Stage 7D top-K neighbor tensors.

---

## 9. Stage 7E-embed — Embedding Export

Stage 7E-embed applies the existing Stage 5D encoder to the Stage 7E-core context dataset.

Recommended command:

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs \
  --overwrite
```

In `--context_dataset_dir` mode, Stage 7E must:

```text
load context_traj.npy directly
check checkpoint["context_dim"] == context_traj.shape[-1]
export embedding.npy
copy metadata.csv and planner_policy_indices/*.npy
not rebuild context from Stage 7D neighbor_seq
```

Smoke validation should record:

```text
context_layout_used = stage5d_context_dataset_direct
base_context_layout = context_traj.npy
base_context_dim = 83
checkpoint_context_dim = 83
final_context_dim = 83
context_padded_to_checkpoint_dim = false
does_not_rebuild_context_from_stage7d_neighbor_seq = true
nonfinite_context_values_replaced_with_zero = 0
nonfinite_embedding_values = 0
```

The final script does not expose zero-padding or Stage 7D neighbor reconstruction modes; all embedding evidence must flow through `--context_dataset_dir`.

---

## 10. A/B Indices and Planner Policy Groups

A/B indices specify which rows in `embedding.npy` belong to each planner / policy group.

For current IDM 5-log row order:

```text
row 0  = scenario_0 × simple_planner
row 1  = scenario_0 × idm_longitudinal_conservative
row 2  = scenario_0 × idm_longitudinal_comfort
row 3  = scenario_0 × idm_longitudinal_aggressive
row 4  = scenario_1 × simple_planner
...
```

Expected indices:

```text
simple_planner.npy                = [0, 4, 8, 12, 16]
idm_longitudinal_conservative.npy = [1, 5, 9, 13, 17]
idm_longitudinal_comfort.npy      = [2, 6, 10, 14, 18]
idm_longitudinal_aggressive.npy   = [3, 7, 11, 15, 19]
```

Stage 6 comparisons use these files as A/B groups.

Examples:

```text
A = idm_longitudinal_conservative.npy
B = idm_longitudinal_aggressive.npy
```

No A/B group should be constructed from background agents.

---

## 11. Stage 7F — Reuse Stage 6 BDD / Report Card Engine

Stage 7F reuses Stage 6 modules directly.

It must not reimplement BDD, MMD, task-conditioned BDD, scenario-balanced BDD, or report card logic.

Reused tools:

```text
tools/stage6_compare_unpaired_style.py
tools/stage6_generate_report_card.py
tools/stage6b_compare_baselines.py
tools/stage6b_scenario_balanced_bdd.py
tools/stage6c_build_behavior_events_v2.py
tools/stage6c_task_conditioned_bdd_report.py
```

IDM smoke command:

```bash
python tools/stage7f_run_idm_stage6_bdd_report.py \
  --dataset_dir outputs/stage7d_stage6_dataset_idm_5logs \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs \
  --output_dir outputs/stage7f_idm_bdd_report_5logs \
  --overwrite
```

Required IDM comparisons:

```text
idm_longitudinal_conservative vs idm_longitudinal_comfort
idm_longitudinal_conservative vs idm_longitudinal_aggressive
idm_longitudinal_comfort vs idm_longitudinal_aggressive
```

Each comparison should produce Stage 6-style outputs:

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

Stage 7F is where BDD/report-card conclusions are made. Stage 7D exports evaluation data; Stage 7E exports embeddings; Stage 7F runs the canonical Stage 6 evaluation engine.

---

## 12. Stage 7G — Final Stage 7 Thesis Evidence

Stage 7G consolidates all planner families into thesis-ready evidence.

Final evidence structure:

```text
1. IDM longitudinal-only validation
   Shows Stage 6 BDD/report card detects controlled longitudinal planner differences.

2. PDM longitudinal + lateral validation
   Shows Stage 6 task-conditioned BDD detects richer planner-profile differences.

3. ML Planner longitudinal + lateral validation
   Shows the method extends from rule-based planners to learned planner outputs.

4. Cross-planner-family analysis
   Compares whether BDD/report card behaves consistently across IDM, PDM, and ML planner data.

5. Limitations
   Documents nuPlan scale, nonreactive/reactive agents, planner parameterization limits,
   Stage5D common-core assumptions, lane-aware assignment fallback rates,
   and planner profile labels vs human driving styles.
```

Allowed claim:

```text
Using official nuPlan simulation, we generated controllable same-scenario planner behavior data and showed that the existing Stage 5D embedding plus Stage 6 BDD/report-card engine can detect planner-induced behavior/style differences across longitudinal-only IDM, richer PDM, and learned ML planner profiles.
```

Not allowed:

```text
IDM proves full driving style.
Planner labels equal human driver personality.
5-log mini results are a full benchmark.
Feature-only smoke metrics are equivalent to final BDD.
Pseudo rollout is acceptable.
Stage 7 has its own independent embedding or BDD implementation.
```

---

## 13. Immediate Next Actions

### P0 — Stage 7E Final Thesis Path Cleanup and Runtime Validation

Implemented architectural baseline:

```text
Stage5D common context core = single source of truth.
Waymo and nuPlan are adapters.
```

Current implementation status:

- Stage5D common-core refactor: implemented.
- nuPlan builder imports Stage5D core constants/functions rather than defining a local Stage5D schema.
- Lane-aware assignment path is implemented through the nuPlan lane adapter and `lane_aware_with_geometric_fallback` mode.
- Derived-formula parity is implemented in the common core, but still requires output-level validation on generated nuPlan context artifacts before final thesis evidence.

Current P0 actions:

Stage7E final embedding cleanup: DONE

A. Run nuPlan lane-aware context build with real `--nuplan_map_root`:

```bash
python tools/build_nuplan_5neighbor_context_dataset.py \
  --sim_dir outputs/stage7c2c2_idm_longitudinal_5logs \
  --output_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --assignment_mode lane_aware_with_geometric_fallback \
  --nuplan_map_root /home/forwardxp/00_nuplan_E2E_eva/nuplan/dataset/maps \
  --overwrite
```

B. Verify `warnings.json` / report fields:

```text
validation.pass
map_query_success
lane_info_count
lane_assignment_available
fallback_assignment_used_rate
ego_lane_projection_success_rate
candidate_lane_projection_success_rate
stage5d_core_reused
stage5d_slot_schema_matched
stage5d_slot_order_matched
stage5d_static_derived_formula_matched
stage5d_closing_formula_matched
stage5d_ttc_formula_matched
stage5d_delta_xy_formula_matched
stage5d_temporal_derived_formula_matched
stage5d_accel_yaw_rate_formula_matched
slot_id_switch_rate_by_slot
```

C. Run Stage7E embedding using `--context_dataset_dir`:

```bash
python tools/stage7e_embed_stage6_dataset.py \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --output_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --overwrite
```

D. Run Stage7F BDD/report-card smoke.

### P1 — Stage 7F IDM Smoke

After P0 fixes, run Stage 7F IDM BDD/report-card smoke.

### P2 — Scale and Planner Expansion

After IDM smoke:

```text
increase from 5 scenarios to 50/100+ scenarios
run PDM planner profiles
run ML planner profiles
compare cross-planner-family behavior drift
```

## Stage7E lane-aware failure diagnosis protocol

Stage7 must not implement a separate lane-aware assignment algorithm. The only lane-aware assignment implementation is Stage5D CORE, `tools.lane_aware_assignment.assign_neighbors_lane_aware`. Stage7 nuPlan code is an adapter: it converts nuPlan map objects and tracked objects into Stage5-compatible `LaneInfo` and candidate-state inputs.

When nuPlan shows high geometric fallback or low candidate lane projection success, run:

```bash
python tools/compare_lane_aware_diagnostics.py \
  --waymo_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --nuplan_dir outputs/<stage7e_nuplan_context_output> \
  --out_dir outputs/lane_aware_diagnostic_comparison \
  --max_rows 2000
```

The comparison report must include `lane_assignment_available`, `fallback_assignment_used_rate`, `candidate_projection_success_rate`, `adjacency_source_counts`, `lane_context_quality` counts, rejection reason counts, slot coverage by slot, and slot switch rate by slot for both Waymo Stage5 and nuPlan Stage7E outputs.

Interpretation rule:

- If Waymo `fallback_assignment_used_rate` and `candidate_projection_success_rate` are both unavailable, the comparison is `inconclusive_missing_waymo_metrics`; collect/export comparable Waymo metrics before blaming the nuPlan adapter.
- Compare fallback rates only when both Waymo and nuPlan fallback rates are available.
- Compare candidate projection success only when both Waymo and nuPlan candidate projection success rates are available.
- If both datasets show similar weakness, treat it as a generic Stage5D lane-aware assignment limitation and improve only `tools/lane_aware_assignment.py` / Stage5D CORE so both Waymo and nuPlan benefit.
- If nuPlan is much worse than Waymo under the same Stage5D assignment call, treat it as a nuPlan LaneInfo adapter / map topology / adjacency / projection quality issue and improve only `tools/nuplan_lane_utils.py` or the nuPlan adapter path.

## Stage7E lane-aware adapter diagnostics update

Stage7E must not become a separate lane-aware assignment implementation. The only lane-aware assignment implementation remains Stage5D CORE in `tools/lane_aware_assignment.py`. Stage7E is an adapter: it converts nuPlan map lanes and tracked-object states into Stage5-compatible `LaneInfo` and candidate states, then calls Stage5D CORE.

New diagnostic artifacts:

- `nuplan_lane_projection_debug_summary.json`: bounded summary of ego/candidate projection success, rejection reasons, lane relation counts, and fallback cause counts.
- `nuplan_lane_projection_debug_report.md`: human-readable summary of the same metrics.
- `nuplan_lane_projection_debug.csv`: optional bounded sampled candidate rows, enabled by `--write_projection_debug` and capped by `--debug_projection_sample_rows`, `--debug_projection_max_candidates_per_frame`, and `--debug_projection_max_frames_per_row`.
- `waymo_lane_aware_diagnostics.json/.md/.csv`: comparable Waymo Stage5D diagnostic export from existing outputs.

Interpretation of comparison verdicts:

- `nuplan_adapter_or_map_projection_issue`: comparable Waymo metrics exist and nuPlan is substantially worse in candidate projection success or fallback rate, which points to the nuPlan adapter / map projection path rather than a new Stage7 assignment problem.
- `generic_stage5_lane_aware_limitation_or_dataset_common_issue`: both datasets show similarly low projection success or high fallback, so the limitation may be in shared Stage5D assumptions or common dataset/map conditions.
- `inconclusive_missing_comparable_metrics`: Waymo / nuPlan comparable metrics are missing, so the comparison must not claim a nuPlan-specific issue.
- `no_clear_nuplan_adapter_issue`: metrics are comparable and nuPlan is not clearly worse.

## Stage7E / Stage5 Waymo lane-aware filtering mismatch diagnosis update

Waymo Stage5 clean lane-aware output was built with a strict filtering philosophy: `--assignment_mode lane_aware_only` plus `--drop_if_no_lane_map`, `--drop_if_ego_lane_missing`, `--drop_if_lane_context_bad`, and `--drop_if_lane_context_ambiguous`. Therefore Waymo `fallback=0` means the dataset was filtered to lane-aware-only rows; it is not directly comparable to a nuPlan Stage7E output that preserves all official planner rollout rows and uses `lane_aware_with_geometric_fallback`.

Stage7E nuPlan main output must continue to preserve official row semantics: one row is one `scenario × planner-controlled rollout`. Strict filtering is diagnostic-only unless explicitly written to a named diagnostic output with `--write_strict_filtered_dataset`.

The diagnosis plan is updated as follows:

1. First reproduce the Stage5 filtering philosophy as a nuPlan diagnostic, writing `nuplan_laneaware_strict_filter_summary.json` and `nuplan_laneaware_strict_filter_report.md`.
2. Compare Waymo strict-filtered Stage5 output against nuPlan fallback-preserving output only with a filtering-mismatch warning and downgraded confidence.
3. Use fair strict-filter comparison only when the nuPlan strict-filter diagnostic is provided. Verdicts distinguish `comparable_strict_filter_pass`, `nuplan_strict_filter_low_keep_rate`, and `inconclusive_due_to_filtering_mismatch`.
4. Do not implement Stage7-specific assignment. Stage5D CORE / `tools.lane_aware_assignment.py` remains the only lane-aware assignment implementation.
5. Waymo diagnostic exports must record `filtering_mode`; use `strict_filter_lane_aware_only` only when confirmed by the Stage5 command or a reliable source file. Unknown Waymo filtering mode remains low-confidence / limited-comparability evidence.
6. nuPlan strict-filter diagnostics support `--strict_filter_min_laneaware_ratio`; `1.0` preserves the all-valid-frames behavior, while `0.8` emulates the Stage5 Waymo `--min_valid_ratio 0.8` filtering philosophy for diagnostic comparison.
7. Optional `--strict_filter_ratio_sweep 1.0 0.9 0.8 0.7 0.6` reports threshold sensitivity without creating multiple datasets.

## Stage7F execution order: full main chain before strict-filter sensitivity

Current branch order for `20260611_stage7_conclusion`:

1. Run Stage7F full fallback-preserving main report first.
2. Then run strict-filter ratio=0.8 clean-subset sensitivity if a real filtered context dataset and embedding have been written.
3. Run Stage5 parameter / lane-aware threshold sweep later, not before the Stage7F main chain.

Stage7F consumes Stage7E embeddings and keeps Stage7E full fallback-preserving output as the primary planner-evaluation dataset. The row semantics remain one row per `scenario × planner-controlled nuPlan ego rollout`. Strict-filter ratio=0.8 is a supplemental clean-subset sensitivity experiment only: it can be useful because the diagnostic may keep fallback-free rows with slot sanity passing, but it drops scenarios and therefore must not replace the primary full official-rollout evaluation.

Main command:

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --output_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --mode full \
  --run_stage6_pairwise \
  --overwrite
```

Strict-filter sensitivity command, only after a real strict-filter embedding exists:

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

If the strict-filter ratio=0.8 output currently exists only as `nuplan_laneaware_strict_filter_summary.json` / report diagnostics, do not fabricate an embedding. First rerun the Stage7E context builder with `--write_strict_filtered_dataset`, then embed that written `strict_filtered_dataset/` with `tools/stage7e_embed_stage6_dataset.py`.

`tools/stage7f_run_report_card.py` is intentionally a thin runner. It validates embedding/metadata row counts, planner axis, complete scenario × planner alignment in full mode, fallback-preserving versus strict-sensitivity mode, and delegates pairwise BDD/report-card work to existing Stage6 tools when feature inputs are available. It does not modify Stage6 metric definitions, Stage5D CORE, or `tools/lane_aware_assignment.py`.

## Stage 7F Pairwise Aggregation Utility

Stage7F now includes a lightweight collector over existing Stage6 pairwise outputs. This utility is intentionally a collector only: it does not change Stage6 BDD/MMD definitions, does not implement new report-card metrics, does not change Stage5D CORE, does not change lane-aware assignment, and does not change Stage7E embedding row semantics.

### Recommended sequence

A. Run the Stage7F full fallback-preserving main report and existing Stage6 pairwise tools:

```bash
python tools/stage7f_run_report_card.py \
  --embedding_dir outputs/stage7e_idm_embeddings_5logs_laneaware \
  --context_dataset_dir outputs/stage7e_nuplan_5neighbor_context_idm_5logs_laneaware_v2 \
  --output_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --mode full \
  --run_stage6_pairwise \
  --overwrite
```

B. Review:

```text
outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_report.md
outputs/stage7f_idm_5logs_full_fallback_preserving/stage7f_pairwise_summary.md
```

C. Treat pairwise differences as exploratory because each pair currently has `n_A=n_B=5`. BDD measures embedding-space distribution drift magnitude only, not direction; category and feature deltas are interpretation layers.

D. Later run strict-filter ratio=0.8 sensitivity and the Stage5 lane-aware parameter sweep. This is especially important for the full fallback-preserving run because the known fallback rate is about 41.9%.

### Collector-only command

If `stage6_pairwise/*/` already exists, regenerate only the aggregate files with:

```bash
python tools/stage7f_collect_pairwise_summary.py \
  --stage7f_dir outputs/stage7f_idm_5logs_full_fallback_preserving \
  --overwrite
```

Expected aggregate outputs:

```text
stage7f_pairwise_summary.csv
stage7f_pairwise_summary.json
stage7f_pairwise_summary.md
```
## Stage7P / PDM Closed Planner Progress Update

### Motivation

The original Stage7 goal is to validate whether the proposed style-monitoring pipeline can detect realized behavioral differences under same-scenario closed-loop simulation. Earlier Stage7 work focused on rule-based / synthetic-style comparisons. The current extension evaluates whether a real nuPlan-compatible planner, PDM Closed Planner, can be used as a controllable policy source for empirical style validation.

The key question is not whether two YAML profiles are nominally different, but whether those nominal planner differences produce measurable rollout differences after official nuPlan closed-loop simulation, Stage5D-compatible context construction, Stage6 embedding, and Stage7F paired-delta / BDD evaluation.

### Current PDM v1 planner profiles

The current implementation defines two PDM Closed Planner profiles in `tools/stage7c1_run_nuplan_simulation.py`:

#### `pdm_closed_conservative_v1`

```python
{
    "idm_policies.speed_limit_fraction": [0.2, 0.4, 0.6, 0.8],
    "idm_policies.fallback_target_velocity": 10.0,
    "idm_policies.min_gap_to_lead_agent": 2.0,
    "idm_policies.headway_time": 2.0,
    "idm_policies.accel_max": 1.0,
    "idm_policies.decel_max": 3.0,
    "lateral_offsets": [-0.5, 0.5],
}
```

#### `pdm_closed_assertive_v1`

```python
{
    "idm_policies.speed_limit_fraction": [0.4, 0.6, 0.8, 1.0],
    "idm_policies.fallback_target_velocity": 18.0,
    "idm_policies.min_gap_to_lead_agent": 0.5,
    "idm_policies.headway_time": 1.0,
    "idm_policies.accel_max": 2.0,
    "idm_policies.decel_max": 3.5,
    "lateral_offsets": [-1.5, 1.5],
}
```

Both profiles are treated as external Hydra PDM planners. The important implementation fix was to make all PDM closed profiles use `planner_type = external_hydra_planner`; otherwise Stage7C tries to discover `PDMClosedPlanner` as a built-in nuPlan planner and fails.

At this stage, these parameters are still defined in Python code rather than injected from command-line JSON. This is acceptable for the current v1 experiment, but future experiments should support a `--planner_profile_json` option so that conservative/assertive parameters can be modified directly from the experiment shell script.

### What PDM v1 can and cannot prove

PDM v1 is useful for testing realized longitudinal and lateral-offset style differences. The current v1 parameters mainly control:

* speed target sampling through `speed_limit_fraction`
* fallback target velocity
* minimum lead-agent gap
* IDM headway time
* maximum acceleration and deceleration
* lateral proposal offsets

Therefore, the strongest expected effects are speed, headway, gap, and general interaction behavior. PDM v1 should not be over-claimed as a true lane-change intention planner. A wider `lateral_offsets` range can produce more assertive lateral proposals, but it is not equivalent to an explicit target-lane gap-acceptance or adjacent-lane decision module.

For thesis writing, the correct claim is:

> PDM closed planner variants induce realized closed-loop behavioral differences under identical nuPlan scenarios.

The claim should not be:

> PDM aggressive has stronger lane-change intention.

That stronger claim requires a future explicit adjacent-lane proposal / gap-acceptance extension.

### nuPlan mini scenario tags: what the counts mean

The scenario labels used for candidate selection come from nuPlan's native `scenario_tag` table. They are not manually assigned by this project.

However, `scenario_tag` counts are tag-row counts, not full scenario counts. One `.db` file is a log database, not one scenario. A `scenario_tag` row marks a tagged `lidar_pc_token` anchor. The actual nuPlan simulation scenario is constructed later by the nuPlan scenario builder around that token.

The current mini inventory shows:

```text
db_files = 64
total_lidar_pc_rows = 518999
total_scenario_tag_rows = 892204
distinct_tagged_scenario_tokens = 390186
strict_changing_lane_tag_rows = 44
strict_changing_lane_unique_tokens = 22
```

Strict lane-change DB tags are:

```text
changing_lane = 22 tag rows
changing_lane_to_left = 15 tag rows
changing_lane_to_right = 7 tag rows
```

After de-duplicating by scenario token, strict lane-change candidates contain 22 unique tokens.

A previous bug wrote SQLite BLOB tokens into CSV as Python bytes strings such as `b'\xf6\xf9...'`. That made Stage7C attempt zero official simulations. The corrected implementation must convert SQLite BLOB tokens to hex strings, for example:

```text
f6f9afda75e251ae
```

not

```text
b'\xf6\xf9\xaf\xdau\xe2Q\xae'
```

### Strict lane-change probe result

After fixing BLOB-token conversion and probing strict changing-lane DB-tag candidates through Stage7C, the current known-good actual lane-change candidates are:

| log_name                                 | scenario_token     | actual_scenario_type     |
| ---------------------------------------- | ------------------ | ------------------------ |
| `2021.05.25.14.16.10_veh-35_01690_02183` | `f6f9afda75e251ae` | `changing_lane_to_right` |
| `2021.06.07.18.53.26_veh-26_00005_00427` | `a59a8c3490f154e2` | `changing_lane_to_left`  |
| `2021.06.08.12.54.54_veh-26_04262_04732` | `9945129405795b72` | `changing_lane_to_right` |
| `2021.06.09.17.23.18_veh-38_00773_01140` | `e3b38485532e575e` | `changing_lane_to_left`  |
| `2021.06.23.15.56.12_veh-16_00839_01285` | `6d1811320c635e82` | `changing_lane_to_left`  |
| `2021.08.17.16.57.11_veh-08_01200_01636` | `9e30155b8bb55fd9` | `changing_lane_to_left`  |
| `2021.08.30.14.54.34_veh-40_00439_00835` | `05f9092e219d59f2` | `changing_lane_to_right` |
| `2021.10.05.07.10.04_veh-52_01442_01802` | `713fa73e30435d98` | `changing_lane_to_right` |

The actual-type probe also found non-lane-change realizations among DB-tag strict candidates, including `traversing_pickup_dropoff`, `stationary`, and `medium_magnitude_speed`. This confirms that DB tags are useful candidate signals, but actual simulation output must still be verified.

### Current smoke-test milestone

A 2-scenario × 2-planner smoke test has passed end to end:

```text
2 known-good lane-change scenarios
× 2 PDM planners
= 4 official nuPlan closed-loop rollouts
```

Stage7C status:

* official nuPlan closed-loop simulation succeeded
* no pseudo rollout was generated
* strict same-scenario token alignment passed
* actual scenario types were `changing_lane_to_left` and `changing_lane_to_right`

Stage7E status:

```text
context_traj.npy = [4, 149, 83]
83 = ego 8 + 5 semantic neighbor slots × 15 channels
row semantics = one row per scenario × planner-controlled nuPlan ego rollout
```

Stage7E embedding status:

```text
embedding.npy = [4, 64]
context_dim = 83
checkpoint_context_dim = 83
multi_agent_ego_expansion = false
```

Stage7F paired-delta status:

```text
planner A = pdm_closed_assertive_v1
planner B = pdm_closed_conservative_v1
delta convention = A - B
paired_scenarios = 2
```

Observed pilot results:

```text
assertive > conservative mean_speed: 2 / 2
mean delta_mean_speed: +2.416 m/s
mean delta_max_speed: +1.666 m/s
mean embedding_l2_distance: 7.576
mean embedding_cosine_distance: 0.097
```

The current pilot supports the statement:

> PDM v1 assertive realizes higher speed and non-zero embedding displacement relative to PDM v1 conservative under same-scenario official nuPlan simulation.

It does not support the stronger statement that assertive always has higher acceleration or jerk. In the 2-scenario pilot, assertive showed lower RMS acceleration and jerk. This likely means that the v1 profiles primarily change speed/gap behavior rather than uniformly increasing dynamic aggressiveness.

### Lane-change should not be the only PDM experiment axis

Strict lane-change candidates are valuable for case studies, but they are too few for the main BDD experiment. Even if all 22 strict DB-tag candidates were usable, that is still a small sample for distributional embedding analysis.

The main PDM v1 experiment should therefore use a balanced scenario set that includes lane-change but is not dominated by it. This is more consistent with the PDM v1 parameter changes, which mainly affect speed, headway, gap, acceleration/deceleration, and interaction response.

Recommended 20-scenario mini-pilot design:

| bucket                          | count | purpose                                                          |
| ------------------------------- | ----: | ---------------------------------------------------------------- |
| actual verified lane-change     |     8 | keep lateral / lane-change coverage                              |
| following / lead interaction    |     4 | evaluate headway, gap, front distance                            |
| stop-go / signal / congestion   |     4 | evaluate speed, braking, acceleration, waiting / launch behavior |
| interaction-rich / lateral-rich |     4 | evaluate surrounding-vehicle density and lateral dynamics        |

Concrete bucket proposal:

```text
actual_verified_lane_change: 8
following_lane_with_slow_lead: 2
following_lane_with_lead: 1
stopping_with_lead or stopping_at_traffic_light_with_lead: 1
stationary_in_traffic: 2
accelerating_at_traffic_light*: 1
stopping_at_traffic_light*: 1
near_multiple_vehicles: 2
high_lateral_acceleration: 1
near_high_speed_vehicle: 1
```

The goal of the 20-scenario run is not to prove a final thesis result, but to decide whether PDM v1 is worth scaling. A useful success criterion is:

```text
paired_scenarios >= 15
assertive > conservative mean_speed in most paired scenarios
embedding distance is consistently non-zero
BDD bootstrap is directionally stable
```

### Recommended experiment scale

For Stage7F / BDD interpretation:

| paired scenario count | interpretation                                                          |
| --------------------: | ----------------------------------------------------------------------- |
|                     2 | smoke test only; verify pipeline                                        |
|                     5 | mini-pilot; inspect direction only                                      |
|                 10–20 | exploratory evidence; decide whether to continue                        |
|                 30–50 | minimum useful range for thesis-level BDD analysis                      |
|                  100+ | stronger conference-level evidence, especially for task-conditioned BDD |

For task-conditioned BDD, each task slice should ideally contain at least 10 paired scenarios, preferably 20–30. Otherwise, slice-level BDD is likely dominated by individual scenarios.

### Current decision

Stage7 should proceed with a balanced 20-scenario PDM v1 mini-pilot:

```text
20 scenarios × 2 planners = 40 official nuPlan rollouts
```

At the current runtime estimate of about 75 seconds per planner-scenario rollout, this is expected to fit roughly within one hour of simulation time, plus context/embedding/report overhead.

If the balanced 20-scenario pilot shows stable speed and embedding differences, expand to 30–50 paired scenarios. If the difference remains mostly speed-only and weak in BDD, then either:

1. keep PDM v1 as an auxiliary validation experiment, or
2. move to PDM v2 with deeper scorer / comfort / tracker parameter exposure, or
3. implement explicit adjacent-lane proposal and target-lane gap acceptance for true lane-change-intention experiments.

## Stage 7 Milestone 1 PDM Balanced20 Data Credibility Audit（2026-07-25）

审计确认 20 个候选中 17 个场景、34 个 planner run 成功；selected index 12、13、19 在 scenario extraction 阶段失败。Stage7C tensor 正确使用非连续轴 `0..11,14..18`，但 Stage7E 未读取该轴并按连续 `range(17)` 重建 metadata；同时 `find_msgpack()` 存在全局首文件 fallback。后 5 个 tensor 场景、共 10 行因此发生 ego、neighbor、metadata 错配。

当前 `balanced17` Stage7F 结果不能作为论文统计证据。Full geometric fallback rate 为 `0.865104`；strict-0.8 仅保留 4 行/2 对；raw lateral acceleration/curvature 亦有物理告警。完整产物见 `outputs/stage7_m1_pdm_data_quality_audit_v1/`。

进入 Milestone 2 前必须修复 scenario-axis 映射和 msgpack identity validation，增加非连续轴回归测试，并用已有 17 个成功 official rollouts 重建 Stage7E/7F。

## Stage 7 Milestone 1 修复后重审（2026-07-25）

已完成 scenario-axis / msgpack identity 修复，并使用原有 17 个成功 official rollout 重建到 `v2_aligned` 目录。Stage7E 现在强制读取 `simulated_ego_seq_index.json`，明确区分 tensor position 与原始 `scenario_index`；每个 tensor 单元都必须匹配 `status=succeeded`、planner id/name、scenario token，并且 msgpack 必须唯一位于对应的 `scenario_<index>/<planner>/` 目录且文件名 token 一致。全局 msgpack fallback 已删除。

重审结果：

```text
scenario_axis: [0..11,14..18]
alignment rows passed: 34 / 34
context shape: [34,150,83]
embedding shape: [34,64]
Stage7F full paired alignment: 17 / 17 scenarios
Milestone 1 verdict: PASS_WITH_LIMITATIONS
```

主要重建结果：

- full BDD MMD² = `0.0357706`，permutation p = `0.752475`，仍为小样本 exploratory、无显著分布差异；
- assertive > conservative mean speed = `17/17`；
- assertive RMS acceleration 更高 = `14/17`；
- assertive mean THW 更小 = `15/17`；
- task-conditioned all-strength 有 5 个有效 task，strong-only 有 4 个有效 task，均无显著 p-value；
- following 与 queue positive rows / paired scenarios 完全重合（Jaccard=`1.0`），不能作为两份独立证据；
- `task_yield_conflict` 在修复后对 34 行全部为正，属于 degenerate task，因此不再进入有效 task BDD。

对齐修复不改变地图投影限制：geometric fallback 仍为 `0.865104`，strict-0.8 仍只有 4 行/2 个完整场景。因此 full 数据可以作为结构正确的 exploratory planner comparison，不能作为 strong lane-aware 证据；strict subset 不能用于 BDD。

物理告警已定位到具体帧：9 个超阈值点中，8 个 lateral-acceleration 点发生在 `t=149` 的无效 padding 帧；唯一有效帧异常为 scenario index `1`、token `a59a8c3490f154e2`、conservative planner、`t=49` 的 raw absolute curvature `4.1180`。这说明 Stage6C 当前 raw physical diagnostics 会把 padding 边界差分计入统计，下一步应先传播/消费 rollout validity mask，再重新生成 behavior-event 与物理诊断。

完整重审产物见 `outputs/stage7_m1_pdm_data_quality_audit_v2_aligned/`；修复前的 `v1` 产物仅保留用于问题溯源，不再用于论文统计。

## Stage 7 Milestone 1 validity-mask follow-up（2026-07-26）

已完成上一节要求的 rollout validity mask 修复：

- Stage7E context 正式写出 `ego_seq_mask.npy [34,150]`，与 Stage7C `simulated_ego_seq_mask.npy` 逐元素一致；
- 34 行有效长度为 149 或 150，共排除 22 个 padding 帧；
- `interaction_feat_style.npy` 只聚合 mask=true 的帧；
- Stage6C 在 smoothing、导数、事件检测与 raw physical diagnostics 之前同步裁剪 ego/neighbor 序列；
- Stage7F paired kinematic metrics 同样消费 mask；
- 曲率 `yaw_rate/speed` 只在 `|speed| >= 0.5 m/s` 时定义，近静止帧记为 NaN。

修复后的 behavior-event warnings：

```text
rollout_validity_mask_applied: true
padding_frames_excluded: 22
metric_physical_range_warning: 0
raw_metric_physically_implausible: false
```

此前 9 个超阈值点中，8 个确认为无效 `t=149` padding 边界，已从所有下游指标中排除；剩余的 4.118 curvature 来自 `speed=0.009 m/s` 的近零速除法，加入 0.5 m/s 有效速度门槛后不再被视为物理曲率样本。修复后有效 rollout 帧上的 physical anomaly 数为 0。

主 embedding 没有改变，因此 full BDD 仍为 `MMD²=0.0357706, p=0.752475`。Pairwise/category 解释层已使用 mask-aware interaction features 重新生成；paired speed/accel/THW 方向和 task-conditioned BDD 结果保持不变。

Milestone 1 最终状态仍为 `PASS_WITH_LIMITATIONS`，但 physical-metric limitation 已关闭。剩余限制只有：

1. geometric fallback rate=`0.865104`；
2. strict-0.8 只有 4/34 行、2 个完整场景。

因此下一阶段不应先扩大样本量，而应优先诊断 nuPlan lane projection / adjacency 覆盖，降低 fallback 后再决定是否扩展到 30–50 paired scenarios。

## Stage 7 Milestone 2A lane projection / adjacency repair（2026-07-26）

Milestone 2A 已完成。原 `v2_aligned` 数据的 `fallback_assignment_used_rate=0.865104` 不是投影阈值本身过严，主要由两项地图适配错误造成：

1. lane extraction 是以 ego rollout 为中心的局部查询，但 cache 只使用 `map_name` 作为 key，导致同一地图上的后续远距离场景复用第一个场景的局部 lane 集合；
2. Stage7C index 没有 location 时，旧命令把 Las Vegas CLI map 当作所有场景的地图。实际 balanced17 同时包含 Las Vegas、Pittsburgh、Boston 和 Singapore。nuPlan DB 中还存在 `location=las_vegas`、`map_version=us-nv-las-vegas-strip` 的别名差异。

修复后的适配规则：

- lane cache key 为 `(canonical map_name, original scenario_index)`；
- 地图查询坐标合并同一场景的全部 planner valid frames；
- map API 按 canonical map name 复用，但局部 LaneInfo 不跨场景复用；
- 地图名优先级为 row metadata、scenario metadata、nuPlan DB `log.map_version`、CLI fallback；
- dense rollout 坐标按空间覆盖选取查询 anchor，减少重复 map API 查询；
- 全部 34 行输出 frame-weighted fallback 统计，候选级 projection debug 记录具体失败原因；
- strict-filter 主阈值和 ratio sweep 均使用非连续原始 scenario ID。

最终 `v3_m2a` 结果：

```text
source scenarios: 17
planner rows: 34
canonical maps: 4
lane cache entries: 17
map API cache entries: 4
fallback rate: 0.0161481
ego projection success rate: 0.983852
candidate projection success rate: 0.649271
lane_map_unavailable fallback frames: 0
remaining fallback frames: 82 (heading_difference_exceeded)
strict-0.8 rows: 20 / 34
strict-0.8 complete paired scenarios: 7
Milestone 2A verdict: PASS
```

相对 `v2_aligned`，fallback 绝对下降 `0.848956`，相对减少 `98.13%`。因此 high geometric fallback 和 strict-subset-too-small 两项 Milestone 1 limitation 已关闭。

下游使用相同 17 个 official rollout 重建后：

```text
context: [34,150,83]
embedding: [34,64]
full BDD MMD²: 0.0292350
permutation p: 0.891089
paired scenarios: 17
assertive > conservative mean speed: 17/17
assertive > conservative RMS accel: 14/17
assertive smaller mean THW: 13/17
valid-frame physical anomalies: 0
```

lane-aware 修复使 context/embedding 和 behavior-event task slices 发生实质变化，但总体结论不变：assertive 与 conservative 在轨迹运动学上存在一致差异，当前 17 对样本的 embedding distribution BDD 仍小且不显著。Milestone 2A 解决的是数据可信度与地图语义覆盖，不应被描述为提高统计显著性的调参步骤。

## Stage 7 Milestone 2B lane-context quality stratification（2026-07-26）

Milestone 2B 已完成，最终审计为 `PASS`，扩容判定为 `READY_TO_SCALE`。

旧 strict filter 会因任意一个 ambiguous frame 否决整条 rollout，造成 34 行只保留
20 行，而且 planner 留存不对称（conservative 12、assertive 8）。这种逐 planner
的 realized-rollout 过滤会破坏 same-scenario pairing，并可能引入
post-treatment selection bias。

M2B 改为先计算每行有效帧上的精确质量比例，再在 scenario pair 层取两个 planner
中的较差 tier：

```text
Tier A: fallback <= 0.05, ambiguity <= 0.05, bad frame = 0
Tier B: fallback <= 0.20, ambiguity <= 0.20, bad frame = 0
Tier C: otherwise
```

最终 row tiers 为 `A=31, B=2, C=1`，pair tiers 为 `A=15, B=1, C=1`。
只有 scenario index 5 为 Tier C（conservative fallback=`0.2953`），scenario
index 6 为 Tier B（conservative ambiguity=`0.0733`）；其余 15 对均为 Tier A。
全部 tier subset 都保持 assertive/conservative 成对选择。

全部 17 个完整 planner pair 继续作为主分析。Tier A 和 Tier A+B 仅作为质量
敏感性分析，不能替代主估计，原因是 lane-context quality 来自 planner 已经实现的
rollout。逐 planner 或只报告高质量子集会改变 estimand，并可能偏向某个 planner。

| 数据集 | pairs | MMD² | permutation p |
| --- | ---: | ---: | ---: |
| full | 17 | 0.0292350 | 0.891089 |
| Tier A | 15 | 0.0365372 | 0.722772 |
| Tier A+B | 16 | 0.0326079 | 0.702970 |

三个集合均支持相同定性结论：assertive 与 conservative 的 embedding
distribution 差异较小，在当前样本量下不显著。质量分层没有揭示被低质量
lane assignment 掩盖的显著 BDD。

M2B context 重建增加了逐行 ambiguity/bad/quality-eligible 统计和地图来源字段。
`context_traj.npy`、`ego_seq.npy`、`ego_seq_mask.npy`、`neighbor_seq.npy`、
`interaction_feat_style.npy`、`neighbor_slot_ids.npy` 与 M2A 版本逐字节一致，
因此复用 M2A embedding 是严格的数据等价复用，不是近似复用。

projection debug 中的 candidate-level `lane_relation_unknown` 主要由
`lane_connector_unhandled`、`missing_adjacency` 和候选投影失败构成。它统计的是
采样候选 lane，其中包括不属于正确 ego topology 的候选，不能解释为 accepted
ego/neighbor assignment 的失败率，也不是当前扩容 blocker。

`READY_TO_SCALE` 只表示结构、地图、对齐、有效帧、pair symmetry 和质量敏感性
检查均通过。17 对仍然是 exploratory evidence；下一步应扩大到至少 30–50 个
完整 paired scenarios，并在扩容数据上原样执行 full-primary、Tier-sensitivity
政策。若目标是 task-conditioned BDD，每个 task slice 应优先达到至少 10 对，
理想为 20–30 对。

## Stage 7 Milestone 3 Balanced50 scale-up（2026-07-26，已完成）

Milestone 3 已正式启动。目标不是通过扩容“追求显著性”，而是在 M1–M2B
已经通过的数据可信度协议上，将 17-pair exploratory evidence 扩展到论文最低
有用范围。

冻结设计为：

| bucket | target |
| --- | ---: |
| actual verified lane-change | 8 |
| following interaction | 10 |
| stop-go / signal | 10 |
| dense interaction | 8 |
| lateral / turning | 7 |
| speed context | 7 |
| total | 50 |

选择集包含 M2B 已成功的 17 个场景和 33 个新候选，共覆盖 37 个 log，每个 log
最多 2 个场景。历史 balanced20 中的 3 个 technical extraction failure token 已
排除。另冻结 20 个 reserve，但 replacement 只能针对记录明确的技术失败并严格
遵循 reserve rank，不能查看 planner behavior 或 BDD 后再换样本。

冻结选择 manifest：

```text
SHA-256 = a59b003ee517237d5a888e9774f939879ce812ac99d09a8f41e23c6d7e196313
target = 50 scenarios × 2 planners = 100 official rollouts
```

统计协议保持不变：

1. 所有成功的 complete planner pairs 构成 primary full dataset；
2. 不允许只保留一个 planner row；
3. Tier A 和 Tier A+B 是 symmetric pair-level sensitivity analysis；
4. 不以 embedding distance、BDD 大小或 p-value 决定场景保留与替换；
5. M3 的通过条件是数据结构、配对、质量和最小规模通过，不要求统计显著。

official simulation 的完整进度记录保存在
`outputs/stage7_m3_pdm_balanced50_stage7c_v1/stage7c_progress.json`。执行期间修正了
启动脚本缺少 `NUPLAN_MAPS_ROOT` 的环境问题；修正前的无效任务未进入正式结果。
修正后已依次完成 Stage7E context、embedding、Stage7F
full/task-conditioned analysis、M2B quality stratification 和 M3 final audit。

最终 100 个 official tasks 全部执行结束。5 个场景在 nuPlan scenario extraction
阶段对两个 planner 对称失败，错误均为 `No scenarios found to simulate`；其余
45 个场景形成 90 个严格 token-aligned official rollouts。因此没有使用 reserve
追补，也没有根据 planner outcome 选择替换样本。

最终数据质量：

```text
complete pairs: 45
successful rollouts: 90
context shape: [90,150,83]
embedding shape: [90,64]
fallback assignment rate: 0.00737156
ego lane projection success rate: 0.992628
Tier A pairs: 40
Tier A+B pairs: 44
valid-frame physical anomalies: 0
```

轨迹层结果：

```text
assertive higher mean speed: 44 / 45
assertive higher RMS acceleration: 40 / 45
assertive smaller mean THW: 24 / 45
mean delta mean-speed: +1.4277 m/s
mean embedding L2 distance: 5.6227
```

BDD 质量敏感性：

| dataset | pairs | MMD² | permutation p |
| --- | ---: | ---: | ---: |
| full | 45 | 0.0142209 | 0.742574 |
| Tier A | 40 | 0.0163792 | 0.683168 |
| Tier A+B | 44 | 0.0164485 | 0.673267 |

三层结果均为小且不显著，说明 M2B 的 lane-context 质量筛选不会改变 M3
embedding-distribution 结论。相比 17-pair pilot，MMD² 进一步下降；这不能解释为
“assertive 和 conservative 没有行为差异”，因为 same-scenario 轨迹指标仍显示
高度一致的速度与加速度方向。更准确的结论是：当前 encoder 的总体 embedding
distribution BDD 对这些 PDM 参数变化较弱，而 trajectory-level paired metrics
能够稳定识别 realized behavior difference。

task-conditioned 结果生成了 6 个有效 task，但解释仍受以下限制：

1. `task_lead_brake_response` 只有 7 个 complete positive pairs，低于预设的 10；
2. following 与 queue 的 paired-scenario Jaccard=`0.833`，不能当作两份独立证据；
3. lead-brake 和 cut-in detector 仍为 proxy-dominant；
4. 所有 task-conditioned p-value 均不显著。

Milestone 3 最终状态：

```text
overall verdict: PASS_WITH_LIMITATIONS
thesis scale status: MINIMUM_USEFUL_SCALE_REACHED
```

这表示 Stage7 已达到预设的论文最低有用 paired scale，并完成结构、地图、有效帧、
质量分层和敏感性审计；不表示已经获得显著的 embedding-distribution difference。

## Stage 7 Milestone 4 formal statistical evidence（2026-07-26）

M4 冻结 M3 的45个 complete pairs，不再改变场景、planner 参数或质量阈值。
需要明确：M4 主端点是在已经观察 M3 exploratory result 后冻结，因此属于
retrospective formalization，不是独立 preregistered confirmation。

主统计 family 为：

1. assertive minus conservative mean speed，方向 `> 0`；
2. assertive minus conservative RMS acceleration，方向 `> 0`；
3. assertive minus conservative mean THW，方向 `< 0`。

每项使用10000次 paired bootstrap mean CI、单侧 Wilcoxon signed-rank、exact
sign test、paired Cohen's dz、rank-biserial 和 Hodges–Lehmann delta。Wilcoxon
与 sign test 分别在三项 family 内执行 Holm correction。

结果：

| endpoint | n | mean delta | 95% paired bootstrap CI | paired dz | Wilcoxon Holm p | sign Holm p |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| mean speed | 45 | +1.4277 m/s | [1.0106, 1.8723] | 0.948 | 1.71e-13 | 3.92e-12 |
| RMS acceleration | 45 | +0.2562 m/s² | [0.1701, 0.3416] | 0.862 | 3.00e-7 | 7.88e-8 |
| mean THW | 35 | -7.9987 s | [-31.1524, 12.3490] | 0.119 | 0.0177 | 0.00677 |

速度与RMS加速度同时满足均值CI不跨零、两类非参数检验经Holm校正后显著，并具有
较大的 paired standardized effect。它们构成目前最稳健的 realized trajectory
evidence。

THW 的 median=`-2.3005 s`、Hodges–Lehmann delta=`-3.2037 s`，方向计数为
24 smaller、9 larger、2 zero；非参数检验通过。但10个场景没有 finite
front-agent THW contrast，mean bootstrap CI 跨零，且均值受极端长THW影响。
因此论文中不能声称“assertive 稳定降低平均THW”；可表述为：

> 在具有有效前车THW对比的35个场景中，THW的稳健位置统计显示向更小值偏移，
> 但平均差异不稳定，且该结论受available-case选择和极端值影响。

BDD 使用1000次 bootstrap 与1000次 permutation 重算：

| dataset | pairs | MMD² | permutation p |
| --- | ---: | ---: | ---: |
| full | 45 | 0.0142209 | 0.733267 |
| Tier A | 40 | 0.0163792 | 0.697303 |
| Tier A+B | 44 | 0.0164485 | 0.593407 |

高分辨率复算保持 M3 结论：embedding distribution BDD 小且不显著，并且不依赖
lane-context quality tier。BDD bootstrap interval 表示重采样变异，不是以0为
null的置信区间；统计显著性应读取 permutation p-value。

六个 task-conditioned BDD 在独立的 exploratory family 内进行 Holm correction，
调整后 p-value 均为1.0。结合 task overlap 与 proxy detector 限制，没有
task-level embedding distribution significance。

Milestone 4 状态：

```text
verdict: PASS_WITH_LIMITATIONS
analysis status: RETROSPECTIVE_FORMALIZATION_OF_M3_EXPLORATORY_RESULTS
```

当前最强结论是：PDM assertive 参数在同场景 official rollout 中稳定提高速度和
RMS加速度，但现有 behavior encoder 的总体及task-conditioned embedding BDD
未检测到显著分布漂移。轨迹层敏感而embedding distribution层较弱，是研究结果，
不应通过继续增加 permutation 次数或选择子集来追求显著性。

## Stage 7 Milestone 5 representation mechanism（2026-07-26）

M5 用三类相同90行、45个scenario pairs的表示解释为何轨迹配对统计显著，而
embedding marginal BDD 不显著：

1. learned behavior embedding，64维；
2. interaction features，33维；
3. trajectory summary，12维。

三个诊断回答不同问题：

- paired sign-flip：同一个scenario内部的 A-B 表示差是否具有一致平均方向；
- scenario-grouped probe：表示能否推广到未出现在训练fold的新scenario并识别planner；
- marginal MMD：忽略scenario pairing后，两组总体边际分布是否不同。

grouped probe 使用5-fold GroupKFold，scenario不会跨train/test。缺失值填补、
standard scaling和logistic classifier全部在training fold内拟合。Permutation
null 在每个scenario pair内部交换planner label，保留一对一设计。

| representation | paired concentration | sign-flip p | grouped ROC-AUC | pair-swap p | marginal MMD p |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned embedding | 0.326 | 0.000100 | 0.638 | 0.00699 | 0.733 |
| interaction features | 0.429 | 0.000100 | 0.773 | 0.000999 | 0.126 |
| trajectory summary | 0.464 | 0.000400 | 0.704 | 0.000999 | 0.123 |

learned embedding 的 paired shift 和scenario-disjoint probe均显著，说明它并非
完全没有planner behavior信息。但interaction features和trajectory summary具有
更强的probe AUC，说明encoder压缩后丢失了部分易分辨信号。

更关键的是，三种表示的marginal MMD均未显著。即使直接使用interaction/trajectory
表示，忽略pairing后的总体分布检验仍受scenario heterogeneity主导。因此M3/M4的
BDD结果不应只解释为“encoder失败”，而应解释为：

> planner参数产生了稳定的within-scenario behavior shift；该shift相对于跨场景
> variation较小。paired estimand和scenario-grouped prediction能够利用设计结构，
> marginal distribution BDD则回答更强、也不同的总体分布问题。

Embedding pair distance 与 absolute mean-speed delta 的 Spearman rho=`0.454`，
Holm p=`0.00518`；与RMS acceleration和THW delta的距离相关性未通过Holm校正。
这表明embedding距离对速度变化具有可解释敏感性，但对comfort/headway差异仍弱。

不依赖observed effect的paired-t设计敏感度：

| paired n | MDE dz, alpha=.05 | MDE dz, alpha=.05/3 |
| ---: | ---: | ---: |
| 30 | 0.465 | 0.564 |
| 45 | 0.376 | 0.454 |
| 60 | 0.325 | 0.391 |
| 90 | 0.264 | 0.317 |
| 120 | 0.228 | 0.274 |

MDE不是post-hoc achieved power，也不能直接外推MMD所需样本量。

Milestone 5 状态：

```text
verdict: PASS_WITH_LIMITATIONS
analysis status: EXPLANATORY_POST_M4_MECHANISM_ANALYSIS
```

M5是在观察M3/M4结果后设计的解释性分析。linear probe不是新的planner performance
metric，interaction/trajectory baselines也来自相同realized rollouts，因此这些
结果用于解释measurement mechanism，而不是独立确认planner优越性。

## Stage 7 Milestone 6 scenario-conditioned BDD 与跨域重训练协议（2026-07-26）

M6 将 M5 的 mechanism finding 落到设计匹配的 BDD。对相同90行、45个 complete
same-scenario pairs，M6 固定 pooled median RBF bandwidth，并比较 pooled shuffle
与 within-pair label swap：

| analysis | space | MMD² | p-value |
| --- | --- | ---: | ---: |
| frozen M4 marginal BDD | original | 0.0142209 | 0.733267 |
| fixed-kernel marginal recheck | original | 0.0141802 | 0.737126 |
| paired-label-swap BDD | original | 0.0141802 | 0.002300 |
| scenario-residualized paired BDD | pair-midpoint residual | 0.0994187 | 0.000100 |

同一个 original-space MMD² 仅改变 permutation exchangeability assumption 后从
不显著变为显著，说明当前 embedding 已包含可检测的 within-scenario planner
shift。Residual BDD 进一步去除每个 scenario pair 的 midpoint，但它与 original
space 的 MMD² 不能直接按数值比较。

M6 不覆盖 marginal BDD：matched simulation 以 paired BDD 为主，marginal BDD 为
补充；真实 unpaired logs 继续使用 Stage6 unpaired-first 与 scenario control。

训练侧诊断是 Stage5D 33维 supervision 缺少 `mean_speed`，且现有 checkpoint
从未使用 nuPlan pairs 做 domain adaptation。后续新 checkpoint family 将增加
versioned kinematic targets、same-scenario pair ranking 与 Waymo teacher
consistency。现有45对已经被反复分析，只作为 exploratory set；最终确认必须使用
新的 scenario-disjoint locked test。

当前本机的 Waymo full51 manifest 指向不存在的 `/Users/liuqing/...` shards，
所以联合重训练在数据恢复前不宣称完成。详细协议见
`docs/stage7_cross_domain_style_sensitive_training_protocol.md`。

Milestone 6A 状态：

```text
verdict: PASS_WITH_LIMITATIONS
analysis status: POST_M5_DESIGN_AWARE_ESTIMAND_CORRECTION
```

## Stage 7 Milestone 6.1 方法冻结（2026-07-29）

M6.1 将原 M6 从解释性修正推进到可复现的方法冻结，但当前45对仍是
`METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY`。冻结 primary 为原始64维 embedding
上的 single-RBF biased V-statistic MMD²、精确 pooled positive off-diagonal
median bandwidth、within-pair label swap、100000 permutations 和 plus-one
p-value。

实际 primary MMD²=`0.0141802`，exceedance=`175/100000`，
p=`0.001760`；pooled-shuffle control exceedance=`74086/100000`，
p=`0.740863`。Residual secondary exceedance=`0/100000`，按 Monte Carlo
分辨率报告 `p<=0.000010`。45/45 pairs 完整，duplicate token、missing planner、
row conflict、unequal within-pair horizon 和 non-finite embedding 均为0。
Tier A 40对和 Tier A+B 44对的 primary 结果经 Holm correction 后仍显著；已测
fallback/ambiguous-rate 与 embedding pair distance 的相关性经 Holm correction
均不显著。

这完成的是方法校准，不是新数据上的确认。下一步先做：

1. 新 log/scenario-disjoint pairs 的盲化、独立 selection-config 冻结与确认；
2. 预处理定义的 task-conditioned paired BDD 和 representation controls；
3. 只有锁定证据显示关键任务的表示敏感性不足时，才恢复 Waymo shards 并启动
   新 checkpoint family 的条件式重训练消融。

Waymo-only Stage5D-balanced-v2 继续作为主跨域模型，nuPlan 继续作为外部验证域。

## Stage 7 Milestone 6.2 锁定确认入口（2026-07-29）

M6.2 已实现锁定 manifest、planner 参数指纹、pre-treatment task mapping、小样本
exact paired randomization 和 representation controls。新确认数据的 log 和
scenario token 必须与45对开发集不相交，但被比较的 planner 参数必须与开发阶段
冻结 treatment 完全相同。

| task | complete pairs | learned embedding exact p | Holm p |
| --- | ---: | ---: | ---: |
| following interaction | 9 | 0.167969 | 0.671875 |
| lane change | 8 | 0.789062 | 0.875000 |
| stop-go control | 9 | 0.175781 | 0.671875 |
| high motion dynamics | 9 | 0.003906 | 0.019531 |
| dense/vulnerable interaction | 8 | 0.437500 | 0.875000 |

五个任务都低于12对运行下限，只有 high-motion dynamics 在开发集上通过 Holm
correction。因此 M6.2 状态是 `DEVELOPMENT_VALIDATED_NOT_CONFIRMATORY`。下一步
是完成 simulation-based power justification，并按冻结 task targets 生成新
log/scenario-disjoint rollout pairs。

## Stage 7 Milestone 6.3 仿真功效与采集冻结（2026-07-29）

M6.3 已完成 empirical-pilot simulation power analysis，并将机器可读的 power
justification 接入 M6.2 locked-confirmation fail-closed 检查。主设计冻结
effect scale=`0.75`、power target=`0.80`、500 simulations/cell、999 planning
swaps、最终确认100000 swaps 和20% attrition。

| design | complete pairs/task | gross pairs/task | gross total | simultaneous power |
| --- | ---: | ---: | ---: | ---: |
| primary：75% pilot effect | 60 | 75 | 375 | 0.918 |
| sensitivity：50% pilot effect | 160 | 200 | 1000 | 0.936 |

Primary 的 simultaneous power 95% CI 为 `[0.891,0.939]`；50% sensitivity 为
`[0.911,0.954]`。Overall 的 power-selected n=45，但执行时保留 M6.2 的80对
complete-pair 质量下限。过去每任务12对只是不允许更小样本进入正式流程的运行
下限，现由每任务60对的冻结功效配额取代为主确认目标。

该结果仍属于 `DEVELOPMENT_PILOT_POWER_PLANNING_NOT_CONFIRMATORY`。下一步不是
重新训练，而是按主配额生成新 log/scenario-disjoint、planner-fingerprint-identical
rollouts；采集不得根据中途 effect size 停止。只有锁定确认显示关键任务在充分
功效下仍缺乏 learned-embedding sensitivity，且机制对照支持存在真实行为 shift，
才进入条件式重训练。

## Stage 7 Milestone 6.4 锁定采集预检（2026-08-04）

M6.4 已实现 `tools/stage7_m6_4_freeze_locked_collection.py`，把 M6.2/M6.3 的
disjointness、planner fingerprints、task mapping、75 primary/task 和15
reserve/task 转化为机器可执行的仿真启动门。选择全程只用 pre-treatment DB
scenario tags；跨任务歧义 token 排除，固定 salt 排序，primary+reserve 每 log
最多2个。

当前真实 mini inventory 审计结果：

| task | eligible candidates | primary selected under log cap / 75 |
| --- | ---: | ---: |
| following interaction | 3077 | 14 / 75 |
| lane change | 2 | 2 / 75 |
| stop-go control | 25408 | 15 / 75 |
| high motion dynamics | 37565 | 14 / 75 |
| dense/vulnerable interaction | 4943 | 13 / 75 |

mini inventory 有63个 log，开发集覆盖34个，故只有29个新 log 可用；在每 log
最多2个的冻结多样性约束下，primary 375个场景至少需要188个新 log。Lane-change
即使忽略该总量约束也仅剩2个，是独立的硬缺口。M6.4 当前状态因此为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，没有输出 locked collection
manifest，也没有启动任何新仿真。

下一步为 M6.4A data acquisition：增加足够的 nuPlan log DB，重建同 schema 的
`all_scenario_tags.csv`，然后原样重跑预检。只有 status 变为
`FROZEN_BEFORE_LOCKED_ROLLOUTS` 后，才进入 M6.4B 的750个 primary rollouts
（375 scenarios × 2 planners）；不能通过改变冻结 task family 或复用开发 log
绕过该门槛。

## Stage 7 Milestone 6.4A 可复现多 DB inventory pipeline（2026-08-07）

M6.4A 新增 `tools/stage7p_build_scenario_inventory.py`，将旧 mini
`all_scenario_tags.csv` 的生成过程固化为正式 CLI。实现范围由 GitHub Issue #236
冻结。工具支持重复 `--db_root`，只读扫描每个 root 的直接子目录 DB，并严格关联
`scenario_tag -> lidar_pc -> scene -> log`。BLOB token 统一为 lowercase hex；
M6.4 的 `scenario_token` 继续使用 `scenario_tag.lidar_pc_token`，原 DB scene token
单独保留为 `db_scene_token`。

为控制未来 train DB 的规模，builder 使用临时 SQLite staging 做流式写入、稳定
排序、exact token/type/log/DB 去重和 token-location 冲突审计，不把全量 CSV 放入
内存。多个 scenario types 不会事先合并，M6.4 仍按冻结规则排除跨 task 歧义。
DB basename 冲突、缺表/缺列、断裂关联、空 token、同 token 多 log/DB 均 fail
closed。

每次运行生成：

- `all_scenario_tags.csv`；
- `scenario_inventory_inputs.csv`，含逐 DB 路径、大小、mtime、SHA-256 和 row counts；
- `scenario_inventory_summary.json`，含 schema、counts、工具/Git/runtime provenance
  和输出 hash；
- `scenario_inventory_report.md`；
- 可选 `--flat_db_root` 相对符号链接 pool，供 M6.4 单层 `db_root / db_file` 合同使用。

该工具只建立 pre-treatment inventory，不读取 planner outcome、trajectory、embedding
或 BDD，也不自动启动 M6.4 preflight。mini-only 重建只用于可复现性 smoke；其容量
不足结论不会改变。必须在新增 DB 到位、expanded inventory 构建完成并原样重跑
M6.4 后，才判断能否进入 M6.4B。

Mac mini-only reproducibility smoke 已完成：64个 DB、63个 logs、892204个原始
scenario-tag rows 经 token/type/log/DB 去重后形成821831行 inventory，移除70373个
重复 tag；unique scenario tokens=`390186`，token-location conflicts=`0`，并建立
64个相对符号链接。用新 inventory 原样重跑 M6.4 后，冻结类型 unique tokens
仍为`177313`、eligible candidates仍为`70995`、eligible logs仍为`29`，新旧
`m6_4_task_capacity.csv` 逐字节一致。状态继续为
`BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY`，没有生成 locked manifest。

Pittsburgh DB-only 扩展于2026-08-07完成。ZIP精确为`30620248893` bytes，1562个
entries中包含1560个 DB；路径穿越审计、CRC测试和全部 SQLite header 检查通过，
安全解压大小为51.90 GiB。3个 DB 与 mini 同名且 SHA-256 完全相同；expanded
输入保留 Pittsburgh 副本，并使用61个非重叠 mini DB，得到1621个 DB、1576个
logs。

expanded inventory 包含9695626个原始 tag rows、9604184个去重 rows 和5386575个
unique tokens；移除91442个重复 tag，token-location conflicts为0。M6.4 preflight
首次达到：

```text
status = FROZEN_BEFORE_LOCKED_ROLLOUTS
ready_to_launch_locked_rollouts = true
primary = 75 per task, 375 scenarios, 750 planned rollouts
reserve = 15 per task, 75 scenarios
primary distinct logs = 306
primary + reserve distinct logs = 350
max scenarios per log = 2
development token overlap = 0
development log overlap = 0
missing DB files = 0
Stage7C primary context rows = 375
```

planner fingerprints与M6.2冻结值一致，全部 task deficit为0。Primary manifest
SHA-256为`c825d87826b951bcdd6ed987195aeea25b02290eacca7cc6a2fc2b9e91ba8839`；
reserve manifest SHA-256为
`c6c148d6298a0c6b8cdccd083f363cded1335f41845ba802148967e3f5328904`。
这标志M6.4A通过并进入M6.4B readiness阶段，不表示rollouts已经运行。Mac正式启动
前仍需恢复精确tuPlan Garage commit、完成PDM readiness和单场景official smoke，
再决定本机或Linux x86_64执行方案。

## Stage 7 Milestone 6.4B Mac readiness 与 locked smoke（2026-08-07）

Mac 环境已恢复并核验指定外部代码版本：nuPlan devkit commit 为
`e9241677997dd86bfc0bcd44817ab04fe631405b`，tuPlan Garage commit 为
`b51d5d04fac1bd4389653b9ab2ff73ea88f435a3`。nuPlan 专用 Python 3.9 环境按 devkit
lock 修复 AWS/HTTP 依赖，并补齐 `scikit-learn==1.2.2`、
`positional-encodings==6.0.1`；PDM Closed Planner 与 official
`run_simulation.py` 均可导入。Readiness 输出位于
`outputs/stage7p_pdm_readiness_check_v2_mac/`，状态为 `pdm_available=true`、
`required_next_action=ready_for_pdm_smoke`。

启动前重新核验 manifest，以下冻结值均保持不变：

```text
Stage7C tool SHA-256: 076b35d2112e126008eec5c96bf3e7b159ded75a40be7999212956423cb3e530
primary CSV SHA-256: 91ef586988856b144a7e7fa5f7d7c187750d9bfec3d9951f0f15015f742a0ca5
reserve CSV SHA-256: a73dd1d60c24ad09ec69aa11c0466d4c82ad68b4f5222ab064fca3147ff74cad
assertive fingerprint: 18772fd36a1b0421109a3d93ed494eac5555d2f8f96571be68ff7641a2bac4dc
conservative fingerprint: 9988f615a5aae5c67b3780076110a5150c3303b65e8beb98cc61090d9e19baac
```

只对 locked primary collection 的第一行运行了
`pdm_closed_assertive_v1` / `pdm_closed_conservative_v1` 双 planner official smoke。
目标 log 为 `2021.09.13.18.55.23_veh-45_02099_02822`，nuPlan scenario token 为
`6b5a9da8c0b353b9`。命令同时将 `scenario_builder.db_files` 限定为对应 Pittsburgh
DB，并用 `scenario_filter=all_scenarios` 清除 nuPlan 默认
`one_continuous_log` 的无关 log filter，再注入锁定 token。

结果位于 `outputs/stage7_m6_4b_locked_smoke_1scene_mac_v1/`：

```text
status: PASS
official command successes: 2 / 2
simulated_ego_seq shape: (1, 2, 149, 8)
valid timesteps: 298
msgpack files parsed: 2 / 2
required pose valid ratio: 1.0
missing scenario-planner pairs: 0
same-log alignment: pass
strict nuPlan token alignment: pass
pseudo rollout: false
wall time: about 41 seconds
```

该 token 在 SQLite `scenario_tag` 中同时有冻结 task 标签 `near_long_vehicle` 和
非冻结标签 `stationary`；nuPlan serializer 将 actual scenario type 写为
`stationary`，但 actual log/token 与锁定目标完全一致。M6.4 的 task assignment
仍按 outcome 前冻结的 `near_long_vehicle` metadata 执行，不根据 rollout 后的目录
标签改写。两个 official log 仅有 metric aggregator 未找到 challenge-named 聚合输入
的警告，trajectory、per-scenario metric files 和 runner report 均已生成。后续批量
模板应令 `job_name` 包含 `closed_loop_nonreactive_agents`，并使用绝对 output path
避免相对目录嵌套。

本次只产生2个 smoke rollouts，不构成 locked confirmation data，也没有查看或用于
调整 selection、planner treatment、effect size 或 stopping rule。750个 primary
rollouts 中仍有748个未运行；在批量启动前应先冻结 Mac 批处理、断点续跑、失败分类
和 reserve 消耗的操作方案。

## Stage 7 Milestone 6.4B locked batch orchestration（2026-08-07）

GitHub Issue #237 新增 `tools/stage7_m6_4b_run_locked_rollouts.py`，作为冻结
Stage7C 之上的 orchestration layer。它不修改 `stage7c1_run_nuplan_simulation.py`，
因此 locked manifest 中 Stage7C SHA-256 继续为
`076b35d2112e126008eec5c96bf3e7b159ded75a40be7999212956423cb3e530`。

启动安全门包括：manifest status/ready flag、primary/reserve 文件与 canonical
manifest hash、planner 参数指纹、连续 collection/task rank、跨 primary/reserve
token 唯一性、selection salt、task counts、450个 DB 路径、nuPlan commit 和
tuPlan Garage commit。默认运行模式是 dry-run；真实执行同时要求 `--execute` 与
显式复述 primary manifest SHA-256。批次支持 order range、`--max_scenarios` 和
`--resume`，但始终保持冻结顺序和完整双 planner pair。

每个 primary scenario 写入
`rollouts/order_NNNN_<token>/attempt_NNN/stage7c_output/`。成功 skip 前会重新检查
official successes、trajectory rows、tensor pair completeness、planner axis、same-log
和 strict-token alignment。失败或损坏 attempt 默认阻塞；只有显式
`--retry_failed` 才在新 attempt 目录重试，旧输出不覆盖。`batch_events.jsonl` 为
append-only attempt ledger；`batch_state.json` 原子更新；CSV status 与 Markdown
report 可直接审计。

Reserve 不属于自动执行范围。只有已记录为 official command、timeout、trajectory
export、incomplete pair、alignment 或质量门失败的 primary 才会按 task-specific
reserve rank 进入 `reserve_replacement_proposal.csv`，且状态固定为
`PROPOSED_NOT_APPROVED_NOT_EXECUTED`。环境/config、missing DB 或 orchestration failure
不会自动消耗 reserve。任何 reserve rollout 都需要另行审查和批准。

最终冻结的真实数据验收位于 `outputs/stage7_m6_4b_locked_batch_mac_v2/`。Batch tool
SHA-256=`ef0026b3cc20942846035ac23d0d16d616a3d7dd6675e9a0f9c2612871d7fb06`，并与
command timeout 一起写入 immutable batch manifest；代码或 timeout 变化时 resume
会 fail closed。Full dry-run 验证
375 primary、75 reserve 和450个 DB；随后只执行 order 1 的已观察 smoke token，
2/2 official commands 成功、298 trajectory rows、same-log 与 strict-token 均通过，
耗时36.3秒。第二次完全相同的 `--resume` 只输出 `SKIP`，event ledger 保持2行且
未创建 `attempt_002`。当前状态为：

```text
SUCCEEDED: 1 scenario / 2 rollouts
PENDING: 374 scenarios / 748 rollouts
FAILED_REVIEW_REQUIRED: 0
reserve proposals: 0
```

nuPlan metric aggregator 仍为每个 planner 打印两行“no metric files found for
aggregation”警告，即使 job name 已包含 challenge；但 per-scenario metric parquet、
runner report、msgpack 和 Stage7C trajectory 均存在，batch validation 为 PASS。该
警告当前不作为 trajectory collection failure，也不用于 reserve replacement。
本步没有读取 embedding、BDD 或 effect size，也没有启动 order 2–375。

### Order 2–6 技术 canary 与时间预算（2026-08-07）

在 immutable batch tool hash 保持一致的前提下，按冻结 collection order 执行
order 2–6。该 canary 只检查运行、解析、pair completeness 和对齐，不读取 planner
effect、embedding 或 BDD。五个场景全部通过，随后原样 `--resume` 全部 `SKIP`；
event ledger 从执行后的12行保持不变，没有 `attempt_002`、失败或 reserve proposal。

| order | frozen task | assertive (s) | conservative (s) | scenario end-to-end (s) |
| ---: | --- | ---: | ---: | ---: |
| 1 | following interaction | 16.92 | 16.45 | 36.26 |
| 2 | lane change | 14.49 | 14.26 | 32.34 |
| 3 | stop-go control | 17.57 | 17.78 | 38.39 |
| 4 | high motion dynamics | 13.91 | 13.76 | 30.70 |
| 5 | dense/vulnerable interaction | 15.59 | 15.50 | 34.13 |
| 6 | following interaction | 19.18 | 18.79 | 41.05 |

六场景 mean=`35.48 s`、median=`35.20 s`、sample SD=`3.86 s`；order 2–6 连续
canary 实际 wall time=`176.64 s`，有效吞吐=`35.33 s/scenario`。顺序单 worker
外推为：

```text
原始 374 pending scenarios: 3h 40m 13s central estimate
canary 后剩余 369 scenarios: 3h 37m 16s central estimate
按观测最快/最慢值外推 374 scenarios: 3h 11m – 4h 16m
建议正式预留: 4.5–5h（温控降频、磁盘与技术重试余量）
```

六个输出平均约13.66 MiB/scenario，剩余369个按当前 artifact 集合估计约4.92 GiB
新增磁盘占用。该估算样本仍小，不能视为服务级时限；Mac休眠、其他高负载进程或
场景复杂度尾部都可能延长 wall time。当前状态为6 `SUCCEEDED`、369 `PENDING`、
0 `FAILED_REVIEW_REQUIRED`、0 reserve proposals。

## Stage 7 Milestone 6.4C locked technical recovery（2026-08-07）

M6.4B 已完成全部375个 frozen primary 场景，原始状态为283成功、92技术失败。
GitHub Issue #238 新增独立的 M6.4C audit/recovery 层，不修改冻结的 M6.4B runner
或 Stage7C。审计输入限于 frozen collection metadata、SQLite scene 结构和 batch
technical status；明确禁止读取 embedding、BDD、effect size、trajectory metric 与
planner outcome。

权威审计 `outputs/stage7_m6_4c_locked_recovery_audit_v2/` 将92个失败拆分为：90个
`INVALID_SCENE_POSITION` 和2个 `RETRY_WITH_QUOTED_HYDRA_TOKEN`。前者来自 nuPlan
官方 scene 查询对最前两帧及最后两帧的边界排除，不应消耗重试；后者 token 分别被
OmegaConf 解释为科学计数法和整数，scene 本身有效。75个冻结 reserve 中58个技术
可运行、17个同样被 scene position 排除。

恢复严格按 audit 生成的 frozen plan 执行。2个 quoted-primary retry 全部通过；随后
lane-change 10个和 high-motion 10个 frozen reserve 也全部通过。22个恢复场景均为
2/2 official planner successes，trajectory pair 完整，same-log 与 strict-token
alignment 通过，0失败。quoted retry 平均33.67秒/场；reserve 平均33.00秒/场。

最终可用完整 pairs 为305：following=60、lane-change=60、stop-go=67、
high-motion=55、dense/vulnerable=63。Lane-change 已补齐，high-motion 仍比冻结目标
少5对，且该任务冻结 primary/reserve 已耗尽。M6.4C 在此 fail closed：不得根据本轮
结果临时挑选集合外样本。下一里程碑应先形成 pre-treatment supplemental protocol
amendment，冻结新增 high-motion 候选的独立性、排序 salt、追加数量和全部 hash，
再运行补充仿真；在补足前不得宣称五任务 simultaneous confirmatory quota 达标。

## Stage 7 Milestone 6.4D high-motion supplement（2026-08-08）

Issue #239 将 M6.4C 剩余的5对 high-motion 缺口固化为独立 outcome-blind protocol
amendment。补充选择不读取 planner outcome、embedding、BDD、effect size 或 trajectory
metric；只从原 eligible inventory 使用冻结 task mapping、identity metadata 和
SQLite official scene-position runnability。

新 salt `stage7-m6.4d-high-motion-supplement-v1` 排序后，排除全部 development 和原
M6.4 primary/reserve token/log，并限制补充集合每 log 1条。前16个候选产生10个
technically runnable distinct-log 场景、4个 invalid scene position 和2个 duplicate
supplement log。冻结5 primary + 5 reserve；token/log overlap audit 全为0。

5个 supplemental primary 全部完成，10/10 official planner runs 成功，trajectory
pair completeness、same-log 与 strict-token alignment 全部通过；端到端均值31.14秒，
0技术失败，因此5个 reserve 未执行。最终完整 pairs 为310：following=60、
lane-change=60、stop-go=67、high-motion=60、dense/vulnerable=63。五任务现均达到
M6.3 冻结的每任务60对要求。

下一阶段进入 confirmatory analysis：必须沿用 M6.2/M6.3 冻结的 task-conditioned
paired BDD、permutation、Holm correction、quality sensitivity 和 planner treatment
fingerprints。M6.4D 不授权根据新增结果修改 embedding 或统计 estimand。

## Stage 7 Milestone 6.5 locked confirmation（2026-08-08）

Issue #240 将 M6.4B/C/D 的310个 complete pairs 固化为新 log/scenario-disjoint
confirmation population。来源为283 primary successes、2 quoted-primary retries、20
frozen reserves 和5 high-motion supplement；五任务计数为60/60/67/60/63。逐场
Stage7C re-audit、development overlap、planner fingerprints 和 M6.3 sample targets
全部通过。

分析在读取确认 embedding 前冻结：原始64D learned embedding、biased single-RBF
MMD² V-statistic、exact pooled positive off-diagonal median bandwidth、100000次
within-pair swaps、plus-one p、五任务 learned-embedding Holm family、Tier A/A+B
quality Holm family，以及 interaction/trajectory mechanism controls。

正确 Mac context 需要显式 `PYTHONPATH=../tuplan_garage`；缺少路径的退化预检因
neighbor slots 全空被隔离且未用于统计。修正后 `[620,150,83]` context 在23分56秒
完成，neighbor coverage 非零。Overall primary MMD²=`0.0044694`，0/100000 exceedances，
p=`9.9999e-6`；五个 pre-treatment tasks 均通过 Holm，确认了跨任务的 planner-
conditioned behavior distribution difference。

限制同样冻结报告：全局 fallback=`10.59%`，高于旧 M2B 5% scale-readiness 门；
fallback 与 embedding distance 强相关。Tier A 和 Tier A+B original sensitivities 通过
Holm，但 Tier A residual p=`0.126`。因此不能从 BDD 推断安全性、planner 优越性，
也不能声称全部分布差异均不受 lane-context quality 影响。

## Stage 7 Milestone 6.6 confirmation evidence package（2026-08-08）

Issue #241 将 M6.5 锁定结果整理为可复现的论文证据包，不修改 M6.1/M6.2/M6.5
工具、样本、embedding、estimand、permutation 或 correction family。M6.6 builder 在
写出前重新计算 analysis lock 和 M6.5 result summary 中记录的每个输入 SHA-256，并
fail-closed 检查310 pairs、620 rows、development disjointness、power target、pair
completeness、五任务 Holm 和冻结质量计数。

权威输出 `outputs/stage7_m6_6_confirmation_evidence_v1/` 包含10组 CSV/Markdown 表、
6组 PNG/PDF 图、machine-readable summary、provenance、报告和中英文 manuscript
段落。总体 primary 与五任务结果只被读取和转录，没有在 M6.6 重算确认性 p 值。

探索性 lane-quality attribution 使用固定 seed、10000次 bootstrap，报告总体任务分层、
task-adjusted rank residual 及五个任务内关联。最大 paired fallback 的总体 Spearman
rho=`0.5088`、95% CI `[0.4086,0.6035]`；任务调整后 rho=`0.4499`、95% CI
`[0.3842,0.5719]`。fallback quality 关联在控制 pre-treatment task composition 后仍然
存在，因此最终状态为 `PASS_WITH_QUALITY_LIMITATIONS`。所有 lane-quality 变量均在
rollout 后实现，只能用于 descriptive/exploratory limitation，不能用于 causal
adjustment、planner superiority 或 safety claim。

## Stage 6L–6N follow-up（2026-08-11）

Stage6L完整性审计发现原Stage6K dose50/75 context为零邻车覆盖；原因是Mac运行时未同时
提供nuPlan devkit与tuPlan Garage路径。旧输出未删除，修复版写入新目录，并在context build
和正式freeze两处增加fail-closed非零覆盖检查。修复后25/50/75/100%完整64D overall BDD
仍全部通过四档Holm，Stage6J纯纵向窄主张保留。

representation消融显示完整64D的median Z_BDD/task-dose pass为7.539/7，邻车置零为
11.066/11，显式ego13D为21.082/12，手工46D为5.384/2。下一步不应把lane pipeline调参
当作提高BDD的首选；完整context-v2只能独立冻结后用于稳健性复验。

面向真实异场景release的Stage6M复用800-pair池，对raw、task、context-balanced和
task+context四方法分别A/A标定。n=400 detection为63.0%/65.0%/66.5%/64.5%，FPR均约5%；
context balance相对raw的+3.5pp不显著且跨样本量不稳定。因此路线图进入Stage6N独立训练
协议准备：扩大Waymo纵向coverage并增加contrastive/ranking与纵向auxiliary objectives，
但旧checkpoint和Stage6J/6K/6M冻结证据不得覆盖。完整报告见
`docs/stage6n_context_balanced_retraining_decision.md`。
