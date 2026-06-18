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
