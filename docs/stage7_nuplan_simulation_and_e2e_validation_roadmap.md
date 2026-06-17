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
| 7E-core | nuPlan adapter to Stage 5D context contract | Stage 7C official simulation + nuPlan map | context_traj.npy [N,T,83] | PASS for geometric smoke; needs Stage5-core/lane-aware refactor before final evidence |
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
stage5d_derived_formula_matched == true
context_traj_no_nonfinite == true
planner_indices_non_empty == true
stage5d_core_reused == true
```

Current final Stage 7E thesis path:

```text
tools/build_nuplan_5neighbor_context_dataset.py
  -> context_traj.npy [N,T,83]
  -> tools/stage7e_embed_stage6_dataset.py --context_dataset_dir
  -> embedding.npy / embeddings/shard_000000/embeddings.npy
```

Deprecated path:

```text
tools/stage7e_embed_stage6_dataset.py --dataset_dir ... --context_layout stage5d83
```

This old path tried to relabel Stage 7D distance top-K `neighbor_seq[:, :5]` as Stage 5D semantic slots and is not valid thesis evidence. It must raise a clear error; only `--dataset_dir --context_layout pad_to_checkpoint_dim` remains available as explicit smoke/debug.

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

`pad_to_checkpoint_dim` or zero-padding is smoke-only and must not be used as final thesis evidence. The deprecated `--dataset_dir --context_layout stage5d83` reconstruction path must fail with: `Final Stage7E Stage5D context must be built by build_nuplan_5neighbor_context_dataset.py and loaded via --context_dataset_dir. Stage7D top-K neighbor_seq cannot be relabeled as Stage5D semantic slots.`

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

### P0 — Stage 5D Common-Core Refactor

Refactor so Stage 5D schema, slot names, channel order, formulas, and assignment logic have one source of truth.

Required invariant:

```text
Stage 7 nuPlan builder imports Stage 5D core constants / functions.
It must not define its own SLOT_NAMES or neighbor channel order.
```

Validation fields:

```text
stage5d_core_reused = true
stage5d_slot_names_source = tools.lane_aware_assignment.SLOT_NAMES or tools.stage5d_context_core.SLOT_NAMES
stage5d_feature_formula_source = tools.stage5d_context_core
stage5d_slot_schema_matched = true
stage5d_slot_order_matched = true
stage5d_derived_formula_matched = true
```

### P0 — Derived-Formula Parity

Align nuPlan builder with Stage 5 Waymo formulas for:

```text
delta_x / delta_y
closing
TTC
THW
accel
yaw_rate
```

Do not overuse the word proxy. Use:

```text
direct_from_state
derived_same_as_stage5
approximated_or_not_stage5_matched
```

### P1 — nuPlan Lane-Aware Slot Assignment

Add:

```text
--assignment_mode lane_aware_with_geometric_fallback
```

Use nuPlan map/lane objects through a LaneInfo adapter, then call Stage 5 `assign_neighbors_lane_aware`.

Record:

```text
ego_lane_projection_success_rate
candidate_lane_projection_success_rate
fallback_assignment_used_rate
lane_context_quality counts
slot coverage and sanity by slot
```

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
