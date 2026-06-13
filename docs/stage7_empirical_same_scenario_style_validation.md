# Stage 7 — Empirical Same-Scenario Style Separability Validation

## 1. Why Stage 6D Matched Pseudo Is Not Enough

Stage 6C v2 three-split validation is complete, but the next main contribution should move beyond pseudo splits.

A matched-task Stage 6D pseudo split would improve over a global pseudo split because it can control task context more carefully. However, it is still pseudo-label based. It can show that the representation separates hand-constructed style groups, but it does not provide direct empirical evidence that two real policies, models, or drivers produce different behavior embedding distributions under the same scenario set.

The next target is therefore empirical same-scenario policy / driver validation rather than another pseudo split.

## 2. Empirical Goal

Core hypothesis:

> Under the same scenario set and the same driving task, different driving policies, models, or drivers should produce significantly different behavior embedding distributions.

Stage 7 evaluates this hypothesis by holding scenario exposure fixed and comparing policy / driver behavior within matched task slices.

Required controls:

- Same scenario set.
- Same driving task / behavior-event slice.
- Different policy, model, or driver style.
- Task-conditioned BDD computed within each task.

Example comparisons:

- E2E / planner policy A vs policy B on the same replay scenarios.
- Human driver style A vs human driver style B under matched scenario families.
- Conservative vs aggressive simulated agents on the same CARLA scenarios.


## Current Status Snapshot

| Stage | Name | Status | Evidence |
|---|---|---|---|
| Stage 7A.0 | nuPlan mini readiness check | PASS | 64 mini DBs, 4 map.gpkg, all key SQLite tables readable, no DB open failures |
| Stage 7B.1 | expert ego/object context export | PASS | 5 DB × 5 scenes export succeeded; 25 scenes; 4797 ego rows; 47970 nearby object rows; warnings none; direct lidar_pc.scene_token = scene.token for all scenes |
| Stage 7B.2 | Stage6C-compatible dynamic context converter | PASS | 23 generated windows; ego_seq [23,80,8]; neighbor_seq [23,5,80,15]; context_traj [23,80,83]; context_mask [23,80,5]; metadata rows 23; interaction features [23,33] |
| Stage 6C smoke on nuPlan expert context | PASS | Stage 6C behavior-event builder ran successfully; total_rows 23; shard_count 1; no manifest/layout/metadata errors |
| Stage 7B.3 | nuPlan map/ODD feature builder | NOT STARTED | map/ODD features reserved but not yet built |
| Stage 7C | same-scenario conservative/aggressive policy rollout | NOT STARTED | pending Stage 7B.3 |
| Stage 7D | policy-style task-conditioned BDD validation | NOT STARTED | pending Stage 7C |

Stage 7B.1 and Stage 7B.2 used nuPlan expert / historical trajectories only as infrastructure validation. They are not final evidence of policy-style separability.

## Two-Pipeline Architecture: Waymo vs nuPlan

Waymo pipeline:

- Passive offline real-world trajectory data.
- Used to build and validate behavior representation, behavior-event metrics, and task-conditioned BDD.
- Cannot provide same-scenario policy A/B rollouts.
- Role: method construction, sanity checks, negative controls, pseudo aggressive / conservative validation, and protocol validation.

nuPlan pipeline:

- Active same-scenario empirical validation pipeline.
- Used because the same scenario set can be run by different planners / policies.
- Role: validate whether behavior embedding + task-conditioned BDD can separate different policy / E2E driving styles under matched scenarios.
- Stage 7C / 7D are the main empirical proof.
- Stage 7A / 7B are infrastructure.

Summary: Waymo proves the method can be defined and validated offline. nuPlan proves the method can distinguish different policies under the same scenario set.

Stage 7 is **not**:

- another pseudo split;
- simply re-running the Waymo pipeline on nuPlan expert human driving;
- full nuPlan benchmark evaluation;
- training a new E2E model from scratch.

Stage 7 is:

- same scenario set;
- different policy / planner / E2E model;
- same common context interface;
- task-conditioned BDD and style-delta report.

## 3. Candidate Data Sources

### 3.1 Company E2E Model A/B Data

This is the strongest option if available.

- Same replay scenario set or same road-test task route.
- Model A vs model B trajectories.
- Directly aligned with the dissertation claim about E2E autonomous driving behavior-style drift.
- Strongest empirical evidence, but may involve data, privacy, and IP constraints.

### 3.2 nuPlan Closed-Loop Planner Rollout

nuPlan is the recommended open-source next step if company E2E A/B data is unavailable.

- Public autonomous driving planning benchmark.
- Supports closed-loop planner rollout.
- Uses real-world driving scenarios.
- Allows different planners / policies to be evaluated on the same scenario set.
- Better aligned with autonomous driving evaluation than human-driver-only classification.

Candidate Stage 7 use:

- Select a fixed nuPlan scenario subset.
- Run planner A and planner B on the same scenarios.
- Convert resulting ego rollouts and available surrounding-agent context into the Stage 7 common schema.
- Compute task-conditioned BDD between planner A and planner B.

### 3.3 CARLA Multi-Agent Rollout

CARLA is the second open-source choice.

- Same scenario, different agents or different style configurations.
- Can use rule-based conservative / aggressive agents or pretrained CARLA leaderboard agents.
- Good for controlled same-scenario validation.
- More simulation-heavy and less directly tied to real-world planner benchmark evaluation.

### 3.4 Human Driver Public Datasets

Human-driver datasets are useful as auxiliary validation, but they should not be treated as the main E2E validation target unless driver identity and task context are sufficiently stable.

- UAH-DriveSet: useful for aggressive / normal / drowsy single-driver behavior-style validation, but lacks full surrounding-vehicle task context.
- highD / inD / rounD / Waymo: useful for matched traffic-scene human response analysis, but usually lacks stable driver identity.

Human-driver public data should be treated as auxiliary human-style validation, not the main E2E policy validation.

## 4. Recommended Priority

1. Company E2E A/B data, if accessible.
2. nuPlan closed-loop planner rollouts.
3. CARLA same-scenario multi-agent rollouts.
4. Human-driver public datasets as auxiliary validation.

If no company data is available, start with nuPlan. It is an autonomous driving planning benchmark, supports closed-loop planner rollout, uses real-world driving scenarios, allows same-scenario policy comparison, and is closer to E2E AD evaluation than human-driver-only datasets.

## 5. Stage 7 Common Rollout Schema

All Stage 7 sources should first be converted into a common rollout schema independent of source.

Required fields:

| field | meaning |
|---|---|
| `scenario_id` | stable scenario identifier |
| `policy_id` or `driver_id` | model / policy / driver identity |
| `timestamp` | time index or timestamp |
| `ego_x` | ego x position |
| `ego_y` | ego y position |
| `ego_vx` | ego x velocity |
| `ego_vy` | ego y velocity |
| `ego_speed` | ego speed |
| `ego_accel` | ego acceleration |
| `ego_heading` | ego heading |
| `ego_yaw_rate` | ego yaw rate |

Optional neighbor fields:

| field | meaning |
|---|---|
| `neighbor_id` | neighboring agent identity |
| `neighbor_x` | neighbor x position |
| `neighbor_y` | neighbor y position |
| `neighbor_vx` | neighbor x velocity |
| `neighbor_vy` | neighbor y velocity |
| `neighbor_speed` | neighbor speed |
| `neighbor_heading` | neighbor heading |
| `neighbor_type` | vehicle / pedestrian / cyclist / other |

Output target:

Convert any Stage 7 source into the existing sharded context dataset format:

- `ego_seq.npy`
- `neighbor_seq.npy`
- `metadata.csv` or `metadata.npy`
- `shard_manifest.json`
- `feature_schema.json`

Then reuse:

- `tools/stage6c_build_behavior_events_v2.py`
- `tools/stage6c_task_conditioned_bdd_report.py`

## 6. Stage 7 Experiment Protocol

1. Select a scenario set `S`.
2. Run or collect trajectories from policy / driver A and policy / driver B on `S`.
3. Convert rollouts into the common sharded context format.
4. Build behavior embeddings using the existing embedding model.
5. Build Stage 6C v2 behavior events.
6. For each task, compute task-conditioned BDD:
   - `task_following`
   - `task_lane_change`
   - `task_yield_conflict`
   - `task_hesitation`
   - optional auxiliary tasks: `task_cutin_response`, `task_lead_brake_response`, `task_queue_approach`
7. Report style deltas:
   - Following: THW, distance, deceleration, jerk.
   - Lane change: lateral speed, lateral acceleration, yaw rate, sharpness.
   - Yield conflict: assertiveness, yielding, conflict acceleration.
   - Cut-in / lead-brake / queue: auxiliary only if detector is proxy-heavy.

## 7. Implementation Roadmap

### Stage 7A — Data Source Decision

Choose the first empirical source:

- company E2E A/B data;
- nuPlan closed-loop planner rollouts;
- CARLA same-scenario rollouts;
- human-driver auxiliary data.

### Stage 7B — Rollout Schema Converter

Implement a source-specific converter to the common schema:

- parse source rollouts;
- normalize coordinate and timestamp fields;
- compute ego speed / acceleration / yaw rate if missing;
- preserve `scenario_id` and `policy_id` / `driver_id`;
- export sharded context dataset files.

### Stage 7B.1 — nuPlan Expert Ego/Object Context Export

Stage 7B.1 exports nuPlan expert ego trajectory and nearby dynamic objects from SQLite DBs into intermediate CSVs.

Inputs:

- nuPlan mini DBs
- `scene`
- `lidar_pc`
- `ego_pose`
- `lidar_box`
- `track`
- `category`

Outputs:

- `expert_ego_trajectory.csv`
- `expert_nearby_objects.csv`
- `selected_scenes.csv`
- `warnings.json`
- `expert_context_export_report.md`

Current small export result:

- selected DB count: 5
- selected scene count: 25
- ego row count: 4797
- nearby object row count: 47970
- warnings: None
- join strategy: direct `lidar_pc.scene_token = scene.token` for all 25 scenes

Interpretation:

- nuPlan SQLite → ego trajectory + nearby object context export is validated.
- This is an infrastructure step only.
- It does not run planner simulation and does not generate policy rollouts.

### Stage 7B.2 — Stage6C-Compatible Dynamic Context Dataset Converter

Stage 7B.2 converts Stage 7B.1 intermediate CSVs into the same dynamic context layout expected by Stage 6C.

Inputs:

- `expert_ego_trajectory.csv`
- `expert_nearby_objects.csv`
- `selected_scenes.csv`

Outputs:

- `shard_manifest.json`
- `feature_schema.json`
- `shards/shard_000000/ego_seq.npy`
- `shards/shard_000000/neighbor_seq.npy`
- `shards/shard_000000/context_traj.npy`
- `shards/shard_000000/context_mask.npy`
- `shards/shard_000000/context_mask_window.npy`
- `shards/shard_000000/metadata.csv`
- `shards/shard_000000/split.npy`
- `shards/shard_000000/interaction_feat_style.npy`
- `conversion_report.md`
- `warnings.json`

Current validation result:

- scenes read: 25
- source dt median: about 0.05001 sec
- source_hz median: about 19.996 Hz
- generated windows: 23
- ego_seq shape: [23, 80, 8]
- neighbor_seq shape: [23, 5, 80, 15]
- context_traj shape: [23, 80, 83]
- context_mask shape: [23, 80, 5]
- context_mask_window shape: [23, 5]
- metadata rows: 23
- interaction feature shape: [23, 33]

Canonical Stage 6C-compatible layout:

`ego_seq`:

- shape [N, T, 8]
- feature order: 0 x; 1 y; 2 vx; 3 vy; 4 heading; 5 speed; 6 accel; 7 yaw_rate

`neighbor_seq`:

- shape [N, 5, T, 15]
- slot order: 0 front; 1 left_front; 2 left_rear; 3 right_front; 4 right_rear
- feature order: 0 valid; 1 dx; 2 dy; 3 rvx; 4 rvy; 5 distance; 6 local_x; 7 local_y; 8 closing_rate; 9 ttc; 10 thw; 11 neighbor_speed; 12 neighbor_accel; 13 relative_heading; 14 neighbor_yaw_rate

`context_traj`:

- shape [N, T, 83]
- constructed as `ego_seq [N,T,8]` concatenated with flattened `neighbor_seq [N,T,5*15]`

`context_mask`:

- shape [N, T, 5]

`context_mask_window`:

- shape [N, 5]

Notes:

- Stage 7B.2 uses geometric neighbor slot assignment only.
- Map / lane-aware assignment will be improved in Stage 7B.3 / 7B.4.
- Neighbor kinematics are currently estimated by finite differences when direct object velocity is unavailable.
- Some short scenes are skipped if they cannot provide 80 frames after downsampling.

### Stage 6C Smoke Test on nuPlan Expert Dynamic Context

The Stage 6C behavior-event builder was run on the Stage 7B.2 generated nuPlan expert context dataset.

Result:

- total_rows: 23
- shard_count: 1
- no array shape error
- no manifest error
- no metadata error
- `behavior_event_metrics_v2.csv` generated
- `behavior_event_bins_v2.csv` generated
- `behavior_event_report_v2.md` generated
- `behavior_event_schema_v2.json` generated
- `behavior_event_warnings_v2.json` generated

Task diagnostics:

- valid tasks:
  - `task_following`: positive_ratio 0.043
  - `task_queue_approach`: positive_ratio 0.087
  - `task_lane_change`: positive_ratio 0.087
  - `task_cutin_response`: positive_ratio 0.087
  - `task_hesitation`: positive_ratio 0.043
- degenerate tasks in this small smoke:
  - `task_lead_brake_response`: positive_ratio 0
  - `task_overtake_opportunity`: positive_ratio 0
  - `task_overtake_executed`: positive_ratio 0
  - `task_yield_conflict`: positive_ratio 1

Interpretation:

- Stage 6C can consume nuPlan-generated dynamic context data.
- The smoke test validates interface compatibility, not policy-style separability.
- Degenerate tasks are acceptable for the small 23-window smoke sample and should not be overinterpreted.
- `yield_conflict` being all-positive suggests that geometric-only slot assignment and map-free context are not sufficient for final validation.

Metric quality notes:

- raw curvature produced a physical range warning and clipping was applied.
- This is not a blocking failure.
- Keep smoothing, clipping, and physical-range diagnostics in future Stage 7 experiments.

### Stage 7B.3 — nuPlan Map/ODD Feature Builder

Status: NOT STARTED.

Reason: Stage 7B.2 currently aligns dynamic context with Stage 6C, but map / ODD context is not built yet. Stage 6 used map / ODD features, so Stage 7 must add equivalent map context before main policy A/B validation.

Target outputs:

- `map_odd_feat.npy`
- `map_odd_meta.csv`
- `map_odd_feature_schema.json`
- `map_odd_report.md`
- `warnings.json`

Target Stage 6-style map / ODD features:

- distance_to_crosswalk_min
- has_crosswalk_near_30m
- distance_to_stop_sign_min
- has_stop_sign_near_40m
- lane_curvature_mean
- lane_curvature_max
- lane_heading_change_total
- lane_count_near_30m
- road_line_count_near_30m
- road_edge_count_near_30m
- crosswalk_count_near_30m
- stop_sign_count_near_40m
- speed_bump_count_near_30m
- map_complexity_score
- intersection_proxy
- map_match_valid
- fallback_full_scenario_path

Stage 7B.3 should align map features by `sample_id` / `scenario_id` / `scene_token` / window start/end timestamp. The first implementation should be map-lite ODD features, not full vector-map polyline tensors. Full map-vector input can be a later enhancement.

### Stage 7C — Same-Scenario Conservative/Aggressive Policy Rollout

Stage 7C is a future main experiment. It should run conservative and aggressive policies on the same fixed nuPlan scenario set, preserve `scenario_id` and `policy_id`, convert rollouts using the same Stage 7B context interface, and ensure same-scenario pairing.

### Stage 7D — Policy-Style Task-Conditioned BDD Validation

Stage 7D is a future main experiment. It should run task-conditioned BDD with a negative control based on random split within the same policy and a main comparison of conservative vs aggressive. Prioritize `task_following`, `task_lane_change`, `task_hesitation`, and `task_yield_conflict` after map / ODD correction. Treat cut-in, lead-brake, and overtake tasks as auxiliary if they remain proxy-heavy.

### Stage 7E — Report

Produce:

- `policy_A_vs_B_task_bdd_report.md`
- style-delta tables;
- top drift cases;
- reliability-tier interpretation.

## 8. Placeholder Command Templates

These commands are placeholders until a source-specific converter exists.

Build Stage 7 common rollout dataset:

```bash
python tools/stage7_convert_rollouts_to_context_dataset.py \
  --source_type nuplan \
  --input_rollout_dir <path> \
  --output_dir outputs/stage7/<experiment_name>/context_dataset \
  --overwrite
```

Build behavior events:

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7/<experiment_name>/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7/<experiment_name>/context_dataset/feature_schema.json \
  --output_dir outputs/stage7/<experiment_name>/behavior_events_v2 \
  --overwrite
```

Compute task-conditioned BDD:

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

## 9. Relationship to Stage 6C

Stage 6C v2 three-split validation should be treated as representation and protocol validation:

- negative-control random split: sanity check;
- pseudo aggressive-vs-conservative split: positive control;
- scene-confounding split: confounding-awareness diagnostic.

Stage 6C scene-confounding is not the main empirical proof of policy / driver style separability. Stage 7 is the next empirical validation step: same scenario set, same driving task, different real policy / model / driver.
