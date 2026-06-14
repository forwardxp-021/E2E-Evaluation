# Stage 7A — nuPlan Same-Scenario Policy Validation

> **Current roadmap note:** This document is retained as Stage 7A / early nuPlan-readiness notes. The current Stage 7 A–G roadmap is [`stage7_nuplan_simulation_and_e2e_validation_roadmap.md`](stage7_nuplan_simulation_and_e2e_validation_roadmap.md). In the current roadmap, E2E model integration belongs to Stage 7F, and final thesis-facing synthesis belongs to Stage 7G.


## 0. Scope Clarification

Stage 7A.0 / Stage 7A.1 expert-data inspection is only an infrastructure step. The final Stage 7 objective is same-scenario policy A/B rollout and task-conditioned BDD. Expert trajectory export is not the main validation result.

For the full Stage 7 A-E structure, see [Stage 7 Master Plan — Same-Scenario Policy / E2E BDD Validation](stage7_master_plan_same_scenario_policy_bdd.md).

## 1. Motivation

Stage 6 pseudo splits are useful for validating the behavior embedding and task-conditioned BDD protocol, but they are still pseudo-label based. They show that the representation can separate constructed style groups, not that real policies or drivers produce separable behavior distributions under matched scenarios.

Stage 7A moves toward empirical validation with same-scenario rollouts. nuPlan is selected as the first open-source target because it provides real-world planning scenarios and closed-loop planner simulation. This makes it closer to autonomous-driving policy evaluation than human-driver-only style classification.

## 2. Hardware-Aware Strategy

Available hardware is limited:

- MacBook Air M5 16GB.
- Intel Ultra5 CPU with 8GB discrete GPU.

Recommended use:

- MacBook Air M5 16GB: documentation, lightweight analysis, code editing, result inspection.
- Intel Ultra5 + 8GB GPU: main nuPlan runtime machine.
- Use Ubuntu or WSL2 Ubuntu for nuPlan.
- Use Python 3.9 because the nuPlan devkit is tested primarily on Python 3.9 on Ubuntu.
- Start with nuPlan mini, not the full dataset.
- Avoid large-scale model training and sensor-based E2E training at this stage.
- Start with rule-based / configurable planner variants.

## 3. Stage 7A Goal

Goal:

- Same scenario set `S`.
- Policy A = conservative planner.
- Policy B = aggressive planner.
- Run both policies on the same scenario set.
- Export ego trajectories and neighbor context.
- Convert rollouts to the existing context dataset format.
- Reuse the existing Stage 6C behavior-event builder and task-conditioned BDD report.

This is not yet E2E model A/B validation, but it is empirical same-scenario policy A/B validation. It is stronger than a pseudo split because the A/B labels come from actual rollout policies rather than feature-derived pseudo labels.

## 4. Initial Policy Definitions

> **Current Stage 7C planner-strategy note:** Stage 7C should first use existing nuPlan devkit / official-compatible planners before any custom planner. Preferred order is expert / log replay, simple planner, IDM planner, configurable IDM-style variants, and only then a minimal nuPlan `AbstractPlanner`-compatible wrapper if needed. Offline pseudo rollout and numpy rewriting of logged trajectories are not acceptable Stage 7C evidence. See the current roadmap section [`Stage 7C Planner Strategy`](stage7_nuplan_simulation_and_e2e_validation_roadmap.md#stage-7c-planner-strategy).

Start with configurable planner variants when official nuPlan/devkit support is available.

Conservative planner:

- lower target speed;
- larger headway;
- lower acceleration;
- earlier braking.

Aggressive planner:

- higher target speed;
- smaller headway;
- higher acceleration;
- later braking.

These planners are an empirical first step. Later, full E2E model A/B trajectories can replace rule-based policy variants without changing the Stage 7A interface.

## 5. Rollout Data Schema

Required ego fields:

| field | meaning |
|---|---|
| `scenario_id` | stable scenario identifier |
| `policy_id` | conservative / aggressive / planner name |
| `timestamp` | timestamp or simulation time |
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
| `neighbor_id` | neighboring agent identifier |
| `neighbor_x` | neighbor x position |
| `neighbor_y` | neighbor y position |
| `neighbor_vx` | neighbor x velocity |
| `neighbor_vy` | neighbor y velocity |
| `neighbor_speed` | neighbor speed |
| `neighbor_heading` | neighbor heading |
| `neighbor_type` | vehicle / pedestrian / cyclist / other |

## 6. Output Target

The converter should produce the existing sharded context format:

- `ego_seq.npy`
- `neighbor_seq.npy`
- `metadata.npy` or `metadata.csv`
- `shard_manifest.json`
- `feature_schema.json`

Then reuse:

- `tools/stage6c_build_behavior_events_v2.py`
- `tools/stage6c_task_conditioned_bdd_report.py`

## 7. Minimal Experiment Plan

### Stage 7A.0 — Environment Check

- Install the nuPlan devkit on Ubuntu or WSL2 Ubuntu.
- Download nuPlan mini and maps.
- Confirm scenarios can be loaded.
- Run a small planner tutorial or minimal planner simulation.

### Stage 7A.1 — Rollout Export

- Select 20–50 nuPlan mini scenarios.
- Run conservative and aggressive planner variants.
- Export rollout parquet / CSV files.
- Preserve `scenario_id`, `policy_id`, and timestamps.

### Stage 7A.2 — Converter

- Convert exported rollouts to the existing context dataset format.
- Generate `ego_seq.npy`, `neighbor_seq.npy`, metadata, `shard_manifest.json`, and `feature_schema.json`.
- Build behavior events.
- Compute task-conditioned BDD.

### Stage 7A.3 — Report

- Compare policy A/B under the same scenarios.
- Report task-conditioned BDD for:
  - `task_following`
  - `task_lane_change`
  - `task_yield_conflict`
  - `task_hesitation`
- Treat cut-in, lead-brake, queue, and overtake as auxiliary / proxy-heavy diagnostics.

## 8. Placeholder Commands

Export conservative rollouts:

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

Export aggressive rollouts:

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

Convert rollouts to context dataset:

```bash
python tools/stage7a_convert_rollouts_to_context_dataset.py \
  --rollout_dir outputs/stage7A_nuplan \
  --output_dir outputs/stage7A_nuplan/context_dataset \
  --overwrite
```

Build behavior events:

```bash
python tools/stage6c_build_behavior_events_v2.py \
  --shard_manifest outputs/stage7A_nuplan/context_dataset/shard_manifest.json \
  --feature_schema_path outputs/stage7A_nuplan/context_dataset/feature_schema.json \
  --output_dir outputs/stage7A_nuplan/behavior_events_v2 \
  --overwrite
```

Compute task-conditioned BDD:

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

These commands are placeholders until the nuPlan rollout exporter and converter are connected to real nuPlan APIs.

## 9. Limitations

- This is not full E2E yet.
- Rule-based planner variants are only the first empirical validation.
- Full E2E A/B trajectories can replace conservative / aggressive planner variants later.
- nuPlan mini is small; a final dissertation experiment may need more scenarios if hardware and time allow.
- Sensor-based E2E model training is intentionally out of scope for Stage 7A.

## Stage 7B.2 — expert dynamic context converter

Stage 7B.2 converts the Stage 7B.1 expert export CSV files into a Stage 6-style **dynamic-only** context dataset.  It reads expert ego trajectory rows, nearby dynamic object rows, and optional selected-scene metadata, then writes fixed-length `ego_seq.npy` / `neighbor_seq.npy` windows plus metadata and schema files.

This step is infrastructure validation only.  It does not run planner simulation, does not generate fake rollout data, does not modify Stage 6C result files, and does not change BDD logic.

### 命令

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

### 输出

- `ego_seq.npy`: `[N, 80, 7]` dynamic ego windows.
- `neighbor_seq.npy`: `[N, 80, 10, 9]` nearby-object windows.
- `metadata.csv`: one row per generated window, with `source=nuplan_expert`, `policy_id=expert`, and `map_odd_status=not_built`.
- `shard_manifest.json`: dynamic dataset manifest with `map_odd_feat_path=null`, `map_feature_status=not_built`, and `next_map_stage=Stage 7B.3 map/ODD feature builder`.
- `feature_schema.json`: dynamic feature order and reserved Stage 6-style map/ODD feature names.
- `conversion_report.md` and `warnings.json`: conversion summary and structured warnings.

## Stage 7B.3 — nuPlan map/ODD feature builder placeholder

Stage 7B.3 is reserved for building Stage 6-style map/ODD context for each Stage 7B.2 generated window.  It is not implemented yet.

Planned purpose:

- Build Stage 6-style map/ODD features for each generated window.
- Use nuPlan maps / `map.gpkg` or the nuPlan map API.
- Align map features by ego path and scene/map location.

Planned outputs:

- `map_odd_feat.npy`
- `map_odd_meta.csv`
- `map_odd_feature_schema.json`
- `map_odd_report.md`
- `warnings.json`

Stage 7B.2 reserves this interface in `shard_manifest.json` and `feature_schema.json`; it deliberately does not parse maps or fabricate map/ODD values.
