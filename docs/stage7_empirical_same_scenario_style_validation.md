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

### Stage 7C — Same-Scenario A/B Rollout Generation

Collect or simulate policy A/B trajectories on the same scenario set.

For nuPlan:

- select a fixed scenario subset;
- run planner A and planner B in closed loop;
- export rollouts with consistent scenario IDs.

For CARLA:

- define fixed routes / scenarios;
- run conservative and aggressive agents or two policy variants;
- export synchronized ego and neighbor trajectories.

### Stage 7D — Behavior Embedding and Task-Conditioned BDD

Reuse the existing Stage 6C pipeline:

- build context features and embeddings;
- build behavior-event task slices;
- compute task-conditioned BDD between A and B.

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
