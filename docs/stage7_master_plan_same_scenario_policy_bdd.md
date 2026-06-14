# Stage 7 Master Plan — Same-Scenario Policy / E2E BDD Validation

> **Current roadmap note:** This older master plan is retained for historical A–E policy-BDD context. The current primary roadmap is [`stage7_nuplan_simulation_and_e2e_validation_roadmap.md`](stage7_nuplan_simulation_and_e2e_validation_roadmap.md), with corrected official-nuPlan-simulation guardrails plus Stage 7F and Stage 7G.


## 1. Research Goal

Stage 7 is the empirical bridge from Stage 6 pseudo-label validation to real policy / model style validation.

The core Stage 7 research goal is:

> Use nuPlan simulation / rollout to validate that, under the same scenario set, different policy styles or E2E model versions produce separable behavior embedding distributions measured by task-conditioned BDD.

In short:

```text
same scenario + different policy/E2E model
→ different driving style
→ large task-conditioned BDD
```

Stage 7 should create empirical same-scenario policy A/B rollouts with nuPlan first, then keep the same context-dataset and BDD interface when replacing rule-based planners with learning-based planner checkpoints or company E2E model versions.

## 2. What Stage 7 Is NOT

Stage 7 must not be interpreted as another Waymo-style pseudo split or as a direct re-run of the existing pipeline on nuPlan expert human trajectories.

Stage 7 is **not**:

- another pseudo split;
- simply applying the Waymo pipeline to nuPlan expert human driving;
- training a new E2E model from scratch;
- full nuPlan benchmark evaluation;
- using expert trajectory export as the final validation result.

Expert trajectory export is only for schema discovery and converter validation. It is infrastructure work, not the empirical proof of behavior-style separability.

## 3. Stage 7 Sub-Stages

### Stage 7A — nuPlan Data Readiness and Schema Understanding

Purpose:

- Verify nuPlan mini DBs and maps are available.
- Inspect SQLite tables.
- Confirm the following tables / concepts are readable:
  - `ego_pose`
  - `lidar_pc`
  - `lidar_box`
  - `track`
  - `scenario_tag`

Outputs:

- `mini_db_inventory.csv`
- `mini_schema_report.json`
- `mini_check_report.md`

Important clarification:

- Stage 7A does **not** prove policy style separability.
- Stage 7A only proves data readiness and schema understanding.
- Stage 7A.0 readiness checks must remain available because they are the safe first diagnostic before any rollout work.

### Stage 7B — Rollout / Context Dataset Interface Validation

Purpose:

- Export expert ego trajectory and nearby object context only to understand schema and validate conversion.
- Convert nuPlan-like trajectory data into the existing context dataset format:
  - `ego_seq.npy`
  - `neighbor_seq.npy`
  - `metadata`
  - `shard_manifest.json`
  - `feature_schema.json`

Important clarification:

- Expert export is converter debug data.
- Expert export is not the final empirical validation.
- Expert export should not be written as the main Stage 7 result.
- The point of Stage 7B is to make sure the nuPlan-derived data interface can feed the existing behavior embedding and BDD pipeline.

### Stage 7C — Same-Scenario Policy A/B Rollout Generation

Purpose:

- Select the same scenario set `S` from nuPlan mini.
- Run Policy A and Policy B on exactly the same scenarios.
- Policy A = conservative planner.
- Policy B = aggressive planner.

Policy definitions:

Conservative:

- lower target speed;
- larger headway;
- lower acceleration;
- earlier braking;
- more conservative gap acceptance.

Aggressive:

- higher target speed;
- smaller headway;
- higher acceleration;
- later braking;
- more assertive gap acceptance.

Outputs:

- conservative rollout ego/object CSV;
- aggressive rollout ego/object CSV;
- `scenario_list.csv`;
- `rollout_manifest.json`.

This is the first true empirical step beyond pseudo split because the A/B labels come from actual rollout policies on matched scenarios, not from feature-derived pseudo labels.

### Stage 7D — Policy-Style BDD Empirical Validation

Purpose:

- Convert policy A/B rollouts to the context dataset format.
- Build behavior events.
- Run task-conditioned BDD.

Required comparisons:

1. Negative control:
   - random A/B split within the same policy should have low BDD.
2. Policy style comparison:
   - conservative vs aggressive should have higher BDD.

Primary tasks:

- `task_following`
- `task_lane_change`
- `task_yield_conflict`
- `task_hesitation`

Auxiliary tasks:

- `task_cutin_response`
- `task_lead_brake_response`
- `task_queue_approach`
- `task_overtake_opportunity`

Expected style deltas:

Following:

- aggressive has lower THW / smaller distance;
- higher deceleration / jerk;
- higher aggressiveness score.

Lane change:

- aggressive has higher lateral speed;
- higher lateral acceleration;
- higher yaw rate;
- higher sharpness score.

Yield conflict:

- aggressive has higher assertiveness;
- higher conflict acceleration;
- lower yielding / courtesy if available.

Hesitation:

- interpret as maneuver execution / hesitation-like behavior, not psychological hesitation.

Stage 7D is the main empirical policy-style BDD validation. Stage 7C and Stage 7D are the core proof; Stage 7A and Stage 7B are infrastructure.

### Stage 7E — Scaling and Real E2E Replacement

Purpose:

- Scale scenario count from 20 to 50 / 100 / 300 if feasible.
- Replace rule-based policy variants with learning-based planner checkpoints if available.
- Replace nuPlan policy rollout with company E2E model A/B rollout when available.
- Keep the same data interface and BDD pipeline.

The important design constraint is that Stage 7E should not require changing the BDD computation logic. The source of rollout trajectories may change, but the context dataset interface and task-conditioned BDD protocol should remain stable.

## 4. Acceptance Standard for Stage 7 Conclusions

A Stage 7 result can support the dissertation claim only if it demonstrates all of the following:

1. The compared trajectories come from the same scenario set or a strictly matched scenario set.
2. The compared groups differ by policy / planner / E2E model identity, not by a feature-derived pseudo split.
3. A negative control random split within the same policy has low task-conditioned BDD.
4. Conservative vs aggressive or model A vs model B has higher task-conditioned BDD on primary behavior tasks.
5. The report explicitly distinguishes infrastructure outputs from empirical validation outputs.

