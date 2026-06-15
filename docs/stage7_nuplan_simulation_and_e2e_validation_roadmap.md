# Stage 7 — nuPlan Simulation and E2E Validation Roadmap

## 1. Top-Level Purpose

Stage 7 uses the nuPlan official simulation environment to generate same-scenario planner / policy / E2E behavior data, then validates whether behavior embedding and Behavioral Distribution Distance (BDD) can detect policy-induced and E2E-induced driving style differences.

The key dataset distinction is:

- **Waymo / earlier stages:** offline logged data and pseudo/synthetic validation.
- **nuPlan / Stage 7:** official simulation-generated behavior data under the same scenarios.

> **Strong warning:** Stage 7C and later must not be described as offline pseudo rollout or numpy trajectory rewriting. Stage 7C, Stage 7F, and all downstream conclusions must be based on official nuPlan simulation outputs.

## 2. Compact A–G Roadmap

| Stage | Purpose | Main data source | Main output | Current status |
|---|---|---|---|---|
| 7A | nuPlan readiness | nuPlan DB/map/devkit | readiness evidence | PASS |
| 7B | context construction | nuPlan logs/maps | `merged_context_feat` | PASS |
| 7C | official simulation with rule/traditional planners | nuPlan simulation | simulated planner trajectories | 7C.1A/7C.1B/7C.1C PASS; 7C.2A/7C.2B PASS; 7C.2C IDM longitudinal-only multi-planner rollout IN PROGRESS; 7C.3 PDM / lateral-interaction planner extension TODO |
| 7D | BDD validation on planner sim data | Stage 7C | paired/unpaired/ODD BDD | TODO |
| 7E | planner-only consolidation | Stage 7D | planner report cards | TODO |
| 7F | E2E model simulation | E2E planner + nuPlan sim | E2E simulated trajectories | TODO |
| 7G | final Stage 7 summary | 7C/7D/7F | final thesis evidence | TODO |

## 3. Stage 7A — nuPlan Readiness

**Definition:** Stage 7A = nuPlan environment / data / map / scenario / simulation API readiness.

**Purpose:**

- Check nuPlan DB access.
- Check map root access.
- Check scenario metadata.
- Check expert ego pose and object extraction.
- Discover available simulation APIs and planner classes.

**Current status:** PASS.

**Expected outputs / evidence:**

- nuPlan DB readable.
- Map readable.
- Selected scenarios readable.
- Ego/object extraction works.
- Simulation APIs can be discovered.

Stage 7A is an infrastructure-readiness stage. It does not by itself prove planner-induced or E2E-induced behavior drift.

## 4. Stage 7B — nuPlan Context Dataset Construction

**Definition:** Stage 7B = build a strict, row-aligned, auditable scenario context dataset for downstream nuPlan simulation and BDD analysis.

**Sub-stages:**

- 7B.1 expert ego/object export.
- 7B.2 dynamic context conversion.
- 7B.3 map/ODD-lite feature builder.
- 7B.4 dynamic + map/ODD merge/alignment.

**Current status:**

- Stage 7B.1: PASS.
- Stage 7B.2: PASS.
- Stage 6C smoke on nuPlan expert context: PASS.
- Stage 7B.3: PASS.
- Stage 7B.4: PASS.
- Stage 7B overall: PASS.

**Validated final directory:**

```text
outputs/stage7b4_nuplan_context_merged/
```

**Validated shapes:**

```text
ego_seq.npy:                  [23, 80, 8]
neighbor_seq.npy:             [23, 5, 80, 15]
context_traj.npy:             [23, 80, 83]
context_mask.npy:             [23, 80, 5]
dynamic_feat_style.npy:       [23, 33]
map_odd_feat.npy:             [23, 37]
merged_context_feat.npy:      [23, 70]
```

Stage 7B is not itself simulation. It is the context and alignment foundation used to select, align, condition, and diagnose later nuPlan simulation and BDD experiments.

## 5. Stage 7C — Official nuPlan Simulation with Rule-Based / Traditional Planners

**Definition:** Stage 7C = run multiple non-E2E planner / policy variants in the official nuPlan closed-loop simulation environment on the same selected scenarios, then export simulated ego trajectories.

**Important correction:**

- Stage 7C must use official nuPlan simulation.
- Stage 7C must not be pseudo rollout-lite.
- Stage 7C must not simply rewrite logged ego trajectories with numpy interpolation.

### Stage 7C Planner Strategy

The first success criterion of Stage 7C is not planner sophistication. The first success criterion is that official nuPlan simulation runs and exports non-empty `simulated_ego_trajectory.csv` and `simulated_ego_seq.npy`. Stage 7C should therefore first use existing nuPlan devkit / official-compatible planners, because they reduce engineering risk and prove the official simulation/export pipeline before adding planner implementation complexity.

Stage 7C should use planners in this priority order:

1. expert / log replay planner if available;
2. simple planner if available;
3. IDM planner if available;
4. configurable IDM-style planner variants;
5. minimal custom `AbstractPlanner`-compatible wrapper only if needed.

Recommended Stage 7C planner variants:

| ID | planner variant | intended role |
|---|---|---|
| P0 | `expert_or_log_replay` | official replay / expert reference and pipeline sanity check |
| P1 | `simple_planner` | minimal official-compatible baseline |
| P2 | `idm_longitudinal_conservative` | longitudinal-only cautious car-following / interaction positive control |
| P3 | `idm_longitudinal_aggressive` | longitudinal-only assertive car-following / interaction positive control |
| P4 | `idm_longitudinal_comfort` | longitudinal-only comfort-oriented car-following / interaction positive control |

For smoke testing, it is acceptable to start with fewer planners:

- `expert_or_log_replay`
- `simple_planner`

For full Stage 7C validation, the goal is to include behaviorally distinct variants, preferably:

- `idm_longitudinal_conservative`
- `idm_longitudinal_aggressive`
- `idm_longitudinal_comfort`

If `IDMPlanner` or another configurable official-compatible planner is available, Stage 7C should define planner styles by planner parameters, not by fake trajectory rewriting. Exact parameter names depend on the installed nuPlan planner API, but the conceptual differences are:

- `idm_longitudinal_conservative`: lower target speed, larger headway, gentler acceleration, earlier braking.
- `idm_longitudinal_aggressive`: higher target speed, smaller headway, stronger acceleration, later braking.
- `idm_longitudinal_comfort`: moderate target speed, lower acceleration, smoother longitudinal response, lower jerk / comfort-oriented behavior.


#### Longitudinal-only vs full driving style

IDM conservative/comfort/aggressive profiles are **longitudinal-only rule-based profiles**. They are intended as controlled positive controls for longitudinal BDD validation, not as complete driving-style models. This interpretation follows the Stage 6C v2 behavior-event taxonomy in [`docs/stage6c_behavior_event_taxonomy_v2.md`](stage6c_behavior_event_taxonomy_v2.md), where driving style is task-conditioned across following, lead-brake response, queue approach, lane change, cut-in response, overtake opportunity/execution, hesitation, and yield-conflict tasks.

IDM profiles are suitable for:

- following;
- lead_brake_response;
- queue_approach;
- cutin_response longitudinal component;
- yield_conflict partial longitudinal component.

IDM profiles are not suitable for:

- lane_change sharpness;
- lane_change willingness;
- overtake execution;
- hesitation / abort-like maneuver;
- target-lane rear-gap pressure;
- full courtesy/yielding behavior.

Lane_change, overtake, hesitation, and yield_conflict require task-conditioned interpretation and cannot be fully validated by IDM longitudinal parameters alone. We first validate whether BDD can detect controlled longitudinal behavior drift using parameterized IDM profiles. Lateral and interaction style dimensions should be evaluated separately through lane-change / overtaking / yield-conflict task-conditioned BDD once a lane-change-capable planner or E2E policy is available.

Backward-compatible aliases `idm_conservative`, `idm_comfort`, and `idm_aggressive` may remain in code metadata, but documentation should prefer `idm_longitudinal_conservative`, `idm_longitudinal_comfort`, and `idm_longitudinal_aggressive` so they are not confused with complete driving-style profiles.

Custom planners are allowed only if existing nuPlan planners are unavailable or insufficient. If a custom planner is implemented later, it must satisfy all requirements below:

1. It must be compatible with nuPlan `AbstractPlanner` or the installed equivalent planner interface.
2. It must run inside the official nuPlan simulation framework.
3. It must output trajectories through the nuPlan simulation loop.
4. It must not bypass nuPlan simulation.
5. It must not generate offline pseudo trajectories.
6. It must not use numpy interpolation to rewrite logged expert trajectories and call that simulation.

Allowed: writing a planner that runs inside official nuPlan simulation. Not allowed: writing our own offline simulator or rewriting logged ego trajectories. E2E model planner integration remains Stage 7F, not Stage 7C.

Starting with custom planners would make failure diagnosis ambiguous. If results are poor, we would not know whether the problem is BDD, nuPlan simulation connectivity, trajectory export, a bad custom planner implementation, or planner behavior differences that are too weak. The correct order is therefore:

1. Run official nuPlan simulation with existing planner(s).
2. Export simulated ego trajectories.
3. Verify simulation output is non-empty and finite.
4. Add parameterized planner variants.
5. Only then add minimal custom planner wrappers if needed.
6. Keep E2E planner integration in Stage 7F.

Stage 7C smoke PASS requires:

- official nuPlan simulation API or official nuPlan CLI is used;
- `pseudo_rollout` is false;
- at least one planner runs successfully;
- at least one scenario runs successfully;
- `simulated_ego_trajectory.csv` is non-empty;
- `simulated_ego_seq.npy` is non-empty;
- numeric outputs are finite;
- `simulation_report.md` says PASS.


### Stage 7C.1 Official nuPlan Simulation Smoke Result

Latest Stage 7C.1 smoke status:

- Stage 7C.1A — official simulation smoke: **PASS**.
- Stage 7C.1B — official msgpack trajectory export: **PASS**.
- Stage 7C.1C — exact log + actual nuPlan scenario token wrapper smoke: **PASS**.
- Stage 7C.1C — strict Stage7B scene_token == nuPlan scenario_token: **NOT REQUIRED / mismatch observed**.
- Stage 7C.2A — `simple_planner × 3 distinct logs`: **PASS**.
- Stage 7C.2B — `simple_planner × 5 distinct logs`: **PASS**.
- Stage 7C.2C-0 — native IDM default/conservative/comfort/aggressive smoke: **PASS**.
- Stage 7C.2C-1 — wrapper multi-planner rollout: **READY / TODO**.
- Stage 7D — BDD validation on planner-generated trajectories: **TODO**.

Recorded smoke metrics:

| item | value |
|---|---|
| Planner | `simple_planner` |
| Official nuPlan simulation command succeeded | `1` |
| `validation.pass` | `true` |
| `official_success_count` | `1` |
| Pseudo rollout | `false` |
| Official simulation log parsed | `simulation_log/**/*.msgpack.xz` |
| Parsed official artifact | `official_nuplan_runs/scenario_0/simple_planner/simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.22.00.38_veh-35_01008_01518/000e00790bc45da7/000e00790bc45da7.msgpack.xz` |
| Parsed trajectory rows | `149` |
| `smoke_pass` | `true` |
| `uses_official_nuplan_simulation` | `true` |
| `same_scenario_alignment_required` | `false` |
| `simulated_ego_seq.npy` shape | `[1, 1, 149, 8]` |
| `simulated_ego_seq_mask.npy` shape | `[1, 1, 149]` |
| `required_pose_valid_ratio` | `1.0` |
| x/y/yaw non-sentinel ratios | `1.0 / 1.0 / 1.0` |
| valid timestep count | `149` |
| msgpack simulation log files found / parsed | `1 / 1` |
| msgpack trajectory rows extracted | `149` |
| warnings | `[]` |

This smoke proves the official nuPlan simulation → `msgpack.xz` simulation log → trajectory parser → `[N, P, T, C]` tensor export path is working.

It does **not** yet prove full Stage 7C. Multi-planner/multi-scenario validation remains TODO.

Stage 7C.1C now separates same-log alignment from strict nuPlan token alignment. The exact-token wrapper smoke has **PASS** evidence from an official `simple_planner` run that parsed the official `.msgpack.xz` simulation artifact. New exact-filter local evidence shows that Stage 7B.4 `scene_token` should be preserved as source metadata, but it must not be assumed to equal the value accepted by nuPlan `scenario_filter.scenario_tokens`: target log `2021.05.12.22.00.38_veh-35_01008_01518` matched successfully, while Stage 7B.4 `scene_token=165060762e765a5a` differed from actual nuPlan scenario token `000e00790bc45da7`.

For nuPlan exact reruns, the verified key is:

```text
log_name + actual_nuPlan_scenario_token
```

For the validated smoke, use:

```text
log_name = 2021.05.12.22.00.38_veh-35_01008_01518
actual_nuPlan_scenario_token = 000e00790bc45da7
```

The resulting alignment conclusion is:

```text
same_log_alignment_passed: true
strict_stage7b_scene_token_match: false
exact_nuplan_token_rerun_supported: true
alignment_status: PASS_LOG_AND_NUPLAN_TOKEN_RERUN
```

Therefore log match plus an available actual nuPlan token is `PASS_LOG_AND_NUPLAN_TOKEN_RERUN`; strict token match is only `PASS_STRICT` when Stage 7B.4 `scene_token` also equals actual nuPlan scenario token. This does not overclaim full Stage 7C completion, because Stage 7C.2 multi-planner/multi-scenario rollout remains TODO.

**Environment / interaction limitation:** the behavior of other traffic agents depends on the selected nuPlan simulation configuration. If the current simulation uses log-replay or non-reactive observations, it must be documented as a limitation. If reactive agents / IDM agents are enabled later, that configuration must be documented separately. Do not overclaim interaction realism unless the simulation configuration actually supports it.

**Expected output:**

```text
outputs/stage7c1_nuplan_simulation/
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
└── warnings.json
```

**Purpose:** Generate real nuPlan simulation behavior data for baseline / rule-based / traditional planners.

**Latest Stage 7 status after Stage 7C.2B and Stage 7C.2C-0:**

```text
Stage 7A: PASS
Stage 7B: PASS
Stage 7B.4: PASS
Stage 7C.1A official simulation smoke: PASS
Stage 7C.1B official msgpack trajectory export: PASS
Stage 7C.1C exact-token wrapper smoke: PASS
Stage 7C.2A simple_planner × 3 distinct logs: PASS
Stage 7C.2B simple_planner × 5 distinct logs: PASS
Stage 7C.2C IDM longitudinal-only multi-planner rollout: IN PROGRESS
Stage 7C.3 PDM / lateral-interaction planner extension: TODO
Stage 7D BDD validation: TODO
```

**Interpretation note:** IDM profiles are longitudinal-only rule-based positive controls. They should not be described as complete conservative / comfort / aggressive driving styles. They cover following, lead-brake response, queue approach, and partial longitudinal components of cut-in/yield conflicts. They do not cover lane-change willingness, lane-change sharpness, overtaking execution, hesitation, target-lane rear-gap pressure, or full courtesy/yield behavior.

**Research statement:** We first validate whether BDD can detect controlled longitudinal behavior drift using parameterized IDM profiles in official nuPlan simulation. Lateral and interaction style dimensions will be evaluated later through PDM or another lane-change-capable planner/E2E policy.


Stage 7C.2C-0 native IDM smoke validated all four official nuPlan IDM runs on the same exact scenario: `log_name=2021.05.12.22.00.38_veh-35_01008_01518`, `nuPlan scenario_token=000e00790bc45da7`, `planner_name=IDMPlanner`. The verified wrapper profiles for Stage 7C.2C-1 are `simple_planner`, `idm_longitudinal_conservative`, `idm_longitudinal_comfort`, and `idm_longitudinal_aggressive`; the IDM profiles use official `planner=idm_planner` with Hydra overrides on `planner.idm_planner.target_velocity`, `planner.idm_planner.min_gap_to_lead_agent`, `planner.idm_planner.headway_time`, `planner.idm_planner.accel_max`, and `planner.idm_planner.decel_max`. The wrapper command template now uses `{planner_hydra_overrides}` so Stage 7C.2C-1 can produce `[1, 4, T, 8]` when all four planners succeed.

Stage 7C.2B validated `simple_planner` on five distinct logs with official nuPlan simulation outputs, no pseudo rollout, same-log alignment required, and tensor shape `[5, 1, 149, 8]`. The selected Stage 7B sample IDs were `sample_000000`, `sample_000005`, `sample_000010`, `sample_000015`, and `sample_000019`; the selected log names were the five distinct mini logs in Stage 7B.4 metadata. Validation passed with `official_success_count=5`, `trajectory_rows=745`, `msgpack_simulation_log_files_found=5`, `msgpack_simulation_log_files_parsed=5`, `required_pose_valid_ratio=1.0`, and `pseudo_rollout=false`.

The five parsed official Stage 7C.2B artifacts were:

```text
scenario_0/simple_planner/simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.22.00.38_veh-35_01008_01518/000e00790bc45da7/000e00790bc45da7.msgpack.xz
scenario_1/simple_planner/simulation_log/SimplePlanner/stationary_in_traffic/2021.05.12.22.28.35_veh-35_00620_01164/001f3d5282985bbb/001f3d5282985bbb.msgpack.xz
scenario_2/simple_planner/simulation_log/SimplePlanner/traversing_traffic_light_intersection/2021.05.12.23.36.44_veh-35_00152_00504/00015fc2840d5313/00015fc2840d5313.msgpack.xz
scenario_3/simple_planner/simulation_log/SimplePlanner/traversing_intersection/2021.05.12.23.36.44_veh-35_01133_01535/0004544fe3715b27/0004544fe3715b27.msgpack.xz
scenario_4/simple_planner/simulation_log/SimplePlanner/high_magnitude_speed/2021.05.12.23.36.44_veh-35_02035_02387/0004bf5585cf5f26/0004bf5585cf5f26.msgpack.xz
```

**Remaining Stage 7C TODOs:**

- Stage 7C.2C-1: run wrapper smoke with `--planners simple_planner idm_longitudinal_conservative idm_longitudinal_comfort idm_longitudinal_aggressive` and require official nuPlan outputs only; expected tensor shape is `[1, 4, T, 8]` for one log when all four planners succeed, and `[5, 4, T, 8]` for five logs when every scenario-planner pair succeeds. Metadata must expose `planner_name`, `planner_id`, `planner_class`, `planner_type`, `policy_style`, `style_scope`, `nuplan_planner_config`, `hydra_overrides`, `supported_behavior_tasks`, `unsupported_behavior_tasks`, and `parameters_json`.
- Stage 7C.2C-2: run the wrapper rollout on five distinct logs × four planners after the one-log wrapper smoke passes.
- Stage 7C.3: add a PDM / lateral-interaction-capable planner extension later to cover lateral, lane-change, overtaking, hesitation, rear-gap pressure, and interaction/yielding style.
- Stage 7C.2D: produce planner behavior report card after multi-planner rollout data exist.

Stage 7D is still not started; BDD validation on planner-generated trajectories remains TODO until Stage 7C.2C outputs exist.

## 6. Stage 7D — BDD Validation on Planner Simulation Data

**Definition:** Stage 7D = convert Stage 7C simulated planner trajectories into behavior datasets, then validate behavior embedding / BDD on same-scenario and distribution-level planner differences.

**Sub-stages:**

- 7D.1 simulation trajectory → behavior dataset.
- 7D.2 planner behavior sanity check.
- 7D.3 paired same-scenario BDD.
- 7D.4 unpaired distribution BDD.
- 7D.5 ODD-conditioned / task-conditioned BDD.

**Expected results:**

- expert/replay vs expert/replay: very small.
- expert/replay vs comfort: small-medium.
- expert/replay vs conservative: medium.
- expert/replay vs aggressive: medium-large.
- conservative vs aggressive: largest.

**Interpretation:**

- Paired same-scenario BDD is the academic validation core.
- Unpaired distribution BDD corresponds to real-world company model-version comparison.
- ODD-conditioned BDD explains where the style drift occurs.

## 7. Stage 7E — Rule-Based / Traditional Planner Experiment Consolidation

**Definition:** Stage 7E = summarize Stage 7C/7D results before E2E integration.

**Purpose:**

- Prove the pipeline works on official nuPlan simulation data.
- Prove BDD detects planner-induced behavior drift.
- Establish baseline planner behavior report cards.
- Prepare the baseline for Stage 7F E2E comparison.

**Expected outputs:**

```text
outputs/stage7e_planner_summary/
├── planner_behavior_summary.csv
├── planner_bdd_matrix.csv
├── planner_report_card.md
├── planner_ablation_report.md
├── planner_robustness_report.md
└── figures/
```

Stage 7E must not claim final E2E conclusions. It is planner-only consolidation.

## 8. Stage 7F — E2E Model Simulation in nuPlan

**Definition:** Stage 7F = integrate an E2E driving model into the official nuPlan simulation environment as a planner, run closed-loop simulation on the same scenario set, and export E2E simulated behavior data.

**Important requirements:**

- Stage 7F must use the official nuPlan simulation environment.
- The E2E model should be wrapped as a nuPlan `AbstractPlanner`-compatible planner or equivalent.
- Do not evaluate E2E by offline trajectory rewriting.

**Sub-stages:**

- 7F.1 E2E planner interface / wrapper.
- 7F.2 E2E model input adapter.
- 7F.3 E2E closed-loop simulation smoke.
- 7F.4 E2E full mini simulation.
- 7F.5 E2E simulation output export.
- 7F.6 E2E behavior / BDD comparison.

**Possible E2E sources:**

- Lightweight trained neural planner.
- nuPlan-compatible ML planner baseline.
- Repo-local model wrapped as planner.
- Minimal neural planner baseline if no pretrained E2E model is available.

**Expected output:**

```text
outputs/stage7f_e2e_nuplan_simulation/
├── e2e_simulated_ego_trajectory.csv
├── e2e_simulated_ego_seq.npy
├── e2e_model_metadata.csv
├── e2e_scenario_index.csv
├── e2e_simulation_summary.csv
├── e2e_simulation_schema.json
├── e2e_simulation_report.md
└── warnings.json
```

**Purpose:** Move from planner-style validation to actual E2E autonomous driving system behavior evaluation.

Stage 7F is intentionally not implemented by this documentation update.

## 9. Stage 7G — Final Stage 7 Summary After E2E Simulation

**Definition:** Stage 7G = combine rule-based planner simulation, traditional planner simulation, and E2E model simulation results into the final Stage 7 experimental evidence package.

**Main questions:**

1. Can BDD distinguish different planners under the same scenarios?
2. Can BDD distinguish E2E model behavior from rule-based / traditional planners?
3. Is the E2E model closer to expert, conservative, aggressive, or comfort behavior?
4. Under which ODD/task conditions does E2E behavior drift most?
5. Do paired same-scenario BDD and unpaired distribution BDD tell a consistent story?
6. Does this support the thesis claim: behavior embedding + BDD can evaluate E2E autonomous driving style drift?

**Expected outputs:**

```text
outputs/stage7g_final_summary/
├── stage7_final_bdd_matrix.csv
├── stage7_policy_e2e_report_card.csv
├── stage7_odd_conditioned_summary.csv
├── stage7_ablation_summary.csv
├── stage7_robustness_summary.csv
├── stage7_final_report.md
└── figures/
```

Stage 7G is the final thesis-facing synthesis. It should be written only after Stage 7F E2E simulation data has been added.

## 10. Direction Guardrails

1. Stage 7C and 7F must use official nuPlan simulation.
2. Stage 7C should prefer existing nuPlan/devkit planners before custom planners.
3. Custom planners are allowed only as nuPlan simulation-compatible planners.
4. Custom planners must not bypass official nuPlan simulation.
5. Offline pseudo rollout and pseudo trajectory rewriting are not acceptable as Stage 7C/7F evidence.
6. Stage 7C validates the simulation/export pipeline and planner-induced behavior drift before E2E integration.
7. Same-scenario alignment must be preserved.
8. Paired BDD and unpaired BDD are both required.
9. ODD-conditioned BDD must use Stage 7B.4 context features.
10. E2E model integration belongs to Stage 7F, not Stage 7C.
11. Final thesis-facing conclusion belongs to Stage 7G, not Stage 7E.

## 11. Documentation Relationship

- `docs/stage7_empirical_same_scenario_style_validation.md` is retained as earlier empirical same-scenario notes and historical Stage 7 planning context.
- `docs/stage7a_nuplan_same_scenario_policy_validation.md` is retained as a Stage 7A / early nuPlan-readiness and same-scenario-policy validation note.
- This file is the current Stage 7 A–G roadmap and should be used as the primary reference for future Stage 7C–7G implementation issues.
