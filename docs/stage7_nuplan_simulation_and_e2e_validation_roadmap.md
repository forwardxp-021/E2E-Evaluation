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
| 7C | official simulation with rule/traditional planners | nuPlan simulation | simulated planner trajectories | TODO |
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
| P2 | `idm_conservative` | cautious longitudinal behavior variant |
| P3 | `idm_aggressive` | assertive longitudinal behavior variant |
| P4 | `idm_comfort` | comfort-oriented behavior variant |

For smoke testing, it is acceptable to start with fewer planners:

- `expert_or_log_replay`
- `simple_planner`

For full Stage 7C validation, the goal is to include behaviorally distinct variants, preferably:

- `idm_conservative`
- `idm_aggressive`
- `idm_comfort`

If `IDMPlanner` or another configurable official-compatible planner is available, Stage 7C should define planner styles by planner parameters, not by fake trajectory rewriting. Exact parameter names depend on the installed nuPlan planner API, but the conceptual differences are:

- `idm_conservative`: lower target speed, larger headway, gentler acceleration, earlier braking.
- `idm_aggressive`: higher target speed, smaller headway, stronger acceleration, later braking.
- `idm_comfort`: moderate target speed, lower acceleration, smoother longitudinal response, lower jerk / comfort-oriented behavior.

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
- Stage 7C.1C — same-scenario alignment: **TODO**.
- Stage 7C.2 — multi-planner/multi-scenario rollout: **TODO**.
- Stage 7D — BDD validation on planner-generated trajectories: **TODO**.

Recorded smoke metrics:

| item | value |
|---|---|
| Planner | `simple_planner` |
| Official nuPlan simulation command succeeded | `1` |
| Pseudo rollout | `false` |
| Official simulation log parsed | `simulation_log/**/*.msgpack.xz` |
| Parsed official artifact | `official_nuplan_runs/scenario_0/simple_planner/simulation_log/SimplePlanner/near_multiple_vehicles/2021.06.08.14.35.24_veh-26_02555_03004/1f151e15c9cf5c81/1f151e15c9cf5c81.msgpack.xz` |
| Parsed trajectory rows | `150` |
| `simulated_ego_seq.npy` shape | `[1, 1, 150, 8]` |
| `simulated_ego_seq_mask.npy` shape | `[1, 1, 150]` |
| `required_pose_valid_ratio` | `1.0` |
| x/y/yaw non-sentinel ratios | `1.0 / 1.0 / 1.0` |
| warnings | `[]` |

This smoke proves the official nuPlan simulation → `msgpack.xz` simulation log → trajectory parser → `[N, P, T, C]` tensor export path is working.

It does **not** yet prove full Stage 7C. Same-scenario alignment with Stage 7B.4 and multi-planner/multi-scenario validation remain TODO.

Stage 7C.1C should align Stage 7B.4 metadata with the actual simulated nuPlan scenario. The smoke used `scenario_filter=one_of_each_scenario_type` and `scenario_filter.limit_total_scenarios=1`, so it proves the simulation/export pipeline but does not yet prove that the simulated scenario is exactly the same as the Stage 7B.4 metadata row.

**Environment / interaction limitation:** the behavior of other traffic agents depends on the selected nuPlan simulation configuration. If the current simulation uses log-replay or non-reactive observations, it must be documented as a limitation. If reactive agents / IDM agents are enabled later, that configuration must be documented separately. Do not overclaim interaction realism unless the simulation configuration actually supports it.

**Expected output:**

```text
outputs/stage7c1_nuplan_simulation/
├── simulated_ego_trajectory.csv
├── simulated_ego_seq.npy
├── simulated_planner_metadata.csv
├── scenario_planner_index.csv
├── simulation_summary.csv
├── simulation_schema.json
├── simulation_report.md
└── warnings.json
```

**Purpose:** Generate real nuPlan simulation behavior data for baseline / rule-based / traditional planners.

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
