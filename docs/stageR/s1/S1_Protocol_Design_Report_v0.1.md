# S1 Protocol Design Report v0.1

**S1_PROTOCOL_READY_FOR_OWNER_FREEZE; S2_NOT_AUTHORIZED; RBR_TRAINING_NOT_AUTHORIZED.**

S1-A through I are covered by six normative Markdown contracts, the firewall and canonical schema, this report, one primary design manifest and one deterministic historical audit output. Q sample size is combined with Q protocol, avoiding a duplicate manifest. Only one new zero-run tool and one fixture test file were added. QUICK_REFERENCE contains the reproduction commands.

## S1.1 Scientific Owner closure repair

Final repair status: **S1_PROTOCOL_READY_FOR_OWNER_FREEZE**;
**S2_NOT_AUTHORIZED**; **RBR_TRAINING_NOT_AUTHORIZED**.
Scientific Owner has approved scope/claims, BDD Primary, H+RBR Secondary,
unpaired-preferred/paired-fallback, ΔBDD, α=.05, m=20, π=.50, δ*=.10,
FPR qualification, Q20/max40, all-pass/first-scientific-failure-stop,
no survivors, Q→D preregistration, U-only selection and the 30D H structure.
This repair records those approvals without reopening the approved SAP.

| Repair | Normative result |
|---|---|
| F0 / matching | F_match is the four existing TSB descriptors; F0_project is existing ego13, both project-standard predefined/routine representation and explicitly reported historical baseline. It is not a corporate KPI inventory. H stays the sole Primary handcrafted comparator; O stays mechanism positive control. Wording is Residual-to-F0_project. |
| Applicability | initial_speed ≥3.61 m/s, derived from 1.0 + maximum exact nominal cumulative loss 2.61. POST_DEVELOPMENT_PROSPECTIVE_SCOPE_AMENDMENT / NOMINAL_MEASURABILITY_SCREEN / NOT_A_CLOSED_LOOP_GUARANTEE. R1 ≥2.0 is historical only. Post-selection LOW_SPEED_ENDSTOP remains scientific failure; no post-Q increase. |
| Training boundary | shared z64 receives backpropagation from a U-only Human Semantic Head and generic representation/temporal heads; no TSB/O/Q/D/E supervision or handcrafted-distance/ego13-geometry alignment. BDD uses frozen z64 directly. Post-hoc probes stay Secondary. Exact layers, weights and SSL objective remain for future pre-training protocol. Stage6T provides auxiliary-head precedent. |
| Operating point | π=.50 is a prospectively frozen SYNTHETIC MODERATE-DRIFT BENCHMARK prevalence, not real-fleet prevalence, production-change share or ODD estimate. No operating-point/dose search. |
| Q20 / Q12 | Q_TARGET=20 independent logs/pairs, MAX_RUNS=40. Q12 is an Owner fallback only after metadata-only census and before any Q rollout/outcome exposure. Q20 cannot become Q12 after starting; 12/12 early success is not qualification success. |

All supported pre-outcome technical/reference/reset/independence requirements
remain. No numeric acceleration or curvature limit is invented. Final protocol
freeze and separate S2 execution authorization are still required; metadata-only
eligible census, runtime bindings and production integration remain future
execution prerequisites, not reasons to reopen approved scientific choices.
E sample-size/power/coverage and no-extension rules remain unchanged. No roster
is selected, no new data role is instantiated and no training is authorized.

## Preserved S1 deterministic development evidence

Read only the 16 trace paths in the frozen TSB round-0 results, covering 8 already exposed independent logs. No scientific identities were newly queried. Reproduction: `python tools/s1_protocol_schema.py --development-audit` in the nuPlan environment. `S1_Development_Evidence_Audit_v0.1.json` contains exact input hashes and aggregate/per-historical-run values.

- Initial speed 4.123176–13.071771 m/s; minimum observed realized speed 1.242186 m/s. This evidence does not establish applicability near 2.0 m/s.
- Physical sample spacing .099918–.100088 s. Preserve actual timestamps. No new numerical cadence exclusion threshold has been introduced.
- Mechanism remeasurement: baseline one phase / treatment two phases, all 16 status OK. H produces finite 30 columns for all 16 traces.
- Signed treatment−baseline F deltas (mean, sample SD): mean speed **+.650820, .001779 m/s**; end−start speed **+.537865, .003278 m/s**; path length **+5.152602, .014210 m**; mean absolute acceleration **+.102057, .001088 m/s²**. These are development descriptive values, not independent scientific effect confirmation. All are positive across the 8 pairs despite frozen F_match pass. Thus matching does not eliminate existing low-order monitoring signals.
- Exact nominal command arithmetic gives baseline loss 2.61 m/s over 18 active samples; treatment net loss 2.50 over 31 nonzero samples. This is fixed-profile arithmetic, not a rollout or controller surrogate validation. S1.1 now uses the resulting 3.61 m/s nominal screen prospectively, without calling it a safe closed-loop speed threshold.
- Missing from the production current_ego trace payload: full initial acceleration/steering/controller internal state. Deriving a conservative realized bound requires bound runtime/config and additional already-exposed telemetry evidence; no numeric acceleration or curvature restriction is invented.

The narrow F-delta SDs from eight calibrated logs must not be used as the variance of future RBR detection gain. Q=20 provides useful development scale, not independent AA tail evaluation or unpaired release replication.

## Historical evidence and reconciliation

| Source | S1 implication |
|---|---|
| R0 v1.0 full protocol, final report/status v1.1, D1/D3 records | Information readability differs from geometry/detection; D3 INCONCLUSIVE; development calipers are not formal physical equivalence |
| Stage6 unpaired protocol §§25–28 and Stage6P reliability report | Strong paired signal may not give reliable unpaired alarms; do not compare raw MMD²; reusable release trials do not add independent logs |
| Stage7 M6.5/M6.6 protocols and locked results | Same-scenario attribution, explicit log grouping, task multiplicity and data-quality limits |
| Stage7L-C/E prospective protocol and final report | B3407 Primary failed despite mechanism success; ego13's strong secondary result does not replace Primary |
| R1 mechanism/applicability/F_match/safety and timestamp-aware production code | Exact frozen gates and inherited 2.0 origin; preserve nominal-count duration semantics |
| R2-A replanning transfer and R2-B/TSB candidate/results | Future trajectory preview, re-anchoring and LQR fitting matter; 8/8 is development only |
| R2-BK scope/closure/capacity | TSB-only amendment; HLC closed; eligibility capacity not materialized |
| B0/B0.1/B0.2 and B1/B1.1 | Atomic attempt budget, full reset, 80/79 distinction, passive return identity; real serializer/analyzer field agreement, no fallback key guessing |

No direct contradiction to v2.2's current scientific constraints was found. Its deliberate open questions remain open. Repository evidence contradicts stronger interpretations sometimes carried in older handover text: D3 “NOT_SUPPORTED” as a total formal state, HLC still awaiting a possible final rescue, TSB “clean residual” after F_match, and old remote HEAD references. Those historical files are preserved, not edited. v2.2's HLC ideal-tracking/overshoot wording is the accepted latest review synthesis, also present in v2.1; this S1 task did not rerun the HLC recovery analyzer or independently establish impossibility.

An important implementation qualification: JSON's “0.3 s / 0.5 s” language maps to frozen 3-/5-sample gates; the evaluator uses physical derivatives but not elapsed-duration comparisons. S1 makes that explicit without revising historical rules. HLC endpoint/engineering bounds cannot silently be transferred to TSB (production TSB evaluator returns those HLC fields as inapplicable).

## Canonical design and test scope

Future flow: approved single manifest → one TSB executor/full official lifecycle → one existing passive recorder → strict canonical trace serializer → one analyzer entry delegating to the production dispatcher/evaluator and official parquet canonicalizer. No stacked new launcher wrappers. S1 has **no simulator executor implementation or callable run entry**; static `validate_plan` checks only hypothetical synthetic plans.

Canonical current_ego fields are taken directly from the inherited production `_payload`, not reconstructed with alternative key names. Frozen trace extras are explicitly projected onto the three canonical keys; scientific fields are never renamed. Tests use synthetic scalar/state doubles at the simulator boundary, but no mocked scientific analyzer, mechanism return or safety parser. Real parquet fixtures reach the actual canonicalizer. Real passive recorder serializes its returned-command rows. The real non-stopping controller dynamics/reset lifecycle still require separate zero-run production integration evidence before Q; doubles alone cannot prove it.

Coverage: 80 states/79 transitions; physical irregularity; missing/wrong keys/nesting; wrong units; nonfinite/time errors; one/two phase and LOW_SPEED_ENDSTOP; rounded F_match boundaries; safety failure; duplicate identity; fresh output root/budget bounds; passive return-object identity and persistence failure; deserialization-to-production-analyzer parity. No HLC recovery execution occurred.

Final numeric thresholds are calibrated on separately reserved D logs before E is constructed or accessed; E contains only independent FPR evaluation and AB evaluation. Calibration costs are included explicitly in the sample-size envelopes.

Validation commands and outcomes are listed in QUICK_REFERENCE and the manifest. Final targeted suite: 19 tests passed (16 S1 tests including parameterized cases plus 3 existing passive-recorder regressions). Python compilation and forbidden temporary-dependency scan passed. Third-party matplotlib/pyparsing deprecation warnings only. Deterministic DEV-CAL audit completed; no fresh-data smoke was run because S1 explicitly forbids simulation/new identities, not because historical data is missing.

## Provenance and preservation

Work began at local HEAD `20b3f594a20584f00c72e0951644016c381ebe3e`, committed tree `e77bfc3cc1fece5a50081747a513584b18262a3b`, branch `20260825_stageR_new`. Last successful live remote observation: `798bd72d33e78f7e434cdc57ea4c6dfcdd924184`. Local tracking ref was stale (status reported ahead 61); no fetch/merge/reset was performed. A later HTTP2 recheck failed and a bounded HTTP1.1 recheck timed out; the remote SHA is the earlier live observation, not a freshness guarantee. Final commit SHA, if any, is reported in the task response rather than self-bound into this report.

Pre-existing dirty state: protected CSV modified, many untracked historical output directories/files. The single manifest records a baseline status digest/count and protected CSV hash. The protected CSV is excluded from all staging, edits and cleanup. No historical output path is written. S1 artifacts and the narrow QUICK_REFERENCE addition are the only intentional changes.

**Actual actions: simulation=0; runner.run=0; new scientific identities exposed=0; RBR training=0; E construction/unblinding=0.**


## S1.1 provenance and validation

Requested and actual repair base: `d7b2f711c0a3ba8081dbc47a0d6e04d8b7c4ba29`.
The prior local S1 commit `4e8b7eb48716d1d24eb07911256ff0ab7d7dfa3a`
and this remote base have the identical tree
`dcd23565b75373e9943b960210d4b10aacf15226`. Prior local history was preserved
on `s1-pre-s11-local-history-4e8b7eb`; aligning the active branch to the specified
base changed no index/worktree bytes. The S1 provenance above remains historical.

S1.1 changes only these normative documents, firewall/manifest metadata and the
narrow QUICK_REFERENCE note. Canonical schema, H implementation, tests, candidate,
all frozen evaluators and historical outputs remain byte-identical. Unit and
static validation results are recorded in the updated manifest; nominal schedule
validation reads frozen parameters/code only, never Q/E or simulator state.

Actual S1.1: simulation=0; runner.run=0; new scientific identities exposed=0;
RBR training=0; E access=0. Final commit and remote SHA are returned externally
so the manifest has no self-referential commit hash.
