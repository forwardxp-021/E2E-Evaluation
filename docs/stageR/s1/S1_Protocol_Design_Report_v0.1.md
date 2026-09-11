# S1 Protocol Design Report v0.1

**REVIEW PACKAGE COMPLETE; SCIENTIFIC FREEZE NOT CLAIMED; FRESH SIMULATION NOT AUTHORIZED.**

S1-A through I are covered by six normative Markdown contracts, the firewall and canonical schema, this report, one primary design manifest and one deterministic historical audit output. Q sample size is combined with Q protocol, avoiding a duplicate manifest. Only one new zero-run tool and one fixture test file were added. QUICK_REFERENCE contains the reproduction commands.

## Recommendations and Owner decisions

| Item | Recommendation | Unresolved before Q authorization |
|---|---|---|
| Scope | Level 1 framework; conditional TSB-specific Level 2 BDD utility | Approve scope and formal interpretation corrections |
| Applicability | Keep inherited ≥2.0 legacy floor; require conservative candidate/controller feasibility certificate | R2 speed/acceleration/curvature/reference and cadence support are not fully established; no replacement numeric floor invented |
| Q | 20 independent logs/pairs, cap 40 entries; minimum 12/24; bounded larger option 32/64 | Choose one budget; authorize metadata-only eligible census separately; no roster exists |
| Whole roster | Every frozen pair jointly passes; stop irrevocably on first scientific failure | Accept stringency, no survivor qualification, no rerun/replacement |
| Primary | RBR−H detection probability at α=.05, m=20, π=.50 | Approve operating point, useful gain .10 and FPR evaluation tolerance .025 |
| Design | Unpaired release emulation if capacity and power pass; pre-E paired fallback | Census is missing; 1,425 logs is an upper bound, not eligibility evidence |
| H | Fixed development-informed 30-column implementation | Approve exact H and the proposed F0 interpretation as current routine set |
| E size | D-only full-pipeline variance/power/coverage rule, smallest valid cost envelope | No final N justified; validate source-log interval method and capacity before E; never look at E then extend |
| Firewall | U-only encoder selection; Q→D registered before Q | Sign transition/role rules and complete exclusion ledger provenance |
| Technical simplification | One future executor, existing passive recorder, canonical serializer and existing scientific dispatcher | Before any Q execution, bind full production lifecycle/reset/config hashes and schema-faithful integration; schema fixtures are not live-runtime qualification |

Current Owner outcome remains **NOT_AUTHORIZED pending review and closure of blockers**. This package does not assert that a single signature can cure missing applicability/capacity evidence. S1 ends here; it does not authorize S2, future E or RBR.

## Deterministic development evidence

Read only the 16 trace paths in the frozen TSB round-0 results, covering 8 already exposed independent logs. No scientific identities were newly queried. Reproduction: `python tools/s1_protocol_schema.py --development-audit` in the nuPlan environment. `S1_Development_Evidence_Audit_v0.1.json` contains exact input hashes and aggregate/per-historical-run values.

- Initial speed 4.123176–13.071771 m/s; minimum observed realized speed 1.242186 m/s. This evidence does not establish applicability near 2.0 m/s.
- Physical sample spacing .099918–.100088 s. Preserve actual timestamps. No new numerical cadence exclusion threshold has been introduced.
- Mechanism remeasurement: baseline one phase / treatment two phases, all 16 status OK. H produces finite 30 columns for all 16 traces.
- Signed treatment−baseline F deltas (mean, sample SD): mean speed **+.650820, .001779 m/s**; end−start speed **+.537865, .003278 m/s**; path length **+5.152602, .014210 m**; mean absolute acceleration **+.102057, .001088 m/s²**. These are development descriptive values, not independent scientific effect confirmation. All are positive across the 8 pairs despite frozen F_match pass. Thus matching does not eliminate existing low-order monitoring signals.
- Exact nominal command arithmetic gives baseline loss 2.61 m/s over 18 active samples; treatment net loss 2.50 over 31 nonzero samples. This is fixed-profile arithmetic, not a new rollout, controller surrogate validation or a safe speed threshold.
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
