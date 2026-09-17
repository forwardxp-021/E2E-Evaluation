# S1R Exposure Policy v1 — PROPOSED OWNER AMENDMENT

Status: `PROPOSED_ONLY`; `OWNER_DECISION = PENDING`.

## Normative contamination definition

A historical nuPlan outcome is claim-relevant contamination when it actually influenced a method, statistical rule, operating point, cohort decision, or sample-size decision tested by the current claim. The counterfactual test is: **without observing that outcome, could the tested method, protocol, metric, inclusion decision, or stopping decision have been different?** If yes, record the influence and apply E3 or E4.

Prohibited paths include nuPlan-outcome selection of the RBR encoder/checkpoint or best seed; RBR-vs-H-result-driven changes to H features/readout; favorable-result selection of statistic, normalization, kernel, bandwidth, calibration, FPR operating point, eligibility, inclusion/exclusion, survivor set, sample size, or stopping rule.

Metadata inspection, ordinary historical rollout, unrelated Stage6/7 evaluation, old64/ego13 analysis, HLC/TSB engineering, controller-transfer diagnosis, F_match, safety, and applicability engineering are disclosed but do not automatically establish scientific unusability.

## Frozen taxonomy

| Class | Definition and examples | Allowed future roles | Prohibited roles | Confirmatory eligibility | Required disclosure |
|---|---|---|---|---|---|
| E0_METADATA_ONLY | Identity, schema, loader, route, infrastructure or non-scientific smoke exposure | B, V, C after ordinary eligibility | None solely from E0 | Eligible | Exact inspected metadata and reservation history |
| E1_UNRELATED_HISTORICAL_USE | Scientific outcome unrelated to the current TSB or Primary method decisions | B, V | Cannot be described as never historically used | V eligible; C only with explicit low-exposure rationale | Stage, outcome type and no-influence basis |
| E2_BENCHMARK_ENGINEERING | TSB/HLC/controller/mechanism/F_match/safety/applicability development | B and Claim-B V after freeze | Claim-A unseen-generalization evidence | V eligible for Claim B; normally not C | Component tuned, outcome observed and affected claim |
| E3_PRIMARY_METHOD_DEVELOPMENT | Outcome used to develop H, RBR readout, statistic, normalization, kernel/bandwidth, calibration, power or detector | B/development and predefined sensitivity | Confirmatory use for the component it tuned | Not eligible for the corresponding confirmatory claim without a predeclared independence design | Exact decision influenced and identities |
| E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION | Primary-like RBR-vs-H outcome used to change method, cohort, metric, operating point, sample size or stopping | Debugging and transparent supporting analysis only | V/C evidence for the same Primary claim | Ineligible | Full adaptation chronology and affected claim |
| E5_UNTOUCHED_CONFIRMATORY | Ledger-supported low/no claim-relevant outcome influence | V and preferred C | Cannot be asserted without provenance | Preferred C | Sources searched and residual uncertainty |
| UNKNOWN | Identity or decision-influence provenance is insufficient | B or deferred review | V/C by default | Ineligible until resolved | Missing evidence and resolution owner |

Multiple exposures retain the full list. The operative label is the highest claim-relevant risk: E4 > E3 > E2 > E1 > E0; E5 applies only where E1–E4 are absent for the stated claim. UNKNOWN fails closed.
