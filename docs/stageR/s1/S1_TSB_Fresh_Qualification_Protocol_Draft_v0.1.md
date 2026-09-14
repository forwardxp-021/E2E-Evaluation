# TSB Fresh Qualification and Q Size — Draft v0.1

**Owner review draft, not authorization.** The S1.1 nominal measurability screen, eligible capacity, candidate/runtime bindings, canonical production integration and Owner signoff are prerequisites. Sole primary manifest: `S1_Protocol_Design_Manifest_v0.1.json`; its only current run budget is zero.

## Population and acquisition

Use the historically frozen source universe (`r1_fresh_smoke_source_universe_v0.1.json`) without expansion. Before any future roster selection, apply the Owner-approved S1.1 initial_speed ≥3.61 m/s nominal measurability screen and obtain authorization for a one-time metadata-only complete eligibility census. Current census is NOT_MATERIALIZED; 1,425 post-governance logs is only a structural upper bound. S1 does not materialize fresh identities.

Exclude union of historical outcome-exposed tokens AND logs, permanent engineering reservations and source groups allocated to U/D/E; preserve reserved-but-unexposed roles distinctly. Include Stage6/7/7L, R0/R1/R2 and aborted/partially exposed attempts, not just successful runs. Missing log provenance fails closed. One token per independent log, globally unique tokens. Same driving-session dependence, if identifiable, elevates the grouping unit to session. Census and exclusion ledger hashes must precede selection.

Future selection algorithm: fixed salt `S1_TSB_Q_v0.1`; sort by SHA256(salt|log_id), then within log SHA256(salt|scenario_token), with lexical ties. Take first eligible token per selected log. This is a rule only, no actual list or rank calculation in S1. Record the complete denominator and pre-outcome reasons. Freeze all Q members, order, runtime/config/candidate/recorder/analyzer hashes, output root, budget, Q→D transition and stage authorization before run 1. No reserves or replacements.

## Execution and reset

Sequential baseline→treatment for each pair, roster order fixed. Construct a fresh scenario, planner, controller, tracker, motion model, history buffer, callbacks, recorder and random state per arm. Bind identical initial full EgoState, replay observations, map/route, seed, clock, scenario start and controller configuration. Clear filesystem callback state and lazy caches that can alter behavior; read-only immutable map caches may be shared only after demonstrated noninterference. Equality of precontext hashes is necessary, not sufficient to prove full reset.

One future executor, frozen official lifecycle, one passive recorder, one analyzer. Count every actual simulator entry, including exceptions after entry. Atomically claim one budget unit before sole entry; no refund for failure. S1 includes only static budget validation, not a runnable executor. Before future authorization, bind lifecycle callback completion, exactly one successful runner report per arm, actual official metric artifacts, 80 planner/trace rows and 79 controller returns. Never manufacture completion booleans.

## Gates and adjudication

Every pair must satisfy applicability, context identity, technical completeness, measurement validity, baseline exactly one phase, treatment exactly two phases, release ≥0.15, second peak ratio ≥0.50, all four frozen TSB F_match calipers, official safety for BOTH arms. Frozen production evaluator/dispatcher is authoritative. F_match is development balance, not proof that all low-order nuisance is eliminated.

- Technical completeness: exact schema, finite values, physical time, artifact provenance, official callbacks/reports and cardinality. Corrupt/missing files, schema mismatch or runner failure are infrastructure failures.
- LOW_SPEED_ENDSTOP on an otherwise complete trace, weak/merged/absent mechanism, F_match failure or unsafe outcome are scientific failures; do not rename them technical to gain a rerun.
- A realized measurement failure despite passing the nominal measurability screen remains in the Q denominator and prevents qualification; it is not retrospective ineligibility.
- On first scientific failure: irrevocable Q FAIL, stop further runs to bound cost; report every frozen member as pass/fail/not-run-after-stop. No completed-success subset becomes a qualified cohort.
- On infrastructure failure: stop all; Q INCOMPLETE/INFRASTRUCTURE_STOP, not scientific negative. Preserve every byte and attempted-run ledger. No automated retry.
- Technical recovery: offline reparsing of existing complete artifacts only, with cause, exact diff, pre-recovery hash and versioned recovered state, separately Owner-approved. No threshold changes, simulation repetition, missing-state synthesis or conversion of original historical stop to PASS. If complete artifacts do not exist, close incomplete. No scientific reruns.

## Whole-roster choice

Owner-approved rule: **all fixed pairs jointly pass**, with early futility stopping. It is a benchmark construction requirement, not a test of >90% generator-population reliability. It supports an interpretable mechanism-qualified asset and matches the current BK draft; an all-pass result does not justify universal claims. At true pair pass probability .95, chance all 20 pass is only .95^20 ≈.358, so this rule is stringent and may close a useful generator.

A fixed ≥18/20 mechanism rule with all 20 safety/technical valid could be scientifically defensible for an intention-to-evaluate software-treatment estimand. It would retain all pairs in BDD, including failures, and call it a partially realized intervention. It is **not recommended here** because it changes the “all pairs mechanism-confirmed” claim and requires a new cohort-level effect criterion. This historical design alternative is not active: Scientific Owner selected whole-roster all-pass in S1.1. Survivor selection is not an option.

## Sample-size recommendation and cost

| Owner option | Independent logs/pairs | Maximum simulator entries | Interpretation |
|---|---:|---:|---|
| Pre-execution Owner fallback only | 12 | 24 | May be selected only after metadata-only census and before any Q rollout/outcome exposure |
| Approved target | 20 | 40 | Q_TARGET=20 independent logs/pairs; MAX_RUNS=40 |

No default 29/29. DEV-CAL 8/8 is selected development evidence, not a binomial sample from the fresh population. Under an ideal iid all-success illustration only, one-sided 95% lower bound p is .05^(1/n): 0.779/0.861 for 12/20. These are illustrative, not the objective or a warranted claim for hash-selected/source-restricted Q. For approximately normal independent scalar margins, relative SE of SD ≈1/sqrt(2(n−1)): 21.3%/16.2%. Non-normal morphology margins and narrow DEV support can be worse.

Twenty pairs is a resource recommendation, not powered BDD confirmation. Q cannot estimate unseen RBR gain, prove nuisance elimination, provide independent AA-tail calibration or supply final E. Stop rules above cap costs; do not top up to a desired success count. Q_TARGET=20 and MAX_RUNS=40 are approved. Q_MINIMUM_OPTION=12 is ONLY a pre-execution Owner fallback after the metadata-only eligibility census and before any Q rollout or scientific outcome exposure. Once Q20 execution starts, it cannot be redefined as Q12; 12/12 early success is not qualification success. There is no active Q32 option. No extra smoke budget. All future E/training budgets remain zero until separately approved from the BDD SAP capacity calculation.

Q→D is preregistered in `S1_Data_Firewall_Draft_v0.1.json`: full Q closes first; passing Q becomes D for named measurement-development uses; failed/incomplete Q is diagnosis-only. Encoder choice remains U-only.
