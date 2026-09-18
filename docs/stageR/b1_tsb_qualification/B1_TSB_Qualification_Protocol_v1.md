# B1 Frozen TSB Closed-Loop Benchmark Qualification Protocol v1

Status: `FROZEN_BEFORE_B1_SCIENTIFIC_ROLLOUT`.

## Scope

B1 asks whether the frozen TSB candidate reliably realizes its predefined one-phase baseline and two-stage treatment in official closed-loop nuPlan execution. B1 is role `B` benchmark qualification. It does not train or evaluate RBR, inspect H/BDD, or perform a Primary comparison.

The target is 20 independent SESSION pairs and 40 arms. Selection uses only `STATIC_ELIGIBLE` identities and ranks within exposure class by `SHA256(B1_FROZEN_TSB_QUALIFICATION_V1|session_id|scenario_token)`. Before roster freeze, the selector scans E2 first and uses E1 only if the context-compatible E2 capacity is below 20; E5 is never used. It takes the first 20 identities for which the frozen official B1 context adapter can produce its fail-closed pre-treatment binding. Every rejected identity and reason is written to `B1_TSB_Qualification_Selection_Audit_v1.csv`. It never tries a second scenario within a rejected SESSION. This is a metadata/map/context gate completed before any rollout; it is not outcome-driven replacement. There are no reserves, retries, or survivor selection after roster freeze. Arm order is deterministically interleaved by SHA256 parity of `pair_id`. Each arm runs in a fresh operating-system process through the frozen B1 successor of the S2R production executor.

## Frozen intervention

The authoritative configuration is `docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json`:

- Baseline: `-1.45 m/s² × 1.8 s`.
- Treatment: `-2.4 m/s² × 0.9 s`, `+1.4 m/s² × 1.3 s`, `-2.4 m/s² × 0.9 s`.
- Intervention begins at 1.1 s.
- Nominal applicability screen: execution initial speed ≥3.61 m/s.
- LOW_SPEED_ENDSTOP floor: 1.0 m/s under the frozen analyzer.

No intervention, measurement, matching, safety, smoothing, window, or analyzer value may change after this freeze.

## Pair gates

Every pair remains in the denominator and must satisfy:

1. both arms technically complete through official lifecycle and artifacts;
2. identical `PRECONTEXT_ID` and `ROUTE_ID`;
3. valid realized-current-ego measurements over iterations 0–79;
4. baseline exactly one braking phase;
5. treatment exactly two braking phases, interstage release fraction ≥0.15, and second peak ratio ≥0.50;
6. all four frozen F_match calipers pass: mean speed, end-minus-start speed, path length, and mean absolute acceleration;
7. official safety passes in both arms;
8. no LOW_SPEED_ENDSTOP.

Allowed pair statuses are `NOT_RUN`, `TECHNICAL_INCOMPLETE`, `MEASUREMENT_INVALID`, `SCIENTIFIC_FAIL`, and `PASS`. `PAIR_INVALID_PRECONTEXT` is a technical/protocol invalidity reason.

## Estimands and success rule

Report technical completion, measurement validity, baseline mechanism, treatment mechanism, joint mechanism, F_match, official safety, and joint qualification as n/N over 20 independent SESSION pairs. Report a descriptive two-sided 95% Wilson interval for joint qualification. This interval does not create or modify a success threshold.

The preregistered success rule is `ALL_20_PAIRS_PASS_FROZEN_JOINT_QUALIFICATION`. Any scientific or measurement failure makes the B1 verdict FAIL, while the complete frozen roster is still executed under the stop-policy addendum. Any infrastructure failure stops further execution and makes the verdict `INCOMPLETE_INFRASTRUCTURE`.

`F_match PASS` does not establish elimination of all low-order nuisance. Regardless of B1 outcome:

```text
LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED
TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED
RBR_TRAINING = NOT_AUTHORIZED
PRIMARY_EVALUATION = NOT_AUTHORIZED
V_EXECUTION = NOT_AUTHORIZED
C_EXECUTION = NOT_AUTHORIZED
```
