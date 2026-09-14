# TSB Applicability — Draft v0.1

**S1_PROTOCOL_READY_FOR_OWNER_FREEZE — S1.1 Owner closure repair. S2_NOT_AUTHORIZED; RBR_TRAINING_NOT_AUTHORIZED.** Frozen candidate, mechanism, F_match and safety thresholds are unchanged. The prospective applicability screen below is an explicit post-development scope amendment; historical R1 provenance is preserved.

## Actual implementation and the 2.0 m/s floor

Candidate: `../r2/r2_bh_tsb_family_development_candidate_v1.0.json`, SHA256 `7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538`. Parameters and generator retain their bound SHA. Baseline −1.45 m/s² for 1.8 s; treatment −2.4 for 0.9 s, +1.4 for 1.3 s, −2.4 for 0.9 s. Start 1.1 s. No candidate tuning.

Code facts are **SUPPORTED**: `R2BControllerAwarePlannerV1._tsb_states` reintegrates a future 80-state horizon from current speed on each call, with a 0.2 m/s planned-speed clamp and route-aligned sampling. `_absolute_episode_clock` inherited from V2.1 uses index×0.1 for command phase; physical time is separately recorded. Before index 11 both arms use the complete baseline future trajectory, so preview already affects pre-divergence dynamics. The LQR fits poses to velocity/curvature, targets a 10-step/1 s lookahead, then TwoStageController propagates the returned command through its motion model. Calling controller.reset alone only clears current state; it is not a full experimental reset.

The inherited R1 ≥2.0 m/s floor came from its −1.0 m/s² ×10-sample baseline. Keep it only as historical provenance: it is NOT sufficient for current R2 TSB and is not the current selection rule.

**POST_DEVELOPMENT_PROSPECTIVE_SCOPE_AMENDMENT / NOMINAL_MEASURABILITY_SCREEN / NOT_A_CLOSED_LOOP_GUARANTEE.** For the frozen candidate, evaluate exact nominal commands at k×0.1 s (k=0…78), and let cumulative commanded loss at state j be `−sum(k=0…j−1, a_arm(k×0.1)×0.1)`. Across both arms and all 80 states, the greatest loss is baseline **2.61 m/s** (18 commands at −1.45); treatment maximum loss is 2.50 m/s (its first-stage loss is 2.16). Require `initial_speed − maximum_nominal_cumulative_loss >= 1.0 m/s`, hence **initial_speed >= 3.61 m/s**, inclusive.

This Owner-approved prospective screen avoids selecting scenarios nominally incompatible with the frozen LOW_SPEED_ENDSTOP measurement contract. It is not a closed-loop guarantee: preview, repeated re-anchoring, initial acceleration, tracker stopping behavior and motion lag still affect realized speed. A post-selection closed-loop LOW_SPEED_ENDSTOP is SCIENTIFIC_FAILURE, retained in the full Q denominator. No retrospective floor increase after Q is allowed. No simulator, controller propagation or roster query is needed for this nominal command calculation.

Frozen Option-A: median-of-3 speed; `np.gradient(speed, physical_time, edge_order=2)`; LOW_SPEED_ENDSTOP if median speed <1.0 for ≥5 consecutive samples. Brake threshold ≤−0.80 for ≥3 samples; release separation requires ≥3 samples at ≥−0.20; exact two phases, release fraction ≥0.15, second peak ratio ≥0.50. Duration gates remain sample counts, including on irregular timestamps. Do not reinterpret five samples as a measured elapsed 0.5 seconds.

## Proposed rule audit

| Proposed requirement | Classification | Evidence / decision |
|---|---|---|
| Initial speed ≥3.61 m/s nominal measurability screen | SUPPORTED by exact discrete nominal schedule; Owner-approved scope amendment | R1 ≥2.0 is historical provenance only; not a closed-loop guarantee |
| Guaranteed realized speed headroom through all rolling lookaheads | NOT ESTABLISHED | Not claimed or required as a proven certificate by this nominal screen; realized LOW_SPEED_ENDSTOP remains scientific failure |
| Exactly resolvable scenario; at least 81 time-controller iterations; 80 states/79 transitions without padding | SUPPORTED | R1 Primary80 and B0.2 |
| Complete replay observations, unambiguous native route, no extrapolation, enough forward distance for each rolling horizon | SUPPORTED | `_tsb_required_distance`, native route sampler; terminal short-segment curvature artifacts must not count as valid reference |
| Bind initial pose, speed, acceleration, steering, angular state, controller/motion parameters and history identically | SUPPORTED reset requirement | No validated new numeric initial-acceleration limit exists |
| Closed-loop acceleration/curvature envelope for R2 | NOT ESTABLISHED | No numeric acceleration/curvature limits invented; no HLC moving-floor transfer |
| Pre-outcome collision/drivable eligibility screen | PLAUSIBLE | Define from initial geometry, replay background and conservative reachable envelope; cannot screen using future realized safety outcomes |
| Official zero-at-fault-collision and drivable-area pass on both arms | SUPPORTED post-run gate | This is an outcome, never a roster selection variable |
| Numeric cadence exclusion limit for H/BDD | NOT ESTABLISHED | Preserve physical timestamps, report nominal-dt ego13 limitation; no new exclusion cutoff without measurement-error derivation |

**Current prospective rule:** use the inclusive 3.61 m/s nominal measurability screen plus all supported pre-outcome requirements above: 80 states / 79 transitions, finite strictly increasing physical timestamps, complete native reference and replay support, no extrapolation, required forward route support, identical full reset and source/log independence. Optional stronger controller-envelope/cadence investigations are not substituted for this approved screen and do not introduce new numeric gates. Neither pre-outcome screening nor nominal feasibility guarantees official post-run safety.

The existing development audit records the discrete command losses and historical speed/cadence observations. It is unchanged. The screen is a disclosed prospective amendment after development, not a retrospective claim that R1 or DEV-CAL used 3.61 m/s.

Identity and source eligibility follow the qualification protocol. S1 neither queries fresh scenario records nor selects a roster. No threshold revision after Q or E exposure is permitted.
