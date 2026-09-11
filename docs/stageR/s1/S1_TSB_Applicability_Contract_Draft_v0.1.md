# TSB Applicability — Draft v0.1

**DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW. Execution blocked pending the development calculation and Owner decisions below.** No inherited rule is changed by this document.

## Actual implementation and the 2.0 m/s floor

Candidate: `../r2/r2_bh_tsb_family_development_candidate_v1.0.json`, SHA256 `7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538`. Parameters and generator retain their bound SHA. Baseline −1.45 m/s² for 1.8 s; treatment −2.4 for 0.9 s, +1.4 for 1.3 s, −2.4 for 0.9 s. Start 1.1 s. No candidate tuning.

Code facts are **SUPPORTED**: `R2BControllerAwarePlannerV1._tsb_states` reintegrates a future 80-state horizon from current speed on each call, with a 0.2 m/s planned-speed clamp and route-aligned sampling. `_absolute_episode_clock` inherited from V2.1 uses index×0.1 for command phase; physical time is separately recorded. Before index 11 both arms use the complete baseline future trajectory, so preview already affects pre-divergence dynamics. The LQR fits poses to velocity/curvature, targets a 10-step/1 s lookahead, then TwoStageController propagates the returned command through its motion model. Calling controller.reset alone only clears current state; it is not a full experimental reset.

The inherited R1 floor came from its −1.0 m/s² ×10-sample baseline, not this R2 sequence. Nominal unclamped integrals are baseline speed loss 2.61 m/s, treatment first loss 2.16 and total loss 2.50. Thus ideal continuous baseline would need at least 3.61 m/s to remain ≥1 throughout; **3.61 is an illustrative integral, NOT a proposed applicability threshold**. Floating boundary sample counts, preview, repeated re-anchoring, initial acceleration, tracker stopping branch and motion lag prevent interpreting it as a realized guarantee. Initial speed 2.0 alone is **NOT ESTABLISHED** as sufficient for R2.

Frozen Option-A: median-of-3 speed; `np.gradient(speed, physical_time, edge_order=2)`; LOW_SPEED_ENDSTOP if median speed <1.0 for ≥5 consecutive samples. Brake threshold ≤−0.80 for ≥3 samples; release separation requires ≥3 samples at ≥−0.20; exact two phases, release fraction ≥0.15, second peak ratio ≥0.50. Duration gates remain sample counts, including on irregular timestamps. Do not reinterpret five samples as a measured elapsed 0.5 seconds.

## Proposed rule audit

| Proposed requirement | Classification | Evidence / decision |
|---|---|---|
| Retain inherited initial speed ≥2.0 as a necessary screening floor | SUPPORTED as inherited restriction; NOT ESTABLISHED as sufficient | Do not silently replace or waive it |
| Positive speed headroom for both arms throughout Primary80 and all rolling lookaheads | PLAUSIBLE | Need conservative bound using exact discrete commands, full precontext preview and controller/motion configuration |
| Exactly resolvable scenario; at least 81 time-controller iterations; 80 states/79 transitions without padding | SUPPORTED | R1 Primary80 and B0.2 |
| Complete replay observations, unambiguous native route, no extrapolation, enough forward distance for each rolling horizon | SUPPORTED | `_tsb_required_distance`, native route sampler; terminal short-segment curvature artifacts must not count as valid reference |
| Bind initial pose, speed, acceleration, steering, angular state, controller/motion parameters and history identically | SUPPORTED reset requirement | No validated new numeric initial-acceleration limit exists |
| Numeric speed/acceleration/curvature envelope for R2 | NOT ESTABLISHED | No HLC 3 m/s moving-floor transfer; no arbitrary curvature cutoff |
| Pre-outcome collision/drivable eligibility screen | PLAUSIBLE | Define from initial geometry, replay background and conservative reachable envelope; cannot screen using future realized safety outcomes |
| Official zero-at-fault-collision and drivable-area pass on both arms | SUPPORTED post-run gate | This is an outcome, never a roster selection variable |
| Numeric cadence exclusion limit for H/BDD | NOT ESTABLISHED | Preserve physical timestamps, report nominal-dt ego13 limitation; no new exclusion cutoff without measurement-error derivation |

**Recommendation:** retain 2.0 as legacy floor, add a pre-outcome conservative feasibility certificate, and leave execution closed until that certificate has a justified domain. The certificate must cover the fixed controller configuration and both arms, not predict future measured mechanism success. If no conservative bound is supportable, Owner must explicitly accept a development-supported restricted domain and its limitations before any fresh selection. Do not choose a new floor from future Q failures.

Required development-only calculation: on already exposed R2 DEV-CAL traces, measure initial/minimum speed, physical cadence, initial acceleration availability, route completeness and planned-clamp incidence; separately derive the exact discrete cumulative speed losses over every nominal replanning phase, using the frozen command function and full baseline preview. Audit frozen tracker stopping/longitudinal parameters and motion lag to bound realized deviations; evaluate joint speed×acceleration×curvature support rather than Cartesian marginal extrema. Missing initial/controller state or curvature metadata must be reported, never imputed as evidence of feasibility. The report includes the available trace audit; a proven controller-envelope bound remains unresolved.

Identity and source eligibility follow the qualification protocol. S1 neither queries fresh scenario records nor selects a roster. No threshold revision after Q or E exposure is permitted.
