# Handcrafted Comparison — Draft v0.1

**DRAFT_FOR_SCIENTIFIC_OWNER_REVIEW.** One implemented H proposal, fixed before E and explicitly **DEVELOPMENT_INFORMED_HANDCRAFTED_CHALLENGER**. No mechanism-label-specific phase count enters H. No post-E feature expansion or performance-driven weakening.

## Exact repository audit and roles

**F0 recommendation:** current TSB monitoring subset of the repository's frozen F summaries, in order mean_speed, end_minus_start_speed, path_length, mean_abs_accel. Owner must confirm that this is the intended routine monitoring set; repository code establishes a matching set, not organizational routine-monitoring practice. Do not claim that a historical “F0” identifier already existed. F0 BDD is secondary; H is the sole Primary comparator.

Frozen production F implementation: `r1_closed_loop_benchmark_v2_1.trajectory_descriptors_timestamp_aware`, used by `R1OfficialTechnicalSmokeEvaluatorV2_1`. Arithmetic mean speed; end−start speed; sum of Euclidean rear-axle steps; mean absolute `np.gradient(raw_speed, physical_time, edge_order=2)`. Descriptors rounded to six decimals. Fifth output heading_change_abs_total sums absolute differences of unwrapped stored heading; it is not a TSB Primary caliper. Calipers in `r1_prospective_generator_contract_v2`: 0.708203939 m/s, 0.978755681 m/s, 5.38423459 m, 0.11777666 m/s² respectively. Absolute deltas rounded to six decimals then compared with tolerance 1e−12. Historical alternative `trajectory_descriptors` functions use different derivatives; do not substitute them.

**ego13:** `stage6l_prepare_context_representation_ablation.ego_kinematic_features`. Input [N,T,8], mask [N,T]; valid frames compacted; speed column 5, heading column 4, position columns 0:2. Fixed dt=.1; no smoothing. Order:

| # | Feature | Unit |
|---|---|---|
| 1–4 | speed mean, population SD, q95, end−start | m/s |
| 5–7 | acceleration RMS, mean absolute, q95 absolute | m/s² |
| 8–9 | jerk RMS, q95 absolute | m/s³ |
| 10–11 | yaw rate RMS, mean absolute | rad/s |
| 12 | total absolute wrapped heading increments | rad |
| 13 | summed Euclidean position increments | m |

Acceleration = diff(speed, prepend=first)/.1; jerk = diff(acceleration, prepend=first)/.1; yaw rate prepends zero to wrapped heading differences/.1. Quantiles are NumPy default linear; SD ddof=0. Existing scaler fits median and max(IQR, scale_floor), fills nonfinite with reference median, returns float32. Historical masks/scalers/outputs are not changed. This fixed-dt definition is distinct from timestamp-aware frozen F and median3 mechanism evaluation.

## Single H implementation

Executable specification: `tools/s1_protocol_schema.py::handcrafted_h` (30 raw columns):

- 13 existing ego13 values, unchanged formulas.
- 1 frozen F mean_abs_accel, distinct derivative definition. F mean/end/path duplicate ego13 mathematically apart from rounding, so do not include duplicate columns; no data-driven duplicate detection.
- 8 sample-mean signed acceleration bins, half-open [0,1),…,[7,8) seconds from physical first timestamp, over median3 speed then physical-time gradient edge_order=2. No event alignment, label conditioning or phase-count features. No clipping or extra smoothing.
- 3 fixed-index-lag autocorrelations at 5/10/20 samples: sum centered a[i]a[i+k] divided by full centered sum of squares. No lag search; denominator ≤1e−12 yields 0 and false validity.
- 2 braking-mass summaries: trapezoidal time weights times max(−a,0); weighted temporal centroid and square-root weighted variance in seconds. Total mass ≤1e−12 yields zero summaries and false validity.
- 2 validity columns (ACF denominator valid, braking mass valid) and 1 cadence diagnostic (maximum absolute physical Δt−0.1, seconds). Keep fixed columns, even constant; do not drop after observing D/E performance.

All 80 states must be finite and ordered. Preserve historical ego13 nominal-dt semantics on irregular timestamps and report that limitation; new temporal columns and frozen F use physical time. No newly invented cadence cutoff, interpolation/extrapolation or compaction across holes. Empty fixed physical-time bins are invalid. A scientifically justified cadence applicability rule, if required, must be derived from development measurement-error evidence before Q; it is presently unresolved. Any future Q/E input invalidity prevents full-cohort BDD qualification, not selective deletion. Numeric unit declarations cannot detect deliberately mislabeled plausible values; production serializer provenance is required.

H and RBR BDD preprocessing: separately fit D_fit baseline mean and population SD per column; zero/SD≤1e−12 columns use scale 1. No winsorizing, feature removal, whitening or projection. Reject missing/nonfinite input, never impute selected E failures. RBR checkpoint input preprocessing stays separately U-frozen. H has no fitting except its scaler; no model-dependent feature searches.

**O:** frozen Option-A brake phase count, interstage release fraction, second-brake peak ratio and status/validity flags. Used only for mechanism confirmation and interpretation. A perfect O classifier neither invalidates residual-to-F0 nor sets a requirement for RBR to beat O.

H includes generic temporal shape because the development mechanism is known; it is not a mechanism-blind discovery baseline. Its definition is frozen before E, with no expansion after RBR or H performance is seen. The separate F0/ego13/O roles prevent replacing an inconvenient strong challenger with a weak routine set.
