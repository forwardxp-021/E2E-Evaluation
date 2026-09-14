# PRIMARY BDD SAP — Draft v0.1

**S1_PROTOCOL_READY_FOR_OWNER_FREEZE.** Exactly one recommended Primary metric and operating point. Scientific Owner approved these design choices in S1.1; they are not empirically optimized values or permission to execute.

## Estimand and decision

Primary: **ΔBDD = P(alarm_RBR) − P(alarm_H)** at nominal FPR α=.05, **m=20 independent logs per release arm**, drift fraction **π=.50** (10 treatment logs and 10 baseline logs in target), fixed eligible TSB domain and frozen nonreactive replay setting. One contrast: RBR versus H. Direction baseline reference→mixed target is fixed. No raw MMD² comparison across representations, no switching to sample efficiency, Z, best dose, task or seed.

S1.1 operating-point semantics: π=.50 is a prospectively frozen **SYNTHETIC MODERATE-DRIFT BENCHMARK** prevalence for fair representation comparison. It is not an estimate of real-fleet prevalence, not an assertion that 50% of production scenarios change, and not an ODD population estimate. Claims are conditional on this operating point. The no-dose-search/no-operating-point-switch rule remains unchanged.

Primary RBR-BDD consumes frozen shared z64 directly under the existing scaler/kernel budget, not the training semantic head's predictions. See RBR_TRAINING_ARCHITECTURE_BOUNDARY in the Scope contract. H remains the sole Primary handcrafted comparator; F0_project=ego13 is an explicitly reported Secondary historical baseline, distinct from the four matching descriptors F_match.

This is the probability of detection under the specified release-construction distribution over independent eligible logs. Report finite-study-pool conditional probability separately from the inferred source-domain probability. Repeated release draws from a fixed pool do not constitute fresh independent releases from a fleet.

Minimum useful gain **δ*=.10** is a proposed scientific utility requirement, not estimated from TSB effect. Use a two-sided 95% log-aware interval [L,U]. Success: valid Q/E mechanism and measurement contracts, valid independent FPR gate for both pipelines, L>0 AND point Δ≥.10. Fail: valid analysis with U<.10 (rules applied after success, so no overlap); otherwise INCONCLUSIVE. A negative or imprecise result is retained; do not enlarge E. This distinguishes statistical superiority from a practical point-estimate criterion; a stronger rule L>.10 is an Owner alternative requiring repowering before execution.

FPR gate: both representations' one-sided simultaneous 95% upper bounds ≤.075 (Bonferroni .025 tail per representation). Nominal calibration remains .05; .025 is proposed evaluation tolerance. If lower confidence bound >.075, calibration validity FAIL; if neither, FPR qualification INCONCLUSIVE. Neither permits a superiority claim. No E threshold readjustment. Owner approval of α, m, π, δ*, tolerance and design is recorded in S1.1; do not choose the operating point on D to manufacture a gap.

## Preferred unpaired design and fallback

Prefer unpaired release emulation. At each release draw, sample 20 reference baseline logs and 20 different target logs, without replacement and without any source log on both sides. Choose exactly 10 of target logs by the frozen randomization to supply treatment, 10 baseline; use only that selected arm per log. Store both potential arms when collected for mechanism qualification but never insert a pair into opposite sides of an unpaired draw. Same draws, target masks and source weights for H and RBR. Selection depends on assigned intervention, not realized success. Uniform log weights over the approved eligible population; no post-treatment context matching. This is a source-specific marginal release-emulation estimand, not an ODD-universal claim.

AA calibration and independent AA evaluation use baseline-only releases, same m=20 per side and disjoint logs within each draw. Final AA calibration uses a separately reserved D_AA_final_calibration pool, disjoint from D_fit, Q and U. The two E pools (AA evaluation and AB evaluation) are disjoint from each other and from ALL U/Q/D logs. No E data calibrates the original threshold. AB draws may reuse source logs across Monte Carlo draws; that reuse is fully retained in uncertainty estimation.

**Capacity fallback fixed before E identity construction:** if census or D-only power/coverage audit fails the unpaired requirements below under approved budget, select paired prospective BDD only, retaining α=.05, m=20 PAIRS, π=.50 and the same Δ metric. Reference is all baseline; target is the same 20 source logs with 10 assigned treatment. Paired null uses log-level within-pair exchangeability; AA pseudo-null draws use preregistered pair swaps, not a borrowed unpaired threshold. Thresholds are recalibrated under that design, and independent held-out pair logs evaluate null false alarms. This tests a controlled assignment/swap null, not operational baseline release FPR; label it explicitly. No unpaired release-emulation claim survives. If paired capacity/power also fails, stop with Level 1; do not silently shrink m or increase π.

## Statistic and preprocessing

Use unbiased quadratic-time two-sample MMD²:

`sum_{i!=j} k(x_i,x_j)/(m(m−1)) + sum_{i!=j} k(y_i,y_j)/(m(m−1)) − 2 sum_{i,j} k(x_i,y_j)/m²`.

Negative unbiased values are retained. RBF k=exp(−||x−y||²/(2σ²)). Each representation is separately standardized by its frozen D_fit baseline mean/SD (constant columns scale 1). No outcome-based projection or feature selection. Bandwidth is sqrt(median of strictly positive pairwise squared distances/2) on standardized D_fit baseline vectors; if no positive distances, mark pipeline degenerate, not an epsilon rescue. This corresponds to exp(−distance²/median_distance²). Apply identical mathematical rule to H and RBR, not identical numeric bandwidths.

Fair tuning budget: **one scaler and one median-bandwidth rule per representation; zero supervised kernel/readout selection for Primary**. No rank search, no best-kernel search, no post-D feature additions. D may diagnose and estimate power, not alter this Primary rule. Auxiliary readout fitting is separately bounded in the Secondary SAP. Encoder architecture/checkpoint selection exclusively U; never select encoder using Q/D alarms, mechanism labels or H performance.

## Calibration, alarm and uncertainty

After frozen encoder and statistical contracts and separate calibration authorization, BEFORE any E access, use only D_AA_final_calibration to compute 1,999 release draws (seed 2026091101). For each representation threshold is ordered AA statistic at rank ceil(.95×(1999+1))=1900 (1-based), alarm iff statistic > threshold, ties do not alarm. This finite-Monte-Carlo quantile is conditional on the calibration log pool, not an exact population-FPR guarantee. All comparisons use the fixed original calibrated thresholds.

E_AA_evaluation: 10,000 draws (seed 2026091102). E_AB_evaluation: 10,000 shared draws (seed 2026091103). Conditional Monte Carlo SE for Δ uses the paired alarm difference d∈{−1,0,1}: sample SD(d)/sqrt(B); do not use independent-binomial SE for RBR and H. AA analogous. Monte Carlo precision may be improved only by increasing draws under a predeclared numerical error criterion (SE≤.005), maximum 40,000; never add source logs or change thresholds. Report actual draws.

Scientific interval: 2,000 outer stratified-by-role log-cluster bootstrap replicates, seed 2026091104. Each source log carries all its arms, features, masks and failures together. Bootstrap the frozen D calibration pool and E evaluation pools independently; recompute the calibration threshold for each bootstrap replicate solely to propagate calibration uncertainty, NOT to modify the original pipeline or result. Recompute the paired H/RBR detection difference with shared inner draw randomness (2,000 draws per outer replicate). Use percentile 2.5/97.5 limits; AA upper gates use 97.5 percentile for each pipeline. In source-group resampling, repeated copies of a log must never masquerade as distinct logs on opposing sides of a release: implement weighted source-log resampling with disjoint original-ID allocation within each release; report effective unique log support. If this bootstrap has insufficient support or lacks development coverage validation, final source-domain interval is NOT_ESTABLISHED and Primary INCONCLUSIVE. Do not report Monte Carlo binomial intervals as a substitute.

This bootstrap is a proposed finite-pool/source-log uncertainty procedure requiring zero-simulation D/historical validation for bias and coverage before E. Reused synthetic releases are dependent through shared logs; McNemar treating all 10,000 draws as independent scientific trials is prohibited. If genuinely independent release blocks can be afforded, their paired alarm-difference interval is preferable, but a switch requires Owner pre-E amendment and sample redesign.

## E sample-size rule and budget

Q size does not determine E size. After registered Q→D and any separately authorized U-only encoder choice, perform power estimation using D/historical exposed data only. No fresh E identities or E outcomes. Q=20 does not support 40-log unpaired batches by itself; it cannot validate the final null tail or provide RBR effect estimates before an encoder exists. Historical Stage6 effects are scenario-specific pilot scenarios, not TSB RBR effect estimates.

Owner capacity options, all **planning envelopes, not powered recommendations**:

| Unpaired E envelope | AA-cal logs | AA-eval logs | AB-eval logs | Calibration + E entry cap, both arms everywhere |
|---|---:|---:|---:|---:|
| Small | 80 | 80 | 160 | 640 |
| Medium | 120 | 120 | 240 | 960 |
| Large | 160 | 160 | 320 | 1,280 |

Distinct-log totals 320/480/640 include the newly reserved D_AA_final_calibration logs and E logs; they exclude existing U/D/Q and engineering reservations. Both-arm collection provides common mechanism qualification and paired diagnostics; all eligibility/qualification failures retain their denominators. No actual E allocation in S1. The first column of allocation is D calibration, so E alone contains 240/360/480 logs (480/720/960 entries); total budgets above include the separate calibration collection. These numbers are bounded resource proposals, not assertions of eligible capacity or statistical adequacy. Q+final-calibration+E total caps under recommended Q would be 680/1000/1320, still unauthorized. Paired fallback may examine log-pool allocations 40/40/80, 60/60/120, 80/80/160 for null-cal/null-eval/AB respectively, with 320/480/640 total calibration-plus-E entry caps; pairing does not eliminate source uncertainty.

For each candidate envelope in ascending cost order, use log-level development resampling to evaluate the full algorithm (including calibration uncertainty and fixed failure policy), at effect scenarios 0, .10 and .20 detection gain with explicitly recorded construction assumptions; do not assert an unknown learned effect. Estimate discordant-alarm variance and log influence, require coverage of nominal intervals near 95%, null false-positive rejection ≤.05, and power ≥.80 at the .20 design alternative under the .10 practical criterion. Use conservative upper variance/low effect scenarios; repeat across development source groups where support permits. Small D cannot demonstrate coverage: record insufficient evidence rather than extrapolate precision.

Freeze the numerical original thresholds on final D calibration before constructing/unblinding E; calibration never becomes E and never feeds encoder selection. Select the lowest-cost envelope satisfying these preregistered checks, independent support and approved capacity. If none, paired fallback; if neither, Level 1 closure. Bind final N, allocations, role hashes and overall run ceiling ONCE before E construction/unblinding. Owner may reject cost; cannot relax power, change m/π, invent favorable RBR gain or extend E after a near miss. No scalar final E number is scientifically justified yet.

## Multiplicity and missingness

One Primary contrast/operating point; two-sided 95% CI, no Primary multiplicity adjustment. FPR validity gate is an intersection requirement with simultaneous limits. Secondary formal hypotheses form the fixed family in the Secondary SAP; no replacement of Primary by a favorable secondary.

No removal for mechanism/safety/low-speed failure. Whole-roster qualification failure prevents a mechanism-confirmed Primary utility claim, though complete traces may be described. Missing/corrupt representation inputs yield INCONCLUSIVE under full-roster design; report complete-case descriptive values explicitly as such, not Primary. No E rerun, outlier removal, imputation rescue, additional seed or second encoder qualification.
