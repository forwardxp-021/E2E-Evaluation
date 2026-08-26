# R0 Statistical Analysis Plan v1.0

Status: `R0_PROTOCOL_V1_0_FROZEN`. This is the frozen v1.0 binding of SAP
v0.3; it introduces no new scientific hypothesis or numerical primary gate.

## Global analysis contract

- alpha = 0.05; confidence level = 0.95.
- Multiplicity: Holm within each module/family; D4 formal equivalence uses
  family-specific TOST/IUT only after its R4 pre-unblind upgrade gate.
- Bootstrap: 5,000 log-cluster repetitions (scenario/source-group fallback
  requires recorded downgrade). Permutation: 49,999 repetitions using the
  frozen paired/group unit.
- Fixed seeds: 2026082601, 3407, 3408, 3409.
- D0 gate: absolute paired standardized retention difference ≥0.10, 95% CI
  excludes 0, direction consistent in ≥2/3 seeds.
- D1 gates: continuous R² ≥0.10 with CI lower >0; categorical BA ≥0.60 with
  CI lower >0.50; each family ≥2/3 core targets; insufficient support is
  `INCONCLUSIVE`.
- D2: frozen matching strata and q99 OOD rule; natural-data pairing/shuffle is
  not a causal coupling claim.
- D3: RBF kernel, treatment-label-blind median bandwidth, ranks {1,2,4,8,16},
  and FPR 0.05 with 95% CI upper ≤0.075.
- D4: family-specific primary roles R-HLC=4, R-TSB=4, R-IP=3. The frozen
  robust-IQR fallback is development-only, not physical or R4 confirmatory
  equivalence.

## Evidence and result model

`R0_AUDIT_HOLDOUT = NOT_AVAILABLE`; the allowed evidence level is
`DEVELOPMENT_DIAGNOSTIC_EVIDENCE`. Execution status is `COMPLETE` or
`BLOCKED`; each hypothesis result is `SUPPORTED`, `NOT_SUPPORTED`, `MIXED`,
`INCONCLUSIVE`, or `NOT_EVALUABLE`. No hypothesis-level result is implied by
this freeze.

## Frozen hypothesis inventory (24)

- `D0_LENGTH_EFFECT` (D0)
- `D0_POSITION_RETENTION_ASSOCIATION` (D0)
- `D0_POOLING_EFFECT` (D0)
- `D0_MASK_PADDING_SENSITIVITY` (D0)
- `D1_KNOWN_SEMANTIC_INFORMATION_PRESENT` (D1)
- `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER` (D1)
- `D1_GEOMETRY_DEGENERACY` (D1)
- `D2_RESPONSE_SENSITIVITY` (D2)
- `D2_CONTEXT_SENSITIVITY` (D2)
- `D2_PAIRING_SENSITIVITY` (D2)
- `D2_SHORTCUT_RISK` (D2)
- `D2_ABLATION_OOD_RISK` (D2)
- `D3_FULL64_SIGNAL_DILUTION` (D3)
- `D3_PROJECTED_READOUT_GAIN` (D3)
- `D3_NULL_CALIBRATION_PRESERVED` (D3)
- `D4_DESCRIPTOR_EQUIVALENCE_R_HLC` (D4)
- `D4_MECHANISM_DIFFERENCE_R_HLC` (D4)
- `D4_OUTCOME_BLIND_FEASIBILITY_R_HLC` (D4)
- `D4_DESCRIPTOR_EQUIVALENCE_R_TSB` (D4)
- `D4_MECHANISM_DIFFERENCE_R_TSB` (D4)
- `D4_OUTCOME_BLIND_FEASIBILITY_R_TSB` (D4)
- `D4_DESCRIPTOR_EQUIVALENCE_R_IP` (D4)
- `D4_MECHANISM_DIFFERENCE_R_IP` (D4)
- `D4_OUTCOME_BLIND_FEASIBILITY_R_IP` (D4)

Machine-readable source of truth: `docs/stageR/r0/manifests/r0_statistical_analysis_plan_v1.0.json` (SHA256:
`804512b50468f8f8534a702ed13db63fed21e2c7db8cfa0cc3518cff9b66f58d`).
