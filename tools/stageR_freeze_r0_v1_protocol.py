#!/usr/bin/env python3
"""Generate the Phase-A, content-only R0 v1.0 protocol freeze artifacts.

This tool is intentionally a document/manifest generator.  It does not load
model checkpoints, tensors, or outputs, and it does not run any scientific
analysis.  Its only inputs are the already frozen R0 v0.x governance assets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASELINE = "b929cd62109f3e5cdc015903a958fa574d181e40"
BRANCH = "20260825_stageR_new"
PROTECTED_DIRTY = (
    "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/"
    "behavior_events_v2/behavior_event_metrics_v2.csv"
)


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def gate_for(hypothesis_id: str) -> tuple[str, str, str, str]:
    """Return primary metric, primary gate, seed rule, fallback identifier."""
    if hypothesis_id.startswith("D0_"):
        return (
            "predeclared temporal retention/readout contrast",
            "absolute paired standardized retention difference >= 0.10; 95% CI excludes 0",
            ">=2/3 fixed seeds direction-consistent",
            "NONE",
        )
    if hypothesis_id == "D1_KNOWN_SEMANTIC_INFORMATION_PRESENT":
        return (
            "held-out grouped R2 (continuous) or balanced accuracy (categorical)",
            "continuous: R2 >= 0.10 and log-cluster 95% CI lower > 0; categorical: BA >= 0.60 and 95% CI lower > 0.50; each semantic family >=2/3 core targets",
            ">=2/3 learned representation families; A/B/C require >=2/3 seed direction consistency",
            "NONE",
        )
    if hypothesis_id.startswith("D1_"):
        return (
            "predeclared grouped probe/geometry metric",
            "fixed D1 probe contract, grouped split and log-cluster CI; insufficient independent support is INCONCLUSIVE",
            ">=2/3 applicable fixed seeds when replicated",
            "NONE",
        )
    if hypothesis_id.startswith("D2_"):
        return (
            "predeclared matched/shuffle/ablation contrast",
            "D2 fixed matching strata and OOD boundary; no causal coupling claim from natural-data shuffle alone",
            ">=2/3 fixed seeds when replicated",
            "NONE",
        )
    if hypothesis_id == "D3_NULL_CALIBRATION_PRESERVED":
        return (
            "empirical null false-positive rate",
            "nominal FPR=0.05 and two-sided 95% CI upper <= 0.075; insufficient independent null units is INCONCLUSIVE",
            ">=2/3 fixed seeds direction-consistent",
            "NONE",
        )
    if hypothesis_id.startswith("D3_"):
        return (
            "full64 versus frozen projected readout effect",
            "fixed RBF kernel; treatment-label-blind median bandwidth; ranks {1,2,4,8,16}; smallest rank within 1 SE after retention and calibration gates",
            ">=2/3 fixed seeds direction-consistent",
            "NONE",
        )
    if "DESCRIPTOR_EQUIVALENCE" in hypothesis_id:
        return (
            "family-specific primary F_match balance",
            "all frozen primary F_match absolute pair differences within 0.10 x frozen robust-IQR caliper; development feasibility only",
            "not applicable to deterministic matching contract",
            "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
        )
    if "MECHANISM_DIFFERENCE" in hypothesis_id:
        return (
            "family-specific M_behavior mechanism contrast",
            "predeclared whole-roster mechanism estimate with no outcome-driven pair removal",
            ">=2/3 fixed seeds when applicable",
            "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
        )
    return (
        "outcome-blind feasibility ledger",
        "pre-treatment eligibility, runnability, family coverage and power ledger complete; no representation/BDD/probe outcome used",
        "not applicable to deterministic feasibility ledger",
        "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
    )


def decision_rules(gate: str) -> dict[str, str]:
    return {
        "supported_rule": f"analysis evaluable and the frozen primary gate is met: {gate}",
        "not_supported_rule": "analysis evaluable and the frozen primary gate is not met",
        "mixed_rule": "predeclared applicable strata, representations, or fixed seeds have material directional conflict",
        "inconclusive_rule": "independent units, class support, uncertainty, or prespecified evaluability requirements are insufficient",
        "not_evaluable_rule": "required frozen input, implementation contract, or valid analysis unit is unavailable",
    }


def main() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"refusing generation outside {BRANCH}")
    if git("rev-parse", "HEAD") != BASELINE:
        raise SystemExit(f"refusing generation outside baseline {BASELINE}")

    docs = REPO / "docs/stageR/r0"
    protocol_dir = docs / "protocol"
    governance_dir = docs / "governance"
    manifest_dir = docs / "manifests"

    v06_protocol = protocol_dir / "R0_Representation_Measurement_Audit_Protocol_v0.6_zh.md"
    v03_sap = manifest_dir / "r0_statistical_analysis_plan_v0.3.json"
    owner_v01 = manifest_dir / "r0_scientific_owner_approval_v0.1.json"
    d1_gate = manifest_dir / "r0_d1_core_semantic_targets_v0.1.json"
    d4_contract = manifest_dir / "r0_d4_family_specific_feature_roles_v0.1.csv"
    d4_fallback = manifest_dir / "r0_d4_development_balance_fallback_v1.json"
    asset_inventory = manifest_dir / "r0_asset_inventory_v0.3.csv"
    contract_inventory = manifest_dir / "r0_contract_inventory_v0.3.csv"
    target_definition = manifest_dir / "r0_target_definition_v0.2.json"
    d0_policy = protocol_dir / "R0_D0_Temporal_Audit_Policy_v0.1.md"
    mask_policy = protocol_dir / "R0_Mask_Padding_Audit_Policy_v0.1.md"

    sap_source = json.loads(v03_sap.read_text(encoding="utf-8"))
    owner_source = json.loads(owner_v01.read_text(encoding="utf-8"))
    d1_source = json.loads(d1_gate.read_text(encoding="utf-8"))
    if len(sap_source["hypotheses"]) != 24:
        raise SystemExit("SAP v0.3 must contain exactly 24 hypotheses")
    if d1_source["core_target_count"] != 9:
        raise SystemExit("D1 manifest must contain exactly nine core targets")

    # A. Scientific owner approval v0.2.
    owner = dict(owner_source)
    owner.update(
        {
            "schema_version": "r0_scientific_owner_approval_v0.2",
            "recorded_date": "2026-08-26",
            "approval_source": "explicit scientific-owner final approval addendum for R0 V1 Formal Protocol Freeze",
            "supersedes": rel(owner_v01),
            "d1": {
                "continuous_information_presence_gate": {
                    "point_estimate": "held-out grouped R2 >= 0.10",
                    "uncertainty": "cluster-aware 95% CI lower bound > 0",
                },
                "categorical_information_presence_gate": {
                    "point_estimate": "balanced accuracy >= 0.60",
                    "uncertainty": "95% CI lower bound > 0.50",
                },
                "family_gate": "each semantic family has at least 2/3 core targets passing",
                "insufficient_sample_result": "INCONCLUSIVE",
                "interpretation_boundary": [
                    "R0_MINIMUM_SEMANTIC_INFORMATION_PRESENCE only",
                    "not RBR semantic superiority",
                    "not RBR semantic noninferiority",
                    "not human-perceptible semantic fidelity",
                    "interaction front-distance, relative-speed, or pressure decoding alone is not correct ego-context causal coupling; interpret with D2 pairing and shortcut results",
                ],
                "status": "SCIENTIFIC_OWNER_APPROVED",
                "gate_manifest": rel(d1_gate),
                "gate_manifest_sha256": sha(d1_gate),
            },
            "d4_family_specific_matching_contract": {
                "R-HLC_primary_f_match_count": 4,
                "R-TSB_primary_f_match_count": 4,
                "R-IP_primary_f_match_count": 3,
                "status": "SCIENTIFIC_OWNER_APPROVED",
                "contract_path": rel(d4_contract),
                "contract_sha256": sha(d4_contract),
            },
            "d4_development_fallback": {
                "fallback_id": "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
                "status": "CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT",
                "permanent_interpretation_boundaries": [
                    "NOT_FORMAL_PHYSICAL_EQUIVALENCE",
                    "NOT_R4_CONFIRMATORY_EQUIVALENCE",
                ],
                "fallback_path": rel(d4_fallback),
                "fallback_sha256": sha(d4_fallback),
            },
            "training_authorization": "NOT_AUTHORIZED",
        }
    )
    owner_path = manifest_dir / "r0_scientific_owner_approval_v0.2.json"
    write_json(owner_path, owner)
    write_text(
        governance_dir / "R0_Scientific_Owner_Approval_Record_v0.2.md",
        f"""# R0 Scientific Owner Approval Record v0.2

## Binding decision

This record preserves the 18/18 parameter approvals in v0.1 and formally adds
the D1 and D4 decisions required for the R0 Protocol v1.0 freeze.

- D1 continuous presence: held-out grouped R² ≥ 0.10 and cluster-aware 95% CI
  lower bound > 0.
- D1 categorical presence: balanced accuracy ≥ 0.60 and 95% CI lower bound >
  0.50.
- D1 family gate: every semantic family must have at least 2/3 core targets
  passing; insufficient support is `INCONCLUSIVE`.
- These are `R0_MINIMUM_SEMANTIC_INFORMATION_PRESENCE` gates only. They are
  not RBR superiority, RBR noninferiority, or human-perceptible semantic
  fidelity gates. Interaction decoding must be interpreted with D2
  pairing/shortcut results and does not alone establish ego-context causal
  coupling.
- D4 family-specific primary F_match counts are approved: R-HLC=4, R-TSB=4,
  R-IP=3.
- `D4_DEVELOPMENT_BALANCE_FALLBACK_V1` is
  `CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT`, but permanently
  `NOT_FORMAL_PHYSICAL_EQUIVALENCE` and `NOT_R4_CONFIRMATORY_EQUIVALENCE`.

This approval does not authorize RBR-A/B/C formal training. The authoritative
machine record is `{rel(owner_path)}` (SHA256: `{sha(owner_path)}`).
""",
    )

    # B. R4 acquisition anti-selection addendum.
    r4_manifest_path = manifest_dir / "r0_future_r4_reserved_source_or_generator_freeze_v0.2.json"
    r4_manifest = {
        "schema_version": "r0_future_r4_reserved_source_or_generator_freeze_v0.2",
        "status": "FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR_FROZEN",
        "frozen_date": "2026-08-26",
        "source_branch": BRANCH,
        "pre_freeze_baseline_commit": BASELINE,
        "freeze_form": "FROZEN_PROSPECTIVE_ACQUISITION_RULE",
        "rule_id": "R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1",
        "reserved_source_or_generator_rule": {
            "eligible_source": "first newly acquired, research-licensed, nuPlan-compatible source batch satisfying schema/map/runnability and complete identity-ledger requirements after this freeze",
            "deterministic_source_tie_break": "lexicographic(dataset_release_id, source_manifest_sha256) among simultaneously eligible batches",
            "identity_exclusions": [
                "all Waymo train/val/historical-test identities",
                "all Stage6/Stage7/Stage7L identities",
                "all R0 development/audit identities",
                "any source/log/token exposed to representation outcome",
            ],
            "selection_algorithm": "SHA256(2026082601|source_release|log_name|scenario_token), log-disjoint whole roster",
        },
        "r4_source_acquisition_choice": {
            "status": "NOT_YET_FROZEN",
            "training_authorization_precondition": True,
            "legal_forms": [
                "exact reserved source universe",
                "deterministic external source eligibility universe",
                "frozen acquisition request/channel manifest",
                "exact controlled-generator base source universe",
            ],
            "rule_or_source_id": None,
            "freeze_timestamp": None,
            "source_eligibility_rule": None,
            "acquisition_channel_or_source_universe": None,
            "outcome_accessed": False,
            "representation_outcome_accessed": False,
            "formal_rbr_outcome_may_not_determine_choice": True,
            "if_unsatisfied_before_training": "RBR_A/B/C_TRAINING_AUTHORIZATION=NOT_AUTHORIZED",
            "protocol_freeze_effect": "NOT_A_PROTOCOL_FREEZE_BLOCKER",
        },
        "generation_contract": {
            "design": "paired baseline/treatment controlled planner",
            "families": ["R-HLC", "R-TSB", "R-IP"],
            "pre_treatment_eligibility_only": True,
            "whole_roster_intention_to_evaluate": True,
            "realized_mechanism_exclusion": "PROHIBITED",
            "representation_outcome_selection": "PROHIBITED",
        },
        "final_confirmation_roster": "NOT_FROZEN_BY_DESIGN",
        "formal_r4_pre_unblind_requirements": [
            "exact source/log/token roster",
            "planner/config/code SHA",
            "dose grid",
            "failure/missingness policy",
            "power/family allocation",
            "family-specific physical/material equivalence margins and TOST/IUT",
        ],
        "rbr_training_authorization": "NOT_AUTHORIZED",
    }
    write_json(r4_manifest_path, r4_manifest)
    write_text(
        governance_dir / "R0_Future_R4_Reserved_Source_or_Generator_Proposal_v0.3.md",
        f"""# R0 Future R4 Reserved Source or Generator Proposal v0.3

## Frozen source/generator boundary and anti-selection rule

`FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR = FROZEN`; the final confirmation
roster remains `NOT_FROZEN_BY_DESIGN`. The prospective rule is retained from
v0.2: a newly acquired, research-licensed, nuPlan-compatible batch with a
complete identity ledger, selected deterministically if multiple batches are
simultaneously eligible.

Before any formal RBR-A/B/C training, a distinct
`R4_SOURCE_ACQUISITION_CHOICE` must be frozen as exactly one legal source
universe/channel form. It must record its rule/source ID, timestamp,
eligibility rule, acquisition channel or source universe, and both
`outcome_accessed=false` and `representation_outcome_accessed=false`. A formal
RBR outcome must never determine this choice.

That prerequisite is a training-authorization gate, not a protocol-freeze
blocker. Until it is satisfied, all RBR-A/B/C authorizations remain
`NOT_AUTHORIZED`. Machine binding: `{rel(r4_manifest_path)}` (SHA256:
`{sha(r4_manifest_path)}`).
""",
    )

    # C. Protocol v1.0, preserving v0.6 method content and adding only freeze semantics.
    protocol_text = v06_protocol.read_text(encoding="utf-8")
    protocol_text = protocol_text.replace(
        "# R0 Representation & Measurement Audit Protocol v0.6（Protocol Consistency Repair）",
        "# R0 Representation & Measurement Audit Protocol v1.0（Formal Protocol Freeze）",
        1,
    ).replace(
        "文档状态：`READY_FOR_R0_V1_PROTOCOL_FREEZE`",
        "文档状态：`R0_PROTOCOL_V1_0_FROZEN`",
        1,
    ).replace(
        "v0.6 source baseline commit（2026-08-26）：`b4bcc9699c534ea6341c19b9a247f80c9e279cbe`",
        f"v1.0 pre-freeze baseline commit（2026-08-26）：`{BASELINE}`",
        1,
    )
    protocol_text = re.sub(
        r"当前研究状态：\n>\n> ```text\n.*?\n> ```",
        """当前研究状态：
>
> ```text
> R0_PROTOCOL_V1_0_FROZEN
> R0_EXECUTION_AUTHORIZED_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE
> R0_AUDIT_HOLDOUT_NOT_AVAILABLE
> RBR_A_TRAINING_AUTHORIZATION_NOT_AUTHORIZED
> RBR_B_TRAINING_AUTHORIZATION_NOT_AUTHORIZED
> RBR_C_TRAINING_AUTHORIZATION_NOT_AUTHORIZED
> R4_SOURCE_OR_GENERATOR_FROZEN
> R4_CONFIRMATION_ROSTER_NOT_FROZEN
> ```""",
        protocol_text,
        count=1,
        flags=re.DOTALL,
    )
    protocol_text = protocol_text.replace(
        "它仍不是最终 v1.0 冻结稿，不授权训练 RBR-A/B/C，不改变 Stage6/Stage7/Stage7L 已冻结历史结论。",
        "v1.0 仅将既有科学内容、最终 owner approval、R4 anti-selection training gate 与 artifact binding 正式冻结；它不授权 RBR-A/B/C training，也不改变 Stage6/Stage7/Stage7L 已冻结历史结论。",
        1,
    )
    protocol_v1_path = protocol_dir / "R0_Representation_Measurement_Audit_Protocol_v1.0_zh.md"
    write_text(
        protocol_v1_path,
        protocol_text
        + f"""

---

## 27. v1.0 formal freeze binding and execution boundary

This v1.0 file is scientifically based on v0.6. No new hypothesis, threshold,
target, readout, margin, or outcome-based selection rule is introduced here.
The final D1 gates and D4 fallback approval are bound through
`r0_scientific_owner_approval_v0.2.json`; the R4 anti-selection training gate
is bound through `r0_future_r4_reserved_source_or_generator_freeze_v0.2.json`.

`R0_AUDIT_HOLDOUT = NOT_AVAILABLE` and the currently permitted evidence level
is `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`. These are explicit evidence boundaries
and do not change `R0_PROTOCOL_V1_0_FROZEN`. They support the distinct status
`R0_EXECUTION_AUTHORIZED_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE`; this document
does not execute any D0--D4 analysis.

There is no `PROTOCOL_DEFINITION_BLOCKER`. Formal RBR-A/B/C training remains
separately `NOT_AUTHORIZED`, because R0 scientific execution/decision records
and candidate-specific activation gates, including the exact
`R4_SOURCE_ACQUISITION_CHOICE`, have not been completed.

Pre-freeze source branch: `{BRANCH}`. Pre-freeze baseline commit:
`{BASELINE}`. Complete artifact SHA bindings are in
`r0_protocol_frozen_v1.0.json` and Phase-B Git binding is intentionally stored
separately to avoid a self-referential commit SHA.
""",
    )

    # D. SAP v1.0 JSON and readable protocol document.
    sap_v1 = dict(sap_source)
    sap_v1.update(
        {
            "schema_version": "r0_statistical_analysis_plan_v1.0",
            "status": "R0_PROTOCOL_V1_0_FROZEN",
            "source_baseline": {"path": rel(v03_sap), "sha256": sha(v03_sap)},
            "evidence_level_rule": {
                "r0_audit_holdout": "NOT_AVAILABLE",
                "current_allowed_level": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
                "execution_authorization": "R0_EXECUTION_AUTHORIZED_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
                "confirmatory_wording": "PROHIBITED without a future eligible audit/confirmation asset",
            },
            "final_owner_approval": {"path": rel(owner_path), "sha256": sha(owner_path)},
            "r4_training_gate": {"path": rel(r4_manifest_path), "sha256": sha(r4_manifest_path)},
            "primary_gate_completeness": {
                "undefined_numerical_primary_gate_count": 0,
                "unresolved_protocol_definition_count": 0,
                "hypothesis_count": len(sap_source["hypotheses"]),
            },
        }
    )
    sap_v1["d1"] = {
        "core_target_count": 9,
        "core_target_manifest": rel(d1_gate),
        "core_target_manifest_sha256": sha(d1_gate),
        "continuous_gate": "held-out grouped R2 >= 0.10 AND log-cluster 95% CI lower > 0",
        "categorical_gate": "balanced accuracy >= 0.60 AND 95% CI lower > 0.50",
        "family_gate": "each family >=2/3 core targets; insufficient support -> INCONCLUSIVE",
        "interpretation_boundary": "minimum semantic information presence, not RBR superiority/noninferiority/human fidelity; interaction decode requires D2 pairing/shortcut interpretation",
    }
    sap_v1["d4"]["family_specific_contract_path"] = rel(d4_contract)
    sap_v1["d4"]["family_specific_contract_sha256"] = sha(d4_contract)
    sap_v1["d4"]["development_fallback"] = {
        "fallback_id": "D4_DEVELOPMENT_BALANCE_FALLBACK_V1",
        "path": rel(d4_fallback),
        "sha256": sha(d4_fallback),
        "status": "CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT",
        "not_formal_physical_equivalence": True,
        "not_r4_confirmatory_equivalence": True,
    }
    sap_v1_path = manifest_dir / "r0_statistical_analysis_plan_v1.0.json"
    write_json(sap_v1_path, sap_v1)
    hypothesis_lines = "\n".join(f"- `{h['hypothesis_id']}` ({h['module']})" for h in sap_source["hypotheses"])
    write_text(
        protocol_dir / "R0_Statistical_Analysis_Plan_v1.0.md",
        f"""# R0 Statistical Analysis Plan v1.0

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
- D3: RBF kernel, treatment-label-blind median bandwidth, ranks {{1,2,4,8,16}},
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

{hypothesis_lines}

Machine-readable source of truth: `{rel(sap_v1_path)}` (SHA256:
`{sha(sap_v1_path)}`).
""",
    )

    # E. Formal data-role split manifest.
    split_path = manifest_dir / "r0_split_manifest_v1.0.csv"
    split_cols = [
        "asset_role", "status", "source_identity", "source_identity_sha256",
        "historical_use_status", "overlap_rule", "allowed_use", "forbidden_use",
        "evidence_level", "freeze_or_availability_note",
    ]
    split_rows = [
        {
            "asset_role": "R0_DEVELOPMENT", "status": "FROZEN_BOUND",
            "source_identity": rel(asset_inventory), "source_identity_sha256": sha(asset_inventory),
            "historical_use_status": "HISTORICALLY_UNBLINDED_DEVELOPMENT_EVIDENCE",
            "overlap_rule": "may include historical assets; must not be relabelled untouched holdout or future R4 confirmation",
            "allowed_use": "protocol execution, diagnostic estimation, fixed-method development, power and feasibility reporting",
            "forbidden_use": "confirmatory claim, outcome-driven future-R4 selection, RBR authorization by itself",
            "evidence_level": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
            "freeze_or_availability_note": "FROZEN/BOUND to inventory v0.3",
        },
        {
            "asset_role": "R0_AUDIT_HOLDOUT", "status": "NOT_AVAILABLE",
            "source_identity": "NONE_CURRENT_NUPLAN_AUDIT_HOLDOUT", "source_identity_sha256": "NOT_APPLICABLE",
            "historical_use_status": "NO_CURRENT_ASSET", "overlap_rule": "not applicable until a future identity ledger is frozen",
            "allowed_use": "none", "forbidden_use": "availability claim, evidence inflation, threshold selection",
            "evidence_level": "NOT_AVAILABLE; DEVELOPMENT_DIAGNOSTIC_EVIDENCE remains the current boundary",
            "freeze_or_availability_note": "absence is nonblocking for protocol freeze",
        },
        {
            "asset_role": "FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR", "status": "FROZEN",
            "source_identity": "R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1", "source_identity_sha256": sha(r4_manifest_path),
            "historical_use_status": "NO_R4_OUTCOME_ACCESSED", "overlap_rule": "log/token disjoint from Waymo, Stage6/7/7L, R0 and representation-outcome-exposed identities",
            "allowed_use": "future outcome-blind source/generator acquisition under frozen rule",
            "forbidden_use": "formal RBR outcome selection, representation outcome selection, final roster claim",
            "evidence_level": "FUTURE_PROSPECTIVE_ONLY",
            "freeze_or_availability_note": "source/generator rule frozen; exact R4 source acquisition choice not yet frozen",
        },
        {
            "asset_role": "FUTURE_R4_CONFIRMATION_ROSTER", "status": "NOT_FROZEN_BY_DESIGN",
            "source_identity": "NONE_FINAL_ROSTER", "source_identity_sha256": "NOT_APPLICABLE",
            "historical_use_status": "NO_CURRENT_ASSET", "overlap_rule": "must be log/token disjoint and outcome-blind when generated from reserved universe",
            "allowed_use": "future R1/R4 outcome-blind roster formation only", "forbidden_use": "pre-generated roster claim, outcome-driven selection",
            "evidence_level": "FUTURE_CONFIRMATORY_ONLY",
            "freeze_or_availability_note": "intentionally not frozen at R0 v1.0",
        },
    ]
    write_csv(split_path, split_cols, split_rows)

    # F. One row per frozen SAP hypothesis.
    decision_path = manifest_dir / "r0_decision_table_v1.0.csv"
    decision_cols = [
        "hypothesis_id", "module", "primary_metric", "minimum_effect_or_gate", "ci_rule",
        "multiplicity_family", "seed_rule", "evidence_requirement", "supported_rule",
        "not_supported_rule", "mixed_rule", "inconclusive_rule", "not_evaluable_rule",
        "blocks_rbr_A", "blocks_rbr_B", "blocks_rbr_C", "fallback_id",
    ]
    decision_rows: list[dict[str, object]] = []
    for h in sap_source["hypotheses"]:
        metric, gate, seed_rule, fallback = gate_for(h["hypothesis_id"])
        rules = decision_rules(gate)
        decision_rows.append(
            {
                "hypothesis_id": h["hypothesis_id"], "module": h["module"],
                "primary_metric": metric, "minimum_effect_or_gate": gate,
                "ci_rule": "95% log-cluster CI; BCa when estimable, declared percentile fallback",
                "multiplicity_family": h["analysis_family"], "seed_rule": seed_rule,
                "evidence_requirement": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE; no confirmatory wording while audit holdout is NOT_AVAILABLE",
                **rules,
                "blocks_rbr_A": "true", "blocks_rbr_B": "true", "blocks_rbr_C": "true",
                "fallback_id": fallback,
            }
        )
    write_csv(decision_path, decision_cols, decision_rows)

    # G. Execution environment, command, and deviation schemas.
    environment_path = manifest_dir / "r0_environment_schema_v1.0.json"
    write_json(environment_path, {
        "schema_version": "r0_environment_schema_v1.0", "status": "FROZEN_FOR_R0_EXECUTION",
        "required_fields": ["execution_id", "timestamp_utc", "git_commit", "git_branch", "dirty_worktree", "os", "python", "package_lock_or_environment_sha256", "hardware", "seed", "input_artifact_sha256", "protocol_frozen_sha256"],
        "prohibited_omission": "all required fields must be present before a scientific execution record is accepted",
        "evidence_level_default": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
    })
    command_path = manifest_dir / "r0_command_ledger_schema_v1.0.json"
    write_json(command_path, {
        "schema_version": "r0_command_ledger_schema_v1.0", "status": "FROZEN_FOR_R0_EXECUTION",
        "required_fields": ["execution_id", "command_id", "timestamp_utc", "operator", "command", "working_directory", "git_commit", "input_artifact_sha256", "output_artifact_sha256", "exit_code", "seed", "environment_record_id", "protocol_deviation_id"],
        "rules": ["append-only", "record before outcome interpretation", "no silent rerun replacement", "outcome access must be declared"],
    })
    deviation_path = manifest_dir / "r0_protocol_deviation_log_v1.0.csv"
    deviation_cols = ["deviation_id", "timestamp_utc", "detected_after_outcome_access", "description", "affected_protocol_section", "affects_primary", "evidence_downgrade", "mitigation", "scientific_owner_disposition", "closed_timestamp_utc"]
    write_csv(deviation_path, deviation_cols, [])

    # H. Candidate-specific formal RBR authorization remains blocked but hashes are prebound.
    artifact_paths = {
        "protocol_sha": protocol_v1_path,
        "decision_table_sha": decision_path,
        "asset_inventory_sha": asset_inventory,
        "contract_inventory_sha": contract_inventory,
        "split_manifest_sha": split_path,
        "sap_sha": sap_v1_path,
        "target_definition_sha": target_definition,
        "D1_gate_sha": d1_gate,
        "D4_contract_sha": d4_contract,
        "R4_source_or_generator_freeze_sha": r4_manifest_path,
    }
    candidate_bindings = {key: sha(path) for key, path in artifact_paths.items()}
    candidate_bindings["fallback_id"] = "D4_DEVELOPMENT_BALANCE_FALLBACK_V1"
    candidates = {}
    for candidate in ("RBR_A", "RBR_B", "RBR_C"):
        candidates[candidate] = {
            "authorization": "NOT_AUTHORIZED",
            "required_bindings": candidate_bindings,
            "missing_required_sha": [],
            "r4_source_acquisition_choice_status": "NOT_YET_FROZEN",
            "scientific_execution_decision_records": "NOT_YET_FORMED",
            "candidate_specific_activation_gates": "NOT_EXECUTED",
            "authorization_rationale": "Protocol is frozen, but R0 scientific execution/decision records, candidate-specific activation gates, and exact R4 source acquisition choice are not complete. Any missing required SHA would also require NOT_AUTHORIZED.",
        }
    training_path = manifest_dir / "r0_training_authorization_manifest_v1.0.json"
    write_json(training_path, {
        "schema_version": "r0_training_authorization_manifest_v1.0",
        "status": "R0_PROTOCOL_FROZEN_RBR_FORMAL_TRAINING_NOT_AUTHORIZED",
        "RBR_A_TRAINING_AUTHORIZATION": "NOT_AUTHORIZED",
        "RBR_B_TRAINING_AUTHORIZATION": "NOT_AUTHORIZED",
        "RBR_C_TRAINING_AUTHORIZATION": "NOT_AUTHORIZED",
        "automatic_not_authorized_rule": "if any required SHA is absent, mismatched, or unverifiable, authorization is NOT_AUTHORIZED",
        "candidates": candidates,
    })

    # I. Content-only frozen protocol manifest. It intentionally does not bind its own SHA.
    frozen_artifacts = {
        "protocol_v1": protocol_v1_path,
        "sap_v1_markdown": protocol_dir / "R0_Statistical_Analysis_Plan_v1.0.md",
        "sap_v1_json": sap_v1_path,
        "decision_table": decision_path,
        "asset_inventory": asset_inventory,
        "contract_inventory": contract_inventory,
        "split_manifest": split_path,
        "target_definition": target_definition,
        "d0_temporal_policy": d0_policy,
        "mask_padding_policy": mask_policy,
        "d1_gate": d1_gate,
        "d4_family_specific_contract": d4_contract,
        "d4_development_fallback": d4_fallback,
        "r4_reserved_source_or_generator_freeze": r4_manifest_path,
        "scientific_owner_approval": owner_path,
        "training_authorization": training_path,
        "environment_schema": environment_path,
        "command_ledger_schema": command_path,
        "protocol_deviation_log_schema": deviation_path,
    }
    status_lines = git("status", "--porcelain=v1").splitlines()
    frozen_path = manifest_dir / "r0_protocol_frozen_v1.0.json"
    write_json(frozen_path, {
        "schema_version": "r0_protocol_frozen_v1.0",
        "status": "R0_PROTOCOL_V1_0_FROZEN",
        "source_branch": BRANCH,
        "pre_freeze_baseline_commit": BASELINE,
        "worktree": {
            "dirty": bool(status_lines),
            "preexisting_protected_dirty_output": {
                "path": PROTECTED_DIRTY,
                "sha256_at_freeze_generation": sha(REPO / PROTECTED_DIRTY),
                "git_status": "modified_tracked",
                "included_in_freeze_commit": False,
                "exclusion_proof_required_before_commit": "git diff --cached --name-only must not contain this path",
            },
            "untracked_historical_output_count": sum(1 for line in status_lines if line.startswith("?? outputs/")),
        },
        "execution_authorization": "R0_EXECUTION_AUTHORIZED_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
        "protocol_definition_blocker": False,
        "r0_audit_holdout": "NOT_AVAILABLE",
        "evidence_level": "DEVELOPMENT_DIAGNOSTIC_EVIDENCE",
        "r4_source_or_generator": "FROZEN",
        "r4_confirmation_roster": "NOT_FROZEN_BY_DESIGN",
        "artifact_sha256": {key: {"path": rel(path), "sha256": sha(path)} for key, path in frozen_artifacts.items()},
        "phase_b_binding": "r0_v1_freeze_binding.json is deliberately created after the Phase-A content commit to avoid self-referential Git binding",
    })

    print(json.dumps({
        "phase": "A_CONTENT_GENERATED", "baseline": BASELINE,
        "protocol": {"path": rel(protocol_v1_path), "sha256": sha(protocol_v1_path)},
        "sap": {"path": rel(sap_v1_path), "sha256": sha(sap_v1_path)},
        "decision_table": {"path": rel(decision_path), "sha256": sha(decision_path)},
        "split_manifest": {"path": rel(split_path), "sha256": sha(split_path)},
        "training_authorization": {"path": rel(training_path), "sha256": sha(training_path)},
        "protocol_frozen": {"path": rel(frozen_path), "sha256": sha(frozen_path)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
