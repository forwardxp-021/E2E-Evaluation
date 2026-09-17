#!/usr/bin/env python3
"""Generate the proposed S1R-A Scientific Owner amendment package.

This is a zero-run governance tool. It verifies bound repository evidence and
creates prospective documents; it does not execute simulation, training, or
scientific evaluation and does not amend frozen S1 by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/s1r_owner_amendment"
S1 = ROOT / "docs/stageR/s1"
S1R = ROOT / "docs/stageR/s1r_protocol_correction"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
M6_METADATA = ROOT / "outputs/stage7e_pdm_v1_balanced50_paired45_context_v1_m3/metadata.csv"
SESSION_RE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs() -> dict[str, Any]:
    from tools.s2_preflight_metadata_census import verify_s1

    verify_s1()
    prior = read_json(S1R / "S1R_Amendment_Manifest_v1.json")
    for relative, expected in prior["artifacts"].items():
        path = ROOT / relative
        if sha256_file(path) != expected:
            raise ValueError(f"S1R artifact changed: {relative}")
    if sha256_file(ROOT / prior["generator"]["path"]) != prior["generator"]["sha256"]:
        raise ValueError("S1R generator changed")
    if sha256_file(PROTECTED) != PROTECTED_SHA:
        raise ValueError("protected CSV hash changed")
    return prior


def e3_reaudit_rows() -> tuple[list[dict[str, str]], dict[str, int]]:
    prior_rows = list(csv.DictReader((S1R / "S1R_Session_Exposure_Reclassification_v1.csv").open(encoding="utf-8")))
    by_session = {row["session_id"]: row for row in prior_rows}
    sessions: set[str] = set()
    with M6_METADATA.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            match = SESSION_RE.match(Path(row["log_name"]).name)
            if match:
                sessions.add(match.group(1))
    if len(sessions) != 31 or not sessions <= set(by_session):
        raise ValueError(f"unexpected M6 method-development sessions: {len(sessions)}")
    rows = []
    for session_id in sorted(sessions):
        prior = by_session[session_id]
        rows.append({
            "session_id": session_id,
            "prior_primary_class": prior["observed_exposure_class"],
            "reaudited_primary_class": "E3_PRIMARY_METHOD_DEVELOPMENT",
            "affected_component": "PAIRED_BDD_ESTIMAND_AND_FIXED_MEDIAN_KERNEL_METHOD_BOUNDARY",
            "decision_influence_evidence": "M6.1 designed after inspecting M3-M5; current 45-pair dataset explicitly method-development-only",
            "future_role_effect": "DEVELOPMENT_ONLY_FOR_AFFECTED_COMPONENT; EXCLUDE_FROM_CORRESPONDING_CONFIRMATORY_CLAIM",
            "source_metadata": str(M6_METADATA.relative_to(ROOT)),
        })
    counts = Counter(row["observed_exposure_class"] for row in prior_rows)
    for row in rows:
        counts[row["prior_primary_class"]] -= 1
    counts["E3_PRIMARY_METHOD_DEVELOPMENT"] = len(rows)
    counts["E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION"] = 0
    counts["UNKNOWN"] = 0
    return rows, dict(sorted((key, value) for key, value in counts.items() if value or key in {"E3_PRIMARY_METHOD_DEVELOPMENT", "E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION", "UNKNOWN"}))


def mapping_rows() -> list[dict[str, str]]:
    fields = (
        "source_contract", "old_text", "new_text", "status", "scientific_rationale",
        "historical_effect", "future_effect", "owner_decision_required",
    )
    values = [
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q = completely fresh TSB qualification", "B benchmark qualification may use transparently labeled development-exposed nuPlan sessions", "REPLACED", "Benchmark validity depends on mechanism and measurement validity; absolute historical freshness is not necessary for Claim B", "Preserve every historical Q result and its original label", "Future benchmark qualification uses B role and exposure ledger", "true"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q PASS automatically becomes D", "Evidence role is assigned prospectively under U/B/V/C and never changes automatically", "REPLACED", "Automatic conversion obscures which decision an outcome influenced", "No historical role is relabeled", "Every future identity receives an explicit purpose and claim binding", "true"),
        ("S1_Data_Firewall_Draft_v0.1.json", "Q can never participate in E", "Benchmark-development exposure does not automatically prohibit V; E3/E4 restrictions remain claim-specific", "REPLACED", "Controlled evaluation validity turns on adaptation of the tested method, not mere prior use", "Old exclusions remain valid under the old firewall", "Role-aware S2R must certify each future V identity", "true"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "E is disjoint from every U/Q/D log and every historical scientific exposure", "C prioritizes low/no relevant exposure; V excludes prohibited claim-relevant contamination", "REPLACED", "Large frozen controlled evaluation and low-exposure robustness answer distinct questions", "242/243/5 capacity findings remain immutable historical results", "Future V and C are separately bound before outcome access", "true"),
        ("S1_Data_Firewall_Draft_v0.1.json", "Encoder cannot use Q/D/E outcomes", "Encoder architecture, objective, checkpoint and seed handling cannot use B/V/C Primary-related outcomes", "PRESERVED_REWORDED", "U-only encoder selection remains the strongest anti-bias firewall", "No effect on historical training records", "Any violation is E3 or E4 and blocks the corresponding claim", "true"),
        ("S1_Handcrafted_Challenger_Contract_Draft_v0.1.md", "No post-E H modification", "No H modification from V/C outcome after PRIMARY_FREEZE_POINT", "PRESERVED_REWORDED", "The prospective timing boundary is retained under the new roles", "No historical H result is changed", "Exact H implementation and preprocessing freeze before V", "true"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "Numeric thresholds freeze on final D calibration before E", "Calibration data and procedure freeze before V/C; V/C never recalibrate the original threshold", "PRESERVED_REWORDED", "Calibration/evaluation separation directly prevents outcome-driven adaptation", "Historical calibration conclusions remain unchanged", "S2R must bind calibration identities and hashes", "true"),
        ("S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md", "Q20/Q12 fresh-session capacity controls entry", "Benchmark sample size and V/C capacity are separate prospective decisions", "REPLACED", "Benchmark engineering size cannot define controlled-evaluation eligibility", "Q20 capacity blocker remains an old-policy result", "B size and V/C size require separate justification", "true"),
        ("S2_Preflight_Readiness_Report_v1.md", "S2_PREFLIGHT_BLOCKED_Q20_CAPACITY", "If approved, label capacity component S2_STRICT_FIREWALL_PREFLIGHT_HISTORICAL_RESULT and run role-aware S2R", "SUPERSEDED_IF_APPROVED", "The old preflight correctly tested a contract that the amendment prospectively replaces", "No deletion or correction of the old report", "Capacity blocker retires only after approval; engineering blockers remain", "true"),
        ("S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md", "Paired/unpaired boundary", "Keep paired controlled attribution and unpaired release-emulation estimands unchanged", "PRESERVED", "Exposure policy does not change statistical estimands", "No historical effect", "Any future change requires a separate Owner amendment", "false"),
    ]
    return [dict(zip(fields, row)) for row in values]


def exposure_policy() -> str:
    return """# S1R Exposure Policy v1 — PROPOSED OWNER AMENDMENT

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
"""


def role_contract() -> str:
    return """# S1R U/B/V/C Data Role Contract v1 — PROPOSED

Status: `PROPOSED_ONLY`; `OWNER_DECISION = PENDING`.

## U — Representation Training

U alone may select RBR architecture, training objective, checkpoint, seed-handling rule and training validation decisions. nuPlan B/V/C Primary-related outcomes MUST NOT select the encoder. U identities, objectives and selection rule must be bound before training.

## B — Benchmark Development

B supports TSB/HLC engineering, controller transfer, mechanism measurement, F_match, safety, applicability and technical lifecycle. Historical reuse is allowed and disclosed. B exposure does not automatically exclude Claim-B V, but it cannot support Claim-A unseen-session generalization.

## V — Controlled Evaluation

V estimates RBR-BDD versus H-BDD utility under a frozen controlled intervention. V begins only after PRIMARY_FREEZE_POINT and role-aware S2R certification. It requires no E3/E4 contamination for the tested claim, a complete preregistered sampling frame, full denominator, no replacement by outcome, no survivor selection and SESSION-aware inference.

## C — Confirmatory Robustness

C tests whether direction and practical magnitude persist in the cleanest available low/no-relevant-exposure subset, preferably E5. C strengthens robustness; it is not the sole scientifically valid evidence. C size, identities and interpretation must be frozen before access, and low power must be reported as uncertainty rather than automatic contradiction.

Roles are purpose bindings, not permanent intrinsic session labels. Reuse requires a new claim-specific ledger entry; rerunning a contaminated session does not erase E3/E4.
"""


def freeze_contract() -> str:
    return """# S1R PRIMARY_FREEZE_POINT v1 — PROPOSED

Status: `PROPOSED_ONLY`; `OWNER_DECISION = PENDING`.

`PRIMARY_FREEZE_POINT` is the final prospective, hash-bound boundary before any V or C outcome access. It must bind:

1. RBR encoder checkpoint, architecture, objective, seed rule and representation interface;
2. exact H feature implementation, preprocessing, normalization and readout;
3. Primary BDD statistic and the fair RBR/H tuning budget;
4. kernel and bandwidth rule;
5. null construction, calibration procedure and calibration identities;
6. nominal FPR, operating point, batch size and drift composition;
7. scenario applicability and eligibility rules;
8. denominator, failure retention, missingness and no-replacement rules;
9. V and C role rules, identity hashes and exposure strata;
10. sample-size, budget ceiling and stopping rule;
11. Primary estimand, success/fail/inconclusive rules and multiplicity family;
12. paired/unpaired design and its claim boundary;
13. SESSION cluster-aware analysis and resampling plan;
14. code/config/environment hashes and authorized execution binding.

After this point, Primary outcomes cannot change any item above. Infrastructure-only repair requires a prospective written classification showing that no Primary scientific value was used. Otherwise the affected result is E4 and cannot serve the same confirmatory claim. No favorable cohort deletion, survivor selection, best-seed rescue, sample extension, alternative metric, feature addition, kernel search or threshold recalibration is permitted.
"""


def reaudit_text(counts: dict[str, int], evidence: dict[str, str]) -> str:
    return f"""# S1R E3/E4 Re-audit v1

Status: `PROPOSED_EVIDENCE_AUDIT`; no scientific execution was performed.

## Determination

`E3_ZERO_CONFIRMED = NO`

`E4_ZERO_CONFIRMED = YES_REPOSITORY_EVIDENCE_SCOPE`

E3 is nonzero because the Stage7 M6 record explicitly states that its 45-pair dataset was used for method development after M3–M5 inspection. Those 45 scenarios map to 31 SESSION clusters. The paired BDD estimand and fixed median-kernel method boundary are retained in the current protocol, so highest relevant risk classification applies. A machine-readable override ledger accompanies this report.

The E4 confirmation is limited to versioned repository evidence. It does not certify unrecorded personal recollection or external decisions. Discovery of a contrary decision-influence record requires immediate reclassification and capacity recomputation.

## H provenance

The frozen H contract defines a deterministic 30D implementation: existing ego13, one frozen F descriptor, fixed time bins, fixed lags, braking-mass summaries, validity flags and cadence diagnostic. It explicitly prohibits data-driven duplicate detection, lag search, feature search and performance-driven weakening. The eight bound R2-B logs were used for TSB mechanism/F_match descriptions and H finiteness, not session-level feature-performance selection. They remain E2 for Claim B.

## BDD, kernel, bandwidth, calibration and readout provenance

The frozen SAP says the Primary metric and operating point are not empirically optimized; uses a fixed MMD/scaler/median-bandwidth rule for both representations; allows zero supervised kernel/readout selection; and states that D may diagnose power but cannot alter the Primary rule. Final numeric calibration has not occurred, V/C identities do not exist, and the current RBR encoder has not been trained or selected.

Historical Stage7 M6 work is explicitly method-development evidence. Although it is a different historical paired analysis, the current S1 preserves the paired fallback and related MMD/kernel method boundary. Its 31 SESSION clusters are therefore E3 for that affected component and cannot serve the corresponding confirmatory claim. Stage7L outcomes remain historical evidence and did not execute the current exact 30D-H versus U-only future-RBR Primary comparison.

## E4 impossibility under current recorded lifecycle

The current exact RBR-vs-H Primary pipeline has never been executed: RBR training, final D calibration, V construction, Primary evaluation and C analysis are all unauthorized/unmaterialized. Therefore no current Primary outcome exists from which an E4 adaptation could have been made.

## Re-audited 248-session counts

| Class | Sessions |
|---|---:|
| E0_METADATA_ONLY | {counts.get('E0_METADATA_ONLY', 0)} |
| E1_UNRELATED_HISTORICAL_USE | {counts.get('E1_UNRELATED_HISTORICAL_USE', 0)} |
| E2_BENCHMARK_ENGINEERING | {counts.get('E2_BENCHMARK_ENGINEERING', 0)} |
| E3_PRIMARY_METHOD_DEVELOPMENT | {counts.get('E3_PRIMARY_METHOD_DEVELOPMENT', 0)} |
| E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION | 0 |
| E5_UNTOUCHED_CONFIRMATORY | {counts.get('E5_UNTOUCHED_CONFIRMATORY', 0)} |
| UNKNOWN | 0 |

These are exposure-policy classes, not V certification. Under the fail-closed rule that excludes E3 from the affected Primary confirmatory pool, `THEORETICAL_V_UPPER_BOUND = 217`; `CERTIFIED_V_CAPACITY = 0`.

## Bound evidence

""" + "\n".join(f"- `{path}` — `{digest}`" for path, digest in evidence.items()) + "\n"


def amendment_text(counts: dict[str, int], base_head: str) -> str:
    return f"""# S1R-A Scientific Owner Amendment v1

Document status: `PROPOSED_OWNER_AMENDMENT`; `OWNER_DECISION = PENDING`.

Decision options, exactly one to be completed by the Scientific Owner:

- `S1R_OWNER_AMENDMENT = APPROVED`
- `S1R_OWNER_AMENDMENT = REJECTED`
- `S1R_OWNER_AMENDMENT = DEFERRED_PENDING_EVIDENCE`

Prepared from repository HEAD `{base_head}`. This document does not approve itself and does not modify the frozen S1 package.

## Scientific basis

nuPlan is the closed-loop controlled validation platform for this claim, not the RBR encoder training corpus. Historical use alone does not bias the current Primary comparison. The prospective exclusion rule is therefore changed from blanket historical-outcome exclusion to `CLAIM_RELEVANT_OUTCOME_DRIVEN_CONTAMINATION`: an outcome contaminates a claim when it influenced the method, statistical rule, operating point, cohort or sample-size decision tested by that claim.

This correction is supported by data-role reasoning and bias-path analysis. It is not justified by the desire to rescue capacity. Applicability, mechanism, safety, technical completeness, denominator, dependence and calibration requirements remain intact.

## Normative amendment

If approved, the U/Q/D/E absolute-fresh role semantics are prospectively superseded by U/B/V/C and the accompanying E0–E5/UNKNOWN exposure policy. Frozen S1 files and all historical reports remain immutable. Claim A, TSB unseen-session generalization, requires truly benchmark-disjoint evidence. Claim B, the Level-2 Primary RBR-BDD versus H-BDD comparison under a frozen intervention, may use E0/E1/E2 sessions after PRIMARY_FREEZE_POINT and role-aware eligibility certification; E3/E4 remain restricted for the affected claim.

## UNCHANGED_SCIENTIFIC_CONTRACTS

- BDD utility remains Primary; H+RBR incremental prediction remains Secondary.
- HLC remains closed by scope after engineering nonconvergence; impossibility is not established.
- TSB remains a frozen development candidate; clean residual and low-order nuisance elimination are not established.
- Encoder architecture/objective/checkpoint/seed selection remains U-only.
- H remains the development-informed strong handcrafted challenger.
- No survivor selection, favorable replacement or post-outcome sample extension.
- SESSION remains the dependence and clustering unit.
- Paired evidence remains controlled attribution/sensitivity; unpaired evidence remains release-emulation distributional evaluation.
- A BDD alarm means behavior change, not automatic degradation, safety failure or release rejection.

## E3/E4 disposition and capacity

Re-audit finds E3={counts.get('E3_PRIMARY_METHOD_DEVELOPMENT', 0)} and confirms E4=0 within repository evidence. Counts are E0={counts.get('E0_METADATA_ONLY', 0)}, E1={counts.get('E1_UNRELATED_HISTORICAL_USE', 0)}, E2={counts.get('E2_BENCHMARK_ENGINEERING', 0)}, E3={counts.get('E3_PRIMARY_METHOD_DEVELOPMENT', 0)}, E4=0, E5={counts.get('E5_UNTOUCHED_CONFIRMATORY', 0)}, UNKNOWN=0. The fail-closed theoretical V upper bound is 217; certified V remains 0. This package does not materialize a roster.

## Historical capacity result

The 242 outcome-exposed, 243 exposure/reservation-conflict, five strict-fresh remaining, and `CAPACITY_RECOVERY_NONE` findings remain correct under the old absolute-fresh firewall. If this amendment is approved, label them `S2_STRICT_FIREWALL_PREFLIGHT_HISTORICAL_RESULT` and retire only their capacity component as an active future blocker. Production execution binding, full-arm reset and precontext identity blockers remain active.

## Prospective effect only

If approved, role-aware S2R metadata/engineering preflight becomes the next stage. Approval does not authorize simulation, RBR training, roster execution, V/C access or Primary evaluation. Those require separate evidence, freeze completion and explicit Scientific Owner authorization.

## Stage route

`S1R-A Owner Amendment → S2R role-aware preflight and engineering remediation → V eligibility certification → Owner authorization → S3 U-only training/freeze and PRIMARY_FREEZE_POINT → controlled V evaluation → C robustness if available → S4 evidence closure`.

## Zero-run declaration

`SIMULATION=0; RUNNER_RUN=0; TSB_ROLLOUT=0; HLC_ROLLOUT=0; RBR_TRAINING=0; NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0; PRIMARY_EVALUATION=0`.
"""


def run(output: Path) -> None:
    if git("rev-parse", "--abbrev-ref", "HEAD") != "20260825_stageR_new":
        raise RuntimeError("unexpected branch")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    prior = verify_inputs()
    counts = prior["session_reclassification"]["classification_counts"]
    expected = {"E0_METADATA_ONLY": 1, "E1_UNRELATED_HISTORICAL_USE": 176, "E2_BENCHMARK_ENGINEERING": 66, "E5_UNTOUCHED_CONFIRMATORY": 5}
    if counts != expected:
        raise ValueError(f"unexpected S1R counts: {counts}")
    e3_rows, reaudited_counts = e3_reaudit_rows()

    evidence_paths = [
        S1 / "S1_Handcrafted_Challenger_Contract_Draft_v0.1.md",
        S1 / "S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md",
        S1 / "S1_Scope_and_Claim_Freeze_Draft_v0.1.md",
        S1 / "S1_Protocol_Design_Report_v0.1.md",
        S1 / "S1_Development_Evidence_Audit_v0.1.json",
        ROOT / "outputs/stage7_m6_1_paired_bdd_method_freeze_v1/milestone6_paired_bdd_summary.json",
        ROOT / "docs/stage7l_e_prospective_bdd_manifest_v1.json",
        ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_results_v1.0.json",
    ]
    evidence = {str(path.relative_to(ROOT)): sha256_file(path) for path in evidence_paths}
    base_head = git("rev-parse", "HEAD")
    paths = {
        "amendment": output / "S1R_Owner_Amendment_v1.md",
        "mapping": output / "S1R_Old_to_New_Contract_Mapping_v1.csv",
        "exposure": output / "S1R_Exposure_Policy_v1.md",
        "roles": output / "S1R_Data_Role_Contract_UBVC_v1.md",
        "freeze": output / "S1R_Primary_Freeze_Point_v1.md",
        "reaudit": output / "S1R_E3_E4_Reaudit_v1.md",
        "e3_overrides": output / "S1R_E3_Session_Overrides_v1.csv",
        "manifest": output / "S1R_Owner_Amendment_Manifest_v1.json",
    }
    write_text(paths["amendment"], amendment_text(reaudited_counts, base_head))
    rows = mapping_rows()
    write_csv(paths["mapping"], rows)
    write_text(paths["exposure"], exposure_policy())
    write_text(paths["roles"], role_contract())
    write_text(paths["freeze"], freeze_contract())
    write_text(paths["reaudit"], reaudit_text(reaudited_counts, evidence))
    write_csv(paths["e3_overrides"], e3_rows)

    frozen_inputs = sorted(S1.glob("S1_*"))
    artifacts = [path for name, path in paths.items() if name != "manifest"]
    payload = {
        "schema_version": "S1R_Owner_Amendment_Manifest_v1",
        "status": "PROPOSED_OWNER_AMENDMENT",
        "OWNER_DECISION": "PENDING",
        "allowed_owner_decisions": ["APPROVED", "REJECTED", "DEFERRED_PENDING_EVIDENCE"],
        "S1_PROTOCOL_AMENDED": False,
        "S2R_AUTHORIZED": False,
        "SIMULATION_AUTHORIZED": False,
        "RBR_TRAINING_AUTHORIZED": False,
        "PRIMARY_EVALUATION_AUTHORIZED": False,
        "branch": "20260825_stageR_new",
        "base_HEAD": base_head,
        "date": "2026-09-17",
        "reaudit": {
            "E3_ZERO_CONFIRMED": "NO",
            "E4_ZERO_CONFIRMED": "YES_REPOSITORY_EVIDENCE_SCOPE",
            "session_counts": reaudited_counts,
            "E3_session_override_count": len(e3_rows),
            "THEORETICAL_V_UPPER_BOUND": 217,
            "CERTIFIED_V_CAPACITY": 0,
            "evidence": evidence,
        },
        "historical_capacity": {
            "label": "S2_STRICT_FIREWALL_PREFLIGHT_HISTORICAL_RESULT",
            "outcome_exposed": 242,
            "exposure_or_reservation_conflict": 243,
            "strict_fresh_remaining": 5,
            "capacity_recovery": "CAPACITY_RECOVERY_NONE",
            "historical_files_modified": False,
        },
        "independent_engineering_blockers": ["PRODUCTION_EXECUTION_BINDING", "FULL_ARM_RESET_CONTRACT", "PRECONTEXT_IDENTITY_CONTRACT"],
        "frozen_S1_inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_inputs},
        "governing_inputs": {
            "v2_3": sha256_file(ROOT / "RBR-64_博士研究总体方案_v2.3.md"),
            "s1r_manifest": sha256_file(S1R / "S1R_Amendment_Manifest_v1.json"),
            "handover0917": sha256_file(ROOT / "handover0917.md"),
            "v2_2": sha256_file(ROOT / "RBR-64_博士研究总体方案_v2.2_S1正式入口版.md"),
            "AGENTS": sha256_file(ROOT / "AGENTS.md"),
        },
        "mapping_status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "generator": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256_file(Path(__file__).resolve())},
        "artifacts": {f"docs/stageR/s1r_owner_amendment/{path.name}": sha256_file(path) for path in artifacts},
        "zero_run_counters": {name: 0 for name in ("SIMULATION", "RUNNER_RUN", "TSB_ROLLOUT", "HLC_ROLLOUT", "RBR_TRAINING", "NEW_SCIENTIFIC_OUTCOME_EXPOSURE", "PRIMARY_EVALUATION")},
        "protected_csv": {"path": str(PROTECTED.relative_to(ROOT)), "expected_sha256": PROTECTED_SHA, "sha256": sha256_file(PROTECTED), "unchanged": sha256_file(PROTECTED) == PROTECTED_SHA},
        "prospective_if_approved": {
            "old_capacity_blocker": "HISTORICAL_ONLY",
            "next_stage": "ROLE_AWARE_S2R_PREFLIGHT_AUTHORIZABLE_SEPARATELY",
            "simulation": "NOT_AUTOMATICALLY_AUTHORIZED",
            "RBR_training": "NOT_AUTOMATICALLY_AUTHORIZED",
            "Primary_evaluation": "NOT_AUTOMATICALLY_AUTHORIZED",
        },
    }
    with paths["manifest"].open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": payload["status"], "owner_decision": payload["OWNER_DECISION"], "reaudit": payload["reaudit"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    run(parser.parse_args().output_dir)
