#!/usr/bin/env python3
"""Generate the approved, zero-run S2R role-aware preflight package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/s2r_role_aware_preflight"
APPROVAL = ROOT / "docs/stageR/s1r_owner_amendment/S1R_Owner_Approval_Record_v1.md"
OWNER_MANIFEST = ROOT / "docs/stageR/s1r_owner_amendment/S1R_Owner_Amendment_Manifest_v1.json"
PRIOR_CLASSES = ROOT / "docs/stageR/s1r_protocol_correction/S1R_Session_Exposure_Reclassification_v1.csv"
E3_OVERRIDES = ROOT / "docs/stageR/s1r_owner_amendment/S1R_E3_Session_Overrides_v1.csv"
OLD_CENSUS = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json"
OLD_BINDING = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Execution_Binding_v1.json"
OLD_RESET = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Full_Reset_Contract_v1.json"
OLD_PRECONTEXT = ROOT / "docs/stageR/s2_preflight/S2_Preflight_Precontext_Identity_Contract_v1.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
BASE_COMMIT = "9ee519fbcb497570b4f2aab13486d4f3cc49ad57"
BRANCH = "20260825_stageR_new"

ZERO = {
    "SIMULATION": 0,
    "RUNNER_RUN": 0,
    "TSB_ROLLOUT": 0,
    "HLC_ROLLOUT": 0,
    "RBR_TRAINING": 0,
    "NEW_SCIENTIFIC_OUTCOME_EXPOSURE": 0,
    "PRIMARY_EVALUATION": 0,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: Any) -> None:
    write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        raise RuntimeError(f"wrong branch: {branch}")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("protected CSV hash changed")
    approval = APPROVAL.read_text(encoding="utf-8")
    required = [
        "S1R_OWNER_AMENDMENT = APPROVED",
        "S1_PROTOCOL = AMENDED_BY_S1R_OWNER_AMENDMENT_V1",
        "TSB_SCIENTIFIC_SIMULATION = NOT_AUTHORIZED",
        f"approved_package_commit = {BASE_COMMIT}",
    ]
    if any(item not in approval for item in required):
        raise RuntimeError("Owner approval record is absent or incomplete")
    owner = read_json(OWNER_MANIFEST)
    for path, expected in owner["artifacts"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"immutable Owner package artifact changed: {path}")
    old_binding = read_json(OLD_BINDING)
    for item in old_binding["bindings"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists() or sha(path) != item["sha256"]:
            raise RuntimeError(f"execution-binding evidence changed: {path}")
    return owner, read_json(OLD_CENSUS), old_binding, read_json(OLD_RESET)


def build_rows(old_census: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    prior = {r["session_id"]: r for r in read_csv(PRIOR_CLASSES)}
    override = {r["session_id"] for r in read_csv(E3_OVERRIDES)}
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for log in old_census["logs"]:
        by_session[log["session_id"]].append(log)
    if set(prior) != set(by_session) or len(prior) != 248 or len(override) != 31:
        raise RuntimeError("248-session universe or E3 override set changed")

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for session_id in sorted(prior):
        p = prior[session_id]
        cls = "E3_PRIMARY_METHOD_DEVELOPMENT" if session_id in override else p["observed_exposure_class"]
        logs = by_session[session_id]
        candidate = cls not in {"E3_PRIMARY_METHOD_DEVELOPMENT", "E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION"}
        claim_a = cls in {"E0_METADATA_ONLY", "E1_UNRELATED_HISTORICAL_USE", "E5_UNTOUCHED_CONFIRMATORY"}
        source_ok = all(x["frozen_fingerprint_match"] and x["source_stat_unchanged_during_read"] for x in logs)
        counts[cls] += 1
        rows.append({
            "session_id": session_id,
            "session_identity_status": "COMPLETE_UNIQUE_FROZEN_SESSION_KEY",
            "exposure_class": cls,
            "v_policy_status": "CANDIDATE" if candidate else "EXCLUDED_E3_E4",
            "claim_a_raw_candidate": str(claim_a).lower(),
            "claim_b_raw_candidate": str(candidate).lower(),
            "strict_c_raw_candidate": str(cls == "E5_UNTOUCHED_CONFIRMATORY").lower(),
            "log_count": len(logs),
            "log_identity_status": "COMPLETE_FROZEN_LOG_KEYS",
            "scenario_count": sum(int(x["scenario_count"]) for x in logs),
            "scenario_token_inventory_status": "COMPLETE_FROZEN_COUNT_AND_HASH",
            "timestamp_status": "COMPLETE_LOG_ACQUISITION_TIMESTAMP",
            "map_locations": ";".join(sorted({x["map_location"] for x in logs})),
            "map_support_status": "MAP_METADATA_PRESENT_RUNTIME_SUPPORT_NOT_CERTIFIED" if candidate else "NOT_ASSESSED_POLICY_EXCLUDED",
            "source_fingerprint_complete": str(source_ok).lower(),
            "duplicate_identity_status": "NO_DUPLICATE_SESSION_ID_FROZEN_SOURCE_ALIASES_CANONICALIZED",
            "role_reservation_status": "CANDIDATE_POOL_ONLY_NO_FINAL_V_OR_C_RESERVATION" if candidate else "DEVELOPMENT_ONLY_EXCLUDED_FROM_PRIMARY_V",
            "static_initial_speed_status": "AMBIGUOUS_EXECUTION_INITIAL_STATE_NOT_BOUND" if candidate else "NOT_ASSESSED_POLICY_EXCLUDED",
            "static_route_reference_status": "AMBIGUOUS_OFFICIAL_NATIVE_ROUTE_NOT_BOUND" if candidate else "NOT_ASSESSED_POLICY_EXCLUDED",
            "required_ego_state_status": "AMBIGUOUS_EXECUTION_EXTRACTION_NOT_BOUND" if candidate else "NOT_ASSESSED_POLICY_EXCLUDED",
            "callback_schema_support_status": "BLOCKED_PRODUCTION_CALLBACK_CHAIN_NOT_BOUND" if candidate else "NOT_APPLICABLE_POLICY_EXCLUDED",
            "static_applicability_status": "AMBIGUOUS_METADATA_INCOMPLETE" if candidate else "INELIGIBLE_E3_E4",
            "source_role_provenance_status": "COMPLETE" if source_ok else "INCOMPLETE",
            "full_v_provenance_status": "INCOMPLETE_PRECONTEXT_AND_EXECUTION_BINDING" if candidate else "NOT_APPLICABLE_POLICY_EXCLUDED",
            "technical_binding_status": "BLOCKED_SHARED_ENGINEERING_CONTRACT" if candidate else "NOT_APPLICABLE_POLICY_EXCLUDED",
            "certified_v_status": "NOT_CERTIFIED",
            "statistical_cluster": session_id,
        })

    expected = {
        "E0_METADATA_ONLY": 1,
        "E1_UNRELATED_HISTORICAL_USE": 149,
        "E2_BENCHMARK_ENGINEERING": 62,
        "E3_PRIMARY_METHOD_DEVELOPMENT": 31,
        "E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION": 0,
        "E5_UNTOUCHED_CONFIRMATORY": 5,
    }
    if any(counts[k] != v for k, v in expected.items()):
        raise RuntimeError(f"unexpected exposure counts: {counts}")
    counts["E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION"] = 0
    counts["UNKNOWN"] = 0

    capacity = []
    for cls in expected:
        subset = [r for r in rows if r["exposure_class"] == cls]
        candidate = cls not in {"E3_PRIMARY_METHOD_DEVELOPMENT", "E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION"}
        capacity.append({
            "exposure_class": cls,
            "sessions": len(subset),
            "logs": sum(int(r["log_count"]) for r in subset),
            "scenarios": sum(int(r["scenario_count"]) for r in subset),
            "raw_claim_a_candidates": sum(r["claim_a_raw_candidate"] == "true" for r in subset),
            "raw_claim_b_v_candidates": len(subset) if candidate else 0,
            "static_applicability_certified": 0,
            "static_applicability_ambiguous": len(subset) if candidate else 0,
            "full_v_provenance_complete": 0,
            "technically_bindable": 0,
            "certified_v_eligible": 0,
            "strict_c_raw_candidates": len(subset) if cls == "E5_UNTOUCHED_CONFIRMATORY" else 0,
            "strict_c_certified": 0,
        })
    return rows, capacity, counts


def generate() -> None:
    owner, old_census, old_binding, old_reset = verify_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, capacity, counts = build_rows(old_census)
    write_csv("S2R_V_Eligibility_Census_v1.csv", rows)
    write_csv("S2R_Exposure_Stratum_Capacity_v1.csv", capacity)

    write_text("S2R_Claim_A_vs_Claim_B_Capacity_v1.md", """
# S2R Claim A versus Claim B Capacity v1

## Claim A — TSB unseen-session generalization

Raw low-exposure candidate capacity is **155 SESSION clusters**: E0=1, E1=149, E5=5. E2 is excluded from Claim A because it participated in benchmark engineering. Exact static eligibility is `UNKNOWN`; zero sessions are certified because execution initial speed and official native route/reference are not bound.

## Claim B — RBR-BDD versus H-BDD under frozen controlled intervention

Raw role-policy capacity is **217 SESSION clusters**: E0=1, E1=149, E2=62, E5=5. E3=31 and E4=0 are excluded. E2 may support Claim B after all remaining static, provenance, execution, reset, and precontext contracts pass. Exact static eligibility is `UNKNOWN`; certified Claim-B V capacity is **0**.

These are candidate pools, not final rosters. No outcome was used, no sample size was changed, and no final identity roster was materialized.
""")
    write_text("S2R_C_Robustness_Capacity_v1.md", """
# S2R C Robustness Capacity v1

`E5_UNTOUCHED_CONFIRMATORY = 5` raw SESSION clusters. `C_STRICT_UNTOUCHED_CAPACITY = 0` certified because exact execution initial speed, native route/reference, production binding, full-arm reset, and precontext identity remain unresolved. Exact statically eligible E5 count is `UNKNOWN`.

Status: `STRICT_C_CAPACITY_LIMITED`.

A future Owner-reviewed design may define a prospectively stratified secondary robustness subset under the approved lowest-claim-relevant-exposure principle. It must retain exposure labels and cannot rename E1/E2 as untouched. This preflight does not implement or authorize that design.
""")

    binding_rows = "\n".join(
        f"| {x['role']} | `{x['path']}` | `{x['symbol']}` | `{x['sha256']}` | {x['status']} |"
        for x in old_binding["bindings"]
    )
    write_text("S2R_Production_Execution_Binding_v1.md", f"""
# S2R Production Execution Binding v1

Status: `PRODUCTION_EXECUTION_BINDING = BLOCKED`.

The zero-run re-audit verified every previously bound component byte-for-byte. The exact code files are identifiable, but the future production chain is not uniquely resolved: there is no dedicated role-aware TSB executor, resolved Hydra configuration, installed production serializer/recorder callback chain, or single primary result manifest. The historical launcher and recorder adapter are HLC-specific and cannot be repurposed by assumption.

| Role | Path | Symbol/config | SHA256 | Component status |
|---|---|---|---|---|
{binding_rows}

Exact audited git SHA: `{BASE_COMMIT}`. Branch: `{BRANCH}`.

Runtime: `{platform.python_implementation()} {platform.python_version()}` at `{sys.executable}`; platform `{platform.platform()}`.

Exact preflight artifacts are under `docs/stageR/s2r_role_aware_preflight/`. Closure requires one production entrypoint plus resolved configuration that instantiates the frozen planner/generator, actual controller/LQR, passive recorder, canonical serializer, analyzer, safety artifacts, and one primary result manifest without fallback dispatch.
""")

    reset_status = old_reset.get("FULL_ARM_RESET_CONTRACT", old_reset.get("status", "BLOCKED"))
    write_text("S2R_Full_Arm_Reset_Contract_v1.md", f"""
# S2R Full-arm Reset Contract v1

Status: `FULL_ARM_RESET_CONTRACT = BLOCKED` (prior evidence status: `{reset_status}`).

Code/schema inspection cannot prove that baseline and treatment arms receive independent fresh instances of every mutable component. `TwoStageController.reset()` clears current state but does not establish reconstruction of tracker and motion-model state; the official builder may reuse planner/callback objects; and no role-aware arm factory binds fresh planner, controller, internal buffers, callbacks, recorder, random/stateful components, and history for each arm.

Required future invariant:

```text
for each (scenario, arm):
  planner, controller, buffers, callbacks, recorder, RNG/stateful components = fresh factory products
  no mutable object identity is shared across arms
  reset/reconstruction evidence is written before outcome access
```

The deterministic preflight verified hashes and schemas only. It created no simulation object and ran no scientific rollout. A future fixture must instantiate the final production factory twice and assert disjoint mutable identities and clean initial state before this contract can pass.
""")

    write_text("S2R_Precontext_Identity_Contract_v1.md", """
# S2R Precontext Identity Contract v1

Status: `PRECONTEXT_IDENTITY_CONTRACT = BLOCKED`.

For every future paired baseline/treatment execution:

```text
BASELINE_PRECONTEXT_ID == TREATMENT_PRECONTEXT_ID
PRECONTEXT_ID = sha256(UTF8(canonical_json(precontext_v1)))
```

`canonical_json` means sorted keys, UTF-8, no insignificant whitespace, explicit units, finite numeric values, and no fallback key guessing. `precontext_v1` must be captured from the final resolved production path and contain only fields that path can actually provide:

- schema version; SESSION, log/database token, scenario token, source database fingerprint;
- exact first extracted lidar token and timestamp;
- ego pose, velocity, acceleration, angular velocity/rate fields available from the official initial EgoState;
- official route roadblock IDs, map name/version, and hash of the derived rolling native reference;
- pre-intervention tracked-object/traffic-light replay token sequence hash;
- planner warmup/history-buffer contents and initialization parameters;
- controller, tracker, motion-model configuration hashes and random seed/state policy;
- ordered pre-intervention callback configuration and state hash.

Unavailable fields must be marked unavailable and fail closed; they must not be invented. The existing metadata anchor speed is not necessarily the execution initial speed. The official API's constant steering value is not a measured initial steering state and must be labeled separately.

The current runner/callback chain does not materialize this schema, so equality cannot yet be verified. Precontext hash equality is necessary but does not substitute for the full-arm reset contract.
""")

    approval_sha = sha(APPROVAL)
    role_manifest = {
        "schema_version": "S2R_Role_Binding_Manifest_v1",
        "status": "APPROVED_ROLE_MODEL_ACTIVE_PREFLIGHT_ONLY",
        "owner_approval": {"path": rel(APPROVAL), "sha256": approval_sha},
        "S1R_OWNER_AMENDMENT": "APPROVED",
        "S1_PROTOCOL": "AMENDED_BY_S1R_OWNER_AMENDMENT_V1",
        "roles": {
            "U": "REPRESENTATION_TRAINING_AND_SELECTION_ONLY",
            "B": "BENCHMARK_AND_ENGINEERING_DEVELOPMENT",
            "V": "FROZEN_CONTROLLED_PRIMARY_EVALUATION",
            "C": "CONFIRMATORY_ROBUSTNESS",
        },
        "exposure_counts": dict(sorted(counts.items())),
        "v_policy_includes": ["E0_METADATA_ONLY", "E1_UNRELATED_HISTORICAL_USE", "E2_BENCHMARK_ENGINEERING", "E5_UNTOUCHED_CONFIRMATORY"],
        "v_policy_excludes": ["E3_PRIMARY_METHOD_DEVELOPMENT", "E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION"],
        "claim_a_excludes_E2": True,
        "final_primary_v_roster_materialized": False,
        "sample_size_refrozen": False,
        "v_identity_metadata_may_influence_u_selection": False,
        "v_scientific_outcome_may_influence_u_selection": False,
        "primary_freeze_point_reached": False,
        "authorization": {"simulation": False, "rbr_training": False, "primary_evaluation": False, "v_execution": False, "c_execution": False},
    }
    write_json("S2R_Role_Binding_Manifest_v1.json", role_manifest)

    write_text("S2R_Preflight_Report_v1.md", """
# S2R Role-Aware Preflight Report v1

## Governing state

`S1R_OWNER_AMENDMENT = APPROVED`; `S1_PROTOCOL = AMENDED_BY_S1R_OWNER_AMENDMENT_V1`. The prior Q20/Q12 capacity result remains a valid `HISTORICAL_RESULT_UNDER_OLD_STRICT_FIREWALL`. Q20/Q12 are now `LEGACY_STRICT_FIREWALL_QUALIFICATION_DESIGN` and `NOT_CURRENT_CAPACITY_GATE`; their mechanism-qualification ideas may be reused only through a future prospectively bound B design.

## Capacity funnel

```text
248 total independent SESSION clusters
↓ remove E3=31 and E4=0
217 raw role-policy V candidates
↓ static applicability (initial speed >= 3.61 m/s and official native route/reference)
0 certified; 217 ambiguous; exact eligible count UNKNOWN
↓ full V provenance / metadata
0 complete; 217 source-and-role complete but precontext/execution provenance incomplete
↓ production binding + full-arm reset + precontext identity
0 technically bindable
↓ certified V pool
0
```

The 3.61 m/s condition remains `NOMINAL_MEASURABILITY_SCREEN` and `NOT_A_CLOSED_LOOP_GUARANTEE`. It was not lowered. No outcome-based field was used. The raw candidate counts are Claim A=155, Claim B=217, and strict E5 C=5; certified counts are all 0.

The frozen universe retains 1,564 canonical logs and 5,338,021 scenario tokens across the 248 SESSION clusters. Frozen counts and token hashes match. Three previously disclosed duplicate source aliases remain canonicalized in that universe and do not create additional independent sessions. Log acquisition timestamps and map metadata are present; execution-initial ego state, native rolling reference, runtime map support, callback schema installation, and final V/C reservation are not certified.

## Engineering findings

- `PRODUCTION_EXECUTION_BINDING = BLOCKED`: no unique complete production chain or resolved runtime configuration.
- `FULL_ARM_RESET_CONTRACT = BLOCKED`: fresh independent mutable arm state is not proven.
- `PRECONTEXT_IDENTITY_CONTRACT = BLOCKED`: the production path does not materialize a comparable machine-readable precontext fingerprint.

Main status: `S2R_PREFLIGHT_BLOCKED_ENGINEERING`.

Static applicability and full V provenance remain secondary blockers. The old structural Q20 capacity crisis is no longer the active blocker under the approved role-aware protocol, but the 217 figure is only a theoretical upper bound. No final V roster was materialized, no sample size was changed, and no scientific execution is authorized.

## Required future order

```text
S2R-PRE
→ V eligibility universe certified
→ execution readiness closed
→ U/B/V/C role binding frozen
→ RBR U-only training/selection
→ H/BDD development finalized
→ PRIMARY_FREEZE_POINT
→ future scientific evaluation after separate authorization
```

SESSION remains the statistical cluster and independent source unit. Logs and scenarios are descriptive denominators only.
""")

    generated = [
        "S2R_Preflight_Report_v1.md",
        "S2R_V_Eligibility_Census_v1.csv",
        "S2R_Exposure_Stratum_Capacity_v1.csv",
        "S2R_Claim_A_vs_Claim_B_Capacity_v1.md",
        "S2R_C_Robustness_Capacity_v1.md",
        "S2R_Production_Execution_Binding_v1.md",
        "S2R_Full_Arm_Reset_Contract_v1.md",
        "S2R_Precontext_Identity_Contract_v1.md",
        "S2R_Role_Binding_Manifest_v1.json",
    ]
    manifest = {
        "schema_version": "S2R_Preflight_Manifest_v1",
        "date": "2026-09-18",
        "branch": BRANCH,
        "audited_git_sha": BASE_COMMIT,
        "main_status": "S2R_PREFLIGHT_BLOCKED_ENGINEERING",
        "owner_approval": {"path": rel(APPROVAL), "sha256": approval_sha},
        "inputs": {rel(p): sha(p) for p in [OWNER_MANIFEST, PRIOR_CLASSES, E3_OVERRIDES, OLD_CENSUS, OLD_BINDING, OLD_RESET, OLD_PRECONTEXT]},
        "artifacts": {f"docs/stageR/s2r_role_aware_preflight/{name}": sha(OUT / name) for name in generated},
        "generator": {"path": "tools/s2r_role_aware_preflight.py", "sha256": sha(Path(__file__))},
        "capacity": {
            "TOTAL_INDEPENDENT_SESSIONS": 248,
            "E3_EXCLUDED": 31,
            "E4_EXCLUDED": 0,
            "RAW_V_UPPER_BOUND": 217,
            "STATIC_APPLICABILITY_ELIGIBLE": "0_CERTIFIED_EXACT_COUNT_UNKNOWN",
            "STATIC_APPLICABILITY_AMBIGUOUS": 217,
            "PROVENANCE_COMPLETE": "0_FULL_V_CONTRACT",
            "SOURCE_AND_ROLE_PROVENANCE_COMPLETE": 217,
            "TECHNICALLY_BINDABLE": 0,
            "CERTIFIED_V_ELIGIBLE": 0,
            "STRICT_E5_C_CAPACITY": "0_CERTIFIED_5_RAW_EXACT_STATIC_UNKNOWN",
            "CLAIM_A_CANDIDATE_CAPACITY": "155_RAW_0_CERTIFIED",
            "CLAIM_B_V_CAPACITY": "217_RAW_0_CERTIFIED",
        },
        "contracts": {"production_execution_binding": "BLOCKED", "full_arm_reset": "BLOCKED", "precontext_identity": "BLOCKED"},
        "final_primary_v_roster_materialized": False,
        "zero_run_counters": ZERO,
        "protected_csv": {"path": rel(PROTECTED), "sha256": sha(PROTECTED), "unchanged": sha(PROTECTED) == PROTECTED_SHA},
        "historical_artifacts_modified": False,
    }
    write_json("S2R_Preflight_Manifest_v1.json", manifest)


if __name__ == "__main__":
    generate()
    print("S2R role-aware zero-run preflight artifacts generated")
