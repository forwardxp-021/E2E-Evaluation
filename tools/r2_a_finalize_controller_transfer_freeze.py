#!/usr/bin/env python3
"""Close the additive R2-A data firewall and SHA-bind the completed DEV audit.

This utility only reads completed R2-A artifacts and writes small governance/
binding JSON.  It never constructs or runs a simulator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json"
EXCLUSION = R2 / "r2_a_controller_id_permanent_exclusion_ledger_v1.0.json"
EXECUTION = R2 / "r2_a_controller_transfer_execution_audit_v1.0.json"
MANIFEST = R2 / "r2_a_controller_transfer_identification_binding_manifest_v1.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def close_exclusion(roster: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    entries = list(current["entries"])
    pre_selection = [row for row in entries if "R2_A_CONTROLLER_IDENTIFICATION_DEV_IDENTITY" not in row.get("reasons", [])]
    if len(pre_selection) != 69:
        raise RuntimeError(f"R2_A_PRE_SELECTION_FIREWALL_COUNT_MISMATCH:{len(pre_selection)}")
    pre_base = {
        key: value
        for key, value in current.items()
        if key not in {"pre_selection_firewall_canonical_sha256", "selected_DEV_identity_addition_is_post_selection"}
    }
    pre_sha = hashlib.sha256(
        json.dumps(
            {**pre_base, "entries": pre_selection, "counts": {
                "historical_permanent_exclusions": 45,
                "R1_official_outcome_exposed": 24,
                "effective_unique_identities": 69,
            }},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    # The roster binds the exact pre-selection firewall produced before any R2-A run.
    if pre_sha != roster["effective_firewall_canonical_sha256"]:
        raise RuntimeError("R2_A_PRE_SELECTION_FIREWALL_SHA_MISMATCH")
    selected = []
    for row in roster["entries"]:
        selected.append(
            {
                "scenario_token": row["scenario_token"],
                "log_id": row["log_id"],
                "family": row["family"],
                "sources": [str(ROSTER.relative_to(ROOT))],
                "reasons": ["R2_A_CONTROLLER_IDENTIFICATION_DEV_IDENTITY"],
                "OUTCOME_EXPOSED": False,
                "PERMANENT_R2_ENGINEERING_ONLY": True,
                "R2_CONFIRMATORY_USE_FORBIDDEN": True,
                "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
            }
        )
    pairs = [(row["scenario_token"], row["log_id"]) for row in pre_selection + selected]
    if len(pairs) != 85 or len(set(pairs)) != 85:
        raise RuntimeError("R2_A_FINAL_PERMANENT_EXCLUSION_UNIQUENESS_FAIL")
    return {
        **current,
        "pre_selection_firewall_canonical_sha256": pre_sha,
        "entries": pre_selection + selected,
        "counts": {
            "historical_permanent_exclusions": 45,
            "R1_official_outcome_exposed": 24,
            "R2_A_fresh_engineering_only": 16,
            "effective_unique_identities": 85,
        },
        "selected_DEV_identity_addition_is_post_selection": True,
    }


def main() -> int:
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("PROTECTED_CSV_SHA_MISMATCH")
    roster = load(ROSTER)
    execution = load(EXECUTION)
    if execution["status"] != "80_OF_80_FROZEN_DEV_RUNS_TECHNICAL_COMPLETE":
        raise RuntimeError("R2_A_EXECUTION_NOT_COMPLETE")
    exclusion = close_exclusion(roster, load(EXCLUSION))
    dump(EXCLUSION, exclusion)

    component_paths = [
        "configs/r1_official_technical_smoke_hydra/planner/r2_a_controller_transfer_dev_v1.yaml",
        "tools/r2_a_freeze_controller_id_design.py",
        "tools/r2_a_controller_transfer_dev_planner_v1.py",
        "tools/r2_a_execute_controller_transfer_dev.py",
        "tools/r2_a_recover_controller_transfer_dev.py",
        "tools/r2_a_analyze_controller_transfer.py",
        "tools/r2_a_finalize_controller_transfer_freeze.py",
        "tests/test_r2_a_controller_transfer_identification.py",
        "QUICK_REFERENCE.md",
        "docs/stageR/r2/r2_a_scientific_owner_engineering_simulation_authorization_v1.0.json",
        "docs/stageR/r2/r2_a_controller_id_permanent_exclusion_ledger_v1.0.json",
        "docs/stageR/r2/r2_a_controller_id_dev_canary_roster_v1.0.json",
        "docs/stageR/r2/r2_a_hlc_excitation_grid_v1.0.json",
        "docs/stageR/r2/r2_a_tsb_excitation_grid_v1.0.json",
        "docs/stageR/r2/r2_a_controller_id_selection_audit_v1.0.json",
        "docs/stageR/r2/r2_a_controller_transfer_run_ledger_v1.0.json",
        "docs/stageR/r2/r2_a_zero_run_construction_audit_v1.0.json",
        "docs/stageR/r2/r2_a_controller_transfer_execution_audit_v1.0.json",
        "docs/stageR/r2/r2_a_hlc_transfer_identification_v1.json",
        "docs/stageR/r2/r2_a_tsb_transfer_identification_v1.json",
        "docs/stageR/r2/r2_a_controller_transfer_surrogate_v1.json",
        "docs/stageR/r2/R2_A_TSB_Replanning_Transfer_Audit_v1.md",
        "docs/stageR/r2/R2_A_Controller_Transfer_Identification_Report_v1.md",
        "docs/stageR/r2/R2_A_R2B_Generator_Architecture_Decision_v1.md",
    ]
    components = []
    for rel in component_paths:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        components.append({"path": rel, "sha256": sha(path)})

    telemetry = []
    for row in execution["effective_runs"]:
        files = []
        for key in ("trace_path", "planner_telemetry_path", "controller_command_path"):
            rel = row[key]
            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(rel)
            files.append({"role": key, "path": rel, "sha256": sha(path)})
        telemetry.append(
            {
                "frozen_run_id": row["frozen_run_id"],
                "effective_run_id": row["effective_run_id"],
                "family": row["family"],
                "files": files,
            }
        )
    manifest = {
        "schema_version": "r2_a_controller_transfer_identification_binding_manifest_v1.0",
        "status": "R2_A_CONTROLLER_TRANSFER_DIAGNOSTIC_FROZEN_COMPLETE",
        "baseline_remote_commit": "78f3a94c60f6b9571f56974f77e6d9ce285cd51f",
        "components": components,
        "effective_runtime_telemetry": telemetry,
        "counts": {
            "fresh_DEV_identities": 16,
            "frozen_effective_runs": 80,
            "technical_reruns": execution["counts"]["technical_reruns"],
            "actual_engineering_runs": execution["counts"]["actual_engineering_runs"],
            "bound_effective_telemetry_sets": len(telemetry),
        },
        "scientific_threshold_changed": False,
        "final_R2_generator_implemented": False,
        "R2_confirmatory_roster_selected": False,
        "RBR_started": False,
        "protected_CSV_sha256": PROTECTED_SHA,
    }
    dump(MANIFEST, manifest)
    print(json.dumps({"status": manifest["status"], "final_exclusion_count": 85, "telemetry_sets": 80}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
