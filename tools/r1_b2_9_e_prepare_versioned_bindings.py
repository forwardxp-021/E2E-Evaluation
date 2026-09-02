#!/usr/bin/env python3
"""Mechanically version B2.9-D schedule/pair bindings without scientific reselection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
OLD_SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v3.0.json"
NEW_SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v3.1.json"
OLD_PAIRS = R1 / "r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json"
NEW_PAIRS = R1 / "r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1.json"
SCHEDULE_AUDIT = R1 / "r1_b2_9_e_schedule_v3_0_to_v3_1_parity_audit_v1.json"
PAIR_AUDIT = R1 / "r1_b2_9_e_pair_binding_v2_0_to_v2_1_parity_audit_v1.json"
ROSTER_SHA = "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def version_id(value: str) -> str:
    if not value.startswith("R1B29D-"):
        raise ValueError(f"OLD_NAMESPACE_REQUIRED:{value}")
    return "R1B29E-" + value[len("R1B29D-") :]


def schedule_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(row[key])
        for key in ("family", "scenario_token", "log_id", "arm", "run_order")
    }


def pair_scientific_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"pair_id", "baseline_run_id", "treatment_run_id", "schedule_rows"}
    return {key: copy.deepcopy(value) for key, value in row.items() if key not in excluded}


def main() -> int:
    if sha256(ROSTER) != ROSTER_SHA:
        raise ValueError("IMMUTABLE_ROSTER_V3_SHA_MISMATCH")
    old_schedule = read_json(OLD_SCHEDULE)
    old_pairs = read_json(OLD_PAIRS)
    new_schedule = copy.deepcopy(old_schedule)
    new_schedule["schema_version"] = "r1_official_compliant_technical_smoke_schedule_v3.1"
    new_schedule["status"] = "FROZEN_NEW_VERSIONED_EXECUTION_PACKAGE_NOT_AUTHORIZED"
    new_schedule["predecessor_attempt"] = "B2_9_D_STOPPED_TECHNICAL_FAILURE"
    new_schedule["old_attempts_consumed"] = 2
    new_schedule["OFFICIAL_SMOKE_AUTHORIZED"] = False
    new_schedule["NEW_RUN_BUDGET"] = 0
    new_schedule["simulation_launched"] = False
    for row in new_schedule["runs"]:
        row["run_id"] = version_id(str(row["run_id"]))
        row["pair_id"] = version_id(str(row["pair_id"]))
    old_runs = sorted(old_schedule["runs"], key=lambda row: int(row["run_order"]))
    new_runs = sorted(new_schedule["runs"], key=lambda row: int(row["run_order"]))
    if len(new_runs) != 48 or [int(row["run_order"]) for row in new_runs] != list(range(1, 49)):
        raise ValueError("NEW_SCHEDULE_48_ORDER_CLOSURE_FAILED")
    parity_rows = []
    for old, new in zip(old_runs, new_runs):
        semantic_equal = schedule_projection(old) == schedule_projection(new)
        namespace_only = new["run_id"] == version_id(old["run_id"]) and new["pair_id"] == version_id(old["pair_id"])
        if not semantic_equal or not namespace_only:
            raise ValueError(f"SCHEDULE_PARITY_FAILED:{old['run_order']}")
        parity_rows.append(
            {
                "run_order": int(old["run_order"]),
                "old_run_id": old["run_id"],
                "new_run_id": new["run_id"],
                "old_pair_id": old["pair_id"],
                "new_pair_id": new["pair_id"],
                "scientific_semantics_exact": True,
            }
        )
    write_new(NEW_SCHEDULE, new_schedule)
    schedule_sha = sha256(NEW_SCHEDULE)

    new_pairs = copy.deepcopy(old_pairs)
    new_pairs["schema_version"] = "r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1"
    new_pairs["status"] = "FROZEN_24_OF_24_PRE_OUTCOME_PAIR_BINDINGS_COMPLETE_NEW_PACKAGE"
    new_pairs["schedule_sha256"] = schedule_sha
    new_pairs["package_provenance"] = {
        "predecessor": "B2_9_D_STOPPED_TECHNICAL_FAILURE",
        "mechanical_source_pair_binding_sha256": sha256(OLD_PAIRS),
        "mechanical_versioner_sha256": sha256(Path(__file__)),
        "old_outputs_reused": False,
        "old_scientific_result_reused": False,
    }
    schedule_by_pair: Dict[str, list[Dict[str, Any]]] = {}
    for row in new_runs:
        schedule_by_pair.setdefault(str(row["pair_id"]), []).append(copy.deepcopy(row))
    pair_parity = []
    for old, new in zip(old_pairs["pairs"], new_pairs["pairs"]):
        old_pair_id = str(old["pair_id"])
        new_pair_id = version_id(old_pair_id)
        rows = sorted(schedule_by_pair[new_pair_id], key=lambda row: int(row["run_order"]))
        if len(rows) != 2:
            raise ValueError(f"NEW_PAIR_SCHEDULE_CARDINALITY:{new_pair_id}")
        new["pair_id"] = new_pair_id
        new["baseline_run_id"] = rows[0]["run_id"]
        new["treatment_run_id"] = rows[1]["run_id"]
        new["schedule_rows"] = rows
        semantic_equal = pair_scientific_projection(old) == pair_scientific_projection(new)
        if not semantic_equal:
            raise ValueError(f"PAIR_SCIENTIFIC_SEMANTIC_PARITY_FAILED:{old_pair_id}")
        pair_parity.append(
            {
                "old_pair_id": old_pair_id,
                "new_pair_id": new_pair_id,
                "family": old["family"],
                "scenario_token": old["scenario_token"],
                "log_id": old["log_id"],
                "scientific_semantics_exact": True,
                "allowed_changes": ["pair_id", "baseline_run_id", "treatment_run_id", "schedule_rows"],
            }
        )
    if len(new_pairs["pairs"]) != 24:
        raise ValueError("NEW_PAIR_BINDING_24_CLOSURE_FAILED")
    write_new(NEW_PAIRS, new_pairs)

    write_new(
        SCHEDULE_AUDIT,
        {
            "schema_version": "r1_b2_9_e_schedule_v3_0_to_v3_1_parity_audit_v1",
            "status": "48_OF_48_EXACT_SCIENTIFIC_SEMANTIC_PARITY_RUN_ID_NAMESPACE_ONLY",
            "old_schedule_sha256": sha256(OLD_SCHEDULE),
            "new_schedule_sha256": schedule_sha,
            "rows": parity_rows,
            "counts": {"total": 48, "semantic_exact": 48, "run_order_exact": 48},
            "selector_invoked": False,
            "source_universe_scanned": False,
            "scientific_identity_changed": False,
        },
    )
    write_new(
        PAIR_AUDIT,
        {
            "schema_version": "r1_b2_9_e_pair_binding_v2_0_to_v2_1_parity_audit_v1",
            "status": "24_OF_24_EXACT_SCIENTIFIC_SEMANTIC_PARITY",
            "old_pair_binding_sha256": sha256(OLD_PAIRS),
            "new_pair_binding_sha256": sha256(NEW_PAIRS),
            "pairs": pair_parity,
            "counts": {"total": 24, "scientific_semantics_exact": 24},
            "old_realized_trace_used": False,
            "old_safety_result_used": False,
            "old_scientific_result_used": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "roster_sha256": sha256(ROSTER),
                "schedule_v3_1_sha256": sha256(NEW_SCHEDULE),
                "pair_binding_v2_1_sha256": sha256(NEW_PAIRS),
                "selector_invoked": False,
                "scientific_identity_changed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
