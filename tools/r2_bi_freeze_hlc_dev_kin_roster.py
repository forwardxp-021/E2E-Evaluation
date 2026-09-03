#!/usr/bin/env python3
"""Freeze fresh outcome-blind R2-BI HLC DEV-KIN identities after zero-run gates."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_b2_9_d_freeze_pair_bindings import _one as freeze_pair_binding  # noqa: E402
from tools.r1_future_compliant_smoke_selector_v1_3 import canonical_sha  # noqa: E402
from tools.r2_a_freeze_controller_id_design import _select_unique_family_suffix  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
BH_ROSTER = R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json"
BH_EXCLUSION = R2 / "r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json"
ENTRY = R2 / "r2_bi_mandatory_zero_run_entry_gate_audit_v1.json"
SPACE = R2 / "r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json"
CONTRACT = R2 / "r2_bi_hlc_kinematic_capture_architecture_contract_v3.0.json"
TAXONOMY = R2 / "r2_bi_hlc_architecture_failure_taxonomy_v1.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
ROUND_ROOT = R2 / "r2_bi_hlc_dev_kin_rounds"
OUT = {
    "roster": R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json",
    "exclusion": R2 / "r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0.json",
    "pairs": R2 / "r2_bi_hlc_dev_kin_pair_bindings_v1.0.json",
    "ledger": R2 / "r2_bi_hlc_dev_kin_run_ledger_v1.0.json",
    "round0": ROUND_ROOT / "r2_bi_hlc_dev_kin_round_0_parameters_v3.0.json",
}


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _runs(rows: list[Mapping[str, Any]], round_index: int) -> list[Dict[str, Any]]:
    result = []
    for index, row in enumerate(rows, 1):
        pair_id = f"R2BI-DEV-KIN-HLC-{index:02d}"
        for arm in ("BASELINE", "TREATMENT"):
            result.append({
                "run_order": len(result) + 1,
                "run_id": f"R2BI-HLC-R{round_index}-{index:02d}-{arm}",
                "pair_id": pair_id, "family": "R-HLC", "arm": arm,
                "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            })
    return result


def main() -> int:
    if _sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    entry = _read(ENTRY)
    if entry["status"] != "R2_BI_ZERO_RUN_ENTRY_GATES_PASS" or not entry["all_mandatory_gates_pass"]:
        raise PermissionError("R2_BI_SIMULATION_NOT_AUTHORIZED_ENTRY_GATES_NOT_PASS")
    bh_roster, prior = _read(BH_ROSTER), _read(BH_EXCLUSION)
    if prior["counts"]["effective_unique_identities"] != 109:
        raise RuntimeError("R2_BI_EXPECTED_109_PRE_SELECTION_FIREWALL")
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    cutoff = max(row["selector_rank_sha256"] for row in bh_roster["entries"])
    rows, audits, source_audit = _select_unique_family_suffix("R-HLC", cutoff, prior, {}, set(), set())
    for row in rows:
        row.update({
            "PERMANENT_ENGINEERING_ONLY": True, "R2C_USE_FORBIDDEN": True,
            "CONFIRMATORY_USE_FORBIDDEN": True, "RBR_USE_FORBIDDEN": True,
            "selection_role": "OUTCOME_BLIND_R2_BI_HLC_DEV_KIN",
        })
    if len(rows) != 8 or len({row["scenario_token"] for row in rows}) != 8 or len({row["log_id"] for row in rows}) != 8:
        raise RuntimeError("R2_BI_DEV_KIN_CARDINALITY_OR_UNIQUENESS_FAIL")
    old_tokens = {row["scenario_token"] for row in prior["entries"]}
    old_logs = {row["log_id"] for row in prior["entries"]}
    if any(row["scenario_token"] in old_tokens or row["log_id"] in old_logs for row in rows):
        raise RuntimeError("R2_BI_DEV_KIN_FIREWALL_OVERLAP")
    roster = {
        "schema_version": "r2_bi_hlc_dev_kin_roster_v1.0",
        "status": "FROZEN_AFTER_ZERO_RUN_GATE_BEFORE_ANY_R2_BI_OUTCOME",
        "selection_semantics": "CONTINUE_FROZEN_V1_3_HASH_RANK_AFTER_R2_BH_HLC_PREFIX",
        "source_universe": bh_roster["source_universe"],
        "entry_gate": {"path": str(ENTRY.relative_to(ROOT)), "sha256": _sha(ENTRY)},
        "pre_selection_firewall": {"path": str(BH_EXCLUSION.relative_to(ROOT)), "sha256": _sha(BH_EXCLUSION), "count": 109},
        "entries": rows, "count": 8, "candidate_audits": audits, "source_audit": source_audit,
        "allowed_selection_inputs": ["context", "map", "route_reference", "Primary80", "technical_runtime_applicability"],
        "outcome_mechanism_F_match_safety_BDD_representation_used": False,
    }
    new_entries = list(prior["entries"]) + [{
        "scenario_token": row["scenario_token"], "log_id": row["log_id"], "family": "R-HLC",
        "sources": [str(OUT["roster"].relative_to(ROOT))], "reasons": ["R2_BI_HLC_DEV_KIN_IDENTITY"],
        "PERMANENT_ENGINEERING_ONLY": True, "R2C_USE_FORBIDDEN": True,
        "CONFIRMATORY_USE_FORBIDDEN": True, "RBR_USE_FORBIDDEN": True,
    } for row in rows]
    exclusion = {
        "schema_version": "r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_R2_BI_FIREWALL",
        "source_ledger": {"path": str(BH_EXCLUSION.relative_to(ROOT)), "sha256": _sha(BH_EXCLUSION)},
        "entries": new_entries,
        "counts": {"pre_R2_BI": 109, "R2_BI_HLC_DEV_KIN": 8, "effective_unique_identities": 117},
        "entry_removal_or_reduction_allowed": False,
    }
    pair_cache: Dict[str, Any] = {}
    pairs = []
    for index, row in enumerate(rows, 1):
        binding = freeze_pair_binding(row, pair_cache)
        binding["pair_id"] = f"R2BI-DEV-KIN-HLC-{index:02d}"
        binding["future_realized_trace_used"] = False
        binding["future_safety_result_used"] = False
        binding["future_scientific_gate_result_used"] = False
        pairs.append(binding)
    pair_payload = {
        "schema_version": "r2_bi_hlc_dev_kin_pair_bindings_v1.0",
        "status": "FROZEN_8_OF_8_PRE_OUTCOME_BINDINGS_COMPLETE", "pairs": pairs,
        "scientific_measurement_numerics_changed": False,
    }
    parameters = _read(SPACE)["round0"]
    round0 = {
        "schema_version": "r2_bi_hlc_dev_kin_round_0_parameters_v3.0",
        "status": "FROZEN_BEFORE_ROUND0_SIMULATION", "round_index": 0,
        "parameters": parameters, "runs": _runs(rows, 0),
        "global_no_identity_specific_parameters": True,
        "round1_update_source": str(TAXONOMY.relative_to(ROOT)),
    }
    ledger = {
        "schema_version": "r2_bi_hlc_dev_kin_run_ledger_v1.0",
        "status": "FROZEN_PRE_EXECUTION",
        "roster_canonical_sha256": canonical_sha(roster),
        "pair_binding_canonical_sha256": canonical_sha(pair_payload),
        "architecture_contract_sha256": _sha(CONTRACT), "parameter_space_sha256": _sha(SPACE),
        "failure_taxonomy_sha256": _sha(TAXONOMY),
        "round0_parameter_canonical_sha256": canonical_sha(round0),
        "rounds": [], "maximum_rounds": 2,
        "round1_condition": "NO_PRE_REGISTERED_ARCHITECTURE_STOP_AND_ONLY_NUMERICAL_GLOBAL_CALIBRATION_FAILURE",
        "technical_rerun_policy": "FRESH_RUN_ID_AND_ROOT_TECHNICAL_INFRASTRUCTURE_FAILURE_ONLY",
        "identity_replacement": False, "TSB_simulation_calls": 0,
        "scientific_simulation_calls": 0, "R2C_started": False,
        "confirmatory_smoke_started": False, "RBR_started": False,
    }
    for key, value in (("roster", roster), ("exclusion", exclusion), ("pairs", pair_payload), ("round0", round0), ("ledger", ledger)):
        _write_new(OUT[key], value)
    print(json.dumps({"status": roster["status"], "fresh_HLC_DEV_KIN": 8, "overlap": 0, "effective_exclusions": 117}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
