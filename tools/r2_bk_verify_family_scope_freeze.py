#!/usr/bin/env python3
"""Verify the zero-run R2-BK family-scope and TSB-only design freeze."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
EXPECTED = {
    "docs/stageR/r2/r2_bj_b1_1_offline_recovery_analyzer_complete_result_v1.0.json": "761532070a61dc744e742c212c72c24e3d36bc8fee12b44805f63e41824e16ed",
    "docs/stageR/r2/r2_bj_b1_1_post_recovery_sha_binding_manifest_v1.0.json": "78d480e4ec76b3b00d01d1e17fe8c634131d1c3cab8cfae1c801a7f0bd1ef3ab",
    "docs/stageR/r2/r2_bh_tsb_family_development_candidate_v1.0.json": "7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538",
    "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json": "1833b245e2b2f74bc19aad7013f6339f554d9d700cc3077151ab0474169c716d",
    "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_results_v1.0.json": "6f561df652867aee2204026b2390cacc5e075e8f75c393b880e6cbd21902ca88",
    "tools/r2_b_controller_aware_generator_v1.py": "d166c4746d0a70668b8e26a532890c641b5166f55661a2d34d7f796a9830e3eb",
    "docs/stageR/r2/r2_b_generator_binding_manifest_v1.0.json": "fc2cbe9823bb91c486b6def490e4ce48a82601b9244f3aaad3fd5b29e8b8f1b0",
    "docs/stageR/r2/r2_bh_hlc_arch_development_binding_manifest_v1.0.json": "dbcf129b379c18ccf18aaa78f8cce3ad36af0c59c1f7458a6d018710292253bc",
    "docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json": "414d04cfe9c440125f37d031ff83eb57b9982ba4b9a158ee0dbff995804dfd8e",
}
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def verify_manifest_components(path: Path) -> int:
    manifest = read_json(path)
    components = manifest.get("components", [])
    for component in components:
        target = ROOT / str(component["path"])
        if not target.is_file() or sha256(target) != str(component["sha256"]):
            raise RuntimeError(f"TRANSITIVE_COMPONENT_SHA_MISMATCH:{component['path']}")
    return len(components)


def verify() -> Dict[str, Any]:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"FROZEN_INPUT_SHA_MISMATCH:{relative}")
    if sha256(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("PROTECTED_CSV_SHA_MISMATCH")

    recovery = read_json(R2 / "r2_bj_b1_1_offline_recovery_analyzer_complete_result_v1.0.json")
    gates = recovery["recovery_analyzer_result"]["gates"]
    expected_gates = {
        "mechanism_pass": False, "endpoint_pass": False, "F_match_pass": True,
        "engineering_pass": True, "official_safety_pass": False,
        "actual_shadow_observability_pass": True,
        "treatment_target_offset_declines": True, "post_deadline_hard_jump_absent": True,
    }
    if gates != expected_gates:
        raise RuntimeError("HLC_FROZEN_GATE_DISPOSITION_MISMATCH")

    candidate = read_json(R2 / "r2_bh_tsb_family_development_candidate_v1.0.json")
    counts = candidate["DEV_CAL_result_summary"]["counts"]
    required_counts = ("mechanism_pass", "F_match_pass", "safety_pass", "measurement_OK", "baseline_one_phase", "treatment_two_phase")
    if candidate["status"] != "TSB_FAMILY_DEVELOPMENT_CANDIDATE_FROZEN" or any(int(counts[key]) != 8 for key in required_counts):
        raise RuntimeError("TSB_FROZEN_CANDIDATE_RESULT_MISMATCH")
    margins = candidate["DEV_CAL_result_summary"]["mechanism_margin_distributions"]
    if float(margins["release_fraction_minus_0p15"]["min"]) != 0.588132:
        raise RuntimeError("TSB_RELEASE_MARGIN_MISMATCH")
    if float(margins["second_peak_ratio_minus_0p50"]["min"]) != 0.70024:
        raise RuntimeError("TSB_SECOND_PEAK_MARGIN_MISMATCH")

    r2b_components = verify_manifest_components(R2 / "r2_b_generator_binding_manifest_v1.0.json")
    r2bh_components = verify_manifest_components(R2 / "r2_bh_hlc_arch_development_binding_manifest_v1.0.json")
    if (r2b_components, r2bh_components) != (39, 33):
        raise RuntimeError("TSB_TRANSITIVE_COMPONENT_COUNT_MISMATCH")

    bifurcation = read_json(R2 / "r2_bk_family_scope_bifurcation_contract_v1.0.json")
    capacity = read_json(R2 / "r2_bk_tsb_r2c_prospective_eligible_capacity_census_v1.0.json")
    firewall = read_json(R2 / "r2_bk_data_firewall_audit_v1.0.json")
    if bifurcation["claims"]["CROSS_FAMILY_POOLING"] != "PROHIBITED":
        raise RuntimeError("CROSS_FAMILY_POOLING_NOT_PROHIBITED")
    if capacity["eligible_pool_capacity"]["complete_census"] is not False or capacity["roster_selected"] is not False:
        raise RuntimeError("CAPACITY_FAIL_CLOSED_OR_NO_ROSTER_STATE_MISMATCH")
    if any(int(firewall[key]) != 0 for key in ("runner_run", "engineering_simulation", "scientific_simulation", "TSB_simulation")):
        raise RuntimeError("R2_BK_ZERO_RUN_INVARIANT_MISMATCH")
    return {
        "status": "PASS_R2_BK_ZERO_RUN_FREEZE_VERIFICATION",
        "TSB_candidate_integrity": "PASS",
        "transitive_component_SHA_closure": {"R2_B": r2b_components, "R2_BH": r2bh_components},
        "eligible_capacity": "FAIL_CLOSED_NOT_MATERIALIZED",
        "runner_run": 0,
        "offline_recovery_analyzer_invocations": 0,
        "roster_selected": False,
    }


def main() -> int:
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
