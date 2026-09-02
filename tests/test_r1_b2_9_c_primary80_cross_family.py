from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from tools.r1_b2_8_r3_prospective_selector import official_env

official_env()

from tests.test_r1_b2_9_b_route_continuous_canary import _paired_map
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3
from tools.r1_closed_loop_benchmark_v2_2 import build_hlc_route_continuous_reference_v2_2
from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1


ROOT = Path(__file__).resolve().parents[1]


def test_primary80_controller_long_exact_and_short_contract() -> None:
    assert R1Primary80ScientificTimeControllerV1(
        SimpleNamespace(get_number_of_iterations=lambda: 200)
    ).number_of_iterations() == 81
    assert R1Primary80ScientificTimeControllerV1(
        SimpleNamespace(get_number_of_iterations=lambda: 81)
    ).number_of_iterations() == 81
    with pytest.raises(ValueError, match="NOT_EVALUABLE"):
        R1Primary80ScientificTimeControllerV1(
            SimpleNamespace(get_number_of_iterations=lambda: 80)
        ).number_of_iterations()


def test_v2_3_enforces_target_same_frozen_route_progression() -> None:
    ego = {"rear_axle": {"x": 1.0, "y": 0.0, "heading": 0.0}, "speed_mps": 2.0, "time_us": 1}
    accepted = build_hlc_route_continuous_reference_v2_3(
        _paired_map(), ["r0", "r1"], "s0", "t0", ego, 20.0
    )
    historical = build_hlc_route_continuous_reference_v2_2(
        _paired_map(), ["r0", "r1"], "s0", "t0", ego, 20.0
    )
    assert np.array_equal(accepted["source_reference_xy"], historical["source_reference_xy"])
    assert np.array_equal(accepted["target_reference_xy"], historical["target_reference_xy"])
    assert accepted["route_progression_invariant"]["status"].startswith("TARGET_AND_SOURCE")
    mismatched = _paired_map()
    mismatched.edges["t1"]._roadblock = "not_frozen_route"
    with pytest.raises(ValueError, match="TARGET_ROADBLOCK_MISMATCH"):
        build_hlc_route_continuous_reference_v2_3(
            mismatched, ["r0", "r1"], "s0", "t0", ego, 20.0
        )


def test_prepared_route_audit_and_scientific_firewall() -> None:
    audit = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_c_route_progression_invariant_audit_v1.json").read_text())
    roster = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_c_cross_family_engineering_canary_roster_v1.0.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["counts"]["target_route_consistency_violations"] == 0
    assert audit["current_scientific_identity_runner_runs"] == 0
    assert roster["counts"] == {"R-HLC": 3, "R-TSB": 3, "identities": 6, "arms": 12}
    assert all(row["SCIENTIFIC_USE_FORBIDDEN"] for row in roster["entries"])
    assert all(row["PERMANENT_FUTURE_SELECTOR_EXCLUSION"] for row in roster["entries"])


def test_final_canary_dispatch_and_sha_closure() -> None:
    ledger = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_c_cross_family_canary_run_ledger_v1.0.json").read_text())
    dispatch = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_c_full_stack_dispatch_audit_v1.json").read_text())
    manifest = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0.json").read_text())
    assert ledger["status"] == "12_OF_12_CROSS_FAMILY_ENGINEERING_CANARY_PASS"
    assert ledger["counts"]["fresh_actual_runs"] == 12
    assert ledger["counts"]["reruns"] == 0
    assert ledger["counts"]["exact_80_row_traces"] == 12
    assert ledger["counts"]["secondary_planner_calls"] == 0
    assert ledger["counts"]["metric_callback_complete"] == 12
    assert ledger["counts"]["safety_adapter_structural_complete"] == 12
    assert dispatch["counts"] == {
        "HLC_pair_dispatcher_complete": 3,
        "TSB_pair_dispatcher_complete": 3,
        "total": 6,
    }
    assert manifest["status"] == "READY_FOR_SCIENTIFIC_SELECTOR_ROSTER_REBUILD_REVIEW"
    assert manifest["official_smoke_ready_or_authorized"] is False
    assert manifest["RBR_authorized"] is False
