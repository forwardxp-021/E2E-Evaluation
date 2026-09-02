import hashlib
import json
from pathlib import Path

import pytest

from tools.r1_b2_9_d_execute_frozen_48run_smoke import FrozenBudgetLedger


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def _load(name: str) -> dict:
    return json.loads((R1 / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_effective_exclusions_are_closed_and_attempt1_is_consumed() -> None:
    ledger = _load("r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json")
    rows = ledger["entries"]
    tokens = {row["scenario_token"] for row in rows}
    assert ledger["counts"]["entries"] == 45
    assert {
        "b1be12bca092597a",
        "25944935eadb52f1",
        "ef3172a208cc5dd7",
        "b486f9cf33a85455",
        "3edcce9e7e19573f",
        "ff152a4cf9c4503b",
    }.issubset(tokens)
    attempt1 = [row for row in rows if row["scenario_token"] == "b1be12bca092597a"]
    assert len(attempt1) == 1
    assert attempt1[0]["OFFICIAL_ATTEMPT_CONSUMED"] is True


def test_roster_v3_is_24_unique_outcome_blind_identities() -> None:
    roster = _load("r1_official_compliant_technical_smoke_roster_v3.0.json")
    exclusions = _load("r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json")["entries"]
    comparison = _load("r1_b2_9_d_roster_v2_1_to_v3_0_comparison_v1.json")
    entries = roster["entries"]
    excluded_tokens = {row["scenario_token"] for row in exclusions}
    excluded_logs = {row["log_id"] for row in exclusions}
    assert len(entries) == 24
    assert len({(row["scenario_token"], row["log_id"]) for row in entries}) == 24
    assert sum(row["family"] == "R-HLC" for row in entries) == 12
    assert sum(row["family"] == "R-TSB" for row in entries) == 12
    assert all(row["scenario_token"] not in excluded_tokens for row in entries)
    assert all(row["log_id"] not in excluded_logs for row in entries)
    assert comparison["counts"] == {
        "HLC_retained": 11,
        "HLC_replaced": 1,
        "TSB_retained": 11,
        "TSB_replaced": 1,
    }
    assert comparison["scientific_rollout_outcome_used"] is False


def test_schedule_and_pair_bindings_are_frozen_pre_outcome() -> None:
    schedule = _load("r1_official_compliant_technical_smoke_schedule_v3.0.json")
    binding_doc = _load("r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json")
    runs = schedule["runs"]
    pairs = binding_doc["pairs"]
    assert len(runs) == 48
    assert [row["run_order"] for row in runs] == list(range(1, 49))
    assert len({row["run_id"] for row in runs}) == 48
    assert all(row["run_id"].startswith("R1B29D-") for row in runs)
    assert len(pairs) == 24 and len({row["pair_id"] for row in pairs}) == 24
    assert binding_doc["pre_outcome_complete"] is True
    for pair in pairs:
        assert pair["future_realized_trace_used"] is False
        assert pair["future_safety_result_used"] is False
        assert pair["future_scientific_gate_result_used"] is False
        if pair["family"] == "R-HLC":
            assert pair["PLANNER_REFERENCE_SEMANTICS"] == "ROUTE_CONTINUOUS_V2_3"
            assert pair["MEASUREMENT_REFERENCE_SEMANTICS"] == "FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT"
            assert pair["measurement_numerics_changed"] is False
        else:
            assert pair["pretreatment_clearance"] is None
            assert "source_reference_xy" not in pair


def test_zero_run_and_dispatcher_closures_have_no_simulation() -> None:
    zero = _load("r1_b2_9_d_zero_run_final_construction_audit_v1.0.json")
    dispatch = _load("r1_b2_9_d_dispatcher_structural_audit_v1.0.json")
    assert zero["status"] == "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS"
    assert zero["counts"] == {
        "exact_resolutions": 48,
        "planner_v3_1_bindings": 48,
        "Primary80_controller_bindings": 48,
        "runner_constructions": 48,
        "pair_binding_lookups": 48,
    }
    assert all(row["planner_class"] == "R1OfficialTechnicalSmokePlannerV3_1" for row in zero["runs"])
    assert all(row["time_controller_class"] == "R1Primary80ScientificTimeControllerV1" for row in zero["runs"])
    assert all(row["controller_number_of_iterations"] == 81 for row in zero["runs"])
    assert zero["claim_49"] == "HARD_FAIL_BEFORE_RUNNER_RUN_CAP_48"
    assert zero["runner_run_calls"] == 0 and zero["simulation_started"] is False
    assert dispatch["status"] == "24_OF_24_FROZEN_PAIR_DISPATCHER_STRUCTURAL_PASS"
    assert dispatch["counts"]["pass"] == 24
    assert dispatch["runner_run_calls"] == 0 and dispatch["simulation_started"] is False


def test_budget_claim_48_then_49_fails_before_runner() -> None:
    ledger = FrozenBudgetLedger()
    for index in range(48):
        ledger.claim(f"R1B29D-dry-{index:02d}")
    with pytest.raises(RuntimeError, match="HARD_FAIL_BEFORE_RUNNER_RUN_CAP_48"):
        ledger.claim("R1B29D-forbidden-49")


def test_final_manifest_has_complete_valid_sha_closure() -> None:
    manifest = _load("r1_b2_9_d_final_execution_binding_manifest_v2.0.json")
    assert manifest["status"] == "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION"
    assert manifest["complete_transitive_sha_closure"] == "PASS"
    assert manifest["closure_component_count"] == len(manifest["complete_transitive_component_sha256"])
    assert manifest["closure_component_count"] >= 60
    for component, expected in manifest["complete_transitive_component_sha256"].items():
        path = Path(component)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), component
        assert _sha(path) == expected, component
    assert manifest["official_runs"] == 0
    assert manifest["consumed_real_budget"] == 0
    assert manifest["authorization"] == {
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "RBR_A/B/C": "NOT_AUTHORIZED",
    }
    protected = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
    assert _sha(protected) == "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
