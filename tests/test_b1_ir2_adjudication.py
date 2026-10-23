import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tools import b1_ir2_adjudication as ir2
from tools import b1_native_route_precheck as precheck
from tools.r1_b2_8_r3_1_official_safety_adapter import adapt_official_safety


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1

    def forbidden(*args, **kwargs):
        pytest.fail("ZERO_RUN_VIOLATION: scientific runtime entry called")

    monkeypatch.setattr(SimulationRunner, "run", forbidden)
    monkeypatch.setattr(TwoStageController, "update_state", forbidden)
    monkeypatch.setattr(LQRTracker, "track_trajectory", forbidden)
    monkeypatch.setattr(KinematicBicycleModel, "propagate_state", forbidden)
    monkeypatch.setattr(R2BControllerAwarePlannerV1, "compute_trajectory", forbidden)
    monkeypatch.setattr(R2BControllerAwarePlannerV1, "compute_planner_trajectory", forbidden)


def test_frozen_safety_is_deterministic_and_full_contract_is_complete():
    arms, pairs = ir2.adjudicate_safety()
    assert (arms, pairs) == ir2.adjudicate_safety()
    assert len(arms) == 34 and len(pairs) == 17
    assert all(row["joint_arm_safety"] == "PASS" for row in arms)
    full = ir2.full_contract_rows(pairs)
    assert len(full) == 17
    assert all(row["full_contract_status"] == "FULL_CONTRACT_COMPLETE" for row in full)
    assert all(row["scientific_status"] == "SCIENTIFIC_PASS" for row in full)


def test_official_safety_failure_cannot_be_overridden(tmp_path):
    source = ir2.B1_OUTPUT / "B1-TSB-01-BASELINE/raw/metrics"
    for name in ("no_ego_at_fault_collisions", "drivable_area_compliance"):
        shutil.copy2(source / f"{name}.parquet", tmp_path / f"{name}.parquet")
    collision = tmp_path / "no_ego_at_fault_collisions.parquet"
    table = pd.read_parquet(collision)
    table["number_of_all_at_fault_collisions_stat_value"] = 1
    table.to_parquet(collision)
    assert adapt_official_safety(tmp_path)["frozen_arm_safety_pass"] is False
    _, pairs = ir2.adjudicate_safety()
    pairs[0]["pair_safety"] = "PAIR_SAFETY_FAIL"
    result = ir2.full_contract_rows(pairs)[0]
    assert result["mechanism_valid"] == "YES"
    assert result["full_contract_status"] == "FULL_CONTRACT_COMPLETE"
    assert result["scientific_status"] == "SCIENTIFIC_FAIL"


def test_recovered_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    audit = json.loads(ir2.RECOVERY_AUDIT.read_text())
    audit["arms"][0]["output_sha256"]["no_ego_at_fault_collisions"] = "0" * 64
    path = tmp_path / "bad_audit.json"
    path.write_text(json.dumps(audit))
    monkeypatch.setattr(ir2, "RECOVERY_AUDIT", path)
    with pytest.raises(RuntimeError, match="SAFETY_ARTIFACT_HASH_MISMATCH"):
        ir2.adjudicate_safety()


@pytest.mark.parametrize("initial_failure,tail_failure,expected", [
    (True, True, "INCOMPATIBLE"),
    (False, True, "AMBIGUOUS"),
    (False, False, "COMPATIBLE"),
])
def test_initial_prerequisite_and_tail_risk_are_distinct(monkeypatch, initial_failure, tail_failure, expected):
    from tools import r1_closed_loop_benchmark_v2_1 as production
    spec = json.loads(ir2.SPECS.read_text())["arms"][0]
    calls = []

    def route_builder(api, route, current, required):
        calls.append(required)
        if initial_failure or (required == float("inf") and tail_failure):
            raise ValueError("NATIVE_ROUTE_FAIL: no native outgoing successor into 19339")
        if required == float("inf"):
            raise ValueError("NATIVE_ROUTE_FAIL: insufficient native forward coverage")
        return {}

    monkeypatch.setattr(production, "build_native_route_reference_v1_1", route_builder)
    result = precheck.precheck_with_production_builder(spec, object())
    assert result["route_precheck_status"] == expected
    assert calls[0] < float("inf")


def test_retrospective_cases_and_remaining_arm_classification():
    specs = json.loads(ir2.SPECS.read_text())["arms"]
    rows, detail = ir2.route_case_rows(specs)
    by_run = {row["arm_id"]: row for row in rows}
    for number, max_index in ((6, 1), (10, 4), (14, 1)):
        for arm in ("BASELINE", "TREATMENT"):
            row = by_run[f"B1-TSB-{number:02d}-{arm}"]
            assert row["historical_execution_valid"] == "YES"
            assert row["runtime_successful_route_calls"] == 80
            assert row["runtime_max_required_route_index"] == max_index
            assert row["future_route_status"] == "ROUTE_AMBIGUOUS"
    failed = detail["B1-TSB-18-TREATMENT"]["retrospective_production_replay"]
    assert failed["first_failure"]["iteration"] == 53
    assert failed["first_failure"]["failure_code"].endswith("into 19339")
    assert failed["first_failure"]["required_forward_m"] > failed["last_recorded_input_capacity"]["max_resolvable_forward_m_upper_bound"]
    assert by_run["B1-TSB-20-BASELINE"]["future_route_status"] == "ROUTE_AMBIGUOUS"
    assert by_run["B1-TSB-20-TREATMENT"]["runtime_trace_rows"] == 0
    plan = ir2.resume_rows(specs, rows, precheck.precheck_specs(ir2.SPECS))
    assert len(plan) == 40
    assert all("NO_RERUN" in row["next_action"] for row in plan if int(row["pair_id"][-2:]) <= 17)
    assert all("DO_NOT_ENTER_RUNNER" in row["next_action"] for row in plan if row["pair_id"] == "B1-TSB-20")


def test_lifecycle_rejects_missing_controller_transition(tmp_path, monkeypatch):
    spec = json.loads(ir2.SPECS.read_text())["arms"][0]
    root = tmp_path / spec["run_id"]
    shutil.copytree(ir2.B1_OUTPUT / spec["run_id"], root)
    controller = root / "telemetry/actual_lqr_controller_telemetry.jsonl"
    controller.write_text("\n".join(controller.read_text().splitlines()[:-1]) + "\n")
    recovered = json.loads(ir2.RECOVERY_AUDIT.read_text())["arms"][0]
    monkeypatch.setattr(ir2, "B1_OUTPUT", tmp_path)
    gates = ir2.audit_arm_lifecycle(spec, recovered)
    assert gates["recorder_complete"] is False
    assert gates["artifact_hash_valid"] is False


def test_frozen_ir2_package_has_valid_hashes_and_closed_authority():
    base = ir2.ROOT / "docs/stageR/b1_ir2_adjudication"
    manifest = json.loads((base / "B1_IR2_Manifest_v1.json").read_text())
    assert manifest["status"] == "B1_IR2_BLOCKED_ROUTE_ADJUDICATION"
    assert manifest["planned_denominator"] == 20
    assert all(value == "NOT_AUTHORIZED" for value in manifest["authorization"].values())
    assert all(value == 0 for value in manifest["zero_run_proof"].values())
    for name, expected in manifest["artifact_sha256"].items():
        assert ir2.sha256_file(base / name) == expected
    for name, expected in manifest["source_sha256"].items():
        assert ir2.sha256_file(ir2.ROOT / name) == expected


def test_full_zero_run_reproduction_matches_frozen_tables(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["b1_ir2_adjudication.py", "--output-dir", str(tmp_path)])
    assert ir2.main() == 0
    frozen = ir2.ROOT / "docs/stageR/b1_ir2_adjudication"
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (frozen / path.name).read_bytes(), path.name
    # A frozen package cannot be overwritten by the generating command.
    monkeypatch.setattr("sys.argv", ["b1_ir2_adjudication.py", "--output-dir", str(frozen)])
    with pytest.raises(FileExistsError, match="FROZEN_IR2_OUTPUT_EXISTS"):
        ir2.main()
