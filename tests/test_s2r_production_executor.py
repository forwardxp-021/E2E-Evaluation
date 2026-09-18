from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.s2r_production_executor import (
    FreshArm,
    S2RExecutionError,
    assert_fresh_pair,
    canonical_sha256,
    execute_authorized_arm,
    execution_manifest,
    isolated_arm_command,
    precontext_id,
    route_id,
)


def payload():
    return {
        "schema_version": "s2r_precontext_v1",
        "session_id": "session",
        "log_id": "log",
        "scenario_token": "0123456789abcdef",
        "start_timestamp_us": 1,
        "ego_initial_state": {"initial_x": 0.0, "initial_y": 0.0, "initial_heading": 0.0, "initial_speed_mps": 4.0, "initial_time_us": 1},
        "route_id": "route",
        "map_identity": "map",
        "traffic_agent_precontext_sha256": "a" * 64,
        "history_buffer_source_sha256": "b" * 64,
        "simulation_initialization": "OFFICIAL_NUPLAN_SCENARIO",
        "intervention_start_s": 1.1,
    }


def spec(arm):
    pre = payload()
    route = ["1", "2"]
    return {
        "run_id": f"pair-{arm}", "pair_id": "pair", "arm": arm,
        "session_id": "session", "log_id": "log", "scenario_token": "0123456789abcdef",
        "exposure_class": "E1_UNRELATED_HISTORICAL_USE", "role": "V", "db_path": "/db",
        "map_name": "map", "route_roadblock_ids": route, "precontext": pre,
        "precontext_id": precontext_id(pre), "route_id": route_id("map", route), "start_timestamp_us": 1,
    }


def arm(tmp_path, side):
    objects = [object() for _ in range(9)]
    return FreshArm(
        spec=spec(side), runner=objects[0], planner=objects[1], simulation=objects[2],
        controller=objects[3], tracker=objects[4], motion_model=objects[5], callbacks=objects[6],
        recorder=objects[7], random_state=objects[8], run_root=tmp_path / side,
        config_hash="c" * 64, precontext_id=precontext_id(payload()), route_id=route_id("map", ["1", "2"]),
    )


def test_precontext_hash_is_canonical_and_sensitive():
    left = payload()
    right = dict(reversed(list(left.items())))
    assert precontext_id(left) == precontext_id(right)
    right["start_timestamp_us"] = 2
    assert precontext_id(left) != precontext_id(right)


def test_fresh_pair_proves_all_mutable_components_are_distinct(tmp_path):
    proof = assert_fresh_pair(arm(tmp_path, "BASELINE"), arm(tmp_path, "TREATMENT"))
    assert proof["precontext_equal"] and proof["route_equal"]
    assert set(proof["independent_components"]) == {"runner", "planner", "simulation", "controller", "tracker", "motion_model", "callbacks", "recorder", "random_state"}


def test_reused_component_fails_closed(tmp_path):
    baseline, treatment = arm(tmp_path, "BASELINE"), arm(tmp_path, "TREATMENT")
    treatment.tracker = baseline.tracker
    with pytest.raises(S2RExecutionError, match="CROSS_ARM_OBJECT_REUSE"):
        assert_fresh_pair(baseline, treatment)


def test_precontext_or_route_mismatch_invalidates_pair(tmp_path):
    baseline, treatment = arm(tmp_path, "BASELINE"), arm(tmp_path, "TREATMENT")
    treatment.precontext_id = "0" * 64
    with pytest.raises(S2RExecutionError, match="PAIR_INVALID_PRECONTEXT"):
        assert_fresh_pair(baseline, treatment)
    treatment.precontext_id = baseline.precontext_id
    treatment.route_id = "1" * 64
    with pytest.raises(S2RExecutionError, match="PAIR_INVALID_ROUTE"):
        assert_fresh_pair(baseline, treatment)


def test_manifest_contains_required_binding(tmp_path):
    value = execution_manifest(spec("BASELINE"), arm(tmp_path, "BASELINE"), "f" * 40, "ZERO_RUN_FIXTURE")
    assert value["precontext_id"] == precontext_id(payload())
    assert value["route_id"] == route_id("map", ["1", "2"])
    assert value["random_state_declaration"]["execution_process"] == "ONE_FRESH_PROCESS_PER_ARM"
    assert value["execution_status"] == "ZERO_RUN_FIXTURE"


def test_isolated_command_is_one_arm_process(tmp_path):
    command = isolated_arm_command(tmp_path / "spec.json", tmp_path / "authorization.json", tmp_path / "output")
    assert command[1:3] == ["-B", str(Path(__import__("tools.s2r_production_executor", fromlist=["x"]).__file__).resolve())]
    assert command[3] == "--execute-authorized-arm"


def test_closed_authorization_stops_before_factory(monkeypatch, tmp_path):
    import tools.s2r_production_executor as executor

    calls = []
    monkeypatch.setattr(executor, "build_fresh_arm", lambda *_args, **_kwargs: calls.append("build"))
    with pytest.raises(S2RExecutionError, match="SCIENTIFIC_EXECUTION_NOT_AUTHORIZED"):
        execute_authorized_arm(spec("BASELINE"), {"S2R_SCIENTIFIC_EXECUTION_AUTHORIZED": False}, tmp_path)
    assert calls == []


def test_hash_uses_json_semantics_not_object_identity():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
