import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import LineString

from tools import b1_native_route_infrastructure as native
from tools import b1_ir3_route_closure as audit


@pytest.fixture(autouse=True)
def zero_scientific_execution(monkeypatch):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1

    def forbidden(*args, **kwargs):
        pytest.fail("IR3_ZERO_RUN_VIOLATION")
    for cls, name in (
        (SimulationRunner, "run"), (TwoStageController, "update_state"),
        (LQRTracker, "track_trajectory"), (KinematicBicycleModel, "propagate_state"),
        (R2BControllerAwarePlannerV1, "compute_trajectory"),
        (R2BControllerAwarePlannerV1, "compute_planner_trajectory"),
    ):
        monkeypatch.setattr(cls, name, forbidden)


class Edge:
    def __init__(self, edge_id, roadblock, x):
        self.id, self.roadblock, self.outgoing_edges = edge_id, roadblock, []
        self.baseline_path = SimpleNamespace(
            linestring=LineString([(x, 0), (x + 1, 0)]),
            get_nearest_pose_from_position=lambda point: SimpleNamespace(heading=0.0),
            get_nearest_arc_length_from_position=lambda point: 0.0,
        )

    def get_roadblock_id(self):
        return self.roadblock


def fixture_graph():
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    start, bad, good, dead, viable, end = [
        Edge(*args) for args in (("0", "A", 0), ("10", "B", 1), ("20", "B", 1),
                                ("11", "C", 2), ("21", "C", 2), ("30", "D", 3))
    ]
    start.outgoing_edges = [good, bad]  # Deliberately unsorted: the resolver owns deterministic ordering.
    bad.outgoing_edges, good.outgoing_edges, viable.outgoing_edges = [dead], [viable], [end]
    api = SimpleNamespace(get_all_map_objects=lambda point, layer: [start] if layer == SemanticMapLayer.LANE else [])
    current = {"rear_axle": {"x": 0.0, "y": 0.0, "heading": 0.0}, "speed_mps": 4.0, "time_us": 0}
    return api, ["A", "B", "C", "D"], current, viable


def test_same_production_successor_repairs_greedy_dead_end():
    api, route, current, _ = fixture_graph()
    with pytest.raises(ValueError, match="no native outgoing successor into D"):
        native.build_native_route_reference_v1_1(api, route, current, 3.5)
    result = native.build_native_route_reference_v1_2(api, route, current, 3.5)
    assert result["native_edge_ids"] == ["0", "20", "21", "30"]
    assert result["native_successor_repair"]["native_adjacency_verified"]
    assert result["native_successor_repair"]["max_junction_gap_m"] == 0.0
    assert not result["bridge_used"] and not result["extrapolation_used"]


def test_existing_success_is_exactly_unchanged_even_on_a_later_dead_branch():
    api, route, current, _ = fixture_graph()
    before = native.build_native_route_reference_v1_1(api, route, current, 1.5)
    after = native.build_native_route_reference_v1_2(api, route, current, 1.5)
    assert before["native_edge_ids"] == ["0", "10"]
    assert audit.reference_bytes(before) == audit.reference_bytes(after)


def test_missing_physical_connection_is_never_synthesized():
    api, route, current, viable = fixture_graph()
    viable.outgoing_edges = []
    with pytest.raises(ValueError, match="no native path on the unchanged ordered route"):
        native.build_native_route_reference_v1_2(api, route, current, 3.5)


@pytest.mark.parametrize("required", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
def test_all_successful_fixture_distances_retain_exact_return(required):
    api, route, current, _ = fixture_graph()
    assert audit.reference_bytes(native.build_native_route_reference_v1_1(api, route, current, required)) == audit.reference_bytes(native.build_native_route_reference_v1_2(api, route, current, required))


def test_non_successor_errors_are_preserved():
    api, route, current, _ = fixture_graph()
    with pytest.raises(ValueError, match="insufficient native forward coverage"):
        native.build_native_route_reference_v1_2(api, ["A"], current, 4.0)


def test_production_and_executor_use_the_same_successor():
    from tools import r2_b_controller_aware_planner_v1 as planner
    from tools import b1_tsb_qualification_executor as executor
    assert planner.build_native_route_reference_v1_1 is native.build_native_route_reference_v1_2
    assert executor.precheck_with_production_builder is native.production_route_precheck_v1_2


def test_old_execution_authorization_cannot_start_new_stack():
    from tools import b1_tsb_qualification_executor as executor
    authorization = json.loads((audit.B1_DOCS / "B1_TSB_Qualification_Authorization_v1.json").read_text())
    with pytest.raises(executor.B1ExecutionError, match="B1_EXECUTOR_SHA_MISMATCH"):
        executor.validate_authorization(authorization, audit.SPECS, audit.B1_DOCS / "B1_TSB_Qualification_Roster_v1.csv", "B1-TSB-18-TREATMENT")


@pytest.mark.parametrize("include_repair_source", [False, True])
def test_rebinding_only_executor_and_planner_does_not_authorize_unbound_repair_source(include_repair_source):
    from tools import b1_tsb_qualification_executor as executor
    authorization = json.loads((audit.B1_DOCS / "B1_TSB_Qualification_Authorization_v1.json").read_text())
    authorization["executor_sha256"] = audit.sha256_file(Path(executor.__file__))
    authorization["planner_sha256"] = audit.sha256_file(executor.PLANNER)
    if include_repair_source:
        authorization["source_file_hashes"]["tools/b1_native_route_infrastructure.py"] = audit.sha256_file(Path(native.__file__))
    with pytest.raises(executor.B1ExecutionError, match="B1_ROUTE_INFRASTRUCTURE_SOURCE_BINDING_MISSING"):
        executor.validate_authorization(authorization, audit.SPECS, audit.B1_DOCS / "B1_TSB_Qualification_Roster_v1.csv", "B1-TSB-18-TREATMENT")


def test_tampered_identity_is_rejected_before_map_resolution():
    spec = json.loads(audit.SPECS.read_text())["arms"][0]
    spec["route_id"] = "0" * 64
    gate = native.production_route_precheck_v1_2(spec, object())
    assert gate["route_precheck_status"] == "TECHNICAL_ERROR"
    assert not gate["route_identity_valid"]


def test_reproduction_and_all_historical_references(tmp_path):
    evidence = audit.generate(tmp_path / "reproduced")
    assert evidence["summary"]["old_success_byte_identical"] == 2773
    assert evidence["summary"]["existing17_reference_queries"] == 2720
    assert evidence["summary"]["focus_06_10_14_queries"] == 480
    assert evidence["summary"]["old_failure_repaired"] == 1
    assert evidence["summary"]["new_simulations"] == 0
    for run_id, gate in evidence["gates"].items():
        assert gate["route_precheck_status"] == "COMPATIBLE", run_id
        assert gate["full_ordered_route_native_path"]["native_adjacency_verified"]
        assert gate["full_ordered_route_native_path"]["max_junction_gap_m"] == 0.0
    frozen = audit.ROOT / "docs/stageR/b1_ir3_route_closure"
    for path in (tmp_path / "reproduced").iterdir():
        assert path.read_bytes() == (frozen / path.name).read_bytes(), path.name
    with pytest.raises(FileExistsError, match="IR3_FRESH_OUTPUT_DIRECTORY_REQUIRED"):
        audit.generate(frozen)


def test_final_manifest_hashes_and_unchanged_historical_assets():
    base = audit.ROOT / "docs/stageR/b1_ir3_route_closure"
    manifest = json.loads((base / "B1_IR3_Manifest_v1.json").read_text())
    assert manifest["status"] == "B1_IR3_READY_FOR_RESUME_OWNER_REVIEW"
    assert all(value == "NOT_AUTHORIZED" for value in manifest["authorization"].values())
    assert manifest["max_total_technical_attempts_per_arm"] == 2
    for name, expected in manifest["artifact_sha256"].items():
        assert audit.sha256_file(base / name) == expected
    for name, expected in manifest["source_sha256"].items():
        assert audit.sha256_file(audit.ROOT / name) == expected
    evidence = json.loads((base / "B1_IR3_Static_Evidence_v1.json").read_text())
    assert evidence["historical_artifact_sha256"] == audit.historical_hashes()
    assert all(value == 0 for value in manifest["zero_run_proof"].values())
