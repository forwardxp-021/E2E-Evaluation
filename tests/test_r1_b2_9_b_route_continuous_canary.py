from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString

from tools.r1_b2_8_r3_prospective_selector import official_env

official_env()

from tools.r1_b2_9_b_canary_time_controller import R1B29BEngineeringCanary80CallTimeController
from tools.r1_closed_loop_benchmark_v2_2 import build_hlc_route_continuous_reference_v2_2
from tools.r1_official_technical_smoke_planner_v3_0 import Primary80AndSecondaryTraceWriterV1_1


ROOT = Path(__file__).resolve().parents[1]


class Edge:
    def __init__(self, edge_id: str, roadblock: str, xy: list[list[float]], connector: bool = False) -> None:
        self.id, self._roadblock = edge_id, roadblock
        self.baseline_path = SimpleNamespace(linestring=LineString(xy))
        self.outgoing_edges: list[Edge] = []
        self.adjacent_edges = (None, None)
        self._connector = connector

    def get_roadblock_id(self) -> str:
        return self._roadblock


class Connector(Edge):
    pass


class FakeMap:
    def __init__(self, edges: list[Edge]) -> None:
        self.edges = {edge.id: edge for edge in edges}

    def get_map_object(self, edge_id: str, layer: object) -> Edge | None:
        edge = self.edges.get(edge_id)
        if edge is None:
            return None
        name = getattr(layer, "name", str(layer))
        return edge if ("CONNECTOR" in name) == isinstance(edge, Connector) else None


def _paired_map(ambiguous: bool = False) -> FakeMap:
    source = Edge("s0", "r0", [[0, 0], [10, 0]])
    target = Edge("t0", "r0", [[0, 1], [10, 1]])
    source.adjacent_edges, target.adjacent_edges = (None, target), (source, None)
    source_next = Edge("s1", "r1", [[10, 0], [30, 0]])
    target_next = Edge("t1", "r1", [[10, 1], [30, 1]])
    source_next.adjacent_edges, target_next.adjacent_edges = (None, target_next), (source_next, None)
    source.outgoing_edges, target.outgoing_edges = [source_next], [target_next]
    edges = [source, target, source_next, target_next]
    if ambiguous:
        duplicate = Edge("s2", "r1", [[10, 0], [30, 0.5]])
        duplicate.adjacent_edges = (None, target_next)
        target_next.adjacent_edges = (source_next, duplicate)
        source.outgoing_edges.append(duplicate)
        edges.append(duplicate)
    return FakeMap(edges)


def test_route_continuous_unique_pair_and_ambiguity_fail_closed() -> None:
    ego = {"rear_axle": {"x": 1.0, "y": 0.0, "heading": 0.0}, "speed_mps": 2.0, "time_us": 1}
    result = build_hlc_route_continuous_reference_v2_2(_paired_map(), ["r0", "r1"], "s0", "t0", ego, 20.0)
    assert [item["edge_id"] for item in result["source_components"]] == ["s0", "s1"]
    assert result["extrapolation_used"] is False
    with pytest.raises(ValueError, match="AMBIGUITY_FAIL_CLOSED"):
        build_hlc_route_continuous_reference_v2_2(_paired_map(True), ["r0", "r1"], "s0", "t0", ego, 20.0)


def test_engineering_time_controller_yields_80_planner_calls() -> None:
    scenario = SimpleNamespace(get_number_of_iterations=lambda: 200)
    controller = R1B29BEngineeringCanary80CallTimeController(scenario)
    assert controller.number_of_iterations() == 81
    assert controller.reached_end() is False


def test_trace_writer_separates_primary_and_secondary(tmp_path: Path) -> None:
    class Ego:
        time_us = 123
        rear_axle = SimpleNamespace(x=1.0, y=2.0, heading=0.3)
        dynamic_car_state = SimpleNamespace(speed=4.0)

    writer = Primary80AndSecondaryTraceWriterV1_1(str(tmp_path))
    for index in (79, 80):
        current = SimpleNamespace(iteration=SimpleNamespace(index=index, time_us=123), history=SimpleNamespace(current_state=(Ego(), None)))
        writer.write(current)
    assert json.loads(writer.path.read_text().strip())["iteration_index"] == 79
    secondary = json.loads(writer.secondary_path.read_text().strip())
    assert secondary["iteration_index"] == 80
    assert secondary["trace_role"] == "SECONDARY_DIAGNOSTIC_NOT_PRIMARY"


def test_final_canary_ledger_and_scientific_firewall() -> None:
    ledger = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_b_engineering_canary_run_ledger_v1.0.json").read_text())
    exclusion = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_b_engineering_canary_exclusion_ledger_v1.0.json").read_text())
    current12 = json.loads((ROOT / "docs/stageR/r1/r1_b2_9_b_current12_route_continuous_diagnostic_v1.json").read_text())
    assert ledger["status"] == "ROUTE_CONTINUOUS_ENGINEERING_CANARY_PASS"
    assert ledger["counts"]["latest_required_runs_complete"] == 6
    assert ledger["counts"]["final_native_coverage_failures"] == 0
    assert ledger["counts"]["final_other_technical_failures"] == 0
    assert ledger["scientific_identities_simulated"] is False
    assert all(row["SCIENTIFIC_USE_FORBIDDEN"] for row in exclusion["entries"])
    assert current12["counts"] == {"identities": 12, "coverage_pass": 12, "coverage_fail": 0, "topology_ambiguity": 0}
