#!/usr/bin/env python3
"""Native-edge-only route successor repair; no planner or simulator execution.

The immutable v1.1 implementation is the fast path. Only its missing-successor
failure enables deterministic backtracking over native edges on the same ordered
roadblock sequence. Both production and preflight use this versioned successor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1

ROOT = Path(__file__).resolve().parents[1]
VERSION = "build_native_route_reference_v1_2_native_successor_backtracking"


def _initial_native_edge(map_api: Any, route: Sequence[str], current: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    """Obtain the exact initial edge/cursor selected by the immutable v1.1 code."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    initial = build_native_route_reference_v1_1(map_api, route, current, 0.0)
    point = Point2D(float(current["rear_axle"]["x"]), float(current["rear_axle"]["y"]))
    local = list(map_api.get_all_map_objects(point, SemanticMapLayer.LANE))
    local += list(map_api.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR))
    selected = [edge for edge in local if str(edge.id) == initial["native_edge_ids"][0] and str(edge.get_roadblock_id()) == str(route[initial["route_occurrence_cursor"]])]
    if len(selected) != 1:
        raise ValueError("NATIVE_ROUTE_REPAIR_FAIL: initial native edge not uniquely recoverable")
    return selected[0], initial


def _native_path(
    edge: Any, route: Sequence[str], cursor: int, remaining_m: float, require_route_end: bool,
) -> list[Any] | None:
    """Lexicographic DFS on native outgoing edges, advancing one route index."""
    if not require_route_end and remaining_m <= 0.0:
        return [edge]
    if cursor == len(route) - 1:
        return [edge] if require_route_end else None
    candidates = sorted(
        (candidate for candidate in edge.outgoing_edges if str(candidate.get_roadblock_id()) == str(route[cursor + 1])),
        key=lambda candidate: str(candidate.id),
    )
    for candidate in candidates:
        suffix = _native_path(
            candidate, route, cursor + 1,
            remaining_m - float(candidate.baseline_path.linestring.length), require_route_end,
        )
        if suffix is not None:
            return [edge, *suffix]
    return None


def resolve_native_path_v1_2(
    map_api: Any, route_roadblock_ids: Sequence[str], current_ego: Mapping[str, Any],
    required_forward_m: float, *, require_route_end: bool = False,
) -> tuple[list[Any], Mapping[str, Any]]:
    """Shared path resolver for finite production repair and static topology audit."""
    route = [str(value) for value in route_roadblock_ids]
    edge, initial = _initial_native_edge(map_api, route, current_ego)
    available = float(edge.baseline_path.linestring.length) - float(initial["current_route_arc_m"])
    path = _native_path(
        edge, route, int(initial["route_occurrence_cursor"]),
        float(required_forward_m) - available, require_route_end,
    )
    if path is None:
        raise ValueError("NATIVE_ROUTE_REPAIR_FAIL: no native path on the unchanged ordered route")
    return path, initial


def native_path_evidence(path: Sequence[Any]) -> Mapping[str, Any]:
    gaps = [
        float(np.linalg.norm(np.asarray(left.baseline_path.linestring.coords[-1]) - np.asarray(right.baseline_path.linestring.coords[0])))
        for left, right in zip(path, path[1:])
    ]
    return {
        "native_edge_ids": [str(edge.id) for edge in path],
        "native_roadblock_ids": [str(edge.get_roadblock_id()) for edge in path],
        "native_adjacency_verified": all(
            any(str(candidate.id) == str(right.id) and str(candidate.get_roadblock_id()) == str(right.get_roadblock_id()) for candidate in left.outgoing_edges)
            for left, right in zip(path, path[1:])
        ),
        "junction_gap_m": gaps,
        "max_junction_gap_m": max(gaps, default=0.0),
    }


def build_native_route_reference_v1_2(
    map_api: Any, route_roadblock_ids: Sequence[str], current_ego: Mapping[str, Any], required_forward_m: float,
) -> dict[str, Any]:
    """Preserve all old successes; repair only a proven greedy missing successor."""
    try:
        return build_native_route_reference_v1_1(map_api, route_roadblock_ids, current_ego, required_forward_m)
    except ValueError as error:
        if not str(error).startswith("NATIVE_ROUTE_FAIL: no native outgoing successor into "):
            raise
        original_error = str(error)
    if not math.isfinite(float(required_forward_m)):
        raise ValueError(original_error)
    path, initial = resolve_native_path_v1_2(map_api, route_roadblock_ids, current_ego, required_forward_m)
    points: list[list[float]] = []
    for edge in path:
        coords = [[float(x), float(y)] for x, y in edge.baseline_path.linestring.coords]
        if points and coords and points[-1] == coords[0]:
            coords = coords[1:]
        points.extend(coords)
    reference = np.asarray(points, dtype=np.float64)
    from tools.r1_prospective_generator_contract_v2 import polyline_arclength
    polyline_arclength(reference)
    return {
        "reference_xy": reference,
        "current_route_arc_m": initial["current_route_arc_m"],
        "route_occurrence_cursor": initial["route_occurrence_cursor"],
        "native_edge_ids": [str(edge.id) for edge in path],
        "bridge_used": False,
        "extrapolation_used": False,
        "builder_version": VERSION,
        "native_successor_repair": {"original_failure": original_error, **native_path_evidence(path)},
    }


def production_route_precheck_v1_2(spec: Mapping[str, Any], map_api: Any) -> Mapping[str, Any]:
    """Same production finite builder plus bounded ordered-route graph resolution."""
    from tools.b1_native_route_precheck import initial_current
    from tools.r2_b_controller_aware_planner_v1 import _tsb_required_distance
    from tools.s2r_production_executor import precontext_id, route_id

    parameters = json.loads((ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json").read_text())["parameters"]
    current = initial_current(spec)
    required = _tsb_required_distance(current["speed_mps"], 0.0, "BASELINE", parameters)
    base = {
        "route_id": spec["route_id"],
        "route_identity_valid": route_id(spec["map_name"], spec["route_roadblock_ids"]) == spec["route_id"],
        "precontext_identity_valid": precontext_id(spec["precontext"]) == spec["precontext_id"],
        "production_equivalence_version": VERSION,
        "production_equivalence_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_required_forward_m": required,
        "initial_route_query_pass": False,
        "runtime_route_compatibility_scope": "NATIVE_PATH_FOR_INITIAL_QUERY_AND_UNCHANGED_ORDERED_ROUTE;NOT_ALL_FUTURE_EGO_STATES",
        "full_episode_compatibility_evidence": "STATIC_NATIVE_TOPOLOGY_AND_FROZEN_QUERY_REGRESSION;NO_NEW_CLOSED_LOOP_EXECUTION",
    }
    if not base["route_identity_valid"] or not base["precontext_identity_valid"]:
        return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": "FROZEN_ROUTE_OR_PRECONTEXT_IDENTITY_MISMATCH"}
    try:
        finite = build_native_route_reference_v1_2(map_api, spec["route_roadblock_ids"], current, required)
    except ValueError as error:
        return {**base, "route_precheck_status": "INCOMPATIBLE", "failure_code": str(error)}
    base = {**base, "initial_route_query_pass": True, "initial_native_edge_ids": finite["native_edge_ids"]}
    try:
        path, initial = resolve_native_path_v1_2(map_api, spec["route_roadblock_ids"], current, required, require_route_end=True)
    except ValueError as error:
        return {**base, "route_precheck_status": "AMBIGUOUS", "latent_topology_risk": "UNRESOLVED", "failure_code": str(error)}
    path_evidence = native_path_evidence(path)
    expected_suffix = [str(value) for value in spec["route_roadblock_ids"][initial["route_occurrence_cursor"]:]]
    if path_evidence["native_roadblock_ids"] != expected_suffix or not path_evidence["native_adjacency_verified"]:
        return {**base, "route_precheck_status": "TECHNICAL_ERROR", "failure_code": "NATIVE_PATH_OR_ORDERED_ROUTE_PROOF_FAILED"}
    try:
        build_native_route_reference_v1_1(map_api, spec["route_roadblock_ids"], current, float("inf"))
    except ValueError as error:
        legacy_diagnostic = str(error)
    else:
        legacy_diagnostic = "UNEXPECTED_INFINITE_REQUEST_SUCCESS"
    return {
        **base,
        "route_precheck_status": "COMPATIBLE",
        "latent_topology_risk": "GREEDY_DEAD_END_REPAIRED_BY_EXISTING_NATIVE_BRANCHES" if "no native outgoing successor into " in legacy_diagnostic else "NONE_DETECTED_IN_STATIC_ROUTE_SCOPE",
        "legacy_full_route_diagnostic": legacy_diagnostic,
        "route_occurrence_cursor": initial["route_occurrence_cursor"],
        "full_ordered_route_native_path": path_evidence,
        "failure_code": None,
    }
