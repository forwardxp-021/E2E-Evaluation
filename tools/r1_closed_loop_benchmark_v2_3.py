#!/usr/bin/env python3
"""R1 HLC route-continuous builder with target-route invariant enforcement."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1
from tools.r1_closed_loop_benchmark_v2_2 import (
    JOIN_GAP_THRESHOLD_M,
    build_hlc_route_continuous_reference_v2_2,
)


BUILDER_VERSION = "build_hlc_route_continuous_reference_v2_3"


def _assert_route_progression_invariant(
    corridor: Mapping[str, Any], route_roadblock_ids: Sequence[str]
) -> Dict[str, Any]:
    route = [str(value) for value in route_roadblock_ids]
    source = list(corridor["source_components"])
    target = list(corridor["target_components"])
    if len(source) != len(target):
        raise ValueError("ROUTE_PROGRESSION_SOURCE_TARGET_COMPONENT_COUNT_MISMATCH")
    cursor = -1
    audited = []
    for source_item, target_item in zip(source, target):
        source_rb = str(source_item["roadblock_id"])
        target_rb = str(target_item["roadblock_id"])
        if source_rb != target_rb:
            raise ValueError(
                "ROUTE_PROGRESSION_TARGET_ROADBLOCK_MISMATCH:"
                f"source={source_rb}:target={target_rb}"
            )
        occurrences = [i for i in range(cursor + 1, len(route)) if route[i] == source_rb]
        if len(occurrences) != 1:
            raise ValueError(
                "ROUTE_PROGRESSION_OCCURRENCE_NOT_EXACTLY_ONE:"
                f"roadblock={source_rb}:after_cursor={cursor}:matches={len(occurrences)}"
            )
        cursor = occurrences[0]
        audited.append(
            {
                "source_edge_id": str(source_item["edge_id"]),
                "target_edge_id": str(target_item["edge_id"]),
                "shared_frozen_route_roadblock_id": source_rb,
                "frozen_route_occurrence_index": cursor,
            }
        )
    return {
        "status": "TARGET_AND_SOURCE_EXACT_SAME_FROZEN_ROUTE_PROGRESSION",
        "component_pairs_audited": len(audited),
        "components": audited,
    }


def build_hlc_route_continuous_reference_v2_3(
    map_api: Any,
    route_roadblock_ids: Sequence[str],
    source_lane_id: str,
    target_lane_id: str,
    current_ego: Mapping[str, Any],
    required_forward_m: float,
) -> Dict[str, Any]:
    corridor = build_hlc_route_continuous_reference_v2_2(
        map_api,
        route_roadblock_ids,
        source_lane_id,
        target_lane_id,
        current_ego,
        required_forward_m,
    )
    audit = _assert_route_progression_invariant(corridor, route_roadblock_ids)
    return {**corridor, "builder_version": BUILDER_VERSION, "route_progression_invariant": audit}


def build_hlc_route_continuous_geometry_v2_3(
    map_api: Any,
    route_roadblock_ids: Sequence[str],
    source_lane_id: str,
    target_lane_id: str,
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    arm: str,
) -> Tuple[Sequence[Dict[str, Any]], Dict[str, Any]]:
    required = max(0.2, float(current_ego["speed_mps"])) * 7.9
    corridor = build_hlc_route_continuous_reference_v2_3(
        map_api,
        route_roadblock_ids,
        source_lane_id,
        target_lane_id,
        current_ego,
        required,
    )
    states = build_hlc_native_geometry_v1_1(
        current_ego,
        absolute_episode_time_s,
        corridor["source_reference_xy"],
        corridor["target_reference_xy"],
        corridor["source_current_arc_m"],
        corridor["target_current_arc_m"],
        arm,
    )
    return states, corridor


__all__ = [
    "BUILDER_VERSION",
    "JOIN_GAP_THRESHOLD_M",
    "build_hlc_route_continuous_geometry_v2_3",
    "build_hlc_route_continuous_reference_v2_3",
]
