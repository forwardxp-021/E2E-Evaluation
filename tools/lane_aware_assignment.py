#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

SLOT_NAMES = ["front", "left_front", "left_rear", "right_front", "right_rear"]


@dataclass
class SlotAssignResult:
    slot_to_agent: Dict[str, str]
    lane_assignment_available: bool
    fallback_assignment_used: bool
    fallback_reason: str
    per_slot_debug: List[dict]


def _to_ego_frame(dx: float, dy: float, h: float) -> Tuple[float, float]:
    c = np.cos(-h)
    s = np.sin(-h)
    return dx * c - dy * s, dx * s + dy * c


def assign_neighbors_geometric(ego_xyh: np.ndarray, candidates_xy: Dict[str, np.ndarray]) -> SlotAssignResult:
    ex, ey, eh = ego_xyh
    scored = {k: [] for k in SLOT_NAMES}
    for aid, xy in candidates_xy.items():
        dx, dy = float(xy[0] - ex), float(xy[1] - ey)
        lon, lat = _to_ego_frame(dx, dy, eh)
        dist = float(np.hypot(dx, dy))
        if lon > 0:
            scored["front"].append((abs(lat) + 0.1 * lon, aid, lon, lat, dist))
        if lat > 0 and lon > 0:
            scored["left_front"].append((dist, aid, lon, lat, dist))
        if lat > 0 and lon <= 0:
            scored["left_rear"].append((dist, aid, lon, lat, dist))
        if lat <= 0 and lon > 0:
            scored["right_front"].append((dist, aid, lon, lat, dist))
        if lat <= 0 and lon <= 0:
            scored["right_rear"].append((dist, aid, lon, lat, dist))

    slot_to_agent: Dict[str, str] = {}
    used = set()
    debug = []
    for slot in SLOT_NAMES:
        rows = sorted(scored[slot], key=lambda x: x[0])
        chosen = None
        for row in rows:
            if row[1] not in used:
                chosen = row
                break
        if chosen is not None:
            used.add(chosen[1])
            slot_to_agent[slot] = str(chosen[1])
            debug.append(dict(slot_name=slot, assignment_method="geometric", neighbor_id=str(chosen[1]), fallback_used=True,
                              fallback_reason="lane_aware_unavailable", distance=chosen[4], longitudinal_gap=chosen[2], lateral_gap=chosen[3]))
        else:
            debug.append(dict(slot_name=slot, assignment_method="geometric", neighbor_id="", fallback_used=True,
                              fallback_reason="no_candidate_for_slot", distance=np.nan, longitudinal_gap=np.nan, lateral_gap=np.nan))

    return SlotAssignResult(slot_to_agent, False, True, "lane_aware_map_projection_not_implemented", debug)


def assign_neighbors_lane_aware(ego_xyh: np.ndarray, candidates_xy: Dict[str, np.ndarray], assignment_mode: str) -> SlotAssignResult:
    # Placeholder interface for future lane map-projection logic.
    if assignment_mode not in {"lane_aware_with_geometric_fallback", "geometric_only"}:
        raise ValueError(f"Unknown assignment_mode: {assignment_mode}")
    return assign_neighbors_geometric(ego_xyh, candidates_xy)
