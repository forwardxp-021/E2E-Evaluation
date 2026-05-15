#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from tools.waymo_lane_utils import find_best_lane_for_agent, project_point_to_lane, wrap_to_pi

SLOT_NAMES = ["front", "left_front", "left_rear", "right_front", "right_rear"]


@dataclass
class SlotAssignResult:
    slot_to_agent: Dict[str, str]
    lane_assignment_available: bool
    fallback_assignment_used: bool
    fallback_reason: str
    per_slot_debug: List[dict]
    current_lane_id: str = ""
    left_lane_id: str = ""
    right_lane_id: str = ""
    adjacency_source: str = "none"


def _to_ego_frame(dx: float, dy: float, h: float) -> Tuple[float, float]:
    c = np.cos(-h); s = np.sin(-h)
    return dx * c - dy * s, dx * s + dy * c


def assign_neighbors_geometric(ego_xyh: np.ndarray, candidates_xy: Dict[str, np.ndarray], reason="lane_aware_unavailable") -> SlotAssignResult:
    ex, ey, eh = ego_xyh
    scored = {k: [] for k in SLOT_NAMES}
    for aid, xy in candidates_xy.items():
        dx, dy = float(xy[0] - ex), float(xy[1] - ey)
        lon, lat = _to_ego_frame(dx, dy, eh)
        dist = float(np.hypot(dx, dy))
        if lon > 0: scored["front"].append((abs(lat) + 0.1 * lon, aid, lon, lat, dist))
        if lat > 0 and lon > 0: scored["left_front"].append((dist, aid, lon, lat, dist))
        if lat > 0 and lon <= 0: scored["left_rear"].append((dist, aid, lon, lat, dist))
        if lat <= 0 and lon > 0: scored["right_front"].append((dist, aid, lon, lat, dist))
        if lat <= 0 and lon <= 0: scored["right_rear"].append((dist, aid, lon, lat, dist))
    slot_to_agent = {}; used = set(); debug = []
    for slot in SLOT_NAMES:
        chosen = next((r for r in sorted(scored[slot], key=lambda x: x[0]) if r[1] not in used), None)
        if chosen is not None:
            used.add(chosen[1]); slot_to_agent[slot] = str(chosen[1])
            debug.append(dict(slot_name=slot, assignment_method="geometric_fallback", neighbor_id=str(chosen[1]), fallback_used=True, fallback_reason=reason,
                              distance=chosen[4], longitudinal_gap=chosen[2], lateral_gap=chosen[3]))
        else:
            debug.append(dict(slot_name=slot, assignment_method="empty", neighbor_id="", fallback_used=True, fallback_reason="no_candidate_for_slot",
                              distance=np.nan, longitudinal_gap=np.nan, lateral_gap=np.nan))
    return SlotAssignResult(slot_to_agent, False, True, reason, debug)


def assign_neighbors_lane_aware(ego_state, candidate_states, lane_infos=None, assignment_mode="lane_aware_with_geometric_fallback", config=None, ego_projection=None, candidate_projections=None):
    cfg = config or {}
    if assignment_mode == "geometric_only":
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, "geometric_only_mode")
    if not lane_infos:
        if assignment_mode == "lane_aware_only":
            return SlotAssignResult({}, False, False, "lane_map_unavailable", [dict(slot_name=s, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason="lane_map_unavailable") for s in SLOT_NAMES])
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, "lane_map_unavailable")

    max_lat = float(cfg.get("lane_max_lateral_distance", 3.0)); max_hd = np.deg2rad(float(cfg.get("lane_max_heading_diff_deg", 45.0)))
    if ego_projection is None:
        ego_proj, reason, _ = find_best_lane_for_agent(np.array([ego_state["x"], ego_state["y"]]), ego_state["heading"], lane_infos, max_lat, max_hd,
            search_radius=float(cfg.get("lane_search_radius",20.0)), topk_candidates=int(cfg.get("lane_topk_candidates",32)),
            disable_spatial_index=bool(cfg.get("disable_lane_spatial_index",False)))
    else:
        ego_proj, reason = ego_projection, "ok"
    if ego_proj is None:
        if assignment_mode == "lane_aware_only":
            return SlotAssignResult({}, False, False, reason, [dict(slot_name=s, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason=reason) for s in SLOT_NAMES])
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, reason)

    cur = ego_proj["lane_id"]; left = ""; right = ""; src = "none"
    ln = lane_infos[cur]
    if ln.left_neighbor_lane_ids or ln.right_neighbor_lane_ids:
        left = ln.left_neighbor_lane_ids[0] if ln.left_neighbor_lane_ids else ""
        right = ln.right_neighbor_lane_ids[0] if ln.right_neighbor_lane_ids else ""
        src = "proto_topology"
    else:
        src = "geometric"
        min_off = float(cfg.get("adjacent_lane_min_offset", 2.0)); max_off = float(cfg.get("adjacent_lane_max_offset", 5.5)); max_adj_hd = np.deg2rad(float(cfg.get("adjacent_lane_max_heading_diff_deg", 35.0)))
        for lid, linfo in lane_infos.items():
            if lid == cur: continue
            p = project_point_to_lane(np.array([ego_state["x"], ego_state["y"]]), linfo)
            if not p["projection_success"]: continue
            off = p["l"]; hd = abs(wrap_to_pi(p["heading"] - ego_proj["heading"]))
            if hd > max_adj_hd or not (min_off <= abs(off) <= max_off): continue
            if off > 0 and left == "": left = lid
            if off < 0 and right == "": right = lid

    slot_to_agent = {}; used = set(); debug = []
    ego_s_map = {cur: ego_proj["s"]}
    if left in lane_infos: ego_s_map[left] = project_point_to_lane(np.array([ego_state["x"], ego_state["y"]]), lane_infos[left])["s"]
    if right in lane_infos: ego_s_map[right] = project_point_to_lane(np.array([ego_state["x"], ego_state["y"]]), lane_infos[right])["s"]

    proj_cache = candidate_projections.copy() if candidate_projections else {}
    for aid, st in candidate_states.items():
        if aid in proj_cache:
            continue
        p, _, _ = find_best_lane_for_agent(np.array([st["x"], st["y"]]), st.get("heading", np.nan), lane_infos, max_lat, max_hd,
            search_radius=float(cfg.get("lane_search_radius",20.0)), topk_candidates=int(cfg.get("lane_topk_candidates",32)),
            disable_spatial_index=bool(cfg.get("disable_lane_spatial_index",False)))
        if p is not None: proj_cache[aid] = p

    def choose(slot, lane_id, front=True):
        if lane_id not in ego_s_map: return None
        es = ego_s_map[lane_id]; rows = []
        for aid, p in proj_cache.items():
            if aid in used or p["lane_id"] != lane_id: continue
            ds = p["s"] - es
            if front and ds <= 0: continue
            if (not front) and ds >= 0: continue
            rows.append((abs(ds), aid, ds, p))
        return min(rows, key=lambda x: x[0]) if rows else None

    plan = [("front", cur, True), ("left_front", left, True), ("left_rear", left, False), ("right_front", right, True), ("right_rear", right, False)]
    for slot, lid, fr in plan:
        ch = choose(slot, lid, fr)
        if ch:
            used.add(ch[1]); slot_to_agent[slot] = ch[1]
            debug.append(dict(slot_name=slot, assignment_method="lane_aware", neighbor_id=str(ch[1]), fallback_used=False, fallback_reason="", neighbor_lane_id=lid,
                              ego_lane_id=cur, ego_s=ego_s_map.get(lid, np.nan), neighbor_s=ch[3]["s"], delta_s=ch[2], ego_l=ego_proj["l"], neighbor_l=ch[3]["l"], projection_distance=ch[3]["distance_to_lane"]))
        else:
            debug.append(dict(slot_name=slot, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason="no_laneaware_candidate", neighbor_lane_id=lid, ego_lane_id=cur))

    return SlotAssignResult(slot_to_agent, True, False, "", debug, cur, left, right, src)
