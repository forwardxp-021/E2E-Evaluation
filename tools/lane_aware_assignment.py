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
    lane_context_quality: str = "bad"
    lane_context_quality_reasons: List[str] = None
    slot_rejection_reason_counts: Dict[str, Dict[str, int]] = None


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
    return SlotAssignResult(slot_to_agent, False, True, reason, debug, lane_context_quality="fallback", lane_context_quality_reasons=["geometric_fallback_used"])


def assign_neighbors_lane_aware(ego_state, candidate_states, lane_infos=None, assignment_mode="lane_aware_with_geometric_fallback", config=None, ego_projection=None, candidate_projections=None):
    cfg = config or {}
    if assignment_mode == "geometric_only":
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, "geometric_only_mode")
    if not lane_infos:
        if assignment_mode == "lane_aware_only":
            return SlotAssignResult({}, False, False, "lane_map_unavailable", [dict(slot_name=s, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason="lane_map_unavailable") for s in SLOT_NAMES], lane_context_quality="bad", lane_context_quality_reasons=["lane_map_unavailable"])
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, "lane_map_unavailable")

    max_lat = float(cfg.get("lane_max_lateral_distance", 3.0)); max_hd = np.deg2rad(float(cfg.get("lane_max_heading_diff_deg", 45.0)));
    slot_hd = np.deg2rad(float(cfg.get("slot_heading_diff_deg", 45.0)))
    lat_tol = float(cfg.get("lane_lateral_tolerance", 2.0))
    front_max = float(cfg.get("front_max_distance", 120.0)); side_front_max = float(cfg.get("side_front_max_distance", 80.0)); side_rear_max = float(cfg.get("side_rear_max_distance", 120.0))
    static_th = float(cfg.get("static_speed_threshold", 0.5))

    ego_projection_precomputed = bool(cfg.get("ego_projection_precomputed", False))
    if ego_projection is None and not ego_projection_precomputed:
        ego_proj, reason, _ = find_best_lane_for_agent(np.array([ego_state["x"], ego_state["y"]]), ego_state["heading"], lane_infos, max_lat, max_hd,
            search_radius=float(cfg.get("lane_search_radius",20.0)), topk_candidates=int(cfg.get("lane_topk_candidates",32)),
            disable_spatial_index=bool(cfg.get("disable_lane_spatial_index",False)))
    elif ego_projection is None:
        ego_proj, reason = None, "precomputed_ego_projection_failed"
    else:
        ego_proj, reason = ego_projection, "ok"
    if ego_proj is None:
        if assignment_mode == "lane_aware_only":
            return SlotAssignResult({}, False, False, reason, [dict(slot_name=s, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason=reason) for s in SLOT_NAMES], lane_context_quality="bad", lane_context_quality_reasons=[f"ego_projection_failed:{reason}"])
        return assign_neighbors_geometric(np.array([ego_state["x"], ego_state["y"], ego_state["heading"]], np.float32), {k: np.array([v["x"], v["y"]]) for k, v in candidate_states.items()}, reason)

    cur = ego_proj["lane_id"]; left = ""; right = ""; src = "none"
    ln = lane_infos[cur]
    def active_proto_neighbors(relations):
        segment = int(ego_proj.get("segment_index", -1))
        return [relation for relation in relations if relation["self_start_index"] <= segment <= relation["self_end_index"]]

    left_relations = getattr(ln, "left_neighbor_relations", [])
    right_relations = getattr(ln, "right_neighbor_relations", [])
    left_relation = None
    right_relation = None
    left_options = []
    right_options = []
    if left_relations or right_relations:
        active_left = active_proto_neighbors(left_relations)
        active_right = active_proto_neighbors(right_relations)
        left_relation = active_left[0] if active_left else None
        right_relation = active_right[0] if active_right else None
        left_options = [(relation["lane_id"], relation) for relation in active_left]
        right_options = [(relation["lane_id"], relation) for relation in active_right]
        left = left_relation["lane_id"] if left_relation else ""
        right = right_relation["lane_id"] if right_relation else ""
        src = "proto_topology"
    elif ln.left_neighbor_lane_ids or ln.right_neighbor_lane_ids:
        # Backward-compatible synthetic/test LaneInfo without interval metadata.
        left = ln.left_neighbor_lane_ids[0] if ln.left_neighbor_lane_ids else ""
        right = ln.right_neighbor_lane_ids[0] if ln.right_neighbor_lane_ids else ""
        left_options = [(left, None)] if left else []
        right_options = [(right, None)] if right else []
        src = "proto_topology_without_local_range"
    elif bool(cfg.get("allow_geometric_adjacent_lane_inference", True)):
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
        left_options = [(left, None)] if left else []
        right_options = [(right, None)] if right else []
    else:
        src = "none"

    slot_to_agent = {}; used = set(); debug = []
    ego_s_map = {cur: ego_proj["s"]}
    for lane_id, _ in left_options + right_options:
        if lane_id in lane_infos and lane_id not in ego_s_map:
            ego_s_map[lane_id] = project_point_to_lane(np.array([ego_state["x"], ego_state["y"]]), lane_infos[lane_id])["s"]

    proj_cache = candidate_projections.copy() if candidate_projections else {}
    if not bool(cfg.get("candidate_projections_complete", False)):
        for aid, st in candidate_states.items():
            if aid in proj_cache:
                continue
            p, _, _ = find_best_lane_for_agent(np.array([st["x"], st["y"]]), st.get("heading", np.nan), lane_infos, max_lat, max_hd,
                search_radius=float(cfg.get("lane_search_radius",20.0)), topk_candidates=int(cfg.get("lane_topk_candidates",32)),
                disable_spatial_index=bool(cfg.get("disable_lane_spatial_index",False)))
            if p is not None:
                proj_cache[aid] = p

    rejection_counts = {s: {"too_far":0,"lateral_offset_too_large":0,"heading_diff_too_large":0,"wrong_lane":0,"wrong_direction_s":0,"no_candidate":0} for s in SLOT_NAMES}

    def choose(slot, lane_id, min_ds, max_ds, neighbor_relation=None):
        if lane_id not in ego_s_map:
            rejection_counts[slot]["no_candidate"] += 1
            return None
        es = ego_s_map[lane_id]; rows = []
        for aid, p in proj_cache.items():
            if aid in used:
                continue
            if p.get("lane_id", "") != lane_id:
                rejection_counts[slot]["wrong_lane"] += 1
                continue
            if neighbor_relation is not None:
                candidate_segment = int(p.get("segment_index", -1))
                if not (neighbor_relation["neighbor_start_index"] <= candidate_segment <= neighbor_relation["neighbor_end_index"]):
                    rejection_counts[slot]["wrong_lane"] += 1
                    continue
            ds = float(p["s"] - es)
            if not (min_ds <= ds <= max_ds):
                if ds == 0.0 or (min_ds >= 0.0 and ds < min_ds) or (max_ds <= 0.0 and ds > max_ds):
                    rejection_counts[slot]["wrong_direction_s"] += 1
                else:
                    rejection_counts[slot]["too_far"] += 1
                continue
            nl = float(p.get("l", np.nan)); prd = float(p.get("distance_to_lane", np.nan))
            if not np.isfinite(nl) or abs(nl) > lat_tol:
                rejection_counts[slot]["lateral_offset_too_large"] += 1
                continue
            hd = abs(float(wrap_to_pi(float(candidate_states[aid].get("heading", 0.0)) - float(p.get("heading", 0.0)))))
            if not np.isfinite(hd) or hd > slot_hd:
                rejection_counts[slot]["heading_diff_too_large"] += 1
                continue
            spd = float(candidate_states[aid].get("speed", np.hypot(candidate_states[aid].get("velocity_x", 0.0), candidate_states[aid].get("velocity_y", 0.0))))
            rows.append((abs(ds), prd, abs(nl), hd, aid, ds, nl, spd, p))
        if not rows:
            rejection_counts[slot]["no_candidate"] += 1
            return None
        return sorted(rows, key=lambda x: (x[0], x[1], x[2], x[3]))[0]

    plan = [
        ("front", [(cur, None)], 0.0, front_max),
        ("left_front", left_options, 0.0, side_front_max),
        ("left_rear", left_options, -side_rear_max, 0.0),
        ("right_front", right_options, 0.0, side_front_max),
        ("right_rear", right_options, -side_rear_max, 0.0),
    ]
    for slot, options, mn, mx in plan:
        choices = []
        for lid, relation in options:
            candidate = choose(slot, lid, mn + (1e-6 if mn >= 0 else 0.0), mx - (1e-6 if mx <= 0 else 0.0), relation)
            if candidate is not None:
                choices.append((candidate, lid))
        ch, lid = min(choices, key=lambda value: value[0][:4]) if choices else (None, options[0][0] if options else "")
        if ch:
            used.add(ch[4]); slot_to_agent[slot] = ch[4]
            debug.append(dict(slot_name=slot, assignment_method="lane_aware", neighbor_id=str(ch[4]), fallback_used=False, fallback_reason="", neighbor_lane_id=lid,
                              ego_lane_id=cur, slot_lane_id=lid, ego_s=ego_s_map.get(lid, np.nan), neighbor_s=ch[8]["s"], delta_s=ch[5], ego_l=ego_proj.get("l", np.nan), neighbor_l=ch[6], projection_distance=ch[1],
                              candidate_lateral_offset=ch[6], candidate_heading_diff=float(np.rad2deg(ch[3])), neighbor_speed=ch[7], neighbor_is_static=bool(ch[7] < static_th),
                              distance_threshold_used=front_max if slot=="front" else (side_front_max if 'front' in slot else side_rear_max), lane_lateral_tolerance=lat_tol, slot_heading_diff_threshold=float(np.rad2deg(slot_hd))))
        else:
            debug.append(dict(slot_name=slot, assignment_method="empty", neighbor_id="", fallback_used=False, fallback_reason="no_laneaware_candidate", neighbor_lane_id=lid, ego_lane_id=cur, slot_lane_id=lid,
                              candidate_lateral_offset=np.nan, candidate_heading_diff=np.nan, neighbor_speed=np.nan, neighbor_is_static=False,
                              distance_threshold_used=front_max if slot=="front" else (side_front_max if 'front' in slot else side_rear_max), lane_lateral_tolerance=lat_tol, slot_heading_diff_threshold=float(np.rad2deg(slot_hd)),
                              rejection_reason="no_laneaware_candidate"))

    lane_context_quality = "good"
    reasons = ["current_lane_found", "empty_slots_allowed"]
    if src == "proto_topology":
        reasons.append("proto_topology_adjacency")
    elif src == "geometric":
        reasons.append("geometric_adjacency")
    else:
        reasons.append("adjacency_source_none")
        lane_context_quality = "ambiguous_intersection"

    ego_dist = float(ego_proj.get("distance_to_lane", np.nan))
    ego_hd = abs(float(wrap_to_pi(float(ego_state.get("heading", 0.0)) - float(ego_proj.get("heading", 0.0)))))
    near_lat = 0.8 * max_lat
    near_hd = 0.8 * max_hd
    if np.isfinite(ego_dist) and ego_dist > near_lat:
        reasons.append("ego_projection_distance_near_threshold")
        if lane_context_quality == "good":
            lane_context_quality = "ambiguous_intersection"
    if np.isfinite(ego_hd) and ego_hd > near_hd:
        reasons.append("ego_heading_diff_near_threshold")
        if lane_context_quality == "good":
            lane_context_quality = "ambiguous_intersection"
    return SlotAssignResult(slot_to_agent, True, False, "", debug, cur, left, right, src, lane_context_quality, reasons, rejection_counts)
