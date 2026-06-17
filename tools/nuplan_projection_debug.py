#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import math



def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _bucket_distance(d: float) -> str:
    if d < 10: return "00_10m"
    if d < 25: return "10_25m"
    if d < 50: return "25_50m"
    if d < 100: return "50_100m"
    return "100m_plus"


def _rel_xy(ex: float, ey: float, eh: float, x: float, y: float) -> tuple[float, float]:
    c = float(math.cos(eh)); s = float(math.sin(eh))
    dx = x - ex; dy = y - ey
    return c * dx + s * dy, -s * dx + c * dy


def _candidate_rejection_reason(result: Any, candidate_id: str, candidate_lane_id: str) -> tuple[str, str]:
    for slot_debug in result.per_slot_debug or []:
        if str(slot_debug.get("neighbor_id", "")) == str(candidate_id) and slot_debug.get("assignment_method") == "lane_aware":
            return "accepted", str(slot_debug.get("slot_name", ""))
    if result.fallback_assignment_used:
        return "geometric_fallback", str(result.fallback_reason or "")
    if not candidate_lane_id:
        return "candidate_projection_failed", "no_best_lane"
    slot_counts = result.slot_rejection_reason_counts or {}
    flat = Counter()
    for by_reason in slot_counts.values():
        if isinstance(by_reason, dict):
            flat.update({str(k): int(v) for k, v in by_reason.items()})
    reason = flat.most_common(1)[0][0] if flat else "not_selected"
    return reason, "aggregate_stage5d_rejection_counts"


def collect_nuplan_projection_debug_rows(
    *,
    global_row: int,
    scenario_index: int,
    planner_id: int,
    planner_name: str,
    map_name: str,
    assignment_mode: str,
    stage7c_seq: np.ndarray,
    mask: np.ndarray,
    tracks: Dict[str, Dict[int, tuple]],
    lane_infos: Dict[str, Any],
    assign_debug: List[dict],
    config: Dict[str, Any],
    max_frames_per_row: int,
    max_candidates_per_frame: int,
) -> List[Dict[str, Any]]:
    from tools.waymo_lane_utils import find_best_lane_for_agent, wrap_to_pi
    rows: List[Dict[str, Any]] = []
    if not assign_debug:
        return rows
    max_lat = float(config.get("lane_max_lateral_distance", 3.0))
    max_hd = math.radians(float(config.get("lane_max_heading_diff_deg", 45.0)))
    frame_indices = [int(i) for i in __import__("numpy").flatnonzero(mask)[:max_frames_per_row]]
    for t in frame_indices:
        ex = float(stage7c_seq[t, 0]); ey = float(stage7c_seq[t, 1]); eh = float(stage7c_seq[t, 2])
        ego_proj = None
        if lane_infos:
            ego_proj, _, _ = find_best_lane_for_agent(__import__("numpy").array([ex, ey]), eh, lane_infos, max_lat, max_hd,
                search_radius=float(config.get("lane_search_radius", 20.0)), topk_candidates=int(config.get("lane_topk_candidates", 32)),
                disable_spatial_index=bool(config.get("disable_lane_spatial_index", False)))
        dbg = assign_debug[t] if t < len(assign_debug) else {}
        slot_by_agent = {str(ps.get("neighbor_id")): str(ps.get("slot_name")) for ps in dbg.get("per_slot_debug", []) if ps.get("assignment_method") == "lane_aware" and ps.get("neighbor_id")}
        candidates = list(tracks.items())[:max_candidates_per_frame]
        for ci, (track_id, by_t) in enumerate(candidates):
            st = by_t.get(t)
            if st is None:
                continue
            x, y, vx, vy, heading, speed = [float(v) for v in st[:6]]
            rel_x, rel_y = _rel_xy(ex, ey, eh, x, y)
            dist = float(math.hypot(x - ex, y - ey))
            cand_proj = None
            if lane_infos:
                cand_proj, _, _ = find_best_lane_for_agent(__import__("numpy").array([x, y]), heading, lane_infos, max_lat, max_hd,
                    search_radius=float(config.get("lane_search_radius", 20.0)), topk_candidates=int(config.get("lane_topk_candidates", 32)),
                    disable_spatial_index=bool(config.get("disable_lane_spatial_index", False)))
            cand_lane = str(cand_proj.get("lane_id", "")) if cand_proj else ""
            ego_lane = str(ego_proj.get("lane_id", "")) if ego_proj else str(dbg.get("current_lane_id", ""))
            accepted = str(track_id) in slot_by_agent
            reason, detail = _candidate_rejection_reason(type("R", (), dbg) if False else _DebugObj(dbg), str(track_id), cand_lane)
            left = str(dbg.get("left_lane_id", "")); right = str(dbg.get("right_lane_id", ""))
            ego_info = lane_infos.get(ego_lane) if ego_lane else None
            cand_info = lane_infos.get(cand_lane) if cand_lane else None
            cand_in_left = bool(ego_info and cand_lane in set(ego_info.left_neighbor_lane_ids))
            cand_in_right = bool(ego_info and cand_lane in set(ego_info.right_neighbor_lane_ids))
            shares_topology = bool(ego_info and cand_info and (set(ego_info.entry_lane_ids) & set(cand_info.entry_lane_ids) or set(ego_info.exit_lane_ids) & set(cand_info.exit_lane_ids) or cand_lane in set(ego_info.entry_lane_ids + ego_info.exit_lane_ids) or ego_lane in set(cand_info.entry_lane_ids + cand_info.exit_lane_ids)))
            s_diff = None
            if cand_proj and ego_proj and cand_lane == ego_lane and cand_proj.get("s") is not None and ego_proj.get("s") is not None:
                s_diff = _finite_float(float(cand_proj["s"]) - float(ego_proj["s"]))
            relation = "same_lane" if cand_lane == ego_lane and cand_lane else ("left_adjacent" if cand_in_left or (cand_lane and cand_lane == left) else ("right_adjacent" if cand_in_right or (cand_lane and cand_lane == right) else "unknown"))
            if not ego_proj:
                relation_failure = "ego_projection_failed"
            elif not cand_proj:
                relation_failure = "candidate_projection_failed"
            elif cand_info and "connector" in str(cand_info.lane_type).lower() and not (cand_info.left_neighbor_lane_ids or cand_info.right_neighbor_lane_ids):
                relation_failure = "lane_connector_unhandled"
            elif relation != "unknown":
                relation_failure = "none"
            elif ego_info and not (ego_info.left_neighbor_lane_ids or ego_info.right_neighbor_lane_ids):
                relation_failure = "missing_adjacency"
            elif cand_proj and ego_proj and abs(wrap_to_pi(float(cand_proj.get("heading", heading)) - float(ego_proj.get("heading", eh)))) > math.radians(45.0):
                relation_failure = "direction_mismatch"
            elif not shares_topology:
                relation_failure = "topology_disconnected"
            else:
                relation_failure = "other"
            rows.append({
                "global_row": global_row, "scenario_index": scenario_index, "planner_id": planner_id, "planner_name": planner_name,
                "timestep": t, "candidate_index": ci, "track_id": track_id, "object_type": "", "map_name": map_name, "assignment_mode": assignment_mode,
                "ego_x": ex, "ego_y": ey, "ego_heading": eh, "ego_lane_id": ego_lane, "ego_lane_projection_success": bool(ego_proj),
                "ego_lane_lateral_offset": _finite_float((ego_proj or {}).get("l")), "ego_lane_heading_diff_deg": float(math.degrees(abs(wrap_to_pi(eh - float((ego_proj or {}).get("heading", eh)))))) if ego_proj else None,
                "ego_lane_s": _finite_float((ego_proj or {}).get("s")), "ego_lane_context_quality": dbg.get("lane_context_quality", "unknown"),
                "candidate_x": x, "candidate_y": y, "candidate_heading": heading, "candidate_speed": speed,
                "candidate_distance_to_ego": dist, "candidate_rel_x": rel_x, "candidate_rel_y": rel_y, "candidate_rel_heading_deg": float(math.degrees(wrap_to_pi(heading - eh))),
                "candidate_best_lane_id": cand_lane, "candidate_projection_success": bool(cand_proj),
                "candidate_lane_lateral_offset": _finite_float((cand_proj or {}).get("l")), "candidate_lane_heading_diff_deg": float(math.degrees(abs(wrap_to_pi(heading - float((cand_proj or {}).get("heading", heading)))))) if cand_proj else None,
                "candidate_lane_s": _finite_float((cand_proj or {}).get("s")), "candidate_lane_distance_to_ego_lane": None,
                "candidate_lane_type": getattr(cand_info, "lane_type", ""), "ego_lane_type": getattr(ego_info, "lane_type", ""),
                "candidate_in_ego_left_adjacency": cand_in_left, "candidate_in_ego_right_adjacency": cand_in_right,
                "shares_predecessor_or_successor_with_ego": shares_topology, "s_difference": s_diff,
                "relation_failure_category": relation_failure,
                "is_same_lane": bool(cand_lane and cand_lane == ego_lane), "is_left_adjacent": bool(cand_in_left or (cand_lane and cand_lane == left)), "is_right_adjacent": bool(cand_in_right or (cand_lane and cand_lane == right)),
                "is_successor_or_predecessor": shares_topology, "adjacency_source": getattr(ego_info, "topology_source", dbg.get("adjacency_source", "none")), "adjacency_confidence": "high" if getattr(ego_info, "topology_source", "") == "nuplan_topology" else ("medium" if getattr(ego_info, "topology_source", "") == "geometric_lane_adjacency" else "low"),
                "lane_relation_used_by_assignment": relation,
                "accepted_by_lane_aware": accepted, "assigned_slot": slot_by_agent.get(str(track_id), ""), "rejection_reason": "" if accepted else reason, "rejection_reason_detail": "" if accepted else detail,
            })
    return rows


class _DebugObj:
    def __init__(self, d: Dict[str, Any]):
        self.fallback_assignment_used = bool(d.get("fallback_assignment_used"))
        self.fallback_reason = d.get("fallback_reason", "")
        self.per_slot_debug = d.get("per_slot_debug", [])
        self.slot_rejection_reason_counts = d.get("slot_rejection_reason_counts", {})


def summarize_projection_debug(rows: List[Dict[str, Any]], assignment_debug_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cand_total = len(rows)
    cand_success = sum(1 for r in rows if r.get("candidate_projection_success"))
    ego_total = len(rows)
    ego_success = sum(1 for r in rows if r.get("ego_lane_projection_success"))
    by_type = defaultdict(lambda: [0, 0])
    by_dist = defaultdict(lambda: [0, 0])
    lat_vals: List[float] = []; hd_vals: List[float] = []
    rej = Counter(); rej_by_slot = defaultdict(Counter)
    for r in rows:
        ot = str(r.get("object_type") or "unknown"); by_type[ot][1] += 1; by_type[ot][0] += int(bool(r.get("candidate_projection_success")))
        db = _bucket_distance(float(r.get("candidate_distance_to_ego") or 0.0)); by_dist[db][1] += 1; by_dist[db][0] += int(bool(r.get("candidate_projection_success")))
        if r.get("candidate_lane_lateral_offset") is not None: lat_vals.append(abs(float(r["candidate_lane_lateral_offset"])))
        if r.get("candidate_lane_heading_diff_deg") is not None: hd_vals.append(abs(float(r["candidate_lane_heading_diff_deg"])))
        if not r.get("accepted_by_lane_aware"):
            reason = str(r.get("rejection_reason") or "unknown"); rej[reason] += 1; rej_by_slot[str(r.get("assigned_slot") or "unassigned")][reason] += 1
    fallback_no_ego = sum(1 for d in assignment_debug_rows if d.get("fallback_assignment_used") and str(d.get("fallback_reason", "")).startswith("ego"))
    fallback_projection = sum(1 for d in assignment_debug_rows if d.get("fallback_assignment_used") and d.get("fallback_reason") not in {"lane_map_unavailable", "no_candidate_for_slot", ""})
    unknown_categories = ["missing_adjacency", "topology_disconnected", "direction_mismatch", "candidate_projection_failed", "ego_projection_failed", "lane_connector_unhandled", "other"]
    unknown_breakdown_counter = Counter(str(r.get("relation_failure_category") or "other") for r in rows if r.get("lane_relation_used_by_assignment") == "unknown")
    unknown_breakdown = {k: int(unknown_breakdown_counter.get(k, 0)) for k in unknown_categories}
    for k, v in unknown_breakdown_counter.items():
        if k not in unknown_breakdown:
            unknown_breakdown[k] = int(v)
    return {
        "sampled_candidate_rows": cand_total,
        "ego_projection_success_rate": ego_success / ego_total if ego_total else None,
        "candidate_projection_success_rate": cand_success / cand_total if cand_total else None,
        "candidate_projection_success_rate_by_object_type": {k: v[0] / v[1] for k, v in by_type.items() if v[1]},
        "candidate_projection_success_rate_by_distance_bucket": {k: v[0] / v[1] for k, v in sorted(by_dist.items()) if v[1]},
        "rejection_reason_counts": dict(rej),
        "rejection_reason_counts_by_slot": {k: dict(v) for k, v in rej_by_slot.items()},
        "best_lane_lateral_offset_distribution": _dist(lat_vals),
        "best_lane_heading_diff_distribution": _dist(hd_vals),
        "same_lane_count": sum(1 for r in rows if r.get("is_same_lane")),
        "left_adjacent_count": sum(1 for r in rows if r.get("is_left_adjacent")),
        "right_adjacent_count": sum(1 for r in rows if r.get("is_right_adjacent")),
        "lane_relation_unknown_count": sum(1 for r in rows if r.get("lane_relation_used_by_assignment") == "unknown"),
        "lane_relation_unknown_breakdown": unknown_breakdown,
        "fallback_frames_due_to_no_ego_lane": fallback_no_ego,
        "fallback_frames_due_to_no_candidates": sum(1 for d in assignment_debug_rows if d.get("fallback_reason") == "no_candidate_for_slot"),
        "fallback_frames_due_to_projection_failure": fallback_projection,
        "fallback_frames_due_to_no_valid_lane_relation": sum(1 for d in assignment_debug_rows if d.get("fallback_assignment_used") and d.get("lane_context_quality") in {"bad", "ambiguous_intersection"}),
    }


def _dist(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "max": None}
    vals_sorted = sorted(float(v) for v in vals)
    def pct(p: float) -> float:
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        pos = (len(vals_sorted) - 1) * p / 100.0
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        return vals_sorted[lo] * (hi - pos) + vals_sorted[hi] * (pos - lo)
    return {"count": len(vals_sorted), "p50": pct(50), "p90": pct(90), "p95": pct(95), "max": max(vals_sorted)}


def write_projection_debug_artifacts(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any], write_csv_flag: bool) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    summary_path = out_dir / "nuplan_lane_projection_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts["summary_json"] = str(summary_path)
    report_path = out_dir / "nuplan_lane_projection_debug_report.md"
    lines = ["# nuPlan Lane Projection Debug", "", f"- sampled_candidate_rows: `{summary.get('sampled_candidate_rows')}`", f"- ego_projection_success_rate: `{summary.get('ego_projection_success_rate')}`", f"- candidate_projection_success_rate: `{summary.get('candidate_projection_success_rate')}`", f"- rejection_reason_counts: `{summary.get('rejection_reason_counts')}`", f"- lane_relation_unknown_breakdown: `{summary.get('lane_relation_unknown_breakdown')}`", f"- candidate_projection_success_rate_by_distance_bucket: `{summary.get('candidate_projection_success_rate_by_distance_bucket')}`"]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts["report_md"] = str(report_path)
    unknown_rows = [r for r in rows if r.get("lane_relation_used_by_assignment") == "unknown" or r.get("rejection_reason") == "wrong_lane"]
    if unknown_rows:
        unk_path = out_dir / "nuplan_lane_relation_unknown_debug.csv"
        fields = list(unknown_rows[0].keys())
        with unk_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(unknown_rows)
        artifacts["relation_unknown_csv"] = str(unk_path)
    if write_csv_flag:
        csv_path = out_dir / "nuplan_lane_projection_debug.csv"
        fields = list(rows[0].keys()) if rows else ["global_row", "timestep", "candidate_index"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        artifacts["csv"] = str(csv_path)
    return artifacts
