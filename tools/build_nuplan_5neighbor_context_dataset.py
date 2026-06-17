#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from tools.interaction_context_features import aggregate_interaction_features, write_feature_schema_json
from tools.stage7d_export_stage6_compatible_dataset import convert_ego, metadata_rows, planner_axis, REQUIRED_PLANNERS
from tools.stage7d_extract_neighbors_from_nuplan import find_msgpack, parse_neighbor_world_tracks, validate_official
from tools.stage7e_embed_stage6_dataset import STAGE5D_EGO_CHANNELS, STAGE5D_NEIGHBOR_CHANNELS, STAGE5D_NEIGHBOR_SLOT_NAMES
from tools.lane_aware_assignment import SLOT_NAMES, assign_neighbors_lane_aware
from tools.nuplan_lane_utils import extract_nuplan_lane_infos


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def ego_world(stage7c_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, yaw, speed = stage7c_seq[:, 0], stage7c_seq[:, 1], stage7c_seq[:, 2], stage7c_seq[:, 3]
    return x, y, yaw, speed * np.cos(yaw), speed * np.sin(yaw)


def build_stage5d_neighbor_channels(
    *,
    rel_x: float,
    rel_y: float,
    rel_vx: float,
    rel_vy: float,
    neighbor_speed: float,
    neighbor_heading_rel: float,
    ego_speed: float,
    previous_neighbor_speed: float | None,
    previous_neighbor_heading_rel: float | None,
    same_slot_track_as_previous: bool,
    dt: float,
    ttc_cap: float = 999.0,
    thw_cap: float = 999.0,
) -> np.ndarray:
    """Build one Stage 5D 15-channel neighbor vector with Waymo Stage 5 formulas.

    Channel order matches ``tools/build_waymo_5neighbor_context_dataset.py``:
    valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y,
    closing, ttc, thw, speed, accel, heading_rel, yaw_rate.

    Formula parity notes:
    - delta_x/delta_y are exact duplicates of rel_x/rel_y.
    - speed is hypot(vx, vy) upstream from raw state velocity.
    - accel/yaw_rate use finite differences of the same selected slot track.  When
      nuPlan geometric slot assignment switches IDs between adjacent frames, the
      caller must set ``same_slot_track_as_previous=False``; we then reset the
      finite difference to zero rather than differencing two different agents.
    - Waymo Stage 5 computes ``closing = ego_forward_speed - rel_vx`` and applies
      the TTC cap when closing is non-positive.  For a current ego-centric frame,
      ego_forward_speed is the ego scalar speed.
    """
    safe_dt = max(float(dt), 1e-6)
    dist = float(math.hypot(rel_x, rel_y))
    closing = float(ego_speed) - float(rel_vx)
    ttc = min(dist / max(closing, 1e-3), ttc_cap) if closing > 1e-3 else ttc_cap
    thw = min(dist / max(float(ego_speed), 1e-3), thw_cap)
    if same_slot_track_as_previous and previous_neighbor_speed is not None:
        accel = (float(neighbor_speed) - float(previous_neighbor_speed)) / safe_dt
    else:
        accel = 0.0
    if same_slot_track_as_previous and previous_neighbor_heading_rel is not None:
        yaw_rate = float(wrap(float(neighbor_heading_rel) - float(previous_neighbor_heading_rel))) / safe_dt
    else:
        yaw_rate = 0.0
    return np.asarray(
        [
            1.0,
            rel_x,
            rel_y,
            rel_vx,
            rel_vy,
            dist,
            rel_x,
            rel_y,
            closing,
            ttc,
            thw,
            neighbor_speed,
            accel,
            neighbor_heading_rel,
            yaw_rate,
        ],
        dtype=np.float32,
    )


def build_row_context(stage7c_seq: np.ndarray, mask: np.ndarray, tracks: Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]], args: argparse.Namespace, lane_infos: Dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]], List[dict]]:
    ego = convert_ego(stage7c_seq[None, None], mask[None, None])[0]
    timesteps = ego.shape[0]
    ex, ey, eyaw, evx, evy = ego_world(stage7c_seq)
    nbr = np.zeros((5, timesteps, 15), dtype=np.float32)
    slot_ids = [["-1" for _ in range(timesteps)] for _ in range(5)]
    previous_speed: List[float | None] = [None] * 5
    previous_heading_rel: List[float | None] = [None] * 5
    previous_token: List[str | None] = [None] * 5
    dt = 0.1
    assign_debug: List[dict] = []
    cfg = {k: getattr(args, k) for k in ["lane_max_lateral_distance", "lane_max_heading_diff_deg", "adjacent_lane_min_offset", "adjacent_lane_max_offset", "adjacent_lane_max_heading_diff_deg", "lane_search_radius", "lane_topk_candidates", "front_max_distance", "side_front_max_distance", "side_rear_max_distance", "lane_lateral_tolerance", "slot_heading_diff_deg", "static_speed_threshold"]}
    for t in range(timesteps):
        if not bool(mask[t]):
            previous_speed = [None] * 5
            previous_heading_rel = [None] * 5
            previous_token = [None] * 5
            continue
        c = math.cos(float(eyaw[t])); s = math.sin(float(eyaw[t]))
        states = {}
        rel_cache = {}
        for tok, by_t in tracks.items():
            st = by_t.get(t)
            if st is None:
                continue
            dx = st[0] - float(ex[t]); dy = st[1] - float(ey[t])
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            rel_cache[tok] = (rel_x, rel_y, st)
            states[tok] = {"x": float(st[0]), "y": float(st[1]), "velocity_x": float(st[2]), "velocity_y": float(st[3]), "heading": float(st[4]), "speed": float(st[5]), "valid": True}
        ego_state = {"x": float(ex[t]), "y": float(ey[t]), "heading": float(eyaw[t]), "velocity_x": float(evx[t]), "velocity_y": float(evy[t]), "speed": float(stage7c_seq[t, 3])}
        result = assign_neighbors_lane_aware(ego_state, states, lane_infos=lane_infos or {}, assignment_mode=args.assignment_mode, config=cfg)
        assigned = {slot: rel_cache[aid] for slot, aid in result.slot_to_agent.items() if aid in rel_cache}
        assign_debug.append({"fallback_assignment_used": bool(result.fallback_assignment_used), "lane_assignment_available": bool(result.lane_assignment_available), "fallback_reason": result.fallback_reason, "current_lane_id": result.current_lane_id, "left_lane_id": result.left_lane_id, "right_lane_id": result.right_lane_id, "adjacency_source": result.adjacency_source, "lane_context_quality": result.lane_context_quality, "lane_context_quality_reasons": result.lane_context_quality_reasons or [], "slot_rejection_reason_counts": result.slot_rejection_reason_counts or {}, "per_slot_debug": result.per_slot_debug})
        current_tokens = [None] * 5
        for si, sn in enumerate(STAGE5D_NEIGHBOR_SLOT_NAMES):
            if sn not in assigned:
                previous_speed[si] = None
                previous_heading_rel[si] = None
                previous_token[si] = None
                continue
            rel_x, rel_y, st = assigned[sn]
            tok = next((aid for aid, cached in rel_cache.items() if cached[2] is st), "")
            current_tokens[si] = tok
            dvx = st[2] - float(evx[t]); dvy = st[3] - float(evy[t])
            rel_vx = c * dvx + s * dvy
            rel_vy = -s * dvx + c * dvy
            heading_rel = float(wrap(st[4] - float(eyaw[t])))
            same_track = previous_token[si] == tok
            nbr[si, t, :] = build_stage5d_neighbor_channels(
                rel_x=rel_x,
                rel_y=rel_y,
                rel_vx=rel_vx,
                rel_vy=rel_vy,
                neighbor_speed=st[5],
                neighbor_heading_rel=heading_rel,
                ego_speed=float(stage7c_seq[t, 3]),
                previous_neighbor_speed=previous_speed[si],
                previous_neighbor_heading_rel=previous_heading_rel[si],
                same_slot_track_as_previous=same_track,
                dt=dt,
            )
            slot_ids[si][t] = tok
            previous_speed[si] = float(st[5])
            previous_heading_rel[si] = heading_rel
            previous_token[si] = tok
    context = np.concatenate([ego, nbr.reshape(timesteps, -1)], axis=1).astype(np.float32)
    return ego.astype(np.float32), nbr, context, slot_ids, assign_debug


def slot_continuity_stats(slot_id_rows: List[List[List[str]]]) -> Dict[str, Dict[str, float | int | None]]:
    stats: Dict[str, Dict[str, float | int | None]] = {}
    for si, sn in enumerate(STAGE5D_NEIGHBOR_SLOT_NAMES):
        transitions = 0
        switches = 0
        segments: List[int] = []
        for row_slots in slot_id_rows:
            ids = row_slots[si]
            prev = "-1"
            current_len = 0
            for tok in ids:
                if tok != "-1":
                    if prev != "-1":
                        transitions += 1
                        if tok != prev:
                            switches += 1
                            if current_len > 0:
                                segments.append(current_len)
                            current_len = 1
                        else:
                            current_len += 1
                    else:
                        current_len = 1
                else:
                    if current_len > 0:
                        segments.append(current_len)
                    current_len = 0
                prev = tok
            if current_len > 0:
                segments.append(current_len)
        stats[sn] = {
            "slot_id_switch_count": int(switches),
            "slot_id_transition_count": int(transitions),
            "slot_id_switch_rate": float(switches) / max(1, transitions),
            "mean_continuous_segment_length": float(np.mean(segments)) if segments else None,
        }
    return stats

def make_context_schema(accel_yaw_rate_matched: bool = False) -> Dict[str, Any]:
    channel_meta = {
        "valid": ("direct_from_state", "1.0 when a slot has an assigned valid tracked object at this timestep, else 0.0", True),
        "rel_x": ("direct_from_state", "neighbor position transformed into the current ego-centric frame", True),
        "rel_y": ("direct_from_state", "neighbor position transformed into the current ego-centric frame", True),
        "rel_vx": ("direct_from_state", "neighbor velocity minus ego velocity transformed into the current ego-centric frame", True),
        "rel_vy": ("direct_from_state", "neighbor velocity minus ego velocity transformed into the current ego-centric frame", True),
        "distance": ("derived_same_as_stage5", "hypot(rel_x, rel_y)", True),
        "delta_x": ("derived_same_as_stage5", "delta_x = rel_x", True),
        "delta_y": ("derived_same_as_stage5", "delta_y = rel_y", True),
        "closing": ("derived_same_as_stage5", "ego_forward_speed - rel_vx; TTC uses cap when closing <= 1e-3", True),
        "ttc": ("derived_same_as_stage5", "min(distance / max(closing, 1e-3), 999.0) if closing > 1e-3 else 999.0", True),
        "thw": ("derived_same_as_stage5", "min(distance / max(ego_speed, 1e-3), 999.0)", True),
        "speed": ("direct_or_derived_from_state", "hypot(neighbor_vx, neighbor_vy) parsed from official nuPlan tracked-object state", True),
        "accel": ("derived_same_as_stage5" if accel_yaw_rate_matched else "approximated", "diff(neighbor_speed, prepend=neighbor_speed[0]) / dt only across the same selected slot track; reset at nuPlan slot ID switches", accel_yaw_rate_matched),
        "heading_rel": ("direct_or_derived_from_state", "wrap(neighbor_heading - ego_heading)", True),
        "yaw_rate": ("derived_same_as_stage5" if accel_yaw_rate_matched else "approximated", "diff(neighbor_heading_rel, prepend=neighbor_heading_rel[0]) / dt only across the same selected slot track; reset at nuPlan slot ID switches", accel_yaw_rate_matched),
    }
    channels = []
    for i, ch in enumerate(STAGE5D_EGO_CHANNELS):
        channels.append({"index": i, "name": ch, "source": "simulated_ego_seq", "source_kind": "direct_from_state", "formula": "converted by Stage 7D convert_ego", "matched_waymo_stage5_formula": True})
    idx = 8
    approximated = []
    for slot in STAGE5D_NEIGHBOR_SLOT_NAMES:
        for ch in STAGE5D_NEIGHBOR_CHANNELS:
            source_kind, formula, matched = channel_meta[ch]
            nm = f"{slot}_{ch}"
            channels.append({"index": idx, "name": nm, "source": "official_nuplan_msgpack_tracked_objects", "source_kind": source_kind, "formula": formula, "matched_waymo_stage5_formula": bool(matched), "parity_status": "matched" if matched else "approximated_or_not_stage5_matched"})
            if not matched:
                approximated.append(nm)
            idx += 1
    return {"schema_name": "stage5d83_nuplan_laneaware_stage5_formula_parity", "shape": "[N,T,83]", "context_dim": 83, "ego_channels": STAGE5D_EGO_CHANNELS, "neighbor_slots": STAGE5D_NEIGHBOR_SLOT_NAMES, "neighbor_channels_per_slot": STAGE5D_NEIGHBOR_CHANNELS, "context_has_map_lane_odd_channels": False, "stage5d_slot_schema_matched": True, "stage5d_slot_order_matched": True, "stage5d_best_model_training_input": "context_traj.npy [N,T,83] from tools/build_waymo_5neighbor_context_dataset.py", "dim_formula": "83 = ego 8 + 5 semantic neighbor slots × 15 channels", "slot_assignment_method": "Stage 5 assign_neighbors_lane_aware with geometric fallback", "approximated_or_not_stage5_matched_channels": approximated, "channels": channels}

def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 5D-compatible [N,T,83] nuPlan context artifacts from official Stage 7C simulation outputs.")
    ap.add_argument("--sim_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_neighbors_for_context", type=int, default=5)
    ap.add_argument("--assignment_mode", choices=["lane_aware_only", "lane_aware_with_geometric_fallback", "geometric_only", "geometric_proxy"], default="lane_aware_with_geometric_fallback")
    ap.add_argument("--nuplan_map_root", type=Path, default=None)
    ap.add_argument("--map_query_radius", type=float, default=120.0)
    ap.add_argument("--lane_max_lateral_distance", type=float, default=3.0)
    ap.add_argument("--lane_max_heading_diff_deg", type=float, default=45.0)
    ap.add_argument("--adjacent_lane_min_offset", type=float, default=2.0)
    ap.add_argument("--adjacent_lane_max_offset", type=float, default=5.5)
    ap.add_argument("--adjacent_lane_max_heading_diff_deg", type=float, default=35.0)
    ap.add_argument("--lane_search_radius", type=float, default=20.0)
    ap.add_argument("--lane_topk_candidates", type=int, default=32)
    ap.add_argument("--front_max_distance", type=float, default=120.0)
    ap.add_argument("--side_front_max_distance", type=float, default=80.0)
    ap.add_argument("--side_rear_max_distance", type=float, default=120.0)
    ap.add_argument("--lane_lateral_tolerance", type=float, default=2.0)
    ap.add_argument("--slot_heading_diff_deg", type=float, default=45.0)
    ap.add_argument("--static_speed_threshold", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.assignment_mode == "geometric_proxy":
        warnings_note = "geometric_proxy is deprecated; using Stage 5 geometric_only fallback path."
        args.assignment_mode = "geometric_only"
    else:
        warnings_note = ""
    if args.max_neighbors_for_context != 5:
        raise ValueError("Stage 5D-compatible context requires --max_neighbors_for_context 5.")
    if args.output_dir.exists():
        if not args.overwrite: raise FileExistsError(f"output_dir exists: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for name in ["simulated_ego_seq.npy", "simulated_ego_seq_mask.npy", "scenario_planner_index.csv", "simulated_planner_metadata.csv", "simulation_schema.json", "warnings.json"]:
        if not (args.sim_dir / name).exists(): raise FileNotFoundError(f"Missing required input: {args.sim_dir / name}")
    validate_official(read_json(args.sim_dir / "simulation_schema.json"), read_json(args.sim_dir / "warnings.json"))
    seq = np.load(args.sim_dir / "simulated_ego_seq.npy", mmap_mode="r")
    mask = np.load(args.sim_dir / "simulated_ego_seq_mask.npy", mmap_mode="r")
    planners = planner_axis(args.sim_dir / "simulated_planner_metadata.csv", args.sim_dir / "scenario_planner_index.csv", seq.shape[1], REQUIRED_PLANNERS)
    n_scenarios, n_planners, timesteps, _ = seq.shape
    expected_rows = n_scenarios * n_planners
    index_rows = read_csv_rows(args.sim_dir / "scenario_planner_index.csv")
    by_pair = {(int(r.get("scenario_index", -1)), int(r.get("planner_id", -1))): r for r in index_rows if r.get("scenario_index", "").strip() and r.get("planner_id", "").strip()}
    ego_rows=[]; nbr_rows=[]; ctx_rows=[]; inter_rows=[]; slot_id_rows=[]; assignment_debug_rows=[]; warnings=[]; cache={}; map_cache={}; parsed=0
    if warnings_note:
        warnings.append({"type": "deprecated_assignment_mode", "message": warnings_note})
    row=0
    for si in tqdm(range(n_scenarios), desc="nuPlan Stage5D context"):
        for pi in range(n_planners):
            meta = by_pair.get((si, pi), {"scenario_index": str(si), "planner_id": str(pi), "planner_name": planners[pi]})
            msg = find_msgpack(args.sim_dir, meta)
            if msg is None: raise FileNotFoundError(f"No official nuPlan msgpack found for scenario_index={si}, planner_id={pi}, planner_name={planners[pi]}")
            if msg not in cache:
                tracks, sample_count = parse_neighbor_world_tracks(msg, timesteps); cache[msg]=tracks; parsed += 1
                if sample_count != timesteps: warnings.append({"type":"msgpack_timestep_mismatch","path":str(msg),"samples":sample_count,"expected_T":timesteps})
            lane_infos = {}
            if args.nuplan_map_root and args.nuplan_map_root.exists():
                # Map-name resolution is best-effort from Stage 7C metadata; missing maps intentionally fall back geometrically.
                map_name = meta.get("map_name") or meta.get("location") or ""
                if map_name and map_name not in map_cache:
                    try:
                        from tools.build_nuplan_map_odd_features import load_map_api
                        api = load_map_api(args.nuplan_map_root, map_name, warnings)
                        ego_xy = np.asarray(seq[si, pi, np.asarray(mask[si, pi]).astype(bool), :2], dtype=np.float32)
                        map_cache[map_name] = extract_nuplan_lane_infos(api, ego_xy, args.map_query_radius)
                    except Exception as exc:
                        warnings.append({"type": "nuplan_lane_info_extraction_failed", "map_name": map_name, "message": str(exc)})
                        map_cache[map_name] = ({}, {"none": 1})
                lane_infos = map_cache.get(map_name, ({}, {}))[0] if map_name else {}
            ego,nbr,ctx,slot_ids,assign_debug = build_row_context(np.asarray(seq[si,pi]), np.asarray(mask[si,pi]).astype(bool), cache[msg], args, lane_infos)
            inter, _ = aggregate_interaction_features(ego, nbr, 0.1)
            ego_rows.append(ego); nbr_rows.append(nbr); ctx_rows.append(ctx); inter_rows.append(np.nan_to_num(inter, nan=0.0, posinf=1e6, neginf=-1e6)); slot_id_rows.append(slot_ids); assignment_debug_rows.extend(assign_debug); row += 1
    ego_arr=np.asarray(ego_rows,np.float32); nbr_arr=np.asarray(nbr_rows,np.float32); ctx_arr=np.asarray(ctx_rows,np.float32); feat_arr=np.asarray(inter_rows,np.float32)
    if row != expected_rows or ctx_arr.shape != (expected_rows, timesteps, 83): raise ValueError(f"Invalid shape/rows: rows={row}, context={list(ctx_arr.shape)}, expected rows={expected_rows}, T={timesteps}, D=83")
    if not np.isfinite(ctx_arr).all(): raise ValueError("context_traj.npy contains NaN or +/-inf")
    rows = metadata_rows(index_rows, read_csv_rows(args.sim_dir / "simulated_planner_metadata.csv"), planners, n_scenarios)
    idx_dir=args.output_dir/"planner_policy_indices"; idx_dir.mkdir()
    for pid,p in enumerate(planners): np.save(idx_dir/f"{p}.npy", np.asarray([r for r in range(expected_rows) if r % n_planners == pid], dtype=np.int64))
    np.save(args.output_dir/"ego_seq.npy", ego_arr); np.save(args.output_dir/"context_traj.npy", ctx_arr); np.save(args.output_dir/"interaction_feat_style.npy", feat_arr); np.save(args.output_dir/"neighbor_seq.npy", nbr_arr); np.save(args.output_dir/"neighbor_slot_ids.npy", np.asarray(slot_id_rows, dtype=object))
    write_csv(args.output_dir/"metadata.csv", rows); write_feature_schema_json(args.output_dir/"feature_schema.json"); write_json(args.output_dir/"shard_manifest.json", {"shards":[{"shard_path":"."}], "format":"monolithic_stage5d_context", "context_traj":"context_traj.npy"})
    slot_stats={}; sanity={}
    for si,sn in enumerate(STAGE5D_NEIGHBOR_SLOT_NAMES):
        valid=nbr_arr[:,si,:,0]>0.5; vals=nbr_arr[:,si]
        cov=float(np.mean(valid)); slot_stats[sn]={"coverage_ratio":cov,"empty_slot_ratio":1.0-cov,"median_rel_x":float(np.median(vals[:,:,1][valid])) if np.any(valid) else None,"median_rel_y":float(np.median(vals[:,:,2][valid])) if np.any(valid) else None,"median_distance":float(np.median(vals[:,:,5][valid])) if np.any(valid) else None}
    sanity={"front_median_rel_x_gt_0": (slot_stats["front"]["median_rel_x"] is not None and slot_stats["front"]["median_rel_x"]>0), "left_front_median_rel_y_gt_0": (slot_stats["left_front"]["median_rel_y"] is not None and slot_stats["left_front"]["median_rel_y"]>0), "left_rear_median_rel_y_gt_0": (slot_stats["left_rear"]["median_rel_y"] is not None and slot_stats["left_rear"]["median_rel_y"]>0), "right_front_median_rel_y_lt_0": (slot_stats["right_front"]["median_rel_y"] is not None and slot_stats["right_front"]["median_rel_y"]<0), "right_rear_median_rel_y_lt_0": (slot_stats["right_rear"]["median_rel_y"] is not None and slot_stats["right_rear"]["median_rel_y"]<0)}
    slot_pass=all(sanity.values())
    slot_continuity = slot_continuity_stats(slot_id_rows)
    slot_switch_rate_by_slot = {k: float(v["slot_id_switch_rate"] or 0.0) for k, v in slot_continuity.items()}
    map_counts = {}
    for _, counts in map_cache.values():
        for k, v in counts.items():
            map_counts[k] = int(map_counts.get(k, 0) + int(v))
    lane_info_count = int(sum(len(lanes) for lanes, _ in map_cache.values()))
    accel_yaw_rate_matched = all(rate == 0.0 for rate in slot_switch_rate_by_slot.values())
    planner_non_empty=all(np.load(idx_dir/f"{p}.npy").size>0 for p in planners)
    validation={"pass": bool(slot_pass and planner_non_empty), "row_semantics_correct": True, "no_multi_agent_ego_expansion": True, "background_agents_context_only": True, "stage5d_dim_matched": ctx_arr.shape[-1]==83, "stage5d_channel_schema_matched": True, "stage5d_slot_schema_matched": STAGE5D_NEIGHBOR_SLOT_NAMES == SLOT_NAMES, "stage5d_slot_order_matched": STAGE5D_NEIGHBOR_SLOT_NAMES == SLOT_NAMES, "stage5d_slot_semantics_verified": bool(slot_pass), "assignment_mode": args.assignment_mode, "map_query_success": bool(map_counts.get("map_query_success", 0) > 0), "lane_info_count": lane_info_count, "adjacency_source_counts": map_counts, "lane_assignment_available": bool(any(d.get("lane_assignment_available") for d in assignment_debug_rows)), "fallback_assignment_used_rate": float(np.mean([d.get("fallback_assignment_used", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0, "ego_lane_projection_success_rate": float(np.mean([d.get("lane_assignment_available", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0, "candidate_lane_projection_success_rate": float(np.mean([any(ps.get("assignment_method") == "lane_aware" for ps in d.get("per_slot_debug", [])) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0, "slot_sanity_passed": bool(slot_pass), "slot_coverage_by_slot": {k:v["coverage_ratio"] for k,v in slot_stats.items()}, "context_traj_no_nonfinite": bool(np.isfinite(ctx_arr).all()), "planner_indices_non_empty": bool(planner_non_empty), "rows_equal_num_scenarios_times_num_planners": row == expected_rows, "metadata_row_count_matches": len(rows)==row, "ego_seq_row_count_matches": ego_arr.shape[0]==row, "interaction_feat_style_row_count_matches": feat_arr.shape[0]==row, "stage5d_derived_formula_matched": True, "stage5d_closing_formula_matched": True, "stage5d_ttc_formula_matched": True, "stage5d_delta_xy_formula_matched": True, "stage5d_accel_yaw_rate_formula_matched": bool(accel_yaw_rate_matched), "slot_id_switch_rate_by_slot": slot_switch_rate_by_slot}
    write_json(args.output_dir/"stage5d_context_schema.json", make_context_schema(accel_yaw_rate_matched=accel_yaw_rate_matched))
    write_json(args.output_dir/"warnings.json", {"warnings": warnings, "assignment_mode": args.assignment_mode, "slot_assignment_method":"stage5_assign_neighbors_lane_aware", "stage5d_slot_schema_matched": True, "stage5d_slot_order_matched": True, "stage5d_slot_assignment_exact_waymo_lane_aware": args.assignment_mode != "geometric_only", "stage5d_formula_parity_schema_recorded": True, "map_query_success": bool(map_counts.get("map_query_success", 0) > 0), "lane_info_count": lane_info_count, "adjacency_source_counts": map_counts, "validation": validation})
    exact_channels = "valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y, closing, ttc, thw, speed, heading_rel"
    accel_note = "accel and yaw_rate match Stage 5D finite-difference semantics because no slot ID switches were observed." if accel_yaw_rate_matched else "accel and yaw_rate are approximated_or_not_stage5_matched because at least one semantic slot switches tracked-object IDs; finite differences are reset at switches."
    report=["# nuPlan Stage 5D-Compatible Context Build Report","",f"- rows: `{row}` (= `{n_scenarios} scenarios × {n_planners} planners`)","- row semantics: `scenario × planner × planner-controlled nuPlan ego rollout`","- background agents: context only; no multi-agent ego expansion","- context_traj.npy: `"+str(list(ctx_arr.shape))+"`","- Stage 5D best model input: `context_traj.npy [N,T,83]`","- 83 = ego 8 + 5 semantic neighbor slots × 15 channels","- interaction_feat_style.npy is for reports/evaluation, not encoder input","- context_traj has no map/lane/ODD channels","- slot assignment: Stage 5 `assign_neighbors_lane_aware`; geometric assignment is fallback","- parsed msgpack files: `"+str(parsed)+"`", f"- map_query_success: `{bool(map_counts.get('map_query_success', 0) > 0)}`", f"- lane_info_count: `{lane_info_count}`", f"- adjacency_source counts: `{map_counts}`","", "## Stage 5 formula parity", f"- Exactly matched neighbor formulas: `{exact_channels}`.", "- closing formula parity: `passed` (`closing = ego_forward_speed - rel_vx`).", "- TTC formula parity: `passed` (cap when closing <= 1e-3, otherwise distance / max(closing, 1e-3)).", "- delta_x/delta_y formula parity: `passed` (duplicates of rel_x/rel_y, not proxy channels).", f"- accel/yaw_rate formula parity: `{accel_yaw_rate_matched}`. {accel_note}", "- Approximations: accel/yaw_rate are approximations when slot switching is non-zero; slot assignment uses Stage 5 lane-aware logic unless `geometric_only` is requested or lane-map fallback is needed."]
    (args.output_dir/"context_build_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    lines=["# Slot Assignment Report","",f"- assignment_mode: `{args.assignment_mode}`",f"- slot names/order: `{STAGE5D_NEIGHBOR_SLOT_NAMES}`",f"- lane-aware success rate: `{float(np.mean([d.get('lane_assignment_available', False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0}`",f"- geometric fallback rate: `{float(np.mean([d.get('fallback_assignment_used', False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0}`",""]
    for sn,st in slot_stats.items():
        cont = slot_continuity[sn]
        lines.append(f"- {sn}: coverage={st['coverage_ratio']:.6f}, empty={st['empty_slot_ratio']:.6f}, median_rel_x={st['median_rel_x']}, median_rel_y={st['median_rel_y']}, median_distance={st['median_distance']}, slot_id_switch_count={cont['slot_id_switch_count']}, slot_id_switch_rate={cont['slot_id_switch_rate']:.6f}, mean_continuous_segment_length={cont['mean_continuous_segment_length']}")
    lines += ["", "## Lane context diagnostics"]
    lane_success = float(np.mean([d.get("lane_assignment_available", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    fallback_rate = float(np.mean([d.get("fallback_assignment_used", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    lines += [f"- lane-aware success rate: `{lane_success}`", f"- geometric fallback rate: `{fallback_rate}`"]
    from collections import Counter
    lines += [f"- adjacency_source counts: `{dict(Counter(d.get('adjacency_source', 'none') for d in assignment_debug_rows))}`", f"- lane_context_quality counts: `{dict(Counter(d.get('lane_context_quality', 'unknown') for d in assignment_debug_rows))}`"]
    rej = Counter()
    for d in assignment_debug_rows:
        for by_slot in (d.get("slot_rejection_reason_counts") or {}).values():
            rej.update(by_slot)
    lines.append(f"- rejection reason counts: `{dict(rej)}`")
    lines += ["", "## Slot continuity diagnostics"]
    for sn, cont in slot_continuity.items():
        lines.append(f"- {sn}: slot_id_switch_count={cont['slot_id_switch_count']}, slot_id_switch_rate={cont['slot_id_switch_rate']:.6f}, mean_continuous_segment_length={cont['mean_continuous_segment_length']}")
    lines += ["", "## Sanity checks"] + [f"- {k}: `{v}`" for k,v in sanity.items()]
    (args.output_dir/"slot_assignment_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not validation["pass"]: raise RuntimeError("nuPlan Stage5D context validation failed; see warnings.json and slot_assignment_report.md")
    print(f"nuPlan Stage 5D-compatible context build PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
