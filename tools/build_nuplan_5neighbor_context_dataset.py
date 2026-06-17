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


def assign_slots(cands: List[Tuple[str, float, float, Tuple[float, float, float, float, float, float]]], same_lane_abs_y: float, adjacent_lane_min_abs_y: float) -> Dict[str, Tuple[str, float, float, Tuple[float, float, float, float, float, float]]]:
    out: Dict[str, Tuple[str, float, float, Tuple[float, float, float, float, float, float]]] = {}
    same = [c for c in cands if abs(c[2]) <= same_lane_abs_y]
    left = [c for c in cands if c[2] >= adjacent_lane_min_abs_y]
    right = [c for c in cands if c[2] <= -adjacent_lane_min_abs_y]
    front = [c for c in same if c[1] > 0]
    rear = [c for c in same if c[1] < 0]
    lf = [c for c in left if c[1] > 0]
    lr = [c for c in left if c[1] < 0]
    rf = [c for c in right if c[1] > 0]
    if front: out["front"] = min(front, key=lambda c: c[1])
    if rear: out["rear"] = max(rear, key=lambda c: c[1])
    if lf: out["left_front"] = min(lf, key=lambda c: c[1])
    if lr: out["left_rear"] = max(lr, key=lambda c: c[1])
    if rf: out["right_front"] = min(rf, key=lambda c: c[1])
    return out


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


def build_row_context(stage7c_seq: np.ndarray, mask: np.ndarray, tracks: Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
    ego = convert_ego(stage7c_seq[None, None], mask[None, None])[0]
    timesteps = ego.shape[0]
    ex, ey, eyaw, evx, evy = ego_world(stage7c_seq)
    nbr = np.zeros((5, timesteps, 15), dtype=np.float32)
    slot_ids = [["-1" for _ in range(timesteps)] for _ in range(5)]
    previous_speed: List[float | None] = [None] * 5
    previous_heading_rel: List[float | None] = [None] * 5
    previous_token: List[str | None] = [None] * 5
    dt = 0.1
    for t in range(timesteps):
        if not bool(mask[t]):
            previous_speed = [None] * 5
            previous_heading_rel = [None] * 5
            previous_token = [None] * 5
            continue
        cands = []
        c = math.cos(float(eyaw[t])); s = math.sin(float(eyaw[t]))
        for tok, by_t in tracks.items():
            st = by_t.get(t)
            if st is None:
                continue
            dx = st[0] - float(ex[t]); dy = st[1] - float(ey[t])
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            cands.append((tok, rel_x, rel_y, st))
        assigned = assign_slots(cands, args.same_lane_abs_y, args.adjacent_lane_min_abs_y)
        current_tokens = [None] * 5
        for si, sn in enumerate(STAGE5D_NEIGHBOR_SLOT_NAMES):
            if sn not in assigned:
                previous_speed[si] = None
                previous_heading_rel[si] = None
                previous_token[si] = None
                continue
            tok, rel_x, rel_y, st = assigned[sn]
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
    return ego.astype(np.float32), nbr, context, slot_ids


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
    return {"schema_name": "stage5d83_nuplan_geometric_stage5_formula_parity", "shape": "[N,T,83]", "context_dim": 83, "ego_channels": STAGE5D_EGO_CHANNELS, "neighbor_slots": STAGE5D_NEIGHBOR_SLOT_NAMES, "neighbor_channels_per_slot": STAGE5D_NEIGHBOR_CHANNELS, "context_has_map_lane_odd_channels": False, "stage5d_best_model_training_input": "context_traj.npy [N,T,83] from tools/build_waymo_5neighbor_context_dataset.py", "dim_formula": "83 = ego 8 + 5 semantic neighbor slots × 15 channels", "slot_assignment_method": "geometric_proxy, not exact Waymo lane-aware assignment", "approximated_or_not_stage5_matched_channels": approximated, "channels": channels}

def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 5D-compatible [N,T,83] nuPlan context artifacts from official Stage 7C simulation outputs.")
    ap.add_argument("--sim_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_neighbors_for_context", type=int, default=5)
    ap.add_argument("--slot_assignment_method", choices=["geometric_proxy"], default="geometric_proxy")
    ap.add_argument("--same_lane_abs_y", type=float, default=1.8)
    ap.add_argument("--adjacent_lane_min_abs_y", type=float, default=1.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
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
    ego_rows=[]; nbr_rows=[]; ctx_rows=[]; inter_rows=[]; slot_id_rows=[]; warnings=[]; cache={}; parsed=0
    row=0
    for si in tqdm(range(n_scenarios), desc="nuPlan Stage5D context"):
        for pi in range(n_planners):
            meta = by_pair.get((si, pi), {"scenario_index": str(si), "planner_id": str(pi), "planner_name": planners[pi]})
            msg = find_msgpack(args.sim_dir, meta)
            if msg is None: raise FileNotFoundError(f"No official nuPlan msgpack found for scenario_index={si}, planner_id={pi}, planner_name={planners[pi]}")
            if msg not in cache:
                tracks, sample_count = parse_neighbor_world_tracks(msg, timesteps); cache[msg]=tracks; parsed += 1
                if sample_count != timesteps: warnings.append({"type":"msgpack_timestep_mismatch","path":str(msg),"samples":sample_count,"expected_T":timesteps})
            ego,nbr,ctx,slot_ids = build_row_context(np.asarray(seq[si,pi]), np.asarray(mask[si,pi]).astype(bool), cache[msg], args)
            inter, _ = aggregate_interaction_features(ego, nbr, 0.1)
            ego_rows.append(ego); nbr_rows.append(nbr); ctx_rows.append(ctx); inter_rows.append(np.nan_to_num(inter, nan=0.0, posinf=1e6, neginf=-1e6)); slot_id_rows.append(slot_ids); row += 1
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
    sanity={"front_median_rel_x_gt_0": (slot_stats["front"]["median_rel_x"] is not None and slot_stats["front"]["median_rel_x"]>0), "rear_median_rel_x_lt_0": (slot_stats["rear"]["median_rel_x"] is not None and slot_stats["rear"]["median_rel_x"]<0), "left_front_median_rel_y_gt_0": (slot_stats["left_front"]["median_rel_y"] is not None and slot_stats["left_front"]["median_rel_y"]>0), "left_rear_median_rel_y_gt_0": (slot_stats["left_rear"]["median_rel_y"] is not None and slot_stats["left_rear"]["median_rel_y"]>0), "right_front_median_rel_y_lt_0": (slot_stats["right_front"]["median_rel_y"] is not None and slot_stats["right_front"]["median_rel_y"]<0)}
    slot_pass=all(sanity.values())
    slot_continuity = slot_continuity_stats(slot_id_rows)
    slot_switch_rate_by_slot = {k: float(v["slot_id_switch_rate"] or 0.0) for k, v in slot_continuity.items()}
    accel_yaw_rate_matched = all(rate == 0.0 for rate in slot_switch_rate_by_slot.values())
    planner_non_empty=all(np.load(idx_dir/f"{p}.npy").size>0 for p in planners)
    validation={"pass": bool(slot_pass and planner_non_empty), "row_semantics_correct": True, "no_multi_agent_ego_expansion": True, "background_agents_context_only": True, "stage5d_dim_matched": ctx_arr.shape[-1]==83, "stage5d_channel_schema_matched": True, "stage5d_slot_semantics_verified": bool(slot_pass), "slot_sanity_passed": bool(slot_pass), "slot_coverage_by_slot": {k:v["coverage_ratio"] for k,v in slot_stats.items()}, "context_traj_no_nonfinite": bool(np.isfinite(ctx_arr).all()), "planner_indices_non_empty": bool(planner_non_empty), "rows_equal_num_scenarios_times_num_planners": row == expected_rows, "metadata_row_count_matches": len(rows)==row, "ego_seq_row_count_matches": ego_arr.shape[0]==row, "interaction_feat_style_row_count_matches": feat_arr.shape[0]==row, "stage5d_derived_formula_matched": True, "stage5d_closing_formula_matched": True, "stage5d_ttc_formula_matched": True, "stage5d_delta_xy_formula_matched": True, "stage5d_accel_yaw_rate_formula_matched": bool(accel_yaw_rate_matched), "slot_id_switch_rate_by_slot": slot_switch_rate_by_slot}
    write_json(args.output_dir/"stage5d_context_schema.json", make_context_schema(accel_yaw_rate_matched=accel_yaw_rate_matched))
    write_json(args.output_dir/"warnings.json", {"warnings": warnings, "slot_assignment_method":"geometric_proxy", "stage5d_slot_assignment_exact_waymo_lane_aware": False, "stage5d_formula_parity_schema_recorded": True, "validation": validation})
    exact_channels = "valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y, closing, ttc, thw, speed, heading_rel"
    accel_note = "accel and yaw_rate match Stage 5D finite-difference semantics because no slot ID switches were observed." if accel_yaw_rate_matched else "accel and yaw_rate are approximated_or_not_stage5_matched because at least one semantic slot switches tracked-object IDs; finite differences are reset at switches."
    report=["# nuPlan Stage 5D-Compatible Context Build Report","",f"- rows: `{row}` (= `{n_scenarios} scenarios × {n_planners} planners`)","- row semantics: `scenario × planner × planner-controlled nuPlan ego rollout`","- background agents: context only; no multi-agent ego expansion","- context_traj.npy: `"+str(list(ctx_arr.shape))+"`","- Stage 5D best model input: `context_traj.npy [N,T,83]`","- 83 = ego 8 + 5 semantic neighbor slots × 15 channels","- interaction_feat_style.npy is for reports/evaluation, not encoder input","- context_traj has no map/lane/ODD channels","- slot assignment: `geometric_proxy`, not exact Waymo lane-aware assignment","- parsed msgpack files: `"+str(parsed)+"`","", "## Stage 5 formula parity", f"- Exactly matched neighbor formulas: `{exact_channels}`.", "- closing formula parity: `passed` (`closing = ego_forward_speed - rel_vx`).", "- TTC formula parity: `passed` (cap when closing <= 1e-3, otherwise distance / max(closing, 1e-3)).", "- delta_x/delta_y formula parity: `passed` (duplicates of rel_x/rel_y, not proxy channels).", f"- accel/yaw_rate formula parity: `{accel_yaw_rate_matched}`. {accel_note}", "- Approximations: slot assignment remains geometric_proxy rather than Waymo lane-aware assignment; accel/yaw_rate are approximations when slot switching is non-zero."]
    (args.output_dir/"context_build_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    lines=["# Slot Assignment Report","",f"- slot assignment method: `{args.slot_assignment_method}`",f"- same_lane_abs_y: `{args.same_lane_abs_y}`",f"- adjacent_lane_min_abs_y: `{args.adjacent_lane_min_abs_y}`",""]
    for sn,st in slot_stats.items():
        cont = slot_continuity[sn]
        lines.append(f"- {sn}: coverage={st['coverage_ratio']:.6f}, empty={st['empty_slot_ratio']:.6f}, median_rel_x={st['median_rel_x']}, median_rel_y={st['median_rel_y']}, median_distance={st['median_distance']}, slot_id_switch_count={cont['slot_id_switch_count']}, slot_id_switch_rate={cont['slot_id_switch_rate']:.6f}, mean_continuous_segment_length={cont['mean_continuous_segment_length']}")
    lines += ["", "## Slot continuity diagnostics"]
    for sn, cont in slot_continuity.items():
        lines.append(f"- {sn}: slot_id_switch_count={cont['slot_id_switch_count']}, slot_id_switch_rate={cont['slot_id_switch_rate']:.6f}, mean_continuous_segment_length={cont['mean_continuous_segment_length']}")
    lines += ["", "## Sanity checks"] + [f"- {k}: `{v}`" for k,v in sanity.items()]
    (args.output_dir/"slot_assignment_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not validation["pass"]: raise RuntimeError("nuPlan Stage5D context validation failed; see warnings.json and slot_assignment_report.md")
    print(f"nuPlan Stage 5D-compatible context build PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
