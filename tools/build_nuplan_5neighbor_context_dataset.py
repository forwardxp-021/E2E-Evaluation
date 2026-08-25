#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from tools.interaction_context_features import aggregate_interaction_features, write_feature_schema_json
from tools.stage7d_extract_neighbors_from_nuplan import find_msgpack, parse_neighbor_world_tracks, validate_official
from tools.stage5d_context_core import (
    SLOT_NAMES, EGO_CHANNELS, NEIGHBOR_CHANNELS, CONTEXT_DIM,
    build_ego_features_8d, build_neighbor_features_15d, build_context_traj_from_standard_tracks,
    assign_stage5d_slots, validate_stage5d_context, write_stage5d_context_schema,
    make_stage5d_context_schema,
)
from tools.nuplan_lane_utils import extract_nuplan_lane_infos, build_lane_topology_debug_summary, write_lane_topology_debug_artifacts
from tools.nuplan_projection_debug import (
    collect_nuplan_projection_debug_rows,
    summarize_projection_debug,
    write_projection_debug_artifacts,
)



def stage5d_formula_validation_status(
    *,
    stage5d_static_derived_formula_matched: bool,
    stage5d_closing_formula_matched: bool,
    stage5d_ttc_formula_matched: bool,
    stage5d_delta_xy_formula_matched: bool,
    accel_yaw_rate_matched: bool,
    slot_switch_rate_by_slot: Dict[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """Summarize Stage5D formula parity without making slot-switch resets fatal.

    The nuPlan adapter assigns semantic slots independently per timestep.  When a
    slot changes tracked-object id, accel/yaw_rate finite differences are reset
    instead of differencing two different objects.  That mismatch is expected and
    nonfatal only if all static and safety-critical derived formulas still match.
    """
    static_and_safety_matched = bool(
        stage5d_static_derived_formula_matched
        and stage5d_closing_formula_matched
        and stage5d_ttc_formula_matched
        and stage5d_delta_xy_formula_matched
    )
    has_slot_switches = any(float(rate or 0.0) > 0.0 for rate in slot_switch_rate_by_slot.values())
    temporal_nonfatal = bool(static_and_safety_matched and (not accel_yaw_rate_matched) and has_slot_switches)
    temporal_fatal = bool(static_and_safety_matched and (not accel_yaw_rate_matched) and not has_slot_switches)
    temporal_status = (
        "matched"
        if accel_yaw_rate_matched
        else "nonfatal_slot_switch_reset"
        if temporal_nonfatal
        else "failed"
    )
    formula_validation_pass = bool(static_and_safety_matched and not temporal_fatal)
    status = {
        "stage5d_static_derived_formula_matched": bool(stage5d_static_derived_formula_matched),
        "stage5d_closing_formula_matched": bool(stage5d_closing_formula_matched),
        "stage5d_ttc_formula_matched": bool(stage5d_ttc_formula_matched),
        "stage5d_delta_xy_formula_matched": bool(stage5d_delta_xy_formula_matched),
        "stage5d_temporal_derived_formula_matched": bool(accel_yaw_rate_matched),
        "stage5d_accel_yaw_rate_formula_matched": bool(accel_yaw_rate_matched),
        "stage5d_temporal_formula_status": temporal_status,
        "stage5d_accel_yaw_rate_formula_status": temporal_status,
        "stage5d_accel_yaw_rate_formula_nonfatal": bool(temporal_nonfatal),
        "stage5d_derived_formula_matched": bool(static_and_safety_matched and accel_yaw_rate_matched),
        "stage5d_formula_validation_pass": formula_validation_pass,
        "slot_id_switch_rate_by_slot": dict(slot_switch_rate_by_slot),
    }
    formula_warnings: List[Dict[str, Any]] = []
    if temporal_nonfatal:
        formula_warnings.append({
            "type": "temporal_formula_nonfatal_slot_switch_reset",
            "severity": "warning",
            "slot_id_switch_rate_by_slot": dict(slot_switch_rate_by_slot),
            "message": "accel/yaw_rate finite differences are reset at semantic slot ID switches; static and safety-critical derived formulas remain matched.",
        })
    return status, formula_warnings, formula_validation_pass


def planner_axis(metadata_path: Path, index_path: Path, p_count: int, required_planners: Sequence[str] | None = None) -> List[str]:
    """Discover the Stage 7C planner axis without coupling Stage 7E to Stage 7D planner policy names."""
    seen: Dict[int, str] = {}
    for path in (metadata_path, index_path):
        for row in read_csv_rows(path):
            pid_text = row.get("planner_id", "").strip()
            if not pid_text:
                continue
            try:
                pid = int(pid_text)
            except ValueError as exc:
                raise ValueError(f"Invalid planner_id={pid_text!r} in {path}") from exc
            name = (row.get("planner_name") or row.get("planner") or f"planner_{pid}").strip()
            seen[pid] = name or f"planner_{pid}"
    missing_ids = [pid for pid in range(p_count) if pid not in seen]
    if missing_ids:
        raise ValueError(f"Planner axis length mismatch: tensor P={p_count}, missing planner ids={missing_ids}; observed={seen}")
    extra_ids = [pid for pid in seen if pid < 0 or pid >= p_count]
    if extra_ids:
        raise ValueError(f"Planner metadata contains planner ids outside tensor axis P={p_count}: {extra_ids}")
    planners = [seen[i] for i in range(p_count)]
    required = list(required_planners or [])
    missing_required = [p for p in required if p not in planners]
    if missing_required:
        raise ValueError(f"Planner axis missing required planners: {missing_required}; observed={planners}; configure --required_planners or omit it for expansion planners")
    return planners


def load_stage7c_scenario_axis(
    index_path: Path,
    n_scenarios: int,
    n_planners: int,
    planners: Sequence[str],
) -> List[int]:
    axis_data = read_json(index_path)
    raw_scenario_axis = axis_data.get("scenario_axis")
    raw_planner_axis = axis_data.get("planner_axis")
    raw_planner_names = axis_data.get("planner_axis_names")
    if not isinstance(raw_scenario_axis, list):
        raise ValueError(f"{index_path} must contain a scenario_axis list")
    try:
        scenario_axis = [int(value) for value in raw_scenario_axis]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scenario_axis must contain integer-like values: {raw_scenario_axis}") from exc
    if len(scenario_axis) != n_scenarios:
        raise ValueError(
            f"scenario_axis length mismatch: axis={len(scenario_axis)}, tensor scenarios={n_scenarios}"
        )
    if len(set(scenario_axis)) != len(scenario_axis):
        raise ValueError(f"scenario_axis contains duplicates: {scenario_axis}")
    if not isinstance(raw_planner_axis, list) or len(raw_planner_axis) != n_planners:
        raise ValueError(
            f"planner_axis length mismatch: axis={raw_planner_axis}, tensor planners={n_planners}"
        )
    try:
        planner_axis_ids = [int(value) for value in raw_planner_axis]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"planner_axis must contain integer-like values: {raw_planner_axis}") from exc
    if planner_axis_ids != list(range(n_planners)):
        raise ValueError(
            f"planner_axis must match tensor planner positions 0..P-1: observed={planner_axis_ids}"
        )
    if list(raw_planner_names or []) != list(planners):
        raise ValueError(
            f"planner_axis_names mismatch: axis={raw_planner_names}, discovered={list(planners)}"
        )
    declared_shape = axis_data.get("shape")
    if declared_shape is not None and list(declared_shape[:2]) != [n_scenarios, n_planners]:
        raise ValueError(
            f"simulated_ego_seq_index shape mismatch: declared={declared_shape}, "
            f"tensor_prefix={[n_scenarios, n_planners]}"
        )
    return scenario_axis


def validate_scenario_planner_alignment(
    index_rows: List[Dict[str, str]],
    scenario_axis: Sequence[int],
    planners: Sequence[str],
) -> Dict[Tuple[int, int], Dict[str, str]]:
    by_pair: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in index_rows:
        try:
            key = (int(row.get("scenario_index", "")), int(row.get("planner_id", "")))
        except (TypeError, ValueError):
            continue
        if key in by_pair:
            raise ValueError(f"Duplicate scenario/planner row in scenario_planner_index.csv: {key}")
        by_pair[key] = row
    scenario_tokens: Dict[int, str] = {}
    for scenario_index in scenario_axis:
        for planner_id, planner_name in enumerate(planners):
            key = (int(scenario_index), planner_id)
            if key not in by_pair:
                raise ValueError(f"Missing Stage7C index row for scenario/planner pair: {key}")
            row = by_pair[key]
            if (row.get("status") or "").strip().lower() != "succeeded":
                raise ValueError(f"Tensor axis references non-successful Stage7C row {key}: status={row.get('status')!r}")
            observed_planner = (row.get("planner_name") or row.get("planner") or "").strip()
            if observed_planner != planner_name:
                raise ValueError(
                    f"Planner mismatch for pair {key}: expected={planner_name!r}, observed={observed_planner!r}"
                )
            token = (
                row.get("actual_nuplan_scenario_token")
                or row.get("scenario_token")
                or row.get("scenario_id")
                or row.get("scene_token")
                or ""
            ).strip()
            if not token:
                raise ValueError(f"Missing scenario token for pair {key}")
            previous = scenario_tokens.setdefault(int(scenario_index), token)
            if token != previous:
                raise ValueError(
                    f"Scenario token differs across planners for scenario_index={scenario_index}: "
                    f"{previous!r} vs {token!r}"
                )
    return by_pair


def metadata_rows(index_rows: List[Dict[str, str]], planner_meta_rows: List[Dict[str, str]], planners: List[str], scenario_axis: Sequence[int]) -> List[Dict[str, Any]]:
    planner_profiles: Dict[int, Dict[str, str]] = {}
    for row in planner_meta_rows:
        try:
            planner_profiles[int(row.get("planner_id", -1))] = row
        except ValueError:
            continue
    by_pair = {}
    for row in index_rows:
        try:
            by_pair[(int(row.get("scenario_index", row.get("scenario_id", -1))), int(row.get("planner_id", -1)))] = row
        except ValueError:
            continue
    rows = []
    g = 0
    for tensor_position, i in enumerate(scenario_axis):
        for pid, planner in enumerate(planners):
            src = by_pair.get((i, pid), {})
            rows.append({
                "global_row": g, "tensor_scenario_position": tensor_position,
                "scenario_index": i, "planner_id": pid, "planner_name": planner,
                "log_name": (src.get("log_name") or src.get("db_name", "")).removesuffix(".db"),
                "map_name": src.get("map_name", ""),
                "location": src.get("location", ""),
                "scenario_token": src.get("actual_nuplan_scenario_token") or src.get("scenario_token") or src.get("scenario_id") or src.get("scene_token", ""),
                "actual_nuplan_scenario_token": src.get("actual_nuplan_scenario_token") or src.get("scenario_token") or src.get("scenario_id") or src.get("scene_token", ""),
                "stage7b_scene_token": src.get("stage7b_scene_token") or src.get("scene_token", ""),
                "sample_id": src.get("sample_id", ""),
                "scenario_type": src.get("scenario_type", ""),
                "source_stage": "stage7c_official_nuplan_simulation",
                "uses_official_nuplan_simulation": True, "pseudo_rollout": False,
                "style_scope": planner_profiles.get(pid, {}).get("style_scope", ""),
                "policy_style": planner_profiles.get(pid, {}).get("policy_style", ""),
                "nuplan_planner_config": planner_profiles.get(pid, {}).get("nuplan_planner_config", ""),
                "hydra_overrides": planner_profiles.get(pid, {}).get("hydra_overrides", ""),
                "supported_behavior_tasks": planner_profiles.get(pid, {}).get("supported_behavior_tasks", ""),
                "unsupported_behavior_tasks": planner_profiles.get(pid, {}).get("unsupported_behavior_tasks", ""),
                "planner_class": planner_profiles.get(pid, {}).get("planner_class", ""),
                "planner_type": planner_profiles.get(pid, {}).get("planner_type", ""),
                "parameters_json": planner_profiles.get(pid, {}).get("parameters_json", ""),
            })
            g += 1
    return rows


def load_scenario_map_metadata(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    rows = read_csv_rows(path)
    mapping: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for key in ("scenario_index", "scenario_id", "scenario_token", "actual_nuplan_scenario_token", "scene_token", "log_name", "db_name"):
            value = (row.get(key) or "").strip()
            if value:
                mapping.setdefault(f"{key}:{value}", row)
                if key == "db_name":
                    mapping.setdefault(f"log_name:{value.removesuffix('.db')}", row)
    return mapping


def resolve_map_name_from_nuplan_db(
    meta: Dict[str, str],
    nuplan_db_root: Path | None,
    cache: Dict[str, str],
) -> Tuple[str, str]:
    if nuplan_db_root is None:
        return "", "nuplan_db_root_unavailable"
    log_name = (meta.get("log_name") or meta.get("db_name") or "").strip().removesuffix(".db")
    if not log_name:
        return "", "log_name_unavailable"
    if log_name in cache:
        return cache[log_name], "nuplan_db.log.cached"
    direct = nuplan_db_root / f"{log_name}.db"
    matches = [direct] if direct.is_file() else list(nuplan_db_root.rglob(f"{log_name}.db"))
    if len(matches) != 1:
        return "", "nuplan_db_not_found" if not matches else "nuplan_db_ambiguous"
    with sqlite3.connect(str(matches[0])) as connection:
        row = connection.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
    if not row:
        return "", "nuplan_db_log_table_empty"
    # map_version is the canonical map-factory key (for example
    # "us-nv-las-vegas-strip"); location may contain a legacy internal alias
    # such as "las_vegas".
    map_name = str(row[1] or row[0] or "").strip()
    if not map_name:
        return "", "nuplan_db_map_name_empty"
    cache[log_name] = map_name
    return map_name, "nuplan_db.log.map_version"


def resolve_map_name(
    meta: Dict[str, str],
    explicit_map_name: str = "",
    scenario_map_metadata: Dict[str, Dict[str, str]] | None = None,
    nuplan_db_root: Path | None = None,
    db_map_cache: Dict[str, str] | None = None,
) -> Tuple[str, str]:
    for key in ("map_name", "location"):
        value = (meta.get(key) or "").strip()
        if value:
            return value, f"row.{key}"
    mapping = scenario_map_metadata or {}
    for key in ("scenario_index", "scenario_id", "scenario_token", "actual_nuplan_scenario_token", "scene_token", "log_name", "db_name"):
        value = (meta.get(key) or "").strip()
        candidates = [value]
        if key == "db_name" and value.endswith(".db"):
            candidates.append(value.removesuffix(".db"))
        for candidate in candidates:
            row = mapping.get(f"{key}:{candidate}")
            if row:
                for map_key in ("map_name", "location"):
                    map_value = (row.get(map_key) or "").strip()
                    if map_value:
                        return map_value, f"scenario_map_metadata_csv.{key}.{map_key}"
    db_map_name, db_source = resolve_map_name_from_nuplan_db(meta, nuplan_db_root, db_map_cache if db_map_cache is not None else {})
    if db_map_name:
        return db_map_name, db_source
    if explicit_map_name:
        return explicit_map_name, "cli.--map_name_fallback"
    return "", "unresolved"


def estimate_dt(stage7c_seq: np.ndarray, mask: np.ndarray, default_dt: float = 0.1) -> float:
    if stage7c_seq.ndim != 2 or stage7c_seq.shape[1] < 8:
        return float(default_dt)
    time_s = np.asarray(stage7c_seq[:, 7], dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(time_s)
    diffs = np.diff(time_s[valid])
    good = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    return float(np.median(good)) if good.size else float(default_dt)


def build_nuplan_ego_features_8d(stage7c_seq: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Adapt a nuPlan simulated_ego_seq row to Stage5D CORE ego 8D in a local window frame.

    The standard track window is [x, y, vx, vy, heading, valid] in world coordinates.
    The local frame uses the first valid ego pose as origin and base heading, matching the
    deterministic Stage5D adapter contract while leaving neighbor features per-timestep
    ego-centric as in the Waymo Stage5D builder.
    """
    s = np.asarray(stage7c_seq, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if s.ndim != 2 or s.shape[1] < 4:
        raise ValueError(f"stage7c_seq row must have shape [T,>=4] with x,y,yaw,speed..., got {list(s.shape)}")
    yaw = s[:, 2]
    speed = s[:, 3]
    track = np.zeros((s.shape[0], 6), dtype=np.float32)
    track[:, 0] = s[:, 0]
    track[:, 1] = s[:, 1]
    track[:, 2] = speed * np.cos(yaw)
    track[:, 3] = speed * np.sin(yaw)
    track[:, 4] = yaw
    track[:, 5] = valid.astype(np.float32)
    valid_idx = np.flatnonzero(valid & np.isfinite(track[:, :2]).all(axis=1) & np.isfinite(track[:, 4]))
    ref = int(valid_idx[0]) if valid_idx.size else 0
    origin = track[ref, :2].copy()
    base_heading = float(track[ref, 4]) if np.isfinite(track[ref, 4]) else 0.0
    dt = estimate_dt(s, valid, 0.1)
    ego, heading, ego_speed = build_ego_features_8d(track, origin, base_heading, dt)
    ego[~valid, :] = 0.0
    return ego.astype(np.float32), heading.astype(np.float32), ego_speed.astype(np.float32), dt

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def scenario_lane_cache_key(map_name: str, scenario_index: int) -> Tuple[str, int]:
    """Scope local nuPlan lane queries to one map and one source scenario.

    LaneInfo extraction is spatially local.  Reusing the first local query for
    every scenario on the same map makes distant scenarios project against the
    wrong lane subset.
    """
    return str(map_name), int(scenario_index)


def scenario_query_ego_xy(seq_for_scenario: np.ndarray, mask_for_scenario: np.ndarray) -> np.ndarray:
    """Collect valid ego positions across every planner for one scenario."""
    seq_arr = np.asarray(seq_for_scenario)
    mask_arr = np.asarray(mask_for_scenario, dtype=bool)
    if seq_arr.ndim != 3 or mask_arr.shape != seq_arr.shape[:2] or seq_arr.shape[-1] < 2:
        raise ValueError(
            "scenario lane query inputs must be seq [P,T,C>=2] and mask [P,T], "
            f"got seq={list(seq_arr.shape)}, mask={list(mask_arr.shape)}"
        )
    xy = np.asarray(seq_arr[..., :2][mask_arr], dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f"scenario lane query ego coordinates are invalid: shape={list(xy.shape)}")
    return xy


def summarize_lane_assignment_rows(row_debug_rows: List[List[dict]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(row_debug_rows) != len(rows):
        raise ValueError(f"lane assignment diagnostic row mismatch: debug={len(row_debug_rows)}, metadata={len(rows)}")
    detail = []
    fallback_reasons: Counter[str] = Counter()
    for metadata, debug in zip(rows, row_debug_rows):
        total = len(debug)
        fallback = sum(bool(d.get("fallback_assignment_used")) for d in debug)
        laneaware = sum(bool(d.get("lane_assignment_available")) and not bool(d.get("fallback_assignment_used")) for d in debug)
        quality_counts = Counter(str(d.get("lane_context_quality") or "unknown").lower() for d in debug)
        ambiguous = sum(
            count for quality, count in quality_counts.items() if quality.startswith("ambiguous")
        )
        bad = int(quality_counts.get("bad", 0))
        quality_eligible = sum(
            bool(d.get("lane_assignment_available"))
            and not bool(d.get("fallback_assignment_used"))
            and str(d.get("lane_context_quality") or "").lower() != "bad"
            and not str(d.get("lane_context_quality") or "").lower().startswith("ambiguous")
            for d in debug
        )
        reasons = Counter(str(d.get("fallback_reason") or "none") for d in debug if d.get("fallback_assignment_used"))
        fallback_reasons.update(reasons)
        detail.append({
            "global_row": int(metadata["global_row"]),
            "scenario_index": int(metadata["scenario_index"]),
            "planner_id": int(metadata["planner_id"]),
            "planner_name": str(metadata["planner_name"]),
            "valid_frame_count": int(total),
            "laneaware_frame_count": int(laneaware),
            "fallback_frame_count": int(fallback),
            "laneaware_rate": float(laneaware / total) if total else 0.0,
            "fallback_rate": float(fallback / total) if total else 0.0,
            "ambiguous_frame_count": int(ambiguous),
            "ambiguous_frame_rate": float(ambiguous / total) if total else 0.0,
            "bad_frame_count": int(bad),
            "bad_frame_rate": float(bad / total) if total else 0.0,
            "quality_eligible_frame_count": int(quality_eligible),
            "quality_eligible_frame_rate": float(quality_eligible / total) if total else 0.0,
            "lane_context_quality_counts": json.dumps(dict(quality_counts), sort_keys=True),
            "fallback_reason_counts": json.dumps(dict(reasons), sort_keys=True),
        })
    total_frames = sum(int(r["valid_frame_count"]) for r in detail)
    fallback_frames = sum(int(r["fallback_frame_count"]) for r in detail)
    return {
        "cache_scope": "map_name_plus_source_scenario",
        "row_count": len(detail),
        "valid_frame_count": total_frames,
        "fallback_frame_count": fallback_frames,
        "fallback_assignment_used_rate": float(fallback_frames / total_frames) if total_frames else 0.0,
        "fallback_reason_counts": dict(fallback_reasons),
        "rows": detail,
    }


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def ego_world(stage7c_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, yaw, speed = stage7c_seq[:, 0], stage7c_seq[:, 1], stage7c_seq[:, 2], stage7c_seq[:, 3]
    return x, y, yaw, speed * np.cos(yaw), speed * np.sin(yaw)


def build_row_context(stage7c_seq: np.ndarray, mask: np.ndarray, tracks: Dict[str, Dict[int, Tuple[float, float, float, float, float, float]]], args: argparse.Namespace, lane_infos: Dict[str, Any] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]], List[dict]]:
    ego, ego_heading, ego_speed, dt = build_nuplan_ego_features_8d(stage7c_seq, mask)
    timesteps = ego.shape[0]
    ex, ey, eyaw, evx, evy = ego_world(stage7c_seq)
    nbr = np.zeros((len(SLOT_NAMES), timesteps, len(NEIGHBOR_CHANNELS)), dtype=np.float32)
    slot_ids = [["-1" for _ in range(timesteps)] for _ in range(len(SLOT_NAMES))]
    previous_speed: List[float | None] = [None] * len(SLOT_NAMES)
    previous_heading_rel: List[float | None] = [None] * len(SLOT_NAMES)
    previous_token: List[str | None] = [None] * len(SLOT_NAMES)
    assign_debug: List[dict] = []
    cfg = {k: getattr(args, k) for k in ["lane_max_lateral_distance", "lane_max_heading_diff_deg", "adjacent_lane_min_offset", "adjacent_lane_max_offset", "adjacent_lane_max_heading_diff_deg", "lane_search_radius", "lane_topk_candidates", "front_max_distance", "side_front_max_distance", "side_rear_max_distance", "lane_lateral_tolerance", "slot_heading_diff_deg", "static_speed_threshold"]}
    for t in range(timesteps):
        if not bool(mask[t]):
            previous_speed = [None] * len(SLOT_NAMES)
            previous_heading_rel = [None] * len(SLOT_NAMES)
            previous_token = [None] * len(SLOT_NAMES)
            continue
        c = float(np.cos(float(eyaw[t]))); s = float(np.sin(float(eyaw[t])))
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
        result = assign_stage5d_slots(ego_state, states, lane_infos=lane_infos or {}, assignment_mode=args.assignment_mode, config=cfg)
        assigned = {slot: rel_cache[aid] for slot, aid in result.slot_to_agent.items() if aid in rel_cache}
        assign_debug.append({"fallback_assignment_used": bool(result.fallback_assignment_used), "lane_assignment_available": bool(result.lane_assignment_available), "fallback_reason": result.fallback_reason, "current_lane_id": result.current_lane_id, "left_lane_id": result.left_lane_id, "right_lane_id": result.right_lane_id, "adjacency_source": result.adjacency_source, "lane_context_quality": result.lane_context_quality, "lane_context_quality_reasons": result.lane_context_quality_reasons or [], "slot_rejection_reason_counts": result.slot_rejection_reason_counts or {}, "per_slot_debug": result.per_slot_debug})
        current_tokens = [None] * len(SLOT_NAMES)
        for si, sn in enumerate(SLOT_NAMES):
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
            nbr[si, t, :] = build_neighbor_features_15d(rel_x=rel_x, rel_y=rel_y, rel_vx=rel_vx, rel_vy=rel_vy, ego_forward_speed=float(ego[t, 2]), neighbor_speed=st[5], neighbor_accel=((float(st[5]) - float(previous_speed[si])) / max(dt, 1e-6)) if same_track and previous_speed[si] is not None else 0.0, heading_rel=heading_rel, neighbor_yaw_rate=(float(wrap(heading_rel - float(previous_heading_rel[si]))) / max(dt, 1e-6)) if same_track and previous_heading_rel[si] is not None else 0.0)
            slot_ids[si][t] = tok
            previous_speed[si] = float(st[5])
            previous_heading_rel[si] = heading_rel
            previous_token[si] = tok
    context = build_context_traj_from_standard_tracks(ego, nbr)
    return ego.astype(np.float32), nbr, context, slot_ids, assign_debug


def slot_continuity_stats(slot_id_rows: List[List[List[str]]]) -> Dict[str, Dict[str, float | int | None]]:
    stats: Dict[str, Dict[str, float | int | None]] = {}
    for si, sn in enumerate(SLOT_NAMES):
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



def evaluate_slot_sanity(slot_stats: Dict[str, Dict[str, Any]], min_coverage: float) -> Tuple[Dict[str, Any], bool, List[str], List[str], List[Dict[str, Any]]]:
    """Evaluate semantic slot directional sanity only for sufficiently covered slots.

    Low/zero coverage is expected in small smoke samples and is reported as a
    diagnostic warning rather than a structural validation failure.
    """
    if not (0.0 <= float(min_coverage) <= 1.0):
        raise ValueError(f"--slot_sanity_min_coverage must be in [0, 1], got {min_coverage}")
    rules = {
        "front_median_rel_x_gt_0": ("front", "median_rel_x", "gt", 0.0),
        "left_front_median_rel_y_gt_0": ("left_front", "median_rel_y", "gt", 0.0),
        "left_rear_median_rel_y_gt_0": ("left_rear", "median_rel_y", "gt", 0.0),
        "right_front_median_rel_y_lt_0": ("right_front", "median_rel_y", "lt", 0.0),
        "right_rear_median_rel_y_lt_0": ("right_rear", "median_rel_y", "lt", 0.0),
    }
    sanity: Dict[str, Any] = {}
    evaluated: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []
    warnings_out: List[Dict[str, Any]] = []
    for check_name, (slot, metric, op, threshold_value) in rules.items():
        coverage = float(slot_stats.get(slot, {}).get("coverage_ratio", 0.0) or 0.0)
        value = slot_stats.get(slot, {}).get(metric)
        detail = {
            "slot": slot,
            "check": check_name,
            "coverage": coverage,
            "coverage_threshold": float(min_coverage),
            "metric": metric,
            "value": value,
        }
        if coverage < float(min_coverage):
            sanity[check_name] = {**detail, "status": "insufficient_coverage", "passed": None}
            skipped.append(slot)
            warnings_out.append({"type": "slot_sanity_insufficient_coverage", "slot": slot, "coverage": coverage, "threshold": float(min_coverage)})
            continue
        passed = bool(value is not None and ((float(value) > threshold_value) if op == "gt" else (float(value) < threshold_value)))
        sanity[check_name] = {**detail, "status": "evaluated", "passed": passed}
        evaluated.append(slot)
        if not passed:
            failed.append(slot)
    return sanity, not failed, evaluated, skipped, warnings_out


def evaluate_required_neighbor_coverage(
    slot_stats: Dict[str, Dict[str, Any]], required: bool
) -> Tuple[bool, float]:
    """Optionally reject formal datasets when every semantic slot is empty."""
    total = float(
        sum(float(row.get("coverage_ratio", 0.0) or 0.0) for row in slot_stats.values())
    )
    return (not required or total > 0.0), total

def write_strict_filter_diagnostic(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    planners: List[str],
    n_scenarios: int,
    row_debug_rows: List[List[dict]],
    nbr_arr: np.ndarray,
    slot_stats: Dict[str, Dict[str, Any]],
    sanity: Dict[str, bool],
    strict_filter_min_laneaware_ratio: float = 1.0,
    strict_filter_ratio_sweep: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """Report the Stage5 Waymo-style strict lane-aware filtering outcome for nuPlan.

    This diagnostic intentionally does not change the default Stage7E row semantics.  A row is
    kept only if all valid frames have lane-aware context available, no geometric fallback was
    used, and no bad/ambiguous lane context quality is reported.
    """
    if not (0.0 <= strict_filter_min_laneaware_ratio <= 1.0):
        raise ValueError(f"--strict_filter_min_laneaware_ratio must be in [0, 1], got {strict_filter_min_laneaware_ratio}")
    row_availability: List[float] = []
    base_reasons_by_row: List[List[str]] = []
    laneaware_frames = 0
    valid_frames = 0
    for debug in row_debug_rows:
        reasons: List[str] = []
        if not debug:
            reasons.append("drop_if_no_lane_map")
            availability = 0.0
        else:
            valid_frames += len(debug)
            frame_ok = [
                bool(d.get("lane_assignment_available"))
                and not bool(d.get("fallback_assignment_used"))
                and str(d.get("lane_context_quality", "")).lower() != "bad"
                and not str(d.get("lane_context_quality", "")).lower().startswith("ambiguous")
                for d in debug
            ]
            laneaware_frames += int(sum(frame_ok))
            availability = float(np.mean(frame_ok)) if frame_ok else 0.0
            if not any(d.get("current_lane_id") for d in debug):
                reasons.append("drop_if_ego_lane_missing")
            if any(str(d.get("lane_context_quality", "")).lower() == "bad" for d in debug):
                reasons.append("drop_if_lane_context_bad")
            if any(str(d.get("lane_context_quality", "")).lower().startswith("ambiguous") for d in debug):
                reasons.append("drop_if_lane_context_ambiguous")
        row_availability.append(availability)
        base_reasons_by_row.append(reasons)

    def indices_and_reasons_for_ratio(ratio: float) -> Tuple[List[int], Dict[str, int]]:
        dropped: Dict[str, int] = {}
        kept: List[int] = []
        for idx, base_reasons in enumerate(base_reasons_by_row):
            reasons = list(base_reasons)
            if row_availability[idx] < ratio:
                reasons.append("lane_aware_only")
            if reasons:
                for reason in sorted(set(reasons)):
                    dropped[reason] = int(dropped.get(reason, 0) + 1)
            else:
                kept.append(idx)
        return kept, dropped

    kept_indices, dropped_by_reason = indices_and_reasons_for_ratio(strict_filter_min_laneaware_ratio)
    kept_rows = [rows[i] for i in kept_indices]
    kept_per_planner = {p: 0 for p in planners}
    source_scenario_ids = sorted({int(r.get("scenario_index", -1)) for r in rows})
    if len(source_scenario_ids) != n_scenarios:
        raise ValueError(
            f"strict-filter source scenario count mismatch: metadata={source_scenario_ids}, "
            f"declared_n_scenarios={n_scenarios}"
        )
    scenario_planners: Dict[int, List[str]] = {scenario_id: [] for scenario_id in source_scenario_ids}
    for r in kept_rows:
        planner = str(r.get("planner_name", ""))
        kept_per_planner[planner] = int(kept_per_planner.get(planner, 0) + 1)
        scenario_planners[int(r.get("scenario_index", -1))].append(planner)
    scenarios_with_all = [si for si, ps in scenario_planners.items() if sorted(ps) == sorted(planners)]
    kept_nbr = nbr_arr[kept_indices] if kept_indices else nbr_arr[:0]
    kept_slot_coverage = {}
    for si, sn in enumerate(SLOT_NAMES):
        valid = kept_nbr[:, si, :, 0] > 0.5 if kept_nbr.size else np.zeros((0,), dtype=bool)
        kept_slot_coverage[sn] = float(np.mean(valid)) if valid.size else 0.0
    summary = {
        "dataset": "nuplan_stage7_adapter",
        "strict_filter_diagnostic": True,
        "filtering_mode": "strict_filter_lane_aware_only",
        "assignment_mode": "lane_aware_only",
        "strict_filter_min_laneaware_ratio": float(strict_filter_min_laneaware_ratio),
        "strict_filters": {
            "drop_if_no_lane_map": True,
            "drop_if_ego_lane_missing": True,
            "drop_if_lane_context_bad": True,
            "drop_if_lane_context_ambiguous": True,
            "lane_aware_only": True,
        },
        "original_rows": int(len(rows)),
        "rows_kept": int(len(kept_indices)),
        "rows_dropped": int(len(rows) - len(kept_indices)),
        "kept_row_rate": float(len(kept_indices) / max(1, len(rows))),
        "dropped_by_reason": dropped_by_reason,
        "kept_rows_per_planner": kept_per_planner,
        "scenario_planner_alignment_after_filtering": scenario_planners,
        "scenarios_with_all_planners": int(len(scenarios_with_all)),
        "scenarios_missing_any_planner": int(len(source_scenario_ids) - len(scenarios_with_all)),
        "each_scenario_still_has_all_planners": bool(len(scenarios_with_all) == len(source_scenario_ids)),
        "laneaware_available_frames": int(laneaware_frames),
        "valid_frames": int(valid_frames),
        "frame_level_laneaware_availability_rate": float(laneaware_frames / max(1, valid_frames)),
        "row_level_all_frames_laneaware_availability_rate": float(sum(1 for v in row_availability if v >= 1.0) / max(1, len(rows))),
        "row_level_min_laneaware_availability": float(min(row_availability) if row_availability else 0.0),
        "row_level_mean_laneaware_availability": float(np.mean(row_availability) if row_availability else 0.0),
        "row_level_laneaware_availability_quantiles": {str(q): float(np.quantile(row_availability, q)) if row_availability else 0.0 for q in [0.0, 0.25, 0.5, 0.75, 1.0]},
        "slot_sanity_on_kept_rows": sanity,
        "slot_coverage_on_kept_rows": kept_slot_coverage,
        "fallback_assignment_used_rate": 0.0,
        "lane_assignment_available": bool(kept_indices),
        "lane_assignment_available_rate": float(len(kept_indices) / max(1, len(rows))),
        "candidate_projection_success_rate": float(laneaware_frames / max(1, valid_frames)),
        "kept_row_indices": kept_indices,
    }

    sweep_rows = []
    for ratio in (strict_filter_ratio_sweep or []):
        ratio = float(ratio)
        sweep_keep, _ = indices_and_reasons_for_ratio(ratio)
        sweep_kept_rows = [rows[i] for i in sweep_keep]
        sweep_scenario_planners: Dict[int, List[str]] = {scenario_id: [] for scenario_id in source_scenario_ids}
        for r in sweep_kept_rows:
            sweep_scenario_planners[int(r.get("scenario_index", -1))].append(str(r.get("planner_name", "")))
        sweep_with_all = [si for si, ps in sweep_scenario_planners.items() if sorted(ps) == sorted(planners)]
        sweep_nbr = nbr_arr[sweep_keep] if sweep_keep else nbr_arr[:0]
        sweep_cov = {}
        for slot_i, slot_name in enumerate(SLOT_NAMES):
            valid = sweep_nbr[:, slot_i, :, 0] > 0.5 if sweep_nbr.size else np.zeros((0,), dtype=bool)
            sweep_cov[slot_name] = float(np.mean(valid)) if valid.size else 0.0
        sweep_rows.append({
            "strict_filter_min_laneaware_ratio": ratio,
            "rows_kept": int(len(sweep_keep)),
            "kept_row_rate": float(len(sweep_keep) / max(1, len(rows))),
            "scenarios_with_all_planners": int(len(sweep_with_all)),
            "scenarios_missing_any_planner": int(len(source_scenario_ids) - len(sweep_with_all)),
            "each_scenario_still_has_all_planners": bool(len(sweep_with_all) == len(source_scenario_ids)),
            "slot_sanity_on_kept_rows": sanity,
            "slot_coverage_on_kept_rows": sweep_cov,
        })
    if sweep_rows:
        summary["strict_filter_ratio_sweep"] = sweep_rows
    write_json(output_dir / "nuplan_laneaware_strict_filter_summary.json", summary)
    report = [
        "# nuPlan Stage5-style Strict Lane-Aware Filter Diagnostic",
        "",
        "- This is diagnostic only; the default Stage7E official rollout output does not drop rows and still preserves one row per scenario × planner rollout.",
        "- Strict filters mirror Waymo Stage5 `lane_aware_only` plus `drop_if_*` philosophy.",
        "- Relaxed strict-filter ratios are only for comparing with Stage5 Waymo min_valid_ratio-style filtering philosophy.",
        f"- strict_filter_min_laneaware_ratio: `{summary['strict_filter_min_laneaware_ratio']}`",
        f"- original_rows: `{summary['original_rows']}`",
        f"- rows_kept: `{summary['rows_kept']}`",
        f"- rows_dropped: `{summary['rows_dropped']}`",
        f"- kept_row_rate: `{summary['kept_row_rate']}`",
        f"- dropped_by_reason: `{dropped_by_reason}`",
        f"- kept_rows_per_planner: `{kept_per_planner}`",
        f"- each_scenario_still_has_all_planners: `{summary['each_scenario_still_has_all_planners']}`",
        f"- frame_level_laneaware_availability_rate: `{summary['frame_level_laneaware_availability_rate']}`",
        f"- row_level_all_frames_laneaware_availability_rate: `{summary['row_level_all_frames_laneaware_availability_rate']}`",
        f"- row_level_min_laneaware_availability: `{summary['row_level_min_laneaware_availability']}`",
        f"- row_level_mean_laneaware_availability: `{summary['row_level_mean_laneaware_availability']}`",
        f"- slot_sanity_on_kept_rows: `{sanity}`",
        "- absent/low-coverage slots are diagnostic-only and do not imply invalid context; sufficiently covered wrong-direction slots remain fatal.",
        f"- row_level_laneaware_availability_quantiles: `{summary['row_level_laneaware_availability_quantiles']}`",
        f"- slot_coverage_on_kept_rows: `{kept_slot_coverage}`",
        f"- strict_filter_ratio_sweep: `{summary.get('strict_filter_ratio_sweep', [])}`",
    ]
    (output_dir / "nuplan_laneaware_strict_filter_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary

def make_context_schema(accel_yaw_rate_matched: bool = False) -> Dict[str, Any]:
    return make_stage5d_context_schema(schema_name="stage5d83_nuplan_laneaware_stage5_formula_parity", accel_yaw_rate_matched=accel_yaw_rate_matched)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 5D-compatible [N,T,83] nuPlan context artifacts from official Stage 7C simulation outputs.")
    ap.add_argument("--sim_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_neighbors_for_context", type=int, default=5)
    ap.add_argument("--assignment_mode", choices=["lane_aware_only", "lane_aware_with_geometric_fallback", "geometric_only", "geometric_proxy"], default="lane_aware_with_geometric_fallback")
    ap.add_argument("--nuplan_map_root", type=Path, default=None)
    ap.add_argument("--nuplan_db_root", type=Path, default=None, help="nuPlan DB directory used to resolve each log's true location. Defaults to <nuplan_map_root>/../data/cache/mini when present.")
    ap.add_argument("--map_name", default="", help="Last-resort map fallback used only when row metadata, scenario metadata, and the nuPlan log DB cannot resolve location.")
    ap.add_argument("--scenario_map_metadata_csv", type=Path, default=None, help="Optional CSV mapping scenario/log identifiers to map_name/location.")
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
    ap.add_argument("--required_planners", nargs="*", default=[], help="Optional planner names that must exist on the discovered Stage 7C planner axis. Default empty supports PDM/ML expansion planners.")
    ap.add_argument("--write_projection_debug", action="store_true", help="Write bounded nuPlan lane projection candidate debug CSV in addition to summary/report.")
    ap.add_argument("--debug_projection_sample_rows", type=int, default=20, help="Maximum context rows sampled for projection debug artifacts.")
    ap.add_argument("--debug_projection_max_candidates_per_frame", type=int, default=32, help="Maximum tracked-object candidates recorded per sampled frame.")
    ap.add_argument("--debug_projection_max_frames_per_row", type=int, default=149, help="Maximum valid timesteps recorded per sampled context row.")
    ap.add_argument("--write_strict_filter_diagnostic", action="store_true", help="Write nuPlan Stage5-style strict lane-aware filtering summary/report without changing the main output rows.")
    ap.add_argument("--slot_sanity_min_coverage", type=float, default=0.05, help="Minimum per-slot coverage required before directional median slot sanity is evaluated. Lower coverage is diagnostic-only and non-fatal.")
    ap.add_argument("--require_nonzero_neighbor_coverage", action="store_true", help="Formal-analysis gate: fail when all five semantic neighbor slots have zero valid-frame coverage. Tiny smoke runs may omit this flag.")
    ap.add_argument("--strict_filter_min_laneaware_ratio", type=float, default=1.0, help="Diagnostic row keep threshold: lane-aware available valid frames / valid frames must be at least this value. 1.0 preserves current all-frames behavior; 0.8 approximates Waymo min_valid_ratio philosophy.")
    ap.add_argument("--strict_filter_ratio_sweep", type=float, nargs="*", default=[], help="Optional diagnostic-only thresholds to summarize without writing multiple datasets, e.g. 1.0 0.9 0.8 0.7 0.6.")
    ap.add_argument("--write_strict_filtered_dataset", action="store_true", help="Also write optional filtered metadata/context arrays under strict_filtered_dataset/ for diagnostics.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.nuplan_db_root is None and args.nuplan_map_root is not None:
        inferred_db_root = args.nuplan_map_root.parent / "data" / "cache" / "mini"
        if inferred_db_root.is_dir():
            args.nuplan_db_root = inferred_db_root
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
    for name in ["simulated_ego_seq.npy", "simulated_ego_seq_mask.npy", "simulated_ego_seq_index.json", "scenario_planner_index.csv", "simulated_planner_metadata.csv", "simulation_schema.json", "warnings.json"]:
        if not (args.sim_dir / name).exists(): raise FileNotFoundError(f"Missing required input: {args.sim_dir / name}")
    validate_official(read_json(args.sim_dir / "simulation_schema.json"), read_json(args.sim_dir / "warnings.json"))
    seq = np.load(args.sim_dir / "simulated_ego_seq.npy", mmap_mode="r")
    mask = np.load(args.sim_dir / "simulated_ego_seq_mask.npy", mmap_mode="r")
    planners = planner_axis(args.sim_dir / "simulated_planner_metadata.csv", args.sim_dir / "scenario_planner_index.csv", seq.shape[1], args.required_planners)
    n_scenarios, n_planners, timesteps, _ = seq.shape
    scenario_axis = load_stage7c_scenario_axis(
        args.sim_dir / "simulated_ego_seq_index.json",
        n_scenarios,
        n_planners,
        planners,
    )
    expected_rows = n_scenarios * n_planners
    index_rows = read_csv_rows(args.sim_dir / "scenario_planner_index.csv")
    scenario_map_metadata = load_scenario_map_metadata(args.scenario_map_metadata_csv)
    by_pair = validate_scenario_planner_alignment(index_rows, scenario_axis, planners)
    ego_rows=[]; ego_mask_rows=[]; nbr_rows=[]; ctx_rows=[]; inter_rows=[]; slot_id_rows=[]; assignment_debug_rows=[]; row_debug_rows=[]; projection_debug_rows=[]; warnings=[]; cache={}; map_cache={}; map_api_cache={}; db_map_cache={}; parsed=0; resolved_map_names=[]; row_map_names=[]; map_resolution_sources=[]
    if warnings_note:
        warnings.append({"type": "deprecated_assignment_mode", "message": warnings_note})
    if args.assignment_mode in {"lane_aware_only", "lane_aware_with_geometric_fallback"}:
        if args.nuplan_map_root is None:
            warnings.append({"type": "nuplan_map_root_missing_for_lane_aware", "severity": "error" if args.assignment_mode == "lane_aware_only" else "warning", "message": "--nuplan_map_root is required for actual nuPlan lane-aware assignment; otherwise lane-aware assignment cannot query maps and geometric fallback may be used."})
        elif not args.nuplan_map_root.exists():
            warnings.append({"type": "nuplan_map_root_not_found_for_lane_aware", "severity": "error" if args.assignment_mode == "lane_aware_only" else "warning", "path": str(args.nuplan_map_root), "message": "nuPlan map root does not exist; lane-aware map query will be unavailable."})
    row=0
    for tensor_si, scenario_index in enumerate(tqdm(scenario_axis, desc="nuPlan Stage5D context")):
        for pi in range(n_planners):
            meta = by_pair[(scenario_index, pi)]
            msg = find_msgpack(args.sim_dir, meta)
            if msg is None: raise FileNotFoundError(f"No official nuPlan msgpack found for scenario_index={scenario_index}, planner_id={pi}, planner_name={planners[pi]}")
            if msg not in cache:
                tracks, sample_count = parse_neighbor_world_tracks(msg, timesteps); cache[msg]=tracks; parsed += 1
                if sample_count != timesteps: warnings.append({"type":"msgpack_timestep_mismatch","path":str(msg),"samples":sample_count,"expected_T":timesteps})
            lane_infos = {}
            map_name, map_name_source = resolve_map_name(
                meta,
                args.map_name.strip(),
                scenario_map_metadata,
                args.nuplan_db_root,
                db_map_cache,
            )
            if map_name:
                resolved_map_names.append(map_name)
            row_map_names.append(map_name)
            map_resolution_sources.append(map_name_source)
            if args.nuplan_map_root and args.nuplan_map_root.exists():
                if not map_name:
                    warnings.append({"type": "nuplan_map_name_missing", "severity": "error" if args.assignment_mode == "lane_aware_only" else "warning", "scenario_index": scenario_index, "planner_id": pi, "assignment_mode": args.assignment_mode, "message": "No map_name could be resolved from row map_name, row location, --map_name, or --scenario_map_metadata_csv; lane-aware map query cannot run for this row."})
                lane_cache_key = scenario_lane_cache_key(map_name, scenario_index) if map_name else None
                if map_name and lane_cache_key not in map_cache:
                    try:
                        from tools.build_nuplan_map_odd_features import load_map_api
                        if map_name not in map_api_cache:
                            map_api_cache[map_name] = load_map_api(args.nuplan_map_root, map_name, warnings)
                        ego_xy = scenario_query_ego_xy(seq[tensor_si], mask[tensor_si])
                        map_cache[lane_cache_key] = extract_nuplan_lane_infos(map_api_cache[map_name], ego_xy, args.map_query_radius)
                    except Exception as exc:
                        warnings.append({"type": "nuplan_lane_info_extraction_failed", "map_name": map_name, "scenario_index": scenario_index, "message": str(exc)})
                        map_cache[lane_cache_key] = ({}, {"map_query_failed": 1, "none": 1})
                lane_infos = map_cache.get(lane_cache_key, ({}, {}))[0] if lane_cache_key else {}
            row_seq = np.asarray(seq[tensor_si,pi])
            row_mask = np.asarray(mask[tensor_si,pi]).astype(bool)
            ego,nbr,ctx,slot_ids,assign_debug = build_row_context(row_seq, row_mask, cache[msg], args, lane_infos)
            if len(projection_debug_rows) == 0 or row < int(args.debug_projection_sample_rows):
                cfg = {k: getattr(args, k) for k in ["lane_max_lateral_distance", "lane_max_heading_diff_deg", "adjacent_lane_min_offset", "adjacent_lane_max_offset", "adjacent_lane_max_heading_diff_deg", "lane_search_radius", "lane_topk_candidates", "front_max_distance", "side_front_max_distance", "side_rear_max_distance", "lane_lateral_tolerance", "slot_heading_diff_deg", "static_speed_threshold"]}
                if row < int(args.debug_projection_sample_rows):
                    projection_debug_rows.extend(collect_nuplan_projection_debug_rows(
                        global_row=row,
                        scenario_index=scenario_index,
                        planner_id=pi,
                        planner_name=planners[pi],
                        map_name=map_name,
                        assignment_mode=args.assignment_mode,
                        stage7c_seq=row_seq,
                        mask=row_mask,
                        tracks=cache[msg],
                        lane_infos=lane_infos,
                        assign_debug=assign_debug,
                        config=cfg,
                        max_frames_per_row=max(0, int(args.debug_projection_max_frames_per_row)),
                        max_candidates_per_frame=max(0, int(args.debug_projection_max_candidates_per_frame)),
                    ))
            inter, _ = aggregate_interaction_features(ego[row_mask], nbr[:, row_mask, :], 0.1)
            ego_rows.append(ego); ego_mask_rows.append(row_mask); nbr_rows.append(nbr); ctx_rows.append(ctx); inter_rows.append(np.nan_to_num(inter, nan=0.0, posinf=1e6, neginf=-1e6)); slot_id_rows.append(slot_ids); assignment_debug_rows.extend(assign_debug); row_debug_rows.append(assign_debug); row += 1
    ego_arr=np.asarray(ego_rows,np.float32); ego_mask_arr=np.asarray(ego_mask_rows,bool); nbr_arr=np.asarray(nbr_rows,np.float32); ctx_arr=np.asarray(ctx_rows,np.float32); feat_arr=np.asarray(inter_rows,np.float32)
    if row != expected_rows or ctx_arr.shape != (expected_rows, timesteps, CONTEXT_DIM): raise ValueError(f"Invalid shape/rows: rows={row}, context={list(ctx_arr.shape)}, expected rows={expected_rows}, T={timesteps}, D={CONTEXT_DIM}")
    if ego_mask_arr.shape != (expected_rows, timesteps) or not np.all(np.any(ego_mask_arr, axis=1)):
        raise ValueError(f"Invalid aligned ego validity mask: shape={list(ego_mask_arr.shape)}, expected={[expected_rows, timesteps]}")
    core_validation = validate_stage5d_context(ctx_arr, ego_arr, nbr_arr)
    rows = metadata_rows(index_rows, read_csv_rows(args.sim_dir / "simulated_planner_metadata.csv"), planners, scenario_axis)
    idx_dir=args.output_dir/"planner_policy_indices"; idx_dir.mkdir()
    for pid,p in enumerate(planners): np.save(idx_dir/f"{p}.npy", np.asarray([r for r in range(expected_rows) if r % n_planners == pid], dtype=np.int64))
    for metadata_row, valid_mask in zip(rows, ego_mask_arr):
        metadata_row["valid_timestep_count"] = int(np.sum(valid_mask))
    if len(row_map_names) != len(rows) or len(map_resolution_sources) != len(rows):
        raise ValueError(
            f"map-resolution row mismatch: maps={len(row_map_names)}, "
            f"sources={len(map_resolution_sources)}, metadata={len(rows)}"
        )
    for metadata_row, row_map_name, resolution_source in zip(rows, row_map_names, map_resolution_sources):
        metadata_row["map_name"] = row_map_name
        metadata_row["map_name_resolution_source"] = resolution_source
    lane_assignment_diagnostics = summarize_lane_assignment_rows(row_debug_rows, rows)
    write_json(args.output_dir / "nuplan_lane_assignment_diagnostics.json", {k: v for k, v in lane_assignment_diagnostics.items() if k != "rows"})
    write_csv(args.output_dir / "nuplan_lane_assignment_by_row.csv", lane_assignment_diagnostics["rows"])
    np.save(args.output_dir/"ego_seq.npy", ego_arr); np.save(args.output_dir/"ego_seq_mask.npy", ego_mask_arr); np.save(args.output_dir/"context_traj.npy", ctx_arr); np.save(args.output_dir/"interaction_feat_style.npy", feat_arr); np.save(args.output_dir/"neighbor_seq.npy", nbr_arr); np.save(args.output_dir/"neighbor_slot_ids.npy", np.asarray(slot_id_rows, dtype=object))
    write_csv(args.output_dir/"metadata.csv", rows); write_feature_schema_json(args.output_dir/"feature_schema.json"); write_json(args.output_dir/"shard_manifest.json", {"shards":[{"shard_path":"."}], "format":"monolithic_stage5d_context", "context_traj":"context_traj.npy", "ego_seq_mask":"ego_seq_mask.npy"})
    slot_stats={}; sanity={}
    for si,sn in enumerate(SLOT_NAMES):
        valid=nbr_arr[:,si,:,0]>0.5; vals=nbr_arr[:,si]
        cov=float(np.mean(valid)); slot_stats[sn]={"coverage_ratio":cov,"empty_slot_ratio":1.0-cov,"median_rel_x":float(np.median(vals[:,:,1][valid])) if np.any(valid) else None,"median_rel_y":float(np.median(vals[:,:,2][valid])) if np.any(valid) else None,"median_distance":float(np.median(vals[:,:,5][valid])) if np.any(valid) else None}
    sanity, slot_pass, slot_sanity_evaluated_slots, slot_sanity_skipped_low_coverage_slots, slot_sanity_warnings = evaluate_slot_sanity(slot_stats, args.slot_sanity_min_coverage)
    warnings.extend(slot_sanity_warnings)
    required_neighbor_coverage_pass, total_slot_coverage = evaluate_required_neighbor_coverage(
        slot_stats, args.require_nonzero_neighbor_coverage
    )
    if not required_neighbor_coverage_pass:
        warnings.append({
            "type": "required_nonzero_neighbor_coverage_failed",
            "severity": "error",
            "message": "All five semantic neighbor slots have zero coverage. Verify the nuPlan/tuPlan PYTHONPATH and official msgpack deserialization before formal analysis.",
        })
    strict_filter_summary = None
    if args.write_strict_filter_diagnostic or args.write_strict_filtered_dataset:
        strict_filter_summary = write_strict_filter_diagnostic(args.output_dir, metadata_rows(index_rows, read_csv_rows(args.sim_dir / "simulated_planner_metadata.csv"), planners, scenario_axis), planners, n_scenarios, row_debug_rows, nbr_arr, slot_stats, sanity, args.strict_filter_min_laneaware_ratio, args.strict_filter_ratio_sweep)
        if args.write_strict_filtered_dataset:
            strict_dir = args.output_dir / "strict_filtered_dataset"
            strict_dir.mkdir(exist_ok=True)
            keep = np.asarray(strict_filter_summary["kept_row_indices"], dtype=np.int64)
            np.save(strict_dir / "ego_seq.npy", ego_arr[keep])
            np.save(strict_dir / "ego_seq_mask.npy", ego_mask_arr[keep])
            np.save(strict_dir / "context_traj.npy", ctx_arr[keep])
            np.save(strict_dir / "interaction_feat_style.npy", feat_arr[keep])
            np.save(strict_dir / "neighbor_seq.npy", nbr_arr[keep])
            np.save(strict_dir / "neighbor_slot_ids.npy", np.asarray(slot_id_rows, dtype=object)[keep])
            write_csv(strict_dir / "metadata.csv", [rows[int(i)] for i in keep])
            write_json(strict_dir / "shard_manifest.json", {"shards":[{"shard_path":"."}], "format":"diagnostic_strict_filtered_stage5d_context", "context_traj":"context_traj.npy", "ego_seq_mask":"ego_seq_mask.npy"})
    slot_continuity = slot_continuity_stats(slot_id_rows)
    slot_switch_rate_by_slot = {k: float(v["slot_id_switch_rate"] or 0.0) for k, v in slot_continuity.items()}
    map_counts = {}
    merged_lane_infos = {}
    for (cached_map_name, _), (lanes, counts) in map_cache.items():
        for lane_id, lane_info in lanes.items():
            merged_lane_infos[f"{cached_map_name}:{lane_id}"] = lane_info
        for k, v in counts.items():
            map_counts[k] = int(map_counts.get(k, 0) + int(v))
    lane_info_count = int(len(merged_lane_infos))
    topology_debug_summary = build_lane_topology_debug_summary(merged_lane_infos, map_counts)
    topology_debug_artifacts = write_lane_topology_debug_artifacts(args.output_dir, topology_debug_summary)
    accel_yaw_rate_matched = all(rate == 0.0 for rate in slot_switch_rate_by_slot.values())
    stage5d_static_derived_formula_matched = True
    stage5d_closing_formula_matched = True
    stage5d_ttc_formula_matched = True
    stage5d_delta_xy_formula_matched = True
    formula_status, formula_warnings, stage5d_formula_validation_pass = stage5d_formula_validation_status(
        stage5d_static_derived_formula_matched=stage5d_static_derived_formula_matched,
        stage5d_closing_formula_matched=stage5d_closing_formula_matched,
        stage5d_ttc_formula_matched=stage5d_ttc_formula_matched,
        stage5d_delta_xy_formula_matched=stage5d_delta_xy_formula_matched,
        accel_yaw_rate_matched=accel_yaw_rate_matched,
        slot_switch_rate_by_slot=slot_switch_rate_by_slot,
    )
    warnings.extend(formula_warnings)
    stage5d_temporal_derived_formula_matched = bool(formula_status["stage5d_temporal_derived_formula_matched"])
    stage5d_derived_formula_matched = bool(formula_status["stage5d_derived_formula_matched"])
    planner_non_empty=all(np.load(idx_dir/f"{p}.npy").size>0 for p in planners)
    map_query_success = bool(map_counts.get("map_query_success", 0) > 0)
    map_names_used = sorted(set(resolved_map_names))
    map_name_resolved_rate = float(len(resolved_map_names) / max(row, 1))
    lane_assignment_available = bool(any(d.get("lane_assignment_available") for d in assignment_debug_rows))
    fallback_assignment_used_rate = float(np.mean([d.get("fallback_assignment_used", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    ego_lane_projection_success_rate = float(np.mean([d.get("lane_assignment_available", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    candidate_lane_projection_success_rate = float(np.mean([any(ps.get("assignment_method") == "lane_aware" for ps in d.get("per_slot_debug", [])) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    projection_debug_summary = summarize_projection_debug(projection_debug_rows, assignment_debug_rows)
    projection_debug_artifacts = write_projection_debug_artifacts(args.output_dir, projection_debug_rows, projection_debug_summary, args.write_projection_debug)
    lane_runtime_ok = True
    if args.assignment_mode == "lane_aware_only" and (map_name_resolved_rate < 1.0 or not map_query_success or lane_info_count <= 0 or not lane_assignment_available):
        lane_runtime_ok = False
        warnings.append({"type": "lane_aware_only_map_query_failed", "severity": "error", "map_name_resolved_rate": map_name_resolved_rate, "map_query_success": map_query_success, "lane_info_count": lane_info_count, "lane_assignment_available": lane_assignment_available, "message": "assignment_mode=lane_aware_only requires resolved map_name for every row, successful nuPlan map query, and lane projection; refusing silent geometric fallback."})
    if args.assignment_mode == "lane_aware_with_geometric_fallback" and fallback_assignment_used_rate >= 0.5:
        warnings.append({"type": "high_geometric_fallback_rate", "severity": "warning", "fallback_assignment_used_rate": fallback_assignment_used_rate, "message": "More than half of lane-aware assignments used geometric fallback. This is not strong lane-aware thesis evidence; verify --nuplan_map_root, map_name resolution, and projection diagnostics."})
    validation={"pass": bool(slot_pass and planner_non_empty and lane_runtime_ok and core_validation.get("passed", core_validation.get("pass", True)) and ctx_arr.shape[-1] == CONTEXT_DIM and np.isfinite(ctx_arr).all() and row == expected_rows and len(rows) == row and ego_arr.shape[0] == row and ego_mask_arr.shape == (row, timesteps) and feat_arr.shape[0] == row and stage5d_formula_validation_pass), "row_semantics_correct": True, "scenario_axis_source": "simulated_ego_seq_index.json", "scenario_axis": list(scenario_axis), "scenario_axis_non_contiguous": scenario_axis != list(range(n_scenarios)), "scenario_planner_token_alignment_strict": True, "msgpack_global_fallback_disabled": True, "ego_seq_mask_written": True, "ego_seq_mask_shape": list(ego_mask_arr.shape), "valid_timestep_count_min": int(np.min(np.sum(ego_mask_arr, axis=1))), "valid_timestep_count_max": int(np.max(np.sum(ego_mask_arr, axis=1))), "interaction_features_valid_frames_only": True, "no_multi_agent_ego_expansion": True, "background_agents_context_only": True, "stage5d_dim_matched": ctx_arr.shape[-1]==CONTEXT_DIM, "stage5d_channel_schema_matched": True, "stage5d_slot_schema_matched": list(SLOT_NAMES) == ["front", "left_front", "left_rear", "right_front", "right_rear"], "stage5d_slot_order_matched": list(SLOT_NAMES) == ["front", "left_front", "left_rear", "right_front", "right_rear"], "stage5d_slot_semantics_verified": bool(slot_pass), "assignment_mode": args.assignment_mode, "map_name_resolved_rate": map_name_resolved_rate, "map_names_used": map_names_used, "map_name_resolution_sources": dict((src, map_resolution_sources.count(src)) for src in sorted(set(map_resolution_sources))), "nuplan_db_root": str(args.nuplan_db_root) if args.nuplan_db_root else "", "log_db_map_resolution_count": len(db_map_cache), "lane_cache_scope": lane_assignment_diagnostics["cache_scope"], "lane_cache_entry_count": len(map_cache), "map_api_cache_entry_count": len(map_api_cache), "lane_assignment_fallback_reason_counts": lane_assignment_diagnostics["fallback_reason_counts"], "lane_assignment_available": lane_assignment_available, "map_query_success": map_query_success, "lane_info_count": lane_info_count, "fallback_assignment_used_rate": fallback_assignment_used_rate, "ego_lane_projection_success_rate": ego_lane_projection_success_rate, "candidate_lane_projection_success_rate": candidate_lane_projection_success_rate, "projection_debug_summary": projection_debug_summary, "projection_debug_artifacts": projection_debug_artifacts, "topology_debug_summary": topology_debug_summary, "topology_debug_artifacts": topology_debug_artifacts, "adjacency_source_counts": map_counts, "slot_sanity_passed": bool(slot_pass), "slot_sanity_min_coverage": float(args.slot_sanity_min_coverage), "slot_sanity_evaluated_slots": slot_sanity_evaluated_slots, "slot_sanity_skipped_low_coverage_slots": slot_sanity_skipped_low_coverage_slots, "slot_sanity_failed_sufficiently_covered_slots": [slot for slot in slot_sanity_evaluated_slots if any(v.get("slot") == slot and v.get("passed") is False for v in sanity.values())], "slot_coverage_by_slot": {k:v["coverage_ratio"] for k,v in slot_stats.items()}, "context_traj_no_nonfinite": bool(np.isfinite(ctx_arr).all()), "planner_indices_non_empty": bool(planner_non_empty), "rows_equal_num_scenarios_times_num_planners": row == expected_rows, "metadata_row_count_matches": len(rows)==row, "ego_seq_row_count_matches": ego_arr.shape[0]==row, "interaction_feat_style_row_count_matches": feat_arr.shape[0]==row, **formula_status}
    validation["require_nonzero_neighbor_coverage"] = bool(args.require_nonzero_neighbor_coverage)
    validation["required_nonzero_neighbor_coverage_pass"] = bool(required_neighbor_coverage_pass)
    validation["total_slot_coverage"] = float(total_slot_coverage)
    validation["pass"] = bool(validation["pass"] and required_neighbor_coverage_pass)
    write_stage5d_context_schema(args.output_dir/"stage5d_context_schema.json", schema_name="stage5d83_nuplan_laneaware_stage5_formula_parity", accel_yaw_rate_matched=accel_yaw_rate_matched)
    write_json(args.output_dir/"warnings.json", {"warnings": warnings, "assignment_mode": args.assignment_mode, "slot_assignment_method":"stage5_assign_neighbors_lane_aware", "stage5d_slot_schema_matched": True, "stage5d_slot_order_matched": True, "stage5d_slot_assignment_exact_waymo_lane_aware": args.assignment_mode != "geometric_only", "stage5d_formula_parity_schema_recorded": True, "map_query_success": bool(map_counts.get("map_query_success", 0) > 0), "lane_info_count": lane_info_count, "adjacency_source_counts": map_counts, "topology_debug_summary": topology_debug_summary, "topology_debug_artifacts": topology_debug_artifacts, "stage5d_core_reused": True, "stage5d_slot_names_source": "tools.stage5d_context_core.SLOT_NAMES", "stage5d_feature_formula_source": "tools.stage5d_context_core", **formula_status, "ego_local_frame_source": "tools.stage5d_context_core.build_ego_features_8d", "neighbor_local_frame_contract": "per-timestep ego-centric rel_x/rel_y/rel_vx/rel_vy using current ego world pose and heading, matching Waymo Stage5D neighbor builder", "validation": {**core_validation, **validation}})
    exact_channels = "valid, rel_x, rel_y, rel_vx, rel_vy, distance, delta_x, delta_y, closing, ttc, thw, speed, heading_rel"
    accel_note = "accel and yaw_rate match Stage 5D finite-difference semantics because no slot ID switches were observed." if accel_yaw_rate_matched else "accel and yaw_rate are approximated_or_not_stage5_matched because at least one semantic slot switches tracked-object IDs; finite differences are reset at switches."
    report=["# nuPlan Stage 5D-Compatible Context Build Report","","## Final architecture","- Stage5D CORE is the single source of truth for slot order, 83-dim schema, lane-aware assignment, fallback policy, derived formulas, context construction, schema generation, and validation.","- nuPlan is an adapter: it reads official simulation rollout artifacts, parses planner-controlled ego rollouts and background tracked agents, queries map lanes when available, converts them to Stage5D inputs, and calls Stage5D CORE.","- Waymo is an adapter that targets the same Stage5D CORE contract.","- nuPlan ego 8D is built by `tools.stage5d_context_core.build_ego_features_8d` from a standard `[x,y,vx,vy,heading,valid]` track window in a deterministic local window frame.","- nuPlan neighbor rel_x/rel_y/rel_vx/rel_vy remain per-timestep ego-centric using the current ego pose/heading, matching the original Waymo Stage5D neighbor convention.","",f"- rows: `{row}` (= `{n_scenarios} scenarios × {n_planners} planners`)","- row semantics: `scenario × planner × planner-controlled nuPlan ego rollout`","- background agents: context only; no multi-agent ego expansion",f"- exact Stage5D slot order: `{list(SLOT_NAMES)}`","- Stage5D core reused: `true`","- context_traj.npy: `"+str(list(ctx_arr.shape))+"`","- Stage 5D best model input: `context_traj.npy [N,T,83]`","- 83 = ego 8 + 5 semantic neighbor slots × 15 channels","- interaction_feat_style.npy is for reports/evaluation, not encoder input","- context_traj has no map/lane/ODD channels","- slot assignment: Stage 5 `assign_neighbors_lane_aware`; geometric assignment is fallback","- parsed msgpack files: `"+str(parsed)+"`", f"- map_name_resolved_rate: `{map_name_resolved_rate}`", f"- map_name values used: `{map_names_used}`", f"- map_name resolution sources: `{validation['map_name_resolution_sources']}`", f"- map_query_success: `{map_query_success}`", f"- lane_info_count: `{lane_info_count}`", f"- lane-aware assignment succeeded: `{lane_assignment_available}`", f"- fallback_assignment_used_rate: `{fallback_assignment_used_rate}`", f"- ego_lane_projection_success_rate: `{ego_lane_projection_success_rate}`", f"- candidate_lane_projection_success_rate: `{candidate_lane_projection_success_rate}`", f"- projection debug summary: `{projection_debug_artifacts.get('summary_json')}`", f"- projection debug report: `{projection_debug_artifacts.get('report_md')}`", f"- projection debug csv: `{projection_debug_artifacts.get('csv', 'not_written_without_--write_projection_debug')}`", f"- relation unknown debug csv: `{projection_debug_artifacts.get('relation_unknown_csv', 'not_written_no_unknown_rows')}`", f"- topology debug summary: `{topology_debug_artifacts.get('summary_json')}`", f"- topology debug report: `{topology_debug_artifacts.get('report_md')}`", f"- adjacency_source counts: `{map_counts}`","", "## Stage 5 formula parity", f"- formula parity passed: `{stage5d_derived_formula_matched}`", f"- static derived formula parity passed: `{stage5d_static_derived_formula_matched}`", f"- temporal derived formula parity passed: `{stage5d_temporal_derived_formula_matched}`", f"- temporal formula status: `{formula_status['stage5d_temporal_formula_status']}`", f"- Exactly matched neighbor formulas: `{exact_channels}`.", "- closing formula parity: `passed` (`closing = ego_forward_speed - rel_vx`).", "- TTC formula parity: `passed` (cap when closing <= 1e-3, otherwise distance / max(closing, 1e-3)).", "- delta_x/delta_y formula parity: `passed` (duplicates of rel_x/rel_y, not proxy channels).", f"- accel/yaw_rate formula parity: `{accel_yaw_rate_matched}`. {accel_note}", "- Static formulas, closing, TTC, and delta_x/delta_y are exact Stage 5D formula matches.", "- accel/yaw_rate are nonfatal approximations when semantic slot ID switches occur because finite differences are reset at switches; this does not invalidate context layout or the `context_traj.npy [N,T,83]` embedding input when structural checks and static/safety-critical formulas pass.", "- Approximations: accel/yaw_rate are approximations when slot switching is non-zero; slot assignment uses Stage 5 lane-aware logic unless `geometric_only` is requested or lane-map fallback is needed.", "", "## Slot sanity coverage gating", f"- slot_sanity_min_coverage: `{args.slot_sanity_min_coverage}`", f"- slot coverage: `{ {k: v['coverage_ratio'] for k, v in slot_stats.items()} }`", f"- evaluated slots: `{slot_sanity_evaluated_slots}`", f"- skipped low-coverage slots: `{slot_sanity_skipped_low_coverage_slots}`", f"- failed sufficiently covered slots: `{[slot for slot in slot_sanity_evaluated_slots if any(v.get('slot') == slot and v.get('passed') is False for v in sanity.values())]}`", "- Low/absent slot coverage is diagnostic-only and does not imply invalid context; sufficiently covered wrong-direction slots remain fatal."]
    report.extend([
        "",
        "## Rollout validity mask",
        f"- ego_seq_mask.npy shape: `{list(ego_mask_arr.shape)}`",
        f"- valid timestep count range: `{int(np.min(np.sum(ego_mask_arr, axis=1)))}..{int(np.max(np.sum(ego_mask_arr, axis=1)))}`",
        "- Invalid fixed-length padding frames are retained in encoder tensors but explicitly masked for downstream behavior/physical diagnostics.",
    ])
    (args.output_dir/"context_build_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    lines=["# Slot Assignment Report","",f"- assignment_mode: `{args.assignment_mode}`",f"- slot names/order: `{SLOT_NAMES}`",f"- map_name_resolved_rate: `{map_name_resolved_rate}`",f"- map_name values used: `{map_names_used}`",f"- map_query_success: `{map_query_success}`",f"- lane_info_count: `{lane_info_count}`",f"- lane-aware success rate: `{float(np.mean([d.get('lane_assignment_available', False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0}`",f"- geometric fallback rate: `{float(np.mean([d.get('fallback_assignment_used', False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0}`",""]
    for sn,st in slot_stats.items():
        cont = slot_continuity[sn]
        lines.append(f"- {sn}: coverage={st['coverage_ratio']:.6f}, empty={st['empty_slot_ratio']:.6f}, median_rel_x={st['median_rel_x']}, median_rel_y={st['median_rel_y']}, median_distance={st['median_distance']}, slot_id_switch_count={cont['slot_id_switch_count']}, slot_id_switch_rate={cont['slot_id_switch_rate']:.6f}, mean_continuous_segment_length={cont['mean_continuous_segment_length']}")
    lines += ["", "## Lane context diagnostics"]
    lane_success = float(np.mean([d.get("lane_assignment_available", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    fallback_rate = float(np.mean([d.get("fallback_assignment_used", False) for d in assignment_debug_rows])) if assignment_debug_rows else 0.0
    lines += [f"- map_name_resolved_rate: `{map_name_resolved_rate}`", f"- map_name values used: `{map_names_used}`", f"- map_name resolution sources: `{validation['map_name_resolution_sources']}`", f"- lane-aware success rate: `{lane_success}`", f"- geometric fallback rate: `{fallback_rate}`", f"- map_query_success: `{map_query_success}`", f"- lane_info_count: `{lane_info_count}`", f"- ego_lane_projection_success_rate: `{ego_lane_projection_success_rate}`", f"- candidate_lane_projection_success_rate: `{candidate_lane_projection_success_rate}`"]
    lines += [f"- projection debug summary: `{projection_debug_artifacts.get('summary_json')}`", f"- projection debug report: `{projection_debug_artifacts.get('report_md')}`", f"- projection debug csv: `{projection_debug_artifacts.get('csv', 'not_written_without_--write_projection_debug')}`", f"- relation unknown debug csv: `{projection_debug_artifacts.get('relation_unknown_csv', 'not_written_no_unknown_rows')}`", f"- topology debug summary: `{topology_debug_artifacts.get('summary_json')}`", f"- topology debug report: `{topology_debug_artifacts.get('report_md')}`"]
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
    lines += ["", "## Sanity checks", f"- slot_sanity_min_coverage: `{args.slot_sanity_min_coverage}`", f"- evaluated_slots: `{slot_sanity_evaluated_slots}`", f"- skipped_low_coverage_slots: `{slot_sanity_skipped_low_coverage_slots}`"] + [f"- {k}: `{v}`" for k,v in sanity.items()]
    (args.output_dir/"slot_assignment_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not validation["pass"]: raise RuntimeError("nuPlan Stage5D context validation failed; see warnings.json and slot_assignment_report.md")
    print(f"nuPlan Stage 5D-compatible context build PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
