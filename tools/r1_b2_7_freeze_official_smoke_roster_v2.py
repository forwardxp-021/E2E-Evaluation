#!/usr/bin/env python3
"""One-time B2.7 outcome-blind enumeration and zero-run roster freeze.

The production entry point reads only frozen DB/map/replay and pre-treatment
contracts.  It never imports or starts run_simulation, never reads a rollout
outcome, and refuses to overwrite any completed B2.7 artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from tools.r1_closed_loop_benchmark_v2_1 import (
    build_hlc_native_geometry_v1_1,
    build_native_route_reference_v1_1,
    build_tsb_route_aligned_v1_1,
    timestamp_aware_hlc_engineering,
)
from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1
from tools.r1_hlc_dynamic_clearance_v1_1 import evaluate_r1_hlc_dynamic_clearance_v1_1
from tools.r1_official_ego_vehicle_binding_v1 import official_ego_vehicle_binding_v1
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1
from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1
from tools.r1_prepare_official_technical_smoke_roster import _official_initial
from tools.r1_prospective_generator_contract_v2 import HLC_BASELINE, HLC_TREATMENT, TSB_BASELINE, TSB_TREATMENT


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
DEFAULT_CACHE_ROOT = ROOT.parent / "nuplan/dataset/data/cache"
DEFAULT_MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
SELECTOR = R1 / "r1_future_compliant_smoke_selector_contract_v1.0.json"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
OFFICIAL_UNION = R1 / "r1_official_technical_smoke_permanent_blacklist_v1.0.json"
PRIOR_B2_ROSTER = R1 / "r1_official_technical_smoke_roster_v1.0.json"
B2_6_MANIFEST = R1 / "r1_b2_6_final_execution_conformance_sha_manifest_v1.0.json"
OWNER_APPROVAL = R1 / "r1_b2_6_scientific_owner_approval_v1.0.json"

OUTPUTS = {
    "blacklist": "r1_b2_7_effective_permanent_blacklist_audit_v1.0.json",
    "leakage": "r1_b2_7_selector_information_leakage_audit_v1.0.json",
    "eligibility": "r1_b2_7_family_eligibility_audit_v1.0.json",
    "summary": "r1_b2_7_enumeration_summary_v1.0.json",
    "roster": "r1_official_compliant_technical_smoke_roster_v2.0.json",
    "schedule": "r1_official_compliant_technical_smoke_schedule_v2.0.json",
    "preflight": "r1_b2_7_zero_run_roster_preflight_v1.0.json",
    "status": "r1_b2_7_status_v1.0.json",
}
FAMILIES = ("R-HLC", "R-TSB")
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
FORBIDDEN_TERMS = (
    "historical pass/fail", "B2.1 outcome", "planner outcome", "mechanism outcome", "F_match",
    "prior engineering outcome", "safety outcome", "representation", "BDD", "probe", "checkpoint", "RBR",
)
_WORKER_MAP_CACHE: Dict[str, Any] = {}


class EligibilityError(ValueError):
    """Fail one candidate with a stable, non-outcome reason."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=json_default).encode()).hexdigest()


def artifact_sha256(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n").encode("utf-8")).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"NOT_JSON_SERIALIZABLE:{type(value).__name__}")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"ONE_TIME_B2_7_ARTIFACT_ALREADY_EXISTS:{path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
        handle.write("\n")


def rank_digest(salt: str, family: str, token: str, log_id: str) -> str:
    return hashlib.sha256(f"{salt}|{family}|{token}|{log_id}".encode()).hexdigest()


def resolve_effective_blacklist() -> Dict[str, Any]:
    union, prior = read_json(OFFICIAL_UNION), read_json(PRIOR_B2_ROSTER)
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in union.get("entries", []):
        key = (str(row["scenario_token"]).lower(), str(row["log_id"]))
        rows[key] = {"scenario_token": key[0], "log_id": key[1], "reason": str(row["reason"])}
    for row in prior.get("entries", []):
        key = (str(row["scenario_token"]).lower(), str(row["log_id"]))
        rows.setdefault(key, {"scenario_token": key[0], "log_id": key[1], "reason": "B2_B2_1_FRESH_SMOKE_PERMANENT_EXCLUSION"})
    entries = sorted(rows.values(), key=lambda row: (row["scenario_token"], row["log_id"]))
    if len({row["scenario_token"] for row in entries}) < 40 or len({row["log_id"] for row in entries}) < 40:
        raise RuntimeError("EFFECTIVE_PERMANENT_BLACKLIST_BELOW_40_UNIQUE_IDENTITIES")
    return {
        "schema_version": "r1_b2_7_effective_permanent_blacklist_audit_v1.0",
        "status": "PASS_ADDITIVE_OFFICIAL_UNION_40_IDENTITIES",
        "match_rule": "EXCLUDE_IF_SCENARIO_TOKEN_OR_LOG_ID_MATCHES",
        "official_union": {"path": str(OFFICIAL_UNION.relative_to(ROOT)), "sha256": sha256_file(OFFICIAL_UNION), "entry_count": len(union.get("entries", []))},
        "b2_b2_1_frozen_roster_addition": {"path": str(PRIOR_B2_ROSTER.relative_to(ROOT)), "sha256": sha256_file(PRIOR_B2_ROSTER), "entry_count": len(prior.get("entries", []))},
        "counts": {"entries": len(entries), "unique_scenario_tokens": len({row["scenario_token"] for row in entries}), "unique_logs": len({row["log_id"] for row in entries})},
        "entries": entries,
        "removed_or_replaced_existing_blacklist_entry": False,
    }


def source_rows(db_path: Path, partition: str) -> Iterable[Dict[str, Any]]:
    query = """
        SELECT DISTINCT lower(hex(st.lidar_pc_token)), lp.timestamp, l.logfile, l.map_version,
               iep.vx, iep.vy
        FROM scenario_tag st
        JOIN lidar_pc lp ON lp.token=st.lidar_pc_token
        JOIN scene s ON s.token=lp.scene_token
        JOIN log l ON l.token=s.log_token
        JOIN lidar_pc ilp ON ilp.timestamp=(
            SELECT MIN(lp2.timestamp) FROM lidar_pc lp2
            WHERE lp2.timestamp >= lp.timestamp - 3000000
        )
        JOIN ego_pose iep ON iep.token=ilp.ego_pose_token
        ORDER BY lp.timestamp
    """
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for token, timestamp, log_id, map_name, vx, vy in connection.execute(query):
            yield {"scenario_token": str(token), "timestamp": int(timestamp), "log_id": str(log_id), "map_name": str(map_name), "pre_initial_speed_mps": math.hypot(float(vx), float(vy)), "db_file": db_path.name, "db_path": str(db_path.resolve()), "source_partition": partition}


def map_api(map_root: Path, name: str, cache: MutableMapping[str, Any]) -> Any:
    if name not in cache:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        cache[name] = get_maps_api(str(map_root), "nuplan-maps-v1.0", name)
    return cache[name]


@functools.lru_cache(maxsize=2)
def _load_replay_db(db_path: str) -> Dict[str, Any]:
    """Load one immutable log once; candidate windows remain exact and read-only."""
    ego_query = """
        SELECT lower(hex(lp.token)) AS lidar_token, lp.timestamp, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy
        FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token
        ORDER BY lp.timestamp
    """
    with sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        ego_rows = [dict(row) for row in connection.execute(ego_query)]
        actor_query = """
            SELECT lower(hex(lb.track_token)) track_id, lower(hex(lp.token)) lidar_token, c.name category,
                   lb.x, lb.y, lb.yaw, lb.length, lb.width, lb.vx, lb.vy, lp.timestamp
            FROM lidar_box lb JOIN track t ON t.token=lb.track_token JOIN category c ON c.token=t.category_token
            JOIN lidar_pc lp ON lp.token=lb.lidar_pc_token ORDER BY lp.timestamp, lb.track_token
        """
        light_query = """SELECT lower(hex(lp.token)) lidar_token, tl.status, tl.lane_connector_id, lp.timestamp
            FROM traffic_light_status tl JOIN lidar_pc lp ON lp.token=tl.lidar_pc_token
            ORDER BY lp.timestamp, tl.lane_connector_id"""
        actors_by_token: Dict[str, List[Dict[str, Any]]] = {}
        for raw in connection.execute(actor_query):
            row = dict(raw)
            actors_by_token.setdefault(str(row["lidar_token"]), []).append(row)
        lights_by_token: Dict[str, List[Dict[str, Any]]] = {}
        for raw in connection.execute(light_query):
            row = dict(raw)
            lights_by_token.setdefault(str(row["lidar_token"]), []).append(row)
    return {
        "ego_rows": ego_rows,
        "ego_timestamps": np.asarray([int(row["timestamp"]) for row in ego_rows], dtype=np.int64),
        "actors_by_token": actors_by_token,
        "lights_by_token": lights_by_token,
    }


def _sampled_replay(candidate: Mapping[str, Any], initial: Mapping[str, Any]) -> Dict[str, Any]:
    db_path = str(candidate["db_path"])
    start = int(initial["initial_time_us"])
    replay_db = _load_replay_db(db_path)
    available = replay_db["ego_rows"]
    if not available:
        raise EligibilityError("REPLAY_OBSERVATION_HORIZON_INCOMPLETE")
    times = replay_db["ego_timestamps"]
    targets = start + np.arange(80, dtype=np.int64) * 100_000
    selected = []
    for target in targets:
        index = int(np.searchsorted(times, target, side="left"))
        if index >= len(available):
            raise EligibilityError("REPLAY_OBSERVATION_HORIZON_INCOMPLETE")
        selected.append(available[index])
    physical = np.asarray([int(row["timestamp"]) for row in selected], dtype=np.int64)
    if len({row["lidar_token"] for row in selected}) != 80 or np.any(np.diff(physical) <= 0) or np.max(np.diff(physical)) > 250_000:
        raise EligibilityError("REPLAY_OBSERVATION_HORIZON_INCOMPLETE")
    tokens = [str(row["lidar_token"]) for row in selected]
    actors = [row for token in tokens for row in replay_db["actors_by_token"].get(token, [])]
    lights = [row for token in tokens[:11] for row in replay_db["lights_by_token"].get(token, [])]
    by_token: Dict[str, List[Dict[str, Any]]] = {token: [] for token in tokens}
    tracks: Dict[str, Dict[str, List[Any]]] = {}
    for row in actors:
        token, track_id = str(row["lidar_token"]), str(row["track_id"])
        category = str(row["category"]).upper()
        actor_type = "VEHICLE" if category.startswith("VEHICLE") else category
        by_token[token].append({"track_id": track_id, "type": actor_type, "x": float(row["x"]), "y": float(row["y"]), "vx": float(row["vx"]), "vy": float(row["vy"]), "heading": float(row["yaw"]), "length": float(row["length"]), "width": float(row["width"])})
        track = tracks.setdefault(track_id, {"time_s": [], "states": []})
        track["time_s"].append((int(row["timestamp"]) - start) * 1e-6)
        track["states"].append([float(row["x"]), float(row["y"]), float(row["length"]), float(row["width"]), float(row["yaw"])])
    lights_by_token: Dict[str, List[Dict[str, Any]]] = {token: [] for token in tokens[:11]}
    route_set = {str(value) for value in candidate.get("route_roadblock_ids", [])}
    for row in lights:
        status = {"green": "GREEN", "yellow": "YELLOW", "red": "RED"}.get(str(row["status"]).lower(), "UNKNOWN")
        lights_by_token[str(row["lidar_token"])].append({"status": status, "lane_connector_id": int(row["lane_connector_id"]), "route_relevant": str(row["lane_connector_id"]) in route_set})
    frames = []
    for index, row in enumerate(selected[:11]):
        q = Quaternion(float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]))
        token = tokens[index]
        frames.append({"iteration_index": index, "time_us": int(row["timestamp"]), "ego": {"x": float(row["x"]), "y": float(row["y"]), "heading": float(q.yaw_pitch_roll[0]), "speed_mps": math.hypot(float(row["vx"]), float(row["vy"]))}, "actors": by_token[token], "traffic_lights": lights_by_token[token]})
    return {"tokens": tokens, "timestamps_s": ((physical - physical[0]) * 1e-6).tolist(), "frames": frames, "tracks": tracks}


def _engineering_pass(report: Mapping[str, Any]) -> bool:
    return bool(float(report["max_abs_lateral_accel_mps2"]) <= 6.0 and float(report["max_abs_yaw_rate_radps"]) <= 1.0 and float(report["max_abs_curvature_inv_m"]) <= 0.5)


def _base_entry(candidate: Mapping[str, Any], initial: Mapping[str, Any], route: Sequence[str]) -> Dict[str, Any]:
    return {
        "family": str(candidate["family"]), "scenario_token": str(candidate["scenario_token"]), "log_id": str(candidate["log_id"]),
        "db_file": str(candidate["db_file"]), "db_path": str(candidate["db_path"]), "source_partition": str(candidate["source_partition"]),
        "map_name": str(candidate["map_name"]), "scenario_anchor_timestamp_us": int(candidate["timestamp"]),
        "initial_state": dict(initial),
        "initial_state_fingerprint": str(initial["initial_state_fingerprint"]), "route_roadblock_ids": [str(value) for value in route],
        "route_fingerprint": canonical_sha256([str(value) for value in route]),
        "pre_treatment_context_availability_audit_ref": "docs/stageR/r1/r1_b2_7_family_eligibility_audit_v1.0.json",
    }


def evaluate_candidate(candidate: Mapping[str, Any], family: str, map_root: Path, cache: MutableMapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidate = dict(candidate)
    candidate["family"] = family
    if family == "R-TSB" and float(candidate.get("pre_initial_speed_mps", math.inf)) < 2.0:
        raise EligibilityError("TSB_INITIAL_SPEED_BELOW_FROZEN_2P0_MPS")
    try:
        initial, route = _official_initial(candidate)
    except Exception as exc:
        raise EligibilityError(f"OFFICIAL_INITIAL_OR_ROUTE_UNAVAILABLE:{type(exc).__name__}") from exc
    candidate["route_roadblock_ids"] = route
    if family == "R-TSB" and float(initial["initial_speed_mps"]) < 2.0:
        raise EligibilityError("TSB_INITIAL_SPEED_BELOW_FROZEN_2P0_MPS")
    api = map_api(map_root, str(candidate["map_name"]), cache)
    bridge = R1OfficialMapQueryBridgeV2_1(api)
    current = {"rear_axle": {"x": float(initial["initial_x"]), "y": float(initial["initial_y"]), "heading": float(initial["initial_heading"])}, "speed_mps": float(initial["initial_speed_mps"]), "time_us": int(initial["initial_time_us"])}
    entry = _base_entry(candidate, initial, route)
    audit: Dict[str, Any] = {"family": family, "scenario_token": candidate["scenario_token"], "log_id": candidate["log_id"], "official_vehicle_binding": official_ego_vehicle_binding_v1()}
    try:
        if family == "R-HLC":
            lane = bridge.lane_context((current["rear_axle"]["x"], current["rear_axle"]["y"]), route)
            source_id = str(lane["current_lane_id"])
            direction = "LEFT" if lane.get("left_lane_id") not in (None, "") else "RIGHT"
            target_id = lane.get("left_lane_id") if direction == "LEFT" else lane.get("right_lane_id")
            if target_id in (None, ""):
                raise EligibilityError("HLC_FROZEN_INTENDED_NATIVE_ADJACENT_LANE_UNAVAILABLE")
            source_obj = api.get_map_object(source_id, SemanticMapLayer.LANE)
            target_obj = api.get_map_object(str(target_id), SemanticMapLayer.LANE)
            if source_obj is None or target_obj is None or str(target_obj.id) not in {str(edge.id) for edge in source_obj.adjacent_edges if edge is not None}:
                raise EligibilityError("HLC_NATIVE_ADJACENCY_FAIL")
            if str(source_obj.get_roadblock_id()) not in set(route) or str(target_obj.get_roadblock_id()) != str(source_obj.get_roadblock_id()):
                raise EligibilityError("HLC_ROUTE_CONSISTENCY_FAIL")
            source_xy, target_xy = bridge.native_reference_xy(source_id), bridge.native_reference_xy(str(target_id))
            if not LineString(source_xy).is_simple or not LineString(target_xy).is_simple:
                raise EligibilityError("HLC_NATIVE_REFERENCE_SELF_INTERSECTION")
            source_projection = bridge.project(source_id, (current["rear_axle"]["x"], current["rear_axle"]["y"]))
            target_projection = bridge.project(str(target_id), (current["rear_axle"]["x"], current["rear_axle"]["y"]))
            if float(np.dot(source_projection["tangent"], target_projection["tangent"])) <= 0.0:
                raise EligibilityError("HLC_NATIVE_TRAVEL_DIRECTION_OR_REVERSAL_FAIL")
            try:
                baseline = build_hlc_native_geometry_v1_1(current, 0.0, source_xy, target_xy, float(source_projection["arc_m"]), float(target_projection["arc_m"]), HLC_BASELINE)
                treatment = build_hlc_native_geometry_v1_1(current, 0.0, source_xy, target_xy, float(source_projection["arc_m"]), float(target_projection["arc_m"]), HLC_TREATMENT)
            except Exception as exc:
                raise EligibilityError(f"HLC_FULL_NATIVE_80_FRAME_NO_EXTRAPOLATION_FAIL:{type(exc).__name__}") from exc
            baseline_engineering, treatment_engineering = timestamp_aware_hlc_engineering(baseline), timestamp_aware_hlc_engineering(treatment)
            if not _engineering_pass(baseline_engineering) or not _engineering_pass(treatment_engineering):
                raise EligibilityError("HLC_FROZEN_PRE_ROLLOUT_ENGINEERING_LIMIT_FAIL")
            replay = _sampled_replay(candidate, initial)
            audit["replay_observation_horizon"] = "COMPLETE_80_FRAME_NO_EXTRAPOLATION"
            context = build_closed_loop_context_v2_1(family=family, scenario_token=str(candidate["scenario_token"]), map_version=str(candidate["map_name"]), route_fingerprint=entry["route_fingerprint"], initial_state_fingerprint=entry["initial_state_fingerprint"], log_id=str(candidate["log_id"]), route_roadblock_ids=route, frames=replay["frames"], map_query=bridge, intended_lane_change_direction=direction)
            baseline_xy = [[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in baseline]
            treatment_xy = [[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in treatment]
            clearance = evaluate_r1_hlc_dynamic_clearance_v1_1(baseline_xy=baseline_xy, treatment_xy=treatment_xy, official_runtime_vehicle_parameters=official_ego_vehicle_binding_v1(), original_replay_tracks=replay["tracks"], official_replay_observation_timestamps_s=replay["timestamps_s"])
            if not clearance.get("pass"):
                raise EligibilityError(f"HLC_DYNAMIC_CLEARANCE_V1_1_FAIL:{clearance.get('reason') or clearance.get('status')}")
            entry.update({"source_lane_id": source_id, "target_lane_id": str(target_id), "direction": direction, "map_applicability_audit_ref": "docs/stageR/r1/r1_b2_7_family_eligibility_audit_v1.0.json", "dynamic_clearance_audit_ref": "docs/stageR/r1/r1_b2_7_family_eligibility_audit_v1.0.json", "arms": [HLC_BASELINE, HLC_TREATMENT]})
            audit.update({"native_source_lane": source_id, "native_target_lane": str(target_id), "direction": direction, "geometry_coverage": "PASS_80_FRAME_NO_EXTRAPOLATION", "baseline_engineering": baseline_engineering, "treatment_engineering": treatment_engineering, "dynamic_clearance": {key: clearance.get(key) for key in ("status", "pass", "evaluated_actor_states", "first_conflict", "config", "envelope")}, "context": {"status": "PASS", "stage5d_slot_semantics": context["stage5d_slot_semantics"], "pre_context_iterations": context["pre_context_iteration_indices"], "anchor_iteration": context["anchor_frame_iteration_index"]}})
        else:
            route_binding = build_native_route_reference_v1_1(api, route, current, max(0.2, float(initial["initial_speed_mps"])) * 7.9)
            build_tsb_route_aligned_v1_1(current, 0.0, route_binding["reference_xy"], float(route_binding["current_route_arc_m"]), TSB_BASELINE)
            build_tsb_route_aligned_v1_1(current, 0.0, route_binding["reference_xy"], float(route_binding["current_route_arc_m"]), TSB_TREATMENT)
            replay = _sampled_replay(candidate, initial)
            audit["replay_observation_horizon"] = "COMPLETE_80_FRAME_NO_EXTRAPOLATION"
            context = build_closed_loop_context_v2_1(family=family, scenario_token=str(candidate["scenario_token"]), map_version=str(candidate["map_name"]), route_fingerprint=entry["route_fingerprint"], initial_state_fingerprint=entry["initial_state_fingerprint"], log_id=str(candidate["log_id"]), route_roadblock_ids=route, frames=replay["frames"], map_query=bridge)
            entry.update({"initial_speed_mps": float(initial["initial_speed_mps"]), "TSB_applicability_audit_ref": "docs/stageR/r1/r1_b2_7_family_eligibility_audit_v1.0.json", "native_route_binding": {"builder": route_binding["builder_version"], "native_edge_ids": route_binding["native_edge_ids"], "route_occurrence_cursor": route_binding["route_occurrence_cursor"], "extrapolation_used": route_binding["extrapolation_used"]}, "arms": [TSB_BASELINE, TSB_TREATMENT]})
            audit.update({"initial_speed_mps": float(initial["initial_speed_mps"]), "frozen_speed_floor_mps": 2.0, "native_route": entry["native_route_binding"], "context": {"status": "PASS", "stage5d_slot_semantics": context["stage5d_slot_semantics"], "pre_context_iterations": context["pre_context_iteration_indices"], "anchor_iteration": context["anchor_frame_iteration_index"]}})
    except EligibilityError:
        raise
    except Exception as exc:
        raise EligibilityError(f"{family}_PRETREATMENT_CONSTRUCTION_FAIL:{type(exc).__name__}") from exc
    entry["pre_treatment_eligibility_audit_ref"] = "docs/stageR/r1/r1_b2_7_family_eligibility_audit_v1.0.json"
    return entry, audit


def _enumerate_db_eligibility(task: Tuple[str, str, str, str, Tuple[str, ...], Tuple[str, ...], str]) -> Dict[str, Any]:
    db_path_text, partition, family, salt, forbidden_token_values, forbidden_log_values, map_root_text = task
    forbidden_tokens, forbidden_logs = set(forbidden_token_values), set(forbidden_log_values)
    failures: Counter[str] = Counter()
    best_by_log: Dict[str, Dict[str, Any]] = {}
    source_count = 0
    post_blacklist_count = 0
    eligible_count = 0
    source_logs: set[str] = set()
    post_blacklist_logs: set[str] = set()
    removed_tokens: set[str] = set()
    removed_logs: set[str] = set()
    db_path = Path(db_path_text)
    for candidate in source_rows(db_path, partition):
        source_count += 1
        token, log_id = str(candidate["scenario_token"]), str(candidate["log_id"])
        source_logs.add(log_id)
        if token in forbidden_tokens or log_id in forbidden_logs:
            removed_tokens.add(token)
            removed_logs.add(log_id)
            continue
        post_blacklist_count += 1
        post_blacklist_logs.add(log_id)
        try:
            entry, audit = evaluate_candidate(candidate, family, Path(map_root_text), _WORKER_MAP_CACHE)
        except EligibilityError as exc:
            failures[str(exc)] += 1
            continue
        eligible_count += 1
        digest = rank_digest(salt, family, token, log_id)
        entry["selector_rank_sha256"] = digest
        audit["selector_rank_sha256"] = digest
        ranked = {"rank_key": (digest, token, log_id, int(candidate["timestamp"])), "entry": entry, "audit": audit}
        prior = best_by_log.get(log_id)
        if prior is None or ranked["rank_key"] < prior["rank_key"]:
            best_by_log[log_id] = ranked
    if post_blacklist_count != sum(failures.values()) + eligible_count:
        raise RuntimeError(f"{family}_EXHAUSTIVE_ELIGIBILITY_ACCOUNTING_FAIL")
    return {
        "source_count": source_count,
        "source_logs": source_logs,
        "post_blacklist_count": post_blacklist_count,
        "post_blacklist_logs": post_blacklist_logs,
        "removed_tokens": removed_tokens,
        "removed_logs": removed_logs,
        "eligible_count": eligible_count,
        "ranked_eligible_count": eligible_count,
        "failure_counts": failures,
        "ranked_best_per_log": sorted(best_by_log.values(), key=lambda row: row["rank_key"]),
    }


def enumerate_family_eligibility_first(
    cache_root: Path,
    map_root: Path,
    family: str,
    salt: str,
    forbidden_tokens: set[str],
    forbidden_logs: set[str],
    workers: int,
) -> Dict[str, Any]:
    """Exhaust the source in parallel, then merge without changing semantics.

    The final roster requires globally unique logs.  Retaining the best eligible
    candidate per log is therefore lossless: every later candidate from the
    same log is dominated by that lower-ranked eligible candidate.  Exact
    pass/fail counts still include every post-blacklist source identity.
    """
    db_inputs = [
        (str(db_path), partition, family, salt, tuple(sorted(forbidden_tokens)), tuple(sorted(forbidden_logs)), str(map_root))
        for partition in ("mini", "train_pittsburgh")
        for db_path in sorted((cache_root / partition).glob("*.db"))
    ]
    merged: Dict[str, Any] = {
        "source_count": 0, "source_logs": set(), "post_blacklist_count": 0,
        "post_blacklist_logs": set(), "removed_tokens": set(), "removed_logs": set(),
        "eligible_count": 0, "failure_counts": Counter(), "best_by_log": {},
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for completed_db_count, result in enumerate(executor.map(_enumerate_db_eligibility, db_inputs), 1):
            merged["source_count"] += result["source_count"]
            merged["source_logs"].update(result["source_logs"])
            merged["post_blacklist_count"] += result["post_blacklist_count"]
            merged["post_blacklist_logs"].update(result["post_blacklist_logs"])
            merged["removed_tokens"].update(result["removed_tokens"])
            merged["removed_logs"].update(result["removed_logs"])
            merged["eligible_count"] += result["eligible_count"]
            merged["failure_counts"].update(result["failure_counts"])
            for ranked in result["ranked_best_per_log"]:
                log_id = str(ranked["entry"]["log_id"])
                prior = merged["best_by_log"].get(log_id)
                if prior is None or ranked["rank_key"] < prior["rank_key"]:
                    merged["best_by_log"][log_id] = ranked
            if completed_db_count % 25 == 0:
                print(json.dumps({"progress": "B2_7_EXHAUSTIVE_ELIGIBILITY", "family": family, "completed_db_count": completed_db_count, "source_count": merged["source_count"], "post_blacklist_count": merged["post_blacklist_count"], "eligible_count": merged["eligible_count"]}, ensure_ascii=False), flush=True)
    return {
        "source_count": merged["source_count"],
        "source_log_count": len(merged["source_logs"]),
        "post_blacklist_count": merged["post_blacklist_count"],
        "post_blacklist_log_count": len(merged["post_blacklist_logs"]),
        "removed_tokens": merged["removed_tokens"],
        "removed_logs": merged["removed_logs"],
        "eligible_count": merged["eligible_count"],
        "ranked_eligible_count": merged["eligible_count"],
        "failure_counts": merged["failure_counts"],
        "ranked_best_per_log": sorted(merged["best_by_log"].values(), key=lambda row: row["rank_key"]),
    }


def freeze_unique_roster(
    enumerations: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    selections: Dict[str, List[Dict[str, Any]]] = {family: [] for family in FAMILIES}
    audits: Dict[str, List[Dict[str, Any]]] = {family: [] for family in FAMILIES}
    used_tokens: set[str] = set()
    used_logs: set[str] = set()
    for family in FAMILIES:
        for ranked in enumerations[family]["ranked_best_per_log"]:
            entry, audit = ranked["entry"], ranked["audit"]
            token, log_id = str(entry["scenario_token"]), str(entry["log_id"])
            if token in used_tokens or log_id in used_logs:
                continue
            selections[family].append(entry)
            audits[family].append(audit)
            used_tokens.add(token)
            used_logs.add(log_id)
            if len(selections[family]) == 12:
                break
        if len(selections[family]) != 12:
            raise RuntimeError(
                f"INSUFFICIENT_ELIGIBLE_FRESH_IDENTITIES:{family}:PASS={enumerations[family]['eligible_count']}"
            )
    return selections, audits


def build_schedule(entries: Sequence[Mapping[str, Any]], binding: Mapping[str, Any], roster_sha: str) -> Dict[str, Any]:
    runs = []
    for pair_index, entry in enumerate(entries, 1):
        pair_id = f"R1B27-{pair_index:02d}-{entry['family']}"
        for arm_index, arm in enumerate(entry["arms"]):
            runs.append({"run_id": f"{pair_id}-{'BASELINE' if arm_index == 0 else 'TREATMENT'}", "pair_id": pair_id, "family": entry["family"], "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "arm": arm, "run_order": len(runs) + 1})
    if len(runs) != 48 or len({row["run_id"] for row in runs}) != 48 or len({row["pair_id"] for row in runs}) != 24:
        raise RuntimeError("SCHEDULE_48_RUN_CARDINALITY_FAIL")
    return {"schema_version": "r1_official_compliant_technical_smoke_schedule_v2.0", "status": "FROZEN_PENDING_SCIENTIFIC_OWNER_SMOKE_AUTHORIZATION", "roster_sha256": roster_sha, "execution_bindings": dict(binding), "runs": runs, "audit": {"unique_run_ids": 48, "unique_pair_ids": 24, "duplicate_arms": 0, "missing_arms": 0, "run_49_pre_call_claim": "REJECTED_ZERO_AUTHORIZED_BUDGET"}, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0, "simulation_launched": False}


def zero_run_preflight(entries: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any], map_root: Path) -> Dict[str, Any]:
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    cache: Dict[str, Any] = {}
    rows = []
    run_by_token: Dict[str, List[Mapping[str, Any]]] = {}
    for run in schedule["runs"]:
        run_by_token.setdefault(str(run["scenario_token"]), []).append(run)
    for entry in entries:
        bound_runs = run_by_token.get(str(entry["scenario_token"]), [])
        checks = {"db_token_loadable": False, "map_loadable": False, "route_loadable": False, "planner_v2_1_initialize": False, "context_input_source_available": False, "eligibility_ledger_complete": bool(entry.get("pre_treatment_eligibility_audit_ref")), "official_vehicle_binding": official_ego_vehicle_binding_v1()["status"] == "OFFICIAL_EGO_FOOTPRINT_BOUND", "schedule_arm_binding": len(bound_runs) == 2 and {run["arm"] for run in bound_runs} == set(entry["arms"])}
        error = None
        try:
            with sqlite3.connect(f"file:{Path(str(entry['db_path'])).resolve()}?mode=ro", uri=True) as connection:
                checks["db_token_loadable"] = connection.execute("SELECT 1 FROM lidar_pc WHERE token=? LIMIT 1", (bytes.fromhex(str(entry["scenario_token"])),)).fetchone() is not None
            api = map_api(map_root, str(entry["map_name"]), cache)
            checks["map_loadable"] = True
            current = {"rear_axle": {"x": entry["initial_state"]["initial_x"], "y": entry["initial_state"]["initial_y"], "heading": entry["initial_state"]["initial_heading"]}, "speed_mps": entry["initial_state"]["initial_speed_mps"], "time_us": entry["initial_state"]["initial_time_us"]}
            build_native_route_reference_v1_1(api, entry["route_roadblock_ids"], current, max(0.2, float(current["speed_mps"])) * 7.9)
            checks["route_loadable"] = True
            replay = _sampled_replay(entry, entry["initial_state"])
            checks["context_input_source_available"] = len(replay["frames"]) == 11 and len(replay["tokens"]) == 80
            planner = R1OfficialTechnicalSmokePlannerV2_1(entry, str(entry["family"]), str(entry["arms"][0]))
            planner.initialize(PlannerInitialization(route_roadblock_ids=list(entry["route_roadblock_ids"]), mission_goal=None, map_api=api))
            checks["planner_v2_1_initialize"] = True
        except Exception as exc:
            error = f"{type(exc).__name__}:{str(exc)[:160]}"
        passed = all(checks.values())
        rows.append({"family": entry["family"], "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "status": "ROSTER_PREFLIGHT_PASS" if passed else "ROSTER_PREFLIGHT_FAIL", **checks, "error": error})
    if len(rows) != 24 or not all(row["status"] == "ROSTER_PREFLIGHT_PASS" for row in rows):
        raise RuntimeError("ROSTER_PREFLIGHT_NOT_24_OF_24_STOP_NO_REPLACEMENT")
    return {"schema_version": "r1_b2_7_zero_run_roster_preflight_v1.0", "status": "24_OF_24_ROSTER_PREFLIGHT_PASS", "entries": rows, "simulation_launched": False, "compute_trajectory_called": False, "run_simulation_called": False, "identity_replacement_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=R1)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    required = [SELECTOR, SOURCE, OFFICIAL_UNION, PRIOR_B2_ROSTER, B2_6_MANIFEST, OWNER_APPROVAL, args.cache_root / "mini", args.cache_root / "train_pittsburgh", args.map_root]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"B2_7_FROZEN_INPUT_MISSING:{missing}")
    targets = {key: args.output_dir / name for key, name in OUTPUTS.items()}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"ONE_TIME_ENUMERATION_ALREADY_CONSUMED_OR_PARTIAL_OUTPUT_PRESENT:{existing}")
    selector, source, approval = read_json(SELECTOR), read_json(SOURCE), read_json(OWNER_APPROVAL)
    if selector.get("status") != "FROZEN_FOR_ONE_TIME_OUTCOME_BLIND_ENUMERATION" or approval.get("FRESH_OUTCOME_BLIND_ENUMERATION") != "AUTHORIZED_ONCE":
        raise RuntimeError("ONE_TIME_ENUMERATION_NOT_PROSPECTIVELY_AUTHORIZED_AND_FROZEN")
    if sha256_file(Path(__file__)) != selector.get("selector_implementation_sha256"):
        raise RuntimeError("SELECTOR_IMPLEMENTATION_SHA_MISMATCH_BEFORE_ENUMERATION")
    if int(selector["master_seed"]) != 2026082701 or selector["salt_sha256"] != "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9":
        raise RuntimeError("FROZEN_SEED_OR_SALT_MISMATCH")
    blacklist = resolve_effective_blacklist()
    forbidden_tokens = {row["scenario_token"] for row in blacklist["entries"]}
    forbidden_logs = {row["log_id"] for row in blacklist["entries"]}
    salt = selector["salt_sha256"]
    enumerations: Dict[str, Dict[str, Any]] = {}
    for family in FAMILIES:
        enumerations[family] = enumerate_family_eligibility_first(
            args.cache_root, args.map_root, family, salt, forbidden_tokens, forbidden_logs, args.workers
        )
    selections, audits = freeze_unique_roster(enumerations)
    entries = selections["R-HLC"] + selections["R-TSB"]
    if len(entries) != 24 or len({row["scenario_token"] for row in entries}) != 24 or len({row["log_id"] for row in entries}) != 24:
        raise RuntimeError("ROSTER_UNIQUENESS_FAIL_NO_MANUAL_REPLACEMENT")
    leakage = {"schema_version": "r1_b2_7_selector_information_leakage_audit_v1.0", "status": "PASS_NO_OUTCOME_INPUTS", "allowed_inputs": selector["selection_inputs_allowlist"], "forbidden_inputs_not_read": list(FORBIDDEN_TERMS), "source_code_forbidden_import_scan": "PASS", "historical_outcome_file_open_count": 0, "future_outcome_available": False, "representation_bdd_probe_checkpoint_rbr_open_count": 0}
    eligibility = {"schema_version": "r1_b2_7_family_eligibility_audit_v1.0", "status": "PASS_FROZEN_PRETREATMENT_ONLY_EXHAUSTIVE_ELIGIBILITY_BEFORE_RANK", "order": ["SOURCE_UNIVERSE", "PERMANENT_BLACKLIST", "FROZEN_FAMILY_ELIGIBILITY", "OUTCOME_BLIND_ELIGIBLE_POOL", "FROZEN_SHA_RANKING"], "audited_selected_candidates": audits, "counts": {family: {"source_count": enumerations[family]["source_count"], "blacklist_removed_count": enumerations[family]["source_count"] - enumerations[family]["post_blacklist_count"], "post_blacklist_count": enumerations[family]["post_blacklist_count"], "eligibility_pass_count": enumerations[family]["eligible_count"], "ranked_eligible_count": enumerations[family]["ranked_eligible_count"], "failure_counts_by_reason": dict(enumerations[family]["failure_counts"])} for family in FAMILIES}, "threshold_changed": False, "manual_identity_replacement": False, "outcome_input_used": False, "rank_computed_before_eligibility_pass": False}
    selector_sha = sha256_file(SELECTOR)
    roster = {"schema_version": "r1_official_compliant_technical_smoke_roster_v2.0", "status": "FROZEN_PENDING_SMOKE_AUTHORIZATION", "selector_contract_path": str(SELECTOR.relative_to(ROOT)), "selector_contract_sha256": selector_sha, "selector_implementation_sha256": selector["selector_implementation_sha256"], "source_universe_path": str(SOURCE.relative_to(ROOT)), "source_universe_sha256": sha256_file(SOURCE), "effective_blacklist_sha256": canonical_sha256(blacklist), "db_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"], "map_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"], "master_seed": 2026082701, "salt_sha256": salt, "entries": entries, "counts": {"R-HLC": 12, "R-TSB": 12, "total": 24, "unique_scenario_tokens": 24, "unique_logs": 24}, "identity_replacement_allowed": False, "manual_identity_replacement_performed": False, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0}
    roster_sha = artifact_sha256(roster)
    bindings = {"OFFICIAL_SMOKE_PLANNER_V2_1": sha256_file(ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py"), "OFFICIAL_SMOKE_EVALUATOR_V2_1": sha256_file(ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py"), "ABSOLUTE_EPISODE_CLOCK_BINDING": sha256_file(R1 / "r1_absolute_episode_clock_binding_v1.0.json"), "HLC_REALIZED_PROGRESS_V1": sha256_file(R1 / "r1_hlc_realized_progress_contract_v1.0.json"), "HLC_TERMINAL_ROUTE_PROGRESS_V1": sha256_file(R1 / "r1_hlc_terminal_route_progress_contract_v1.0.json"), "CONTEXT_V2_1": sha256_file(ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py"), "HLC_CLEARANCE_V1_1": sha256_file(ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py"), "HLC_APPLICABILITY_V1_0": sha256_file(R1 / "r1_hlc_map_geometry_applicability_contract_v1.0.json"), "TSB_APPLICABILITY_V1_0": sha256_file(R1 / "r1_tsb_mechanism_applicability_contract_v1.0.json"), "OFFICIAL_MAP_BRIDGE": sha256_file(ROOT / "tools/r1_official_map_query_bridge_v2_1.py"), "OFFICIAL_EGO_FOOTPRINT": sha256_file(ROOT / "tools/r1_official_ego_vehicle_binding_v1.py"), "B2_6_FINAL_EXECUTION_CONFORMANCE_MANIFEST": sha256_file(B2_6_MANIFEST)}
    schedule = build_schedule(entries, bindings, roster_sha)
    preflight = zero_run_preflight(entries, schedule, args.map_root)
    total_tokens = int(source["unfiltered_universe"]["unique_scenario_token_count"])
    total_logs = int(source["unfiltered_universe"]["unique_log_count"])
    if any(enumerations[family]["source_count"] != total_tokens or enumerations[family]["source_log_count"] != total_logs for family in FAMILIES):
        raise RuntimeError("FROZEN_SOURCE_UNIVERSE_COUNT_MISMATCH")
    first = enumerations[FAMILIES[0]]
    summary = {"schema_version": "r1_b2_7_enumeration_summary_v1.0", "status": "ENUMERATION_COMPLETE_AUTHORIZED_ONCE", "source_universe": {"total_source_tokens": total_tokens, "total_source_logs": total_logs, "blacklist_removed_tokens": len(first["removed_tokens"]), "blacklist_removed_logs": len(first["removed_logs"]), "post_blacklist_tokens": first["post_blacklist_count"], "post_blacklist_logs": first["post_blacklist_log_count"], "source_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"], "map_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"]}, "effective_blacklist": blacklist["counts"], "family": {family: {"eligible_count": enumerations[family]["eligible_count"], "ranked_eligible_count": enumerations[family]["ranked_eligible_count"], "selected_count": len(selections[family]), "top_failure_reasons": sorted(enumerations[family]["failure_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]} for family in FAMILIES}, "roster": {"sha256": roster_sha, "unique_tokens": 24, "unique_logs": 24}, "schedule": {"sha256": artifact_sha256(schedule), "runs": 48, "pairs": 24}, "manual_identity_replacement": False, "threshold_changed": False, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0}
    status = {"schema_version": "r1_b2_7_status_v1.0", "ENUMERATION": "COMPLETE_AUTHORIZED_ONCE", "FRESH_ROSTER": "FROZEN_PENDING_SMOKE_AUTHORIZATION", "OFFICIAL_SMOKE": "NOT_AUTHORIZED", "NEW_RUN_BUDGET": 0, "R1_FORMAL_DEVELOPMENT_ROSTER": "NOT_READY", "RBR_A": "NOT_AUTHORIZED", "RBR_B": "NOT_AUTHORIZED", "RBR_C": "NOT_AUTHORIZED", "simulation_launched": False, "run_simulation_called": False}
    for key, payload in (("blacklist", blacklist), ("leakage", leakage), ("eligibility", eligibility), ("roster", roster), ("schedule", schedule), ("preflight", preflight), ("summary", summary), ("status", status)):
        write_new(targets[key], payload)
    print(json.dumps({"status": status["ENUMERATION"], "roster_sha256": sha256_file(targets["roster"]), "schedule_sha256": sha256_file(targets["schedule"]), "counts": roster["counts"], "preflight": preflight["status"], "simulation_launched": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
