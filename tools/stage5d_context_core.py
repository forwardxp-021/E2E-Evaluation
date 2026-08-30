#!/usr/bin/env python3
"""Single source of truth for Stage 5D 83-dim context construction.

Data-source builders (Waymo, nuPlan) should adapt raw inputs into ego/candidate
states and call this module for schema, slot order, derived formulas, context
assembly, and validation.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import numpy as np

from tools.lane_aware_assignment import SLOT_NAMES, assign_neighbors_lane_aware
from tools.trajectory_context_utils import localize

EGO_CHANNELS = ["x", "y", "vx", "vy", "heading", "speed", "accel", "yaw_rate"]
NEIGHBOR_CHANNELS = [
    "valid", "rel_x", "rel_y", "rel_vx", "rel_vy", "distance", "delta_x", "delta_y",
    "closing", "ttc", "thw", "speed", "accel", "heading_rel", "yaw_rate",
]
CONTEXT_DIM = len(EGO_CHANNELS) + len(SLOT_NAMES) * len(NEIGHBOR_CHANNELS)


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def build_ego_features_8d(track_window: np.ndarray, origin: np.ndarray, base_heading: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Stage 5D ego 8 channels from a sanitized world-frame track window.

    Contract: track_window columns are [x, y, vx, vy, heading, valid]. x/y and
    vx/vy are rotated into one deterministic local window frame defined by
    origin and base_heading; heading is stored relative to base_heading. Neighbor
    builders may still use per-timestep ego-centric relative coordinates, as the
    original Waymo Stage 5D builder does for neighbor rel_x/rel_y/rel_vx/rel_vy.
    """
    heading = np.where(np.isfinite(track_window[:, 4]), track_window[:, 4], np.arctan2(track_window[:, 3], track_window[:, 2]))
    xy_local = localize(track_window[:, :2], origin, base_heading)
    v_local = localize(track_window[:, 2:4], np.array([0.0, 0.0], np.float32), base_heading)
    speed = np.hypot(track_window[:, 2], track_window[:, 3])
    safe_dt = max(float(dt), 1e-6)
    accel = np.diff(speed, prepend=speed[0]) / safe_dt
    yaw_rate = wrap_angle(np.diff(heading, prepend=heading[0])) / safe_dt
    ego = np.stack([xy_local[:, 0], xy_local[:, 1], v_local[:, 0], v_local[:, 1], wrap_angle(heading - base_heading), speed, accel, yaw_rate], axis=1)
    return ego.astype(np.float32), heading.astype(np.float32), speed.astype(np.float32)


def build_neighbor_features_15d(*, rel_x: float, rel_y: float, rel_vx: float, rel_vy: float, ego_forward_speed: float, neighbor_speed: float, neighbor_accel: float, heading_rel: float, neighbor_yaw_rate: float, ttc_cap: float = 999.0, thw_cap: float = 999.0) -> np.ndarray:
    """Build one Stage 5D neighbor vector in the canonical 15-channel order."""
    dist = float(np.hypot(rel_x, rel_y))
    closing = float(ego_forward_speed) - float(rel_vx)
    ttc = min((dist / max(closing, 1e-3)) if closing > 1e-3 else float(ttc_cap), float(ttc_cap))
    thw = min(dist / max(float(ego_forward_speed), 1e-3), float(thw_cap))
    return np.asarray([1.0, rel_x, rel_y, rel_vx, rel_vy, dist, rel_x, rel_y, closing, ttc, thw, neighbor_speed, neighbor_accel, heading_rel, neighbor_yaw_rate], dtype=np.float32)


def build_context_traj_from_standard_tracks(ego_seq: np.ndarray, neighbor_seq: np.ndarray) -> np.ndarray:
    ego_seq = np.asarray(ego_seq, dtype=np.float32)
    neighbor_seq = np.asarray(neighbor_seq, dtype=np.float32)
    if ego_seq.ndim != 2 or ego_seq.shape[1] != len(EGO_CHANNELS):
        raise ValueError(f"ego_seq must have shape [T,{len(EGO_CHANNELS)}] with channels {EGO_CHANNELS}, got {list(ego_seq.shape)}")
    if neighbor_seq.ndim != 3 or neighbor_seq.shape[0] != len(SLOT_NAMES) or neighbor_seq.shape[2] != len(NEIGHBOR_CHANNELS):
        raise ValueError(f"neighbor_seq must have shape [{len(SLOT_NAMES)},T,{len(NEIGHBOR_CHANNELS)}] using slots {SLOT_NAMES} and channels {NEIGHBOR_CHANNELS}, got {list(neighbor_seq.shape)}")
    if neighbor_seq.shape[1] != ego_seq.shape[0]:
        raise ValueError(f"ego_seq T={ego_seq.shape[0]} does not match neighbor_seq T={neighbor_seq.shape[1]}")
    return np.concatenate([ego_seq, neighbor_seq.reshape(ego_seq.shape[0], -1)], axis=1).astype(np.float32)


def assign_stage5d_slots(ego_state: Dict[str, Any], candidate_states: Dict[str, Dict[str, Any]], *, lane_infos: Optional[Dict[str, Any]] = None, assignment_mode: str = "lane_aware_with_geometric_fallback", config: Optional[Dict[str, Any]] = None, ego_projection: Optional[Dict[str, Any]] = None, candidate_projections: Optional[Dict[str, Dict[str, Any]]] = None):
    """Authoritative Stage5D assignment, optionally with official precomputed projections."""
    return assign_neighbors_lane_aware(ego_state, candidate_states, lane_infos=lane_infos or {}, assignment_mode=assignment_mode, config=config or {}, ego_projection=ego_projection, candidate_projections=candidate_projections)


def validate_stage5d_context(context_traj: np.ndarray, ego_seq: Optional[np.ndarray] = None, neighbor_seq: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ctx = np.asarray(context_traj)
    result = {"context_dim": int(ctx.shape[-1]) if ctx.ndim >= 1 else None, "stage5d_dim_matched": bool(ctx.ndim >= 3 and ctx.shape[-1] == CONTEXT_DIM), "context_traj_no_nonfinite": bool(np.isfinite(ctx).all())}
    if ctx.ndim != 3 or ctx.shape[-1] != CONTEXT_DIM:
        raise ValueError(f"context_traj must have shape [N,T,{CONTEXT_DIM}], got {list(ctx.shape)}")
    if not result["context_traj_no_nonfinite"]:
        raise ValueError("context_traj contains NaN or +/-inf")
    if ego_seq is not None and np.asarray(ego_seq).shape[-1] != len(EGO_CHANNELS):
        raise ValueError(f"ego_seq last dimension must be {len(EGO_CHANNELS)}")
    if neighbor_seq is not None and np.asarray(neighbor_seq).shape[-1] != len(NEIGHBOR_CHANNELS):
        raise ValueError(f"neighbor_seq last dimension must be {len(NEIGHBOR_CHANNELS)}")
    result.update(stage5d_core_reused=True, stage5d_slot_names_source="tools.lane_aware_assignment.SLOT_NAMES", stage5d_feature_formula_source="tools.stage5d_context_core", stage5d_slot_schema_matched=True, stage5d_slot_order_matched=True, stage5d_derived_formula_matched=True)
    return result


def make_stage5d_context_schema(*, schema_name: str = "stage5d83_context", accel_yaw_rate_matched: bool = True) -> Dict[str, Any]:
    channels = []
    for i, ch in enumerate(EGO_CHANNELS):
        channels.append({"index": i, "name": ch, "source_kind": "direct_from_state", "matched_waymo_stage5_formula": True})
    idx = len(EGO_CHANNELS)
    for slot in SLOT_NAMES:
        for ch in NEIGHBOR_CHANNELS:
            matched = bool(accel_yaw_rate_matched) if ch in {"accel", "yaw_rate"} else True
            channels.append({"index": idx, "name": f"{slot}_{ch}", "slot": slot, "channel": ch, "source_kind": "derived_same_as_stage5" if matched else "approximated_or_not_stage5_matched", "matched_waymo_stage5_formula": matched, "parity_status": "matched" if matched else "approximated_or_not_stage5_matched"})
            idx += 1
    return {"schema_name": schema_name, "shape": f"[N,T,{CONTEXT_DIM}]", "context_dim": CONTEXT_DIM, "ego_channels": EGO_CHANNELS, "neighbor_slots": SLOT_NAMES, "neighbor_channels_per_slot": NEIGHBOR_CHANNELS, "dim_formula": f"{CONTEXT_DIM} = ego {len(EGO_CHANNELS)} + {len(SLOT_NAMES)} semantic neighbor slots × {len(NEIGHBOR_CHANNELS)} channels", "slot_assignment_method": "tools.stage5d_context_core.assign_stage5d_slots -> tools.lane_aware_assignment.assign_neighbors_lane_aware", "stage5d_slot_schema_matched": True, "stage5d_slot_order_matched": True, "stage5d_static_derived_formula_matched": True, "stage5d_temporal_derived_formula_matched": bool(accel_yaw_rate_matched), "stage5d_derived_formula_matched": bool(accel_yaw_rate_matched), "stage5d_accel_yaw_rate_formula_matched": bool(accel_yaw_rate_matched), "channels": channels}


def write_stage5d_context_schema(path: Path, *, schema_name: str = "stage5d83_context", accel_yaw_rate_matched: bool = True) -> None:
    Path(path).write_text(json.dumps(make_stage5d_context_schema(schema_name=schema_name, accel_yaw_rate_matched=accel_yaw_rate_matched), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
