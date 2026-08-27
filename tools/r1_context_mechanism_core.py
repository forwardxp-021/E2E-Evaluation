#!/usr/bin/env python3
"""Frozen R1 context canonicalization and mechanism calculators.

This module deliberately has no representation, BDD, probe, checkpoint, RBR,
or planner-runtime dependency.  It implements only the v1.0 frozen context and
measurement contracts used by the R1 technical smoke.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


DT_SECONDS = 0.1
PRE_CONTEXT_FRAMES = 10
NOT_APPLICABLE = "NOT_APPLICABLE_BY_FROZEN_ABSENCE_STATE"
SLOT_NAMES = ("front", "left_front", "left_rear", "right_front", "right_rear")
TSB_HAZARD_PRIORITY = (
    "ROUTE_SIGNAL_RED_OR_YELLOW",
    "STATIC_STOP_CONTROL_AHEAD",
    "OBSERVED_SLOW_LEAD",
    "NONE_OBSERVED",
)


def canonical_json_sha256(value: Any) -> str:
    """Return the deterministic SHA-256 used by frozen R1 records."""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _as_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def median3(values: Sequence[float]) -> np.ndarray:
    """Edge-replicated three-sample median; input must be finite."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("median3 requires a finite one-dimensional sequence with at least three samples")
    padded = np.pad(x, (1, 1), mode="edge")
    return np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)


def _seconds_to_samples(seconds: float) -> int:
    return int(round(float(seconds) / DT_SECONDS))


def _run_ranges(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(mask):
        if bool(item) and start is None:
            start = index
        if start is not None and (not bool(item) or index == len(mask) - 1):
            end = index if bool(item) and index == len(mask) - 1 else index - 1
            result.append((start, end))
            start = None
    return result


def _require_pre_context_frames(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != PRE_CONTEXT_FRAMES:
        raise ValueError("context payload must contain exactly 10 pre-context frames")
    expected = [round(float(payload.get("t_anchor_s", 1.0)) - 1.0 + i * DT_SECONDS, 6) for i in range(10)]
    actual = [_round(_as_float(frame.get("time_s"), f"frames[{i}].time_s")) for i, frame in enumerate(frames)]
    if actual != expected:
        raise ValueError(f"pre-context timestamps must be {expected}, got {actual}")
    return frames


def _slot_summary(frames: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    bits: List[str] = []
    audit: Dict[str, Any] = {}
    for name in SLOT_NAMES:
        entries = [dict(frame.get("slots", {})).get(name) for frame in frames]
        valid = [entry for entry in entries if isinstance(entry, Mapping) and bool(entry.get("valid", False))]
        ids = sorted({str(entry.get("track_id")) for entry in valid if entry.get("track_id") not in (None, "")})
        bit = len(valid) >= 8 and len(ids) == 1
        bits.append("1" if bit else "0")
        audit[name] = {"valid_frame_count": len(valid), "track_ids": ids, "canonical_present": bit}
    return "".join(bits), audit


def _canonical_gap(
    frames: Sequence[Mapping[str, Any]],
    key: str,
    present_state: str,
    absent_state: str,
) -> Tuple[str, Any, Dict[str, Any]]:
    records = [frame.get(key) for frame in frames]
    valid = [item for item in records if isinstance(item, Mapping) and bool(item.get("valid", False))]
    ids = sorted({str(item.get("track_id")) for item in valid if item.get("track_id") not in (None, "")})
    if not valid:
        return absent_state, NOT_APPLICABLE, {"valid_frame_count": 0, "track_ids": [], "state": absent_state}
    gaps = [_as_float(item.get("arc_gap_m"), f"{key}.arc_gap_m") for item in valid]
    if len(valid) < 8 or len(ids) != 1 or any(gap <= 0.0 for gap in gaps):
        raise ValueError(f"{key} PRESENT needs >=8 valid positive arc gaps and one stable track ID")
    return present_state, _round(float(np.median(gaps))), {
        "valid_frame_count": len(valid), "track_ids": ids, "state": present_state,
    }


def _common_context(payload: Mapping[str, Any], family: str) -> Tuple[List[Mapping[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if family not in {"R-HLC", "R-TSB"}:
        raise ValueError("family must be R-HLC or R-TSB")
    frames = _require_pre_context_frames(payload)
    required = ("scenario_token", "map_version", "route_fingerprint", "initial_state_fingerprint", "map_location", "road_class", "log_id", "query_version")
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ValueError(f"missing required context fields: {', '.join(missing)}")
    coverage = []
    for index, frame in enumerate(frames):
        ego_valid = bool(frame.get("ego_valid", False))
        map_valid = bool(frame.get("map_valid", False))
        lane_valid = bool(frame.get("current_required_lane_valid", False))
        coverage.append({"frame_index": index, "ego_valid": ego_valid, "map_valid": map_valid, "current_required_lane_valid": lane_valid})
        _as_float(frame.get("speed_mps"), f"frames[{index}].speed_mps")
        _as_float(frame.get("lane_offset_m", 0.0), f"frames[{index}].lane_offset_m")
        _as_float(frame.get("legal_projected_dynamic_vehicle_count", 0), f"frames[{index}].legal_projected_dynamic_vehicle_count")
    eligible = all(item["ego_valid"] and item["map_valid"] and item["current_required_lane_valid"] for item in coverage)
    raw_payload = {
        "family": family,
        "scenario_token": str(payload["scenario_token"]),
        "map_version": str(payload["map_version"]),
        "route_fingerprint": str(payload["route_fingerprint"]),
        "initial_state_fingerprint": str(payload["initial_state_fingerprint"]),
        "history_source": str(payload.get("history_source", "OFFICIAL_HISTORY_BUFFER")),
        "frames": frames,
    }
    base = {
        "family": family,
        "scenario_token": str(payload["scenario_token"]),
        "map_version": str(payload["map_version"]),
        "route_fingerprint": str(payload["route_fingerprint"]),
        "initial_state_fingerprint": str(payload["initial_state_fingerprint"]),
        "history_source": str(payload.get("history_source", "OFFICIAL_HISTORY_BUFFER")),
        "sampling": {"window": "[t_anchor-1.0s,t_anchor)", "dt_seconds": DT_SECONDS, "exact_valid_frames": PRE_CONTEXT_FRAMES},
        "context_variables": {
            "map_location": str(payload["map_location"]),
            "road_class": str(payload["road_class"]),
            "log_id": str(payload["log_id"]),
            "initial_speed_mps": _round(float(np.median([_as_float(frame["speed_mps"], "speed_mps") for frame in frames]))),
            "traffic_density": int(round(float(np.median([_as_float(frame["legal_projected_dynamic_vehicle_count"], "density") for frame in frames])))),
        },
        "frame_valid_coverage": coverage,
        "eligible": eligible,
        "map_source_ids": dict(payload.get("map_source_ids", {})),
        "query_version": str(payload["query_version"]),
    }
    return frames, raw_payload, base


def build_canonical_context_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one exact ten-frame context input to a frozen canonical record."""
    family = str(payload.get("family"))
    frames, raw_payload, record = _common_context(payload, family)
    pattern, slots = _slot_summary(frames)
    record["context_variables"]["neighbor_availability_pattern"] = pattern
    record["slot_track_ids"] = slots
    if family == "R-HLC":
        direction = str(payload.get("intended_lane_change_direction", ""))
        if direction not in {"LEFT", "RIGHT"}:
            raise ValueError("HLC intended_lane_change_direction must be LEFT or RIGHT")
        front_state, front_gap, front_audit = _canonical_gap(frames, "target_front", "TARGET_FRONT_PRESENT", "TARGET_FRONT_ABSENT")
        rear_state, rear_gap, rear_audit = _canonical_gap(frames, "target_rear", "TARGET_REAR_PRESENT", "TARGET_REAR_ABSENT")
        record["context_variables"].update({
            "intended_lane_change_direction": direction,
            "initial_lane_offset_m": _round(float(np.median([_as_float(frame.get("lane_offset_m", 0.0), "lane_offset_m") for frame in frames]))),
            "target_lane_initial_front_gap_m": front_gap,
            "target_lane_initial_rear_gap_m": rear_gap,
        })
        record["missingness_states"] = {"target_front": front_state, "target_rear": rear_state}
        record["target_track_audit"] = {"target_front": front_audit, "target_rear": rear_audit}
    else:
        front_state, front_gap, front_audit = _canonical_gap(frames, "front", "FRONT_PRESENT", "FRONT_ABSENT")
        if front_state == "FRONT_ABSENT":
            lead_relative_speed, thw = NOT_APPLICABLE, NOT_APPLICABLE
        else:
            front_records = [frame["front"] for frame in frames if isinstance(frame.get("front"), Mapping) and bool(frame["front"].get("valid", False))]
            relative_speeds = [_as_float(item.get("lead_relative_speed_mps"), "front.lead_relative_speed_mps") for item in front_records]
            thws = [_as_float(item.get("thw_s"), "front.thw_s") for item in front_records]
            lead_relative_speed, thw = _round(float(np.median(relative_speeds))), _round(float(np.median(thws)))
        multi_hot = [str(item) for item in payload.get("hazard_multi_hot", ["NONE_OBSERVED"])]
        unknown = set(multi_hot).difference(TSB_HAZARD_PRIORITY)
        if unknown:
            raise ValueError(f"unknown TSB hazard states: {sorted(unknown)}")
        selected = next((state for state in TSB_HAZARD_PRIORITY if state in multi_hot), "NONE_OBSERVED")
        record["context_variables"].update({
            "initial_front_gap_m": front_gap,
            "initial_lead_relative_speed_mps": lead_relative_speed,
            "initial_thw_s": thw,
            "planned_stop_or_hazard_class": selected,
        })
        record["missingness_states"] = {"front": front_state}
        record["front_track_audit"] = front_audit
        record["hazard_multi_hot_audit"] = multi_hot
    record["pre_context_raw_hash"] = canonical_json_sha256(raw_payload)
    hash_payload = dict(record)
    record["canonical_context_json_hash"] = canonical_json_sha256(hash_payload)
    return record


def assert_pair_context_identity(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> Dict[str, Any]:
    fields = ("pre_context_raw_hash", "canonical_context_json_hash")
    equality = {field: baseline.get(field) == treatment.get(field) for field in fields}
    return {"pair_context_identity_pass": all(equality.values()), "fields": equality}


def calculate_hlc_option_b(time_s: Sequence[float], progress_p: Sequence[float], speed_mps: Sequence[float], map_valid: bool = True) -> Dict[str, Any]:
    """Measure exactly frozen HLC Option B from a projected lane-transition signal."""
    time = np.asarray(time_s, dtype=np.float64)
    p_raw = np.asarray(progress_p, dtype=np.float64)
    speed = np.asarray(speed_mps, dtype=np.float64)
    if len(time) < 6 or time.shape != p_raw.shape or time.shape != speed.shape or np.any(np.diff(time) <= 0):
        raise ValueError("HLC inputs must be equal-length, strictly timed vectors of at least six samples")
    if not np.isfinite(time).all() or not np.isfinite(p_raw).all() or not np.isfinite(speed).all():
        raise ValueError("HLC inputs must be finite")
    p = median3(np.clip(p_raw, 0.0, 1.0))
    dt = float(np.median(np.diff(time)))
    if abs(dt - DT_SECONDS) > 1e-6:
        raise ValueError("HLC v1.0 expects dt=0.1s; resampling requires a new contract version")
    result: Dict[str, Any] = {"option": "OPTION_B", "median3_p": [_round(v) for v in p], "map_valid": bool(map_valid)}
    if not map_valid:
        return {**result, "status": "MAP_INVALID", "hesitation_retreat_count": None, "commit_latency_s": None, "monotonic_transition_fraction": None}
    if np.any(speed < 1.0):
        low_speed = _run_ranges(speed < 1.0)
        if any((end - start + 1) * dt >= 0.5 - 1e-9 for start, end in low_speed):
            return {**result, "status": "LOW_SPEED_TRANSITION", "hesitation_retreat_count": None, "commit_latency_s": None, "monotonic_transition_fraction": None}
    departures = np.flatnonzero(p >= 0.10)
    if len(departures) == 0:
        return {**result, "status": "NO_DEPARTURE", "hesitation_retreat_count": 0, "commit_latency_s": None, "monotonic_transition_fraction": None}
    departure = int(departures[0])
    persistence = _seconds_to_samples(0.5)
    commitment: int | None = None
    for start in range(departure, len(p) - persistence + 1):
        if np.all(p[start:start + persistence] >= 0.75):
            commitment = start
            break
    if commitment is None:
        return {**result, "status": "UNFINISHED_TRANSITION", "departure_time_s": _round(time[departure]), "hesitation_retreat_count": 0, "commit_latency_s": None, "monotonic_transition_fraction": None}
    derivative = np.diff(p) / np.diff(time)
    negative_runs = _run_ranges(derivative <= -0.10)
    candidates: List[Tuple[int, int]] = []
    recovery_samples = _seconds_to_samples(0.3)
    for start, end in negative_runs:
        if start < departure or start >= commitment or (end - start + 1) * dt < 0.3 - 1e-9:
            continue
        recovery_start = min(end + 1, commitment)
        for possible in range(end + 1, max(end + 1, commitment - recovery_samples + 1)):
            if np.all(derivative[possible:possible + recovery_samples] >= 0.04):
                recovery_start = possible
                break
        event_end = min(commitment, max(end + 1, recovery_start))
        fall = float(p[start] - np.min(p[start:event_end + 1]))
        duration = float(time[event_end] - time[start])
        if fall >= 0.08 - 1e-9 and duration >= 0.4 - 1e-9:
            candidates.append((start, event_end))
    merged: List[Tuple[int, int]] = []
    for start, end in candidates:
        if merged and time[start] - time[merged[-1][1]] < 0.4 - 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    deltas = np.diff(p[departure:commitment + 1])
    denominator = float(np.sum(np.maximum(np.abs(deltas) - 0.008, 0.0)))
    monotonic = None if denominator < 0.1 else 1.0 - float(np.sum(np.maximum(-deltas - 0.008, 0.0))) / denominator
    return {
        **result,
        "status": "OK" if monotonic is not None else "NOT_EVALUABLE_MONOTONIC_DENOMINATOR",
        "departure_time_s": _round(time[departure]), "commit_time_s": _round(time[commitment]),
        "commit_latency_s": _round(time[commitment] - time[0]),
        "hesitation_retreat_count": len(merged),
        "retreat_episodes": [{"start_time_s": _round(time[start]), "end_time_s": _round(time[end])} for start, end in merged],
        "monotonic_transition_fraction": None if monotonic is None else _round(monotonic),
    }


def qualify_hlc_pair(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    if baseline.get("status") != "OK" or treatment.get("status") != "OK":
        reasons.append("MEASUREMENT_NOT_OK")
    if baseline.get("hesitation_retreat_count") != 0:
        reasons.append("BASELINE_RETREAT_NOT_ZERO")
    if not isinstance(treatment.get("hesitation_retreat_count"), int) or treatment["hesitation_retreat_count"] < 1:
        reasons.append("TREATMENT_RETREAT_LT_ONE")
    latency_delta = None
    if baseline.get("commit_latency_s") is not None and treatment.get("commit_latency_s") is not None:
        latency_delta = float(treatment["commit_latency_s"]) - float(baseline["commit_latency_s"])
        if latency_delta < 0.5 - 1e-9:
            reasons.append("COMMIT_LATENCY_DELTA_LT_0P5")
    else:
        reasons.append("COMMIT_LATENCY_NOT_EVALUABLE")
    monotonic_delta = None
    if baseline.get("monotonic_transition_fraction") is not None and treatment.get("monotonic_transition_fraction") is not None:
        monotonic_delta = float(treatment["monotonic_transition_fraction"]) - float(baseline["monotonic_transition_fraction"])
        if monotonic_delta > -0.1 + 1e-9:
            reasons.append("MONOTONIC_PENALTY_LT_0P1")
    else:
        reasons.append("MONOTONIC_NOT_EVALUABLE")
    return {"status": "HLC_MECHANISM_PAIR_PASS" if not reasons else "HLC_MECHANISM_PAIR_FAIL", "pass": not reasons, "reasons": reasons, "delta_commit_latency_s": None if latency_delta is None else _round(latency_delta), "delta_monotonic_fraction": None if monotonic_delta is None else _round(monotonic_delta)}


def calculate_tsb_option_a(time_s: Sequence[float], longitudinal_speed_mps: Sequence[float]) -> Dict[str, Any]:
    """Measure exactly frozen TSB Option A on timestamps at 0.1 second cadence."""
    time = np.asarray(time_s, dtype=np.float64)
    speed_raw = np.asarray(longitudinal_speed_mps, dtype=np.float64)
    if len(time) < 6 or time.shape != speed_raw.shape or np.any(np.diff(time) <= 0):
        raise ValueError("TSB inputs must be equal-length, strictly timed vectors of at least six samples")
    if not np.isfinite(time).all() or not np.isfinite(speed_raw).all():
        raise ValueError("TSB inputs must be finite")
    if abs(float(np.median(np.diff(time))) - DT_SECONDS) > 1e-6:
        raise ValueError("TSB v1.0 expects dt=0.1s; resampling requires a new contract version")
    speed = median3(speed_raw)
    accel = np.gradient(speed, time, edge_order=2)
    result: Dict[str, Any] = {"option": "OPTION_A", "median3_speed_mps": [_round(v) for v in speed], "acceleration_mps2": [_round(v) for v in accel]}
    low = _run_ranges(speed < 1.0)
    if any((end - start + 1) * DT_SECONDS >= 0.5 - 1e-9 for start, end in low):
        return {**result, "status": "LOW_SPEED_ENDSTOP", "brake_phase_count": None, "interstage_release_fraction": None, "second_brake_peak_ratio": None}
    raw = [(start, end) for start, end in _run_ranges(accel <= -0.80) if (end - start + 1) * DT_SECONDS >= 0.3 - 1e-9]
    phases: List[Tuple[int, int]] = []
    release_samples = _seconds_to_samples(0.3)
    for start, end in raw:
        if not phases:
            phases.append((start, end))
            continue
        prior_start, prior_end = phases[-1]
        gap = start - prior_end - 1
        gap_seconds = gap * DT_SECONDS
        has_release = gap >= release_samples and np.any(_run_ranges(accel[prior_end + 1:start] >= -0.20)) and any((b - a + 1) >= release_samples for a, b in _run_ranges(accel[prior_end + 1:start] >= -0.20))
        if gap_seconds < 0.3 - 1e-9 or not has_release:
            phases[-1] = (prior_start, end)
        else:
            phases.append((start, end))
    phase_records = [{"start_time_s": _round(time[start]), "end_time_s": _round(time[end]), "peak_decel_mps2": _round(float(np.max(-accel[start:end + 1])))} for start, end in phases]
    release_fraction = None
    peak_ratio = None
    if len(phases) >= 2:
        first_start, first_end = phases[0]
        second_start, second_end = phases[1]
        gap_speed = speed[first_end + 1:second_start]
        first_loss = max(float(speed[first_start] - np.min(speed[first_start:first_end + 1])), 0.1)
        release_fraction = float(np.max(gap_speed) - speed[first_end]) / first_loss if len(gap_speed) else 0.0
        first_peak = max(float(np.max(-accel[first_start:first_end + 1])), 0.8)
        second_peak = float(np.max(-accel[second_start:second_end + 1]))
        peak_ratio = second_peak / first_peak
    status = "OK" if phases else "NO_BRAKE_PHASE"
    return {**result, "status": status, "brake_phase_count": len(phases), "brake_phases": phase_records, "interstage_release_fraction": None if release_fraction is None else _round(release_fraction), "second_brake_peak_ratio": None if peak_ratio is None else _round(peak_ratio)}


def qualify_tsb_pair(baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    if baseline.get("status") != "OK" or treatment.get("status") != "OK":
        reasons.append("MEASUREMENT_NOT_OK")
    if baseline.get("brake_phase_count") != 1:
        reasons.append("BASELINE_PHASE_COUNT_NOT_ONE")
    if treatment.get("brake_phase_count") != 2:
        reasons.append("TREATMENT_PHASE_COUNT_NOT_EXACTLY_TWO")
    if treatment.get("interstage_release_fraction") is None or float(treatment["interstage_release_fraction"]) < 0.15 - 1e-9:
        reasons.append("RELEASE_FRACTION_LT_0P15")
    if treatment.get("second_brake_peak_ratio") is None or float(treatment["second_brake_peak_ratio"]) < 0.50 - 1e-9:
        reasons.append("SECOND_PEAK_RATIO_LT_0P50")
    return {"status": "TSB_MECHANISM_PAIR_PASS" if not reasons else "TSB_MECHANISM_PAIR_FAIL", "pass": not reasons, "reasons": reasons}


def trajectory_descriptors(time_s: Sequence[float], xy: Sequence[Sequence[float]], speed_mps: Sequence[float]) -> Dict[str, float]:
    """Frozen descriptor names used by R0 development F_match calipers."""
    time = np.asarray(time_s, dtype=np.float64)
    position = np.asarray(xy, dtype=np.float64)
    speed = np.asarray(speed_mps, dtype=np.float64)
    if position.shape != (len(time), 2) or speed.shape != time.shape:
        raise ValueError("descriptor inputs have incompatible shapes")
    heading = np.unwrap(np.arctan2(np.gradient(position[:, 1], time, edge_order=2), np.gradient(position[:, 0], time, edge_order=2)))
    return {
        "mean_speed": _round(float(np.mean(speed))),
        "end_minus_start_speed": _round(float(speed[-1] - speed[0])),
        "heading_change_abs_total": _round(float(np.sum(np.abs(np.diff(heading))))),
        "mean_abs_accel": _round(float(np.mean(np.abs(np.gradient(speed, time, edge_order=2))))),
        "path_length": _round(float(np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1)))),
    }


def frozen_f_match(baseline: Mapping[str, float], treatment: Mapping[str, float], family: str) -> Dict[str, Any]:
    calipers = {"mean_speed": 0.708203939, "end_minus_start_speed": 0.978755681, "path_length": 5.38423459}
    if family == "R-HLC":
        calipers["heading_change_abs_total"] = 0.0492160141
    elif family == "R-TSB":
        calipers["mean_abs_accel"] = 0.11777666
    else:
        raise ValueError("F_match family must be R-HLC or R-TSB")
    delta = {key: _round(abs(float(treatment[key]) - float(baseline[key]))) for key in calipers}
    pass_by_feature = {key: delta[key] <= caliper + 1e-12 for key, caliper in calipers.items()}
    return {"status": "F_MATCH_PASS" if all(pass_by_feature.values()) else "F_MATCH_FAIL", "pass": all(pass_by_feature.values()), "absolute_delta": delta, "calipers": calipers, "pass_by_feature": pass_by_feature}
