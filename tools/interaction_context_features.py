#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def aggregate_interaction_features(ego_seq: np.ndarray, neighbor_seq: np.ndarray, dt: float) -> Tuple[np.ndarray, List[str]]:
    speed = ego_seq[:, 5]
    accel = ego_seq[:, 6]
    yaw_rate = ego_seq[:, 7]
    jerk = np.diff(accel, prepend=accel[0]) / max(dt, 1e-6)
    curvature = yaw_rate / np.maximum(speed, 1e-3)
    heading = ego_seq[:, 4]

    front = neighbor_seq[0]
    lf, lr, rf, rr = neighbor_seq[1], neighbor_seq[2], neighbor_seq[3], neighbor_seq[4]

    def safe_mean(x):
        return float(np.nanmean(x)) if np.any(np.isfinite(x)) else 0.0
    def safe_min(x):
        return float(np.nanmin(x)) if np.any(np.isfinite(x)) else 0.0
    def safe_p95(x):
        return float(np.nanpercentile(x, 95)) if np.any(np.isfinite(x)) else 0.0

    thw_f = np.where(front[:, 0] > 0.5, front[:, 10], np.nan)
    dist_f = np.where(front[:, 0] > 0.5, front[:, 5], np.nan)
    cr_f = np.where(front[:, 0] > 0.5, front[:, 8], np.nan)

    lane_change_proxy = np.abs(ego_seq[:, 1]) > 2.0
    lc_count = int(np.sum((lane_change_proxy[1:] & (~lane_change_proxy[:-1])).astype(np.int32)))

    feats = [
        np.sqrt(np.mean(accel ** 2)), np.sqrt(np.mean(jerk ** 2)), np.max(np.abs(accel)), np.max(np.abs(jerk)),
        safe_mean(thw_f), safe_min(thw_f), safe_mean(dist_f), safe_min(dist_f), safe_mean(cr_f), safe_p95(cr_f),
        np.sqrt(np.mean(yaw_rate ** 2)), np.sqrt(np.mean(curvature ** 2)), float(np.sum(np.abs(wrap(np.diff(heading))))),
        float(lc_count), float(lc_count / max(len(ego_seq) * dt, 1e-6)),
        float(lc_count // 2), float(lc_count - lc_count // 2),
        float(np.mean(np.diff(np.where(lane_change_proxy, 1, 0), prepend=0) > 0) * dt),
        float(np.max(np.abs(ego_seq[:, 3]))), np.sqrt(np.mean(np.diff(ego_seq[:, 3], prepend=ego_seq[0, 3]) ** 2)),
        float(np.mean(np.abs(np.diff(lane_change_proxy.astype(np.float32), prepend=0)))),
        safe_mean(np.where(front[:, 0] > 0.5, np.maximum(0, 30 - front[:, 5]), np.nan)),
        safe_min(np.where(lf[:, 0] > 0.5, lf[:, 5], np.nan)), safe_min(np.where(lr[:, 0] > 0.5, lr[:, 5], np.nan)),
        safe_min(np.where(rf[:, 0] > 0.5, rf[:, 5], np.nan)), safe_min(np.where(rr[:, 0] > 0.5, rr[:, 5], np.nan)),
        min(safe_min(np.where(lf[:, 0] > 0.5, lf[:, 5], np.nan)), safe_min(np.where(lr[:, 0] > 0.5, lr[:, 5], np.nan))),
        min(safe_min(np.where(rf[:, 0] > 0.5, rf[:, 5], np.nan)), safe_min(np.where(rr[:, 0] > 0.5, rr[:, 5], np.nan))),
        float(np.mean(np.where(lf[:, 0] > 0.5, lf[:, 8] > 0, 0))), float(np.mean(np.where(rf[:, 0] > 0.5, rf[:, 8] > 0, 0))),
        safe_mean(np.where((lr[:, 0] > 0.5) | (rr[:, 0] > 0.5), np.maximum(lr[:, 8], rr[:, 8]), np.nan)),
        safe_mean(np.where(front[:, 0] > 0.5, np.clip(front[:, 8] / np.maximum(front[:, 5], 1e-3), -5, 5), np.nan)),
        float(np.mean(speed > np.nanmean(speed)))
    ]
    names = [
        "rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk", "mean_thw", "min_thw", "mean_front_distance", "min_front_distance",
        "mean_rel_speed", "p95_rel_speed", "rms_yaw_rate", "rms_curvature", "heading_change_total",
        "lane_change_count_proxy", "lane_change_rate_proxy", "lane_change_left_count_proxy", "lane_change_right_count_proxy",
        "lane_change_duration_mean_proxy", "max_lateral_speed", "rms_lateral_accel", "lane_change_oscillation_score_proxy",
        "front_pressure_score", "left_front_min_gap", "left_rear_min_gap", "right_front_min_gap", "right_rear_min_gap", "left_gap_min", "right_gap_min",
        "left_gap_acceptance_proxy", "right_gap_acceptance_proxy", "rear_vehicle_pressure_proxy", "yielding_score_proxy", "assertiveness_score_proxy"
    ]
    return np.asarray(feats, dtype=np.float32), names


FEATURE_SPECS: List[Dict[str, Any]] = [
    {"name": "rms_accel", "description": "Root mean square of ego longitudinal speed derivative.", "source": "aggregate_interaction_features: accel"},
    {"name": "rms_jerk", "description": "Root mean square of ego acceleration derivative.", "source": "aggregate_interaction_features: jerk"},
    {"name": "max_abs_accel", "description": "Maximum absolute ego acceleration.", "source": "aggregate_interaction_features: accel"},
    {"name": "max_abs_jerk", "description": "Maximum absolute ego jerk.", "source": "aggregate_interaction_features: jerk"},
    {"name": "mean_thw", "description": "Mean front time-headway for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,10]"},
    {"name": "min_thw", "description": "Minimum front time-headway for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,10]"},
    {"name": "mean_front_distance", "description": "Mean front distance for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,5]"},
    {"name": "min_front_distance", "description": "Minimum front distance for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,5]"},
    {"name": "mean_rel_speed", "description": "Mean front closing rate for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,8]"},
    {"name": "p95_rel_speed", "description": "95th percentile front closing rate for valid front-neighbor frames.", "source": "aggregate_interaction_features: front[:,8]"},
    {"name": "rms_yaw_rate", "description": "Root mean square of ego yaw-rate proxy.", "source": "aggregate_interaction_features: yaw_rate"},
    {"name": "rms_curvature", "description": "Root mean square of curvature proxy (yaw_rate/speed).", "source": "aggregate_interaction_features: curvature"},
    {"name": "heading_change_total", "description": "Total absolute heading change across the window.", "source": "aggregate_interaction_features: heading"},
    {"name": "lane_change_count_proxy", "description": "Count of lane-change transitions from lateral-offset proxy.", "source": "aggregate_interaction_features: lane_change_proxy"},
    {"name": "lane_change_rate_proxy", "description": "Lane-change transition count normalized by window duration.", "source": "aggregate_interaction_features: lane_change_proxy"},
    {"name": "lane_change_left_count_proxy", "description": "Proxy split of lane-change count attributed to left.", "source": "aggregate_interaction_features: lc_count // 2"},
    {"name": "lane_change_right_count_proxy", "description": "Proxy split of lane-change count attributed to right.", "source": "aggregate_interaction_features: lc_count - lc_count // 2"},
    {"name": "lane_change_duration_mean_proxy", "description": "Mean transition indicator from lane-change proxy.", "source": "aggregate_interaction_features: lane_change_proxy"},
    {"name": "max_lateral_speed", "description": "Maximum absolute local lateral speed.", "source": "aggregate_interaction_features: ego_seq[:,3]"},
    {"name": "rms_lateral_accel", "description": "Root mean square of lateral acceleration proxy.", "source": "aggregate_interaction_features: diff(ego_seq[:,3])"},
    {"name": "lane_change_oscillation_score_proxy", "description": "Mean absolute change in lane-change proxy signal.", "source": "aggregate_interaction_features: lane_change_proxy"},
    {"name": "front_pressure_score", "description": "Mean clipped front-pressure proxy max(0,30-front_distance).", "source": "aggregate_interaction_features: front[:,5]"},
    {"name": "left_front_min_gap", "description": "Minimum left-front gap.", "source": "aggregate_interaction_features: left_front[:,5]"},
    {"name": "left_rear_min_gap", "description": "Minimum left-rear gap.", "source": "aggregate_interaction_features: left_rear[:,5]"},
    {"name": "right_front_min_gap", "description": "Minimum right-front gap.", "source": "aggregate_interaction_features: right_front[:,5]"},
    {"name": "right_rear_min_gap", "description": "Minimum right-rear gap.", "source": "aggregate_interaction_features: right_rear[:,5]"},
    {"name": "left_gap_min", "description": "Minimum of left-front and left-rear minimum gaps.", "source": "aggregate_interaction_features: min(left_front_min_gap,left_rear_min_gap)"},
    {"name": "right_gap_min", "description": "Minimum of right-front and right-rear minimum gaps.", "source": "aggregate_interaction_features: min(right_front_min_gap,right_rear_min_gap)"},
    {"name": "left_gap_acceptance_proxy", "description": "Fraction of frames with positive left-front closing rate.", "source": "aggregate_interaction_features: left_front[:,8]"},
    {"name": "right_gap_acceptance_proxy", "description": "Fraction of frames with positive right-front closing rate.", "source": "aggregate_interaction_features: right_front[:,8]"},
    {"name": "rear_vehicle_pressure_proxy", "description": "Mean rear closing-rate pressure proxy from left/right rear.", "source": "aggregate_interaction_features: max(left_rear[:,8],right_rear[:,8])"},
    {"name": "yielding_score_proxy", "description": "Mean clipped ratio closing_rate/front_distance in [-5,5].", "source": "aggregate_interaction_features: front[:,8]/front[:,5]"},
    {"name": "assertiveness_score_proxy", "description": "Fraction of frames with speed above sequence mean.", "source": "aggregate_interaction_features: speed > mean(speed)"},
]


def get_feature_schema() -> Dict[str, Any]:
    features = [{"index": i, **spec} for i, spec in enumerate(FEATURE_SPECS)]
    if len(features) != 33:
        raise RuntimeError(f"Expected 33 canonical interaction features, got {len(features)}.")
    return {"feature_dim": 33, "features": features}


def write_feature_schema_json(path: Path) -> None:
    schema = get_feature_schema()
    path = Path(path)
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")