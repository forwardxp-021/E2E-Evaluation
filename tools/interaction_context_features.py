#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Tuple
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
        "rms_accel", "rms_jerk", "max_abs_accel", "max_abs_jerk", "mean_thw", "min_thw", "front_mean_distance", "front_min_distance",
        "front_closing_rate_mean", "front_closing_rate_p95", "rms_yaw_rate", "rms_curvature", "heading_change_total",
        "lane_change_count_proxy", "lane_change_rate_proxy", "lane_change_left_count_proxy", "lane_change_right_count_proxy",
        "lane_change_duration_mean_proxy", "max_lateral_speed", "rms_lateral_accel", "lane_change_oscillation_score_proxy",
        "front_pressure_score", "left_front_min_gap", "left_rear_min_gap", "right_front_min_gap", "right_rear_min_gap", "left_gap_min", "right_gap_min",
        "left_gap_acceptance_proxy", "right_gap_acceptance_proxy", "rear_vehicle_pressure_proxy", "yielding_score_proxy", "assertiveness_score_proxy"
    ]
    return np.asarray(feats, dtype=np.float32), names
