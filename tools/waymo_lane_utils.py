#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LaneInfo:
    lane_id: str
    centerline_xy: np.ndarray
    seg_heading: np.ndarray
    seg_len: np.ndarray
    s_prefix: np.ndarray
    left_neighbor_lane_ids: List[str] = field(default_factory=list)
    right_neighbor_lane_ids: List[str] = field(default_factory=list)
    entry_lane_ids: List[str] = field(default_factory=list)
    exit_lane_ids: List[str] = field(default_factory=list)
    lane_type: str = "unknown"
    topology_source: str = "unknown"


def _lane_points_from_feature(feature) -> Optional[np.ndarray]:
    lane = getattr(feature, "lane", None)
    if lane is None:
        return None
    poly = getattr(lane, "polyline", None)
    if not poly:
        return None
    pts = np.asarray([[p.x, p.y] for p in poly], dtype=np.float32)
    if len(pts) < 2:
        return None
    return pts


def extract_lane_polylines(scenario) -> Dict[str, LaneInfo]:
    lanes: Dict[str, LaneInfo] = {}
    for mf in getattr(scenario, "map_features", []):
        pts = _lane_points_from_feature(mf)
        if pts is None:
            continue
        dxy = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(dxy, axis=1)
        valid = seg_len > 1e-6
        if not np.any(valid):
            continue
        seg_len = np.where(valid, seg_len, 1e-6)
        seg_heading = np.arctan2(dxy[:, 1], dxy[:, 0])
        s_prefix = np.concatenate([[0.0], np.cumsum(seg_len)])
        lane = getattr(mf, "lane", None)
        left_ids = []
        right_ids = []
        for fld, dst in (("left_neighbors", left_ids), ("right_neighbors", right_ids)):
            for nb in getattr(lane, fld, []):
                nid = getattr(nb, "feature_id", None)
                if nid is not None:
                    dst.append(str(nid))
        entry_ids = [str(x) for x in getattr(lane, "entry_lanes", [])]
        exit_ids = [str(x) for x in getattr(lane, "exit_lanes", [])]
        lanes[str(getattr(mf, "id"))] = LaneInfo(
            lane_id=str(getattr(mf, "id")), centerline_xy=pts, seg_heading=seg_heading,
            seg_len=seg_len, s_prefix=s_prefix, left_neighbor_lane_ids=left_ids,
            right_neighbor_lane_ids=right_ids, entry_lane_ids=entry_ids, exit_lane_ids=exit_ids,
            lane_type=str(getattr(lane, "type", "unknown")),
            topology_source="proto_topology" if (left_ids or right_ids) else "geometric_lane_adjacency",
        )
    return lanes


def project_point_to_lane(point_xy, lane_info: LaneInfo) -> dict:
    p = np.asarray(point_xy, dtype=np.float64)
    pts = lane_info.centerline_xy.astype(np.float64)
    best = None
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        den = float(np.dot(ab, ab))
        if den < 1e-9:
            continue
        t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
        q = a + t * ab
        d = p - q
        eu = float(np.linalg.norm(d))
        h = float(lane_info.seg_heading[i])
        left = np.array([-math.sin(h), math.cos(h)])
        l = float(np.dot(d, left))
        s = float(lane_info.s_prefix[i] + t * lane_info.seg_len[i])
        row = (eu, i, s, l, h)
        if best is None or row[0] < best[0]:
            best = row
    if best is None:
        return dict(lane_id=lane_info.lane_id, s=np.nan, l=np.nan, heading=np.nan, distance_to_lane=np.inf, projection_success=False)
    return dict(lane_id=lane_info.lane_id, s=best[2], l=best[3], heading=best[4], distance_to_lane=abs(best[3]), projection_success=True)


def find_best_lane_for_agent(point_xy, heading, lane_infos: Dict[str, LaneInfo], max_lateral_distance: float, max_heading_diff: float):
    cand = []
    for lid, ln in lane_infos.items():
        proj = project_point_to_lane(point_xy, ln)
        if not proj["projection_success"]:
            continue
        if proj["distance_to_lane"] > max_lateral_distance:
            continue
        hd = 0.0
        if np.isfinite(heading):
            hd = abs(wrap_to_pi(float(heading) - float(proj["heading"])))
            if hd > max_heading_diff:
                continue
        cand.append((proj["distance_to_lane"], hd, lid, proj))
    if not cand:
        return None, "no_lane_passed_threshold"
    cand.sort(key=lambda x: (x[0], x[1]))
    return cand[0][3], "ok"
