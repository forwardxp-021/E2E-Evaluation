#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse
import math
import time
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
    seg_start_xy: np.ndarray
    seg_vec_xy: np.ndarray
    seg_den: np.ndarray
    bbox_min_xy: np.ndarray
    bbox_max_xy: np.ndarray
    bbox_center_xy: np.ndarray
    left_neighbor_lane_ids: List[str] = field(default_factory=list)
    right_neighbor_lane_ids: List[str] = field(default_factory=list)
    entry_lane_ids: List[str] = field(default_factory=list)
    exit_lane_ids: List[str] = field(default_factory=list)
    lane_type: str = "unknown"
    topology_source: str = "unknown"
    left_neighbor_relations: List[dict] = field(default_factory=list)
    right_neighbor_relations: List[dict] = field(default_factory=list)


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


def _build_lane_geom(pts: np.ndarray):
    dxy = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(dxy, axis=1)
    valid = seg_len > 1e-6
    if not np.any(valid):
        return None
    seg_len_safe = np.where(valid, seg_len, 1e-6)
    seg_heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    s_prefix = np.concatenate([[0.0], np.cumsum(seg_len_safe)])
    seg_start = pts[:-1].astype(np.float64)
    seg_vec = dxy.astype(np.float64)
    seg_den = np.sum(seg_vec * seg_vec, axis=1)
    bbox_min = np.min(pts, axis=0).astype(np.float64)
    bbox_max = np.max(pts, axis=0).astype(np.float64)
    bbox_ctr = ((bbox_min + bbox_max) * 0.5).astype(np.float64)
    return seg_heading, seg_len_safe, s_prefix, seg_start, seg_vec, seg_den, bbox_min, bbox_max, bbox_ctr


def extract_lane_polylines(scenario) -> Dict[str, LaneInfo]:
    lanes: Dict[str, LaneInfo] = {}
    for mf in getattr(scenario, "map_features", []):
        pts = _lane_points_from_feature(mf)
        if pts is None:
            continue
        geom = _build_lane_geom(pts)
        if geom is None:
            continue
        seg_heading, seg_len, s_prefix, seg_start, seg_vec, seg_den, bbox_min, bbox_max, bbox_ctr = geom
        lane = getattr(mf, "lane", None)
        left_ids = []
        right_ids = []
        left_relations = []
        right_relations = []
        for fld, dst, relations in (("left_neighbors", left_ids, left_relations), ("right_neighbors", right_ids, right_relations)):
            for nb in getattr(lane, fld, []):
                nid = getattr(nb, "feature_id", None)
                if nid is not None:
                    dst.append(str(nid))
                    relations.append({
                        "lane_id": str(nid),
                        "self_start_index": int(getattr(nb, "self_start_index", 0)),
                        "self_end_index": int(getattr(nb, "self_end_index", 0)),
                        "neighbor_start_index": int(getattr(nb, "neighbor_start_index", 0)),
                        "neighbor_end_index": int(getattr(nb, "neighbor_end_index", 0)),
                    })
        entry_ids = [str(x) for x in getattr(lane, "entry_lanes", [])]
        exit_ids = [str(x) for x in getattr(lane, "exit_lanes", [])]
        lanes[str(getattr(mf, "id"))] = LaneInfo(
            lane_id=str(getattr(mf, "id")), centerline_xy=pts, seg_heading=seg_heading,
            seg_len=seg_len, s_prefix=s_prefix, seg_start_xy=seg_start, seg_vec_xy=seg_vec,
            seg_den=seg_den, bbox_min_xy=bbox_min, bbox_max_xy=bbox_max, bbox_center_xy=bbox_ctr,
            left_neighbor_lane_ids=left_ids, right_neighbor_lane_ids=right_ids,
            entry_lane_ids=entry_ids, exit_lane_ids=exit_ids,
            lane_type=str(getattr(lane, "type", "unknown")),
            topology_source="proto_topology" if (left_ids or right_ids) else "geometric_lane_adjacency",
            left_neighbor_relations=left_relations,
            right_neighbor_relations=right_relations,
        )
    return lanes


def project_point_to_lane(point_xy, lane_info: LaneInfo, eps: float = 1e-9) -> dict:
    p = np.asarray(point_xy, dtype=np.float64)[None, :]
    den = lane_info.seg_den
    valid = den > eps
    if not np.any(valid):
        return dict(lane_id=lane_info.lane_id, s=np.nan, l=np.nan, heading=np.nan, distance_to_lane=np.inf, projection_success=False)
    A = lane_info.seg_start_xy
    AB = lane_info.seg_vec_xy
    u = np.sum((p - A) * AB, axis=1) / np.where(valid, den, 1.0)
    u = np.clip(u, 0.0, 1.0)
    q = A + u[:, None] * AB
    d = p - q
    eu = np.linalg.norm(d, axis=1)
    eu[~valid] = np.inf
    bi = int(np.argmin(eu))
    if not np.isfinite(eu[bi]):
        return dict(lane_id=lane_info.lane_id, s=np.nan, l=np.nan, heading=np.nan, distance_to_lane=np.inf, projection_success=False)
    h = float(lane_info.seg_heading[bi])
    left = np.array([-math.sin(h), math.cos(h)], dtype=np.float64)
    l = float(np.dot(d[bi], left))
    s = float(lane_info.s_prefix[bi] + u[bi] * lane_info.seg_len[bi])
    return dict(lane_id=lane_info.lane_id, s=s, l=l, heading=h, distance_to_lane=abs(l), projection_success=True, segment_index=bi, segment_fraction=float(u[bi]))


def _candidate_lane_ids(point_xy, lane_infos, search_radius=20.0, topk=32):
    p = np.asarray(point_xy, dtype=np.float64)
    hits = []
    near = []
    for lid, ln in lane_infos.items():
        c = ln.bbox_center_xy
        cd = float(np.hypot(*(p - c)))
        near.append((cd, lid))
        lo = ln.bbox_min_xy - search_radius
        hi = ln.bbox_max_xy + search_radius
        if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]:
            hits.append((cd, lid))
    rows = hits if hits else near
    rows.sort(key=lambda x: x[0])
    return [lid for _, lid in rows[:max(1, int(topk))]], len(hits)


def find_best_lane_for_agent(point_xy, heading, lane_infos: Dict[str, LaneInfo], max_lateral_distance: float, max_heading_diff: float,
                             search_radius: float = 20.0, topk_candidates: int = 32, disable_spatial_index: bool = False):
    candidate_ids = list(lane_infos.keys()) if disable_spatial_index else _candidate_lane_ids(point_xy, lane_infos, search_radius, topk_candidates)[0]
    cand = []
    projectable_count = 0
    lateral_pass_count = 0
    for lid in candidate_ids:
        proj = project_point_to_lane(point_xy, lane_infos[lid])
        if not proj["projection_success"]:
            continue
        projectable_count += 1
        if proj["distance_to_lane"] > max_lateral_distance:
            continue
        lateral_pass_count += 1
        hd = 0.0
        if np.isfinite(heading):
            hd = abs(wrap_to_pi(float(heading) - float(proj["heading"])))
            if hd > max_heading_diff:
                continue
        cand.append((proj["distance_to_lane"], hd, lid, proj))
    if not cand:
        if not candidate_ids:
            reason = "no_candidate_lane"
        elif projectable_count == 0:
            reason = "no_projectable_lane"
        elif lateral_pass_count == 0:
            reason = "lateral_distance_exceeded"
        else:
            reason = "heading_difference_exceeded"
        return None, reason, len(candidate_ids)
    cand.sort(key=lambda x: (x[0], x[1]))
    return cand[0][3], "ok", len(candidate_ids)


def _self_test():
    rng = np.random.default_rng(0)
    lanes = {}
    for i in range(100):
        x = np.linspace(0, 200, 101)
        y = np.full_like(x, i * 2.0)
        pts = np.stack([x, y], axis=1).astype(np.float32)
        g = _build_lane_geom(pts)
        lanes[str(i)] = LaneInfo(str(i), pts, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8])
    points = np.stack([rng.uniform(0, 200, size=1000), rng.uniform(-20, 220, size=1000)], axis=1)
    t0 = time.perf_counter()
    ok = 0
    for p in points:
        proj, _, _ = find_best_lane_for_agent(p, np.nan, lanes, 10.0, math.pi, 20.0, 32, False)
        ok += int(proj is not None and np.isfinite(proj["s"]))
    dt = time.perf_counter() - t0
    print(f"self_test: projected 1000 points over 100x100 segments in {dt:.3f}s; finite={ok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
