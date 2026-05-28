#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None

FEATURES = [
    'distance_to_crosswalk_min', 'has_crosswalk_near_30m', 'distance_to_stop_sign_min', 'has_stop_sign_near_40m',
    'lane_curvature_mean', 'lane_curvature_max', 'lane_heading_change_total', 'lane_count_near_30m', 'road_line_count_near_30m',
    'road_edge_count_near_30m', 'crosswalk_count_near_30m', 'stop_sign_count_near_40m', 'speed_bump_count_near_30m',
    'map_complexity_score', 'intersection_proxy', 'map_match_valid', 'fallback_full_scenario_path'
]

def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def extract_meta(shard_dir: Path):
    cands = [shard_dir / 'metadata.csv', shard_dir / 'meta.csv', shard_dir / 'meta.npy']
    found = [str(p) for p in cands if p.exists()]
    if (shard_dir / 'metadata.csv').exists():
        df = pd.read_csv(shard_dir / 'metadata.csv')
    elif (shard_dir / 'meta.csv').exists():
        df = pd.read_csv(shard_dir / 'meta.csv')
    elif (shard_dir / 'meta.npy').exists():
        arr = np.load(shard_dir / 'meta.npy', allow_pickle=True)
        if hasattr(arr, 'dtype') and arr.dtype.names:
            df = pd.DataFrame({k: arr[k] for k in arr.dtype.names})
        else:
            raise ValueError(f'meta.npy 非结构化数组，无法解析字段: {shard_dir}/meta.npy')
    else:
        raise FileNotFoundError(f'未找到 metadata.csv/meta.csv/meta.npy: {shard_dir}')
    return df, found

def pick_col(cols, aliases):
    return next((c for c in aliases if c in cols), None)

def build_rows(df, shard_id, g0):
    cols = set(df.columns)
    scenario_col = pick_col(cols, ['scenario_id', 'scenarioId', 'scenario'])
    start_col = pick_col(cols, ['window_start', 'start_frame', 'start_idx'])
    end_col = pick_col(cols, ['window_end', 'end_frame', 'end_idx'])
    if scenario_col is None:
        raise ValueError(f'元数据缺少 scenario_id 字段，现有字段: {sorted(cols)}')
    n = len(df)
    return pd.DataFrame({
        'global_row': np.arange(g0, g0 + n),
        'shard_id': shard_id,
        'local_row': np.arange(n),
        'scenario_id': df[scenario_col].astype(str),
        'window_start': df[start_col].values if start_col else -1,
        'window_end': df[end_col].values if end_col else -1,
    })

def load_waymo_parser():
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
        return tf, scenario_pb2
    except Exception as e:
        raise RuntimeError('Waymo proto 解析器不可用。原始错误: ' + str(e)) from e

def _polyline_points(polyline):
    return np.array([[p.x, p.y] for p in polyline], dtype=np.float32) if polyline else np.zeros((0, 2), dtype=np.float32)

def _finite_xy(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.asarray(arr[:, :2], dtype=np.float32)
    return xy[np.isfinite(xy).all(axis=1)]

def parse_scenario_maps(raw_dir: Path, needed_scenario_ids: Optional[set] = None, max_scenarios_scanned: int = 0):
    tf, scenario_pb2 = load_waymo_parser()
    tfrecs = sorted(raw_dir.glob('*.tfrecord*'))
    if not tfrecs:
        raise FileNotFoundError(f'raw_scenario_dir 下未找到 tfrecord: {raw_dir}')
    out = {}
    scanned = 0
    for tfr in tfrecs:
        ds = tf.data.TFRecordDataset([str(tfr)], compression_type='')
        for rec in ds:
            scanned += 1
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(bytes(rec.numpy()))
            sid = str(sc.scenario_id)
            if needed_scenario_ids is not None and sid not in needed_scenario_ids:
                if max_scenarios_scanned > 0 and scanned >= max_scenarios_scanned:
                    return out, scanned, len(tfrecs)
                continue
            d = {'lane': [], 'road_line': [], 'road_edge': [], 'crosswalk': [], 'stop_sign': [], 'speed_bump': []}
            for mf in sc.map_features:
                which = mf.WhichOneof('feature_data')
                if which == 'lane':
                    d['lane'].append(_polyline_points(mf.lane.polyline))
                elif which == 'road_line':
                    d['road_line'].append(_polyline_points(mf.road_line.polyline))
                elif which == 'road_edge':
                    d['road_edge'].append(_polyline_points(mf.road_edge.polyline))
                elif which == 'crosswalk':
                    d['crosswalk'].append(_polyline_points(mf.crosswalk.polygon))
                elif which == 'stop_sign':
                    d['stop_sign'].append(np.array([[mf.stop_sign.position.x, mf.stop_sign.position.y]], dtype=np.float32))
                elif which == 'speed_bump':
                    d['speed_bump'].append(_polyline_points(mf.speed_bump.polygon))
            out[sid] = d
            if needed_scenario_ids is not None and len(out) >= len(needed_scenario_ids):
                return out, scanned, len(tfrecs)
            if max_scenarios_scanned > 0 and scanned >= max_scenarios_scanned:
                return out, scanned, len(tfrecs)
    return out, scanned, len(tfrecs)

def min_dist_points_chunked(path_xy: np.ndarray, points_xy: np.ndarray, chunk_size: int = 4096) -> float:
    if path_xy.size == 0 or points_xy.size == 0:
        return float(np.inf)
    path = _finite_xy(path_xy)
    pts = _finite_xy(points_xy)
    if path.size == 0 or pts.size == 0:
        return float(np.inf)
    best_d2 = np.inf
    for i in range(0, pts.shape[0], chunk_size):
        ch = pts[i:i + chunk_size]
        d2 = ((path[:, None, :] - ch[None, :, :]) ** 2).sum(axis=2)
        local = float(np.min(d2))
        if local < best_d2:
            best_d2 = local
    return float(np.sqrt(best_d2))

def min_dist_to_shape(path_xy: np.ndarray, shape_xy: np.ndarray) -> float:
    return min_dist_points_chunked(path_xy, shape_xy)

def min_dist_to_shapes(path_xy: np.ndarray, shapes: List[np.ndarray]) -> float:
    mins = [min_dist_to_shape(path_xy, s) for s in shapes if s.size > 0]
    return float(np.min(mins)) if mins else float(np.inf)

def filter_shapes_near_path(path_xy: np.ndarray, shapes: List[np.ndarray], radius_m: float) -> List[np.ndarray]:
    path = _finite_xy(path_xy)
    if path.shape[0] < 1:
        return []
    if cKDTree is not None:
        tree = cKDTree(path)
        out = []
        for s in shapes:
            pts = _finite_xy(s)
            if pts.shape[0] == 0:
                continue
            d, _ = tree.query(pts, k=1)
            if np.min(d) <= radius_m:
                out.append(s)
        return out
    out = []
    for s in shapes:
        if min_dist_to_shape(path, s) <= radius_m:
            out.append(s)
    return out

def lane_curvature_stats(lanes):
    curvs = []
    heading_total = 0.0
    for ln in lanes:
        pts = _finite_xy(ln)
        if pts.shape[0] < 3:
            continue
        d = np.diff(pts, axis=0)
        headings = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dh = np.abs(np.diff(headings))
        if dh.size:
            curvs.extend(dh.tolist())
            heading_total += float(dh.sum())
    if not curvs:
        return 0.0, 0.0, 0.0
    return float(np.mean(curvs)), float(np.max(curvs)), heading_total

def extract_ego_path_from_context(ctx_row: np.ndarray) -> Tuple[np.ndarray, str, Optional[str]]:
    arr = np.asarray(ctx_row)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return _finite_xy(arr[:, :2]), 'NTF_direct', None
    if arr.ndim == 3:
        # ATF: assume agent-axis first, pick agent 0 conservatively
        if arr.shape[2] >= 2:
            return _finite_xy(arr[0, :, :2]), 'ATF_agent0', '多agent上下文，默认agent0作为ego，存在不确定性'
        # TAF
        if arr.shape[1] >= 1 and arr.shape[2] >= 2:
            return _finite_xy(arr[:, 0, :2]), 'TAF_agent0', '多agent上下文，默认agent0作为ego，存在不确定性'
    return np.zeros((0, 2), dtype=np.float32), 'unknown', f'不支持的 context row shape: {arr.shape}'

def compute_features(path_xy, md, warnings):
    x = np.zeros((len(FEATURES),), dtype=np.float32)
    near_lanes = filter_shapes_near_path(path_xy, md['lane'], 30.0)
    near_road_lines = filter_shapes_near_path(path_xy, md['road_line'], 30.0)
    near_road_edges = filter_shapes_near_path(path_xy, md['road_edge'], 30.0)
    near_crosswalks = filter_shapes_near_path(path_xy, md['crosswalk'], 30.0)
    near_stop_signs = filter_shapes_near_path(path_xy, md['stop_sign'], 40.0)
    near_speed_bumps = filter_shapes_near_path(path_xy, md['speed_bump'], 30.0)
    d_cross = min_dist_to_shapes(path_xy, md['crosswalk'])
    d_stop = min_dist_to_shapes(path_xy, md['stop_sign'])
    lmean, lmax, lchg = lane_curvature_stats(near_lanes)
    if len(near_lanes) == 0:
        warnings['no_near_lane_rows'] += 1
    x[0] = d_cross
    x[1] = 1.0 if len(near_crosswalks) > 0 else 0.0
    x[2] = d_stop
    x[3] = 1.0 if len(near_stop_signs) > 0 else 0.0
    x[4] = lmean
    x[5] = lmax
    x[6] = lchg
    x[7] = len(near_lanes)
    x[8] = len(near_road_lines)
    x[9] = len(near_road_edges)
    x[10] = len(near_crosswalks)
    x[11] = len(near_stop_signs)
    x[12] = len(near_speed_bumps)
    x[13] = x[7] + x[8] + x[9] + x[10] * 2 + x[11] * 2 + x[12] * 1.5 + x[5]
    x[14] = 1.0 if (x[1] > 0.5 or x[3] > 0.5 or x[7] >= 6 or x[8] >= 8 or x[13] >= 15) else 0.0
    x[15] = 1.0
    x[16] = 0.0
    return x

def main(a):
    sm = load_json(a.shard_manifest)
    base = Path(a.shard_manifest).parent
    shards = sm.get('shards', sm.get('shard_infos', []))
    shard_paths = [s['shard_path'] for s in shards] if shards else sm.get('shard_paths', [])
    all_rows, md_diag, g = [], [], 0
    context_presence = []
    sample_ctx_shape = None
    for i, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        meta_df, found = extract_meta(sd)
        row_df = build_rows(meta_df, i, g)
        ctx_path = sd / 'context_traj.npy'
        has_ctx = ctx_path.exists()
        if has_ctx and sample_ctx_shape is None:
            sample_ctx_shape = tuple(np.load(ctx_path, mmap_mode='r').shape)
        context_presence.append({'shard_path': sp, 'context_traj_exists': has_ctx})
        md_diag.append({'shard_path': sp, 'metadata_files': found, 'metadata_columns': list(meta_df.columns), 'rows': int(len(row_df))})
        all_rows.append(row_df)
        g += feat.shape[0]
    rows_df = pd.concat(all_rows, ignore_index=True)
    proc_sids = set(rows_df['scenario_id'].astype(str).tolist())
    raw_maps, raw_scanned, raw_file_count = parse_scenario_maps(Path(a.raw_scenario_dir), proc_sids, a.max_scenarios)
    raw_sids = set(raw_maps.keys())
    overlap = raw_sids & proc_sids
    missing = sorted(list(proc_sids - raw_sids))
    recommendation = 'OK to run debug build'
    if any(not x['context_traj_exists'] for x in context_presence):
        recommendation = 'context_traj missing, cannot compute ego-local ODD'
    elif len(overlap) == 0:
        recommendation = 'raw scenario directory does not match processed shards'
    diag = {
        'shard_count': len(shard_paths),
        'total_processed_rows': int(len(rows_df)),
        'processed_scenario_count': len(proc_sids),
        'sample_processed_scenario_ids': sorted(list(proc_sids))[:10],
        'raw_tfrecord_file_count': raw_file_count,
        'raw_scenarios_scanned': raw_scanned,
        'raw_scenarios_kept': len(raw_sids),
        'scenario_overlap_count': len(overlap),
        'missing_processed_scenario_count': len(missing),
        'sample_missing_processed_scenario_ids': missing[:10],
        'metadata_diagnostics': md_diag,
        'context_traj_presence': context_presence,
        'sample_context_traj_shape': sample_ctx_shape,
        'window_start_exists': bool((rows_df['window_start'] != -1).any()),
        'window_end_exists': bool((rows_df['window_end'] != -1).any()),
        'recommendation': recommendation,
    }
    if a.inspect_metadata:
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        return
    out = Path(a.output_dir)
    if out.exists() and not a.overwrite:
        raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    warnings = defaultdict(int)
    per_shard_valid = {}
    manifest_rows, reports, g0, valid_rows = [], [], 0, 0
    for i, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        meta_df, _ = extract_meta(sd)
        row_df = build_rows(meta_df, i, g0)
        ctx_path = sd / 'context_traj.npy'
        if not ctx_path.exists():
            raise FileNotFoundError(f'{sp} 缺少 context_traj.npy，无法计算 ego-local ODD。')
        ctx = np.load(ctx_path, mmap_mode='r')
        vals = np.zeros((feat.shape[0], len(FEATURES)), dtype=np.float32)
        shard_valid = 0
        for r in range(feat.shape[0]):
            sid = str(row_df.iloc[r]['scenario_id'])
            if sid not in raw_maps:
                warnings['missing_scenario_rows'] += 1
                continue
            md = raw_maps[sid]
            if sum(len(md[k]) for k in md) == 0:
                warnings['no_map_feature_rows'] += 1
                continue
            path_xy, mode, w = extract_ego_path_from_context(ctx[r])
            if w:
                warnings['invalid_context_rows'] += 1
                continue
            if path_xy.shape[0] < 2:
                warnings['empty_path_rows'] += 1
                continue
            vals[r] = compute_features(path_xy, md, warnings)
            vals[r, 15] = 1.0
            shard_valid += 1
            valid_rows += 1
        per_shard_valid[sp] = {'valid_rows': int(shard_valid), 'total_rows': int(feat.shape[0])}
        shard_out = out / Path(sp).name
        shard_out.mkdir(parents=True, exist_ok=True)
        np.save(shard_out / 'map_odd_feat.npy', vals)
        row_df.to_csv(shard_out / 'map_odd_meta.csv', index=False)
        manifest_rows.append({'shard_id': i, 'shard_path': sp, 'feature_path': str((shard_out / 'map_odd_feat.npy').relative_to(out)), 'meta_path': str((shard_out / 'map_odd_meta.csv').relative_to(out)), 'rows': int(feat.shape[0]), 'global_row_start': int(g0), 'global_row_end': int(g0 + feat.shape[0] - 1)})
        reports.append(f'- {sp}: rows={feat.shape[0]}, map_match_valid={shard_valid}')
        g0 += feat.shape[0]
    match_rate = valid_rows / max(1, g0)
    if match_rate < a.min_match_rate and not a.allow_low_match_rate:
        raise RuntimeError(f'map_match_valid 比例过低: {match_rate:.4f} < {a.min_match_rate}')
    warnings['match_rate'] = match_rate
    warnings['per_shard_valid_counts'] = per_shard_valid
    (out / 'map_odd_warnings.json').write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'map_odd_schema.json').write_text(json.dumps({'feature_names': FEATURES}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'map_odd_manifest.json').write_text(json.dumps({'total_rows': g0, 'shards': manifest_rows, 'map_match_valid_rate': match_rate}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'map_odd_diagnostic.json').write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'map_odd_build_report.md').write_text('# Stage6B map ODD features build report\n\n' + '\n'.join(reports) + f'\n\nmatch_rate={match_rate:.6f}\n', encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--raw_scenario_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_scenarios', type=int, default=0)
    p.add_argument('--inspect_metadata', action='store_true')
    p.add_argument('--min_match_rate', type=float, default=0.1)
    p.add_argument('--allow_low_match_rate', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
