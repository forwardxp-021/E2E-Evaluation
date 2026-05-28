#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

FEATURES = [
    'distance_to_crosswalk_min','has_crosswalk_near_30m','distance_to_stop_sign_min','has_stop_sign_near_40m',
    'lane_curvature_mean','lane_curvature_max','lane_heading_change_total','lane_count_near_30m','road_line_count_near_30m',
    'road_edge_count_near_30m','crosswalk_count_near_30m','stop_sign_count_near_40m','speed_bump_count_near_30m',
    'map_complexity_score','intersection_proxy','map_match_valid','fallback_full_scenario_path'
]

def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def extract_meta(shard_dir: Path):
    cands = [shard_dir/'metadata.csv', shard_dir/'meta.csv', shard_dir/'meta.npy']
    found = [str(p) for p in cands if p.exists()]
    if (shard_dir/'metadata.csv').exists():
        df = pd.read_csv(shard_dir/'metadata.csv')
    elif (shard_dir/'meta.csv').exists():
        df = pd.read_csv(shard_dir/'meta.csv')
    elif (shard_dir/'meta.npy').exists():
        arr = np.load(shard_dir/'meta.npy', allow_pickle=True)
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
    scenario_col = pick_col(cols, ['scenario_id','scenarioId','scenario'])
    start_col = pick_col(cols, ['window_start','start_frame','start_idx'])
    end_col = pick_col(cols, ['window_end','end_frame','end_idx'])
    if scenario_col is None:
        raise ValueError(f'元数据缺少 scenario_id 字段，现有字段: {sorted(cols)}')
    n = len(df)
    return pd.DataFrame({
        'global_row': np.arange(g0, g0+n),
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
        raise RuntimeError(
            'Waymo proto 解析器不可用。请安装 tensorflow + waymo-open-dataset-tf-*，并确保可导入 '\
            'waymo_open_dataset.protos.scenario_pb2。原始错误: ' + str(e)
        ) from e

def _polyline_points(polyline):
    return np.array([[p.x, p.y] for p in polyline], dtype=np.float32) if polyline else np.zeros((0,2), dtype=np.float32)

def parse_scenario_maps(raw_dir: Path, max_scenarios: int=0):
    tf, scenario_pb2 = load_waymo_parser()
    tfrecs = sorted(raw_dir.glob('*.tfrecord*'))
    if not tfrecs:
        raise FileNotFoundError(f'raw_scenario_dir 下未找到 tfrecord: {raw_dir}')
    out = {}
    total = 0
    for tfr in tfrecs:
        ds = tf.data.TFRecordDataset([str(tfr)], compression_type='')
        for rec in ds:
            sc = scenario_pb2.Scenario()
            sc.ParseFromString(bytes(rec.numpy()))
            sid = sc.scenario_id
            d = {'lane':[], 'road_line':[], 'road_edge':[], 'crosswalk':[], 'stop_sign':[], 'speed_bump':[]}
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
            total += 1
            if max_scenarios > 0 and total >= max_scenarios:
                return out
    return out

def min_dist_to_shapes(path_xy, shapes):
    if len(shapes) == 0 or path_xy.size == 0:
        return np.inf
    pts = np.concatenate([s for s in shapes if s.size > 0], axis=0) if any(s.size > 0 for s in shapes) else np.zeros((0,2), dtype=np.float32)
    if pts.size == 0:
        return np.inf
    d2 = ((path_xy[:,None,:] - pts[None,:,:])**2).sum(axis=2)
    return float(np.sqrt(d2.min()))

def lane_curvature_stats(lanes):
    curvs=[]
    heading_total=0.0
    for ln in lanes:
        if ln.shape[0] < 3:
            continue
        d = np.diff(ln, axis=0)
        headings = np.unwrap(np.arctan2(d[:,1], d[:,0]))
        dh = np.abs(np.diff(headings))
        if dh.size:
            curvs.extend(dh.tolist())
            heading_total += float(dh.sum())
    if not curvs:
        return 0.0,0.0,0.0
    return float(np.mean(curvs)), float(np.max(curvs)), heading_total

def compute_features(path_xy, md):
    x = np.zeros((len(FEATURES),), dtype=np.float32)
    d_cross = min_dist_to_shapes(path_xy, md['crosswalk'])
    d_stop = min_dist_to_shapes(path_xy, md['stop_sign'])
    lmean,lmax,lchg = lane_curvature_stats(md['lane'])
    x[0]=d_cross; x[1]=1.0 if d_cross<=30.0 else 0.0
    x[2]=d_stop; x[3]=1.0 if d_stop<=40.0 else 0.0
    x[4]=lmean; x[5]=lmax; x[6]=lchg
    x[7]=len(md['lane']); x[8]=len(md['road_line']); x[9]=len(md['road_edge'])
    x[10]=len(md['crosswalk']); x[11]=len(md['stop_sign']); x[12]=len(md['speed_bump'])
    x[13]=x[7]+x[8]+x[9]+x[10]*2+x[11]*2+x[12]*1.5
    x[14]=1.0 if (x[10]+x[11]>=1 or x[7]>=6) else 0.0
    x[15]=1.0; x[16]=0.0
    return x

def main(a):
    sm = load_json(a.shard_manifest)
    base = Path(a.shard_manifest).parent
    shards = sm.get('shards', sm.get('shard_infos', []))
    shard_paths = [s['shard_path'] for s in shards] if shards else sm.get('shard_paths', [])
    if not shard_paths:
        raise ValueError('shard_manifest 中未找到 shards/shard_paths')

    all_rows=[]; g=0; md_diag=[]
    for i, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        meta_df, found = extract_meta(sd)
        row_df = build_rows(meta_df, i, g)
        if len(row_df) != feat.shape[0]:
            raise ValueError(f'行数不一致: {sp} metadata={len(row_df)} feat={feat.shape[0]}')
        md_diag.append({'shard_path': sp, 'metadata_files': found, 'metadata_columns': list(meta_df.columns), 'rows': int(len(row_df))})
        all_rows.append(row_df)
        g += feat.shape[0]
    rows_df = pd.concat(all_rows, ignore_index=True)

    raw_dir = Path(a.raw_scenario_dir)
    raw_maps = parse_scenario_maps(raw_dir, a.max_scenarios)
    raw_sids = set(raw_maps.keys())
    proc_sids = set(rows_df['scenario_id'].astype(str).tolist())
    overlap = raw_sids & proc_sids

    diag = {
        'shard_paths': shard_paths,
        'metadata_diagnostics': md_diag,
        'scenario_id_exists': True,
        'window_start_exists': bool((rows_df['window_start'] != -1).any()),
        'window_end_exists': bool((rows_df['window_end'] != -1).any()),
        'raw_scenario_count': len(raw_sids),
        'processed_scenario_count': len(proc_sids),
        'scenario_overlap_count': len(overlap),
        'sample_raw_scenario_ids': sorted(list(raw_sids))[:10],
        'sample_processed_scenario_ids': sorted(list(proc_sids))[:10],
        'sample_overlap_scenario_ids': sorted(list(overlap))[:10],
    }
    if a.inspect_metadata:
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        return

    if len(overlap) == 0:
        raise RuntimeError('processed metadata 与 raw scenario 无 scenario_id overlap，拒绝生成伪 ODD。请检查 ID 映射。')

    out = Path(a.output_dir)
    if out.exists() and not a.overwrite:
        raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_rows=[]; reports=[]; g0=0; valid_rows=0
    for i, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        meta_df, _ = extract_meta(sd)
        row_df = build_rows(meta_df, i, g0)
        ctx = np.load(sd / 'context_traj.npy', mmap_mode='r') if (sd / 'context_traj.npy').exists() else None
        vals = np.zeros((feat.shape[0], len(FEATURES)), dtype=np.float32)
        for r in range(feat.shape[0]):
            sid = str(row_df.iloc[r]['scenario_id'])
            if sid not in raw_maps:
                continue
            if ctx is None:
                raise FileNotFoundError(f'{sp} 缺少 context_traj.npy，无法恢复 ego path，拒绝 full-scenario 全回退。')
            ego = np.asarray(ctx[r])
            if ego.ndim != 2 or ego.shape[1] < 2:
                raise ValueError(f'{sp} context_traj row shape 非法: {ego.shape}')
            path_xy = ego[:,:2]
            vals[r] = compute_features(path_xy, raw_maps[sid])
            valid_rows += 1
        shard_out = out / Path(sp).name
        shard_out.mkdir(parents=True, exist_ok=True)
        np.save(shard_out / 'map_odd_feat.npy', vals)
        row_df.to_csv(shard_out / 'map_odd_meta.csv', index=False)
        manifest_rows.append({'shard_id': i, 'shard_path': sp, 'feature_path': str((shard_out/'map_odd_feat.npy').relative_to(out)), 'meta_path': str((shard_out/'map_odd_meta.csv').relative_to(out)), 'rows': int(feat.shape[0]), 'global_row_start': int(g0), 'global_row_end': int(g0+feat.shape[0]-1)})
        reports.append(f'- {sp}: rows={feat.shape[0]}, map_match_valid={(vals[:,15]>0.5).sum()}')
        g0 += feat.shape[0]

    match_rate = valid_rows / max(1, g0)
    if match_rate < a.min_match_rate and not a.allow_low_match_rate:
        raise RuntimeError(f'map_match_valid 比例过低: {match_rate:.4f} < {a.min_match_rate}. 如确认可接受请加 --allow_low_match_rate。')

    (out/'map_odd_schema.json').write_text(json.dumps({'feature_names': FEATURES}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_manifest.json').write_text(json.dumps({'total_rows': g0, 'shards': manifest_rows, 'map_match_valid_rate': match_rate}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_diagnostic.json').write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_build_report.md').write_text('# Stage6B map ODD features build report\n\n'+'\n'.join(reports)+f'\n\nmatch_rate={match_rate:.6f}\n', encoding='utf-8')

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
