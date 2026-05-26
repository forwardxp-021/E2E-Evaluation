#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path
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

def build_rows(df, shard_id, g0):
    cols = set(df.columns)
    scenario_col = next((c for c in ['scenario_id','scenarioId','scenario'] if c in cols), None)
    start_col = next((c for c in ['window_start','start_frame','start_idx'] if c in cols), None)
    end_col = next((c for c in ['window_end','end_frame','end_idx'] if c in cols), None)
    if scenario_col is None:
        raise ValueError(f'元数据缺少 scenario_id 字段，现有字段: {sorted(cols)}')
    n = len(df)
    out = pd.DataFrame({
        'global_row': np.arange(g0, g0+n),
        'shard_id': shard_id,
        'local_row': np.arange(n),
        'scenario_id': df[scenario_col].astype(str),
        'window_start': df[start_col].values if start_col else -1,
        'window_end': df[end_col].values if end_col else -1,
    })
    return out

def main(a):
    out = Path(a.output_dir)
    if out.exists() and not a.overwrite:
        raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    raw = Path(a.raw_scenario_dir)
    tfrecs = list(raw.glob('*.tfrecord*')) if raw.exists() else []
    if not raw.exists() or not tfrecs:
        raise FileNotFoundError('raw_scenario_dir 不存在或无 tfrecord 文件，请提供原始 Waymo scenario tfrecords 目录。')

    sm = load_json(a.shard_manifest)
    base = Path(a.shard_manifest).parent
    shards = sm.get('shards', sm.get('shard_infos', []))
    shard_paths = [s['shard_path'] for s in shards] if shards else sm.get('shard_paths', [])

    warnings = {'metadata_files': [], 'missing_scenario_ids': [], 'note': '当前版本使用 map_match_valid/fallback 标记缺失场景；需接入 Waymo proto 解析后提取真实地图特征。'}
    manifest_rows = []
    reports = []
    g = 0
    for i, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        meta_df, found = extract_meta(sd)
        warnings['metadata_files'].append({'shard': sp, 'files': found, 'columns': list(meta_df.columns)})
        row_df = build_rows(meta_df, i, g)
        if len(row_df) != feat.shape[0]:
            raise ValueError(f'行数不一致: {sp} metadata={len(row_df)} feat={feat.shape[0]}')
        vals = np.zeros((feat.shape[0], len(FEATURES)), dtype=np.float32)
        vals[:, 0] = 1e6
        vals[:, 2] = 1e6
        vals[:, 15] = 0
        vals[:, 16] = 1
        shard_out = out / Path(sp).name
        shard_out.mkdir(parents=True, exist_ok=True)
        np.save(shard_out / 'map_odd_feat.npy', vals)
        row_df.to_csv(shard_out / 'map_odd_meta.csv', index=False)
        manifest_rows.append({'shard_id': i, 'shard_path': sp, 'feature_path': str((shard_out/'map_odd_feat.npy').relative_to(out)), 'meta_path': str((shard_out/'map_odd_meta.csv').relative_to(out)), 'rows': int(feat.shape[0]), 'global_row_start': int(g), 'global_row_end': int(g+feat.shape[0]-1)})
        reports.append(f'- {sp}: rows={feat.shape[0]}, map_match_valid=0 (待接入Waymo proto map parser)')
        g += feat.shape[0]

    (out/'map_odd_schema.json').write_text(json.dumps({'feature_names': FEATURES}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_manifest.json').write_text(json.dumps({'total_rows': g, 'shards': manifest_rows}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_warnings.json').write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'map_odd_build_report.md').write_text('# Stage6B map ODD features build report\n\n'+'\n'.join(reports)+'\n', encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--raw_scenario_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_scenarios', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
