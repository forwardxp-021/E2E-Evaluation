#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd


def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x
        return tqdm


def iter_progress(iterable, enabled=True, **kwargs):
    if not enabled:
        return iterable
    tqdm = get_tqdm()
    return tqdm(iterable, **kwargs)


def qbin(vals, labels, valid_mask):
    ok = np.isfinite(vals) & valid_mask
    if ok.sum() < len(labels):
        return np.array(['unknown'] * len(vals), dtype=object), {'collapsed': True}
    qs = np.quantile(vals[ok], np.linspace(0, 1, len(labels) + 1)[1:-1])
    if len(np.unique(qs)) < len(qs):
        return np.array(['unknown'] * len(vals), dtype=object), {'collapsed': True, 'quantiles': qs.tolist()}
    out = np.array(['unknown'] * len(vals), dtype=object)
    edges = [-np.inf] + list(qs) + [np.inf]
    for i, l in enumerate(labels):
        out[ok & (vals >= edges[i]) & (vals < edges[i + 1])] = l
    return out, {'collapsed': False, 'quantiles': qs.tolist()}


def main(a):
    t0 = time.time()
    progress_enabled = not a.no_progress
    out = Path(a.output_dir)
    if out.exists() and not a.overwrite:
        raise FileExistsError('output_dir exists')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    mm = json.loads(Path(a.map_odd_manifest).read_text(encoding='utf-8'))
    root = Path(a.map_odd_manifest).parent
    rows, feats = [], []
    feature_paths = [s['feature_path'] for s in mm['shards']]
    meta_paths = [s['meta_path'] for s in mm['shards']]
    if len(set(feature_paths)) != len(feature_paths):
        raise RuntimeError('map_odd_manifest 中 feature_path 存在重复')
    if len(set(meta_paths)) != len(meta_paths):
        raise RuntimeError('map_odd_manifest 中 meta_path 存在重复')
    total_loaded_rows = 0
    for s in iter_progress(mm['shards'], enabled=progress_enabled, desc='load map ODD shards', unit='shard'):
        feat = np.load(root / s['feature_path'], mmap_mode='r')
        meta = pd.read_csv(root / s['meta_path'])
        if feat.shape[0] != len(meta):
            raise RuntimeError(f'特征/元数据行数不一致: {s["feature_path"]}={feat.shape[0]} vs {s["meta_path"]}={len(meta)}')
        total_loaded_rows += int(feat.shape[0])
        feats.append(feat)
        rows.append(meta[['global_row', 'shard_id', 'local_row']])
        if feat.shape[1] <= 15:
            raise ValueError('map_odd_feat.npy 缺少 map_match_valid 列(索引15)')
    if total_loaded_rows != int(mm['total_rows']):
        raise RuntimeError(f'manifest total_rows 不一致: loaded={total_loaded_rows} expected={mm["total_rows"]}')

    print('concatenating map ODD features ...')
    X = np.concatenate(feats, 0)
    M = pd.concat(rows, ignore_index=True)
    valid_mask = X[:, 15] > 0.5
    global_match_rate = float(valid_mask.mean())
    valid_count = int(valid_mask.sum())
    invalid_count = int((~valid_mask).sum())

    if global_match_rate < a.min_map_match_rate and not a.allow_low_match_rate:
        raise RuntimeError(f'map_match_valid 比例过低: {global_match_rate:.4f} < {a.min_map_match_rate}，拒绝构建 ODD bins。')

    print('building ODD bins ...')
    cross = np.where(valid_mask, np.where(X[:, 1] > 0.5, 'crosswalk_near', 'no_crosswalk_near'), 'unknown')
    stop = np.where(valid_mask, np.where(X[:, 3] > 0.5, 'stop_sign_near', 'no_stop_sign_near'), 'unknown')
    curv, w1 = qbin(X[:, 5], ['straight', 'mild_curve', 'sharp_curve'], valid_mask)
    inter = np.where(valid_mask, np.where(X[:, 14] > 0.5, 'intersection_like', 'non_intersection_like'), 'unknown')
    comp, w2 = qbin(X[:, 13], ['low', 'mid', 'high'], valid_mask)
    lane, w3 = qbin(X[:, 7], ['simple_lane_context', 'multi_lane_context', 'dense_lane_context'], valid_mask)

    df = M.copy()
    df['odd_crosswalk_bin'] = cross
    df['odd_stop_sign_bin'] = stop
    df['odd_curvature_bin'] = curv
    df['odd_intersection_bin'] = inter
    df['odd_map_complexity_bin'] = comp
    df['odd_lane_count_bin'] = lane
    df['map_match_valid'] = X[:, 15].astype(int)
    df['fallback_full_scenario_path'] = X[:, 16].astype(int)
    if X.shape[0] != int(mm['total_rows']):
        raise RuntimeError(f'拼接后行数不一致: bins_rows={X.shape[0]} expected={mm["total_rows"]}')

    df.to_csv(out / 'odd_bins.csv', index=False)
    np.save(out / 'odd_bins.npy', df.to_records(index=False))

    cols = ['odd_crosswalk_bin', 'odd_stop_sign_bin', 'odd_curvature_bin', 'odd_intersection_bin', 'odd_map_complexity_bin', 'odd_lane_count_bin']
    valid_df = df[df['map_match_valid'] == 1]
    valid_counts = {c: valid_df[c].value_counts(dropna=False).to_dict() for c in cols}
    counts = {c: df[c].value_counts(dropna=False).to_dict() for c in cols}

    degenerate = {
        'odd_map_complexity_bin_all_unknown': bool((df['odd_map_complexity_bin'] == 'unknown').all()),
        'odd_lane_count_bin_all_unknown': bool((df['odd_lane_count_bin'] == 'unknown').all()),
    }
    if (degenerate['odd_map_complexity_bin_all_unknown'] or degenerate['odd_lane_count_bin_all_unknown']) and not a.allow_degenerate_bins:
        raise RuntimeError(f'关键 ODD bins 退化: {degenerate}')
    local_lane_match_valid_rate = float(mm.get('local_lane_match_valid_rate', 0.0))
    (out / 'odd_bin_warnings.json').write_text(json.dumps({'curvature': w1, 'complexity': w2, 'lane_count': w3, 'degenerate': degenerate, 'row_count_validation_passed': True}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'odd_bin_schema.json').write_text(json.dumps({'columns': df.columns.tolist()}, indent=2, ensure_ascii=False), encoding='utf-8')
    runtime_sec = time.time() - t0
    report = '# Stage6B ODD bins\n\n'
    report += f'- valid row count: {valid_count}\n'
    report += f'- invalid row count: {invalid_count}\n'
    report += f'- global match rate: {global_match_rate:.6f}\n'
    report += f'- local_lane_match_valid_rate: {local_lane_match_valid_rate:.6f}\n'
    report += f'- row count validation: PASS ({X.shape[0]} == {mm["total_rows"]})\n'
    report += f'- key bin degenerate: {degenerate}\n'
    report += f'- total runtime seconds: {runtime_sec:.3f}\n\n'
    report += '## bin distributions among valid rows\n\n' + json.dumps(valid_counts, ensure_ascii=False, indent=2)
    report += '\n\n## bin distributions overall\n\n' + json.dumps(counts, ensure_ascii=False, indent=2)
    (out / 'odd_bin_report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--map_odd_manifest', required=True)
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--min_map_match_rate', type=float, default=0.1)
    p.add_argument('--allow_low_match_rate', action='store_true')
    p.add_argument('--no_progress', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow_degenerate_bins', action='store_true')
    main(p.parse_args())
