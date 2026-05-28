#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
from tools.stage6_compare_unpaired_style import _resolve_alias, _build_tertile_bins, load_schema
from tools.stage6b_compare_baselines import load_manifest_arrays, mmd_with_stats


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
    return get_tqdm()(iterable, **kwargs)


def main(a):
    t0=time.time()
    progress_enabled = not a.no_progress
    out = Path(a.output_dir)
    if out.exists() and not a.overwrite:
        raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    (out / 'plots').mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    feat, emb = load_manifest_arrays(a.shard_manifest, a.embedding_manifest)
    names = load_schema(a.feature_schema_path)
    fmap = {n: i for i, n in enumerate(names)}

    a_idx = np.load(a.a_indices_path)
    b_idx = np.load(a.b_indices_path)
    raw = mmd_with_stats(emb[a_idx], emb[b_idx], rng, a.num_bootstrap, a.num_permutation, a.max_mmd_samples)

    if a.odd_bins_path:
        odd = pd.read_csv(a.odd_bins_path)
        if 'global_row' not in odd.columns:
            raise ValueError('odd_bins.csv must include global_row')
        if 'map_match_valid' not in odd.columns:
            raise ValueError('odd_bins.csv 缺少 map_match_valid 列，ODD bins 无法验证有效性。')
        if not a.allow_invalid_map:
            odd = odd[odd['map_match_valid'] == 1]
        if len(odd) == 0:
            raise ValueError('ODD bins are invalid; run stage6b_build_map_odd_features with real map parsing.')
        keys = [k.strip() for k in a.balance_keys.split(',') if k.strip()]
        missing = [k for k in keys if k not in odd.columns]
        if missing:
            raise ValueError(f'balance keys missing in odd bins: {missing}')
        ta = odd[odd.global_row.isin(a_idx)]
        tb = odd[odd.global_row.isin(b_idx)]
        ba, bb, rows = [], [], []
        gb = tb.groupby(keys)
        for kval, ga in iter_progress(list(ta.groupby(keys)), enabled=progress_enabled, desc="balance ODD bins", unit="bin"):

            if kval not in gb.groups:
                rows.append({'bin': str(kval), 'n_A_before': len(ga), 'n_B_before': 0, 'n_used': 0, 'used': False})
                continue
            g2 = gb.get_group(kval)
            n = min(len(ga), len(g2))
            used = n >= a.min_bin_size
            if used:
                ba.append(rng.choice(ga.global_row.values, n, replace=False))
                bb.append(rng.choice(g2.global_row.values, n, replace=False))
            rows.append({'bin': str(kval), 'n_A_before': len(ga), 'n_B_before': len(g2), 'n_used': int(n if used else 0), 'used': used})
        key = a.balance_keys
        fname = 'odd_bins'
    else:
        key_cfg = {
            'lateral_activity_bin': ['lane_change_count_proxy','lane_change_left_count_proxy','lane_change_right_count_proxy','rms_yaw_rate','rms_curvature','heading_change_total'],
            'speed_bin': ['speed_mean','ego_speed_mean','speed_norm_mean','mean_speed'],
            'thw_bin': ['mean_thw','thw_mean','min_thw','thw_min'],
            'interaction_bin': ['interaction_density','neighbor_count','neighbor_valid_count','front_valid_ratio','front_vehicle_valid_ratio'],
        }
        key = a.balance_keys
        if key not in key_cfg:
            raise ValueError(f'unsupported balance_key: {key}')
        fname = _resolve_alias(fmap, key_cfg[key])
        if fname is None:
            raise ValueError(f'{key} 无可用 proxy 特征')
        bins, meta = _build_tertile_bins(np.asarray(feat[:, fmap[fname]], float), ['low', 'mid', 'high'])
        if bins is None:
            raise ValueError(f'{key} 无法构建分箱: {meta}')
        ba, bb, rows = [], [], []
        for label in ['low', 'mid', 'high']:
            ai = a_idx[bins[a_idx] == label]
            bi = b_idx[bins[b_idx] == label]
            n = min(len(ai), len(bi))
            used = n >= a.min_bin_size
            if used:
                ba.append(rng.choice(ai, n, replace=False))
                bb.append(rng.choice(bi, n, replace=False))
            rows.append({'bin': label, 'n_A_before': len(ai), 'n_B_before': len(bi), 'n_used': int(n if used else 0), 'used': used})

    bal_a = np.concatenate(ba) if ba else np.array([], dtype=int)
    bal_b = np.concatenate(bb) if bb else np.array([], dtype=int)
    if len(bal_a) == 0 or len(bal_b) == 0:
        raise ValueError('平衡后样本为空，请降低 --min_bin_size')

    balanced = mmd_with_stats(emb[bal_a], emb[bal_b], rng, a.num_bootstrap, a.num_permutation, a.max_mmd_samples)
    reduction = (raw['mmd2'] - balanced['mmd2']) / raw['mmd2'] * 100.0 if raw['mmd2'] > 0 else 0.0
    table = pd.DataFrame(rows)
    table.to_csv(out / 'bin_balance_table.csv', index=False)
    np.save(out / 'balanced_indices_A.npy', bal_a)
    np.save(out / 'balanced_indices_B.npy', bal_b)

    summary = {
        'balance_key': key,
        'resolved_feature': fname,
        'raw_bdd': raw,
        'balanced_bdd': balanced,
        'reduction_percent': float(reduction),
        'n_A_raw': int(len(a_idx)),
        'n_B_raw': int(len(b_idx)),
        'n_A_balanced': int(len(bal_a)),
        'n_B_balanced': int(len(bal_b)),
        'bins_used': table[table['used']]['bin'].tolist(),
        'bins_skipped': table[~table['used']]['bin'].tolist(),
        'num_bins': int(len(table)),
        'total_runtime_seconds': float(time.time()-t0),
    }
    (out / 'balanced_bdd_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'balanced_report.md').write_text('# Stage6B Scenario-balanced BDD\n\n详见 balanced_bdd_summary.json。\n', encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_manifest', required=True)
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--a_indices_path', required=True)
    p.add_argument('--b_indices_path', required=True)
    p.add_argument('--odd_bins_path', default='')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--allow_invalid_map', action='store_true')
    p.add_argument('--balance_keys', default='lateral_activity_bin')
    p.add_argument('--num_bootstrap', type=int, default=50)
    p.add_argument('--num_permutation', type=int, default=100)
    p.add_argument('--max_mmd_samples', type=int, default=2000)
    p.add_argument('--min_bin_size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no_progress', action='store_true')
    main(p.parse_args())
