#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from tools.stage6_compare_unpaired_style import compute_mmd2, _load_json, _resolve_alias, _build_tertile_bins, load_schema
from tools.stage6b_compare_baselines import load_manifest_arrays, mmd_with_stats


def main(a):
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

    ba, bb = [], []
    rows = []
    for i, label in enumerate(['low', 'mid', 'high']):
        ai = a_idx[bins[a_idx] == label]
        bi = b_idx[bins[b_idx] == label]
        n = min(len(ai), len(bi))
        used = n >= a.min_bin_size
        if used:
            sa = rng.choice(ai, n, replace=False)
            sb = rng.choice(bi, n, replace=False)
            ba.append(sa)
            bb.append(sb)
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
    }
    (out / 'balanced_bdd_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'balanced_report.md').write_text('# Stage6B Scenario-balanced BDD\n\n详见 balanced_bdd_summary.json。\n', encoding='utf-8')

    if plt is not None:
        x = np.arange(len(table))
        w = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(x - w/2, table['n_A_before'], width=w, label='A_before')
        plt.bar(x + w/2, table['n_B_before'], width=w, label='B_before')
        plt.xticks(x, table['bin'])
        plt.legend(); plt.tight_layout()
        plt.savefig(out / 'plots' / 'bin_counts_before_after.png', dpi=150)
        plt.close()
    
        plt.figure(figsize=(5, 4))
        plt.bar(['raw_bdd', 'balanced_bdd'], [raw['mmd2'], balanced['mmd2']])
        plt.tight_layout()
        plt.savefig(out / 'plots' / 'raw_vs_balanced_bdd.png', dpi=150)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_manifest', required=True)
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--a_indices_path', required=True)
    p.add_argument('--b_indices_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--balance_keys', default='lateral_activity_bin')
    p.add_argument('--num_bootstrap', type=int, default=50)
    p.add_argument('--num_permutation', type=int, default=100)
    p.add_argument('--max_mmd_samples', type=int, default=2000)
    p.add_argument('--min_bin_size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
