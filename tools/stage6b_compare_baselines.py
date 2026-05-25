#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from tools.stage6_compare_unpaired_style import compute_mmd2, _load_json, load_schema


def load_manifest_arrays(shard_manifest, embedding_manifest):
    sm = _load_json(shard_manifest)
    em = _load_json(embedding_manifest)
    base = Path(shard_manifest).parent
    shards = sm.get('shards', sm.get('shard_infos', []))
    shard_paths = [s['shard_path'] for s in shards] if shards else sm.get('shard_paths', [])
    emb_paths = em.get('embedding_shard_paths', [])
    if len(shard_paths) != len(emb_paths):
        raise ValueError('feature/embedding shard 数量不一致')
    feats, embs = [], []
    for sp, ep in zip(shard_paths, emb_paths):
        f = np.load(base / sp / 'interaction_feat_style.npy', mmap_mode='r')
        z = np.load(ep, mmap_mode='r')
        if f.shape[0] != z.shape[0]:
            raise ValueError(f'分片行数不一致: {sp}')
        feats.append(np.asarray(f))
        embs.append(np.asarray(z))
    return np.concatenate(feats, 0), np.concatenate(embs, 0)


def robust_norm(x):
    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    iqr = np.maximum(iqr, 1e-6)
    return (x - med) / iqr, med, iqr


def mmd_with_stats(xa, xb, rng, n_boot, n_perm, max_samples):
    obs = compute_mmd2(xa, xb, rng, max_samples, 1024)
    boots = []
    for _ in range(n_boot):
        ia = rng.choice(len(xa), len(xa), replace=True)
        ib = rng.choice(len(xb), len(xb), replace=True)
        boots.append(compute_mmd2(xa[ia], xb[ib], rng, max_samples, 1024))
    perms = []
    z = np.vstack([xa, xb])
    na = len(xa)
    for _ in range(n_perm):
        p = rng.permutation(len(z))
        perms.append(compute_mmd2(z[p[:na]], z[p[na:]], rng, max_samples, 1024))
    p = float((np.sum(np.asarray(perms) >= obs) + 1) / (n_perm + 1)) if n_perm > 0 else 1.0
    return {
        'mmd2': float(obs),
        'ci95_low': float(np.quantile(boots, 0.025)) if boots else float('nan'),
        'ci95_high': float(np.quantile(boots, 0.975)) if boots else float('nan'),
        'p_value': p,
    }


def perm_pvalue(xa, xb, n_perm, rng):
    obs = float(np.mean(xb) - np.mean(xa))
    z = np.concatenate([xa, xb])
    na = len(xa)
    c = 0
    for _ in range(n_perm):
        p = rng.permutation(z)
        d = float(np.mean(p[na:]) - np.mean(p[:na]))
        if abs(d) >= abs(obs):
            c += 1
    return float((c + 1) / (n_perm + 1))


def main(args):
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    (out / 'plots').mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    feat, emb = load_manifest_arrays(args.shard_manifest, args.embedding_manifest)
    names = load_schema(args.feature_schema_path)
    a_idx = np.load(args.a_indices_path)
    b_idx = np.load(args.b_indices_path)
    xa, xb = emb[a_idx], emb[b_idx]

    emb_stats = mmd_with_stats(xa, xb, rng, args.num_bootstrap, args.num_permutation, args.max_mmd_samples)

    fa_raw, fb_raw = np.asarray(feat[a_idx], float), np.asarray(feat[b_idx], float)
    f_all = np.vstack([fa_raw, fb_raw])
    f_norm_all, med, iqr = robust_norm(f_all)
    na = len(fa_raw)
    fa = f_norm_all[:na]
    fb = f_norm_all[na:]

    feat_stats = mmd_with_stats(fa, fb, rng, args.num_bootstrap, args.num_permutation, args.max_mmd_samples)

    pca_dim = min(args.pca_dim, fa.shape[1], len(f_norm_all))
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca_all = pca.fit_transform(f_norm_all)
    pca_a, pca_b = pca_all[:na], pca_all[na:]
    pca_stats = mmd_with_stats(pca_a, pca_b, rng, args.num_bootstrap, args.num_permutation, args.max_mmd_samples)

    cfg = yaml.safe_load(Path(args.feature_groups_config).read_text(encoding='utf-8'))
    groups = cfg.get('category_groups', {})
    group_map = {}
    for g, v in groups.items():
        for f in v.get('features', []):
            group_map[f] = g

    rows = []
    for i, n in enumerate(names):
        a = fa_raw[:, i]
        b = fb_raw[:, i]
        delta = float(np.nanmean(b) - np.nanmean(a))
        pooled = float(np.nanstd(np.r_[a, b]) + 1e-6)
        delta_norm = float(delta / (float(iqr[i]) if i < len(iqr) else 1.0))
        rows.append({
            'feature': n,
            'category': group_map.get(n, 'ungrouped'),
            'mean_A': float(np.nanmean(a)),
            'mean_B': float(np.nanmean(b)),
            'delta': delta,
            'delta_normalized': delta_norm,
            'cohen_d': float(delta / pooled),
            'permutation_p_value': perm_pvalue(a[np.isfinite(a)], b[np.isfinite(b)], args.num_permutation, rng),
        })
    fdf = pd.DataFrame(rows)
    fdf.to_csv(out / 'feature_mean_delta.csv', index=False)
    top = fdf.reindex(fdf['cohen_d'].abs().sort_values(ascending=False).index).head(15)
    top.to_csv(out / 'top_feature_effects.csv', index=False)

    mmd_rows = [
        {'method': 'embedding_bdd', 'mmd2': emb_stats['mmd2'], 'ci95_low': emb_stats['ci95_low'], 'ci95_high': emb_stats['ci95_high'], 'p_value': emb_stats['p_value'], 'n_A': len(a_idx), 'n_B': len(b_idx), 'dim': xa.shape[1]},
        {'method': 'feature_mmd', 'mmd2': feat_stats['mmd2'], 'ci95_low': feat_stats['ci95_low'], 'ci95_high': feat_stats['ci95_high'], 'p_value': feat_stats['p_value'], 'n_A': len(a_idx), 'n_B': len(b_idx), 'dim': fa.shape[1]},
        {'method': 'pca_feature_mmd', 'mmd2': pca_stats['mmd2'], 'ci95_low': pca_stats['ci95_low'], 'ci95_high': pca_stats['ci95_high'], 'p_value': pca_stats['p_value'], 'n_A': len(a_idx), 'n_B': len(b_idx), 'dim': pca_dim},
    ]
    pd.DataFrame(mmd_rows).to_csv(out / 'baseline_mmd.csv', index=False)

    summary = {
        'embedding_bdd': mmd_rows[0],
        'feature_mmd': mmd_rows[1],
        'pca_feature_mmd': mmd_rows[2],
        'top_feature_effects': top[['feature', 'category', 'cohen_d', 'permutation_p_value']].to_dict(orient='records'),
    }
    (out / 'baseline_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'baseline_report.md').write_text('# Stage6B Baseline Report\n\n详见 baseline_summary.json 与 CSV。\n', encoding='utf-8')

    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.bar([r['method'] for r in mmd_rows], [r['mmd2'] for r in mmd_rows])
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out / 'plots' / 'baseline_mmd_bar.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(top['feature'], top['cohen_d'])
        plt.xticks(rotation=75, ha='right')
        plt.tight_layout()
        plt.savefig(out / 'plots' / 'top_feature_effects.png', dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(pca_a[:1500, 0], pca_a[:1500, 1], s=4, alpha=0.4, label='A')
        plt.scatter(pca_b[:1500, 0], pca_b[:1500, 1], s=4, alpha=0.4, label='B')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / 'plots' / 'pca_feature_embedding.png', dpi=150)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_manifest', required=True)
    p.add_argument('--shard_manifest', required=True)
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--a_indices_path', required=True)
    p.add_argument('--b_indices_path', required=True)
    p.add_argument('--feature_groups_config', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_bootstrap', type=int, default=50)
    p.add_argument('--num_permutation', type=int, default=100)
    p.add_argument('--max_mmd_samples', type=int, default=2000)
    p.add_argument('--pca_dim', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
