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


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main(a):
    out = Path(a.output_dir)
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    (out / 'plots').mkdir(parents=True, exist_ok=True)

    rows = {}
    for root in a.experiment_roots:
        r = Path(root)
        exp = r.name
        rows.setdefault(exp, {'experiment': exp})
        if (r / 'bdd_summary.json').exists():
            b = jload(r / 'bdd_summary.json')
            rows[exp].update({'embedding_bdd': b.get('mmd2'), 'embedding_p_value': b.get('p_value'), 'n_A': b.get('n_A'), 'n_B': b.get('n_B')})
        if (r / 'baseline_summary.json').exists():
            b = jload(r / 'baseline_summary.json')
            rows[exp].update({'embedding_bdd': b['embedding_bdd']['mmd2'], 'embedding_p_value': b['embedding_bdd']['p_value'], 'feature_mmd': b['feature_mmd']['mmd2'], 'feature_mmd_p_value': b['feature_mmd']['p_value'], 'pca_feature_mmd': b['pca_feature_mmd']['mmd2'], 'pca_feature_mmd_p_value': b['pca_feature_mmd']['p_value'], 'n_A': b['embedding_bdd']['n_A'], 'n_B': b['embedding_bdd']['n_B']})
            t = b.get('top_feature_effects', [])
            if t:
                rows[exp].update({'top_feature': t[0].get('feature'), 'top_category': t[0].get('category')})
        if (r / 'balanced_bdd_summary.json').exists():
            b = jload(r / 'balanced_bdd_summary.json')
            rows[exp].update({'raw_bdd': b['raw_bdd']['mmd2'], 'balanced_bdd': b['balanced_bdd']['mmd2'], 'balanced_reduction_percent': b['reduction_percent']})

    for e, d in rows.items():
        if 'negative_control' in e:
            d['interpretation'] = 'same-distribution sanity check; no false drift expected'
        elif 'pseudo' in e:
            d['interpretation'] = 'known pseudo style shift; embedding and feature baselines should detect drift'
        elif 'scene' in e:
            d['interpretation'] = 'proxy scene shift; high BDD should be treated as confounding warning, not pure style drift'
        else:
            d['interpretation'] = ''

    df = pd.DataFrame(rows.values()).sort_values('experiment')
    df.to_csv(out / 'stage6b_calibration_table.csv', index=False)
    (out / 'stage6b_summary_report.md').write_text('# Stage6B Summary\n\n详见 stage6b_calibration_table.csv。\n', encoding='utf-8')

    if plt is not None:
        if 'embedding_bdd' in df.columns:
            plt.figure(figsize=(8, 4))
            plt.bar(df['experiment'], df['embedding_bdd'])
            plt.xticks(rotation=30, ha='right'); plt.tight_layout()
            plt.savefig(out / 'plots' / 'bdd_across_experiments.png', dpi=150); plt.close()
    
        subset = df.dropna(subset=['feature_mmd', 'pca_feature_mmd'], how='all') if 'feature_mmd' in df.columns else pd.DataFrame()
        if not subset.empty:
            x = np.arange(len(subset)); w = 0.25
            plt.figure(figsize=(8, 4))
            plt.bar(x-w, subset['embedding_bdd'], width=w, label='embedding_bdd')
            plt.bar(x, subset['feature_mmd'], width=w, label='feature_mmd')
            plt.bar(x+w, subset['pca_feature_mmd'], width=w, label='pca_feature_mmd')
            plt.xticks(x, subset['experiment'], rotation=30, ha='right'); plt.legend(); plt.tight_layout()
            plt.savefig(out / 'plots' / 'baseline_methods_across_experiments.png', dpi=150); plt.close()
    
        sb = df.dropna(subset=['raw_bdd', 'balanced_bdd'], how='any') if 'raw_bdd' in df.columns else pd.DataFrame()
        if not sb.empty:
            x = np.arange(len(sb)); w = 0.35
            plt.figure(figsize=(7, 4))
            plt.bar(x-w/2, sb['raw_bdd'], width=w, label='raw_bdd')
            plt.bar(x+w/2, sb['balanced_bdd'], width=w, label='balanced_bdd')
            plt.xticks(x, sb['experiment'], rotation=20, ha='right'); plt.legend(); plt.tight_layout()
            plt.savefig(out / 'plots' / 'raw_vs_balanced_bdd.png', dpi=150)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_roots', nargs='+', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
