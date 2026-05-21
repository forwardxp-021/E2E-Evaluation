#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main(a):
    d = Path(a.input_dir or a.compare_dir)
    bdd = json.loads((d / 'bdd_summary.json').read_text(encoding='utf-8'))
    c = _safe_read_csv(d / 'category_delta.csv')
    f = _safe_read_csv(d / 'feature_delta.csv')
    s = _safe_read_csv(d / 'scenario_slice_delta.csv')
    t = _safe_read_csv(d / 'top_drift_cases.csv')
    w = json.loads((d / 'stage6_warnings.json').read_text(encoding='utf-8')) if (d / 'stage6_warnings.json').exists() else {'warnings': []}

    topc = c.reindex(c.delta.abs().sort_values(ascending=False).head(10).index) if not c.empty else pd.DataFrame()
    topf = f.reindex(f.delta_normalized.abs().sort_values(ascending=False).head(10).index) if not f.empty else pd.DataFrame()
    if not topf.empty:
        preferred = [c for c in ['feature','delta_normalized','cohen_d','permutation_p_value','group'] if c in topf.columns]
        topf = topf[preferred] if preferred else topf

    lines = [
        '# Style Report Card', '',
        '## Executive Summary',
        f"- BDD(MMD^2): {bdd.get('mmd2', float('nan')):.6f}",
        f"- 95% CI: [{bdd.get('ci95_low', float('nan')):.6f}, {bdd.get('ci95_high', float('nan')):.6f}]",
        f"- permutation p-value: {bdd.get('p_value', float('nan')):.6f}",
        '- BDD 仅衡量嵌入空间分布漂移幅度，不直接代表更保守/更激进。',
        '- 若无 negative/positive control，对 BDD 绝对量纲应视为未标定。', '',
        '## BDD Summary',
        f"- metric: {bdd.get('metric')}",
        f"- n_A: {bdd.get('n_A')}, n_B: {bdd.get('n_B')}, embedding_dim: {bdd.get('embedding_dim')}", '',
        '## Top Category Deltas', topc.to_markdown(index=False) if not topc.empty else '_missing_', '',
        '## Top Feature Deltas', topf.to_markdown(index=False) if not topf.empty else '_missing_', '',
        '## Scenario/Proxy Slice Summary', s.to_markdown(index=False) if not s.empty else '_missing_', '',
        '## Top Drift Cases', t.head(20).to_markdown(index=False) if not t.empty else '_missing_', '',
        '## Warnings', *[f'- {x}' for x in w.get('warnings', [])], '',
        '## Limitations',
        '- BDD measures drift magnitude only; direction requires category/feature/case analysis.',
        '- Unpaired logs may be confounded by scenario/ODD distribution mismatch.',
        '- Embedding space is the primary metric space; category/feature are interpretation layers.',
        '- Negative/positive controls are required for BDD scale calibration.',
    ]
    out_path = Path(a.output_path) if a.output_path else (d / 'style_report_card.md')
    if out_path.exists() and not a.overwrite:
        raise FileExistsError(f'输出文件已存在: {out_path}；如需覆盖请加 --overwrite')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir')
    p.add_argument('--compare_dir')
    p.add_argument('--output_path')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.input_dir and not args.compare_dir:
        raise ValueError('请提供 --input_dir 或 --compare_dir')
    main(args)
