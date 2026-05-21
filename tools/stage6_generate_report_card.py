#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import pandas as pd


def main(a):
    d = Path(a.input_dir)
    bdd = json.loads((d / 'bdd_summary.json').read_text(encoding='utf-8'))
    c = pd.read_csv(d / 'category_delta.csv') if (d / 'category_delta.csv').exists() else pd.DataFrame()
    f = pd.read_csv(d / 'feature_delta.csv') if (d / 'feature_delta.csv').exists() else pd.DataFrame()
    s = pd.read_csv(d / 'scenario_slice_delta.csv') if (d / 'scenario_slice_delta.csv').exists() else pd.DataFrame()
    t = pd.read_csv(d / 'top_drift_cases.csv') if (d / 'top_drift_cases.csv').exists() else pd.DataFrame()
    w = json.loads((d / 'stage6_warnings.json').read_text(encoding='utf-8')) if (d / 'stage6_warnings.json').exists() else {'warnings': []}

    topc = c.reindex(c.delta.abs().sort_values(ascending=False).head(10).index) if not c.empty else pd.DataFrame()
    topf = f.reindex(f.delta_normalized.abs().sort_values(ascending=False).head(10).index) if not f.empty else pd.DataFrame()

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
    (d / 'style_report_card.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    main(p.parse_args())
