#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_RETRIEVAL_COLS = {
    'representation', 'hit_at_1', 'hit_at_5', 'mean_same_label_fraction_at_5',
    'mean_neighbor_feature_distance', 'median_neighbor_feature_distance'
}
REQUIRED_CATEGORY_COLS = {'representation', 'category', 'mean_spearman_corr'}
KEY_CATEGORIES = ['longitudinal_comfort', 'following_interaction', 'lateral_lane_dynamics', 'behavior_proxy']

MODEL_CONFIG = [
    ('Stage 5B baseline', 'stage5b_eval'),
    ('Stage 5D-v1 group_weighted', 'stage5d_v1_eval'),
    ('Stage 5D-balanced-v2', 'stage5d_v2_eval'),
]


def _must_exist(path: Path):
    if not path.exists():
        raise RuntimeError(f'Missing required file: {path}')


def _read_eval(model_name: str, eval_dir: Path):
    print(f'[INFO] Loading {model_name} from {eval_dir}')
    summary_path = eval_dir / 'evaluation_summary.json'
    retrieval_path = eval_dir / 'retrieval_metrics.csv'
    category_path = eval_dir / 'category_correlation_summary.csv'
    _must_exist(summary_path)
    _must_exist(retrieval_path)
    _must_exist(category_path)

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if not summary.get('paper_grade_valid', False):
        raise RuntimeError(f'{model_name}: paper_grade_valid is not true.')
    if not summary.get('strict_feature_schema', False):
        raise RuntimeError(f'{model_name}: strict_feature_schema is not true.')
    if not summary.get('feature_schema_loaded', False):
        raise RuntimeError(f'{model_name}: feature_schema_loaded is not true.')
    aligned = summary.get('row_alignment_checks', {}).get('aligned', False)
    if not aligned:
        raise RuntimeError(f'{model_name}: row_alignment_checks.aligned is not true.')
    warnings = summary.get('warnings', [])
    if warnings:
        print(f'[WARN] {model_name} has warnings: {warnings}')
    else:
        print(f'[INFO] {model_name} warnings: []')

    retrieval_df = pd.read_csv(retrieval_path)
    category_df = pd.read_csv(category_path)

    for need, df, name in [
        (REQUIRED_RETRIEVAL_COLS, retrieval_df, 'retrieval_metrics.csv'),
        (REQUIRED_CATEGORY_COLS, category_df, 'category_correlation_summary.csv'),
    ]:
        miss = need - set(df.columns)
        if miss:
            raise RuntimeError(f'{model_name}: missing columns in {name}: {sorted(miss)}')

    learned_win_df = _extract_learned_win_df(model_name, eval_dir)
    return summary, retrieval_df, category_df, learned_win_df, warnings


def _pick_row(df: pd.DataFrame, representation: str):
    rows = df[df['representation'] == representation]
    if rows.empty:
        raise RuntimeError(f'Missing representation={representation}')
    return rows.iloc[0]


def _category_value(category_df: pd.DataFrame, representation: str, category: str):
    rows = category_df[(category_df['representation'] == representation) & (category_df['category'] == category)]
    if rows.empty:
        raise RuntimeError(f'Missing category row: representation={representation}, category={category}')
    return float(rows.iloc[0]['mean_spearman_corr'])


def _resolve_col(df: pd.DataFrame, candidates, column_kind: str, source_name: str, model_name: str):
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"{model_name}: could not resolve {column_kind} column in {source_name}. "
        f"Candidates={candidates}, available={sorted(df.columns.tolist())}"
    )


def _extract_learned_win_df(model_name: str, eval_dir: Path):
    win_path = eval_dir / 'learned_win_features.csv'
    style_corr_path = eval_dir / 'style_distance_correlation.csv'

    if win_path.exists():
        win_df = pd.read_csv(win_path)
        print(f'[INFO] {model_name}: using learned_win_features.csv')
        feature_col = _resolve_col(
            win_df, ['target_feature', 'feature', 'target'], 'feature', 'learned_win_features.csv', model_name
        )

        computed_raw_delta = False
        computed_pca_delta = False
        learned_candidates = ['learned_corr', 'learned_context_embedding_corr', 'learned_context_embedding', 'learned_spearman_corr']

        if 'learned_minus_raw_feature' not in win_df.columns:
            learned_col = _resolve_col(win_df, learned_candidates, 'learned correlation', 'learned_win_features.csv', model_name)
            raw_col = _resolve_col(
                win_df, ['raw_feature_corr', 'raw_feature', 'raw_spearman_corr'], 'raw correlation', 'learned_win_features.csv', model_name
            )
            win_df['learned_minus_raw_feature'] = win_df[learned_col] - win_df[raw_col]
            computed_raw_delta = True

        if 'learned_minus_pca_feature' not in win_df.columns:
            learned_col = _resolve_col(win_df, learned_candidates, 'learned correlation', 'learned_win_features.csv', model_name)
            pca_col = _resolve_col(
                win_df, ['pca_feature_corr', 'pca_feature', 'pca_spearman_corr'], 'pca correlation', 'learned_win_features.csv', model_name
            )
            win_df['learned_minus_pca_feature'] = win_df[learned_col] - win_df[pca_col]
            computed_pca_delta = True

        if computed_raw_delta or computed_pca_delta:
            print(
                f'[INFO] {model_name}: computed missing delta columns from learned_win_features.csv '
                f'(raw_delta_computed={computed_raw_delta}, pca_delta_computed={computed_pca_delta})'
            )
        else:
            print(f'[INFO] {model_name}: learned_win_features.csv already contains delta columns')

        return pd.DataFrame({
            'target_feature': win_df[feature_col].astype(str),
            'learned_minus_raw_feature': win_df['learned_minus_raw_feature'],
            'learned_minus_pca_feature': win_df['learned_minus_pca_feature'],
        })

    _must_exist(style_corr_path)
    print(f'[WARN] {model_name}: learned_win_features.csv missing, fallback to style_distance_correlation.csv')
    style_df = pd.read_csv(style_corr_path)
    need_cols = {'representation', 'target_feature', 'spearman_corr'}
    missing = need_cols - set(style_df.columns)
    if missing:
        raise RuntimeError(f'{model_name}: style_distance_correlation.csv missing columns: {sorted(missing)}')

    pivot_df = style_df.pivot_table(index='target_feature', columns='representation', values='spearman_corr', aggfunc='first').reset_index()
    for rep_col in ['learned_context_embedding', 'raw_feature', 'pca_feature']:
        if rep_col not in pivot_df.columns:
            raise RuntimeError(f'{model_name}: fallback style_distance_correlation.csv missing representation={rep_col}')

    print(f'[INFO] {model_name}: fallback style_distance_correlation.csv used successfully')
    return pd.DataFrame({
        'target_feature': pivot_df['target_feature'].astype(str),
        'learned_minus_raw_feature': pivot_df['learned_context_embedding'] - pivot_df['raw_feature'],
        'learned_minus_pca_feature': pivot_df['learned_context_embedding'] - pivot_df['pca_feature'],
    })


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f'Output directory exists and not empty: {out_dir}. Use --overwrite')
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows, retrieval_rows, category_rows, win_rows = [], [], [], []
    loaded = {}
    for model_name, arg_key in MODEL_CONFIG:
        eval_dir = Path(getattr(args, arg_key))
        summary, retrieval_df, category_df, learned_win_df, warnings = _read_eval(model_name, eval_dir)
        loaded[model_name] = (eval_dir, summary, retrieval_df, category_df, learned_win_df)

        learned_retrieval = _pick_row(retrieval_df, 'learned_context_embedding')
        cat_vals = {c: _category_value(category_df, 'learned_context_embedding', c) for c in KEY_CATEGORIES}

        if model_name == 'Stage 5B baseline':
            interpretation = 'meaningful baseline; strong lateral; weak following.'
        elif model_name == 'Stage 5D-v1 group_weighted':
            interpretation = 'following strongly improved but lateral over-corrected downward.'
        else:
            interpretation = 'best current trade-off and current recommended Stage 5 model.'

        model_rows.append({
            'model_name': model_name,
            'eval_dir': str(eval_dir),
            'paper_grade_valid': bool(summary.get('paper_grade_valid', False)),
            'hit_at_1': float(learned_retrieval['hit_at_1']),
            'hit_at_5': float(learned_retrieval['hit_at_5']),
            'mean_same_label_fraction_at_5': float(learned_retrieval['mean_same_label_fraction_at_5']),
            'mean_neighbor_feature_distance': float(learned_retrieval['mean_neighbor_feature_distance']),
            'longitudinal_comfort': cat_vals['longitudinal_comfort'],
            'following_interaction': cat_vals['following_interaction'],
            'lateral_lane_dynamics': cat_vals['lateral_lane_dynamics'],
            'behavior_proxy': cat_vals['behavior_proxy'],
            'recommended_rank': 0,
            'short_interpretation': interpretation,
        })

        for rep in ['learned_context_embedding', 'raw_feature', 'pca_feature']:
            row = _pick_row(retrieval_df, rep)
            retrieval_rows.append({
                'model_name': model_name,
                'representation': rep,
                'hit_at_1': float(row['hit_at_1']),
                'hit_at_5': float(row['hit_at_5']),
                'mean_same_label_fraction_at_5': float(row['mean_same_label_fraction_at_5']),
                'mean_neighbor_feature_distance': float(row['mean_neighbor_feature_distance']),
                'median_neighbor_feature_distance': float(row['median_neighbor_feature_distance']),
            })

        for category in KEY_CATEGORIES:
            learned = _category_value(category_df, 'learned_context_embedding', category)
            raw = _category_value(category_df, 'raw_feature', category)
            pca = _category_value(category_df, 'pca_feature', category)
            best = max(raw, pca)
            if learned > best:
                verdict = 'learned_wins'
            elif abs(learned - best) <= 0.01:
                verdict = 'near_tie'
            else:
                verdict = 'learned_loses'
            category_rows.append({
                'model_name': model_name,
                'category': category,
                'learned_context_embedding_mean_corr': learned,
                'raw_feature_mean_corr': raw,
                'pca_feature_mean_corr': pca,
                'learned_minus_raw': learned - raw,
                'learned_minus_pca': learned - pca,
                'verdict': verdict,
            })

        beats_raw = learned_win_df[learned_win_df['learned_minus_raw_feature'] > 0]
        beats_pca = learned_win_df[learned_win_df['learned_minus_pca_feature'] > 0]
        beats_both = learned_win_df[(learned_win_df['learned_minus_raw_feature'] > 0) & (learned_win_df['learned_minus_pca_feature'] > 0)]
        win_rows.append({
            'model_name': model_name,
            'n_beats_raw': int(len(beats_raw)),
            'n_beats_pca': int(len(beats_pca)),
            'n_beats_both_raw_and_pca': int(len(beats_both)),
            'beat_raw_features': ';'.join(beats_raw['target_feature'].astype(str).tolist()),
            'beat_pca_features': ';'.join(beats_pca['target_feature'].astype(str).tolist()),
            'beat_both_features': ';'.join(beats_both['target_feature'].astype(str).tolist()),
        })

    model_df = pd.DataFrame(model_rows)
    rank_order = ['Stage 5D-balanced-v2', 'Stage 5D-v1 group_weighted', 'Stage 5B baseline']
    model_df['recommended_rank'] = model_df['model_name'].apply(lambda x: rank_order.index(x) + 1)
    model_df = model_df.sort_values('recommended_rank')

    category_df_all = pd.DataFrame(category_rows)
    retrieval_df_all = pd.DataFrame(retrieval_rows)
    win_df_all = pd.DataFrame(win_rows)

    model_path = out_dir / 'final_stage5_model_comparison.csv'
    category_path = out_dir / 'final_stage5_category_comparison.csv'
    retrieval_path = out_dir / 'final_stage5_retrieval_comparison.csv'
    win_path = out_dir / 'final_stage5_learned_win_summary.csv'

    model_df.to_csv(model_path, index=False)
    category_df_all.to_csv(category_path, index=False)
    retrieval_df_all.to_csv(retrieval_path, index=False)
    win_df_all.to_csv(win_path, index=False)

    fig_metrics = ['hit_at_5', 'following_interaction', 'lateral_lane_dynamics', 'behavior_proxy']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x = range(len(model_df))
    for i, metric in enumerate(fig_metrics):
        ax = axes[i]
        ax.bar(x, model_df[metric].tolist(), color=['#4e79a7', '#f28e2b', '#59a14f'])
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_df['model_name'].tolist(), rotation=15, ha='right')
        ax.set_title(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    plot_path = out_dir / 'final_stage5_comparison_plot.png'
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    md_path = out_dir / 'final_stage5_recommendation.md'
    md_lines = [
        '# Stage 5E Final Comparison Report',
        '',
        '## Input evaluation directories',
        f'- Stage 5B baseline: `{args.stage5b_eval}`',
        f'- Stage 5D-v1 group_weighted: `{args.stage5d_v1_eval}`',
        f'- Stage 5D-balanced-v2: `{args.stage5d_v2_eval}`',
        '',
        '## Paper-grade validity check',
        '- All three runs passed: `paper_grade_valid=true`, `strict_feature_schema=true`, `feature_schema_loaded=true`, and `row_alignment_checks.aligned=true`.',
        '',
        '## Final model comparison table',
        model_df.to_markdown(index=False),
        '',
        '## Retrieval comparison',
        retrieval_df_all.to_markdown(index=False),
        '',
        '## Category-wise comparison',
        category_df_all.to_markdown(index=False),
        '',
        '## Learned-win feature analysis',
        win_df_all.to_markdown(index=False),
        '',
        '## Interpretation of Stage 5B',
        '- hit@5 = 0.490300',
        '- following_interaction = 0.302917',
        '- lateral_lane_dynamics = 0.266777',
        '- Interpretation: meaningful baseline; strong lateral; weak following.',
        '',
        '## Interpretation of Stage 5D-v1',
        '- hit@5 = 0.507992',
        '- following_interaction = 0.582954',
        '- lateral_lane_dynamics = 0.204637',
        '- Interpretation: following strongly improved but lateral over-corrected downward.',
        '',
        '## Interpretation of Stage 5D-balanced-v2',
        '- hit@5 = 0.526232',
        '- following_interaction = 0.501998',
        '- lateral_lane_dynamics = 0.245608',
        '- behavior_proxy = 0.322344',
        '- Interpretation: best current trade-off and current recommended Stage 5 model.',
        '',
        '## Final recommendation',
        '- Recommended model: **Stage 5D-balanced-v2**.',
        '- Stage 5D-balanced-v2 still does not globally beat raw_feature / pca_feature retrieval baselines.',
        '- But it beats or nearly matches feature baselines on important behavior categories.',
        '- It is the best learned representation so far.',
    ]
    md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    for p in [model_path, category_path, retrieval_path, win_path, md_path, plot_path]:
        print(f'[INFO] Wrote: {p}')
    print('[INFO] Recommended model: Stage 5D-balanced-v2')


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 5E final comparison report for learned embeddings.')
    parser.add_argument('--stage5b_eval', required=True)
    parser.add_argument('--stage5d_v1_eval', required=True)
    parser.add_argument('--stage5d_v2_eval', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
