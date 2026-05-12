#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np


def _read_json(p):
    return json.loads(Path(p).read_text())

def _to_md(df, path):
    Path(path).write_text(df.to_markdown(index=False)+"\n")

def _to_tex(df, path):
    Path(path).write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x,float) else str(x)))

def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    build = _read_json(args.build_summary)
    pseudo = _read_json(args.pseudo_label_summary)
    train = _read_json(args.train_summary)
    export = _read_json(args.export_summary)
    eval_csv = Path(args.eval_dir) / 'baseline_comparison_summary.csv'
    bdf = pd.read_csv(eval_csv)

    split_counts = build.get('split_counts', {})
    ds = pd.DataFrame([{
        'n_files_processed': build.get('n_files_processed'),
        'n_scenarios_processed': build.get('n_scenarios_processed'),
        'n_windows_kept': build.get('n_windows_kept'),
        'train_count': split_counts.get('train'),
        'val_count': split_counts.get('val'),
        'test_count': split_counts.get('test'),
        'front_found_rate': build.get('front_found_rate'),
    }])
    ds.to_csv(out/'table_dataset_statistics.csv', index=False); _to_md(ds, out/'table_dataset_statistics.md')

    psplit = pseudo.get('split_labeled_counts', {})
    pl = pd.DataFrame([{
        'conservative_like': pseudo.get('conservative_like'),
        'aggressive_like': pseudo.get('aggressive_like'),
        'lateral_stable_like': pseudo.get('lateral_stable_like'),
        'unlabeled': pseudo.get('n_unlabeled'),
        'train_labeled': psplit.get('train'),
        'val_labeled': psplit.get('val'),
        'test_labeled': psplit.get('test'),
    }])
    pl.to_csv(out/'table_pseudo_label_distribution.csv', index=False); _to_md(pl, out/'table_pseudo_label_distribution.md')

    notes = {
        'learned':'row-level learned embedding', 'raw_feature':'handcrafted style feature baseline',
        'trajectory_l2':'trajectory geometric baseline', 'random':'random sanity baseline', 'pca_feature':'PCA style feature baseline'
    }
    lvb = bdf[['method','centroid_accuracy_overall','hit_at_1','mean_same_label_fraction_topk','hit_at_1_lift_over_chance']].copy()
    lvb['notes'] = lvb['method'].map(notes).fillna('baseline')
    lvb.to_csv(out/'table_learned_vs_baselines.csv', index=False); _to_md(lvb, out/'table_learned_vs_baselines.md'); _to_tex(lvb, out/'table_learned_vs_baselines.tex')

    cols = ['method','spearman_mean_speed_delta','spearman_rms_jerk_delta','spearman_rms_yaw_rate_delta','spearman_rms_curvature_delta','spearman_mean_thw_delta',
            'valid_pairs_mean_speed_delta','valid_pairs_rms_jerk_delta','valid_pairs_rms_yaw_rate_delta','valid_pairs_rms_curvature_delta','valid_pairs_mean_thw_delta']
    sc = bdf[[c for c in cols if c in bdf.columns]].copy()
    sc.to_csv(out/'table_style_distance_correlation.csv', index=False); _to_md(sc, out/'table_style_distance_correlation.md'); _to_tex(sc, out/'table_style_distance_correlation.tex')

    texp = pd.DataFrame([{
        'n_total': train.get('n_total'), 'n_retained': train.get('n_retained'), 'n_dropped': train.get('n_dropped'),
        'traj_nan_count_raw': train.get('traj_nan_count_raw'), 'traj_nan_count_after_sanitize': train.get('traj_nan_count_after_sanitize'),
        'traj_repaired_count': train.get('traj_repaired_count'), 'feature_clipped_values': train.get('feature_clipped_values'),
        'best_val_loss': train.get('best_val_loss'), 'embedding_shape': export.get('shape'), 'row_aligned': export.get('row_aligned')
    }])
    texp.to_csv(out/'table_training_export_summary.csv', index=False); _to_md(texp, out/'table_training_export_summary.md')

    (out/'paper_tables_summary.md').write_text('# Paper Tables Summary\n\nGenerated from experiment JSON/CSV outputs.\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--eval_dir', default='outputs/waymo_human_v1_full51/eval_with_learned')
    p.add_argument('--train_summary', default='outputs/waymo_human_v1_full51/human_embedding_model/train_summary.json')
    p.add_argument('--export_summary', default='outputs/waymo_human_v1_full51/embedding_export_summary.json')
    p.add_argument('--pseudo_label_summary', default='outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json')
    p.add_argument('--build_summary', default='outputs/waymo_human_v1_full51/build_summary.json')
    p.add_argument('--out_dir', default='outputs/waymo_human_v1_full51/paper_tables')
    run(p.parse_args())
