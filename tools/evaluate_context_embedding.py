#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

CORE_REQUIRED_FEATURES = [
    'rms_accel','rms_jerk','max_abs_accel','max_abs_jerk','mean_thw','min_thw','mean_front_distance','min_front_distance',
    'mean_rel_speed','p95_rel_speed','rms_yaw_rate','rms_curvature','heading_change_total','lane_change_count_proxy',
    'lane_change_rate_proxy','max_lateral_speed','rms_lateral_accel','front_pressure_score','left_front_min_gap','left_rear_min_gap',
    'right_front_min_gap','right_rear_min_gap','left_gap_min','right_gap_min','rear_vehicle_pressure_proxy','yielding_score_proxy','assertiveness_score_proxy'
]

STYLE_TARGET_FEATURES = [
    'rms_accel','rms_jerk','max_abs_accel','max_abs_jerk',
    'mean_thw','min_thw','mean_front_distance','min_front_distance','mean_rel_speed','p95_rel_speed','front_pressure_score','rear_vehicle_pressure_proxy',
    'rms_yaw_rate','rms_curvature','heading_change_total','lane_change_count_proxy','lane_change_rate_proxy','max_lateral_speed','rms_lateral_accel',
    'lane_change_oscillation_score_proxy','left_front_min_gap','left_rear_min_gap','right_front_min_gap','right_rear_min_gap','left_gap_min','right_gap_min',
    'left_gap_acceptance_proxy','right_gap_acceptance_proxy','yielding_score_proxy','assertiveness_score_proxy'
]
PSEUDO_STYLE_FEATURES = [
    'rms_jerk','mean_thw','min_thw','mean_front_distance','min_front_distance','mean_rel_speed','p95_rel_speed','rms_yaw_rate','rms_curvature',
    'front_pressure_score','yielding_score_proxy','assertiveness_score_proxy'
]
CONTEXT_SENSITIVITY_FEATURES = [
    'mean_thw','min_thw','mean_front_distance','min_front_distance','mean_rel_speed','p95_rel_speed','front_pressure_score','rear_vehicle_pressure_proxy',
    'left_front_min_gap','left_rear_min_gap','right_front_min_gap','right_rear_min_gap','left_gap_min','right_gap_min','yielding_score_proxy','assertiveness_score_proxy'
]

CATEGORY_FEATURE_GROUPS = {
    'longitudinal_comfort': [
        'rms_accel', 'rms_jerk', 'max_abs_accel', 'max_abs_jerk'
    ],
    'following_interaction': [
        'mean_thw', 'min_thw', 'mean_front_distance', 'min_front_distance', 'mean_rel_speed', 'p95_rel_speed',
        'front_pressure_score', 'rear_vehicle_pressure_proxy'
    ],
    'lateral_lane_dynamics': [
        'rms_yaw_rate', 'rms_curvature', 'heading_change_total', 'lane_change_count_proxy', 'lane_change_rate_proxy',
        'max_lateral_speed', 'rms_lateral_accel', 'lane_change_oscillation_score_proxy',
        'left_front_min_gap', 'left_rear_min_gap', 'right_front_min_gap', 'right_rear_min_gap',
        'left_gap_min', 'right_gap_min', 'left_gap_acceptance_proxy', 'right_gap_acceptance_proxy'
    ],
    'behavior_proxy': [
        'yielding_score_proxy', 'assertiveness_score_proxy'
    ],
}

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def _safe_load_npy(path: Path, mmap_mode='r'):
    try:
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=True)
    except ValueError:
        return np.load(path, allow_pickle=True)

def _load_feature_schema(path: Path) -> dict:
    obj = _load_json(path)
    feats = obj.get('features', [])
    if not isinstance(feats, list):
        raise RuntimeError(f'Invalid feature schema at {path}: "features" must be a list.')
    ordered = sorted(feats, key=lambda x: int(x['index']))
    dim = int(obj.get('feature_dim', len(ordered)))
    if len(ordered) != dim:
        raise RuntimeError(f'Invalid feature schema at {path}: feature_dim={dim} but features length={len(ordered)}.')
    return {'feature_dim': dim, 'features': ordered, 'names': [str(f['name']) for f in ordered]}

def _build_feature_mapping(feature_names: List[str]) -> Dict[str, int]:
    lowered = {}
    for i, name in enumerate(feature_names):
        key = name.lower()
        if key in lowered:
            raise RuntimeError(f'Duplicate feature name in schema: {name}')
        lowered[key] = i
    return {name: lowered[name.lower()] for name in feature_names}

def _verdict(learned: float, baseline: float, near_tie_threshold: float = 0.01) -> str:
    if learned > baseline:
        return 'win'
    if abs(learned - baseline) <= near_tie_threshold:
        return 'near_tie'
    return 'lose'

def summarize_model_position(category_corr_df: pd.DataFrame, retrieval_df: pd.DataFrame, learned_win_df: pd.DataFrame = None) -> dict:
    def _fmt(name: str) -> str:
        return name.replace('_', ' ')

    retrieval_row = retrieval_df[retrieval_df['representation'] == 'learned_context_embedding']
    raw_row = retrieval_df[retrieval_df['representation'] == 'raw_feature']
    pca_row = retrieval_df[retrieval_df['representation'] == 'pca_feature']
    if retrieval_row.empty or raw_row.empty or pca_row.empty:
        raise RuntimeError('Missing retrieval metrics for learned_context_embedding/raw_feature/pca_feature.')

    learned_hit5 = float(retrieval_row.iloc[0]['hit_at_5'])
    best_baseline_hit5 = max(float(raw_row.iloc[0]['hit_at_5']), float(pca_row.iloc[0]['hit_at_5']))
    global_verdict = _verdict(learned_hit5, best_baseline_hit5)

    category_verdicts = {}
    for category in CATEGORY_FEATURE_GROUPS.keys():
        learned_row = category_corr_df[(category_corr_df['category'] == category) & (category_corr_df['representation'] == 'learned_context_embedding')]
        raw_cat = category_corr_df[(category_corr_df['category'] == category) & (category_corr_df['representation'] == 'raw_feature')]
        pca_cat = category_corr_df[(category_corr_df['category'] == category) & (category_corr_df['representation'] == 'pca_feature')]
        if learned_row.empty or raw_cat.empty or pca_cat.empty:
            continue
        learned_val = float(learned_row.iloc[0]['mean_spearman_corr'])
        best_baseline = max(float(raw_cat.iloc[0]['mean_spearman_corr']), float(pca_cat.iloc[0]['mean_spearman_corr']))
        verdict = _verdict(learned_val, best_baseline)
        category_verdicts[category] = {
            'verdict': verdict,
            'learned_mean_spearman_corr': learned_val,
            'best_feature_baseline_mean_spearman_corr': best_baseline,
            'delta_vs_best_feature_baseline': learned_val - best_baseline,
        }

    win_count = sum(1 for v in category_verdicts.values() if v['verdict'] == 'win')
    near_tie_count = sum(1 for v in category_verdicts.values() if v['verdict'] == 'near_tie')
    if global_verdict == 'win' and win_count >= 2:
        recommended = 'learned_context_embedding'
    elif win_count + near_tie_count >= 2:
        recommended = 'learned_context_embedding_best_tradeoff_so_far'
    else:
        recommended = 'feature_baseline_preferred_for_global_retrieval'

    key_win_features = []
    if learned_win_df is not None and not learned_win_df.empty:
        for _, row in learned_win_df.iterrows():
            if float(row.get('learned_minus_best_feature_baseline', -np.inf)) > 0:
                key_win_features.append(str(row['target_feature']))

    lines = [
        f"Global retrieval (hit@5): learned_context_embedding is {_fmt(global_verdict)} vs best feature baseline.",
    ]
    for category in ['longitudinal_comfort', 'following_interaction', 'lateral_lane_dynamics', 'behavior_proxy']:
        if category in category_verdicts:
            lines.append(f"{_fmt(category)}: learned_context_embedding is {_fmt(category_verdicts[category]['verdict'])} vs best feature baseline.")
    if key_win_features:
        lines.append(f"Feature-level learned wins observed on {len(key_win_features)} targets (see learned_win_features.csv).")
    if recommended == 'learned_context_embedding_best_tradeoff_so_far':
        lines.append('Overall: learned_context_embedding is the best current trade-off learned representation, but not a full global retrieval win over handcrafted baselines.')
    elif recommended == 'learned_context_embedding':
        lines.append('Overall: learned_context_embedding is the recommended model candidate.')
    else:
        lines.append('Overall: handcrafted feature baselines remain stronger for retrieval-oriented ranking.')

    return {
        'dynamic_conclusion': ' '.join(lines),
        'global_retrieval_verdict': global_verdict,
        'category_verdicts': category_verdicts,
        'recommended_model_candidate': recommended,
    }

def run(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise RuntimeError(f'{out} exists and is not empty. Use --overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    warnings = []

    emb_manifest = _load_json(Path(args.embedding_manifest))
    src_manifest_path = Path(args.source_shard_manifest)
    src_manifest = _load_json(src_manifest_path)

    emb_paths = emb_manifest.get('embedding_shard_paths', [])
    shards = src_manifest.get('shards', src_manifest.get('shard_infos', []))
    if not shards and 'shard_paths' in src_manifest:
        shards = [{'shard_path': s} for s in src_manifest['shard_paths']]
    if len(emb_paths) != len(shards):
        warnings.append(f'Embedding shard count ({len(emb_paths)}) != source shard count ({len(shards)}). Using min count.')
    n_shards = min(len(emb_paths), len(shards))

    strict_feature_schema = args.strict_feature_schema
    dataset_root = src_manifest_path.parent
    schema_path = Path(args.feature_schema) if args.feature_schema else (dataset_root / 'feature_schema.json')
    if not schema_path.exists():
        raise RuntimeError(f'Feature schema file is required but not found: {schema_path}')

    fs = _load_feature_schema(schema_path)
    feature_names = fs['names']
    fmap = _build_feature_mapping(feature_names)

    sampled = []
    total_eval_rows = 0
    for sid in tqdm(range(n_shards), desc='Collecting samples', unit='shard'):
        sdir = src_manifest_path.parent / shards[sid]['shard_path']
        split = _safe_load_npy(sdir / 'split.npy')
        if split.dtype.kind not in {'U', 'S', 'O'}:
            split = np.array([['train', 'val', 'test'][int(x)] if int(x) in [0, 1, 2] else str(int(x)) for x in split], dtype=object)
        mask = np.ones_like(split, dtype=bool) if args.eval_split == 'all' else (split.astype(str) == args.eval_split)
        idx = np.flatnonzero(mask)
        total_eval_rows += int(idx.size)
        sampled.extend((sid, int(i)) for i in idx)

    rng = np.random.default_rng(args.seed)
    if len(sampled) > args.max_eval_samples:
        choose = rng.choice(len(sampled), size=args.max_eval_samples, replace=False)
        sampled = [sampled[i] for i in choose]
    sampled = sorted(sampled, key=lambda x: (x[0], x[1]))

    by_shard = {}
    for sid, lid in sampled:
        by_shard.setdefault(sid, []).append(lid)

    X_emb, X_feat, X_ctx = [], [], []
    align_ok = True
    finite = {'embedding_nonfinite': 0, 'feature_nonfinite': 0, 'context_nonfinite': 0}
    for sid, lids in tqdm(by_shard.items(), desc='Loading data', unit='shard'):
        emb = _safe_load_npy(Path(emb_paths[sid]))
        sdir = src_manifest_path.parent / shards[sid]['shard_path']
        feat = _safe_load_npy(sdir / 'interaction_feat_style.npy')
        ctx = _safe_load_npy(sdir / 'context_traj.npy')
        if emb.shape[0] != feat.shape[0]:
            align_ok = False
            warnings.append(f'Row mismatch at shard {sid}: emb={emb.shape[0]} feat={feat.shape[0]}.')
        ids = np.array(lids, dtype=np.int64)
        X_emb.append(np.asarray(emb[ids], dtype=np.float32))
        X_feat.append(np.asarray(feat[ids], dtype=np.float32))
        X_ctx.append(np.asarray(ctx[ids], dtype=np.float32).reshape(len(lids), -1))

    X_emb = np.concatenate(X_emb, axis=0) if X_emb else np.zeros((0, 1), dtype=np.float32)
    X_feat = np.concatenate(X_feat, axis=0) if X_feat else np.zeros((0, 1), dtype=np.float32)
    X_ctx = np.concatenate(X_ctx, axis=0) if X_ctx else np.zeros((0, 1), dtype=np.float32)

    if fs['feature_dim'] != X_feat.shape[1]:
        raise RuntimeError(f'Feature dimension mismatch: schema={fs["feature_dim"]}, loaded feature array={X_feat.shape[1]}.')

    missing_required = [k for k in CORE_REQUIRED_FEATURES if k not in fmap]
    if strict_feature_schema and missing_required:
        raise RuntimeError(f'Strict feature schema enabled but required features are missing: {missing_required}')

    optional_features = sorted(set(STYLE_TARGET_FEATURES + PSEUDO_STYLE_FEATURES + CONTEXT_SENSITIVITY_FEATURES) - set(CORE_REQUIRED_FEATURES))
    missing_optional = [k for k in optional_features if k not in fmap]
    for k in missing_optional:
        warnings.append(f'Optional feature missing from schema; dependent metric(s) skipped: {k}')

    finite['embedding_nonfinite'] = int((~np.isfinite(X_emb)).sum()); X_emb = np.nan_to_num(X_emb)
    finite['feature_nonfinite'] = int((~np.isfinite(X_feat)).sum()); X_feat = np.nan_to_num(X_feat)
    finite['context_nonfinite'] = int((~np.isfinite(X_ctx)).sum()); X_ctx = np.nan_to_num(X_ctx)

    X_pca = PCA(n_components=min(16, X_feat.shape[1], max(2, X_feat.shape[0]-1)), random_state=args.seed).fit_transform(X_feat) if X_feat.shape[0] >= 2 else np.zeros_like(X_feat)
    reps = {'learned_context_embedding': X_emb, 'raw_feature': X_feat, 'pca_feature': X_pca, 'context_l2': X_ctx, 'random': rng.standard_normal(size=X_emb.shape, dtype=np.float32)}

    pseudo_avail = [k for k in PSEUDO_STYLE_FEATURES if k in fmap]
    skipped_style_delta_features = [k for k in PSEUDO_STYLE_FEATURES if k not in fmap]
    style_vec = np.stack([X_feat[:, fmap[k]] for k in pseudo_avail], axis=1)
    qs = np.quantile(style_vec, [0.33, 0.66], axis=0)
    bins = (style_vec > qs[0]).astype(int) + (style_vec > qs[1]).astype(int)
    labels = np.array(['_'.join(map(str, row.tolist())) for row in bins], dtype=object)

    retrieval_rows = []
    for rep_name, X in reps.items():
        nn = NearestNeighbors(n_neighbors=min(11, max(2, X.shape[0])), metric='euclidean').fit(X)
        nbr = nn.kneighbors(return_distance=False)[:, 1:]
        feat_dist = np.linalg.norm(X_feat[:, None, :] - X_feat[nbr], axis=2)
        top5 = nbr[:, :min(5, nbr.shape[1])]
        top10 = nbr[:, :min(10, nbr.shape[1])]
        retrieval_rows.append({'representation': rep_name, 'k': int(min(10, nbr.shape[1])),'mean_neighbor_feature_distance': float(np.mean(feat_dist)), 'median_neighbor_feature_distance': float(np.median(feat_dist)), 'hit_at_1': float(np.mean(labels[nbr[:, 0]] == labels)), 'hit_at_5': float(np.mean(np.any(labels[top5] == labels[:, None], axis=1))), 'mean_same_label_fraction_at_5': float(np.mean(labels[top5] == labels[:, None])), 'mean_same_label_fraction_at_10': float(np.mean(labels[top10] == labels[:, None]))})
    pd.DataFrame(retrieval_rows).to_csv(out / 'retrieval_metrics.csv', index=False)

    corr_rows = []
    style_targets_avail = [k for k in STYLE_TARGET_FEATURES if k in fmap]
    n_pairs = min(args.max_pairs, X_feat.shape[0] * 4)
    i = rng.integers(0, X_feat.shape[0], size=n_pairs); j = rng.integers(0, X_feat.shape[0], size=n_pairs)
    valid = i != j; i, j = i[valid], j[valid]
    for rep_name, X in reps.items():
        d = np.linalg.norm(X[i] - X[j], axis=1)
        for k in style_targets_avail:
            delta = np.abs(X_feat[i, fmap[k]] - X_feat[j, fmap[k]])
            corr, p = spearmanr(d, delta)
            corr_rows.append({'representation': rep_name, 'target_feature': f'{k}_delta', 'spearman_corr': float(corr), 'p_value': float(p), 'n_pairs': int(len(d))})
    pd.DataFrame(corr_rows).to_csv(out / 'style_distance_correlation.csv', index=False)
    corr_df = pd.DataFrame(corr_rows)

    category_rows = []
    for category, features in CATEGORY_FEATURE_GROUPS.items():
        deltas = [f'{f}_delta' for f in features if f in style_targets_avail]
        if not deltas:
            continue
        for rep_name in reps:
            sub = corr_df[(corr_df['representation'] == rep_name) & (corr_df['target_feature'].isin(deltas))]
            if sub.empty:
                continue
            category_rows.append({
                'category': category,
                'representation': rep_name,
                'mean_spearman_corr': float(sub['spearman_corr'].mean()),
                'median_spearman_corr': float(sub['spearman_corr'].median()),
                'number_of_features': int(sub['target_feature'].nunique()),
            })
    category_corr_df = pd.DataFrame(category_rows)
    category_corr_df.to_csv(out / 'category_correlation_summary.csv', index=False)

    baseline_cols = ['raw_feature', 'pca_feature']
    retrieval_summary_rows = []
    for category, features in CATEGORY_FEATURE_GROUPS.items():
        deltas = [f'{f}_delta' for f in features if f in style_targets_avail]
        if not deltas:
            continue
        learned_sub = corr_df[(corr_df['representation'] == 'learned_context_embedding') & (corr_df['target_feature'].isin(deltas))]
        raw_sub = corr_df[(corr_df['representation'] == 'raw_feature') & (corr_df['target_feature'].isin(deltas))]
        pca_sub = corr_df[(corr_df['representation'] == 'pca_feature') & (corr_df['target_feature'].isin(deltas))]
        if learned_sub.empty or raw_sub.empty or pca_sub.empty:
            continue
        retrieval_summary_rows.append({
            'category': category,
            'learned_mean_spearman_corr': float(learned_sub['spearman_corr'].mean()),
            'raw_feature_mean_spearman_corr': float(raw_sub['spearman_corr'].mean()),
            'pca_feature_mean_spearman_corr': float(pca_sub['spearman_corr'].mean()),
            'learned_minus_best_feature_baseline': float(learned_sub['spearman_corr'].mean() - max(raw_sub['spearman_corr'].mean(), pca_sub['spearman_corr'].mean())),
            'number_of_features': int(learned_sub['target_feature'].nunique()),
        })
    if retrieval_summary_rows:
        pd.DataFrame(retrieval_summary_rows).to_csv(out / 'category_retrieval_summary.csv', index=False)

    learned_rows = []
    for k in style_targets_avail:
        tf = f'{k}_delta'
        tf_rows = corr_df[corr_df['target_feature'] == tf]
        if tf_rows.empty:
            continue
        by_rep = {r['representation']: float(r['spearman_corr']) for _, r in tf_rows.iterrows()}
        sorted_reps = sorted(by_rep.items(), key=lambda x: x[1], reverse=True)
        learned_rank = next((idx + 1 for idx, (rep, _) in enumerate(sorted_reps) if rep == 'learned_context_embedding'), None)
        best_baseline = max(by_rep.get('raw_feature', np.nan), by_rep.get('pca_feature', np.nan))
        learned_rows.append({
            'target_feature': k,
            'learned_corr': by_rep.get('learned_context_embedding', np.nan),
            'raw_feature_corr': by_rep.get('raw_feature', np.nan),
            'pca_feature_corr': by_rep.get('pca_feature', np.nan),
            'context_l2_corr': by_rep.get('context_l2', np.nan),
            'random_corr': by_rep.get('random', np.nan),
            'learned_rank': int(learned_rank) if learned_rank is not None else np.nan,
            'learned_minus_best_feature_baseline': float(by_rep.get('learned_context_embedding', np.nan) - best_baseline),
        })
    pd.DataFrame(learned_rows).to_csv(out / 'learned_win_features.csv', index=False)
    learned_win_df = pd.DataFrame(learned_rows)

    context_rows = []
    for k in [x for x in CONTEXT_SENSITIVITY_FEATURES if x in fmap]:
        v = X_feat[:, fmap[k]].astype(np.float32)
        for rep_name, X in reps.items():
            nn = NearestNeighbors(n_neighbors=min(6, X.shape[0]), metric='euclidean').fit(X)
            nbr = nn.kneighbors(return_distance=False)[:, 1:]
            abs_diff = np.mean(np.abs(v[:, None] - v[nbr])) if nbr.size else np.nan
            rank_corr, _ = spearmanr(v, np.mean(v[nbr], axis=1)) if nbr.size else (np.nan, np.nan)
            context_rows.append({'representation': rep_name, 'context_variable': k, 'metric_name': 'mean_abs_neighbor_delta', 'metric_value': float(abs_diff)})
            context_rows.append({'representation': rep_name, 'context_variable': k, 'metric_name': 'nn_value_spearman_corr', 'metric_value': float(rank_corr)})
    pd.DataFrame(context_rows).to_csv(out / 'context_sensitivity_metrics.csv', index=False)

    rdf = pd.DataFrame(retrieval_rows)
    plt.figure(figsize=(8,4)); plt.bar(rdf['representation'], rdf['hit_at_5']); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out/'retrieval_bar.png'); plt.close()
    cdf = corr_df
    focus = cdf[cdf['target_feature'].isin([f'{x}_delta' for x in ['mean_thw','min_thw','mean_front_distance','mean_rel_speed','p95_rel_speed','rms_jerk','rms_yaw_rate','rms_curvature']])]
    agg = focus.groupby('representation', as_index=False)['spearman_corr'].mean()
    plt.figure(figsize=(8,4)); plt.bar(agg['representation'], agg['spearman_corr']); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out/'feature_delta_correlation_bar.png'); plt.close()

    dynamic_summary = summarize_model_position(category_corr_df, rdf, learned_win_df)
    paper_grade_valid = bool(strict_feature_schema and not missing_required)
    summary = {
        'input_paths': {'embedding_manifest': args.embedding_manifest, 'source_shard_manifest': args.source_shard_manifest},
        'feature_schema_path': str(schema_path), 'feature_schema_loaded': True, 'strict_feature_schema': bool(strict_feature_schema),
        'feature_dim': int(X_feat.shape[1]), 'feature_index_mapping': fmap, 'core_required_features': CORE_REQUIRED_FEATURES,
        'missing_required_features': missing_required, 'optional_features': optional_features, 'missing_optional_features': missing_optional,
        'skipped_style_delta_features': [k for k in STYLE_TARGET_FEATURES if k not in fmap], 'warnings': warnings, 'paper_grade_valid': paper_grade_valid,
        'eval_split': args.eval_split, 'max_eval_samples': args.max_eval_samples, 'actual_eval_samples': int(X_emb.shape[0]),
        'representation_list': list(reps.keys()), 'finite_checks': finite,
        'row_alignment_checks': {'aligned': bool(align_ok), 'embedding_shards_used': n_shards, 'total_eval_rows_before_subsample': total_eval_rows},
        'dynamic_conclusion': dynamic_summary['dynamic_conclusion'],
        'global_retrieval_verdict': dynamic_summary['global_retrieval_verdict'],
        'category_verdicts': dynamic_summary['category_verdicts'],
        'recommended_model_candidate': dynamic_summary['recommended_model_candidate'],
    }
    (out/'evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    report = [
        '# Stage 5C Context Embedding Evaluation',
        f'- Strict schema mode used: **{strict_feature_schema}**',
        '- feature_schema.json loaded: **yes**',
        '- No fallback feature index was used: **yes**',
        '- mean_speed and std_rel_speed are not part of the Stage 5 schema and were not evaluated.',
        '- p95_rel_speed is used instead of std_rel_speed.',
        f'- Paper-grade valid: **{"yes" if paper_grade_valid else "no"}**',
        '', '## Retrieval Results', pd.DataFrame(retrieval_rows).to_markdown(index=False), '',
        '## Style-distance Correlation', pd.DataFrame(corr_rows).to_markdown(index=False), '',
        '## Context Sensitivity', pd.DataFrame(context_rows).to_markdown(index=False), '',
        '## Category-wise Correlation Summary', (category_corr_df.to_markdown(index=False) if not category_corr_df.empty else '_No category rows available._'), '',
        '## Dynamic Evaluation Conclusions',
        f"- Global retrieval verdict: **{dynamic_summary['global_retrieval_verdict']}**",
        f"- Longitudinal comfort verdict: **{dynamic_summary['category_verdicts'].get('longitudinal_comfort', {}).get('verdict', 'not_available')}**",
        f"- Following interaction verdict: **{dynamic_summary['category_verdicts'].get('following_interaction', {}).get('verdict', 'not_available')}**",
        f"- Lateral/lane dynamics verdict: **{dynamic_summary['category_verdicts'].get('lateral_lane_dynamics', {}).get('verdict', 'not_available')}**",
        f"- Behavior proxy verdict: **{dynamic_summary['category_verdicts'].get('behavior_proxy', {}).get('verdict', 'not_available')}**",
        f"- Overall recommendation: **{dynamic_summary['recommended_model_candidate']}**",
        '',
        dynamic_summary['dynamic_conclusion'],
        '',
        '## Warnings and Limitations']
    report.extend([f'- {w}' for w in warnings] or ['- None'])
    (out/'evaluation_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_manifest', required=True)
    p.add_argument('--source_shard_manifest', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--feature_schema', default=None)
    p.add_argument('--strict_feature_schema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max_eval_samples', type=int, default=20000)
    p.add_argument('--eval_split', default='test', choices=['train','val','test','all'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_pairs', type=int, default=50000)
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())
