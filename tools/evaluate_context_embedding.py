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
    cdf = pd.DataFrame(corr_rows)
    focus = cdf[cdf['target_feature'].isin([f'{x}_delta' for x in ['mean_thw','min_thw','mean_front_distance','mean_rel_speed','p95_rel_speed','rms_jerk','rms_yaw_rate','rms_curvature']])]
    agg = focus.groupby('representation', as_index=False)['spearman_corr'].mean()
    plt.figure(figsize=(8,4)); plt.bar(agg['representation'], agg['spearman_corr']); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out/'feature_delta_correlation_bar.png'); plt.close()

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
