#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_load_npy(path: Path, mmap_mode='r'):
    try:
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=True)
    except ValueError:
        return np.load(path, allow_pickle=True)


def _find_feature_names(manifest_obj: dict, manifest_dir: Path) -> List[str]:
    candidates = [
        manifest_obj.get('feature_names'),
        manifest_obj.get('interaction_feature_names'),
    ]
    for c in candidates:
        if isinstance(c, list) and c:
            return [str(x) for x in c]
    for meta in ['build_summary.json', 'dataset_summary.json', 'summary.json']:
        p = manifest_dir / meta
        if p.exists():
            obj = _load_json(p)
            for k in ['feature_names', 'interaction_feature_names']:
                if isinstance(obj.get(k), list) and obj.get(k):
                    return [str(x) for x in obj[k]]
    return []


def _feature_mapping(feature_names: List[str], dim: int) -> Tuple[Dict[str, int], List[str]]:
    warnings = []
    aliases = {
        'mean_speed': ['mean_speed'],
        'rms_accel': ['rms_accel'],
        'rms_jerk': ['rms_jerk'],
        'rms_yaw_rate': ['rms_yaw_rate'],
        'rms_curvature': ['rms_curvature'],
        'mean_thw': ['mean_thw'],
        'min_thw': ['min_thw'],
        'mean_front_distance': ['mean_front_distance', 'mean_front_dist'],
        'min_front_distance': ['min_front_distance', 'min_front_dist'],
        'mean_rel_speed': ['mean_rel_speed'],
        'std_rel_speed': ['std_rel_speed'],
    }
    fmap = {}
    if feature_names:
        lowered = {n.lower(): i for i, n in enumerate(feature_names)}
        for key, names in aliases.items():
            for n in names:
                if n.lower() in lowered:
                    fmap[key] = lowered[n.lower()]
                    break
    fallback = {
        'mean_speed': 0, 'rms_accel': 1, 'rms_jerk': 2, 'rms_yaw_rate': 3, 'rms_curvature': 4,
        'mean_thw': 5, 'min_thw': 6, 'mean_front_distance': 7, 'min_front_distance': 8,
        'mean_rel_speed': 9, 'std_rel_speed': 10,
    }
    for k, idx in fallback.items():
        if k not in fmap and idx < dim:
            fmap[k] = idx
            warnings.append(f'Feature "{k}" not found by name; fallback index {idx} used.')
    return fmap, warnings


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
    feature_schema_loaded = False
    feature_names = []
    if schema_path.exists():
        fs = _load_feature_schema(schema_path)
        feature_names = fs['names']
        feature_schema_loaded = True
    elif strict_feature_schema:
        raise RuntimeError(f'Strict feature schema enabled but schema file not found: {schema_path}')
    else:
        feature_names = _find_feature_names(src_manifest, dataset_root)
        warnings.append(f'Feature schema file missing; relaxed mode fallback using manifest/build_summary names: {schema_path}')

    sampled = []
    total_eval_rows = 0
    for sid in tqdm(range(n_shards), desc='Collecting samples', unit='shard'):
        sdir = src_manifest_path.parent / shards[sid]['shard_path']
        split = _safe_load_npy(sdir / 'split.npy')
        if split.dtype.kind not in {'U', 'S', 'O'}:
            split = np.array([['train', 'val', 'test'][int(x)] if int(x) in [0, 1, 2] else str(int(x)) for x in split], dtype=object)
        mask = split.astype(str) == args.eval_split
        idx = np.flatnonzero(mask)
        total_eval_rows += int(idx.size)
        if idx.size == 0:
            continue
        sampled.extend([(sid, int(i)) for i in idx])

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
        feat_path = sdir / 'interaction_feat_style.npy'
        if not feat_path.exists():
            feat_path = sdir / 'interaction_feat_style_raw.npy'
            warnings.append(f'Using raw feature file for shard {sid}.')
        feat = _safe_load_npy(feat_path)
        ctx = _safe_load_npy(sdir / 'context_traj.npy')
        n = len(lids)
        if emb.shape[0] != feat.shape[0]:
            align_ok = False
            warnings.append(f'Row mismatch at shard {sid}: emb={emb.shape[0]} feat={feat.shape[0]}.')
        ids = np.array(lids, dtype=np.int64)
        X_emb.append(np.asarray(emb[ids], dtype=np.float32))
        X_feat.append(np.asarray(feat[ids], dtype=np.float32))
        X_ctx.append(np.asarray(ctx[ids], dtype=np.float32).reshape(n, -1))

    X_emb = np.concatenate(X_emb, axis=0) if X_emb else np.zeros((0, 1), dtype=np.float32)
    X_feat = np.concatenate(X_feat, axis=0) if X_feat else np.zeros((0, 1), dtype=np.float32)
    X_ctx = np.concatenate(X_ctx, axis=0) if X_ctx else np.zeros((0, 1), dtype=np.float32)

    finite['embedding_nonfinite'] = int((~np.isfinite(X_emb)).sum())
    finite['feature_nonfinite'] = int((~np.isfinite(X_feat)).sum())
    finite['context_nonfinite'] = int((~np.isfinite(X_ctx)).sum())
    X_emb = np.nan_to_num(X_emb, nan=0.0, posinf=0.0, neginf=0.0)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_ctx = np.nan_to_num(X_ctx, nan=0.0, posinf=0.0, neginf=0.0)

    if feature_schema_loaded and fs['feature_dim'] != X_feat.shape[1]:
        raise RuntimeError(f'Feature dimension mismatch: schema={fs["feature_dim"]}, loaded feature array={X_feat.shape[1]}.')
    fmap, map_warn = _feature_mapping(feature_names, X_feat.shape[1])
    required = ['mean_thw', 'min_thw', 'mean_front_distance', 'min_front_distance', 'mean_rel_speed', 'std_rel_speed']
    missing_required = [k for k in required if k not in fmap]
    if strict_feature_schema and missing_required:
        raise RuntimeError(f'Strict feature schema enabled but required features are missing: {missing_required}')
    if strict_feature_schema and map_warn:
        raise RuntimeError(f'Strict feature schema enabled; fallback feature index resolution is forbidden. Details: {map_warn}')
    warnings.extend(map_warn if not strict_feature_schema else [])

    print(f'[INFO] Running PCA on {X_feat.shape[0]} samples...')
    pca = PCA(n_components=min(16, X_feat.shape[1], max(2, X_feat.shape[0] - 1)), random_state=args.seed)
    X_pca = pca.fit_transform(X_feat) if X_feat.shape[0] >= 2 else np.zeros_like(X_feat)

    reps = {
        'learned_context_embedding': X_emb,
        'raw_feature': X_feat,
        'pca_feature': X_pca,
        'context_l2': X_ctx,
        'random': rng.standard_normal(size=X_emb.shape, dtype=np.float32),
    }

    style_keys = ['mean_thw', 'min_thw', 'mean_front_distance', 'min_front_distance', 'mean_rel_speed', 'rms_jerk', 'rms_yaw_rate', 'rms_curvature']
    style_avail = [k for k in style_keys if k in fmap]
    if not style_avail:
        style_vec = X_feat[:, :min(3, X_feat.shape[1])]
        warnings.append('No named style features available; using first dims for pseudo labels.')
    else:
        style_vec = np.stack([X_feat[:, fmap[k]] for k in style_avail], axis=1)
    qs = np.quantile(style_vec, [0.33, 0.66], axis=0)
    bins = (style_vec > qs[0]).astype(int) + (style_vec > qs[1]).astype(int)
    labels = np.array(['_'.join(map(str, row.tolist())) for row in bins], dtype=object)

    retrieval_rows = []
    for rep_name, X in tqdm(reps.items(), desc='Running retrieval', unit='rep'):
        n_neighbors = min(11, max(2, X.shape[0]))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X)
        idx = nn.kneighbors(return_distance=False)
        nbr = idx[:, 1:]
        feat_dist = np.linalg.norm(X_feat[:, None, :] - X_feat[nbr], axis=2)
        hit1 = np.mean(labels[nbr[:, 0]] == labels) if nbr.shape[1] >= 1 else np.nan
        top5 = nbr[:, :min(5, nbr.shape[1])]
        top10 = nbr[:, :min(10, nbr.shape[1])]
        same5 = np.mean(labels[top5] == labels[:, None]) if top5.size else np.nan
        same10 = np.mean(labels[top10] == labels[:, None]) if top10.size else np.nan
        hit5 = np.mean(np.any(labels[top5] == labels[:, None], axis=1)) if top5.size else np.nan
        retrieval_rows.append({
            'representation': rep_name, 'k': int(min(10, nbr.shape[1])),
            'mean_neighbor_feature_distance': float(np.mean(feat_dist)),
            'median_neighbor_feature_distance': float(np.median(feat_dist)),
            'hit_at_1': float(hit1), 'hit_at_5': float(hit5),
            'mean_same_label_fraction_at_5': float(same5),
            'mean_same_label_fraction_at_10': float(same10),
        })

    pd.DataFrame(retrieval_rows).to_csv(out / 'retrieval_metrics.csv', index=False)

    corr_rows = []
    target_keys = ['mean_speed', 'rms_accel', 'rms_jerk', 'rms_yaw_rate', 'rms_curvature', 'mean_thw', 'min_thw', 'mean_front_distance', 'min_front_distance', 'mean_rel_speed', 'std_rel_speed']
    n_pairs = min(args.max_pairs, X_feat.shape[0] * 4)
    i = rng.integers(0, X_feat.shape[0], size=n_pairs)
    j = rng.integers(0, X_feat.shape[0], size=n_pairs)
    valid = i != j
    i, j = i[valid], j[valid]
    for rep_name, X in tqdm(reps.items(), desc='Computing correlations', unit='rep'):
        d = np.linalg.norm(X[i] - X[j], axis=1)
        for k in target_keys:
            if k not in fmap:
                continue
            delta = np.abs(X_feat[i, fmap[k]] - X_feat[j, fmap[k]])
            corr, p = spearmanr(d, delta)
            corr_rows.append({'representation': rep_name, 'target_feature': f'{k}_delta', 'spearman_corr': float(corr), 'p_value': float(p), 'n_pairs': int(len(d))})
    pd.DataFrame(corr_rows).to_csv(out / 'style_distance_correlation.csv', index=False)

    context_rows = []
    for k in ['mean_thw', 'min_thw', 'mean_front_distance', 'min_front_distance', 'mean_rel_speed', 'std_rel_speed']:
        if k not in fmap:
            continue
        v = X_feat[:, fmap[k]].astype(np.float32)
        for rep_name, X in reps.items():
            nn = NearestNeighbors(n_neighbors=min(6, X.shape[0]), metric='euclidean').fit(X)
            nbr = nn.kneighbors(return_distance=False)[:, 1:]
            nbr_vals = v[nbr]
            abs_diff = np.mean(np.abs(v[:, None] - nbr_vals)) if nbr.size else np.nan
            context_rows.append({'representation': rep_name, 'context_variable': k, 'metric_name': 'mean_abs_neighbor_delta', 'metric_value': float(abs_diff)})
    pd.DataFrame(context_rows).to_csv(out / 'context_sensitivity_metrics.csv', index=False)

    # plots
    rdf = pd.DataFrame(retrieval_rows)
    plt.figure(figsize=(8, 4)); plt.bar(rdf['representation'], rdf['hit_at_5']); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out / 'retrieval_bar.png'); plt.close()

    cdf = pd.DataFrame(corr_rows)
    focus = cdf[cdf['target_feature'].isin([f'{x}_delta' for x in ['mean_thw', 'min_thw', 'mean_front_distance', 'mean_rel_speed', 'rms_jerk', 'rms_yaw_rate', 'rms_curvature']])]
    agg = focus.groupby('representation', as_index=False)['spearman_corr'].mean()
    plt.figure(figsize=(8, 4)); plt.bar(agg['representation'], agg['spearman_corr']); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out / 'feature_delta_correlation_bar.png'); plt.close()

    pca_emb = PCA(n_components=2, random_state=args.seed).fit_transform(X_emb)
    color = X_feat[:, fmap['mean_thw']] if 'mean_thw' in fmap else np.arange(X_emb.shape[0])
    plt.figure(figsize=(6, 5)); sc = plt.scatter(pca_emb[:, 0], pca_emb[:, 1], c=color, s=4, cmap='viridis'); plt.colorbar(sc); plt.tight_layout(); plt.savefig(out / 'pca_embedding.png', dpi=160); plt.close()

    pca_feat2 = PCA(n_components=2, random_state=args.seed).fit_transform(X_feat)
    color2 = X_feat[:, fmap['mean_front_distance']] if 'mean_front_distance' in fmap else np.arange(X_feat.shape[0])
    plt.figure(figsize=(6, 5)); sc = plt.scatter(pca_feat2[:, 0], pca_feat2[:, 1], c=color2, s=4, cmap='plasma'); plt.colorbar(sc); plt.tight_layout(); plt.savefig(out / 'pca_feature.png', dpi=160); plt.close()

    # optional UMAP
    try:
        import umap  # noqa: F401
    except Exception:
        warnings.append('umap-learn not installed; skipped UMAP visualizations.')

    winner = rdf.sort_values('hit_at_5', ascending=False).iloc[0]['representation'] if not rdf.empty else 'n/a'
    summary = {
        'input_paths': {'embedding_manifest': args.embedding_manifest, 'source_shard_manifest': args.source_shard_manifest},
        'feature_schema_path': str(schema_path),
        'feature_schema_loaded': bool(feature_schema_loaded),
        'strict_feature_schema': bool(strict_feature_schema),
        'eval_split': args.eval_split,
        'max_eval_samples': args.max_eval_samples,
        'actual_eval_samples': int(X_emb.shape[0]),
        'representation_list': list(reps.keys()),
        'key_winner_summary': f'Best hit@5: {winner}',
        'warnings': warnings,
        'feature_names_used': feature_names,
        'feature_index_mapping': fmap,
        'missing_required_features': missing_required,
        'finite_checks': finite,
        'row_alignment_checks': {'aligned': bool(align_ok), 'embedding_shards_used': n_shards, 'total_eval_rows_before_subsample': total_eval_rows},
    }
    (out / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    report = [
        '# Stage 5C Context Embedding Evaluation',
        f'- Eval split: **{args.eval_split}**',
        f'- Sample count: **{X_emb.shape[0]}** (from {total_eval_rows} split rows, max={args.max_eval_samples})',
        f'- Representations: {", ".join(reps.keys())}',
        f'- Feature schema loaded: **{feature_schema_loaded}**',
        f'- Strict feature schema: **{strict_feature_schema}**',
        f'- Fallback feature indices used: **{"yes" if any("fallback index" in w for w in warnings) else "no"}**',
        f'- Paper-grade valid: **{"yes" if feature_schema_loaded and strict_feature_schema and not missing_required else "no (preliminary)"}**',
        '',
        '## Retrieval Results',
        pd.DataFrame(retrieval_rows).to_markdown(index=False),
        '',
        '## Style-distance Correlation',
        pd.DataFrame(corr_rows).to_markdown(index=False),
        '',
        '## Context Sensitivity',
        pd.DataFrame(context_rows).to_markdown(index=False),
        '',
        f'## Effectiveness Verdict\nTop retrieval representation (hit@5): **{winner}**.',
        '',
        '## Warnings and Limitations',
    ]
    report.extend([f'- {w}' for w in warnings] or ['- None'])
    report.extend(['', '## Next Step', '- If learned_context_embedding consistently wins on interaction deltas and retrieval, proceed to cross-stage alignment checks in next stage.'])
    (out / 'evaluation_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_manifest', required=True)
    p.add_argument('--source_shard_manifest', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--feature_schema', default=None)
    p.add_argument('--strict_feature_schema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max_eval_samples', type=int, default=20000)
    p.add_argument('--eval_split', default='test', choices=['train', 'val', 'test', 'all'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_pairs', type=int, default=50000)
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())
