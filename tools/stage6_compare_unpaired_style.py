#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder


def load_schema(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    feats = obj.get('features', [])
    if feats:
        return [f['name'] for f in sorted(feats, key=lambda x: int(x['index']))]
    return obj.get('feature_names', [])


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _load_from_manifests(source_shard_manifest, embedding_manifest):
    src = _load_json(source_shard_manifest)
    emb = _load_json(embedding_manifest)
    shard_paths = src.get('shard_paths', [])
    emb_paths = emb.get('embedding_shard_paths', [])
    if len(shard_paths) != len(emb_paths):
        raise ValueError(f'分片数不一致: source={len(shard_paths)} embedding={len(emb_paths)}')
    feat_list, split_list, z_list = [], [], []
    for sp, ep in zip(shard_paths, emb_paths):
        sd = Path(sp)
        feat_list.append(np.load(sd / 'interaction_feat_style.npy', mmap_mode='r'))
        split_list.append(np.load(sd / 'split.npy', allow_pickle=True))
        z_list.append(np.load(ep, mmap_mode='r'))
    feat = np.concatenate([np.asarray(x) for x in feat_list], axis=0)
    split = np.concatenate([np.asarray(x) for x in split_list], axis=0)
    z = np.concatenate([np.asarray(x) for x in z_list], axis=0)
    if z.shape[0] != feat.shape[0]:
        raise ValueError(f'embedding 与 feature 行数不一致: embedding={z.shape[0]} feature={feat.shape[0]}')
    return feat, split, z


def two_sided_permutation_pvalue(x_a, x_b, num_permutation, rng):
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    observed = float(np.nanmean(x_b) - np.nanmean(x_a))
    combined = np.concatenate([x_a, x_b])
    n_a = len(x_a)
    perm_deltas = []
    for _ in range(num_permutation):
        p = rng.permutation(combined)
        perm_deltas.append(float(np.nanmean(p[n_a:]) - np.nanmean(p[:n_a])))
    perm_deltas = np.asarray(perm_deltas)
    return float((np.sum(np.abs(perm_deltas) >= abs(observed)) + 1) / (num_permutation + 1))


def mk_mmd(x, y, rng, maxn=5000):
    if len(x) > maxn:
        x = x[rng.choice(len(x), maxn, replace=False)]
    if len(y) > maxn:
        y = y[rng.choice(len(y), maxn, replace=False)]
    z = np.vstack([x, y])
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    bws = np.clip(med * np.array([0.25, 0.5, 1, 2, 4]), 1e-6, None)

    def k(a, b):
        dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-dist / (2 * bw * bw)) for bw in bws) / len(bws)

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def emb(context, ckpt, device, bz):
    c = torch.load(ckpt, map_location='cpu')
    model = ContextFlattenGRUEncoder(context.shape[-1], embedding_dim=int(c.get('embedding_dim', 64)))
    model.load_state_dict(c['model'], strict=False)
    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(context), bz):
            batch = torch.from_numpy(context[i:i + bz]).float().to(dev)
            out.append(model(batch).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def resolve_feature(name, fmap, aliases):
    if name in fmap:
        return name
    for alias in aliases.get(name, []):
        if alias in fmap:
            return alias
    return None


def build_slice_tags(features, fmap, idx):
    tags = {}
    if 'speed_mean' in fmap:
        v = float(features[idx, fmap['speed_mean']])
        tags['speed_bin'] = 'low' if v < 5 else ('mid' if v < 15 else 'high')
    if 'mean_thw' in fmap:
        v = float(features[idx, fmap['mean_thw']])
        tags['thw_bin'] = 'tight' if v < 1 else ('normal' if v < 2 else 'safe')
    density_key = 'interaction_density' if 'interaction_density' in fmap else ('neighbor_count' if 'neighbor_count' in fmap else None)
    if density_key:
        v = float(features[idx, fmap[density_key]])
        tags['interaction_density_bin'] = 'sparse' if v < 2 else ('mid' if v < 5 else 'dense')
    if 'front_valid' in fmap:
        tags['front_valid_bin'] = 'valid' if float(features[idx, fmap['front_valid']]) > 0.5 else 'invalid'
    return tags


def main(a):
    out = Path(a.output_dir)
    (out / 'plots').mkdir(parents=True, exist_ok=True)
    warnings = {'warnings': []}

    if not a.smoke_test and not a.embedding_path and not (a.context_traj_path and a.encoder_ckpt):
        raise ValueError('必须提供 --embedding_path，或同时提供 --context_traj_path 和 --encoder_ckpt。')

    rng = np.random.default_rng(a.seed)
    names = load_schema(a.feature_schema_path)
    fmap = {n: i for i, n in enumerate(names)}
    feat = np.load(a.feature_path, mmap_mode='r') if a.feature_path else None

    if a.smoke_test:
        n, d = 400, 32
        synth_feat = np.random.default_rng(1).normal(size=(n, len(names) if names else 10)).astype(np.float32)
        feat = synth_feat
        a_idx = np.arange(0, n // 2)
        b_idx = np.arange(n // 2, n)
        z = np.random.default_rng(2).normal(size=(n, d)).astype(np.float32)
    else:
        a_idx = np.load(a.a_indices_path)
        b_idx = np.load(a.b_indices_path)
        if len(a_idx) == 0 or len(b_idx) == 0:
            raise ValueError('A/B 索引不能为空。')
        if not a.allow_overlap and np.intersect1d(a_idx, b_idx).size > 0:
            raise ValueError('A/B 索引有重叠。如需允许请显式设置 --allow_overlap。')

        if a.source_shard_manifest and a.embedding_manifest:
            feat, split_all, z = _load_from_manifests(a.source_shard_manifest, a.embedding_manifest)
            split_all = split_all.astype(str) if split_all.dtype.kind in {'U', 'S', 'O'} else split_all
            test_idx = np.flatnonzero(split_all == 'test')
            a_idx = test_idx[np.load(a.a_indices_path)] if a.indices_are_test_relative else np.load(a.a_indices_path)
            b_idx = test_idx[np.load(a.b_indices_path)] if a.indices_are_test_relative else np.load(a.b_indices_path)
            warnings['warnings'].append('使用 shard_manifest + embedding_manifest 模式（Stage5 full51 推荐路径）。')
        elif a.embedding_path:
            z = np.load(a.embedding_path, mmap_mode='r')
            warnings['warnings'].append('使用 embedding_path 模式；当 context/feature/split 单体数组缺失时推荐此模式。')
        else:
            ctx = np.load(a.context_traj_path, mmap_mode='r')
            z = emb(np.asarray(ctx, dtype=np.float32), a.encoder_ckpt, a.device, a.batch_size)

        if not np.isfinite(z).all():
            raise ValueError('embedding 含有非有限值。')
        if z.shape[0] != feat.shape[0]:
            raise ValueError(f'embedding 行数 {z.shape[0]} 与 feature 行数 {feat.shape[0]} 不一致。')

    max_index = int(max(np.max(a_idx), np.max(b_idx)))
    if max_index >= feat.shape[0]:
        raise ValueError(f'A/B 索引超出 feature 范围: max={max_index}, feature_rows={feat.shape[0]}')
    if max_index >= z.shape[0]:
        raise ValueError(f'A/B 索引超出 embedding/context 范围: max={max_index}, embed_rows={z.shape[0]}')

    za = np.asarray(z[a_idx], dtype=np.float32)
    zb = np.asarray(z[b_idx], dtype=np.float32)

    cfg = yaml.safe_load(Path(a.feature_groups_config).read_text(encoding='utf-8'))
    aliases = cfg.get('feature_aliases', {})
    groups = cfg.get('category_groups', {})

    mmd = mk_mmd(za, zb, rng, a.max_mmd_samples)
    bs = []
    for _ in range(a.num_bootstrap):
        bs.append(mk_mmd(za[rng.choice(len(za), len(za), replace=True)], zb[rng.choice(len(zb), len(zb), replace=True)], rng, a.max_mmd_samples))
    mix = np.vstack([za, zb])
    n_a = len(za)
    perm = []
    for _ in range(a.num_permutation):
        p = rng.permutation(len(mix))
        perm.append(mk_mmd(mix[p[:n_a]], mix[p[n_a:]], rng, a.max_mmd_samples))

    bdd = {
        'metric': 'BDD_MMD', 'mmd2': float(mmd),
        'ci95_low': float(np.percentile(bs, 2.5)), 'ci95_high': float(np.percentile(bs, 97.5)),
        'p_value': float((np.sum(np.array(perm) >= mmd) + 1) / (len(perm) + 1)),
        'n_A': int(len(a_idx)), 'n_B': int(len(b_idx)), 'embedding_dim': int(za.shape[1]),
    }
    (out / 'bdd_summary.json').write_text(json.dumps(bdd, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame({'mmd2_bootstrap': bs}).to_csv(out / 'bdd_bootstrap_samples.csv', index=False)
    pd.DataFrame({'mmd2_permutation': perm}).to_csv(out / 'bdd_permutation_samples.csv', index=False)

    rows = []
    group_map = {}
    category_feature_map = {}
    for g, v in groups.items():
        resolved, missing = [], []
        for f in v.get('features', []):
            hit = resolve_feature(f, fmap, aliases)
            if hit:
                resolved.append((f, hit, fmap[hit]))
                group_map[hit] = g
            else:
                missing.append(f)
        if not resolved:
            warnings['warnings'].append(f'category={g} 无可解析特征，已跳过。missing={missing}')
            continue

        cols = [x[2] for x in resolved]
        vals = np.asarray(feat[np.r_[a_idx, b_idx]][:, cols], dtype=float)
        med = np.nanmedian(vals, 0)
        iqr = np.nanpercentile(vals, 75, 0) - np.nanpercentile(vals, 25, 0) + 1e-6
        sa = np.asarray(feat[a_idx][:, cols], dtype=float)
        sb = np.asarray(feat[b_idx][:, cols], dtype=float)
        za1, zb1 = (sa - med) / iqr, (sb - med) / iqr
        lower = set(v.get('lower_is_better', []))
        for j, (raw, _, _) in enumerate(resolved):
            if raw in lower:
                za1[:, j] *= -1
                zb1[:, j] *= -1
        score_a, score_b = za1.mean(1), zb1.mean(1)
        pval = two_sided_permutation_pvalue(score_a, score_b, a.num_permutation, rng)
        pooled = np.nanstd(np.r_[score_a, score_b]) + 1e-6
        rows.append({
            'category': g, 'n_features': len(resolved),
            'resolved_features': json.dumps([x[1] for x in resolved], ensure_ascii=False),
            'missing_features': json.dumps(missing, ensure_ascii=False),
            'mean_A': float(np.nanmean(score_a)), 'mean_B': float(np.nanmean(score_b)),
            'delta': float(np.nanmean(score_b) - np.nanmean(score_a)),
            'cohen_d': float((np.nanmean(score_b) - np.nanmean(score_a)) / pooled),
            'p_value': pval, 'positive_direction': v.get('positive_direction', ''),
        })
        category_feature_map[g] = [x[1] for x in resolved]

    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / 'category_delta.csv', index=False)

    frows = []
    for i, n in enumerate(names):
        xa, xb = np.asarray(feat[a_idx, i], float), np.asarray(feat[b_idx, i], float)
        den = np.nanstd(np.r_[xa, xb]) + 1e-6
        pval = two_sided_permutation_pvalue(xa, xb, a.num_permutation, rng)
        frows.append({'feature': n, 'mean_A': np.nanmean(xa), 'mean_B': np.nanmean(xb), 'median_A': np.nanmedian(xa),
                      'median_B': np.nanmedian(xb), 'delta_raw': np.nanmean(xb) - np.nanmean(xa),
                      'delta_normalized': (np.nanmean(xb) - np.nanmean(xa)) / den,
                      'relative_change_percent': 100 * (np.nanmean(xb) - np.nanmean(xa)) / (abs(np.nanmean(xa)) + 1e-6),
                      'cohen_d': (np.nanmean(xb) - np.nanmean(xa)) / den,
                      'permutation_p_value': pval, 'group': group_map.get(n, '')})
    fdf = pd.DataFrame(frows)
    fdf.to_csv(out / 'feature_delta.csv', index=False)

    srows = []
    if 'speed_mean' not in fmap:
        warnings['warnings'].append('缺少 speed_mean，无法构建 speed_bin 切片。')
    else:
        sp = np.asarray(feat[:, fmap['speed_mean']], float)
        q = np.quantile(sp, [1 / 3, 2 / 3])
        bins = np.where(sp < q[0], 'low', np.where(sp < q[1], 'mid', 'high'))
        for b in ['low', 'mid', 'high']:
            ai, bi = a_idx[bins[a_idx] == b], b_idx[bins[b_idx] == b]
            if len(ai) >= a.min_slice_size and len(bi) >= a.min_slice_size:
                srows.append({'slice_name': f'speed_bin:{b}', 'n_A': len(ai), 'n_B': len(bi), 'bdd_mmd': mk_mmd(z[ai], z[bi], rng, 2000)})
    pd.DataFrame(srows).to_csv(out / 'scenario_slice_delta.csv', index=False)

    d = np.linalg.norm(za[:, None, :] - zb[None, :, :], axis=-1)
    ta = np.argsort(d.min(1))[-a.top_k:]
    tb = np.argsort(d.min(0))[-a.top_k:]
    all_vals = np.asarray(feat[np.r_[a_idx, b_idx]], dtype=float)
    med, iqr = np.nanmedian(all_vals, axis=0), np.nanpercentile(all_vals, 75, axis=0) - np.nanpercentile(all_vals, 25, axis=0) + 1e-6

    def row_for(sample_idx, group_name, opp_indices, dist, nearest_idx):
        zvec = (np.asarray(feat[sample_idx], dtype=float) - med) / iqr
        opp_med = np.nanmedian((np.asarray(feat[opp_indices], dtype=float) - med) / iqr, axis=0)
        dif = np.abs(zvec - opp_med)
        top_idx = np.argsort(dif)[-3:][::-1]
        top_features = [names[k] for k in top_idx]
        category_scores = {}
        for cat, fts in category_feature_map.items():
            ids = [fmap[f] for f in fts if f in fmap]
            if ids:
                category_scores[cat] = float(np.mean(np.abs(zvec[ids] - opp_med[ids])))
        dominant = max(category_scores, key=category_scores.get) if category_scores else ''
        return {
            'sample_index': int(sample_idx), 'group': group_name, 'distance_to_opposite': float(dist),
            'nearest_opposite_index': int(nearest_idx), 'dominant_category': dominant,
            'top_changed_features': json.dumps(top_features, ensure_ascii=False),
            'feature_values': json.dumps({k: float(feat[sample_idx, fmap[k]]) for k in top_features if k in fmap}, ensure_ascii=False),
            'slice_tags': json.dumps(build_slice_tags(feat, fmap, sample_idx), ensure_ascii=False),
            'scenario_id': '', 'video_path': ''
        }

    tops = []
    for i in ta:
        n = int(b_idx[d[i].argmin()])
        tops.append(row_for(int(a_idx[i]), 'A', b_idx, d[i].min(), n))
    for j in tb:
        n = int(a_idx[d[:, j].argmin()])
        tops.append(row_for(int(b_idx[j]), 'B', a_idx, d[:, j].min(), n))
    pd.DataFrame(tops).sort_values('distance_to_opposite', ascending=False).to_csv(out / 'top_drift_cases.csv', index=False)

    warnings['warnings'].append('BDD 量纲未经负/正对照标定前不可用于绝对阈值决策。')
    warnings['warnings'].append('缺少视频/场景元数据时，top drift case 的场景解释为 proxy 级别。')
    (out / 'stage6_warnings.json').write_text(json.dumps(warnings, indent=2, ensure_ascii=False), encoding='utf-8')

    if not cdf.empty:
        cdf.plot(x='category', y='delta', kind='bar'); plt.tight_layout(); plt.savefig(out / 'plots/category_delta_bar.png'); plt.close()
    if not fdf.empty:
        top = fdf.reindex(fdf.delta_normalized.abs().sort_values(ascending=False).head(20).index)
        top.plot(x='feature', y='delta_normalized', kind='bar'); plt.tight_layout(); plt.savefig(out / 'plots/feature_delta_bar_top20.png'); plt.close()
    plt.hist(bs, bins=30); plt.axvline(mmd, color='r'); plt.tight_layout(); plt.savefig(out / 'plots/bdd_bootstrap_distribution.png'); plt.close()
    pca = PCA(n_components=2).fit_transform(np.vstack([za, zb]))
    plt.scatter(pca[:len(za), 0], pca[:len(za), 1], s=4, label='A'); plt.scatter(pca[len(za):, 0], pca[len(za):, 1], s=4, label='B')
    plt.legend(); plt.tight_layout(); plt.savefig(out / 'plots/embedding_pca.png'); plt.close()

    subprocess.run([sys.executable, 'tools/stage6_generate_report_card.py', '--input_dir', str(out)], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embedding_path')
    p.add_argument('--context_traj_path')
    p.add_argument('--context_mask_path')
    p.add_argument('--context_mask_window_path')
    p.add_argument('--feature_path', required=False)
    p.add_argument('--feature_schema_path', required=False)
    p.add_argument('--encoder_ckpt')
    p.add_argument('--a_indices_path')
    p.add_argument('--b_indices_path')
    p.add_argument('--feature_groups_config', default='configs/stage6_feature_groups.yaml')
    p.add_argument('--source_shard_manifest', help='Stage5 full51 推荐输入：shard_manifest.json')
    p.add_argument('--embedding_manifest', help='Stage5D 导出 embedding_manifest.json')
    p.add_argument('--indices_are_test_relative', action='store_true', help='若 A/B 索引基于 test 子集位置，则自动映射到全局行号')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--num_bootstrap', type=int, default=200)
    p.add_argument('--num_permutation', type=int, default=500)
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--max_mmd_samples', type=int, default=5000)
    p.add_argument('--min_slice_size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow_overlap', action='store_true')
    p.add_argument('--smoke_test', action='store_true')
    args = p.parse_args()

    if not args.smoke_test:
        for req in ['feature_schema_path', 'a_indices_path', 'b_indices_path']:
            if getattr(args, req) is None:
                raise ValueError(f'缺少必需参数 --{req}')

    if not args.smoke_test:
        if not args.feature_path and not (args.source_shard_manifest and args.embedding_manifest):
            raise ValueError('请提供 --feature_path（legacy），或提供 --source_shard_manifest + --embedding_manifest（推荐）。')
    main(args)
