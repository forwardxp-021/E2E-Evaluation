#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _iter_shard_feature_split(shard_manifest_path):
    manifest = _load_json(shard_manifest_path)
    base = Path(shard_manifest_path).parent
    shards = manifest.get('shards', manifest.get('shard_infos', []))
    if shards:
        shard_paths = [s['shard_path'] for s in shards]
    else:
        shard_paths = manifest.get('shard_paths', [])
    for shard in shard_paths:
        d = base / shard
        yield np.load(d / 'interaction_feat_style.npy', mmap_mode='r'), np.load(d / 'split.npy', allow_pickle=True)


def _load_feature_split(args):
    if args.shard_manifest:
        feats, splits = [], []
        for f, s in _iter_shard_feature_split(args.shard_manifest):
            feats.append(np.asarray(f))
            splits.append(np.asarray(s))
        if not feats:
            raise ValueError(f'shard_manifest 中未找到有效分片: {args.shard_manifest}')
        return np.concatenate(feats, axis=0), np.concatenate(splits, axis=0)
    return np.load(args.feature_path, mmap_mode='r'), np.load(args.split_path, allow_pickle=True)


def load_schema(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    feats = obj.get('features', [])
    return [f['name'] for f in sorted(feats, key=lambda x: int(x['index']))] if feats else obj.get('feature_names', [])


def fmap(names):
    return {n: i for i, n in enumerate(names)}


def col(arr, m, keys, warns):
    for k in keys:
        if k in m:
            return arr[:, m[k]], k
    warns.append(f'missing feature candidates: {keys}')
    return None, None


def main(a):
    out = Path(a.output_dir) / a.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    feat, split = _load_feature_split(a)
    split = split.astype(str) if split.dtype.kind in {'U', 'S', 'O'} else np.array([['train', 'val', 'test'][int(x)] if int(x) in [0, 1, 2] else str(int(x)) for x in split], dtype=object)
    test_idx = np.flatnonzero(split == 'test')
    names = load_schema(a.feature_schema_path)
    m = fmap(names)
    warns, rng = [], np.random.default_rng(a.seed)

    if a.mode == 'negative_control_random':
        sel = rng.permutation(test_idx)
        mid = len(sel) // 2
        A, B = sel[:mid], sel[mid:]
        criteria = ['random split on test indices']

    elif a.mode == 'pseudo_style_aggressive_vs_conservative':
        score = np.zeros(len(test_idx), dtype=float)
        used = []
        rules = [(['mean_thw'], +1), (['min_thw'], +1), (['rms_jerk'], -1), (['assertiveness_score_proxy', 'assertiveness_proxy'], -1)]
        for keys, sign in rules:
            v, n = col(feat[test_idx], m, keys, warns)
            if v is None:
                continue
            s = (v - np.nanmedian(v)) / (np.nanpercentile(v, 75) - np.nanpercentile(v, 25) + 1e-6)
            score += sign * s
            used.append((n, sign))
        if len(used) == 0:
            raise ValueError('pseudo_style_aggressive_vs_conservative 无可用特征，无法构建评分。')
        lo, hi = np.quantile(score, [a.q_low, a.q_high])
        A, B = test_idx[score >= hi], test_idx[score <= lo]
        criteria = [f'conservative score high>=q{a.q_high}, aggressive score low<=q{a.q_low}', f'used={used}']

    else:
        s, sk = col(feat[test_idx], m, ['ego_speed_mean', 'speed_mean'], warns)
        d, dk = col(feat[test_idx], m, ['interaction_density', 'neighbor_count'], warns)
        if s is None or d is None:
            if not a.allow_degraded_split:
                raise ValueError('scene_confounding_control 缺少 speed 或 density proxy；如要降级请设置 --allow_degraded_split。')
            warns.append('degraded split: scene_confounding_control 代理特征不完整。')
            s = np.zeros(len(test_idx)) if s is None else s
            d = np.zeros(len(test_idx)) if d is None else d
        s_lo, s_hi = np.quantile(s, [a.q_low, a.q_high])
        d_lo, d_hi = np.quantile(d, [a.q_low, a.q_high])
        A = test_idx[(s <= s_lo) & (d >= d_hi)]
        B = test_idx[(s >= s_hi) & (d <= d_lo)]
        criteria = [f'A: low_speed<=q{a.q_low} & dense>=q{a.q_high}', f'B: high_speed>=q{a.q_high} & sparse<=q{a.q_low}', f'used_speed={sk}, used_density={dk}']

    np.save(out / 'a_indices.npy', A)
    np.save(out / 'b_indices.npy', B)
    summary = {'mode': a.mode, 'eval_split': 'test', 'n_A': int(len(A)), 'n_B': int(len(B)), 'global_row_indices': True, 'shard_manifest': a.shard_manifest, 'feature_schema_path': a.feature_schema_path, 'seed': int(a.seed), 'criteria': criteria, 'warnings': warns}
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['negative_control_random', 'pseudo_style_aggressive_vs_conservative', 'scene_confounding_control'], required=True)
    p.add_argument('--feature_path')
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--split_path')
    p.add_argument('--shard_manifest', help='推荐：Stage5 full51 使用 shard_manifest.json')
    p.add_argument('--output_dir', default='outputs/stage6A_splits')
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--q_low', type=float, default=0.3)
    p.add_argument('--q_high', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow_degraded_split', action='store_true')
    args = p.parse_args()
    if not args.shard_manifest and (not args.feature_path or not args.split_path):
        raise ValueError('请提供 --shard_manifest，或同时提供 --feature_path 与 --split_path（legacy）。')
    main(args)
