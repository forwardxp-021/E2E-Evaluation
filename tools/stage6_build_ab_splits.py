#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _normalize_split(split):
    arr = np.asarray(split)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        return np.asarray([str(x) for x in arr], dtype=object)
    mapping = {0: 'train', 1: 'val', 2: 'test'}
    out = []
    for x in arr:
        xi = int(x)
        out.append(mapping.get(xi, str(xi)))
    return np.asarray(out, dtype=object)


def _load_manifest_feature_split(shard_manifest):
    m = _load_json(shard_manifest)
    base = Path(shard_manifest).parent
    shards = m.get('shards', m.get('shard_infos', []))
    shard_paths = [s['shard_path'] for s in shards] if shards else m.get('shard_paths', [])
    all_feat, all_split, rows = [], [], []
    g = 0
    for sid, sp in enumerate(shard_paths):
        sd = base / sp
        feat = np.load(sd / 'interaction_feat_style.npy', mmap_mode='r')
        split = _normalize_split(np.load(sd / 'split.npy', allow_pickle=True))
        if feat.shape[0] != split.shape[0]:
            raise ValueError(f'shard 行数不一致: {sp} feature={feat.shape[0]} split={split.shape[0]}')
        all_feat.append(np.asarray(feat))
        all_split.append(split)
        for i, sv in enumerate(split):
            rows.append({'global_row': g + i, 'shard_id': sid, 'shard_path': str(sp), 'local_row': i, 'split': str(sv)})
        g += feat.shape[0]
    return np.concatenate(all_feat, 0), np.concatenate(all_split, 0), pd.DataFrame(rows), shard_paths


def load_schema(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    feats = obj.get('features', [])
    return [f['name'] for f in sorted(feats, key=lambda x: int(x['index']))] if feats else obj.get('feature_names', [])


def main(a):
    out = Path(a.output_dir) / a.experiment_name
    if out.exists() and not a.overwrite:
        raise FileExistsError(f'输出目录已存在: {out}；如需覆盖请加 --overwrite')
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    warnings = []
    if a.shard_manifest:
        feat, split, row_df, shard_paths = _load_manifest_feature_split(a.shard_manifest)
    else:
        feat = np.load(a.feature_path, mmap_mode='r')
        split = _normalize_split(np.load(a.split_path, allow_pickle=True))
        row_df = pd.DataFrame({'global_row': np.arange(len(split)), 'shard_id': -1, 'shard_path': 'legacy_flat', 'local_row': np.arange(len(split)), 'split': split.astype(str)})
        shard_paths = []

    eval_idx = np.flatnonzero(split == a.eval_split)
    rng = np.random.default_rng(a.seed)
    sel = rng.permutation(eval_idx)
    mid = len(sel) // 2
    A, B = sel[:mid], sel[mid:]

    np.save(out / 'a_indices.npy', A)
    np.save(out / 'b_indices.npy', B)
    row_df.set_index('global_row').loc[A].reset_index().to_csv(out / 'a_indices.csv', index=False)
    row_df.set_index('global_row').loc[B].reset_index().to_csv(out / 'b_indices.csv', index=False)

    split_counts = {str(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))}
    summary = {
        'mode': a.mode,
        'eval_split': a.eval_split,
        'n_A': int(len(A)),
        'n_B': int(len(B)),
        'global_row_indices': True,
        'shard_manifest': a.shard_manifest,
        'feature_schema_path': a.feature_schema_path,
        'seed': int(a.seed),
        'n_shards': len(shard_paths),
        'total_rows': int(len(split)),
        'split_counts': split_counts,
        'criteria': ['random split on eval_split rows'],
        'warnings': warnings,
    }
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'split_report.md').write_text(f"# Stage6 A/B Split\n\n- eval_split: {a.eval_split}\n- n_A: {len(A)}\n- n_B: {len(B)}\n", encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['negative_control_random', 'pseudo_style_aggressive_vs_conservative', 'scene_confounding_control'], required=True)
    p.add_argument('--feature_path')
    p.add_argument('--feature_schema_path', required=True)
    p.add_argument('--split_path')
    p.add_argument('--shard_manifest')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--output_dir', default='outputs/stage6A_splits')
    p.add_argument('--experiment_name', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.shard_manifest and (not args.feature_path or not args.split_path):
        raise ValueError('请提供 --shard_manifest，或同时提供 --feature_path 与 --split_path（legacy）。')
    main(args)
