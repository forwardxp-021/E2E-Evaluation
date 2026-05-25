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


def _safe_iqr(x):
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    return iqr if iqr > 1e-8 else 1.0


def _robust_z(x):
    med = np.nanmedian(x)
    return np.nan_to_num((x - med) / _safe_iqr(x), nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_aliases(name_to_idx, alias_groups):
    resolved, missing = {}, {}
    for gname, aliases in alias_groups.items():
        found = [a for a in aliases if a in name_to_idx]
        if found:
            resolved[gname] = found
        else:
            missing[gname] = aliases
    return resolved, missing


def _group_signal(feat_eval, name_to_idx, aliases):
    cols = [feat_eval[:, name_to_idx[a]] for a in aliases if a in name_to_idx]
    if not cols:
        return None
    mat = np.stack([_robust_z(np.asarray(c, dtype=float)) for c in cols], axis=1)
    return np.nanmean(mat, axis=1)


def _quantile_select(score, q_top, max_size=None):
    n = score.shape[0]
    k = max(1, int(np.floor(n * q_top)))
    order = np.argsort(score)
    idx = order[-k:]
    if max_size is not None and idx.shape[0] > max_size:
        idx = idx[np.argsort(score[idx])[-max_size:]]
    return idx


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

    feat_names = load_schema(a.feature_schema_path)
    if feat.shape[1] != len(feat_names):
        raise ValueError(f'特征维度与 schema 不一致: feat_dim={feat.shape[1]}, schema_dim={len(feat_names)}')
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    eval_idx = np.flatnonzero(split == a.eval_split)
    feat_eval = np.asarray(feat[eval_idx], dtype=float)
    rng = np.random.default_rng(a.seed)
    criteria = []
    resolved_features = {}
    missing_features = {}
    score_definition = {}
    overlap_removed = 0

    if a.mode == 'negative_control_random':
        sel = rng.permutation(eval_idx)
        mid = len(sel) // 2
        A, B = sel[:mid], sel[mid:]
        criteria = ['random split on eval_split rows']
    else:
        if a.mode == 'pseudo_style_aggressive_vs_conservative':
            alias_groups = {
                'thw': ['mean_thw', 'thw_mean', 'min_thw', 'thw_min'],
                'jerk': ['rms_jerk', 'max_abs_jerk'],
                'accel': ['rms_accel', 'max_abs_accel'],
                'assertiveness': ['lane_change_count_proxy', 'lane_change_left_count_proxy', 'lane_change_right_count_proxy', 'front_pressure_score', 'yielding_score_proxy'],
                'lateral_activity': ['rms_yaw_rate', 'rms_curvature', 'heading_change_total'],
            }
            signs_cons = {'thw': +1.0, 'jerk': -1.0, 'accel': -1.0, 'assertiveness': -1.0, 'lateral_activity': -1.0}
            group_meaning = {'A': 'conservative_like', 'B': 'aggressive_like'}
        elif a.mode == 'scene_confounding_control':
            alias_groups = {
                'speed': ['speed_mean', 'ego_speed_mean', 'speed_norm_mean', 'speed_std', 'speed_norm_std'],
                'lateral_activity': ['lane_change_count_proxy', 'lane_change_left_count_proxy', 'lane_change_right_count_proxy', 'rms_yaw_rate', 'rms_curvature', 'heading_change_total'],
                'interaction_pressure': ['front_pressure_score', 'neighbor_count', 'neighbor_valid_count', 'front_valid_ratio', 'yielding_score_proxy'],
                'gap_size': ['left_front_min_gap', 'right_front_min_gap', 'left_rear_min_gap', 'right_rear_min_gap'],
            }
            signs_cons = {'speed': -0.3, 'lateral_activity': -1.0, 'interaction_pressure': -1.0, 'gap_size': +1.0}
            group_meaning = {'A': 'easy_scene_like', 'B': 'complex_scene_like'}
        else:
            raise ValueError(f'未知模式: {a.mode}')

        resolved, missing = _resolve_aliases(name_to_idx, alias_groups)
        resolved_features = resolved
        missing_features = missing
        if len(resolved) < 2:
            raise ValueError(f'{a.mode} 可用特征组过少（{len(resolved)}），无法构造稳定分组。')

        agg = np.zeros(len(eval_idx), dtype=float)
        score_definition = {'group_signs': signs_cons, 'normalization': 'robust_z = (x - median) / IQR on eval_split rows'}
        for gname, aliases in resolved.items():
            s = _group_signal(feat_eval, name_to_idx, aliases)
            if s is None:
                continue
            agg += float(signs_cons.get(gname, 0.0)) * s
        conservative_score = agg
        aggressive_score = -agg
        qa = _quantile_select(conservative_score, a.q_low, max_size=a.max_group_size)
        qb = _quantile_select(aggressive_score, 1.0 - a.q_high, max_size=a.max_group_size)
        sa, sb = set(qa.tolist()), set(qb.tolist())
        ov = sa & sb
        overlap_removed = len(ov)
        if ov:
            ov_arr = np.array(sorted(ov), dtype=int)
            drop_from_a = conservative_score[ov_arr] <= aggressive_score[ov_arr]
            remove_a = set(ov_arr[drop_from_a].tolist())
            qa = np.array([i for i in qa if i not in remove_a and i not in ov], dtype=int)
            qb = np.array([i for i in qb if i not in (ov - remove_a)], dtype=int)
        if len(qa) < a.min_group_size or len(qb) < a.min_group_size:
            raise ValueError(f'{a.mode} 分组样本不足: n_A={len(qa)} n_B={len(qb)} min_group_size={a.min_group_size}')
        A = eval_idx[qa]
        B = eval_idx[qb]
        criteria = [
            f'quantile based score split, q_low={a.q_low}, q_high={a.q_high}, min_group_size={a.min_group_size}',
            f"A={group_meaning['A']}, B={group_meaning['B']}",
        ]
        score_definition['group_meaning'] = group_meaning

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
        'resolved_features': resolved_features,
        'missing_features': missing_features,
        'criteria': criteria,
        'score_definition': score_definition,
        'quantiles': {'q_low': float(a.q_low), 'q_high': float(a.q_high)},
        'overlap_removed': int(overlap_removed),
        'warnings': warnings,
    }
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'split_report.md').write_text(
        f"# Stage6 A/B Split\n\n- mode: {a.mode}\n- eval_split: {a.eval_split}\n- n_A: {len(A)}\n- n_B: {len(B)}\n- criteria: {criteria}\n- overlap_removed: {overlap_removed}\n",
        encoding='utf-8'
    )


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
    p.add_argument('--min_group_size', type=int, default=500)
    p.add_argument('--max_group_size', type=int, default=None)
    p.add_argument('--q_low', type=float, default=0.3)
    p.add_argument('--q_high', type=float, default=0.7)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if not args.shard_manifest and (not args.feature_path or not args.split_path):
        raise ValueError('请提供 --shard_manifest，或同时提供 --feature_path 与 --split_path（legacy）。')
    main(args)
