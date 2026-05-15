#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FILES = [
    'ego_seq.npy',
    'neighbor_seq.npy',
    'context_traj.npy',
    'context_mask.npy',
    'context_mask_window.npy',
    'neighbor_slot_ids.npy',
    'meta.npy',
    'split.npy',
    'interaction_feat_style_raw.npy',
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _sum_nested_dict(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            if k not in dst or not isinstance(dst[k], dict):
                dst[k] = {}
            _sum_nested_dict(dst[k], v)
        elif isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v


def _validate_root(root: Path, summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get('nonfinite_output_detected') != 0:
        errors.append(f'{root}: nonfinite_output_detected != 0')
    if float(summary.get('fallback_assignment_rate', -1.0)) != 0.0:
        errors.append(f'{root}: fallback_assignment_rate != 0')
    if float(summary.get('good_lane_context_rate', -1.0)) != 1.0:
        errors.append(f'{root}: good_lane_context_rate != 1')
    if int(summary.get('n_windows_kept', 0)) <= 0:
        errors.append(f'{root}: n_windows_kept <= 0')

    for sp in summary.get('shard_paths', []):
        shard = Path(sp)
        if not shard.is_absolute():
            shard = root / shard
        if not shard.exists():
            errors.append(f'{root}: shard path missing: {shard}')
            continue
        for fn in REQUIRED_FILES:
            if not (shard / fn).exists():
                errors.append(f'{root}: missing {fn} in {shard}')
    return errors


def _recompute_standardization(shard_paths: list[str], out_dir: Path) -> str:
    total_sum = None
    total_sumsq = None
    total_count = 0
    feat_dim = None

    for shard_path in shard_paths:
        shard = Path(shard_path)
        raw = np.load(shard / 'interaction_feat_style_raw.npy')
        splits = np.load(shard / 'split.npy', allow_pickle=True)
        mask = (splits.astype(str) == 'train')
        train_raw = raw[mask]
        if train_raw.shape[0] == 0:
            continue
        if feat_dim is None:
            feat_dim = train_raw.shape[1]
            total_sum = np.zeros((feat_dim,), dtype=np.float64)
            total_sumsq = np.zeros((feat_dim,), dtype=np.float64)
        total_sum += train_raw.sum(axis=0)
        total_sumsq += np.square(train_raw).sum(axis=0)
        total_count += int(train_raw.shape[0])

    if feat_dim is None:
        raise RuntimeError('No train rows found across shards; cannot recompute global standardization.')

    mean = total_sum / max(1, total_count)
    var = np.maximum(total_sumsq / max(1, total_count) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    std_safe = np.where(std < 1e-6, 1e-6, std)

    for shard_path in shard_paths:
        shard = Path(shard_path)
        raw = np.load(shard / 'interaction_feat_style_raw.npy')
        standardized = ((raw - mean) / std_safe).astype(np.float32)
        np.save(shard / 'interaction_feat_style.npy', standardized)

    standardization = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'train_count': int(total_count),
        'clip_value': None,
    }
    _write_json(out_dir / 'interaction_feature_standardization.json', standardization)
    _write_json(out_dir / 'global_feature_standardization_report.json', {
        'message': 'Global standardization recomputed from train split only across all shards.',
        'train_count': int(total_count),
        'feature_dim': int(feat_dim),
    })
    return 'interaction_feature_standardization.json'


def _validate_manifest(manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    shard_paths = [Path(p) for p in manifest.get('shard_paths', [])]
    if not shard_paths:
        raise RuntimeError('manifest has no shard_paths')
    total_rows = 0
    split_counts = defaultdict(int)
    errors = []
    for shard in shard_paths:
        if not shard.exists():
            errors.append(f'shard missing: {shard}')
            continue
        for fn in REQUIRED_FILES + ['interaction_feat_style.npy']:
            if not (shard / fn).exists():
                errors.append(f'missing {fn} in {shard}')
        split = np.load(shard / 'split.npy', allow_pickle=True)
        n = int(split.shape[0])
        if n == 0:
            errors.append(f'zero rows in shard: {shard}')
        total_rows += n
        for s in split.astype(str):
            split_counts[str(s)] += 1
        if (shard / 'interaction_feat_style.npy').exists():
            feat = np.load(shard / 'interaction_feat_style.npy')
            if not np.isfinite(feat).all():
                errors.append(f'NaN/Inf in interaction_feat_style.npy: {shard}')

    if int(manifest.get('total_windows', -1)) != total_rows:
        errors.append(f"total_windows mismatch: manifest={manifest.get('total_windows')} actual={total_rows}")

    summary_path = manifest_path.parent / 'build_summary.json'
    if summary_path.exists():
        summary = _read_json(summary_path)
        if int(summary.get('n_windows_kept', -1)) != total_rows:
            errors.append('build_summary n_windows_kept mismatch')
        for k, v in summary.get('split_counts', {}).items():
            if int(v) != int(split_counts.get(k, 0)):
                errors.append(f'build_summary split_counts mismatch for {k}')

    if errors:
        raise RuntimeError('\n'.join(errors))


def main() -> None:
    p = argparse.ArgumentParser(description='Merge Stage 5A shard outputs by manifest/metadata without monolithic tensor concat.')
    p.add_argument('--input_roots', nargs='*', default=[])
    p.add_argument('--out_dir', type=Path)
    p.add_argument('--recompute_global_standardization', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--validate_only', action='store_true')
    args = p.parse_args()

    if args.validate_only:
        if not args.manifest:
            raise ValueError('--validate_only requires --manifest')
        _validate_manifest(args.manifest)
        print('Validation passed.')
        return

    if not args.input_roots or args.out_dir is None:
        raise ValueError('merge mode requires --input_roots and --out_dir')

    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f'{out_dir} exists and not empty; use --overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_shards: list[str] = []
    errors: list[str] = []
    for r in args.input_roots:
        root = Path(r)
        summary = _read_json(root / 'build_summary.json')
        summaries.append(summary)
        errors.extend(_validate_root(root, summary))
        for sp in summary.get('shard_paths', []):
            shard = Path(sp)
            if not shard.is_absolute():
                shard = root / shard
            all_shards.append(str(shard.resolve()))

    if errors:
        raise RuntimeError('\n'.join(errors))

    merged: dict[str, Any] = {
        'dataset_type': 'waymo_5neighbor_context_laneaware',
        'merged_dataset_type': 'waymo_5neighbor_context',
        'output_format': 'sharded',
        'input_roots': [str(Path(r).resolve()) for r in args.input_roots],
        'shard_paths': all_shards,
        'n_shards': len(all_shards),
        'total_shards': len(all_shards),
    }

    keys_to_sum = [
        'n_windows_kept', 'n_windows_total', 'n_windows_filtered_static', 'n_windows_filtered_invalid',
        'fallback_assignment_count_kept', 'lane_assignment_success_count_kept',
        'current_lane_found_count_kept', 'left_lane_found_count_kept', 'right_lane_found_count_kept',
        'nonfinite_output_detected', 'static_front_count',
    ]
    for k in keys_to_sum:
        merged[k] = int(sum(int(s.get(k, 0)) for s in summaries))

    merged['split_counts'] = {}
    for s in summaries:
        for k, v in s.get('split_counts', {}).items():
            merged['split_counts'][k] = int(merged['split_counts'].get(k, 0) + int(v))

    n_kept = max(1, int(merged['n_windows_kept']))

    empty_counts = defaultdict(int)
    for s in summaries:
        for slot, c in s.get('empty_slot_count_by_slot', {}).items():
            empty_counts[slot] += int(c)
    merged['empty_slot_count_by_slot'] = dict(empty_counts)
    merged['empty_slot_ratio_by_slot'] = {k: float(v) / n_kept for k, v in empty_counts.items()}

    slot_valid_ratio = {}
    for s in summaries:
        if 'slot_valid_count_by_slot' in s:
            continue
    if all('slot_valid_count_by_slot' in s for s in summaries):
        val_counts = defaultdict(int)
        for s in summaries:
            for k, v in s['slot_valid_count_by_slot'].items():
                val_counts[k] += int(v)
        slot_valid_ratio = {k: float(v) / n_kept for k, v in val_counts.items()}
        merged['slot_valid_count_by_slot'] = dict(val_counts)
    else:
        weights = [max(0, int(s.get('n_windows_kept', 0))) for s in summaries]
        denom = max(1, sum(weights))
        slots = set()
        for s in summaries:
            slots.update(s.get('slot_valid_ratio', {}).keys())
        for slot in slots:
            num = 0.0
            for s, w in zip(summaries, weights):
                num += float(s.get('slot_valid_ratio', {}).get(slot, 0.0)) * w
            slot_valid_ratio[slot] = num / denom
    merged['slot_valid_ratio'] = slot_valid_ratio

    merged_nested: dict[str, Any] = {}
    for field in ['assignment_method_counts_by_slot', 'static_neighbor_count_by_slot', 'lane_context_quality_counts']:
        tmp: dict[str, Any] = {}
        for s in summaries:
            _sum_nested_dict(tmp, s.get(field, {}))
        merged[field] = tmp

    merged['fallback_assignment_rate'] = float(merged['fallback_assignment_count_kept']) / n_kept
    good_cnt = merged.get('lane_context_quality_counts', {}).get('good', merged['n_windows_kept'])
    merged['good_lane_context_rate'] = float(good_cnt) / n_kept
    merged['total_windows'] = int(merged['n_windows_kept'])

    std_path = ''
    if args.recompute_global_standardization:
        std_path = _recompute_standardization(all_shards, out_dir)
    merged['interaction_feature_standardization'] = std_path or 'interaction_feature_standardization.json'

    manifest = {
        'merged_dataset_type': 'waymo_5neighbor_context',
        'input_roots': merged['input_roots'],
        'shard_paths': all_shards,
        'total_shards': len(all_shards),
        'total_windows': int(merged['n_windows_kept']),
        'required_files_per_shard': REQUIRED_FILES + ['interaction_feat_style.npy'],
        'global_standardization_path': str((out_dir / 'interaction_feature_standardization.json').resolve()),
    }
    _write_json(out_dir / 'shard_manifest.json', manifest)
    _write_json(out_dir / 'merged_build_summary.json', merged)
    _write_json(out_dir / 'build_summary.json', merged)

    report = """# Stage 5A 分片合并报告

- 本次合并为 **manifest/元数据级别** 合并，不拼接大型 `.npy` 张量。
- `ego_seq.npy` / `neighbor_seq.npy` / `context_traj.npy` 仍保留在原分片目录中。
- 全局标准化使用所有分片的 `train` split 样本重新计算（仅 train 参与统计）。
- 该策略可避免一次性拼接大数组导致的 OOM 风险。
- Stage 5B 训练请使用 `shard_manifest.json` 作为入口进行分片读取。
"""
    (out_dir / 'build_report.md').write_text(report, encoding='utf-8')
    print(f'Merged {len(all_shards)} shards into manifest: {out_dir / "shard_manifest.json"}')


if __name__ == '__main__':
    main()
