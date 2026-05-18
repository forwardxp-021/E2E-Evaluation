#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SLOTS = ['front', 'left_front', 'left_rear', 'right_front', 'right_rear']
ALLOWED_METHODS = ['lane_aware', 'geometric_fallback', 'empty', 'sanitize_failed']
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


def resolve_shard_path(root: Path, sp: str) -> Path:
    p = Path(sp)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(p)
        candidates.append(root / p)
        candidates.append(root / 'shards' / p.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f'Cannot resolve shard path {sp} under root {root}; tried: {candidates}')


def _validate_root(root: Path, summary: dict[str, Any]) -> tuple[list[str], list[Path]]:
    errors: list[str] = []
    shards: list[Path] = []
    for sp in summary.get('shard_paths', []):
        try:
            shard = resolve_shard_path(root, str(sp))
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        shards.append(shard)
        for fn in REQUIRED_FILES:
            if not (shard / fn).exists():
                errors.append(f'{root}: missing {fn} in {shard}')
    if not shards:
        errors.append(f'{root}: no valid shard paths found in build_summary.json')
    return errors, shards


def _meta_value_counter(meta: np.ndarray, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for v in meta[field]:
        if isinstance(v, bytes):
            k = v.decode('utf-8', errors='ignore')
        else:
            k = str(v)
        counts[k] += 1
    return counts


def _recompute_standardization(shard_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    total_sum = None
    total_sumsq = None
    total_count = 0
    feat_dim = None

    for shard in shard_paths:
        raw = np.load(shard / 'interaction_feat_style_raw.npy')
        splits = np.load(shard / 'split.npy', allow_pickle=True).astype(str)
        train_mask = splits == 'train'
        train_raw = raw[train_mask]
        if train_raw.shape[0] == 0:
            continue
        if feat_dim is None:
            feat_dim = int(train_raw.shape[1])
            total_sum = np.zeros((feat_dim,), dtype=np.float64)
            total_sumsq = np.zeros((feat_dim,), dtype=np.float64)
        total_sum += train_raw.sum(axis=0)
        total_sumsq += np.square(train_raw).sum(axis=0)
        total_count += int(train_raw.shape[0])

    if feat_dim is None or total_sum is None or total_sumsq is None:
        raise RuntimeError('No train rows found across shards; cannot recompute global standardization.')

    mean = total_sum / total_count
    var = np.maximum(total_sumsq / total_count - np.square(mean), 1e-12)
    std = np.sqrt(var)
    std_safe = np.where(std < 1e-6, 1e-6, std)

    for shard in shard_paths:
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
    _write_json(
        out_dir / 'global_feature_standardization_report.json',
        {
            'message': 'Global standardization recomputed from train split only across all shards.',
            'train_count': int(total_count),
            'feature_dim': int(feat_dim),
        },
    )
    return {'path': 'interaction_feature_standardization.json', 'train_count': int(total_count)}


def _aggregate_from_shards(shard_paths: list[Path], validate_only: bool) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    split_counts: defaultdict[str, int] = defaultdict(int)
    slot_occ_counts: defaultdict[str, int] = defaultdict(int)
    empty_counts: defaultdict[str, int] = defaultdict(int)
    slot_frame_counts: defaultdict[str, int] = defaultdict(int)
    lane_quality_counts: Counter[str] = Counter()
    assign_method: dict[str, defaultdict[str, int]] = {
        slot: defaultdict(int) for slot in SLOTS
    }

    n_windows_kept = 0
    total_frames = 0
    fallback_used_count = 0
    lane_success_count = 0
    nonfinite_detected = 0

    for shard in shard_paths:
        for fn in REQUIRED_FILES:
            if not (shard / fn).exists():
                raise RuntimeError(f'missing {fn} in {shard}')

        split = np.load(shard / 'split.npy', allow_pickle=True).astype(str)
        n = int(split.shape[0])

        def _rows(fname: str) -> int:
            return int(np.load(shard / fname, allow_pickle=True).shape[0])

        for fname in ['meta.npy', 'ego_seq.npy', 'neighbor_seq.npy', 'context_traj.npy', 'context_mask_window.npy', 'interaction_feat_style_raw.npy']:
            rn = _rows(fname)
            if rn != n:
                raise RuntimeError(f'row mismatch in {shard}: split={n}, {fname}={rn}')

        n_windows_kept += n
        for s in split:
            split_counts[str(s)] += 1

        cmw = np.load(shard / 'context_mask_window.npy')
        if cmw.shape[0] != n or cmw.shape[1] != 5:
            raise RuntimeError(f'context_mask_window.npy shape invalid in {shard}: {cmw.shape}, expected [N, 5]')
        occ = cmw.astype(bool)
        for i, slot in enumerate(SLOTS):
            c = int(occ[:, i].sum())
            slot_occ_counts[slot] += c
            empty_counts[slot] += n - c

        cm_path = shard / 'context_mask.npy'
        if cm_path.exists():
            cm = np.load(cm_path)
            if cm.shape[0] == n and cm.shape[-1] == 5:
                total_frames += int(cm.shape[0] * cm.shape[1])
                valid = cm.astype(bool)
                for i, slot in enumerate(SLOTS):
                    slot_frame_counts[slot] += int(valid[:, :, i].sum())

        meta = np.load(shard / 'meta.npy', allow_pickle=True)
        if meta.dtype.names and 'lane_context_quality' in meta.dtype.names:
            lane_quality_counts.update(_meta_value_counter(meta, 'lane_context_quality'))
        if meta.dtype.names and 'fallback_used' in meta.dtype.names:
            fallback_used_count += int(np.asarray(meta['fallback_used']).astype(np.int64).sum())
        if meta.dtype.names and 'lane_assignment_success' in meta.dtype.names:
            lane_success_count += int(np.asarray(meta['lane_assignment_success']).astype(np.int64).sum())

        debug_csv = shard / 'lane_assignment_debug.csv'
        if debug_csv.exists():
            with debug_csv.open('r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if len(rows) != n * len(SLOTS):
                warnings.append(f'{shard}: lane_assignment_debug.csv 行数异常，期望 {n * len(SLOTS)}，实际 {len(rows)}')
            for r in rows:
                slot = r.get('slot', '')
                method = r.get('assignment_method', '')
                if slot not in SLOTS:
                    continue
                if method not in ALLOWED_METHODS:
                    method = 'sanitize_failed'
                assign_method[slot][method] += 1
        else:
            warnings.append(f'{shard}: 缺少 lane_assignment_debug.csv，assignment_method 统计将不完整')

        check_files = ['ego_seq.npy', 'neighbor_seq.npy', 'context_traj.npy', 'interaction_feat_style_raw.npy']
        if (shard / 'interaction_feat_style.npy').exists():
            check_files.append('interaction_feat_style.npy')
        elif validate_only:
            warnings.append(f'{shard}: validate_only 时缺少 interaction_feat_style.npy')

        for fn in check_files:
            arr = np.load(shard / fn, mmap_mode='r')
            if not np.isfinite(arr).all():
                nonfinite_detected = 1

    summary: dict[str, Any] = {
        'n_windows_kept': int(n_windows_kept),
        'split_counts': {k: int(v) for k, v in sorted(split_counts.items())},
        'slot_occupied_window_count_by_slot': {k: int(slot_occ_counts.get(k, 0)) for k in SLOTS},
        'slot_occupied_window_ratio': {k: float(slot_occ_counts.get(k, 0)) / max(1, n_windows_kept) for k in SLOTS},
        'empty_slot_count_by_slot': {k: int(empty_counts.get(k, 0)) for k in SLOTS},
        'empty_slot_ratio_by_slot': {k: float(empty_counts.get(k, 0)) / max(1, n_windows_kept) for k in SLOTS},
        'nonfinite_output_detected': int(nonfinite_detected),
    }

    if total_frames > 0:
        summary['slot_valid_frame_count_by_slot'] = {k: int(slot_frame_counts.get(k, 0)) for k in SLOTS}
        summary['slot_valid_frame_ratio'] = {k: float(slot_frame_counts.get(k, 0)) / total_frames for k in SLOTS}
        summary['slot_valid_ratio'] = dict(summary['slot_valid_frame_ratio'])
    else:
        summary['slot_valid_ratio'] = dict(summary['slot_occupied_window_ratio'])

    if lane_quality_counts:
        total_quality = sum(lane_quality_counts.values())
        summary['lane_context_quality_counts'] = dict(lane_quality_counts)
        summary['good_lane_context_rate'] = float(lane_quality_counts.get('good', 0)) / max(1, total_quality)
        summary['ambiguous_intersection_rate'] = float(lane_quality_counts.get('ambiguous_intersection', 0)) / max(1, total_quality)
        summary['bad_lane_context_rate'] = float(lane_quality_counts.get('bad', 0)) / max(1, total_quality)
        summary['fallback_lane_context_rate'] = float(lane_quality_counts.get('fallback', 0)) / max(1, total_quality)

    summary['lane_assignment_success_count_kept'] = int(lane_success_count)
    summary['fallback_assignment_count_kept'] = int(fallback_used_count)
    summary['lane_assignment_success_rate'] = float(lane_success_count) / max(1, n_windows_kept)
    summary['fallback_assignment_rate'] = float(fallback_used_count) / max(1, n_windows_kept)

    assignment_counts = {slot: {m: int(assign_method[slot].get(m, 0)) for m in ALLOWED_METHODS} for slot in SLOTS}
    summary['assignment_method_counts_by_slot'] = assignment_counts
    for slot in SLOTS:
        slot_sum = sum(assignment_counts[slot].values())
        if slot_sum != n_windows_kept:
            warnings.append(f'assignment_method_counts_by_slot[{slot}] 汇总 {slot_sum} != n_windows_kept {n_windows_kept}')

    if sum(summary['split_counts'].values()) != n_windows_kept:
        raise RuntimeError('split_counts 汇总不等于 n_windows_kept')

    return summary, warnings


def _validate_manifest(manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    roots = [Path(p) for p in manifest.get('input_roots', [])]
    raw_paths = [str(p) for p in manifest.get('shard_paths', [])]
    if not raw_paths:
        raise RuntimeError('manifest has no shard_paths')

    resolved: list[Path] = []
    errors: list[str] = []
    for sp in raw_paths:
        resolved_one = None
        if Path(sp).is_absolute() and Path(sp).exists():
            resolved_one = Path(sp).resolve()
        else:
            candidates = roots if roots else [manifest_path.parent]
            for r in candidates:
                try:
                    resolved_one = resolve_shard_path(r, sp)
                    break
                except FileNotFoundError:
                    continue
        if resolved_one is None:
            errors.append(f'Cannot resolve shard from manifest: {sp}')
            continue
        resolved.append(resolved_one)

    if errors:
        raise RuntimeError('\n'.join(errors))

    agg, warnings = _aggregate_from_shards(resolved, validate_only=True)
    if int(manifest.get('total_windows', -1)) != int(agg['n_windows_kept']):
        raise RuntimeError(
            f"total_windows mismatch: manifest={manifest.get('total_windows')} actual={agg['n_windows_kept']}"
        )
    for w in warnings:
        print(f'[WARN] {w}')
    print('Validation passed.')


def main() -> None:
    p = argparse.ArgumentParser(description='Merge Stage 5A shard outputs by scanning shard files.')
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
        return

    if not args.input_roots or args.out_dir is None:
        raise ValueError('merge mode requires --input_roots and --out_dir')

    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f'{out_dir} exists and not empty; use --overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_shards: list[Path] = []
    errors: list[str] = []
    for r in args.input_roots:
        root = Path(r)
        summary = _read_json(root / 'build_summary.json')
        e, shards = _validate_root(root, summary)
        errors.extend(e)
        all_shards.extend(shards)

    if errors:
        raise RuntimeError('\n'.join(errors))

    agg, warnings = _aggregate_from_shards(all_shards, validate_only=False)

    std_info = {'path': 'interaction_feature_standardization.json', 'train_count': 0}
    if args.recompute_global_standardization:
        std_info = _recompute_standardization(all_shards, out_dir)
    agg['interaction_feature_standardization'] = std_info['path']

    merged: dict[str, Any] = {
        'dataset_type': 'waymo_5neighbor_context_laneaware',
        'merged_dataset_type': 'waymo_5neighbor_context',
        'output_format': 'sharded',
        'input_roots': [str(Path(r).resolve()) for r in args.input_roots],
        'shard_paths': [str(s) for s in all_shards],
        'n_shards': len(all_shards),
        'total_shards': len(all_shards),
        'total_windows': int(agg['n_windows_kept']),
    }
    merged.update(agg)
    if warnings:
        merged['warnings'] = warnings

    manifest = {
        'merged_dataset_type': 'waymo_5neighbor_context',
        'input_roots': merged['input_roots'],
        'shard_paths': [str(s) for s in all_shards],
        'total_shards': len(all_shards),
        'total_windows': int(agg['n_windows_kept']),
        'required_files_per_shard': REQUIRED_FILES + ['interaction_feat_style.npy'],
        'global_standardization_path': str((out_dir / 'interaction_feature_standardization.json').resolve()),
    }

    _write_json(out_dir / 'shard_manifest.json', manifest)
    _write_json(out_dir / 'merged_build_summary.json', merged)
    _write_json(out_dir / 'build_summary.json', merged)
    _write_json(out_dir / 'neighbor_context_summary.json', merged)

    report = f"""# Stage 5A 分片合并报告

- 分片总数：{len(all_shards)}
- n_windows_kept：{agg['n_windows_kept']}
- split_counts：{json.dumps(agg['split_counts'], ensure_ascii=False)}
- slot occupied window ratio：{json.dumps(agg['slot_occupied_window_ratio'], ensure_ascii=False)}
- slot valid frame ratio：{json.dumps(agg.get('slot_valid_frame_ratio', {}), ensure_ascii=False)}
- empty slot ratio：{json.dumps(agg['empty_slot_ratio_by_slot'], ensure_ascii=False)}
- lane_context_quality_counts：{json.dumps(agg.get('lane_context_quality_counts', {}), ensure_ascii=False)}
- fallback_assignment_rate：{agg['fallback_assignment_rate']:.8f}
- nonfinite_output_detected：{agg['nonfinite_output_detected']}
- global standardization train_count：{std_info['train_count']}
- 说明：本次仅进行清单与统计汇总，大型张量（例如 ego_seq / neighbor_seq / context_traj）保持分片存储，不进行全量拼接。
"""
    (out_dir / 'build_report.md').write_text(report, encoding='utf-8')

    for w in warnings:
        print(f'[WARN] {w}')
    print(f'Merged {len(all_shards)} shards into manifest: {out_dir / "shard_manifest.json"}')


if __name__ == '__main__':
    main()
