#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SLOTS = ["front", "left_front", "left_rear", "right_front", "right_rear"]
METHODS = ["lane_aware", "geometric_fallback", "empty", "sanitize_failed"]
PRESERVE_FIELDS = [
    "dataset_type", "n_files_processed", "n_scenarios_processed", "n_target_agents_considered",
    "n_windows_total", "n_windows_filtered_static", "n_windows_filtered_invalid",
    "n_windows_dropped_no_lane_map", "n_windows_dropped_ego_lane_missing",
    "n_windows_dropped_bad_lane_context", "n_windows_dropped_lane_context_ambiguous",
    "n_windows_dropped_clean_filter_total", "window_len", "dt", "lane_map_available_scenarios",
    "lane_map_missing_scenarios", "lane_projection_attempt_count", "lane_projection_success_count",
    "lane_projection_success_rate", "lane_projection_failure_count", "lane_projection_avg_candidate_lanes",
    "lane_projection_max_candidate_lanes", "lane_spatial_index_enabled", "lane_search_radius",
    "lane_topk_candidates", "trajectory_nan_count_raw", "trajectory_inf_count_raw",
    "trajectory_nan_count_after_sanitize", "trajectory_inf_count_after_sanitize", "ego_windows_repaired",
    "ego_windows_dropped_sanitize_failed", "neighbor_windows_repaired", "neighbor_windows_dropped_sanitize_failed",
    "timing_seconds", "notes",
]


def to_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def finite_check(arr):
    if not np.issubdtype(arr.dtype, np.number):
        return True
    return bool(np.isfinite(arr).all())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate_only", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    shards = sorted((data_root / "shards").glob("shard_*"))

    warnings = []
    errors = []
    if not shards:
        errors.append("no shards found")

    split_counts = Counter()
    n_windows_kept = 0
    required_any = {"split.npy": 0, "meta.npy": 0, "context_mask_window.npy": 0}
    row_mismatch = False

    occ_count = np.zeros(5, dtype=np.int64)
    valid_frame_count = np.zeros(5, dtype=np.int64)
    valid_frame_total = 0

    assign_counts = {s: {m: 0 for m in METHODS} for s in SLOTS}
    static_by_slot = {s: 0 for s in SLOTS}
    static_front_count = 0
    assigned_front_count = 0

    lane_quality_counts = Counter()
    lane_success_count = 0
    fallback_used_count = 0
    has_lane_quality = False
    has_lane_success = False
    has_fallback_used = False

    nonfinite_hits = []

    for shard in shards:
        split_p = shard / "split.npy"
        meta_p = shard / "meta.npy"
        cmw_p = shard / "context_mask_window.npy"
        cm_p = shard / "context_mask.npy"

        for k in required_any:
            if (shard / k).exists():
                required_any[k] += 1

        if not split_p.exists():
            warnings.append(f"{shard}: split.npy missing")
            continue

        split = np.load(split_p, allow_pickle=True)
        n = int(len(split))
        n_windows_kept += n
        for s in split:
            split_counts[str(s)] += 1

        shard_rows = {"split.npy": n}
        if meta_p.exists():
            meta = np.load(meta_p, allow_pickle=True)
            shard_rows["meta.npy"] = int(len(meta))
            fields = set(meta.dtype.names or [])
            if "lane_context_quality" in fields:
                has_lane_quality = True
                for v in meta["lane_context_quality"]:
                    lane_quality_counts[str(v)] += 1
            if "lane_assignment_success" in fields:
                has_lane_success = True
                lane_success_count += int(np.sum([to_bool(v) for v in meta["lane_assignment_success"]]))
            if "fallback_used" in fields:
                has_fallback_used = True
                fallback_used_count += int(np.sum([to_bool(v) for v in meta["fallback_used"]]))
        else:
            warnings.append(f"{shard}: meta.npy missing")

        for fname in ["context_traj.npy", "ego_seq.npy", "neighbor_seq.npy", "context_mask_window.npy", "interaction_feat_style_raw.npy"]:
            p = shard / fname
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                shard_rows[fname] = int(arr.shape[0])

        row_vals = set(shard_rows.values())
        if len(row_vals) > 1:
            row_mismatch = True
            warnings.append(f"{shard}: row count mismatch {shard_rows}")

        if cmw_p.exists():
            cmw = np.load(cmw_p)
            if cmw.ndim != 2 or cmw.shape[1] != 5:
                warnings.append(f"{shard}: context_mask_window shape invalid {cmw.shape}")
            else:
                occ = cmw.astype(bool)
                occ_count += occ.sum(axis=0)
        else:
            warnings.append(f"{shard}: context_mask_window.npy missing")

        if cm_p.exists():
            cm = np.load(cm_p)
            if cm.ndim == 3 and cm.shape[2] == 5:
                valid = cm.astype(bool)
                valid_frame_count += valid.sum(axis=(0, 1))
                valid_frame_total += int(valid.shape[0] * valid.shape[1])
            else:
                warnings.append(f"{shard}: context_mask shape invalid {cm.shape}")

        debug_p = shard / "lane_assignment_debug.csv"
        used_csv = False
        if debug_p.exists():
            with debug_p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if len(rows) >= n:
                used_csv = True
                per_slot_seen = {s: 0 for s in SLOTS}
                for r in rows:
                    slot = str(r.get("slot", "")).strip()
                    if slot not in SLOTS:
                        continue
                    method = str(r.get("assignment_method", "")).strip()
                    if method not in METHODS:
                        method = "lane_aware"
                    assign_counts[slot][method] += 1
                    per_slot_seen[slot] += 1
                    if to_bool(r.get("neighbor_is_static", False)):
                        static_by_slot[slot] += 1
                        if slot == "front":
                            static_front_count += 1
                assigned_front_count += per_slot_seen["front"]
            else:
                warnings.append(f"{shard}: debug csv incomplete, fallback to context_mask_window")
        if (not used_csv) and cmw_p.exists():
            cmw = np.load(cmw_p).astype(bool)
            if cmw.ndim == 2 and cmw.shape[1] == 5:
                warnings.append(f"{shard}: assignment method fallback from context_mask_window")
                for i, slot in enumerate(SLOTS):
                    occn = int(cmw[:, i].sum())
                    assign_counts[slot]["lane_aware"] += occn
                    assign_counts[slot]["empty"] += int(cmw.shape[0] - occn)
                assigned_front_count += int(cmw.shape[0])

        for fname in ["ego_seq.npy", "neighbor_seq.npy", "context_traj.npy", "interaction_feat_style_raw.npy", "interaction_feat_style.npy"]:
            p = shard / fname
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                if not finite_check(arr):
                    nonfinite_hits.append({"shard": str(shard), "file": fname})

    for k, v in required_any.items():
        if v == 0:
            errors.append(f"required file missing from all shards: {k}")

    if not split_counts:
        errors.append("split_counts empty")
    if sum(split_counts.values()) != n_windows_kept:
        errors.append("sum(split_counts) != n_windows_kept")
    if row_mismatch:
        errors.append("shard row count mismatch")
    if nonfinite_hits:
        errors.append("nonfinite detected")

    total = max(1, n_windows_kept)
    slot_occupied_count = {s: int(occ_count[i]) for i, s in enumerate(SLOTS)}
    slot_occupied_ratio = {s: float(occ_count[i] / total) for i, s in enumerate(SLOTS)}
    empty_count = {s: int(total - occ_count[i]) for i, s in enumerate(SLOTS)}
    empty_ratio = {s: float((total - occ_count[i]) / total) for i, s in enumerate(SLOTS)}

    summary = {}
    orig_path = data_root / "build_summary.json"
    if orig_path.exists():
        summary = json.loads(orig_path.read_text(encoding="utf-8"))
    for k in PRESERVE_FIELDS:
        if k in summary:
            summary[k] = summary[k]

    summary.update({
        "n_shards": len(shards),
        "shard_paths": [str(p.relative_to(data_root)) for p in shards],
        "n_windows_kept": int(n_windows_kept),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "slot_occupied_window_count_by_slot": slot_occupied_count,
        "slot_occupied_window_ratio": slot_occupied_ratio,
        "empty_slot_count_by_slot": empty_count,
        "empty_slot_ratio_by_slot": empty_ratio,
        "assignment_method_counts_by_slot": assign_counts,
        "static_neighbor_count_by_slot": static_by_slot,
        "static_front_count": int(static_front_count),
        "static_front_ratio": float(static_front_count / max(1, assigned_front_count)),
        "nonfinite_output_detected": int(len(nonfinite_hits) > 0),
        "summary_rebuilt_from_shards": True,
        "summary_rebuild_time": datetime.now(timezone.utc).isoformat(),
        "summary_rebuild_warnings": warnings,
    })

    if valid_frame_total > 0:
        summary["slot_valid_frame_count_by_slot"] = {s: int(valid_frame_count[i]) for i, s in enumerate(SLOTS)}
        summary["slot_valid_frame_ratio"] = {s: float(valid_frame_count[i] / valid_frame_total) for i, s in enumerate(SLOTS)}
        summary["slot_valid_ratio"] = summary["slot_valid_frame_ratio"]
    else:
        summary["slot_valid_ratio"] = summary["slot_occupied_window_ratio"]

    if has_lane_quality:
        lq = {k: int(v) for k, v in lane_quality_counts.items()}
        summary["lane_context_quality_counts"] = lq
        summary["good_lane_context_rate"] = float(lq.get("good", 0) / total)
        summary["ambiguous_intersection_rate"] = float(lq.get("ambiguous_intersection", 0) / total)
        summary["bad_lane_context_rate"] = float(lq.get("bad", 0) / total)
        summary["fallback_lane_context_rate"] = float(lq.get("fallback", 0) / total)
    else:
        summary["lane_context_quality_counts"] = {}
        warnings.append("lane_context_quality missing in meta")

    if has_lane_success:
        summary["lane_assignment_success_count_kept"] = int(lane_success_count)
        summary["lane_assignment_success_rate"] = float(lane_success_count / total)
    else:
        warnings.append("lane_assignment_success missing in meta")
    if has_fallback_used:
        summary["fallback_assignment_count_kept"] = int(fallback_used_count)
        summary["fallback_assignment_rate"] = float(fallback_used_count / total)
    else:
        warnings.append("fallback_used missing in meta")

    for slot in SLOTS:
        sm = sum(assign_counts[slot].values())
        if sm != n_windows_kept:
            warnings.append(f"assignment method sum mismatch: {slot}={sm}, expected={n_windows_kept}")

    if args.strict and warnings:
        errors.append("strict mode: warnings present")

    if args.validate_only:
        print(json.dumps({"ok": len(errors) == 0, "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
        sys.exit(0 if len(errors) == 0 else 1)

    if (data_root / "build_summary.json").exists() and (not args.overwrite):
        raise SystemExit("build_summary.json exists, use --overwrite")

    (data_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "neighbor_context_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if nonfinite_hits:
        (data_root / "nonfinite_debug_summary.json").write_text(json.dumps(nonfinite_hits, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Stage 5A 重建报告

- 样本数（n_windows_kept）：{summary['n_windows_kept']}
- split_counts：{summary.get('split_counts', {})}
- shard 数量：{summary['n_shards']}
- slot_occupied_window_ratio：{summary.get('slot_occupied_window_ratio', {})}
- slot_valid_frame_ratio：{summary.get('slot_valid_frame_ratio', {})}
- empty_slot_ratio_by_slot：{summary.get('empty_slot_ratio_by_slot', {})}
- lane_context_quality_counts：{summary.get('lane_context_quality_counts', {})}
- fallback_assignment_rate：{summary.get('fallback_assignment_rate', None)}
- static_front_count / static_front_ratio：{summary.get('static_front_count', 0)} / {summary.get('static_front_ratio', 0.0)}
- nonfinite_output_detected：{summary.get('nonfinite_output_detected', 0)}
- 说明：summary_rebuilt_from_shards={summary.get('summary_rebuilt_from_shards', False)}
"""
    (data_root / "build_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"ok": len(errors) == 0, "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
