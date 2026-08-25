#!/usr/bin/env python3
"""Finalize independently built Dynamic Builder v2 parts with global train statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shards(root: Path) -> list[Path]:
    summary = read_json(root / "build_summary.json")
    result = []
    for raw in summary["shard_paths"]:
        path = Path(raw)
        if not path.is_dir():
            path = root / "shards" / path.name
        if not path.is_dir():
            raise FileNotFoundError(raw)
        result.append(path.resolve())
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = [path.resolve() for path in args.part_roots]
    manifests = [read_json(root / "dynamic_builder_v2_manifest.json") for root in roots]
    summaries = [read_json(root / "build_summary.json") for root in roots]
    pilot_decision = read_json(args.pilot_decision)
    if pilot_decision.get("status") != "PILOT_PASS_FULL51_REBUILD_ALLOWED_NOT_TRAINING":
        raise ValueError("Stage 6R pilot did not authorize the full51 rebuild")
    if not all(
        manifest.get("assignment_mode_required") == "lane_aware_only"
        and manifest.get("geometric_adjacent_lane_inference") is False
        for manifest in manifests
    ):
        raise ValueError("all Dynamic v2 parts must use strict semantic assignment without geometric fallback")
    selections = [manifest["source_selection"] for manifest in manifests]
    intervals = sorted((item["file_start"], item["file_end_exclusive"]) for item in selections)
    cursor = 0
    for start, end in intervals:
        if start != cursor or end is None or end <= start:
            raise ValueError(f"source file ranges are not contiguous at {cursor}: {intervals}")
        cursor = end
    if cursor != args.expected_file_count:
        raise ValueError(f"expected files 0..{args.expected_file_count - 1}, observed intervals={intervals}")
    source_files = sorted(args.waymo_dir.resolve().glob("*.tfrecord*"))[: args.expected_file_count]
    if len(source_files) != args.expected_file_count:
        raise ValueError(f"expected {args.expected_file_count} source TFRecords, got {len(source_files)}")
    all_shards = [shard for root in roots for shard in shards(root)]
    schema_hashes = {sha256_file(root / "feature_schema.json") for root in roots}
    if len(schema_hashes) != 1 or any(manifest.get("schema_version") != "waymo_dynamic_interaction_builder_v2" for manifest in manifests):
        raise ValueError("Dynamic v2 schemas differ across parts")
    train_columns: list[list[np.ndarray]] = [[], [], []]
    split_counts = Counter()
    scenario_split: dict[str, str] = {}
    overlap = set()
    for shard in all_shards:
        raw = np.load(shard / "longitudinal_supervision_v2_raw.npy", mmap_mode="r")
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        meta = np.load(shard / "meta.npy", allow_pickle=True)
        train = split == "train"
        for column in range(3):
            train_columns[column].append(np.asarray(raw[train, :, column]).reshape(-1))
        for name, count in zip(*np.unique(split, return_counts=True)):
            split_counts[str(name)] += int(count)
        for row, split_name in zip(meta, split):
            sid = str(row["scenario_id"])
            previous = scenario_split.setdefault(sid, str(split_name))
            if previous != str(split_name):
                overlap.add(sid)
    train_values = [np.concatenate(values).astype(np.float32, copy=False) for values in train_columns]
    q01 = np.asarray([np.quantile(values, 0.01) for values in train_values], dtype=np.float64)
    q99 = np.asarray([np.quantile(values, 0.99) for values in train_values], dtype=np.float64)
    medians = []
    scales = []
    for index, values in enumerate(train_values):
        clipped = np.clip(values, q01[index], q99[index])
        medians.append(float(np.median(clipped)))
        q25, q75 = np.quantile(clipped, [0.25, 0.75])
        scales.append(float(max(q75 - q25, 1e-6)))
    median = np.asarray(medians)
    scale = np.asarray(scales)
    for shard in all_shards:
        raw = np.load(shard / "longitudinal_supervision_v2_raw.npy", mmap_mode="r")
        normalized = ((np.clip(raw, q01, q99) - median) / scale).astype(np.float32)
        np.save(shard / "longitudinal_supervision_v2.npy", normalized)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    schema = {
        "feature_names": ["ego_speed_smoothed", "ego_longitudinal_accel", "ego_longitudinal_jerk"],
        "median_filter_frames": 5,
        "train_q01": q01.tolist(),
        "train_q99": q99.tolist(),
        "train_median_after_winsorize": median.tolist(),
        "train_iqr_after_winsorize": scale.tolist(),
        "train_frame_count": int(len(train_values[0])),
        "statistics_scope": "all_train_rows_across_full51",
        "finite": True,
    }
    schema_path = output / "longitudinal_supervision_v2_global_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    source_sha256 = {path.name: sha256_file(path) for path in source_files}
    shard_artifact_sha256 = {}
    required_artifacts = [
        "ego_seq.npy", "neighbor_seq.npy", "context_traj.npy", "context_mask.npy",
        "context_mask_window.npy", "neighbor_slot_ids.npy", "meta.npy", "split.npy",
        "slot_track_id_timeline.npy", "slot_valid_mask.npy", "slot_identity_switch_mask.npy",
        "slot_derivative_valid_mask.npy", "longitudinal_supervision_v2_raw.npy",
        "longitudinal_supervision_v2.npy",
    ]
    for shard in all_shards:
        missing = [name for name in required_artifacts if not (shard / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{shard} missing required v2 artifacts: {missing}")
        shard_artifact_sha256[str(shard)] = {
            name: sha256_file(shard / name) for name in required_artifacts
        }
    hash_ledger = output / "stage6r_full51_sha256_ledger.json"
    hash_ledger.write_text(json.dumps({"source_tfrecord_sha256": source_sha256, "shard_artifact_sha256": shard_artifact_sha256}, indent=2) + "\n")
    manifest = {
        "schema_version": "stage6r_waymo_dynamic_full51_merged_v1",
        "status": "DYNAMIC_FULL51_FINALIZED_PENDING_STAGE6O_V2",
        "part_roots": [str(root) for root in roots],
        "pilot_decision_sha256": sha256_file(args.pilot_decision),
        "part_manifest_sha256": [sha256_file(root / "dynamic_builder_v2_manifest.json") for root in roots],
        "source_file_intervals": intervals,
        "source_file_count": cursor,
        "shard_paths": [str(path) for path in all_shards],
        "shard_count": len(all_shards),
        "row_count": int(sum(summary["n_windows_kept"] for summary in summaries)),
        "scenario_count": int(sum(summary["n_scenarios_processed"] for summary in summaries)),
        "split_counts": dict(sorted(split_counts.items())),
        "scenario_cross_split_overlap_count": len(overlap),
        "global_longitudinal_schema": str(schema_path),
        "global_longitudinal_schema_sha256": sha256_file(schema_path),
        "feature_schema_sha256": next(iter(schema_hashes)),
        "sha256_ledger": str(hash_ledger),
        "sha256_ledger_sha256": sha256_file(hash_ledger),
        "old_full51_overwritten": False,
        "stage6o_v1_modified": False,
        "embedding_or_checkpoint_read": False,
    }
    manifest["content_signature_sha256"] = canonical_sha({key: value for key, value in manifest.items() if key != "content_signature_sha256"})
    (output / "stage6r_dynamic_full51_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--waymo_dir", type=Path, required=True)
    parser.add_argument("--pilot_decision", type=Path, required=True)
    parser.add_argument("--expected_file_count", type=int, default=51)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
