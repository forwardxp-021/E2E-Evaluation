#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ALIGNMENT_KEY_CANDIDATES = [
    "scenario_id", "scene_token", "db_name", "sample_id", "window_id",
    "start_frame", "end_frame", "start_time_us", "end_time_us",
    # Stage 7B.2/7B.3 current names.
    "start_frame_index", "end_frame_index",
    "start_lidar_pc_timestamp", "end_lidar_pc_timestamp",
]
DYNAMIC_ARRAYS = ["ego_seq.npy", "neighbor_seq.npy", "context_traj.npy", "context_mask.npy", "interaction_feat_style.npy"]


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_shard_paths(dynamic_dir: Path) -> List[Path]:
    manifest_path = dynamic_dir / "shard_manifest.json"
    manifest = read_json(manifest_path)
    raw = []
    if manifest.get("shards"):
        raw = [s.get("shard_path") for s in manifest["shards"]]
    elif manifest.get("shard_infos"):
        raw = [s.get("shard_path") for s in manifest["shard_infos"]]
    else:
        raw = manifest.get("shard_paths", [])
    raw = [p for p in raw if p]
    if not raw:
        raise ValueError(f"No shard paths found in dynamic shard manifest: {manifest_path}")
    return [(Path(p) if Path(p).is_absolute() else dynamic_dir / p) for p in raw]


def load_dynamic(dynamic_dir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]], List[str], Path]:
    shards = manifest_shard_paths(dynamic_dir)
    arrays_by_name: Dict[str, List[np.ndarray]] = {name: [] for name in DYNAMIC_ARRAYS}
    meta_rows: List[Dict[str, str]] = []
    meta_fields: List[str] = []
    for shard in shards:
        if not shard.exists():
            raise FileNotFoundError(f"Shard path listed in manifest does not exist: {shard}")
        for name in DYNAMIC_ARRAYS:
            path = shard / name
            if not path.exists():
                raise FileNotFoundError(f"Required dynamic array missing: {path}")
            arrays_by_name[name].append(np.load(path, mmap_mode="r"))
        fields, rows = load_csv(shard / "metadata.csv")
        if not meta_fields:
            meta_fields = fields
        else:
            for field in fields:
                if field not in meta_fields:
                    meta_fields.append(field)
        meta_rows.extend(rows)
    arrays = {name: np.concatenate(parts, axis=0) if len(parts) > 1 else np.asarray(parts[0]) for name, parts in arrays_by_name.items()}
    n = arrays["interaction_feat_style.npy"].shape[0]
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(f"Dynamic array row mismatch: {name} has {arr.shape[0]} rows, expected {n}")
    if len(meta_rows) != n:
        raise ValueError(f"Dynamic metadata rows ({len(meta_rows)}) do not match dynamic arrays ({n})")
    return arrays, meta_rows, meta_fields, dynamic_dir / "feature_schema.json"


def key_tuple(row: Dict[str, str], keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(k, "")) for k in keys)


def align_indices(dynamic_rows: List[Dict[str, str]], map_rows: List[Dict[str, str]], keys: List[str]) -> Tuple[np.ndarray, bool]:
    dynamic_keys = [key_tuple(r, keys) for r in dynamic_rows]
    map_keys = [key_tuple(r, keys) for r in map_rows]
    if dynamic_keys == map_keys:
        return np.arange(len(dynamic_rows)), True
    lookup: Dict[Tuple[str, ...], int] = {}
    duplicates = set()
    for i, k in enumerate(map_keys):
        if k in lookup:
            duplicates.add(k)
        lookup[k] = i
    if duplicates:
        raise ValueError(f"Cannot safely align rows: duplicate map/ODD keys for columns {keys}; duplicate_count={len(duplicates)}")
    missing = [k for k in dynamic_keys if k not in lookup]
    if missing:
        raise ValueError(f"Cannot safely align rows: {len(missing)} dynamic keys are absent from map/ODD metadata using columns {keys}")
    return np.asarray([lookup[k] for k in dynamic_keys], dtype=np.int64), False


def schema_feature_names(schema: dict, feature_dim: int, source: str) -> List[str]:
    for key in ("interaction_features", "feature_names", "merged_feature_names"):
        names = schema.get(key)
        if isinstance(names, list) and len(names) == feature_dim:
            return [str(n) for n in names]
    prefix = "dynamic_feat" if source == "dynamic" else "map_odd_feat"
    return [f"{prefix}_{i:03d}" for i in range(feature_dim)]


def finite_summary(arrays: Dict[str, np.ndarray]) -> Dict[str, bool]:
    return {name: bool(np.isfinite(arr).all()) for name, arr in arrays.items()}


def make_report(args, n_dyn, n_map, keys, already_aligned, reindexed, shapes, finite, warnings, passed):
    lines = [
        "# Stage 7B.4 Dynamic Context + Map/ODD Merge Alignment Report",
        "",
        "## Purpose",
        "Merge Stage 7B.2 dynamic context arrays with Stage 7B.3 map/ODD-lite features into one row-aligned nuPlan context dataset. No training or rollout is performed.",
        "",
        "## Inputs",
        f"- dynamic_context_dir: `{args.dynamic_context_dir}`",
        f"- map_odd_dir: `{args.map_odd_dir}`",
        "",
        "## Output",
        f"- output_dir: `{args.output_dir}`",
        "",
        "## Row counts",
        f"- dynamic rows: {n_dyn}",
        f"- map/ODD rows: {n_map}",
        "",
        "## Alignment",
        f"- alignment keys used: {', '.join(keys)}",
        f"- row order already aligned: {str(already_aligned).lower()}",
        f"- reindexing needed: {str(reindexed).lower()}",
        "",
        "## Output shapes",
    ]
    for name, shape in shapes.items():
        lines.append(f"- {name}: {shape}")
    lines += ["", "## Finite checks"]
    for name, ok in finite.items():
        lines.append(f"- {name}: {str(ok).lower()}")
    lines += ["", "## Warning summary", f"- warning_count: {len(warnings)}"]
    for w in warnings:
        lines.append(f"- {w}")
    lines += ["", "## PASS/FAIL summary", f"- status: {'PASS' if passed else 'FAIL'}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 7B.4 merge dynamic context and map/ODD-lite features with row alignment checks.")
    p.add_argument("--dynamic_context_dir", required=True)
    p.add_argument("--map_odd_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dynamic_dir = Path(args.dynamic_context_dir)
    map_dir = Path(args.map_odd_dir)
    out = Path(args.output_dir)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists; pass --overwrite to replace it: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    warnings: List[dict] = []
    dyn_arrays, dyn_rows, dyn_fields, dyn_schema_path = load_dynamic(dynamic_dir)
    map_feat_path = map_dir / "map_odd_feat.npy"
    if not map_feat_path.exists():
        raise FileNotFoundError(f"Required map/ODD feature array missing: {map_feat_path}")
    map_feat = np.load(map_feat_path, mmap_mode="r")
    map_fields, map_rows = load_csv(map_dir / "map_odd_meta.csv")
    map_schema_path = map_dir / "map_odd_feature_schema.json"
    dyn_schema = read_json(dyn_schema_path)
    map_schema = read_json(map_schema_path)

    n_dyn = int(dyn_arrays["interaction_feat_style.npy"].shape[0])
    n_map = int(map_feat.shape[0])
    if len(map_rows) != n_map:
        raise ValueError(f"map_odd_meta.csv rows ({len(map_rows)}) do not match map_odd_feat rows ({n_map})")
    if n_dyn != n_map:
        raise ValueError(f"Row count mismatch: dynamic rows={n_dyn}, map/ODD rows={n_map}; safe Stage 7B.4 merge requires equal row counts")
    common_keys = [k for k in ALIGNMENT_KEY_CANDIDATES if k in dyn_fields and k in map_fields]
    if not common_keys:
        raise ValueError(f"No common metadata key fields found. dynamic_columns={dyn_fields}; map_odd_columns={map_fields}")
    map_indices, already_aligned = align_indices(dyn_rows, map_rows, common_keys)
    reindexed = not already_aligned
    if reindexed:
        warnings.append({"type": "map_odd_rows_reindexed", "message": "map_odd_feat/map_odd_meta row order was reindexed to match dynamic metadata order.", "alignment_keys": common_keys})
    map_feat_aligned = np.asarray(map_feat)[map_indices]
    map_rows_aligned = [map_rows[int(i)] for i in map_indices]

    dynamic_feat = np.asarray(dyn_arrays["interaction_feat_style.npy"])
    if dynamic_feat.ndim != 2 or map_feat_aligned.ndim != 2:
        raise ValueError(f"Features must be 2D: interaction_feat_style shape={dynamic_feat.shape}, map_odd_feat shape={map_feat_aligned.shape}")
    merged = np.concatenate([dynamic_feat, map_feat_aligned], axis=1).astype(np.float32, copy=False)
    output_arrays = {
        "ego_seq": np.asarray(dyn_arrays["ego_seq.npy"]),
        "neighbor_seq": np.asarray(dyn_arrays["neighbor_seq.npy"]),
        "context_traj": np.asarray(dyn_arrays["context_traj.npy"]),
        "context_mask": np.asarray(dyn_arrays["context_mask.npy"]),
        "dynamic_feat_style": dynamic_feat,
        "map_odd_feat": map_feat_aligned,
        "merged_context_feat": merged,
    }
    finite = finite_summary(output_arrays)
    if not finite["merged_context_feat"]:
        raise ValueError("merged_context_feat contains NaN or inf; refusing to write invalid Stage 7B.4 output")

    for name, arr in output_arrays.items():
        np.save(out / f"{name}.npy", arr)

    merged_rows = []
    fieldnames = list(dyn_fields)
    for f in map_fields:
        pf = f"map_odd::{f}"
        if pf not in fieldnames:
            fieldnames.append(pf)
    for dr, mr in zip(dyn_rows, map_rows_aligned):
        row = dict(dr)
        for f in map_fields:
            row[f"map_odd::{f}"] = mr.get(f, "")
        merged_rows.append(row)
    write_csv(out / "merged_metadata.csv", merged_rows, fieldnames)

    dyn_names = schema_feature_names(dyn_schema, dynamic_feat.shape[1], "dynamic")
    map_names = schema_feature_names(map_schema, map_feat_aligned.shape[1], "map")
    merged_schema = {
        "stage": "7B.4",
        "feature_type": "nuplan_dynamic_plus_map_odd_context",
        "num_dynamic_features": int(dynamic_feat.shape[1]),
        "num_map_odd_features": int(map_feat_aligned.shape[1]),
        "num_merged_features": int(merged.shape[1]),
        "dynamic_feature_schema_source": str(dyn_schema_path),
        "map_odd_feature_schema_source": str(map_schema_path),
        "merged_feature_names": [f"dynamic::{n}" for n in dyn_names] + [f"map_odd::{n}" for n in map_names],
        "alignment_keys_used": common_keys,
        "row_order_already_aligned": bool(already_aligned),
        "reindexing_needed": bool(reindexed),
        "notes": [],
    }
    write_json(out / "merged_feature_schema.json", merged_schema)
    write_json(out / "warnings.json", {"warnings": warnings})
    shapes = {name: list(arr.shape) for name, arr in output_arrays.items()}
    passed = bool(n_dyn == n_map and all(finite.values()))
    (out / "alignment_report.md").write_text(make_report(args, n_dyn, n_map, common_keys, already_aligned, reindexed, shapes, finite, warnings, passed), encoding="utf-8")
    print(f"Wrote Stage 7B.4 merged context to {out}: merged_context_feat shape={list(merged.shape)}, alignment_keys={common_keys}, row_order_already_aligned={already_aligned}, warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
