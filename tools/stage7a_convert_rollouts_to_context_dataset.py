#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
from pathlib import Path


REQUIRED_COLUMNS = [
    "scenario_id",
    "policy_id",
    "timestamp",
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "ego_speed",
    "ego_accel",
    "ego_heading",
    "ego_yaw_rate",
]

OPTIONAL_NEIGHBOR_COLUMNS = [
    "neighbor_id",
    "neighbor_x",
    "neighbor_y",
    "neighbor_vx",
    "neighbor_vy",
    "neighbor_speed",
    "neighbor_heading",
    "neighbor_type",
]


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_columns(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def parquet_columns(path: Path):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            f"Cannot validate parquet schema because pyarrow is unavailable: {path}. "
            "Install parquet dependencies or export CSV rollouts first."
        ) from exc
    return list(pq.read_schema(path).names)


def discover_rollout_tables(rollout_dir: Path):
    return sorted(
        [p for p in rollout_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}]
    )


def validate_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        columns = csv_columns(path)
    elif suffix == ".parquet":
        columns = parquet_columns(path)
    else:
        columns = []
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    optional_present = [c for c in OPTIONAL_NEIGHBOR_COLUMNS if c in columns]
    return {
        "path": str(path),
        "format": suffix.lstrip("."),
        "columns": columns,
        "missing_required_columns": missing,
        "optional_neighbor_columns_present": optional_present,
        "schema_valid": not missing,
    }


def build_feature_schema():
    features = []
    for idx, name in enumerate(REQUIRED_COLUMNS + OPTIONAL_NEIGHBOR_COLUMNS):
        features.append({
            "index": idx,
            "name": name,
            "source": "stage7A_rollout_schema",
            "required": name in REQUIRED_COLUMNS,
        })
    return {
        "schema_type": "stage7A_rollout_expected_schema",
        "note": "This skeleton validates rollout table columns only. It does not yet generate ego_seq.npy or neighbor_seq.npy.",
        "features": features,
    }


def run(args) -> int:
    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        raise FileNotFoundError(f"rollout_dir does not exist: {rollout_dir}")
    if not rollout_dir.is_dir():
        raise NotADirectoryError(f"rollout_dir is not a directory: {rollout_dir}")

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_tables = discover_rollout_tables(rollout_dir)
    validations = []
    errors = []
    for table in rollout_tables:
        try:
            result = validate_table(table)
        except Exception as exc:
            result = {
                "path": str(table),
                "schema_valid": False,
                "error": str(exc),
            }
        validations.append(result)
        if not result.get("schema_valid", False):
            errors.append(result)

    feature_schema = build_feature_schema()
    write_json(out_dir / "feature_schema.json", feature_schema)

    manifest = {
        "stage": "stage7A_rollout_to_context_dataset",
        "status": "schema_validated_pending_converter_implementation" if rollout_tables and not errors else "blocked",
        "rollout_dir": str(rollout_dir),
        "output_dir": str(out_dir),
        "history_sec": float(args.history_sec),
        "dt": float(args.dt),
        "num_neighbors": int(args.num_neighbors),
        "rollout_table_count": len(rollout_tables),
        "table_validations": validations,
        "expected_outputs_not_yet_generated": [
            "ego_seq.npy",
            "neighbor_seq.npy",
            "metadata.csv or metadata.npy",
            "shard_manifest.json",
        ],
        "todo": [
            "Window each policy rollout into fixed-length ego_seq arrays.",
            "Select and align nearest neighbors for neighbor_seq arrays.",
            "Write metadata with scenario_id, policy_id, local row, and time window.",
            "Write shard_manifest.json compatible with Stage 6C tools.",
        ],
    }
    write_json(out_dir / "conversion_manifest.json", manifest)
    (out_dir / "README.md").write_text(
        "# Stage 7A rollout-to-context converter\n\n"
        "This skeleton validates future rollout CSV/parquet schemas and writes the expected schema.\n"
        "It does not create fake `ego_seq.npy` or `neighbor_seq.npy` files.\n\n"
        f"Status: `{manifest['status']}`\n",
        encoding="utf-8",
    )

    if not rollout_tables:
        print(f"No CSV/parquet rollout tables found under: {rollout_dir}")
        print(f"Wrote expected schema: {out_dir / 'feature_schema.json'}")
        return 2
    if errors:
        print("Some rollout tables do not match the expected Stage 7A schema:")
        for item in errors:
            print(f"- {item.get('path')}: missing={item.get('missing_required_columns')} error={item.get('error')}")
        print(f"Wrote conversion manifest: {out_dir / 'conversion_manifest.json'}")
        return 2

    print("All discovered rollout tables match the expected Stage 7A schema.")
    print("Actual ego_seq.npy / neighbor_seq.npy conversion is still TODO.")
    print(f"Wrote conversion manifest: {out_dir / 'conversion_manifest.json'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Skeleton converter from Stage 7A rollout tables to context dataset format.")
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_sec", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
