#!/usr/bin/env python3
"""Canonicalize the two frozen official nuPlan safety metric payloads for R1.

This is an artifact parser, not a scientific metric calculator.  It binds the
existing Stage7L semantics to the official nuPlan Parquet schema and fails
closed unless exactly one expected file and one unambiguous scenario row exist
for each frozen metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


COLLISION_FILENAME = "no_ego_at_fault_collisions.parquet"
DRIVABLE_FILENAME = "drivable_area_compliance.parquet"
COLLISION_FIELD = "number_of_all_at_fault_collisions_stat_value"
DRIVABLE_FIELD = "drivable_area_compliance_stat_value"
CANONICAL_SCHEMA_VERSION = "r1_official_metric_canonical_payload_v1.0"


class MetricCanonicalizationError(RuntimeError):
    """A fail-closed incompatibility in an official metric artifact."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_metric_path(run_dir: Path, filename: str, metric_name: str) -> Path:
    """Find one and only one exact official filename under an official run root."""
    matches = sorted(path for path in run_dir.rglob(filename) if path.is_file() and path.name == filename)
    if not matches:
        raise MetricCanonicalizationError(f"{metric_name}:MISSING_EXPECTED_METRIC_FILE:{filename}")
    if len(matches) != 1:
        rendered = ", ".join(str(path.relative_to(run_dir)) for path in matches)
        raise MetricCanonicalizationError(f"{metric_name}:DUPLICATE_EXPECTED_METRIC_FILE:{filename}:{rendered}")
    return matches[0]


def _read_exact_row(path: Path, field: str, metric_name: str) -> Any:
    try:
        table = pd.read_parquet(path)
    except Exception as exc:
        raise MetricCanonicalizationError(f"{metric_name}:UNREADABLE_PARQUET:{type(exc).__name__}:{exc}") from exc
    if table.empty:
        raise MetricCanonicalizationError(f"{metric_name}:EMPTY_TABLE")
    if field not in table.columns:
        raise MetricCanonicalizationError(f"{metric_name}:EXPECTED_COLUMN_MISSING:{field}")
    if len(table.index) != 1:
        raise MetricCanonicalizationError(f"{metric_name}:AMBIGUOUS_SCENARIO_ROW:row_count={len(table.index)}")
    value = table.iloc[0][field]
    if pd.isna(value):
        raise MetricCanonicalizationError(f"{metric_name}:NONFINITE_OR_NULL_VALUE:{field}")
    return value


def _canonical_collision(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise MetricCanonicalizationError(f"collision:UNEXPECTED_TYPE:{type(value).__name__}")
    canonical = int(value)
    if canonical < 0:
        raise MetricCanonicalizationError("collision:INVALID_NEGATIVE_COUNT")
    return canonical


def _canonical_drivable(value: Any) -> bool:
    """Stage7L's bool semantics, restricted to the official binary schema."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        raise MetricCanonicalizationError(f"drivable_area:UNEXPECTED_TYPE:{type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MetricCanonicalizationError("drivable_area:NONFINITE_OR_NULL_VALUE")
    if numeric not in (0.0, 1.0):
        raise MetricCanonicalizationError(f"drivable_area:INVALID_BINARY_VALUE:{numeric!r}")
    return bool(int(numeric))


def canonicalize_official_metrics(run_dir: Path) -> Dict[str, Any]:
    """Return exact canonical payload and non-primary artifact provenance hashes."""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise MetricCanonicalizationError(f"official run directory does not exist: {root}")
    collision_path = _expected_metric_path(root, COLLISION_FILENAME, "collision")
    drivable_path = _expected_metric_path(root, DRIVABLE_FILENAME, "drivable_area")
    collision = _canonical_collision(_read_exact_row(collision_path, COLLISION_FIELD, "collision"))
    drivable = _canonical_drivable(_read_exact_row(drivable_path, DRIVABLE_FIELD, "drivable_area"))
    payload = {
        "collision": {COLLISION_FIELD: collision},
        "drivable_area": {DRIVABLE_FIELD: drivable},
    }
    provenance = {
        "collision": {"relative_path": str(collision_path.relative_to(root)), "artifact_provenance_sha256": sha256_file(collision_path)},
        "drivable_area": {"relative_path": str(drivable_path.relative_to(root)), "artifact_provenance_sha256": sha256_file(drivable_path)},
    }
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "canonical_payload": payload,
        "canonical_payload_sha256": canonical_sha256(payload),
        "artifact_provenance": provenance,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Official run directory containing the two exact Parquet files.")
    parser.add_argument("--output", required=True, type=Path, help="Small canonical payload/provenance JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite canonical metric output: {args.output}")
    payload = canonicalize_official_metrics(args.run_dir)
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "canonical_payload_sha256": payload["canonical_payload_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
