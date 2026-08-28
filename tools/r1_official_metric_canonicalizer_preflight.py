#!/usr/bin/env python3
"""Run zero-budget fixtures for R1's canonical official metric parser."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tools.r1_official_metric_canonicalizer import (
    COLLISION_FIELD,
    COLLISION_FILENAME,
    DRIVABLE_FIELD,
    DRIVABLE_FILENAME,
    MetricCanonicalizationError,
    canonicalize_official_metrics,
    canonical_sha256,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_OUTPUT = R1_DIR / "r1_official_metric_canonicalizer_preflight_v1.0.json"
DEFAULT_FIXTURE_ROOT = ROOT / "outputs/r1_official_metric_canonicalizer_preflight_v1"
DEFAULT_V2_RUN = ROOT / "outputs/r1_runtime_determinism_validation_v2/runs/R-HLC__25944935eadb52f1__V2_RUN_A"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_dir(root: Path) -> Path:
    directory = root / "nuplan_output" / "metrics"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_pair(root: Path, collision: Any = 0, drivable: Any = True, collision_column: str = COLLISION_FIELD, empty: bool = False) -> None:
    metrics = _metric_dir(root)
    collision_rows = [] if empty else [{collision_column: collision}]
    drivable_rows = [] if empty else [{DRIVABLE_FIELD: drivable}]
    pd.DataFrame(collision_rows, columns=[collision_column]).to_parquet(metrics / COLLISION_FILENAME, index=False)
    pd.DataFrame(drivable_rows, columns=[DRIVABLE_FIELD]).to_parquet(metrics / DRIVABLE_FILENAME, index=False)


def _expect_failure(name: str, root: Path, expected_marker: str) -> Dict[str, Any]:
    try:
        canonicalize_official_metrics(root)
    except MetricCanonicalizationError as exc:
        if expected_marker not in str(exc):
            raise AssertionError(f"{name} failed with unexpected marker: {exc}") from exc
        return {"name": name, "status": "EXPECTED_FAIL", "failure_marker": expected_marker}
    raise AssertionError(f"{name} unexpectedly canonicalized")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--historical-v2-run", type=Path, default=DEFAULT_V2_RUN)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite metric-parser preflight: {args.output}")
    if args.fixture_root.exists():
        raise FileExistsError(f"refusing to overwrite metric-parser fixture root: {args.fixture_root}")
    if not args.historical_v2_run.is_dir():
        raise FileNotFoundError(f"historical V2 run is required for compatibility-only parser validation: {args.historical_v2_run}")

    fixture0 = args.fixture_root / "fixture_collision0_drivable_true"
    _write_pair(fixture0, collision=0, drivable=True)
    payload0 = canonicalize_official_metrics(fixture0)
    expected0 = {"collision": {COLLISION_FIELD: 0}, "drivable_area": {DRIVABLE_FIELD: True}}
    assert payload0["canonical_payload"] == expected0

    fixture1 = args.fixture_root / "fixture_collision1_drivable_false"
    _write_pair(fixture1, collision=1, drivable=False)
    payload1 = canonicalize_official_metrics(fixture1)
    expected1 = {"collision": {COLLISION_FIELD: 1}, "drivable_area": {DRIVABLE_FIELD: False}}
    assert payload1["canonical_payload"] == expected1

    missing = args.fixture_root / "failure_missing_file"
    (missing / "nuplan_output" / "metrics").mkdir(parents=True)
    pd.DataFrame([{DRIVABLE_FIELD: True}]).to_parquet(missing / "nuplan_output" / "metrics" / DRIVABLE_FILENAME, index=False)

    duplicate = args.fixture_root / "failure_duplicate_file"
    _write_pair(duplicate, collision=0, drivable=True)
    duplicate_path = duplicate / "duplicate" / COLLISION_FILENAME
    duplicate_path.parent.mkdir(parents=True)
    pd.DataFrame([{COLLISION_FIELD: 0}]).to_parquet(duplicate_path, index=False)

    missing_column = args.fixture_root / "failure_missing_column"
    _write_pair(missing_column, collision=0, drivable=True, collision_column="wrong_column")

    empty = args.fixture_root / "failure_empty_table"
    _write_pair(empty, empty=True)

    historical = canonicalize_official_metrics(args.historical_v2_run)
    results = [
        {"name": "synthetic_collision0_drivable_true", "status": "PASS", "canonical_payload": payload0["canonical_payload"]},
        {"name": "synthetic_collision1_drivable_false", "status": "PASS", "canonical_payload": payload1["canonical_payload"]},
        _expect_failure("missing_file", missing, "MISSING_EXPECTED_METRIC_FILE"),
        _expect_failure("duplicate_file", duplicate, "DUPLICATE_EXPECTED_METRIC_FILE"),
        _expect_failure("missing_column", missing_column, "EXPECTED_COLUMN_MISSING"),
        _expect_failure("empty_table", empty, "EMPTY_TABLE"),
    ]
    output = {
        "schema_version": "r1_official_metric_canonicalizer_preflight_v1.0",
        "status": "PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED",
        "scope": "SYNTHETIC_PARQUET_FIXTURES_PLUS_HISTORICAL_V2_SCHEMA_COMPATIBILITY_ONLY",
        "canonicalizer_sha256": sha256_file(ROOT / "tools/r1_official_metric_canonicalizer.py"),
        "fixture_results": results,
        "historical_v2_compatibility": {
            "status": "PASS_CANONICALIZED_WITHOUT_INTERPRETING_METRIC_VALUES",
            "canonical_payload_sha256": historical["canonical_payload_sha256"],
            "artifact_provenance": historical["artifact_provenance"],
        },
        "official_closed_loop_runs_claimed": 0,
        "official_closed_loop_runs_started": 0,
        "budget_consumed": 0,
        "prohibited_actions_not_performed": ["official simulation", "V3 run claim", "scientific metric interpretation", "threshold tuning"],
    }
    write_json(args.output, output)
    print(json.dumps({"status": output["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
