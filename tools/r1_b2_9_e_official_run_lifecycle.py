#!/usr/bin/env python3
"""One official nuPlan lifecycle primitive shared by canary and future scientific execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_SAFETY_PARQUETS = (
    "no_ego_at_fault_collisions.parquet",
    "drivable_area_compliance.parquet",
)


def _exact_file(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file() and path.name == filename)
    if len(matches) != 1:
        raise RuntimeError(f"FULL_LIFECYCLE_EXPECTED_EXACTLY_ONE_FILE:{filename}:{len(matches)}")
    return matches[0]


def run_one_with_full_nuplan_lifecycle(
    *,
    runners: Sequence[Any],
    common_builder: Any,
    profiler_name: str,
    cfg: Any,
    run_output_root: Union[str, Path],
) -> Dict[str, Any]:
    """Execute exactly one runner through nuPlan's full run and post-run callback lifecycle."""
    if len(runners) != 1:
        raise RuntimeError(f"FULL_LIFECYCLE_EXACTLY_ONE_RUNNER_REQUIRED:{len(runners)}")
    root = Path(run_output_root)
    from nuplan.planning.script.utils import run_runners

    run_runners(list(runners), common_builder, profiler_name, cfg)
    expected = {name: _exact_file(root, name) for name in EXPECTED_SAFETY_PARQUETS}
    runner_report = _exact_file(root, str(cfg.runner_report_file))
    parquet_files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not parquet_files:
        raise RuntimeError("FULL_LIFECYCLE_NO_FINAL_METRIC_PARQUET")
    return {
        "lifecycle": "NUPLAN_RUN_RUNNERS_PLUS_POST_RUN_MAIN_CALLBACKS",
        "runner_count": 1,
        "run_runners_called": True,
        "post_run_main_callbacks_complete": True,
        "metric_parquet_count": len(parquet_files),
        "expected_safety_parquets": {name: str(path) for name, path in expected.items()},
        "runner_report": str(runner_report),
        "runner_report_available": True,
        "temporary_metric_files": len([path for path in root.rglob("*.pickle.temp") if path.is_file()]),
        "temporary_metric_is_only_final_output": False,
    }
