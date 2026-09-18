#!/usr/bin/env python3
"""Offline official nuPlan metric finalization for completed B1 arms.

This tool only integrates existing ``.pickle.temp`` metric payloads through
nuPlan's MetricFileCallback.  It never constructs or advances a simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(__file__).resolve().parents[1]
DEVKIT = ROOT.parent / "nuplan-devkit"
if str(DEVKIT) not in sys.path:
    sys.path.insert(0, str(DEVKIT))

REQUIRED_SAFETY_METRICS = frozenset({"no_ego_at_fault_collisions", "drivable_area_compliance"})
EXPECTED_OFFICIAL_METRICS = frozenset(
    {
        "corners_in_drivable_area",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "ego_is_comfortable",
        "ego_is_making_progress",
        "ego_jerk",
        "ego_lane_change",
        "ego_lat_acceleration",
        "ego_lon_acceleration",
        "ego_lon_jerk",
        "ego_progress_along_expert_route",
        "ego_yaw_acceleration",
        "ego_yaw_rate",
        "no_ego_at_fault_collisions",
        "speed_limit_compliance",
        "time_to_collision_within_bound",
    }
)
OFFICIAL_CALLBACK_SOURCE = DEVKIT / "nuplan/planning/simulation/main_callback/metric_file_callback.py"
OFFICIAL_METRIC_CONFIG = DEVKIT / "nuplan/planning/script/config/common/simulation_metric/default_metrics.yaml"
OFFICIAL_CALLBACK_CONFIG = DEVKIT / "nuplan/planning/script/config/simulation/main_callback/metric_file_callback.yaml"


class B1OfflineFinalizeError(RuntimeError):
    """Fail-closed offline metric recovery error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_metric_payload(metric_dir: Path) -> Mapping[str, Any]:
    """Verify that one existing callback payload contains all official metrics."""
    from nuplan.common.utils.io_utils import read_pickle

    inputs = sorted(metric_dir.glob("*.pickle.temp"))
    if len(inputs) != 1:
        raise B1OfflineFinalizeError(f"EXPECTED_ONE_METRIC_TEMP:{metric_dir}:{len(inputs)}")
    payload = read_pickle(inputs[0])
    if not isinstance(payload, list) or not payload:
        raise B1OfflineFinalizeError(f"METRIC_TEMP_PAYLOAD_INVALID:{inputs[0]}")
    names = {str(row.get("metric_statistics_name")) for row in payload if isinstance(row, Mapping)}
    missing = sorted(EXPECTED_OFFICIAL_METRICS - names)
    if missing:
        raise B1OfflineFinalizeError(f"OFFICIAL_METRICS_MISSING_FROM_TEMP:{inputs[0]}:{missing}")
    return {
        "input_file": str(inputs[0]),
        "input_sha256": sha256_file(inputs[0]),
        "metric_names": sorted(names),
        "required_safety_present": REQUIRED_SAFETY_METRICS <= names,
    }


def official_finalize(metric_dir: Path, output_dir: Path | None = None) -> Mapping[str, Any]:
    """Run the official nuPlan metric-file callback over preserved inputs."""
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback

    inspection = inspect_metric_payload(metric_dir)
    destination = metric_dir if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    callback = MetricFileCallback(
        metric_file_output_path=str(destination),
        scenario_metric_paths=[str(metric_dir)],
        delete_scenario_metric_files=False,
    )
    callback.on_run_simulation_end()
    outputs = {path.stem: sha256_file(path) for path in sorted(destination.glob("*.parquet"))}
    missing = sorted(EXPECTED_OFFICIAL_METRICS - set(outputs))
    if missing:
        raise B1OfflineFinalizeError(f"OFFICIAL_PARQUETS_MISSING:{destination}:{missing}")
    return {
        **inspection,
        "status": "OFFLINE_FINALIZED_FROM_ORIGINAL_B1_ARTIFACTS",
        "official_callback": "nuplan.planning.simulation.main_callback.metric_file_callback.MetricFileCallback",
        "delete_scenario_metric_files": False,
        "output_dir": str(destination),
        "output_sha256": outputs,
    }


def recover_b1(output_root: Path) -> Mapping[str, Any]:
    """Finalize every runner/recorder-complete B1 arm, preserving temp inputs."""
    recovered: list[Mapping[str, Any]] = []
    for arm_root in sorted(path for path in output_root.glob("B1-TSB-*-*") if path.is_dir()):
        metric_dir = arm_root / "raw/metrics"
        temps = list(metric_dir.glob("*.pickle.temp")) if metric_dir.is_dir() else []
        if not temps:
            continue
        result = official_finalize(metric_dir)
        original_paths = [
            arm_root / "execution_manifest.json",
            arm_root / "trace/realized_current_ego.jsonl",
            arm_root / "telemetry/planner_transfer.jsonl",
            arm_root / "telemetry/actual_lqr_controller_telemetry.jsonl",
            *sorted((arm_root / "raw/simulation_log").rglob("*.msgpack.xz")),
            *sorted(metric_dir.glob("*.pickle.temp")),
        ]
        missing_original = [str(path) for path in original_paths if not path.is_file()]
        if missing_original:
            raise B1OfflineFinalizeError(f"ORIGINAL_B1_ARTIFACT_MISSING:{arm_root.name}:{missing_original}")
        recovered.append(
            {
                "run_id": arm_root.name,
                "original_artifact_sha256": {
                    str(path.relative_to(arm_root)): sha256_file(path) for path in original_paths
                },
                **result,
            }
        )
    if len(recovered) != 34:
        raise B1OfflineFinalizeError(f"EXPECTED_34_RECOVERABLE_ARMS:{len(recovered)}")
    return {
        "schema_version": "B1_IR_offline_finalize_audit_v1",
        "status": "OFFLINE_FINALIZED_FROM_ORIGINAL_B1_ARTIFACTS",
        "simulation_count": 0,
        "runner_run_count": 0,
        "planner_step_count": 0,
        "offline_finalize_tool_sha256": sha256_file(Path(__file__)),
        "official_metric_callback_source_sha256": sha256_file(OFFICIAL_CALLBACK_SOURCE),
        "official_metric_config_sha256": sha256_file(OFFICIAL_METRIC_CONFIG),
        "official_metric_callback_config_sha256": sha256_file(OFFICIAL_CALLBACK_CONFIG),
        "recovered_arm_count": len(recovered),
        "arms": recovered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-b1", type=Path, required=True, metavar="OUTPUT_ROOT")
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    audit = recover_b1(args.recover_b1)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "recovered_arm_count": audit["recovered_arm_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
