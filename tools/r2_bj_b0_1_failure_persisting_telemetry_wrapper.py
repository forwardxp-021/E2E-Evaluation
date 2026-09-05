#!/usr/bin/env python3
"""Audit-only B0.1 wrapper that atomically persists V4 architecture failures."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_bj_b0_hlc_v4_engineering_planner import (  # noqa: E402
    B0ArchitectureViolation,
    R2BJB0HLCV4EngineeringPlanner,
)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON record in the destination directory, then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class R2BJB01FailurePersistingTelemetryWrapper(R2BJB0HLCV4EngineeringPlanner):
    """Add persistence only; V4 state construction and controller input remain inherited unchanged."""

    def __init__(
        self,
        roster_row: Mapping[str, Any],
        arm: str,
        parameters: Mapping[str, Any],
        trace_dir: str,
        telemetry_dir: str,
        run_id: str,
        pair_id: str,
        component_manifest_sha256: str,
        schedule_sha256: str,
        pair_binding_sha256: str,
    ) -> None:
        super().__init__(roster_row, arm, parameters, trace_dir, telemetry_dir)
        self._b01_run_id = str(run_id)
        self._b01_pair_id = str(pair_id)
        self._b01_telemetry_dir = Path(telemetry_dir).expanduser().resolve()
        self._b01_failure_path = self._b01_telemetry_dir / "architecture_failure_audit.json"
        self._b01_controller_visible_path = self._b01_telemetry_dir / "controller_visible_telemetry.jsonl"
        self._b01_component_manifest_sha256 = str(component_manifest_sha256)
        self._b01_schedule_sha256 = str(schedule_sha256)
        self._b01_pair_binding_sha256 = str(pair_binding_sha256)

    def _call_context(self, current_input: Any) -> Mapping[str, Any]:
        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        # Do not call the inherited stateful episode clock here.  The frozen B0
        # planner must remain the sole owner of advancing that clock once per
        # planner call.  Primary80 absolute time is the frozen 0.1 s iteration
        # clock, so the pre-call failure context can be derived read-only.
        iteration = int(current_input.iteration.index)
        return {
            "iteration": iteration,
            "absolute_episode_time_s": float(iteration) * 0.1,
            "realized_current_ego": current,
        }

    def _persist_failure(self, context: Mapping[str, Any], error: B0ArchitectureViolation) -> None:
        record = {
            "schema_version": "r2_bj_b0_1_atomic_architecture_failure_audit_v1.0",
            "classification": "ARCHITECTURE_FAILURE",
            "run_id": self._b01_run_id,
            "pair_id": self._b01_pair_id,
            "arm": self._development_arm,
            "iteration": context["iteration"],
            "absolute_episode_time_s": context["absolute_episode_time_s"],
            "failure_codes": list(error.codes),
            "error_audit": error.audit,
            "realized_current_ego": context["realized_current_ego"],
            "V4_parameter_sha256": _canonical_sha(self._parameters),
            "B0_component_manifest_sha256": self._b01_component_manifest_sha256,
            "B0_schedule_sha256": self._b01_schedule_sha256,
            "B0_pair_binding_sha256": self._b01_pair_binding_sha256,
            "STOP_CURRENT_RUN": True,
            "STOP_REMAINING_SCHEDULE": True,
            "NO_RERUN": True,
            "NO_REPLACEMENT": True,
            "NO_PARAMETER_UPDATE": True,
        }
        atomic_json(self._b01_failure_path, record)

    def _persist_controller_visible_telemetry(self, context: Mapping[str, Any]) -> None:
        # The inherited B0 planner has already recorded the exact audit that governed the returned trajectory.
        rows = [line for line in self._telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        inherited = json.loads(rows[-1])
        gate = inherited["online_gate"]
        record = {
            "schema_version": "r2_bj_b0_1_controller_visible_telemetry_v1.0",
            "run_id": self._b01_run_id,
            "pair_id": self._b01_pair_id,
            "arm": self._development_arm,
            "iteration": context["iteration"],
            "absolute_episode_time_s": context["absolute_episode_time_s"],
            "controller_visible_steering_rad": gate["controller_visible_steering_rad"],
            "controller_visible_steering_finite": gate["checks"]["controller_visible_steering"],
            "source_planner_telemetry_sha256": hashlib.sha256(rows[-1].encode("utf-8")).hexdigest(),
        }
        self._b01_controller_visible_path.parent.mkdir(parents=True, exist_ok=True)
        with self._b01_controller_visible_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        context = self._call_context(current_input)
        try:
            trajectory = super().compute_planner_trajectory(current_input)
        except B0ArchitectureViolation as error:
            self._persist_failure(context, error)
            raise
        self._persist_controller_visible_telemetry(context)
        return trajectory


__all__ = ["R2BJB01FailurePersistingTelemetryWrapper", "atomic_json"]
