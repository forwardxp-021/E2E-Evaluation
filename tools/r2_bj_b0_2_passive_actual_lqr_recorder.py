#!/usr/bin/env python3
"""Passive, behavior-preserving actual-LQR recorder for the BJ-B0 canary."""

from __future__ import annotations

import json
import math
import os
import tempfile
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_TRANSITIONS = 79
SHADOW_TOLERANCE = 1e-12


def _frozen_velocity_profile(initial_velocity: float, acceleration: float, tracker: Any) -> np.ndarray:
    from nuplan.planning.simulation.controller.tracker.tracker_utils import (
        _generate_profile_from_initial_condition_and_derivatives,
    )
    return _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=np.ones(tracker._tracking_horizon, dtype=np.float64) * acceleration,
        discretization_time=tracker._discretization_time,
    )[: tracker._tracking_horizon]


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"R2_BJ_B0_2_NONFINITE_{label}")
    return number


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Replace the complete JSONL atomically so no partial final row can survive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class PassiveActualLQRRecorderV1:
    """Wrap one frozen LQR instance without changing its inputs or returned object."""

    def __init__(
        self,
        output_path: Path,
        run: Mapping[str, Any],
        component_shas: Mapping[str, str],
        expected_rows: int = EXPECTED_TRANSITIONS,
    ) -> None:
        self.output_path = Path(output_path)
        self.run = dict(run)
        self.component_shas = dict(component_shas)
        self.expected_rows = int(expected_rows)
        self.rows: list[Mapping[str, Any]] = []
        self._tracker: Any = None
        self._original: Any = None
        self._episode_start_s: float | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def validate_installability(self, ego_controller: Any, time_controller: Any) -> None:
        if ego_controller.__class__.__name__ != "TwoStageController":
            raise TypeError(f"R2_BJ_B0_2_EXPECTED_TWO_STAGE_CONTROLLER:{ego_controller.__class__.__name__}")
        tracker = getattr(ego_controller, "_tracker", None)
        if tracker is None or tracker.__class__.__name__ != "LQRTracker":
            raise TypeError(f"R2_BJ_B0_2_EXPECTED_LQR_TRACKER:{type(tracker).__name__}")
        if time_controller.__class__.__name__ != "R1Primary80ScientificTimeControllerV1":
            raise TypeError(f"R2_BJ_B0_2_EXPECTED_PRIMARY80_CONTROLLER:{time_controller.__class__.__name__}")
        if int(time_controller.number_of_iterations()) != 81:
            raise ValueError("R2_BJ_B0_2_PRIMARY80_ITERATIONS_NOT_81")
        if getattr(tracker, "_r2_bj_b0_2_passive_recorder_installed", False):
            raise RuntimeError("R2_BJ_B0_2_RECORDER_ALREADY_INSTALLED")

    def install(self, ego_controller: Any, time_controller: Any) -> None:
        self.validate_installability(ego_controller, time_controller)
        tracker = ego_controller._tracker
        original = tracker.track_trajectory
        recorder = self

        def wrapped(_tracker: Any, current_iteration: Any, next_iteration: Any, initial_state: Any, trajectory: Any) -> Any:
            initial_velocity, initial_lateral = _tracker._compute_initial_velocity_and_lateral_state(
                current_iteration, initial_state, trajectory
            )
            reference_velocity, curvature = _tracker._compute_reference_velocity_and_curvature_profile(
                current_iteration, trajectory
            )
            should_stop = reference_velocity <= _tracker._stopping_velocity and initial_velocity <= _tracker._stopping_velocity
            if should_stop:
                shadow_acceleration, shadow_steering = _tracker._stopping_controller(
                    initial_velocity, reference_velocity
                )
            else:
                shadow_acceleration = _tracker._longitudinal_lqr_controller(initial_velocity, reference_velocity)
                velocity_profile = _frozen_velocity_profile(initial_velocity, shadow_acceleration, _tracker)
                shadow_steering = _tracker._lateral_lqr_controller(initial_lateral, velocity_profile, curvature)

            result = original(current_iteration, next_iteration, initial_state, trajectory)
            actual_acceleration = _finite(result.rear_axle_acceleration_2d.x, "ACTUAL_ACCELERATION")
            actual_steering = _finite(result.tire_steering_rate, "ACTUAL_STEERING_RATE")
            shadow_acceleration = _finite(shadow_acceleration, "SHADOW_ACCELERATION")
            shadow_steering = _finite(shadow_steering, "SHADOW_STEERING_RATE")
            initial_velocity = _finite(initial_velocity, "INITIAL_LONGITUDINAL_STATE")
            lateral = [_finite(value, "INITIAL_LATERAL_STATE") for value in initial_lateral]
            curvature_values = [_finite(value, "REFERENCE_CURVATURE") for value in curvature]
            reference_velocity = _finite(reference_velocity, "REFERENCE_VELOCITY")
            time_s = _finite(current_iteration.time_point.time_s, "ITERATION_TIME")
            if recorder._episode_start_s is None:
                recorder._episode_start_s = time_s
            row = {
                "schema_version": "r2_bj_b0_2_passive_actual_lqr_telemetry_v1.0",
                "instrumentation": "PASSIVE_ACTUAL_LQR_RETURN_PLUS_INDEPENDENT_FROZEN_SHADOW",
                "run_id": str(recorder.run["run_id"]),
                "pair_id": str(recorder.run["pair_id"]),
                "arm": str(recorder.run["arm"]),
                "iteration": int(current_iteration.index),
                "absolute_episode_time_s": time_s - recorder._episode_start_s,
                "actual_acceleration_command_mps2": actual_acceleration,
                "actual_tire_steering_rate_command_radps": actual_steering,
                "shadow_acceleration_command_mps2": shadow_acceleration,
                "shadow_tire_steering_rate_command_radps": shadow_steering,
                "initial_longitudinal_state_mps": initial_velocity,
                "initial_lateral_state": lateral,
                "reference_velocity_mps": reference_velocity,
                "reference_curvature_profile_inv_m": curvature_values,
                "acceleration_direction_agreement": bool(
                    abs(actual_acceleration) <= SHADOW_TOLERANCE and abs(shadow_acceleration) <= SHADOW_TOLERANCE
                    or np.sign(actual_acceleration) == np.sign(shadow_acceleration)
                ),
                "steering_direction_agreement": bool(
                    abs(actual_steering) <= SHADOW_TOLERANCE and abs(shadow_steering) <= SHADOW_TOLERANCE
                    or np.sign(actual_steering) == np.sign(shadow_steering)
                ),
                "absolute_acceleration_difference_mps2": abs(actual_acceleration - shadow_acceleration),
                "absolute_steering_rate_difference_radps": abs(actual_steering - shadow_steering),
                "component_sha256": recorder.component_shas,
                "behavior_changed": False,
            }
            candidate = [*recorder.rows, row]
            if len(candidate) > recorder.expected_rows:
                raise RuntimeError("R2_BJ_B0_2_ACTUAL_CONTROLLER_TELEMETRY_EXCEEDS_79")
            _atomic_jsonl(recorder.output_path, candidate)
            recorder.rows = candidate
            return result

        self._tracker, self._original = tracker, original
        tracker.track_trajectory = types.MethodType(wrapped, tracker)
        tracker._r2_bj_b0_2_passive_recorder_installed = True

    def uninstall(self) -> None:
        if self._tracker is not None and self._original is not None:
            self._tracker.track_trajectory = self._original
            self._tracker._r2_bj_b0_2_passive_recorder_installed = False
        self._tracker = self._original = None

    def validate_complete(self) -> None:
        if self.row_count != self.expected_rows:
            raise RuntimeError(
                f"R2_BJ_B0_2_ACTUAL_CONTROLLER_TELEMETRY_CARDINALITY:{self.row_count}!={self.expected_rows}"
            )
        expected = list(range(self.expected_rows))
        actual = [int(row["iteration"]) for row in self.rows]
        if actual != expected:
            raise RuntimeError("R2_BJ_B0_2_ACTUAL_CONTROLLER_ITERATIONS_NOT_EXACT_0_78")


__all__ = ["EXPECTED_TRANSITIONS", "SHADOW_TOLERANCE", "PassiveActualLQRRecorderV1"]
