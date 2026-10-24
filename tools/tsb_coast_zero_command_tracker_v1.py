#!/usr/bin/env python3
"""Opt-in tracker adapter: disable longitudinal feedback during explicit coast.

No runner, scenario selection, execution authorization or production registration
is provided. The stock TwoStageController/motion model remain unchanged.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVKIT = Path(__file__).resolve().parents[2]/'nuplan-devkit'
if DEVKIT.is_dir() and str(DEVKIT) not in sys.path:
    sys.path.insert(0, str(DEVKIT))

import numpy as np
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker


class ZeroLongitudinalRequestTrackerV1(AbstractTracker):
    """Keep stock tracking outside coast; bypass longitudinal solvers inside.

    Intervals are half-open, integer microseconds relative to an explicit
    episode anchor. They are caller supplied, never selected by this adapter.
    Steering still tracks the route using a constant-current-speed prediction.
    The vehicle model, including its acceleration lag, must still be advanced
    by TwoStageController. This is an abstract zero-request mode, not torque.
    """

    def __init__(self, tracker: LQRTracker, episode_start_time_us: int,
                 coast_intervals_us: Sequence[Tuple[int, int]]):
        if type(tracker) is not LQRTracker:
            raise TypeError('Expected the audited official LQRTracker instance')
        if type(episode_start_time_us) is not int:
            raise ValueError('Episode anchor must be an integer microsecond timestamp')
        intervals = []
        previous_end = 0
        for start, end in coast_intervals_us:
            if type(start) is not int or type(end) is not int or start < previous_end or end <= start:
                raise ValueError('Coast intervals must be ordered nonoverlapping nonnegative integer [start,end) microseconds')
            intervals.append((start, end))
            previous_end = end
        if not intervals:
            raise ValueError('Explicit nonempty coast intervals are required')
        self._base = tracker
        self._episode_start_time_us = episode_start_time_us
        self._coast_intervals_us = tuple(intervals)
        self._last_command_evidence = None

    @property
    def last_command_evidence(self):
        """A copy of the last returned packet's metadata; not actuator telemetry."""
        return None if self._last_command_evidence is None else dict(self._last_command_evidence)

    def coast_active(self, current_time_us: int) -> bool:
        if type(current_time_us) is not int:
            raise ValueError('Current timestamp must be integer microseconds')
        elapsed = current_time_us-self._episode_start_time_us
        if elapsed < 0:
            raise ValueError('Current timestamp precedes explicit episode anchor')
        return any(start <= elapsed < end for start, end in self._coast_intervals_us)

    def track_trajectory(self, current_iteration, next_iteration, initial_state, trajectory):
        self._last_command_evidence = None
        now = int(current_iteration.time_point.time_us)
        next_us = int(next_iteration.time_point.time_us)
        if now != int(initial_state.time_us) or next_us <= now:
            raise ValueError('Current ego/iteration timestamps must agree and next time must advance')
        coast = self.coast_active(now)
        elapsed, next_elapsed = now-self._episode_start_time_us, next_us-self._episode_start_time_us
        if any(elapsed < boundary < next_elapsed for interval in self._coast_intervals_us for boundary in interval):
            raise ValueError('Control step straddles a coast boundary; align steps before execution')
        if not coast:
            # Preserve the exact official behavior and the exact returned object.
            result = self._base.track_trajectory(current_iteration, next_iteration, initial_state, trajectory)
        else:
            # Do not call track_trajectory, longitudinal LQR or the stopping P
            # controller. Future reference speed is extracted but never used
            # to request longitudinal actuation in this branch.
            velocity, lateral_state = self._base._compute_initial_velocity_and_lateral_state(
                current_iteration, initial_state, trajectory)
            _, curvature = self._base._compute_reference_velocity_and_curvature_profile(
                current_iteration, trajectory)
            if not math.isfinite(velocity) or velocity < 0 or not np.isfinite(lateral_state).all() or not np.isfinite(curvature).all():
                raise ValueError('Coast requires finite forward/nonnegative velocity and finite route state')
            if np.shape(lateral_state) != (3,) or np.shape(curvature) != (self._base._tracking_horizon,):
                raise ValueError('Coast route state must have lateral shape (3,) and one horizon of curvature')
            # This matches zero requested acceleration; it deliberately avoids
            # an ungated brake/acceleration forecast in the lateral LQR.
            # It does not predict the stock motion model's residual accel lag.
            velocity_profile = np.full(self._base._tracking_horizon, velocity, dtype=np.float64)
            steering_rate = self._base._lateral_lqr_controller(lateral_state, velocity_profile, curvature)
            if not math.isfinite(steering_rate):
                raise ValueError('Nonfinite steering request')
            result = DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
                rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_rate=steering_rate,
            )
        accel = float(result.rear_axle_acceleration_2d.x)
        steering = float(result.tire_steering_rate)
        if not math.isfinite(accel) or not math.isfinite(steering):
            raise ValueError('Nonfinite final command packet')
        self._last_command_evidence = dict(
            schema_version='zero_longitudinal_request_tracker_v1',
            current_time_us=now, next_time_us=next_us,
            elapsed_episode_time_us=now-self._episode_start_time_us,
            mode='COAST_ZERO_LONGITUDINAL_REQUEST' if coast else 'OFFICIAL_TRACKING',
            longitudinal_tracking_disabled=coast,
            returned_acceleration_request_mps2=accel,
            returned_tire_steering_rate_radps=steering,
            scope='TRACKER_RETURN_PACKET_BEFORE_MOTION_MODEL_FILTER',
        )
        return result
