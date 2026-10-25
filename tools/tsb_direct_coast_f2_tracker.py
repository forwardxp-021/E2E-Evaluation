#!/usr/bin/env python3
"""F2 fixed requests for both arms; no longitudinal LQR or state mutation."""
from __future__ import annotations
import math
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.tracker.tracker_utils import _generate_profile_from_initial_condition_and_derivatives

SEGMENTS={'BASELINE':((2100000,3900000,-1.45),),
          'TREATMENT':((1100000,2100000,-1.305),(3900000,4900000,-1.305))}


def request_at(arm,elapsed_us):
    if arm not in SEGMENTS or type(elapsed_us) is not int or elapsed_us<0:
        raise ValueError('F2 requires BASELINE/TREATMENT and nonnegative integer episode microseconds')
    return next((a for start,end,a in SEGMENTS[arm] if start<=elapsed_us<end),0.)


class ScriptedLongitudinalRequestTrackerV1(AbstractTracker):
    def __init__(self,tracker,episode_start_time_us,arm):
        if type(tracker) is not LQRTracker or type(episode_start_time_us) is not int or arm not in SEGMENTS:
            raise ValueError('F2 requires fresh official LQR, integer anchor and fixed arm')
        self._base=tracker;self._episode_start_time_us=episode_start_time_us;self.arm=arm
        self.lateral_calls=0;self.last_command_evidence=None

    def track_trajectory(self,current_iteration,next_iteration,initial_state,trajectory):
        now=int(current_iteration.time_point.time_us);nxt=int(next_iteration.time_point.time_us)
        elapsed=now-self._episode_start_time_us
        if initial_state.time_us!=now or nxt-now!=100000 or elapsed<0 or elapsed%100000:
            raise ValueError('F2 command/ego clock must be aligned integer 100000us transitions')
        u=request_at(self.arm,elapsed)
        v,lateral=self._base._compute_initial_velocity_and_lateral_state(current_iteration,initial_state,trajectory)
        _,curvature=self._base._compute_reference_velocity_and_curvature_profile(current_iteration,trajectory)
        if not math.isfinite(v) or v<=0 or not np.isfinite(lateral).all() or not np.isfinite(curvature).all():
            raise ValueError('F2 nonfinite or non-forward lateral control state')
        profile=_generate_profile_from_initial_condition_and_derivatives(initial_condition=v,
            derivatives=np.full(self._base._tracking_horizon,u),discretization_time=self._base._discretization_time)[:self._base._tracking_horizon]
        steer=float(self._base._lateral_lqr_controller(lateral,profile,curvature));self.lateral_calls+=1
        if not math.isfinite(steer):raise ValueError('F2 nonfinite lateral LQR request')
        result=DynamicCarState.build_from_rear_axle(rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(u,0.),tire_steering_rate=steer)
        self.last_command_evidence=dict(elapsed_episode_us=elapsed,request_mps2=u,longitudinal_LQR_calls=0,
            lateral_LQR_active=True,mode='COAST' if self.arm=='TREATMENT' and 2100000<=elapsed<3900000 else 'DIRECT_BRAKE' if u<0 else 'ZERO_REQUEST')
        return result
