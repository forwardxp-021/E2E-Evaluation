#!/usr/bin/env python3
"""Strict actual-command to official model propagation recorder, new namespace."""
from __future__ import annotations
import math
import os
from pathlib import Path
import sys
import types
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.r2_bj_b0_2_passive_actual_lqr_recorder import _atomic_jsonl, _frozen_velocity_profile
from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel


def fp_bound(*values):
    return 64*np.finfo(np.float64).eps*max(1.,*(abs(float(x)) for x in values))


class CoastTransitionRecorderV1:
    def __init__(self,path,arm,anchor_us):
        self.output_path=Path(path);self.arm=arm;self.anchor_us=int(anchor_us)
        self.rows=[];self.pending=None;self._installed=False

    @property
    def row_count(self):return len(self.rows)

    def install(self,controller,time_controller=None):
        if self._installed or type(controller) is not TwoStageController or type(controller._motion_model) is not KinematicBicycleModel:
            raise TypeError('Expected fresh official TwoStageController and KinematicBicycleModel')
        tracker=controller._tracker;model=controller._motion_model
        if self.arm=='TREATMENT':
            if type(tracker) is not ZeroLongitudinalRequestTrackerV1 or type(tracker._base) is not LQRTracker:
                raise TypeError('Treatment requires exact audited adapter and inner LQR')
            base=tracker._base
            if tracker._episode_start_time_us!=self.anchor_us or tracker._coast_intervals_us!=((2100000,3900000),):
                raise ValueError('Treatment clock/interval mismatch')
        elif self.arm=='BASELINE' and type(tracker) is LQRTracker:base=tracker
        else:raise TypeError('Baseline requires unwrapped official LQR')
        if getattr(tracker,'_p1r_recorder_installed',False) or getattr(model,'_p1r_recorder_installed',False):
            raise ValueError('Already instrumented')
        if time_controller is not None and (time_controller.__class__.__name__!='R1Primary80ScientificTimeControllerV1' or time_controller.number_of_iterations()!=81):
            raise ValueError('Wrong Primary80 time controller')
        track=tracker.track_trajectory;propagate=model.propagate_state;rec=self

        def wrapped_track(_tracker,current_iteration,next_iteration,initial_state,trajectory):
            i=int(current_iteration.index);now=int(current_iteration.time_point.time_us);nxt=int(next_iteration.time_point.time_us)
            if rec.pending is not None or not 0<=i<79 or i!=len(rec.rows) or now!=rec.anchor_us+i*100000 or nxt!=now+100000 or initial_state.time_us!=now:
                raise ValueError('Duplicate, missing or misaligned transition')
            result=track(current_iteration,next_iteration,initial_state,trajectory)
            coast=rec.arm=='TREATMENT' and 2100000<=now-rec.anchor_us<3900000
            v,lateral=base._compute_initial_velocity_and_lateral_state(current_iteration,initial_state,trajectory)
            ref,curvature=base._compute_reference_velocity_and_curvature_profile(current_iteration,trajectory)
            if coast:
                shadow_a=0.;profile=np.full(base._tracking_horizon,v);label='COAST_ZERO_RULE_NO_COUNTERFACTUAL_LONGITUDINAL_CALL'
                shadow_steer=base._lateral_lqr_controller(lateral,profile,curvature)
            elif ref<=base._stopping_velocity and v<=base._stopping_velocity:
                shadow_a,shadow_steer=base._stopping_controller(v,ref);label='OFFICIAL_APPLIED_SHADOW'
            else:
                shadow_a=base._longitudinal_lqr_controller(v,ref)
                shadow_steer=base._lateral_lqr_controller(lateral,_frozen_velocity_profile(v,shadow_a,base),curvature);label='OFFICIAL_APPLIED_SHADOW'
            a=float(result.rear_axle_acceleration_2d.x);steer=float(result.tire_steering_rate)
            if not all(math.isfinite(x) for x in (a,steer,shadow_a,shadow_steer)) or abs(a-shadow_a)>1e-12 or abs(steer-shadow_steer)>1e-12:
                raise ValueError('Actual/shadow command mismatch')
            if coast and a!=0.:raise ValueError('Nonzero coast command')
            mode='COAST' if coast else ('BRAKING' if (rec.arm=='BASELINE' and 2100000<=now-rec.anchor_us<3900000 or rec.arm=='TREATMENT' and (1100000<=now-rec.anchor_us<2100000 or 3900000<=now-rec.anchor_us<4900000)) else 'TRACKING')
            rec.pending=(result,initial_state,dict(iteration=i,time_us=now,next_time_us=nxt,tracking_mode=mode,
                longitudinal_requested_acceleration=a,steering_request=steer,shadow_acceleration=shadow_a,shadow_kind=label))
            return result

        def wrapped_model(_model,state,ideal_dynamic_state,sampling_time):
            if rec.pending is None:raise ValueError('Propagation without recorded command')
            command,input_state,row=rec.pending
            if command is not ideal_dynamic_state or input_state is not state or sampling_time.time_us!=100000:
                raise ValueError('Command did not reach the bound model or duplicate update')
            before=float(state.dynamic_car_state.rear_axle_acceleration_2d.x);v0=float(state.dynamic_car_state.rear_axle_velocity_2d.x)
            output=propagate(state,ideal_dynamic_state,sampling_time)
            a=float(output.dynamic_car_state.rear_axle_acceleration_2d.x);v1=float(output.dynamic_car_state.rear_axle_velocity_2d.x)
            expected=before+.1/(.1+model._accel_time_constant)*(row['longitudinal_requested_acceleration']-before)
            if output is state or output.time_us!=row['next_time_us'] or not all(math.isfinite(x) for x in (before,a,v0,v1,output.rear_axle.x,output.rear_axle.y)):
                raise ValueError('State propagation missing/nonfinite')
            if abs(a-expected)>fp_bound(a,expected,before) or abs(v1-(v0+.1*a))>fp_bound(v1,v0):
                raise ValueError('Official model equation mismatch')
            row.update(model_realized_acceleration=a,model_acceleration_before=before,realized_speed=v1,
                speed_before=v0,speed_difference_acceleration=(v1-v0)/.1,model_expected_acceleration=expected,
                model_filter_tau_s=float(model._accel_time_constant),coast_command_reached_model=(row['tracking_mode']=='COAST'),
                command_identity_reached_model=True,state_propagation_continues=True,
                displacement_m=float(np.hypot(output.rear_axle.x-state.rear_axle.x,output.rear_axle.y-state.rear_axle.y)))
            _atomic_jsonl(rec.output_path,[*rec.rows,row]);rec.rows.append(row);rec.pending=None
            return output

        tracker.track_trajectory=types.MethodType(wrapped_track,tracker)
        model.propagate_state=types.MethodType(wrapped_model,model)
        tracker._p1r_recorder_installed=model._p1r_recorder_installed=True
        self._tracker=tracker;self._base=base;self._model=model;self._controller=controller;self._installed=True
        self._bound_track=tracker.track_trajectory;self._bound_propagate=model.propagate_state

    def validate_complete(self):
        if len(self.rows)!=79 or self.pending is not None or [x['iteration'] for x in self.rows]!=list(range(79)):
            raise ValueError('Expected all 79 complete bound transitions')
