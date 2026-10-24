#!/usr/bin/env python3
"""Real official model on synthetic state/poses, no scenario or simulation runner."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
from tools.tsb_coast_p1r_recorder import CoastTransitionRecorderV1

ANCHOR=1650000000000000


def make_tracker():
    return LQRTracker(q_longitudinal=[10.],r_longitudinal=[1.],q_lateral=[1.,10.,0.],r_lateral=[1.],
        discretization_time=.1,tracking_horizon=10,jerk_penalty=1e-4,curvature_rate_penalty=1e-2,
        stopping_proportional_gain=.5,stopping_velocity=.2)


def make_state(time_us=ANCHOR,speed=10.,accel=-1.,x=0.):
    return EgoState.build_from_rear_axle(StateSE2(x,0.,0.),StateVector2D(speed,0.),StateVector2D(accel,0.),0.,TimePoint(time_us),get_pacifica_parameters())


def synthetic_trajectory(ego,reference_brake=False):
    v=ego.dynamic_car_state.rear_axle_velocity_2d.x
    # Unit-amplitude analytic test signal, NOT C2 generator or scientific parameter.
    acceleration=-1. if reference_brake else 0.
    return InterpolatedTrajectory([make_state(ego.time_us+i*100000,v+acceleration*i*.1,0.,
        ego.rear_axle.x+v*i*.1+.5*acceleration*(i*.1)**2) for i in range(80)])


def run_fixture(output_path):
    output_path=Path(output_path)
    if output_path.exists():raise FileExistsError('Fixture output must be new')
    base=make_tracker();adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,[(2100000,3900000)])
    model=KinematicBicycleModel(get_pacifica_parameters())
    controller=TwoStageController(None,adapter,model)
    recorder=CoastTransitionRecorderV1(output_path,'TREATMENT',ANCHOR);recorder.install(controller)
    ego=make_state();states=[ego]
    original_long=base._longitudinal_lqr_controller;original_stop=base._stopping_controller;clock={'index':0}
    def forbid_coast_long(*args,**kwargs):
        if 21<=clock['index']<39:raise AssertionError('Longitudinal solver called in coast')
        return original_long(*args,**kwargs)
    def forbid_coast_stop(*args,**kwargs):
        if 21<=clock['index']<39:raise AssertionError('Stopping solver called in coast')
        return original_stop(*args,**kwargs)
    base._longitudinal_lqr_controller=forbid_coast_long;base._stopping_controller=forbid_coast_stop
    for i in range(79):
        clock['index']=i
        current=SimulationIteration(TimePoint(ANCHOR+i*100000),i)
        nxt=SimulationIteration(TimePoint(ANCHOR+(i+1)*100000),i+1)
        controller.update_state(current,nxt,ego,synthetic_trajectory(ego,11<=i<21 or 39<=i<49))
        ego=controller.get_state();states.append(ego)
    recorder.validate_complete()
    return {'evidence_kind':'ISOLATED_OFFICIAL_MODEL_TEST_NOT_NUPLAN_SCIENTIFIC_CLOSED_LOOP',
        'reference':'SYNTHETIC_UNIT_TEST_NOT_C2_GENERATOR','transitions':recorder.rows,
        'states':[{'time_us':s.time_us,'speed_mps':s.dynamic_car_state.speed,'state_acceleration_mps2':s.dynamic_car_state.rear_axle_acceleration_2d.x,
                   'x':s.rear_axle.x,'y':s.rear_axle.y} for s in states],
        'official_model_propagate_calls_isolated':79,'simulation_runner_calls':0,'scientific_attempts':0,
        'coast_longitudinal_solver_calls':0,'coast_stopping_solver_calls':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result=run_fixture(args.output_dir/'official_model_transitions.jsonl')
    with (args.output_dir/'Official_Model_Fixture_v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('79 actual official model transitions on synthetic data; scientific attempts=0')
