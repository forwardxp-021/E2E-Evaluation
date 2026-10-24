#!/usr/bin/env python3
"""Synthetic official-model/controller test driven by the canonical authority."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tsb_coast_p1r_model_fixture import (ANCHOR, make_state, make_tracker,
    synthetic_trajectory, TimePoint, get_pacifica_parameters, KinematicBicycleModel, TwoStageController)
from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
from tools.tsb_coast_p1r_recorder import CoastTransitionRecorderV1
from tools.tsb_coast_p1r_c_clock import CoastCanonicalTimeControllerV1


class SyntheticClockScenario:
    database_interval = .1
    _lidarpc_tokens = tuple('SYNTHETIC_'+str(i) for i in range(81))

    def get_number_of_iterations(self):
        return 81

    def get_time_point(self, index):
        return TimePoint(ANCHOR+index*100000+round(750*math.sin(index*math.pi/13)))

    @property
    def initial_ego_state(self):
        return make_state()


def run_fixture(output_path, arm='TREATMENT'):
    output_path=Path(output_path)
    if output_path.exists():
        raise FileExistsError('Canonical fixture path already exists')
    if arm not in ('BASELINE','TREATMENT'):
        raise ValueError('Fixture arm must be BASELINE or TREATMENT')
    scenario=SyntheticClockScenario();clock=CoastCanonicalTimeControllerV1(scenario)
    clock.require_mapping()
    base=make_tracker()
    tracker=ZeroLongitudinalRequestTrackerV1(base,clock.episode_start_us,[(2100000,3900000)]) if arm=='TREATMENT' else base
    model=KinematicBicycleModel(get_pacifica_parameters())
    controller=TwoStageController(scenario,tracker,model)
    recorder=CoastTransitionRecorderV1(output_path,arm,clock.episode_start_us)
    recorder.install(controller);recorder._canonical_clock_authority=clock
    ego=controller.get_state();states=[ego];step={'index':0}
    long=base._longitudinal_lqr_controller;stop=base._stopping_controller
    def checked_long(*args,**kwargs):
        if arm=='TREATMENT' and 21<=step['index']<39:
            raise AssertionError('Coast must not call longitudinal LQR')
        return long(*args,**kwargs)
    def checked_stop(*args,**kwargs):
        if arm=='TREATMENT' and 21<=step['index']<39:
            raise AssertionError('Coast must not call stopping controller')
        return stop(*args,**kwargs)
    base._longitudinal_lqr_controller=checked_long;base._stopping_controller=checked_stop
    for i in range(79):
        step['index']=i;current=clock.get_iteration();nxt=clock.next_iteration()
        if current.index!=i or nxt.index!=i+1 or current.time_us!=ego.time_us:
            raise ValueError('Canonical fixture time authority mismatch')
        brake=(11<=i<21 or 39<=i<49) if arm=='TREATMENT' else 21<=i<39
        controller.update_state(current,nxt,ego,synthetic_trajectory(ego,brake))
        ego=controller.get_state();states.append(ego)
    recorder.validate_complete()
    if clock.get_iteration().index!=79 or clock.next_iteration() is not None:
        raise ValueError('Canonical fixture terminal step would propagate an extra state')
    return dict(evidence_kind='ISOLATED_CANONICAL_CLOCK_OFFICIAL_MODEL_NOT_SCIENTIFIC_CLOSED_LOOP',
        reference='SYNTHETIC_UNIT_SIGNAL_NOT_C2_CLOSED_LOOP',arm=arm,clock_mapping=clock.mapping_report(),
        states=[dict(time_us=s.time_us,speed_mps=s.dynamic_car_state.speed,
            model_acceleration_mps2=s.dynamic_car_state.rear_axle_acceleration_2d.x,x=s.rear_axle.x,y=s.rear_axle.y) for s in states],
        transitions=recorder.rows,canonical_model_propagations=79,states_count=80,
        simulation_runner_calls=0,formal_SESSION_state_updates=0,scientific_attempts=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for arm in ('BASELINE','TREATMENT'):
        result=run_fixture(args.output_dir/(arm+'_transitions.jsonl'),arm)
        with (args.output_dir/(arm+'_Official_Model_Fixture_v1.json')).open('x') as f:
            f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Two synthetic canonical-clock official-model fixtures complete; scientific attempts=0')
