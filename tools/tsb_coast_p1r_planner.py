#!/usr/bin/env python3
"""Independent C2 planner; reference generation only, no execution entry."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1
from tools.r1_prospective_generator_contract_v2 import TSB_BASELINE, TSB_TREATMENT
from tools.r1_closed_loop_benchmark_v2_1 import _offset_preserving_route_xy, _headings_from_xy
from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2

SEGMENTS = {'BASELINE': ((2100000,3900000,-1.45),),
            'TREATMENT': ((1100000,2100000,-1.305),(3900000,4900000,-1.305))}


def reference_profile(speed, elapsed_us, arm, common_precontext=False):
    if arm not in SEGMENTS or not np.isfinite(speed) or speed < .2:
        raise ValueError('Invalid C2 reference input or speed below reference floor')
    effective = 'BASELINE' if common_precontext else arm
    speeds=[float(speed)];distances=[0.];commands=[]
    for i in range(79):
        left=elapsed_us+i*100000;right=left+100000
        dv=sum(a*max(0,min(right,e)-max(left,s))/1e6 for s,e,a in SEGMENTS[effective])
        nxt=max(.2,speeds[-1]+dv)
        commands.append(dv/.1);distances.append(distances[-1]+.05*(speeds[-1]+nxt));speeds.append(nxt)
    return np.asarray(speeds),np.asarray(distances),np.asarray(commands)


class CoastC2PlannerV1(R1OfficialTechnicalSmokePlannerV3_1):
    def __init__(self, spec, trace_dir, telemetry_dir):
        arm=spec['arm']
        super().__init__(spec,'R-TSB',TSB_BASELINE if arm=='BASELINE' else TSB_TREATMENT,str(trace_dir))
        self._coast_arm=arm;self._anchor_us=int(spec['start_timestamp_us'])
        self._coast_last_index=-1
        self._telemetry=Path(telemetry_dir)/'planner_transfer.jsonl'

    def name(self):
        return 'TSB_COAST_C2_P1R_'+self._coast_arm

    def compute_trajectory(self, current_input):
        index=int(current_input.iteration.index);ego,_=current_input.history.current_state
        if index!=self._coast_last_index+1 or not 0<=index<80 or ego.time_us!=self._anchor_us+index*100000 or current_input.iteration.time_us!=ego.time_us:
            raise ValueError("P1R duplicate/missing planner call or physical clock mismatch")
        result=super().compute_trajectory(current_input)
        self._coast_last_index=index
        return result

    def compute_planner_trajectory(self, current_input):
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
        if self._initialization is None:raise RuntimeError('C2 planner not initialized')
        index=int(current_input.iteration.index);ego,_=current_input.history.current_state
        elapsed=int(ego.time_us)-self._anchor_us
        if not 0<=index<80 or elapsed!=index*100000 or int(current_input.iteration.time_us)!=ego.time_us:
            raise ValueError('P1R physical episode clock mismatch')
        current=self._payload(ego);common=elapsed<1100000
        speed,distance,command=reference_profile(current['speed_mps'],elapsed,self._coast_arm,common)
        route=build_native_route_reference_v1_2(self._initialization.map_api,self._row['route_roadblock_ids'],current,float(distance[-1])+1e-9)
        xy=_offset_preserving_route_xy(route['reference_xy'],route['current_route_arc_m']+distance,current)
        headings=_headings_from_xy(xy)
        states=[EgoState.build_from_rear_axle(StateSE2(*map(float,(*xy[i],headings[i]))),
                    StateVector2D(float(speed[i]),0.),StateVector2D(0.,0.),0.,
                    TimePoint(int(ego.time_us)+i*100000),ego.car_footprint.vehicle_parameters) for i in range(80)]
        states[0]=ego
        self._telemetry.parent.mkdir(parents=True,exist_ok=True)
        with self._telemetry.open('a') as f:f.write(json.dumps({'iteration':index,'elapsed_episode_us':elapsed,
            'common_entire_future_reference':common,'reference_floor_triggered':bool(np.any(speed<=.2)),'reference_acceleration_mps2':command.tolist(),
            'reference_speed_mps':speed.tolist(),'native_route_builder':'v1.2','evidence':'PLANNED_NOT_REALIZED'})+'\n')
        return InterpolatedTrajectory(states)
