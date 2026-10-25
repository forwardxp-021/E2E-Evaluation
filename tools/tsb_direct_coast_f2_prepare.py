#!/usr/bin/env python3
"""F2 isolated-model validation, zero-run production preflight and binding."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
import numpy as np
from tools.s2r_production_executor import ROOT,write_manifest,sha256_file,canonical_sha256
from tools.tsb_direct_coast_f2_production import OUT,fixed_specs,build_f2_arm,validate_f2_production,DirectCoastF2PlannerV1
from tools.tsb_direct_coast_f2_tracker import SEGMENTS,request_at,ScriptedLongitudinalRequestTrackerV1


def new_json(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fixed_initial():
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    return EgoState.build_from_rear_axle(StateSE2(0.,0.,0.),StateVector2D(14.,0.),StateVector2D(0.,0.),0.,TimePoint(1000000000),get_pacifica_parameters())


def fixed_base():
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    return LQRTracker(q_longitudinal=np.array([10.]),r_longitudinal=np.array([1.]),q_lateral=np.array([1.,10.,0.]),r_lateral=np.array([1.]),
        discretization_time=.1,tracking_horizon=10,jerk_penalty=.0001,curvature_rate_penalty=.01,stopping_proportional_gain=.5,stopping_velocity=.2)


def straight_trajectory(state):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    v=state.dynamic_car_state.rear_axle_velocity_2d.x;pose=state.rear_axle
    states=[EgoState.build_from_rear_axle(StateSE2(pose.x+v*i*.1,pose.y,0.),StateVector2D(v,0.),StateVector2D(0.,0.),0.,
        TimePoint(state.time_us+i*100000),state.car_footprint.vehicle_parameters) for i in range(80)]
    states[0]=state
    return InterpolatedTrajectory(states)


def isolate():
    """Exactly one fixed two-arm model preflight, no SESSION/runner/scientific claim."""
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from nuplan.common.actor_state.state_representation import TimePoint
    from tools.tsb_coast_p1r_analysis import audit_proxy_response,fmatch_pair
    destination=OUT/'evidence/F2_Isolated_Model_Validation_v1.json'
    if destination.exists():raise FileExistsError('F2 isolated model validation is one-shot')
    result=dict(status='BLOCKED',scope='FIXED_SYNTHETIC_INITIAL_STATE_OFFICIAL_MODEL_INTERFACE_ONLY_NOT_SCIENTIFIC_QUALIFICATION',scientific_attempts=0,arms=[])
    data=[]
    try:
        for arm in ('BASELINE','TREATMENT'):
            state=fixed_initial();anchor=state.time_us;base=fixed_base();tracker=ScriptedLongitudinalRequestTrackerV1(base,anchor,arm)
            model=KinematicBicycleModel(state.car_footprint.vehicle_parameters);times=[anchor];speeds=[14.];xy=[[0.,0.]];rows=[]
            with patch.object(base,'track_trajectory',side_effect=AssertionError('F2 full longitudinal tracker forbidden')),patch.object(base,'_longitudinal_lqr_controller',side_effect=AssertionError('F2 longitudinal LQR forbidden')),patch.object(base,'_stopping_controller',side_effect=AssertionError('F2 stopping forbidden')):
                for i in range(79):
                    now=SimulationIteration(TimePoint(anchor+i*100000),i);nxt=SimulationIteration(TimePoint(anchor+(i+1)*100000),i+1)
                    command=tracker.track_trajectory(now,nxt,state,straight_trajectory(state));u=float(command.rear_axle_acceleration_2d.x)
                    before=float(state.dynamic_car_state.rear_axle_acceleration_2d.x);v0=float(state.dynamic_car_state.rear_axle_velocity_2d.x)
                    output=model.propagate_state(state,command,TimePoint(100000));a=float(output.dynamic_car_state.rear_axle_acceleration_2d.x);v1=float(output.dynamic_car_state.rear_axle_velocity_2d.x)
                    row=dict(iteration=i,time_us=now.time_us,next_time_us=nxt.time_us,tracking_mode=tracker.last_command_evidence['mode'],
                        reference_request_mps2=request_at(arm,i*100000),longitudinal_requested_acceleration=u,steering_request=float(command.tire_steering_rate),
                        model_acceleration_before=before,model_realized_acceleration=a,model_expected_acceleration=before+(u-before)/3,
                        model_filter_tau_s=.2,speed_before=v0,realized_speed=v1,command_identity_reached_model=True,
                        coast_command_reached_model=arm=='TREATMENT' and 21<=i<39,state_propagation_continues=output is not state and output.time_us==nxt.time_us,
                        displacement_m=float(np.hypot(output.rear_axle.x-state.rear_axle.x,output.rear_axle.y-state.rear_axle.y)),
                        x_m=float(output.rear_axle.x),y_m=float(output.rear_axle.y),lateral_LQR_calls=tracker.lateral_calls,longitudinal_LQR_calls=0)
                    if u!=request_at(arm,i*100000) or u>0 or row['displacement_m']<=0 or not row['state_propagation_continues']:raise ValueError('F2 isolated interface/propagation failure')
                    rows.append(row);state=output;times.append(state.time_us);speeds.append(v1);xy.append([state.rear_axle.x,state.rear_axle.y])
            mechanism=audit_proxy_response(times,speeds,rows,arm)
            if not mechanism['proposal_diagnostic_pass'] or tracker.lateral_calls!=79:raise ValueError('F2 isolated mechanism/lateral validation failed: '+str(mechanism['reasons']))
            result['arms'].append(dict(arm=arm,initial_state=dict(x=0.,y=0.,heading=0.,vx=14.,a=0.,time_us=anchor),mechanism=mechanism,transitions=rows,
                final_speed_mps=speeds[-1],path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),lateral_LQR_calls=tracker.lateral_calls))
            data.append((times,speeds,xy))
        comparison=fmatch_pair(data[0][0],data[0][1],data[0][2],data[1][1],data[1][2])
        result['model_fixture_motion_comparison_not_formal_Fmatch']=comparison
        if not comparison['pass']:raise ValueError('F2 isolated motion closeness did not meet unchanged fixture tolerances')
        result['status']='PASS'
    except Exception as error:
        result['error']=type(error).__name__+': '+str(error);new_json(destination,result);raise
    new_json(destination,result);return result


def preflight():
    from tools.b1_production_preflight import zero_run_guard
    from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
    from tools.tsb_coast_p1r_planner import reference_profile
    target=OUT/'evidence/F2_Production_Preflight_v1.json'
    if target.exists():raise FileExistsError('F2 preflight evidence already exists')
    if json.loads((OUT/'evidence/F2_Isolated_Model_Validation_v1.json').read_text())['status']!='PASS':raise PermissionError('F2 model validation must pass first')
    held=[];proofs=[];result=dict(status='BLOCKED',arms=[],scientific_attempts=0)
    try:
        with zero_run_guard() as calls,patch.object(DirectCoastF2PlannerV1,'initialize',side_effect=RuntimeError('F2_ZERO_RUN_ONLY')),patch.object(DirectCoastF2PlannerV1,'compute_trajectory',side_effect=RuntimeError('F2_ZERO_RUN_ONLY')):
            for spec in fixed_specs():
                arm=build_f2_arm(spec,ROOT/'outputs/stageR/tsb_direct_coast_f2_preflight_v1'/spec['run_id']);held.append(arm)
                proof=validate_f2_production(arm);initial=arm.simulation._scenario.get_ego_state_at_iteration(0)
                current=arm.planner._payload(initial);_,distance,_=reference_profile(current['speed_mps'],0,spec['arm'],True)
                route=build_native_route_reference_v1_2(arm.simulation._scenario.map_api,spec['route_roadblock_ids'],current,float(distance[-1])+1e-9)
                old=json.loads((ROOT/'outputs/stageR/tsb_coast_p1r_d_first_pair_v1'/('P1R-C2-DEV-01-'+spec['arm'])/'canonical_clock_mapping.json').read_text())
                if proof['clock']['rows']!=old['rows']:raise ValueError('F2 original raw observation time/token mapping changed')
                proofs.append(dict(run_id=spec['run_id'],status='PASS',precontext_id=spec['precontext_id'],route_id=spec['route_id'],initial_state=current,
                    validation=proof,config_path=str(arm.run_root/'resolved_config.json'),config_sha256=sha256_file(arm.run_root/'resolved_config.json'),
                    finite_initial_route_query=route['extrapolation_used'],history_rows=len(arm.simulation.history),recorder_rows=arm.recorder.row_count,
                    component_ids={name:id(getattr(arm,name)) for name in ('runner','planner','controller','tracker','motion_model','recorder')}))
            if proofs[0]['initial_state']!=proofs[1]['initial_state'] or any(calls.values()):raise ValueError('F2 initial pair equality or zero-run guard failed')
            for key in proofs[0]['component_ids']:
                if proofs[0]['component_ids'][key]==proofs[1]['component_ids'][key]:raise ValueError('F2 component reuse')
        result.update(status='PASS',arms=proofs,zero_run_guard_calls=calls)
    except Exception as error:
        result.update(error=type(error).__name__+': '+str(error),arms=proofs);new_json(target,result);raise
    new_json(target,result);return result


def freeze():
    """After source commit, bind fresh successful checks and new zero-attempt ledger."""
    from tools.tsb_direct_coast_f2_executor import AUTH,LEDGER,SCIENCE_ROOT,validate_credential
    if AUTH.exists() or LEDGER.exists():raise FileExistsError('F2 authorization or budget already exists')
    for name in ['F2_Isolated_Model_Validation_v1.json','F2_Production_Preflight_v1.json']:
        if json.loads((OUT/'evidence'/name).read_text())['status']!='PASS':raise PermissionError('F2 preflight blocked')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_clock_closure/P1R_C_Execution_Source_Binding_v1.json').read_text())
    source=dict(old['source_config_sha256'])
    for path in list((ROOT/'tools').glob('tsb_direct_coast_f2*.py'))+[ROOT/'tests/test_tsb_direct_coast_f2.py',ROOT/'configs/r1_official_technical_smoke_hydra/planner/tsb_direct_coast_f2.yaml']:
        name=str(path.relative_to(ROOT));source[name]=sha256_file(path)
    for name,digest in source.items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('F2 source hash changed: '+name)
        if not name.startswith('../') and not Path(name).is_absolute():
            import hashlib
            if hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=ROOT)).hexdigest()!=digest:raise ValueError('F2 source not committed: '+name)
    specs=fixed_specs();database=specs[0]['db_path']
    data={name:digest for name,digest in old['data_sha256'].items() if name==database or name.endswith('nuplan-maps-v1.0.json') or ('us-pa-pittsburgh-hazelwood' in name and name.endswith('map.gpkg'))}
    if str(database) not in data:raise ValueError('F2 source DB hash not found')
    for name,digest in data.items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('F2 data hash changed')
    runtime=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_clock_closure/evidence/Runtime_v1.json').read_text())
    if sys.version.split()[0]!=runtime['scientific_runtime_python'] or sys.executable!=runtime['scientific_runtime_executable']:raise ValueError('F2 runtime differs')
    for name,version in runtime['dependency_versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('F2 runtime dependency differs: '+name)
    ledger=dict(stage='TSB_DIRECT_COAST_F2',attempts=[],cpu_seconds_consumed=0.,stopped=False,maximum_new_attempts=2,scientific_CPU_seconds_max=3600,per_arm_wall_seconds_max=1200,automatic_retry_authorized=False,old_C2_ledger_not_reused=True)
    LEDGER.parent.mkdir(parents=True,exist_ok=True);new_json(LEDGER,ledger)
    binding=dict(source_commit=head,source_config_sha256=source,data_sha256=data,runtime=runtime,
        protocol_sha256=sha256_file(OUT/'F2_Owner_Protocol_v1.json'),specs_sha256=sha256_file(OUT/'F2_Arm_Specs_v1.json'),
        isolated_validation_sha256=sha256_file(OUT/'evidence/F2_Isolated_Model_Validation_v1.json'),preflight_sha256=sha256_file(OUT/'evidence/F2_Production_Preflight_v1.json'))
    new_json(OUT/'F2_Execution_Source_Binding_v1.json',binding)
    auth=dict(stage='TSB_DIRECT_COAST_F2',authorization_status='AUTHORIZED',scientific_owner_approved=True,source_commit=head,
        approved_run_ids=[s['run_id'] for s in specs],approved_spec_sha256={s['run_id']:canonical_sha256(s) for s in specs},
        binding_sha256=sha256_file(OUT/'F2_Execution_Source_Binding_v1.json'),protocol_sha256=binding['protocol_sha256'],maximum_new_attempts=2,
        execution_ledger_path=str(LEDGER),scientific_output_root=str(SCIENCE_ROOT),initial_ledger_sha256=sha256_file(LEDGER),
        budget=dict(cpu_seconds=3600,per_arm_wall_seconds=1200),technical_retry_authorized=False,disclosed_observation_skew_acknowledged=True)
    new_json(AUTH,auth)
    for spec in specs:validate_credential(spec['run_id'])
    new_json(OUT/'F2_Initial_Execution_Ledger_v1.json',ledger)
    return dict(status='AUTHORIZED_FIRST_DEV_PAIR_ONLY',source_commit=head)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['isolate','preflight','freeze']);args=parser.parse_args()
    result={'isolate':isolate,'preflight':preflight,'freeze':freeze}[args.mode]()
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2,ensure_ascii=False))
