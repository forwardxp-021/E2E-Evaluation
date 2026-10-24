#!/usr/bin/env python3
"""P1R-D first-pair final preflight only. Never starts a scientific runner."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT, canonical_sha256, sha256_file

OUT=ROOT/'docs/stageR/tsb_coast_p1r_first_pair'
AUTH=OUT/'P1R_D_Owner_FirstPair_Authorization_v1.json'
SOURCE='73e96f7e6fa1c39bca0f2cd0f9f2947646024c93'


def encoded(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(encoded(value))


def normalized_config(value,arm_root,uid):
    if isinstance(value,dict):return {k:normalized_config(v,arm_root,uid) for k,v in value.items()}
    if isinstance(value,list):return [normalized_config(v,arm_root,uid) for v in value]
    if isinstance(value,str):
        for prefix in (str(Path(arm_root).resolve()),os.path.relpath(arm_root,ROOT)):
            value=value.replace(prefix,'<ARM_OUTPUT_ROOT>')
        return value.replace(uid,'<RUN_START_TIMESTAMP>')
    return value


def static_checks(auth):
    if auth['executable_source_commit']!=SOURCE or auth['maximum_new_attempts']!=2 or auth['technical_retry_authorized']:
        raise ValueError('P1R_D source/attempt/retry authorization mismatch')
    if sha256_file(ROOT/auth['owner_instruction_path'])!=auth['owner_instruction_sha256']:
        raise ValueError('P1R_D Owner instruction hash mismatch')
    binding_path=ROOT/auth['execution_source_binding_path'];binding=json.loads(binding_path.read_text())
    for field,key in [('execution_source_binding_sha256','execution_source_binding_path'),('approved_amendment_sha256','approved_amendment_path'),('production_preflight_sha256','production_preflight_path')]:
        if sha256_file(ROOT/auth[key])!=auth[field]:raise ValueError('P1R_D frozen contract/binding hash mismatch: '+field)
    changed=[p for group in ('source_config_sha256','data_sha256') for p,h in binding[group].items() if sha256_file(ROOT/p)!=h]
    if changed:raise ValueError('P1R_D source/config/data changed: '+repr(changed))
    root_sources={p:h for p,h in binding['source_config_sha256'].items() if not p.startswith('../') and not Path(p).is_absolute()}
    for p,h in root_sources.items():
        blob=subprocess.check_output(['git','show',SOURCE+':'+p],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=h:raise ValueError('P1R_D executable source commit mismatch: '+p)
    runtime_path=ROOT/'docs/stageR/tsb_coast_p1r_clock_closure/evidence/Runtime_v1.json'
    runtime=json.loads(runtime_path.read_text())
    if sha256_file(runtime_path)!=binding['runtime_sha256'] or sys.version.split()[0]!=runtime['scientific_runtime_python'] or sys.executable!=runtime['scientific_runtime_executable']:
        raise ValueError('P1R_D Python runtime binding mismatch')
    observed={}
    for name,version in runtime['dependency_versions'].items():
        try:observed[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:observed[name]='METADATA_UNAVAILABLE'
        if observed[name]!=version:raise ValueError('P1R_D dependency version mismatch: '+name)
    ledger_path=ROOT/auth['execution_ledger_path'];ledger=json.loads(ledger_path.read_text())
    if sha256_file(ledger_path)!=auth['prior_budget_sha256'] or ledger['historical_attempts'] or ledger['attempts'] or ledger['cpu_seconds_consumed'] or ledger['stopped']:
        raise ValueError('P1R_D initial cumulative budget is not the reviewed zero-attempt ledger')
    if auth['global_limits']!=dict(development_pairs=3,first_attempts=6,maximum_cumulative_attempts=8,scientific_CPU_seconds=10800,per_arm_wall_seconds=1200):
        raise ValueError('P1R_D frozen global budget changed')
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    approved=json.loads((ROOT/auth['approved_amendment_path']).read_text())
    if approved['approved_mechanism_specification']!=SCIENCE_CONTRACT or approved['D1']!='APPROVED' or approved['D2']!='APPROVED':
        raise ValueError('P1R_D analyzer is inconsistent with approved D1/D2')
    source_paths={p:h for p,h in binding['source_config_sha256'].items()}
    return dict(status='PASS',source_commit=SOURCE,source_config_files_verified=len(source_paths),
        root_git_blobs_verified=len(root_sources),data_files_verified=len(binding['data_sha256']),
        runtime_dependencies=observed,initial_budget=ledger,source_config_sha256=source_paths,
        data_sha256=binding['data_sha256'],analyzer_matches_approved_D1_D2=True)


def final_preflight():
    path=OUT/'evidence/Final_Pair_Preflight_v1.json'
    if path.exists():raise FileExistsError('Final first-pair preflight already exists; no rerun/retry in this task')
    auth=json.loads(AUTH.read_text());payload=dict(status='P1R_D_PREFLIGHT_BLOCKED',arms=[],scientific_attempts=0)
    try:
        if auth['authorization_status']!='CONDITIONAL_PENDING_FINAL_PREFLIGHT':raise ValueError('Independent conditional Owner record required')
        payload['static']=static_checks(auth)
        from tools.tsb_coast_p1r_c_preflight import fixed_specs
        from tools.tsb_coast_p1r_c_production import build_canonical_coast_arm,validate_canonical_production,validate_scientific_credential
        from tools.tsb_coast_p1r_planner import CoastC2PlannerV1,reference_profile
        from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
        from tools.b1_production_preflight import zero_run_guard
        specs=fixed_specs()[:2]
        if [s['run_id'] for s in specs]!=auth['authorized_order']:raise ValueError('P1R_D arm order changed')
        if any(s['session_id']!=auth['session_id'] or s['scenario_token']!=auth['scenario_token'] or canonical_sha256(s)!=auth['approved_spec_sha256'][s['run_id']] for s in specs):
            raise ValueError('P1R_D exact first-pair identity/spec mismatch')
        science_root=Path(auth['scientific_output_root'])
        if science_root.exists():raise FileExistsError('P1R_D scientific output namespace already exists')
        previous={a['run_id']:a for a in json.loads((ROOT/auth['production_preflight_path']).read_text())['arms']}
        held=[];initial=[]
        with zero_run_guard() as calls,patch.object(CoastC2PlannerV1,'initialize',side_effect=RuntimeError('FINAL_PREFLIGHT_ZERO_RUN')),patch.object(CoastC2PlannerV1,'compute_trajectory',side_effect=RuntimeError('FINAL_PREFLIGHT_ZERO_RUN')),patch.object(CoastC2PlannerV1,'compute_planner_trajectory',side_effect=RuntimeError('FINAL_PREFLIGHT_ZERO_RUN')):
            for spec in specs:
                arm=build_canonical_coast_arm(spec,ROOT/'outputs/stageR/tsb_coast_p1r_d_final_preflight_v1'/spec['run_id']);held.append(arm)
                proof=validate_canonical_production(arm);clock=proof['clock']
                if (clock['min_raw_minus_sim_us'],clock['max_raw_minus_sim_us'])!=(-330,323):raise ValueError('P1R_D DEV01 observation skew changed')
                if clock['rows']!=previous[spec['run_id']]['clock']['rows']:raise ValueError('P1R_D original observation mapping changed')
                old_config_path=ROOT/previous[spec['run_id']]['config_path'];old_config=json.loads(old_config_path.read_text())
                current=json.loads((arm.run_root/'resolved_config.json').read_text())
                old_norm=normalized_config(old_config,old_config_path.parent,old_config['experiment_uid'])
                new_norm=normalized_config(current,arm.run_root,current['experiment_uid'])
                if old_norm!=new_norm:raise ValueError('P1R_D production config differs beyond output namespace/start timestamp')
                ego=arm.simulation._scenario.initial_ego_state;d=ego.dynamic_car_state
                init=[ego.time_us,ego.rear_axle.x,ego.rear_axle.y,ego.rear_axle.heading,d.rear_axle_velocity_2d.x,d.rear_axle_velocity_2d.y,d.rear_axle_acceleration_2d.x,d.rear_axle_acceleration_2d.y,ego.tire_steering_angle]
                if not all(math.isfinite(float(v)) for v in init):raise ValueError('P1R_D nonfinite initial state')
                initial.append(init)
                now=dict(time_us=ego.time_us,speed_mps=float(d.speed),rear_axle=dict(x=ego.rear_axle.x,y=ego.rear_axle.y,heading=ego.rear_axle.heading))
                _,distance,_=reference_profile(now['speed_mps'],0,spec['arm'],True)
                route=build_native_route_reference_v1_2(arm.simulation._scenario.map_api,spec['route_roadblock_ids'],now,float(distance[-1])+1e-9)
                payload['arms'].append(dict(run_id=spec['run_id'],status='PASS',spec_sha256=canonical_sha256(spec),
                    precontext_id=arm.precontext_id,route_id=arm.route_id,initial_state=init,
                    clock=clock,callbacks=proof['callback_recorder'],config_sha256=sha256_file(arm.run_root/'resolved_config.json'),
                    config_normalized_sha256=canonical_sha256(new_norm),config_path=str(arm.run_root/'resolved_config.json'),
                    finite_initial_route=dict(required_forward_m=float(distance[-1]),extrapolation_used=route['extrapolation_used']),
                    tracker_type=type(arm.tracker).__name__,model_type=type(arm.motion_model).__name__,
                    component_ids={k:id(getattr(arm,k)) for k in ('runner','planner','controller','tracker','motion_model','recorder')},
                    recorder_rows=arm.recorder.row_count,history_rows=len(arm.simulation.history)))
            if initial[0]!=initial[1] or held[0].precontext_id!=held[1].precontext_id or held[0].route_id!=held[1].route_id:
                raise ValueError('P1R_D paired initial/precontext/route mismatch')
            for k in ('runner','planner','controller','tracker','motion_model','recorder'):
                if id(getattr(held[0],k))==id(getattr(held[1],k)):raise ValueError('P1R_D reused arm object: '+k)
            if any(calls.values()):raise ValueError('P1R_D forbidden scientific call during final preflight')
            payload['zero_run_guard_calls']=calls
        candidate={**auth,'authorization_status':'AUTHORIZED','activation':'FINAL_PAIR_PREFLIGHT_PASS'}
        for spec in specs:validate_scientific_credential(candidate,spec,ROOT/auth['approved_amendment_path'],ROOT/auth['execution_source_binding_path'],ROOT/auth['production_preflight_path'])
        denied=[]
        for spec in fixed_specs()[2::2]:
            try:validate_scientific_credential(candidate,spec,ROOT/auth['approved_amendment_path'],ROOT/auth['execution_source_binding_path'],ROOT/auth['production_preflight_path'])
            except PermissionError:denied.append(spec['run_id'])
            else:raise ValueError('P1R_D DEV02/03 credential was accepted')
        payload.update(status='PASS',denied_other_pair_runs=denied,paired_initial_state_identical=True,full_scientific_config_unchanged=True,
            allowed_config_normalization=['arm output namespace','run-start experiment_uid timestamp'],guard_objects_discarded_not_for_execution=True)
        write_new(path,payload)
        candidate.update(final_pair_preflight_sha256=sha256_file(path),final_preflight_config_sha256={a['run_id']:a['config_sha256'] for a in payload['arms']},
            audit_helper_sha256=sha256_file(Path(__file__)),audit_helper_role='FINAL_ZERO_RUN_PREFLIGHT_ONLY_NOT_PRODUCTION_MODIFICATION')
        AUTH.write_bytes(encoded(candidate))
        return payload
    except Exception as exc:
        payload.update(status='P1R_D_PREFLIGHT_BLOCKED',error_type=type(exc).__name__,error=str(exc),stop_required=True)
        if not path.exists():write_new(path,payload)
        return payload


if __name__=='__main__':
    result=final_preflight();print(json.dumps({k:result[k] for k in ('status','scientific_attempts')},ensure_ascii=False))
    raise SystemExit(0 if result['status']=='PASS' else 2)
