#!/usr/bin/env python3
"""Fixed three-pair metadata/canonical clock preflight, never scientific execution."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT, write_manifest, sha256_file

OUT = ROOT/'docs/stageR/tsb_coast_p1r_clock_closure'
ORIGINAL = ROOT/'docs/stageR/tsb_coast_p1r_entry'


def fixed_specs():
    original = json.loads((ORIGINAL/'evidence/Arm_Specs_v1.json').read_text())['arms']
    expected = [f'P1R-C2-DEV-{i:02d}-{a}' for i in (1,2,3) for a in ('BASELINE','TREATMENT')]
    if [s['run_id'] for s in original] != expected or len({s['session_id'] for s in original}) != 3:
        raise ValueError('P1R_C fixed original roster is missing or altered')
    return original


def preflight(run_root, evidence_dir):
    from tools.b1_production_preflight import zero_run_guard
    from tools.tsb_coast_p1r_c_production import build_canonical_coast_arm, validate_canonical_production
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
    from tools.tsb_coast_p1r_planner import reference_profile
    specs = fixed_specs(); held = []; proofs = []
    binding=json.loads((ORIGINAL/'evidence/Final_Binding_v1.json').read_text())
    sources={**binding['metadata_data_sha256'],**binding['map_data_sha256']}
    if any(sha256_file(ROOT/p)!=h for p,h in sources.items()):
        raise ValueError('P1R_C original database/map/shard hash changed')
    historical={p['run_id']:p['episode_clock']['scenario_timepoints_us'] for p in json.loads((ORIGINAL/'evidence/Production_Clock_Preflight_v1.json').read_text())['arms']}
    evidence_dir = Path(evidence_dir)
    if (evidence_dir/'ThreePair_Preflight_v1.json').exists():
        raise FileExistsError('P1R_C evidence already exists; do not overwrite a completed audit')
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with zero_run_guard() as calls, \
         patch.object(CoastC2PlannerV1, 'initialize', side_effect=RuntimeError('ZERO_RUN_ONLY')), \
         patch.object(CoastC2PlannerV1, 'compute_trajectory', side_effect=RuntimeError('ZERO_RUN_ONLY')), \
         patch.object(CoastC2PlannerV1, 'compute_planner_trajectory', side_effect=RuntimeError('ZERO_RUN_ONLY')):
        for spec in specs:
            arm = build_canonical_coast_arm(spec, Path(run_root)/spec['run_id']); held.append(arm)
            result = validate_canonical_production(arm)
            if [r['raw_log_time_us'] for r in result['clock']['rows']] != historical[spec['run_id']]:
                raise ValueError('P1R_C original log timestamp mapping changed')
            result['clock']['historical_P1R_raw_timestamps_exactly_preserved']=True
            clock = arm.simulation._time_controller
            # Exercise only time-controller index arithmetic, not observations,
            # Simulation, controller.update_state or model propagation.
            iterations = []
            while not clock.reached_end():
                current = clock.get_iteration(); iterations.append([current.index,current.time_us])
                clock.next_iteration()
            if len(iterations) != 80 or [r[0] for r in iterations] != list(range(80)):
                raise ValueError('P1R_C official end semantics did not yield 80 planner indices')
            clock.reset()
            if clock.get_iteration().time_us != spec['start_timestamp_us']:
                raise ValueError('P1R_C reset changed original anchor')
            initial = spec['precontext']['ego_initial_state']
            current = dict(time_us=spec['start_timestamp_us'], speed_mps=initial['initial_speed_mps'],
                rear_axle=dict(x=initial['initial_x'],y=initial['initial_y'],heading=initial['initial_heading']))
            _,distance,_ = reference_profile(current['speed_mps'],0,spec['arm'],True)
            route = build_native_route_reference_v1_2(arm.simulation._scenario.map_api,
                spec['route_roadblock_ids'],current,float(distance[-1])+1e-9)
            proofs.append(dict(run_id=spec['run_id'], pair_id=spec['pair_id'], session_id=spec['session_id'],
                scenario_token=spec['scenario_token'], precontext_id=arm.precontext_id, route_id=arm.route_id,
                status='PASS',clock=result['clock'],callback_recorder=result['callback_recorder'],
                official_80_iteration_interface=iterations,config_hash=arm.config_hash,
                config_path=str(arm.run_root/'resolved_config.json'),
                finite_initial_route=dict(builder_called='build_native_route_reference_v1_2',
                    required_forward_m=float(distance[-1]),extrapolation_used=route['extrapolation_used']),
                component_ids={k:id(getattr(arm,k)) for k in ('runner','planner','controller','tracker','motion_model','recorder')},
                time_controller_id=id(clock), history_rows=0, recorder_rows=arm.recorder.row_count))
        for key in ('runner','planner','controller','tracker','motion_model','recorder'):
            if len({id(getattr(a,key)) for a in held}) != 6:
                raise ValueError('P1R_C cross-arm component reuse')
        for i in (0,2,4):
            left,right=proofs[i:i+2]
            if left['clock']['rows'] != right['clock']['rows'] or left['precontext_id'] != right['precontext_id'] or left['route_id'] != right['route_id']:
                raise ValueError('P1R_C paired time/precontext/route differs')
        if any(calls.values()):
            raise ValueError('P1R_C scientific zero-run guard violation')
    payload = dict(status='PASS',scope='SIX_REAL_OFFICIAL_OBJECT_GRAPHS_WITHOUT_SIMULATION_EXECUTION',
        fixed_spec_source=str(ORIGINAL/'evidence/Arm_Specs_v1.json'),
        fixed_spec_source_sha256=sha256_file(ORIGINAL/'evidence/Arm_Specs_v1.json'),
        arms=proofs,zero_run_guard_calls=calls,scientific_attempts=0,
        original_database_map_and_shard_sha256=sources,
        actual_runtime_metric_finalization='NOT_EXECUTED',complete_episode_route='NOT_ESTABLISHED')
    write_manifest(evidence_dir/'ThreePair_Preflight_v1.json',payload)
    with (evidence_dir/'ThreePair_Preflight_Rows_v1.csv').open('x',newline='') as f:
        fields=['run_id','status','clock_status','min_raw_minus_sim_us','max_raw_minus_sim_us',
                'max_abs_raw_minus_sim_us','positive_skew_count','negative_skew_count','config_hash','precontext_id','route_id']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for p in proofs:
            row={k:p[k] for k in ('run_id','status','config_hash','precontext_id','route_id')}
            row['clock_status']=p['clock']['status']
            row.update({k:p['clock'][k] for k in fields if k in p['clock']})
            writer.writerow(row)
    return payload


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True,help='Fresh local engineering namespace, never scientific outputs')
    parser.add_argument('--evidence-dir',type=Path,required=True,help='Fresh evidence directory')
    args=parser.parse_args();result=preflight(args.output_root,args.evidence_dir)
    print('P1R_C canonical preflight:',result['status'],'; scientific attempts=0')
