#!/usr/bin/env python3
"""P1R one-shot metadata roster and uninitialized production audit; never run."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT,write_manifest,precontext_id,route_id,sha256_file

OUT=ROOT/'docs/stageR/tsb_coast_p1r_entry'
SALT='TSB_COAST_P1R_BDEV_C2_V1'


def select_metadata():
    from tools.r1_prepare_official_technical_smoke_roster import _official_initial
    from tools.s2r_engineering_closure import precontext_payload
    from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
    from tools.tsb_coast_p1r_planner import reference_profile
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    census=list(csv.DictReader((ROOT/'docs/stageR/s2r_engineering_closure/S2R_Static_Eligibility_Census_v1.csv').open()))
    # Already-exposed B development sessions only; no success/outcome fields.
    eligible=[x for x in census if x['static_eligibility_status']=='STATIC_ELIGIBLE' and float(x['execution_initial_speed_mps'])>=5 and x['exposure_class']=='E2_BENCHMARK_ENGINEERING']
    def rank(x):return hashlib.sha256(f"{SALT}|{x['session_id']}|{x['representative_scenario_token']}|{x['representative_log_id']}".encode()).hexdigest()
    selected=sorted(eligible,key=lambda x:(rank(x),x['session_id']))[:3]
    if len(selected)!=3 or len({x['session_id'] for x in selected})!=3:raise ValueError('Insufficient three independent metadata candidates')
    old=json.loads((ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json').read_text())
    specs=[];rows=[];evidence=[];sources={}
    for number,row in enumerate(selected,1):
        log=next(x for x in old['logs'] if x['log_id']==row['representative_log_id'])
        shard=ROOT/'docs/stageR/s2_preflight'/log['shard']
        with gzip.open(shard,'rt') as f:data=json.load(f)
        columns=data['record_columns'];token_idx=columns.index('scenario_token')
        record=next(x for x in data['records'] if x[token_idx]==row['representative_scenario_token'])
        anchor=int(record[columns.index('tag_anchor_timestamp_us')])
        initial,route=_official_initial({'db_path':log['source_file'],'scenario_token':row['representative_scenario_token'],'timestamp':anchor})
        if initial['initial_speed_mps']<5:raise ValueError('Selected identity no longer meets metadata domain; no replacement')
        candidate={'session_id':row['session_id'],'log_id':log['log_id'],'scenario_token':row['representative_scenario_token'],
            'db_path':log['source_file'],'map_name':log['map_version'],'source_fingerprint_sha256':log['observed_db_fingerprint_sha256']}
        rid=route_id(candidate['map_name'],route);pre=precontext_payload(candidate,initial,route,rid)
        common={**candidate,'pair_id':f'P1R-C2-DEV-{number:02d}','candidate':'TSB_COAST_C2_P1R','role':'B','exposure_class':row['exposure_class'],
            'route_roadblock_ids':route,'route_id':rid,'precontext':pre,'precontext_id':precontext_id(pre),'start_timestamp_us':initial['initial_time_us']}
        api=get_maps_api(str(ROOT.parent/'nuplan/dataset/maps'),'nuplan-maps-v1.0',candidate['map_name'])
        current={'rear_axle':{'x':initial['initial_x'],'y':initial['initial_y'],'heading':initial['initial_heading']},'speed_mps':initial['initial_speed_mps'],'time_us':initial['initial_time_us']}
        queries=[]
        for arm in ('BASELINE','TREATMENT'):
            _,distance,_=reference_profile(initial['initial_speed_mps'],0,arm,True)
            native=build_native_route_reference_v1_2(api,route,current,float(distance[-1])+1e-9)
            specs.append({**common,'arm':arm,'run_id':common['pair_id']+'-'+arm})
            queries.append({'arm':arm,'required_forward_m':float(distance[-1]),'builder_called':'build_native_route_reference_v1_2','returned_inner_builder_label':native.get('builder_version'),'native_edge_ids':native['native_edge_ids'],'extrapolation_used':native['extrapolation_used']})
        rows.append({'pair_id':common['pair_id'],'session_id':row['session_id'],'log_id':log['log_id'],'scenario_token':candidate['scenario_token'],
            'initial_speed_mps':initial['initial_speed_mps'],'precontext_id':common['precontext_id'],'route_id':rid,'rank_sha256':rank(row),
            'exposure_class':row['exposure_class'],'status':'CANDIDATE_PENDING_OWNER_CONTRACT_FREEZE'})
        evidence.append({'pair_id':common['pair_id'],'queries':queries,'scope':'INITIAL_FINITE_QUERY_NOT_FULL_FUTURE_EPISODE_PROOF'})
        sources[str(shard.relative_to(ROOT))]=sha256_file(shard)
        sources[log['source_file']]=sha256_file(Path(log['source_file']))
    with (OUT/'P1R_Development_Candidate_Roster_v1.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_manifest(OUT/'evidence/Arm_Specs_v1.json',{'status':'PENDING_OWNER_FREEZE_NOT_EXECUTABLE','arms':specs})
    write_manifest(OUT/'evidence/Metadata_Selection_v1.json',{'salt':SALT,'eligible_E2_count':len(eligible),'selection_rule':'rank current static representatives, E2 and initial speed>=5; first three; no fallback replacement',
        'rows':rows,'finite_route_queries':evidence,'metadata_sources_sha256':sources,'scientific_outcomes_used':False})
    return specs


def production_preflight(specs, namespace="tsb_coast_p1r_zero_run_v1", evidence_name="Production_Preflight_v1.json"):
    from tools import b1_production_preflight as guard
    from tools.tsb_coast_p1r_production import build_coast_arm,validate_callbacks,audit_episode_clock
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    held=[];proof=[]
    raw=ROOT/'outputs/stageR'/namespace
    with guard.zero_run_guard() as calls,patch.object(CoastC2PlannerV1,'initialize',side_effect=RuntimeError('P1R_ZERO_RUN_PLANNER_INITIALIZE')),patch.object(CoastC2PlannerV1,'compute_planner_trajectory',side_effect=RuntimeError('P1R_ZERO_RUN_PLANNER_COMPUTE')):
        for spec in specs:
            arm=build_coast_arm(spec,raw/spec['run_id']);held.append(arm)
            proof.append({'run_id':spec['run_id'],'callback_recorder':validate_callbacks(arm),'episode_clock':audit_episode_clock(arm),'config_hash':arm.config_hash,
                'precontext_id':arm.precontext_id,'route_id':arm.route_id,'tracker_type':type(arm.tracker).__name__,
                'controller_type':type(arm.controller).__name__,'motion_model_type':type(arm.motion_model).__name__,
                'component_ids':{k:id(getattr(arm,k)) for k in ('runner','planner','controller','tracker','motion_model','recorder')},
                'config_path':str(arm.run_root/'resolved_config.json')})
        if any(calls.values()):raise ValueError('Scientific zero-run guard violation')
        for key in ('runner','planner','controller','tracker','motion_model','recorder'):
            if len({id(getattr(a,key)) for a in held})!=6:raise ValueError('Cross-arm object reuse')
    result={'status':'PASS' if all(x['episode_clock']['status']=='PASS' for x in proof) else 'BLOCKED_PHYSICAL_CLOCK_CONTRACT','scope':'SIX_REAL_UNINITIALIZED_OFFICIAL_RUNNERS_NO_SCIENTIFIC_EXECUTION','arms':proof,'zero_run_guard_calls':calls,
        'fresh_component_sets':6,'formal_scientific_attempts':0,'runtime_metric_finalization':'NOT_RUN'}
    write_manifest(OUT/'evidence'/evidence_name,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if not args.prepare:raise SystemExit('--prepare performs metadata/factory checks only; no execution option')
    if (OUT/'evidence/Arm_Specs_v1.json').exists():raise FileExistsError('One-shot roster preparation already exists')
    specs=select_metadata();production_preflight(specs)
    print('P1R metadata/real factory preflight complete; scientific attempts=0')
