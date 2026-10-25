#!/usr/bin/env python3
"""P2-Q0 prospective metadata selection and guarded factory preparation only.

No scientific launch, claim, model propagation or original outcome analysis.
F2 scientific modules are imported unchanged and used only under zero_run_guard.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import gzip
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT,sha256_file,canonical_sha256,precontext_id,route_id

OUT=ROOT/'docs/stageR/tsb_direct_coast_p2_q0'
RULE=OUT/'P2_Q0_Prospective_Selection_Rule_v1.json'
STATIC=ROOT/'docs/stageR/s2r_engineering_closure/S2R_Static_Eligibility_Census_v1.csv'
ROLES=ROOT/'docs/stageR/s2r_role_aware_preflight/S2R_V_Eligibility_Census_v1.csv'
UNIVERSE=ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json'
PREFLIGHT_ROOT=ROOT/'outputs/stageR/tsb_direct_coast_p2_q0_zero_run_v1'


def read_csv(path):return list(csv.DictReader(StringIO(Path(path).read_text())))


def write_new(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows,fields=None):
    if fields is None:fields=list(rows[0])
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def ranked_frame(rows,roles,excluded,protected_C,protected_C_fingerprints,logs,salt):
    """Static metadata only; no result fields are accepted or consulted."""
    by_log={x['log_id']:x for x in logs};frame=[]
    if len(rows)!=len({r['session_id'] for r in rows}):raise ValueError('Duplicate SESSION rows in static frame')
    for row in rows:
        sid=row['session_id'];role=roles.get(sid);reasons=[]
        if role is None or role['exposure_class'] not in {'E0_METADATA_ONLY','E1_UNRELATED_HISTORICAL_USE','E2_BENCHMARK_ENGINEERING','E3_PRIMARY_METHOD_DEVELOPMENT','E4_PRIMARY_OUTCOME_DRIVEN_ADAPTATION','E5_UNTOUCHED_CONFIRMATORY'}:reasons.append('EXPOSURE_UNKNOWN')
        elif role['exposure_class'].startswith(('E3_','E4_')):reasons.append('EXCLUDED_E3_E4')
        if sid in excluded:reasons.append('OLD_B1_OR_COAST_DEV_EXCLUDED')
        if sid in protected_C:reasons.append('PROTECTED_E5_C_CANDIDATE')
        if row['static_eligibility_status']!='STATIC_ELIGIBLE':reasons.append('NOT_STATIC_ELIGIBLE')
        if not row.get('execution_initial_speed_mps') or float(row['execution_initial_speed_mps'])<5.:reasons.append('INITIAL_SPEED_LT_5_OR_MISSING')
        log=by_log.get(row['representative_log_id'])
        if not log or log['session_id']!=sid or not log['frozen_fingerprint_match'] or not log['source_stat_unchanged_during_read']:reasons.append('LOG_SOURCE_IDENTITY_INCOMPLETE')
        elif log['observed_db_fingerprint_sha256'] in protected_C_fingerprints:reasons.append('PROTECTED_C_SOURCE_ALIAS')
        if role and (role['source_role_provenance_status']!='COMPLETE' or role['source_fingerprint_complete']!='true'):reasons.append('ROLE_PROVENANCE_INCOMPLETE')
        if role and role['role_reservation_status'] not in {'CANDIDATE_POOL_ONLY_NO_FINAL_V_OR_C_RESERVATION','DEVELOPMENT_ONLY_EXCLUDED_FROM_PRIMARY_V'}:reasons.append('ROLE_RESERVATION_UNKNOWN')
        digest=hashlib.sha256((salt+'|'+sid).encode()).hexdigest()
        frame.append(dict(session_id=sid,rank_sha256=digest,exposure_class=role['exposure_class'] if role else 'UNKNOWN',
            log_id=row['representative_log_id'],scenario_token=row['representative_scenario_token'],initial_speed_mps=row['execution_initial_speed_mps'],
            static_status=row['static_eligibility_status'],eligible=not reasons,reasons=';'.join(sorted(set(reasons)))))
    eligible=sorted([x for x in frame if x['eligible']],key=lambda x:(x['rank_sha256'],x['session_id']))
    return frame,eligible


def reject_q0_execution(credential,run_id):
    if credential.get('stage')!='TSB_DIRECT_COAST_P2' or credential.get('authorization_status')!='AUTHORIZED':
        raise PermissionError('P2-Q0 rejects old B1/C2/F2 and PROPOSED_ONLY credentials; no scientific execution authorized')
    raise PermissionError('P2-Q0 is preparation only; independent final Owner authorization and exact execution binding required')


def select():
    roster_path=OUT/'P2_Q0_Independent_Session_Roster_v1.csv'
    if roster_path.exists():raise FileExistsError('P2 frozen roster exists; no reselection/replacement')
    rule=json.loads(RULE.read_text())
    if sha256_file(STATIC)!=rule['static_census_sha256']:raise ValueError('Prospectively bound static frame changed')
    roles={r['session_id']:r for r in read_csv(ROLES)};rows=read_csv(STATIC);universe=json.loads(UNIVERSE.read_text());logs=universe['logs']
    if len(roles)!=248 or len(rows)!=248 or len(logs)!=1564:raise ValueError('S1R frozen SESSION/log universe changed')
    b1={r['session_id'] for r in read_csv(ROOT/'docs/stageR/b1_tsb_qualification/B1_TSB_Qualification_Roster_v1.csv')}
    dev={r['session_id'] for r in read_csv(ROOT/'docs/stageR/tsb_coast_p1r_entry/P1R_Development_Candidate_Roster_v1.csv')}
    if len(b1)!=20 or len(dev)!=3:raise ValueError('Original B1/development exclusion identities changed')
    for event in read_csv(ROOT/'docs/stageR/b1_resume_execution_r2/B1_R2_Exposure_Ledger_Update_v1.csv'):
        sid=event['session_id']
        if not roles[sid]['exposure_class'].startswith(('E3_','E4_')):roles[sid]['exposure_class']='E2_BENCHMARK_ENGINEERING'
    for sid in dev:
        if not roles[sid]['exposure_class'].startswith(('E3_','E4_')):roles[sid]['exposure_class']='E2_BENCHMARK_ENGINEERING'
    protected_C={sid for sid,r in roles.items() if r['exposure_class']=='E5_UNTOUCHED_CONFIRMATORY'}
    C_fps={log['observed_db_fingerprint_sha256'] for log in logs if log['session_id'] in protected_C}
    frame,eligible=ranked_frame(rows,roles,b1|dev,protected_C,C_fps,logs,rule['session_rank_salt'])
    selected=eligible[:20];by_static={x['session_id']:x for x in rows};by_log={x['log_id']:x for x in logs}
    if len(selected)==20:
        fingerprints=[by_log[x['log_id']]['observed_db_fingerprint_sha256'] for x in selected]
        tokens=[by_log[x['log_id']]['database_log_token'] for x in selected]
        if len(set(fingerprints))!=20 or len(set(tokens))!=20 or len({x['log_id'] for x in selected})!=20:raise ValueError('Frozen top20 raw LOG independence fails; no replacements permitted')
    ranking=[dict(eligible_rank=i+1,**row,selected=i<20) for i,row in enumerate(eligible)]
    write_csv(OUT/'evidence/P2_Q0_Full_248Session_Eligibility_Audit_v1.csv',frame)
    write_csv(OUT/'evidence/P2_Q0_Full_Eligible_Hash_Ranking_v1.csv',ranking)
    roster=[]
    for number,row in enumerate(selected,1):
        log=by_log[row['log_id']];static=by_static[row['session_id']];pair=f'DIRECTCOAST-P2-{number:02d}'
        roster.append(dict(pair_id=pair,selection_rank=number,session_id=row['session_id'],log_id=row['log_id'],scenario_token=row['scenario_token'],
            database_log_token=log['database_log_token'],db_path=log['source_file'],source_database_sha256=log['observed_db_fingerprint_sha256'],
            metadata_shard=log['shard'],map_name=log['map_version'],initial_speed_mps=row['initial_speed_mps'],precontext_id=static['precontext_id'],route_id=static['native_route_identity'],
            rank_sha256=row['rank_sha256'],exposure_class_before_P2=row['exposure_class'],current_role='B_BENCHMARK_QUALIFICATION_PREPARATION',
            zero_run_exposure='E0_METADATA_EVENT_RETAIN_PRIOR_RISK_CLASS_NOT_E5',future_if_executed='E2_BENCHMARK_QUALIFICATION_EXPOSURE_NOT_PRIMARY_COMPARISON',
            independent_from_old_B1=True,independent_from_all_Coast_development=True,primary_C_candidate=False,
            baseline_run_id=pair+'-BASELINE',treatment_run_id=pair+'-TREATMENT',arm_order='BASELINE_THEN_TREATMENT',execution_status='NOT_RUN_ROSTER_FROZEN_BEFORE_PREFLIGHT'))
    if roster:write_csv(roster_path,roster)
    else:write_csv(roster_path,[],['session_id','execution_status'])
    capacity=dict(status='CAPACITY_SUFFICIENT_FROZEN_20_PENDING_PREFLIGHT' if len(selected)==20 else 'P2_Q0_CAPACITY_BLOCKED',
        total_SESSION=248,static_eligible_before_new_domain=sum(r['static_eligibility_status']=='STATIC_ELIGIBLE' for r in rows),eligible_count=len(eligible),selected_count=len(selected),
        all_old_B1_sessions=sorted(b1),all_Coast_DEV_sessions=sorted(dev),protected_C_E5_sessions=sorted(protected_C),
        operative_exposure_counts=dict(Counter(x['exposure_class'] for x in roles.values())),selected_exposure_counts=dict(Counter(x['exposure_class'] for x in selected)),
        original_duplicate_aliases=universe['duplicate_source_aliases'],roster_sha256=sha256_file(roster_path),selection_rule_sha256=sha256_file(RULE),
        protected_C_policy='ALL_5_E5_SESSION_AND_THEIR_RAW_LOG_FINGERPRINTS_PROTECTED; NO_NEW_C_ROSTER_CREATED',
        explicit_current_C_reservation_evidence='S2R role manifest has no final V/C roster; all E5 candidate sources conservatively excluded',scientific_outcomes_read=False)
    write_new(OUT/'evidence/P2_Q0_Capacity_And_Selection_v1.json',capacity)
    print('Frozen eligible sessions:',len(eligible),'; selected:',len(selected),flush=True)
    return capacity


def metadata():
    from tools.r1_prepare_official_technical_smoke_roster import _official_initial
    from tools.s2r_engineering_closure import precontext_payload
    dest=OUT/'evidence/P2_Q0_Arm_Specs_v1.json'
    if dest.exists():raise FileExistsError('P2 metadata evidence already exists; no replacement')
    cap=json.loads((OUT/'evidence/P2_Q0_Capacity_And_Selection_v1.json').read_text());roster=read_csv(OUT/'P2_Q0_Independent_Session_Roster_v1.csv')
    if cap['selected_count']!=20:raise PermissionError('P2 capacity blocked; do not construct a reduced roster')
    if sha256_file(OUT/'P2_Q0_Independent_Session_Roster_v1.csv')!=cap['roster_sha256']:raise ValueError('P2 frozen roster modified')
    specs=[];audit=[];sources={};metadata_cache={};maps=ROOT.parent/'nuplan/dataset/maps';map_metadata=maps/'nuplan-maps-v1.0.json';versions=json.loads(map_metadata.read_text());sources[str(map_metadata)]=sha256_file(map_metadata)
    for number,row in enumerate(roster,1):
        try:
            db=Path(row['db_path']);digest=sha256_file(db)
            if digest!=row['source_database_sha256']:raise ValueError('Current full database differs from frozen raw source SHA')
            shard=ROOT/'docs/stageR/s2_preflight'/row['metadata_shard'];sources[str(shard.relative_to(ROOT))]=sha256_file(shard)
            with gzip.open(shard,'rt') as f:data=json.load(f)
            cols=data['record_columns'];idx=cols.index('scenario_token');records=[x for x in data['records'] if x[idx]==row['scenario_token']]
            if len(records)!=1:raise ValueError('Scenario token not unique in frozen canonical raw LOG shard')
            anchor=int(records[0][cols.index('tag_anchor_timestamp_us')])
            initial,route=_official_initial(dict(db_path=row['db_path'],scenario_token=row['scenario_token'],timestamp=anchor))
            if initial['initial_speed_mps']<5.:raise ValueError('Frozen representative initial speed below approved 5m/s')
            candidate=dict(session_id=row['session_id'],log_id=row['log_id'],scenario_token=row['scenario_token'],db_path=row['db_path'],map_name=row['map_name'],source_fingerprint_sha256=digest)
            rid=route_id(row['map_name'],route);pre=precontext_payload(candidate,initial,route,rid);pid=precontext_id(pre)
            if rid!=row['route_id'] or pid!=row['precontext_id']:raise ValueError('Frozen route/precontext metadata changed')
            common=dict(**candidate,candidate='TSB_DIRECT_COAST_F2',pair_id=row['pair_id'],role='B',exposure_class=row['exposure_class_before_P2'],
                route_roadblock_ids=route,route_id=rid,precontext=pre,precontext_id=pid,start_timestamp_us=initial['initial_time_us'])
            specs.extend(dict(**copy.deepcopy(common),arm=arm,run_id=row['baseline_run_id'] if arm=='BASELINE' else row['treatment_run_id']) for arm in ['BASELINE','TREATMENT'])
            sources[row['db_path']]=digest
            map_file=maps/row['map_name']/versions[row['map_name']]['version']/'map.gpkg'
            if str(map_file) not in sources:sources[str(map_file)]=sha256_file(map_file)
            audit.append(dict(pair_id=row['pair_id'],status='PASS',initial=initial,precontext_id=pid,route_id=rid))
        except Exception as error:
            audit.append(dict(pair_id=row['pair_id'],status='BLOCKED',error=type(error).__name__+': '+str(error)))
        print('P2 metadata',number,'/20',audit[-1]['status'],row['session_id'],flush=True)
    result=dict(status='PASS' if len(specs)==40 and all(x['status']=='PASS' for x in audit) else 'P2_Q0_BLOCKED_METADATA',arms=specs,pair_audits=audit,
        frozen_roster_sha256=cap['roster_sha256'],source_data_sha256=sources,scientific_attempts=0)
    write_new(dest,result);return result


def preflight():
    from tools.b1_production_preflight import zero_run_guard
    from tools.tsb_direct_coast_f2_production import build_f2_arm,validate_f2_production,DirectCoastF2PlannerV1
    from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
    from tools.tsb_coast_p1r_planner import reference_profile
    destination=OUT/'evidence/P2_Q0_40Arm_Production_Preflight_v1.json'
    if destination.exists():raise FileExistsError('P2 one-shot preflight evidence exists; no rerun/replacement')
    source=json.loads((OUT/'evidence/P2_Q0_Arm_Specs_v1.json').read_text());roster=read_csv(OUT/'P2_Q0_Independent_Session_Roster_v1.csv')
    specs={s['run_id']:s for s in source['arms']};results=[];held=[]
    if len(roster)!=20:raise PermissionError('P2 roster is not exactly20')
    original=json.loads((ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Execution_Source_Binding_v1.json').read_text())
    for name,digest in original['source_config_sha256'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('Frozen F2 scientific source/config changed: '+name)
    if sha256_file(OUT/'P2_Q0_Independent_Session_Roster_v1.csv')!=source['frozen_roster_sha256']:raise ValueError('P2 roster changed before preflight')
    with zero_run_guard() as calls,patch.object(DirectCoastF2PlannerV1,'initialize',side_effect=RuntimeError('P2_Q0_ZERO_RUN_ONLY')),patch.object(DirectCoastF2PlannerV1,'compute_trajectory',side_effect=RuntimeError('P2_Q0_ZERO_RUN_ONLY')):
        for number,row in enumerate(roster,1):
            pair_arms=[]
            for arm in ['BASELINE','TREATMENT']:
                run_id=row['baseline_run_id'] if arm=='BASELINE' else row['treatment_run_id']
                item=dict(pair_id=row['pair_id'],session_id=row['session_id'],arm=arm,run_id=run_id,status='BLOCKED',failure_code='',scientific_execution='NOT_RUN',history_rows=0,recorder_rows=0)
                try:
                    if run_id not in specs:raise ValueError('Frozen metadata for this arm unavailable; no replacement')
                    spec=specs[run_id];built=build_f2_arm(spec,PREFLIGHT_ROOT/run_id);held.append(built);pair_arms.append(built)
                    proof=validate_f2_production(built);ego=built.simulation._scenario.get_ego_state_at_iteration(0);current=built.planner._payload(ego)
                    if ego.dynamic_car_state.speed<5.:raise ValueError('Full precision actual initial speed below5')
                    _,distance,_=reference_profile(current['speed_mps'],0,arm,True)
                    route=build_native_route_reference_v1_2(built.simulation._scenario.map_api,spec['route_roadblock_ids'],current,float(distance[-1])+1e-9)
                    item.update(status='PASS',clock=proof['clock'],callbacks=proof['callbacks'],initial_state=current,precontext_id=spec['precontext_id'],route_id=spec['route_id'],
                        finite_initial_route_query=dict(required_forward_m=float(distance[-1]),extrapolation_used=route['extrapolation_used']),
                        component_ids={name:id(getattr(built,name)) for name in ['runner','planner','controller','tracker','motion_model','recorder']},
                        config_path=str(built.run_root/'resolved_config.json'),config_sha256=sha256_file(built.run_root/'resolved_config.json'))
                except Exception as error:item.update(status='BLOCKED',failure_code=type(error).__name__+': '+str(error))
                results.append(item)
            if len(pair_arms)==2 and results[-2]['status']=='PASS' and results[-1]['status']=='PASS':
                if results[-2]['initial_state']!=results[-1]['initial_state'] or results[-2]['clock']['rows']!=results[-1]['clock']['rows']:
                    for item in results[-2:]:item.update(status='BLOCKED',failure_code='PAIR_INITIAL_STATE_OR_OBSERVATION_MAPPING_MISMATCH')
            print('P2 production preflight',number,'/20',[(x['arm'],x['status']) for x in results[-2:]],flush=True)
        if any(calls.values()):raise RuntimeError('P2 zero-run guard violation')
        for name in ['runner','planner','controller','tracker','motion_model','recorder']:
            ids=[id(getattr(a,name)) for a in held]
            if len(ids)!=len(set(ids)):raise RuntimeError('P2 cross-arm object reuse')
    rows=[]
    for item in results:
        clock=item.get('clock',{});rows.append(dict(pair_id=item['pair_id'],session_id=item['session_id'],arm=item['arm'],run_id=item['run_id'],status=item['status'],failure_code=item['failure_code'],
            clock_status=clock.get('status','NOT_VERIFIED'),min_raw_minus_sim_us=clock.get('min_raw_minus_sim_us',''),max_raw_minus_sim_us=clock.get('max_raw_minus_sim_us',''),
            scientific_execution='NOT_RUN',history_rows=item['history_rows'],recorder_rows=item['recorder_rows'],config_sha256=item.get('config_sha256','')))
    write_csv(OUT/'P2_Q0_40Arm_Production_Preflight_v1.csv',rows)
    summary=dict(status='PASS' if len(results)==40 and all(x['status']=='PASS' for x in results) else 'P2_Q0_PRODUCTION_PREFLIGHT_BLOCKED',arms=results,
        pass_arms=sum(x['status']=='PASS' for x in results),blocked_arms=sum(x['status']!='PASS' for x in results),zero_run_guard_calls=calls,scientific_attempts=0,
        original_roster_preserved=True,replacements=0,route_scope='INITIAL_FINITE_QUERY_ONLY_NOT_FULL_FUTURE_EPISODE_PROOF')
    write_new(destination,summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['select','metadata','preflight']);args=parser.parse_args()
    result={'select':select,'metadata':metadata,'preflight':preflight}[args.mode]();print(json.dumps({k:v for k,v in result.items() if k not in ['arms','pair_audits','source_data_sha256','zero_run_guard_calls']},indent=2,ensure_ascii=False))
