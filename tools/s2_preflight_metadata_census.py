#!/usr/bin/env python3
"""Read-only frozen-source census. No simulator imports, factories or execution API.

The normalized JSON gzip shards are part of the denominator, not a Q roster.
UNKNOWN is a blocking result, never an eligibility pass or a scientific failure.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import subprocess
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / 'docs/stageR/s1'
SOURCE = ROOT / 'docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json'
INVENTORY = ROOT / 'docs/stageR/r1/r1_official_nuplan_db_inventory_rows_v0.1.json'
OUT = ROOT / 'docs/stageR/s2_preflight'
GLOBAL = ROOT / 'docs/stageR/r0/manifests/r0_nuplan_global_identity_ledger_v0.1.csv'
HISTORY = ROOT / 'docs/stageR/r0/manifests/r0_nuplan_historical_use_ledger_v0.1.csv'
PERMANENT = ROOT / 'docs/stageR/r2/r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0.json'
ENGINEERING = ROOT / 'docs/stageR/r2/r2_bj_b0_permanent_engineering_exclusion_ledger_v1.0.json'
A5 = ROOT / 'docs/stageR/r2/r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json'
UNKNOWN = ['NATIVE_ROLLING_REFERENCE_NOT_CERTIFIED', 'FULL_ARM_RESET_NOT_BOUND', 'U_Q_D_E_ROLE_CLOSURE_NOT_COMPLETE']
COLUMNS = ['scenario_token','tag_anchor_speed_mps','tag_anchor_timestamp_us','tag_anchor_scene_token','task_types','raw_scene_roadblock_ids','forward_native_record_count_from_anchor','anchor_timestamp_80_state_support','anchor_pose_available','metadata_flags']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def canonical(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def token_sha(tokens): return hashlib.sha256(''.join(t+'\n' for t in sorted(set(tokens))).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def session(log):
    match=re.fullmatch(r'(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d+_\d+',log)
    return match.group(1) if match else None

def speed_gate(speed): return isinstance(speed,(int,float)) and not isinstance(speed,bool) and math.isfinite(speed) and speed>=3.61

def verify_s1():
    m=read(S1/'S1_Protocol_Design_Manifest_v0.1.json')
    for p,h in {**m['evidence_bindings'],**m['artifacts']}.items():
        if p=='QUICK_REFERENCE.md':
            original=subprocess.check_output(['git','show','9764ea46a91b6495ec9063616c0ec1ce6b1045f3:QUICK_REFERENCE.md'],cwd=ROOT)
            if hashlib.sha256(original).hexdigest()!=h or not (ROOT/p).read_bytes().startswith(original): raise ValueError('HISTORICAL_QUICK_REFERENCE_CHANGED')
        elif sha(ROOT/p)!=h: raise ValueError('FROZEN_SHA_MISMATCH:'+p)
    if sha(ROOT/m['baseline']['protected_csv'])!=m['baseline']['protected_csv_sha256']: raise ValueError('PROTECTED_CSV_CHANGED')

def governance():
    # Reuse the frozen R0 identity-key parser, never its execution/research main.
    from tools.stageR_close_r0_v1_blockers import extract_json_identity, normal_log, normal_token, LOG_KEYS, TOKEN_KEYS, digest_strings
    sets={k:set() for k in ('exposed_logs','exposed_tokens','reserved_logs','reserved_tokens','screened_logs','screened_tokens')}
    sources=[]
    for row in csv.DictReader(HISTORY.open()):
        p=ROOT/row['source_manifest']; tokens=set(); logs=set()
        if sha(p)!=row['manifest_sha256']: raise ValueError('HISTORY_SOURCE_CHANGED:'+str(p))
        if p.suffix.lower()=='.csv':
            for record in csv.DictReader(p.open(encoding='utf-8-sig')):
                for k,v in record.items():
                    if k.lower().strip() in TOKEN_KEYS:
                        t=normal_token(v)
                        if t: tokens.add(t)
                    elif k.lower().strip() in LOG_KEYS:
                        log=normal_log(v)
                        if log: logs.add(log)
        else: extract_json_identity(read(p),tokens,logs)
        if digest_strings(tokens)!=row['nuplan_token_set_sha256'] or digest_strings(logs)!=row['nuplan_log_set_sha256']:
            raise ValueError('HISTORY_IDENTITY_SET_MISMATCH:'+str(p))
        kind='exposed' if row['outcome_already_unblinded']=='true' else 'reserved' if row['use_type']=='FROZEN_OR_LOCKED_ROSTER' else 'screened'
        sets[kind+'_logs'].update(logs); sets[kind+'_tokens'].update(tokens)
        sources.append({'path':row['source_manifest'],'sha256':row['manifest_sha256'],'classification':kind,'identity_hashes_verified':True,'historical_stage':row['historical_stage']})
    permanent=read(PERMANENT)['entries']; engineering=read(ENGINEERING)['entries']; a5=read(A5)['records']
    permanent_logs={r['log_id'] for r in permanent+engineering}
    reserved_logs={r['log_id'] for r in a5+engineering}|sets['reserved_logs']
    exposed_logs=set(sets['exposed_logs'])
    for r in permanent:
        if any('OUTCOME_EXPOSED' in reason for reason in r.get('reasons',[])): exposed_logs.add(r['log_id'])
    return {'historical_identity_sets':{k:sorted(v) for k,v in sets.items()},
            'explicit_exposed_logs':sorted(exposed_logs),'permanent_logs':sorted(permanent_logs),'reserved_logs':sorted(reserved_logs),
            'historical_sessions':sorted({session(l) for l in exposed_logs}-{None}),
            'conflict_sessions':sorted({session(l) for l in permanent_logs|reserved_logs}-{None}),
            'historical_source_verification':sources,'history_source_hashes_pass':True,
            'role_closure':'BLOCKED_NO_COMPLETE_CURRENT_U_Q_D_E_SOURCE_ALLOCATION_LEDGER',
            'classification_note':'Use frozen R0 outcome_already_unblinded declaration, exact identity set SHA, and separate frozen reservations. Mere historical metadata screening is NOT outcome exposure and is NOT added as an exclusion. E is not constructed; all history is already declared exposed or reserved.'}

def inspect_db(row, old_tokens, old_logs, frozen_logs, gov):
    path=Path(row['db_path']); stat=path.stat()
    c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True); c.execute('PRAGMA query_only=ON')
    try:
        tags=c.execute('SELECT lower(hex(lidar_pc_token)),group_concat(DISTINCT type) FROM scenario_tag GROUP BY lidar_pc_token ORDER BY 1').fetchall()
        tokens=[r[0] for r in tags]
        schema=c.execute("SELECT type,name,tbl_name,COALESCE(sql,'') FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
        fp=canonical({'relative_path':str(path.relative_to(Path(read(SOURCE)['source_db']['cache_root']))),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'schema_sha256':canonical(schema),'token_set_sha256':token_sha(tokens)})
        valid=fp==row['db_fingerprint_sha256']
        logs=c.execute('SELECT lower(hex(token)),logfile,vehicle_name,date,timestamp,location,map_version FROM log').fetchall()
        if len(logs)!=1: raise ValueError('AMBIGUOUS_DATABASE_LOG:'+str(path))
        log=logs[0][1]; group=session(log)
        common=[]
        if not valid: common.append('FROZEN_DB_FINGERPRINT_MISMATCH')
        if group is None: common.append('SESSION_PROVENANCE_UNRESOLVED')
        if group in gov['historical_sessions']: common.append('DECLARED_HISTORICAL_OUTCOME_SOURCE_GROUP_CONFLICT')
        if group in gov['conflict_sessions']: common.append('PERMANENT_OR_ENGINEERING_SOURCE_GROUP_CONFLICT')
        if not gov['history_source_hashes_pass']: common.append('HISTORICAL_SOURCE_HASH_CLOSURE_FAILED')
        common=sorted(common+UNKNOWN)
        # This query reads native acquisition metadata only, no generated trajectory.
        pcs=c.execute('SELECT lower(hex(p.token)),p.timestamp,lower(hex(p.scene_token)),e.vx,e.vy,e.x,e.y,e.qw,e.qx,e.qy,e.qz,s.roadblock_ids FROM lidar_pc p LEFT JOIN ego_pose e ON e.token=p.ego_pose_token LEFT JOIN scene s ON s.token=p.scene_token ORDER BY p.timestamp,p.token').fetchall()
        bytoken={r[0]:i for i,r in enumerate(pcs)}
        records=[]; counts=Counter()
        for token,types in tags:
            if token in old_tokens or log in old_logs: continue
            if log not in frozen_logs: raise ValueError('UNFROZEN_LOG')
            reasons=list(common); i=bytoken.get(token); speed=None; stamp=None; sc=None; route=None; forward=0; time_ok=False; pose_ok=False
            if i is None: reasons.append('SCENARIO_NOT_RESOLVABLE')
            else:
                p=pcs[i]; stamp=p[1]; sc=p[2]; route=p[11]
                if p[3] is not None and p[4] is not None and math.isfinite(p[3]) and math.isfinite(p[4]): speed=math.hypot(p[3],p[4])
                pose_ok=all(v is not None and math.isfinite(v) for v in p[5:11])
                # Native DB is 20 Hz; official configured ratio .5 yields every other row.
                # No interpolation, padding, controller or scenario is instantiated.
                window=pcs[i:i+161:2]; forward=len(pcs)-i
                time_ok=len(window)==81 and all(type(w[1]) is int for w in window) and all(b[1]>a[1] for a,b in zip(window,window[1:]))
                if not time_ok: reasons.append('ANCHOR_PRIMARY80_NATIVE_TIMESTAMP_SUPPORT_MISSING')
                if not pose_ok: reasons.append('ANCHOR_EGO_METADATA_MISSING')
                if not route: reasons.append('RAW_SCENE_ROUTE_IDS_MISSING_NOT_OFFICIAL_ROUTE')
                # Presence of native DB rows is necessary, not sufficient for official observations.
                reasons.append('OFFICIAL_REPLAY_AND_SCENARIO_EXTRACTION_NOT_BOUND')
            if speed is None: reasons.append('ANCHOR_SPEED_MISSING_OR_NONFINITE')
            elif not speed_gate(speed): reasons.append('ANCHOR_SPEED_BELOW_3P61_NOT_EXECUTION_INITIAL_SPEED')
            counts.update(reasons)
            records.append([token,speed,stamp,sc,sorted((types or '').split(',')),route,forward,time_ok,pose_ok,sorted(set(reasons)-set(common))])
        metadata={'log_id':log,'session_id':group,'session_basis':'database log.timestamp/date/vehicle plus shared filename acquisition prefix; offsets are chunks, not new independent sessions','database_log_token':logs[0][0],'vehicle_id':logs[0][2],'driver_id':None,'log_date':logs[0][3],'acquisition_start_timestamp_us':logs[0][4],'map_location':logs[0][5],'map_version':logs[0][6],'source_file':str(path),'source_partition':row['source_partition'],'source_dataset':read(SOURCE)['source_db']['release'],'frozen_fingerprint_match':valid,'observed_db_fingerprint_sha256':fp,'source_stat_unchanged_during_read':(path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns),'historical_exposure_status':'EXCLUDED_BY_BOUND_HISTORY' if log in gov['explicit_exposed_logs'] else 'NOT_PROVEN_UNEXPOSED','historical_reservation_status':'RESERVED' if log in gov['reserved_logs'] else 'NO_MATCH_IN_BOUND_RESERVATIONS','provenance_completeness':'SOURCE_BOUND_ROLE_HISTORY_INCOMPLETE','applicability_status':'BLOCKED','initial_speed_mps':None,'initial_speed_status':'BLOCKED_EXACT_SCENARIO_EXTRACTION_NOT_BOUND','official_native_route_availability':'UNVERIFIED_RAW_SCENE_IDS_ARE_NOT_DERIVED_ROUTE','common_exclusion_reasons':common,'record_columns':COLUMNS,'records':records}
        return metadata,tokens,counts,fp
    finally: c.close()

def summarize_capacity(census, gov):
    """Known conflicts give an upper bound, not a zero-eligibility claim."""
    groups=defaultdict(list)
    for row in census['logs']: groups[row['session_id']].append(row)
    exposed=set(gov['historical_sessions']); reserved=set(gov['conflict_sessions'])
    available=set(groups)-exposed-reserved
    census['eligible_capacity']=None
    census['capacity_not_zero_claim']=True
    census['eligible_capacity_status']='UNKNOWN_EXACT_ELIGIBILITY; ZERO_CERTIFIED; KNOWN_CONFLICTS_BOUND_CAPACITY'
    census['independent_capacity_upper_bound_after_known_conflicts']=len(available)
    census['Q20_capacity']='Q20_CAPACITY_NOT_AVAILABLE' if len(available)<20 else 'NOT_ESTABLISHED'
    census['Q12_capacity']='NOT_AVAILABLE_EVEN_BEFORE_OTHER_ELIGIBILITY' if len(available)<12 else 'NOT_ACTIVATED'
    census['denominator_summary']={
        'frozen_candidate_scenarios':census['total_frozen_scenarios'],
        'frozen_candidate_logs':len(census['logs']), 'source_sessions':len(groups),
        'sessions_with_multiple_chunks':sum(len(v)>1 for v in groups.values()),
        'session_timestamp_vehicle_date_agreement':all(len({(r['acquisition_start_timestamp_us'],r['vehicle_id'],r['log_date']) for r in v})==1 for v in groups.values()),
        'declared_outcome_conflict_sessions':len(set(groups)&exposed),
        'known_exposure_or_reservation_conflict_sessions':len(set(groups)&(exposed|reserved)),
        'remaining_session_upper_bound':len(available),
        'remaining_log_upper_bound':sum(len(groups[s]) for s in available),
        'remaining_token_upper_bound':sum(r['scenario_count'] for s in available for r in groups[s]),
        'excluded_by_historical_exposure_scenarios':sum(r['scenario_count'] for r in census['logs'] if r['session_id'] in exposed),
        'excluded_by_reservation_or_permanent_conflict_scenarios_overlapping':sum(r['scenario_count'] for r in census['logs'] if r['session_id'] in reserved),
        'initial_speed_below_3p61_execution_count':None,
        'missing_provenance_source_count':sum(not r['frozen_fingerprint_match'] for r in census['logs']),
        'missing_complete_role_provenance_scenarios':census['total_frozen_scenarios'],
        'official_route_reference_insufficiency_count':None,
        'other_frozen_preoutcome_failure_count':None,
        'remaining_eligibility_unknown':True,
        'certified_eligible_independent_units':0,'certified_eligible_logs':0,
        'certified_eligible_scenario_tokens':0,'certified_eligible_diversity':0}
    return census

def run(out):
    verify_s1(); out.mkdir(parents=True,exist_ok=True)
    if (out/'S2_Preflight_Eligibility_Census_v1.json').exists(): raise FileExistsError('CENSUS_ALREADY_EXISTS')
    shardroot=out/'census'; shardroot.mkdir(exist_ok=False)
    source=read(SOURCE); inv=read(INVENTORY)['rows']; old=read(ROOT/source['eligibility_exclusions']['old_smoke_blacklist_path'])['entries']
    old_tokens={r['scenario_token'] for r in old}; old_logs={r['log_id'] for r in old}; frozen_logs=set(source['eligible_universe']['log_universe'])
    gov=governance(); write(out/'S2_Preflight_Exposure_Exclusion_Ledger_v1.json',gov)
    seen=set(); all_tokens=set(); eligible_tokens=set(); logs=[]; counts=Counter(); shards=[]; fps=[]; duplicates=[]
    for n,row in enumerate(inv):
        metadata,tokens,reasons,fp=inspect_db(row,old_tokens,old_logs,frozen_logs,gov)
        all_tokens.update(tokens); fps.append({'partition':row['source_partition'],'db_file':row['db_file'],'db_fingerprint_sha256':fp})
        log=metadata['log_id']
        if log in seen: duplicates.append({'log_id':log,'duplicate_source':row['db_path'],'token_set_sha256':token_sha(tokens)}); continue
        seen.add(log)
        if not metadata['records']: continue
        eligible_tokens.update(r[0] for r in metadata['records']); counts.update(reasons)
        target=shardroot/f'log_{len(logs)+1:04d}.json.gz'
        with target.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
                with io.TextIOWrapper(z,encoding='utf-8') as f: json.dump(metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
        summary={k:v for k,v in metadata.items() if k not in ('records','record_columns')}; summary['scenario_count']=len(metadata['records']); summary['shard']=str(target.relative_to(out)); summary['sha256']=sha(target); logs.append(summary)
        if n%50==0: print(f'metadata files {n+1}/{len(inv)}, unique tokens {len(eligible_tokens)}',flush=True)
    integrity={'source_root_fingerprint_match':canonical(fps)==source['source_db']['source_root_fingerprint_sha256'],'all_token_count_match':len(all_tokens)==source['unfiltered_universe']['unique_scenario_token_count'],'all_token_hash_match':token_sha(all_tokens)==source['unfiltered_universe']['token_set_sha256'],'frozen_token_count_match':len(eligible_tokens)==source['eligible_universe']['unique_scenario_token_count'],'frozen_token_hash_match':token_sha(eligible_tokens)==source['eligible_universe']['token_set_sha256'],'frozen_log_count_match':len(logs)==source['eligible_universe']['unique_log_count']}
    census={'status':'COMPLETE_METADATA_ENUMERATION_ELIGIBILITY_BLOCKED','complete_candidate_enumeration':all(integrity.values()),'complete_scientific_eligibility_certification':False,'source':{'path':str(SOURCE.relative_to(ROOT)),'sha256':sha(SOURCE),'inventory_sha256':sha(INVENTORY)},'integrity':integrity,'total_frozen_scenarios':len(eligible_tokens),'total_frozen_logs':len(logs),'total_source_sessions':len({r['session_id'] for r in logs}),'certified_eligible_logs':0,'certified_eligible_tokens':0,'eligible_capacity':None,'capacity_not_zero_claim':True,'INDEPENDENCE_UNIT':'SESSION','Q20_capacity':'NOT_ESTABLISHED','roster_materialized':False,'reason_counts_overlapping_scenarios':dict(counts),'unknown_is_not_observed_failure':True,'common_record_contract':{'records_normalized_by_log':True,'all_candidates_retained':True,'reasons':'common_exclusion_reasons are blocking; row metadata_flags describe tagged anchor only, not execution eligibility failures','required_forward_route_support':'frozen _tsb_required_distance(current_speed, absolute_episode_time, arm, params); complete rolling-horizon native route availability unresolved; no numeric replacement','full_native_replay':'UNVERIFIED','execution_initial_speed_mps':None,'initial_speed_below_3p61_count':None,'note':'Tag anchor is not necessarily first extracted state; mapping offsets may be -3s. Raw scene route IDs are not the official trajectory-derived route.','no_outcome_selection':True,'E_identity_materialization':False},'duplicate_source_aliases':duplicates,'logs':logs,'counters':{'simulation':0,'runner_run':0,'new_scientific_outcome_exposure':0,'RBR_training':0,'E_access':0}}
    summarize_capacity(census, gov)
    write(out/'S2_Preflight_Eligibility_Census_v1.json',census); verify_s1()
    print(json.dumps({k:v for k,v in census.items() if k not in ('logs','duplicate_source_aliases','common_record_contract')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output-dir',type=Path,default=OUT); args=ap.parse_args(); run(args.output_dir)
