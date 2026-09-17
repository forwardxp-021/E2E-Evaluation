#!/usr/bin/env python3
"""Static S2-PRE validators. Deliberately no execution or budget-activation API."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def validate_draft(value):
    required={'stage','S2_EXECUTION_AUTHORIZED','RUN_BUDGET_ACTIVE','future_max_simulator_entries','active_budget','consumed_entries','authorization_by_existence','RBR_TRAINING','E_ACCESS','Q_to_D_active','rules','output_root','budget_ledger','activation_requirements','zero_run_counters'}
    if set(value)!=required: raise ValueError('AUTHORIZATION_DRAFT_EXACT_KEYS_REQUIRED')
    for key in ('S2_EXECUTION_AUTHORIZED','RUN_BUDGET_ACTIVE','authorization_by_existence','Q_to_D_active'):
        if value[key] is not False: raise ValueError('CLOSED_DRAFT_REQUIRED:'+key)
    for key,want in [('future_max_simulator_entries',40),('active_budget',0),('consumed_entries',0)]:
        if type(value[key]) is not int or value[key]!=want: raise ValueError('BUDGET_DRAFT_VALUE:'+key)
    if value['RBR_TRAINING']!='NOT_AUTHORIZED' or value['E_ACCESS']!='NOT_AUTHORIZED': raise ValueError('FIREWALL_CLOSED_REQUIRED')
    expected_zero={'SIMULATION_RUNS':0,'RUNNER_RUN_CALLS':0,'NEW_SCIENTIFIC_OUTCOME_EXPOSURE':0,'RBR_TRAINING':0,'E_ACCESS':0}
    if value['zero_run_counters'] != expected_zero: raise ValueError('ZERO_RUN_COUNTERS_REQUIRED')
    return {'static_validation':'PASS','scientific_qualification':False,'execution_possible':False}

def validate_fresh_root(root, parent):
    raw=Path(root); target=raw.resolve(); base=Path(parent).resolve()
    if target==base or base not in target.parents or raw.is_symlink() or target.exists(): raise ValueError('OUTPUT_ROOT_NOT_FRESH_OR_OUTSIDE_RESERVATION')

def select_q20(candidates):
    """Pure future selection algorithm; callers must provide certified metadata only."""
    allowed={'log_id','scenario_token','session_id','eligibility','exclusion_reasons','evidence_sha256'}
    if any(set(r)!=allowed for r in candidates): raise ValueError('METADATA_EXACT_KEYS_REQUIRED_NO_OUTCOMES')
    if any(r['eligibility']!='PASS' or r['exclusion_reasons'] or not r['session_id'] or len(r['evidence_sha256'])!=64 for r in candidates): raise ValueError('UNCERTIFIED_CANDIDATE')
    ids=[r['scenario_token'] for r in candidates]
    if len(ids)!=len(set(ids)): raise ValueError('DUPLICATE_TOKEN')
    grouped={}
    for r in candidates:
        if r['log_id'] in grouped and grouped[r['log_id']]!=r['session_id']: raise ValueError('LOG_SESSION_CONFLICT')
        grouped[r['log_id']]=r['session_id']
    def rank(s): return hashlib.sha256(('S1_TSB_Q_v0.1|'+s).encode()).hexdigest()
    ordered=sorted(candidates,key=lambda r:(rank(r['log_id']),rank(r['scenario_token'])))
    chosen=[]; sessions=set(); logs=set()
    for r in ordered:
        if r['session_id'] in sessions or r['log_id'] in logs: continue
        sessions.add(r['session_id']); logs.add(r['log_id']); chosen.append(r)
        if len(chosen)==20: return chosen
    raise ValueError('Q20_CAPACITY_NOT_AVAILABLE_NO_AUTOMATIC_Q12')

def verify_manifest(root, manifest):
    for row in manifest['bindings']:
        path=Path(row['path']); path=path if path.is_absolute() else Path(root)/path
        if not path.is_file() or sha(path)!=row['sha256']: raise ValueError('SHA_MISMATCH:'+str(path))
    if manifest['scientific_qualification'] is not False: raise ValueError('PREFLIGHT_CANNOT_QUALIFY_SCIENCE')
    return True

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--manifest',type=Path,required=True); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads(args.manifest.read_text()); verify_manifest(root,manifest)
    draft=json.loads((args.manifest.parent/'S2_Preflight_Budget_Authorization_DRAFT_v1.json').read_text())
    result=validate_draft(draft); validate_fresh_root(draft['output_root'],root/'outputs')
    print(json.dumps(result,indent=2))
