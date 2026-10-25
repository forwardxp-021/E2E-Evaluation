#!/usr/bin/env python3
"""Freeze scoped benchmark identities and noncircular source/data credentials."""
from __future__ import annotations
import argparse,copy,os,sys,subprocess
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sdc_v1_selection import OUT,ROOT,read,write,sha,canon
from tools.sdc_v1_execution import BUDGET


def identities(phase):
    source='evidence/DEVELOPMENT_Applicability_v4.json' if phase=='DEVELOPMENT' else 'evidence/QUALIFICATION_Applicability_v1.json'
    census=read(OUT/source);rule=read(OUT/('Development_Design_v4.json' if phase=='DEVELOPMENT' else 'Frozen_Qualification_Protocol_v1.json'))
    eligible=[x for x in census['rows'] if x['status']=='ELIGIBLE'];eligible.sort(key=lambda x:(__import__('hashlib').sha256((rule['selection_salt']+'|'+x['session_id']).encode()).hexdigest(),x['session_id']))
    count=3 if phase=='DEVELOPMENT' else 20;chosen=eligible[:count]
    if len(chosen)<count:
        write(OUT/(phase+'_Capacity_Blocked_v1.json'),dict(status='CAPACITY_BLOCKED',required=count,eligible=len(eligible),no_relaxation=True),exclusive=True);return
    arms=[];roster=[]
    for i,x in enumerate(chosen,1):
        pid=('SDC-DEV-' if phase=='DEVELOPMENT' else 'SDC-QUAL-')+f'{i:02d}'
        for arm in ['BASELINE','TREATMENT']:
            spec=copy.deepcopy(x['spec']);spec.update(pair_id=pid,arm=arm,run_id=pid+'-'+arm,phase=phase);arms.append(spec)
        roster.append(dict(pair_id=pid,session_id=x['session_id'],log_id=x['spec']['log_id'],scenario_token=x['spec']['scenario_token'],anchor_rank=x['anchor_rank'],anchor_rank_hash=x['anchor_rank_hash'],
            session_rank_hash=__import__('hashlib').sha256((rule['selection_salt']+'|'+x['session_id']).encode()).hexdigest(),data=x['evidence']['sources'],exposure_class=x['spec']['exposure_class']))
    if len({x['session_id'] for x in roster})!=count or len({x['log_id'] for x in roster})!=count:raise ValueError('SESSION/log independence failure')
    if phase=='QUALIFICATION':
        dev=set(read(OUT/'Pool_Split_v1.json')['development_SESSION']);hist=set(read(OUT/'Pool_Split_v1.json')['historical_excluded'])
        if any(x['session_id'] in dev|hist for x in roster):raise ValueError('Qualification overlaps development/history')
    write(OUT/(phase+'_Arm_Specs_v1.json'),dict(arms=arms,applicability_path=source,applicability_sha256=sha(OUT/source)),exclusive=True)
    write(OUT/(phase+'_Frozen_Roster_v1.json'),dict(roster=roster,eligible_SESSION=len(eligible),selection_rule_sha256=sha(OUT/('Development_Design_v4.json' if phase=='DEVELOPMENT' else 'Frozen_Qualification_Protocol_v1.json'))),exclusive=True)
    print(phase,'frozen',len(roster),'pairs; eligible',len(eligible))


def bind(phase):
    specfile=OUT/(phase+'_Arm_Specs_v1.json');specs=read(specfile)['arms'];pre=OUT/'evidence'/(phase+'_Production_Preflight_v1.json')
    if read(pre)['status']!='PASS':raise ValueError('Production preflight not PASS')
    paths=['tools/sdc_v1_selection.py','tools/sdc_v1_execution.py','tools/sdc_v1_prepare.py','tests/test_sdc_v1.py']
    sourcecommit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={p:sha(ROOT/p) for p in paths}
    for p,h in sources.items():
        blob=subprocess.check_output(['git','show',sourcecommit+':'+p])
        if __import__('hashlib').sha256(blob).hexdigest()!=h:raise ValueError('Commit all source before binding')
    proofs=[OUT/'Owner_Task_Authority_v1.json',OUT/'Development_Design_v4.json',OUT/'Pool_Split_v1.json',specfile,pre,OUT/(phase+'_Frozen_Roster_v1.json'),OUT/read(specfile)['applicability_path'],ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Execution_Source_Binding_v1.json',ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json']
    if phase=='QUALIFICATION':proofs += [OUT/'Frozen_Qualification_Protocol_v1.json',OUT/'DEVELOPMENT_Execution_Binding_v1.json',OUT/'evidence/Development_Results_Freeze_v1.json']
    data={}
    census=read(OUT/read(specfile)['applicability_path'])
    for spec in specs:
        item=next(x for x in census['rows'] if x['session_id']==spec['session_id']);src=item['evidence']['sources'];data[src['db_path']]=src['db_sha256'];data[src['metadata_shard']]=src['metadata_shard_sha256']
        metadata=ROOT.parent/'nuplan/dataset/maps/nuplan-maps-v1.0.json';meta=read(metadata);data[str(metadata)]=sha(metadata)
        gpkg=metadata.parent/spec['map_name']/meta[spec['map_name']]['version']/'map.gpkg';data[str(gpkg)]=sha(gpkg)
    proofhash={str(p.relative_to(ROOT)):sha(p) for p in proofs}
    for item in read(pre)['arms']:proofhash[item['config_path']]=item['config_sha256']
    auth=dict(stage='E2E_SCOPED_DIRECT_COAST_V1',phase=phase,owner_approved=True,budget=BUDGET,source_commit=sourcecommit,source_sha256=sources,proof_sha256=proofhash,data_sha256=data,run_ids=[s['run_id'] for s in specs],spec_sha256={s['run_id']:canon(s) for s in specs},
        science='Unchanged F2 scientific source; only applicability, identity scheduling and prespecified temporal gate added',qualification_scientific_failure='Continue original full denominator; no retry, no replacement',exposure='B benchmark; source applicability access and execution are E2, never E5')
    write(OUT/(phase+'_Execution_Binding_v1.json'),auth,exclusive=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['identities','bind']);p.add_argument('phase',choices=['DEVELOPMENT','QUALIFICATION']);a=p.parse_args()
    (identities if a.mode=='identities' else bind)(a.phase)
