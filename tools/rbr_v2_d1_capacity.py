#!/usr/bin/env python3
"""SESSION census under unchanged SDC static rules; never reads V/C scientific results."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import copy
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d1_v1';SDC=ROOT/'docs/stageR/e2e_scoped_direct_coast_v1'


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def session(name):
 m=re.fullmatch(r'(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_\d+_\d+\.db',name)
 return m.group(1) if m else None


def inventory():
 roles={r['session_id']:r for r in csv.DictReader((ROOT/'docs/stageR/s2r_role_aware_preflight/S2R_V_Eligibility_Census_v1.csv').open())}
 old=read(ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json')['logs'];names={Path(r['source_file']).name for r in old}
 files=sorted((ROOT.parent/'nuplan/dataset/data/cache').rglob('*.db'));byname={};records=[]
 for p in files:
  s=p.stat();sid=session(p.name);byname.setdefault(p.name,[]).append(p)
  records.append(dict(path=str(p),resolved_path=str(p.resolve()),session_id=sid,bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,previous_census_filename=p.name in names,role_class=roles.get(sid,{}).get('exposure_class','UNKNOWN'),raw_opened=False))
 new=[]
 for name,aliases in sorted(byname.items()):
  if name in names:continue
  p=aliases[0];sid=session(name)
  if roles.get(sid,{}).get('exposure_class','').startswith('E5'):continue
  before=p.stat()
  with sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True) as db:
   columns=[r[1] for r in db.execute('pragma table_info(log)')]
   rows=db.execute('select hex(token),logfile,vehicle_name,map_version from log').fetchall()
  h=sha(p);after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed while inventoried')
  new.append(dict(filename=name,session_id=sid,new_SESSION=sid not in roles,source_path=str(p),aliases=list(map(str,aliases)),source_sha256=h,
                  metadata_columns=columns,log_rows=rows,provenance_status='KNOWN_SESSION_ADDITIONAL_CHUNK_NOT_NEW_INDEPENDENT_UNIT' if sid in roles else 'SOURCE_IDENTIFIED_EXPOSURE_ROLE_UNRESOLVED',allowed_D_V_C=False))
 cat=read(SDC/'Pool_Split_v1.json');existing={}
 for phase,rev in [('DEVELOPMENT',4),('QUALIFICATION',2)]:
  a=read(SDC/f'evidence/{phase}_Applicability_v{rev}.json')
  for row in a['rows']:
   if row['status']!='POLICY_EXCLUDED':existing[row['session_id']]=dict(phase=phase,**row)
 additional=[copy.deepcopy(r) for r in cat['rows'] if not r['policy_reasons'] and r['session_id'] not in existing]
 # Rank and all additional sources frozen before any newly inspected static outcomes.
 salt='RBR_V2_D1_ADDITIONAL_METADATA_ONLY_SAME_SDC_RULE_V1'
 additional.sort(key=lambda r:hashlib.sha256((salt+'|'+r['session_id']).encode()).hexdigest())
 write(OUT/'additional_static_plan.json',dict(rule_path=str((SDC/'Frozen_Qualification_Protocol_v1.json').relative_to(ROOT)),rule_sha256=sha(SDC/'Frozen_Qualification_Protocol_v1.json'),
       selection='ALL_ORIGINAL_POLICY_ELIGIBLE_BUT_NOT_PREVIOUSLY_SDC_STATIC_ASSESSED; no rule relaxation, no outcome subset',order_salt=salt,rows=additional,source_scan_only=True,new_rollout=0))
 write(OUT/'local_source_inventory.json',dict(physical_DB_files=len(files),unique_filename_chunks=len(byname),old_census_chunks=len(names),old_census_SESSION=len(roles),local_filename_SESSION=len(set(session(p.name) for p in files)),
       new_filename_chunks=len(new),new_SESSION=sorted({r['session_id'] for r in new if r['new_SESSION']}),new_source_rows=new,paths=records,
       E5_raw_opened=0,additional_same_rule_static_SESSION=len(additional),new_chunks_not_new_SESSION=True))
 print('inventory',len(files),'physical files;',len(new),'new chunks;',len(additional),'additional same-rule candidates',flush=True)


def scan():
 from tools.sdc_v1_selection import assess,anchor_candidates
 from tools.b1_production_preflight import zero_run_guard
 plan=read(OUT/'additional_static_plan.json');rule=read(ROOT/plan['rule_path'])
 if sha(ROOT/plan['rule_path'])!=plan['rule_sha256']:raise ValueError('Frozen SDC applicability changed')
 target=OUT/'additional_static_results.json'
 if target.exists():raise FileExistsError('No silent replacement of immutable capacity census')
 rows=[]
 with zero_run_guard() as calls:
  for j,row in enumerate(plan['rows']):
   trials=[];chosen=None
   for k,candidate in enumerate(anchor_candidates(row,rule),1):
    try:
     item=assess(candidate,rule);item.update(anchor_rank=k)
    except Exception as e:item=dict(session_id=row['session_id'],status='METADATA_BLOCKED',reasons=[type(e).__name__+': '+str(e)],anchor_rank=k)
    if 'evidence' in item and 'clock' in item['evidence']:
     clock=item['evidence']['clock'];payload=json.dumps(clock.get('rows'),sort_keys=True).encode();item['evidence']['clock_rows_sha256']=hashlib.sha256(payload).hexdigest();item['evidence']['clock']={k:v for k,v in clock.items() if k!='rows'}
    trials.append(item)
    if item['status']=='ELIGIBLE':chosen=item;break
   item=dict(chosen) if chosen else dict(session_id=row['session_id'],status='EXCLUDED' if any(x['status']=='EXCLUDED' for x in trials) else 'METADATA_BLOCKED',reasons=['NO_ELIGIBLE_ANCHOR_IN_UNCHANGED_FIXED_FRAME'])
   item['anchor_trials']=trials;rows.append(item)
   write(OUT/'additional_static_progress.json',dict(rows=rows,processed=len(rows),planned=len(plan['rows'])))
   print('additional static',j+1,'/',len(plan['rows']),row['session_id'],item['status'],len(trials),flush=True)
 if any(calls.values()):raise ValueError('Forbidden scientific call during metadata audit')
 write(target,dict(rows=rows,counts=dict(Counter(r['status'] for r in rows)),rule_sha256=plan['rule_sha256'],zero_run_guard=calls))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['inventory','scan']);a=p.parse_args();inventory() if a.mode=='inventory' else scan()
