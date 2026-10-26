#!/usr/bin/env python3
"""Prospective metadata-only D pool for legacy E3 sources; never restores their V eligibility."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import gzip
import hashlib
import heapq
import json
from pathlib import Path
from collections import Counter
from tools.rbr_v2_d1_capacity import ROOT,OUT,SDC,read,write,sha


def main():
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.s2r_engineering_closure import evaluate_candidate,SALT,TOP_K
 from tools.sdc_v1_selection import assess,anchor_candidates
 from tools.b1_production_preflight import zero_run_guard
 cat=read(SDC/'Pool_Split_v1.json');rows={r['session_id']:r for r in cat['rows'] if r['exposure_class'].startswith('E3')};heaps={s:[] for s in rows}
 old=read(ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json');logs={r['log_id']:r for r in old['logs']};metahashes={}
 for log in old['logs']:
  sid=log['session_id']
  if sid not in rows:continue
  path=ROOT/'docs/stageR/s2_preflight'/log['shard'];metahashes[str(path.relative_to(ROOT))]=sha(path)
  with gzip.open(path,'rt') as f:a=json.load(f)
  c=a['record_columns'];ti=c.index('scenario_token');tm=c.index('tag_anchor_timestamp_us');tags=c.index('task_types');support=c.index('anchor_timestamp_80_state_support')
  for rec in a['records']:
   token=rec[ti];digest=hashlib.sha256(f"{SALT}|{sid}|{token}|{log['log_id']}".encode()).hexdigest()
   item=dict(rank_sha256=digest,session_id=sid,scenario_token=token,anchor_timestamp_us=int(rec[tm]),task_types=rec[tags],anchor_80_state_support=bool(rec[support]),log_id=log['log_id'],db_path=log['source_file'],map_name=log['map_version'],source_fingerprint_sha256=log['observed_db_fingerprint_sha256'],source_partition=log['source_partition'])
   value=(-int(digest,16),token,log['log_id'],item);heap=heaps[sid]
   if len(heap)<TOP_K:heapq.heappush(heap,value)
   elif value[0]>heap[0][0]:heapq.heapreplace(heap,value)
 ranked={s:sorted([v[-1] for v in heap],key=lambda r:(r['rank_sha256'],r['scenario_token'],r['log_id'])) for s,heap in heaps.items()}
 plan=dict(purpose='D_ONLY_LEGACY_E3_METADATA_CAPACITY_NO_V_C_RESTORATION',source_SESSION=sorted(rows),candidate_selection='Original S2R hash/top256 first broadly eligible representative; original3.61m/s gate THEN SDC>=5m/s representative preselection; no alternate winner after speed screen',
           SDC_physical_rule_sha256=sha(SDC/'Frozen_Qualification_Protocol_v1.json'),source_metadata_sha256=metahashes,ranked=ranked,new_rollout=0)
 target=OUT/'D_E3_Static_Plan.json'
 if target.exists():raise FileExistsError('D metadata plan already frozen')
 write(target,plan);rule=read(SDC/'Frozen_Qualification_Protocol_v1.json');results=[];cache={}
 with zero_run_guard() as calls:
  for sid in sorted(rows):
   winner=None;rejects=[]
   for candidate in ranked[sid]:
    win,reason=evaluate_candidate(candidate,cache)
    if win is not None:winner=win;break
    rejects.append(dict(rank_sha256=candidate['rank_sha256'],reason=reason))
   trials=[]
   if winner is None:status='REPRESENTATIVE_NOT_ESTABLISHED';reason=['NO_BROAD_STATIC_REPRESENTATIVE_IN_ORIGINAL_256_PREFIX']
   elif float(winner['initial']['initial_speed_mps'])<5:status='EXCLUDED';reason=['ORIGINAL_REPRESENTATIVE_INITIAL_SPEED_LT_5']
   else:
    row=copy.deepcopy(rows[sid]);row['log']=logs[winner['log_id']];row['static'].update(representative_log_id=winner['log_id'],representative_scenario_token=winner['scenario_token'],execution_initial_speed_mps=winner['initial']['initial_speed_mps'])
    chosen=None
    for i,anchor in enumerate(anchor_candidates(row,rule),1):
     try:r=assess(anchor,rule);r.update(anchor_rank=i)
     except Exception as e:r=dict(status='METADATA_BLOCKED',reasons=[type(e).__name__+': '+str(e)],anchor_rank=i)
     if 'evidence' in r and 'clock' in r['evidence']:r['evidence']['clock']={k:v for k,v in r['evidence']['clock'].items() if k!='rows'}
     trials.append(r)
     if r['status']=='ELIGIBLE':chosen=r;break
    status='ELIGIBLE_D_ONLY' if chosen else 'EXCLUDED';reason=[] if chosen else ['NO_ELIGIBLE_SDC_ANCHOR_IN_FIXED16']
   result=dict(session_id=sid,status=status,reasons=reason,V_C_eligible=False,exposure_class='E3_PRIMARY_METHOD_DEVELOPMENT',representative=winner,representative_rejections=rejects,anchor_trials=trials)
   results.append(result);write(OUT/'D_E3_Static_Progress.json',dict(rows=results));print('D-only legacy E3 static',len(results),'/31',sid,status,flush=True)
 if any(calls.values()):raise ValueError('D metadata guard violation')
 write(OUT/'D_E3_Static_Results.json',dict(rows=results,counts=dict(Counter(r['status'] for r in results)),zero_run_guard=calls,plan_sha256=sha(target),scientific_qualification=False,existing_SDC_arms=0))

if __name__=='__main__':main()
