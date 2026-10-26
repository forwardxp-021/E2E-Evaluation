#!/usr/bin/env python3
"""Incremental official source acquisition + unchanged SDC static assessment, no simulation."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,copy,csv,gzip,hashlib,json,math,shutil,sqlite3,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from tools.rbr_v2_d2_download import ROOT,OUT,LOCAL,LIMIT,RESERVE,read,write,sha,download_entry


def metadata(e):
 from tools.s2_preflight_metadata_census import COLUMNS,session
 from tools.tsb_direct_coast_p2_q0 import historical_db_fingerprint
 path=Path(e['path']);sid=e['session_id'];dest=OUT/'metadata'/(sid+'.json.gz')
 with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
  db.execute('PRAGMA query_only=ON')
  logs=db.execute('SELECT lower(hex(token)),logfile,vehicle_name,date,timestamp,location,map_version FROM log').fetchall()
  if len(logs)!=1 or session(logs[0][1])!=sid or logs[0][1]+'.db'!=e['filename']:raise ValueError('Official DB header/source SESSION identity mismatch')
  tags=db.execute('SELECT lower(hex(lidar_pc_token)),group_concat(DISTINCT type) FROM scenario_tag GROUP BY lidar_pc_token ORDER BY 1').fetchall()
  pcs=db.execute('SELECT lower(hex(p.token)),p.timestamp,lower(hex(p.scene_token)),e.vx,e.vy,e.x,e.y,e.qw,e.qx,e.qy,e.qz,s.roadblock_ids FROM lidar_pc p LEFT JOIN ego_pose e ON e.token=p.ego_pose_token LEFT JOIN scene s ON s.token=p.scene_token ORDER BY p.timestamp,p.token').fetchall()
 bytoken={r[0]:i for i,r in enumerate(pcs)};records=[]
 for token,types in tags:
  i=bytoken.get(token)
  if i is None:continue
  p=pcs[i];window=pcs[i:i+161:2];support=len(window)==81 and all(type(w[1]) is int for w in window) and all(b[1]>a[1] for a,b in zip(window,window[1:]));pose=all(v is not None and math.isfinite(v) for v in p[5:11]);speed=math.hypot(p[3],p[4]) if p[3] is not None and p[4] is not None else None
  records.append([token,speed,p[1],p[2],sorted((types or '').split(',')),p[11],len(pcs)-i,support,pose,[]])
 data=dict(record_columns=COLUMNS,records=records,source_sha256=e['sha256'],official_split=e['official_split'])
 dest.parent.mkdir(exist_ok=True)
 with dest.open('wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:g.write(json.dumps(data,separators=(',',':')).encode())
 log=dict(log_id=logs[0][1],session_id=sid,database_log_token=logs[0][0],vehicle_id=logs[0][2],log_date=logs[0][3],acquisition_start_timestamp_us=logs[0][4],map_location=logs[0][5],map_version=logs[0][6],
          source_file=str(path),source_partition='AWS_V1_1_'+e['official_split'].upper(),source_dataset='OFFICIAL_NUPLAN_V1_1',frozen_fingerprint_match=True,observed_db_fingerprint_sha256=historical_db_fingerprint(path),source_stat_unchanged_during_read=True,shard=str(dest.resolve()),scenario_count=len(records),sha256=sha(dest))
 return log,data


def scan(e):
 from tools.sdc_v1_selection import assess,anchor_candidates
 from tools.s2r_engineering_closure import evaluate_candidate,SALT
 from tools.b1_production_preflight import zero_run_guard
 dest=OUT/'static'/(e['session_id']+'.json')
 if dest.exists():
  old=read(dest)
  if old['download_sha256']!=e['sha256']:raise ValueError('Immutable downloaded source changed')
  return old
 started=time.monotonic();log,data=metadata(e);ranked=[]
 cols=data['record_columns']
 for r in data['records']:
  digest=hashlib.sha256((SALT+'|'+e['session_id']+'|'+r[0]+'|'+log['log_id']).encode()).hexdigest()
  ranked.append(dict(rank_sha256=digest,scenario_token=r[0],session_id=e['session_id'],anchor_timestamp_us=r[2],task_types=r[4],anchor_80_state_support=r[7],log_id=log['log_id'],db_path=log['source_file'],map_name=log['map_version'],source_fingerprint_sha256=log['observed_db_fingerprint_sha256'],source_partition=log['source_partition']))
 ranked=sorted(ranked,key=lambda c:(c['rank_sha256'],c['scenario_token']))[:256];rejections=[];win=None;trials=[];chosen=None
 rule=read(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/Frozen_Qualification_Protocol_v1.json')
 with zero_run_guard() as calls:
  for candidate in ranked:
   w,reason=evaluate_candidate(candidate,{})
   if w is not None:win=w;break
   rejections.append(dict(rank_sha256=candidate['rank_sha256'],reason=reason))
  if not win:reasons=['NO_BROAD_STATIC_REPRESENTATIVE_IN_FIXED_PREFIX'];status='EXCLUDED'
  elif float(win['initial']['initial_speed_mps'])<5:reasons=['REPRESENTATIVE_INITIAL_SPEED_LT_5'];status='EXCLUDED'
  else:
   row=dict(session_id=e['session_id'],log=log,static=dict(representative_scenario_token=win['scenario_token']),exposure_class='E0_METADATA_ONLY')
   for rank,anchor in enumerate(anchor_candidates(row,rule),1):
    try:
     item=assess(anchor,rule);item.update(anchor_rank=rank)
    except ValueError as error:
     message=str(error)
     if message.startswith(('OFFICIAL_UNIQUE_RUNNABLE_IDENTITY_REQUIRED','NATIVE_ROUTE_','CURRENT_EGO_OUTSIDE_NATIVE','Initial ego','canonical clock','NO_NATIVE','Cannot')):
      item=dict(status='METADATA_REJECTED',reasons=[message],anchor_rank=rank)
     else:raise
    trials.append(item)
    if item['status']=='ELIGIBLE':chosen=item;break
   status='ELIGIBLE' if chosen else 'EXCLUDED';reasons=[] if chosen else ['NO_SDC_ANCHOR_IN_FIXED16']
 if any(calls.values()):raise RuntimeError('Forbidden scientific call during source scan')
 result=dict(session_id=e['session_id'],log=log,status=status,reasons=reasons,source_split=e['official_split'],download_sha256=e['sha256'],
             representative=win,representative_rejections=rejections,anchor_trials=trials,selected=chosen,zero_run_guard=calls,
             wall_seconds=time.monotonic()-started,scanner_sha256=sha(Path(__file__)),frozen_SDC_source_sha256=sha(ROOT/'tools/sdc_v1_selection.py'),new_scientific_runs=0)
 write(dest,result);return result


def get(e):
 receipt=OUT/'downloads'/(e['filename']+'.json')
 if receipt.exists():return download_entry(e)
 errors=[]
 for attempt in range(3):
  try:return download_entry(e)
  except RuntimeError as err:
   if str(err) in ('DOWNLOAD_50GB_BUDGET_STOP','DISK_RESERVE_STOP'):raise
   errors.append(dict(attempt=attempt+1,error=repr(err)));write(OUT/'download_errors'/(e['session_id']+'.json'),errors)
   if attempt==2:raise
  except Exception as err:
   errors.append(dict(attempt=attempt+1,error=repr(err)));write(OUT/'download_errors'/(e['session_id']+'.json'),errors)
   if attempt==2:raise


def run(max_batches=None):
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env();plan=read(OUT/'acquisition_plan.json');started=time.monotonic();stop='SOURCE_FRAME_EXHAUSTED';i=0;batchno=0
 known_status={p.stem:read(p)['status'] for p in (OUT/'static').glob('*.json')};counts=Counter(known_status.values())
 while i<len(plan['rows']) and plan['rows'][i]['session_id'] in known_status:i+=1
 while i<len(plan['rows']):
  if max_batches is not None and batchno>=max_batches:stop='BOUNDED_ENGINEERING_BATCH_CHECKPOINT';break
  if counts.get('ELIGIBLE',0)>=plan['target_static_capacity']:stop='PREREGISTERED_STATIC_CAPACITY_TARGET';break
  batch=[];estimated_raw=0;estimated_bytes=0;budget=read(OUT/'download_budget.json')
  for e in plan['rows'][i:i+plan['batch_SESSION']]:
   if (OUT/'downloads'/(e['filename']+'.json')).exists():batch.append(e);continue
   if shutil.disk_usage(ROOT).free-estimated_raw-e['uncompressed_bytes']<RESERVE:stop='DISK_RESERVE_STOP';break
   if budget['received_bytes']+budget['reserved_bytes']+estimated_bytes+e['compressed_bytes']+16384>LIMIT:stop='DOWNLOAD_50GB_BUDGET_STOP';break
   batch.append(e);estimated_raw+=e['uncompressed_bytes'];estimated_bytes+=e['compressed_bytes']+16384
  if not batch:break
  with ThreadPoolExecutor(max_workers=4) as pool:
   for e,future in zip(batch,[pool.submit(get,e) for e in batch]):
    receipt=future.result()
    try:result=scan(receipt)
    except Exception as error:
     write(OUT/'scan_errors'/(e['session_id']+'.json'),dict(error=repr(error),source_sha256=receipt['sha256'],scanner_sha256=sha(Path(__file__)),scientific_runs=0));raise
    i+=1
    if e['session_id'] not in known_status:known_status[e['session_id']]=result['status'];counts[result['status']]+=1
    write(OUT/'acquisition_progress.json',dict(processed=i,total_planned=len(plan['rows']),counts=dict(counts),received_bytes=read(OUT/'download_budget.json')['received_bytes'],disk_free_bytes=shutil.disk_usage(ROOT).free,this_process_wall_seconds=time.monotonic()-started))
    print('source',i,'/',len(plan['rows']),e['session_id'],result['status'],'eligible',counts.get('ELIGIBLE',0),'download_GB',round(read(OUT/'download_budget.json')['received_bytes']/1e9,3),flush=True)
  batchno+=1
  if len(batch)<plan['batch_SESSION']:break
 write(OUT/'acquisition_status.json',dict(stop_reason=stop,processed=i,counts=dict(Counter(read(p)['status'] for p in (OUT/'static').glob('*.json'))),download=read(OUT/'download_budget.json'),disk_free_bytes=shutil.disk_usage(ROOT).free,
                                        this_process_wall_seconds=time.monotonic()-started))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--max-batches',type=int);a=p.parse_args();run(a.max_batches)
