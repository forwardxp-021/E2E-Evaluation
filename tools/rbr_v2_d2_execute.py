#!/usr/bin/env python3
"""One-use D-only SDC scientific executor; independent ledger, hard budgets, no retries or V/C."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,json,resource,secrets,signal,subprocess,time
from pathlib import Path
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha,accounting
SCIENCE=ROOT/'outputs/stageR/rbr_v2_d2_D';STATE=OUT/'science_ledger.json';BIND=OUT/'D_execution_binding.json'


def validate(run_id=None,all_data=False):
 b=read(BIND)
 if b['stage']!='RBR_V2_D2_D_ONLY_V1' or not b['owner_task_authorized'] or b['first_arms']!=80 or b['technical_retry']!=0:raise PermissionError('Exact D-only authorization required')
 if run_id is not None and run_id not in b['run_ids']:raise PermissionError('D run identity not authorized')
 if sha(BIND)!=__import__('hashlib').sha256(subprocess.check_output(['git','show','HEAD:'+str(BIND.relative_to(ROOT))],cwd=ROOT)).hexdigest():raise PermissionError('D freeze must be committed before first scientific step')
 for name,h in b['source_sha256'].items():
  if sha(ROOT/name)!=h:raise PermissionError('Frozen D source changed '+name)
 for name,h in b['data_sha256'].items():
  if all_data and sha(name)!=h:raise PermissionError('Raw DB changed '+name)
 return b


def preflight():
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.b1_production_preflight import zero_run_guard
 from tools.tsb_direct_coast_f2_production import build_f2_arm,validate_f2_production
 b=read(BIND);rows=[];held=[];root=OUT/'preflight'
 if root.exists():raise FileExistsError('Preflight evidence already exists; preserve failed revision separately')
 with zero_run_guard() as calls:
  for spec in b['arms']:
   a=build_f2_arm(spec,root/spec['run_id']);held.append(a);p=validate_f2_production(a)
   static=read(OUT/'static'/(spec['session_id']+'.json'))['selected']['evidence']['clock']
   if {k:v for k,v in p['clock'].items() if k not in ['observer','shared_original_scenario']}!=static:raise ValueError('New source preflight clock differs from frozen static mapping')
   rows.append(dict(run_id=spec['run_id'],status='PASS',config_path=str(a.run_root/'resolved_config.json'),config_sha256=sha(a.run_root/'resolved_config.json'),clock=p['clock']))
   print('D-only zero-run preflight',len(rows),'/80',spec['run_id'],flush=True)
  for name in ['controller','tracker','motion_model','recorder']:
   if len({id(getattr(a,name)) for a in held})!=80:raise ValueError('Cross-arm mutable object reuse '+name)
 if any(calls.values()):raise RuntimeError('Preflight invoked science')
 write(OUT/'D_preflight.json',dict(status='PASS',arms=rows,zero_run_guard=calls,binding_sha256=sha(BIND)))


def child(run_id,ticket):
 b=validate(run_id);d=read(STATE);a=d['attempts'][-1]
 if a['run_id']!=run_id or a['status']!='CLAIMED' or a['parent_pid']!=os.getppid() or a['ticket']!=ticket or d['stopped']:raise PermissionError('One-use live parent claim required')
 spec=next(s for s in b['arms'] if s['run_id']==run_id)
 if sha(spec['db_path'])!=b['data_sha256'][spec['db_path']]:raise ValueError('Current D source bytes changed')
 resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 import tools.tsb_direct_coast_f2_production as p
 arm=p.build_f2_arm(spec,SCIENCE/run_id);p.validate_f2_production(arm)
 previous=next(r for r in read(OUT/'D_preflight.json')['arms'] if r['run_id']==run_id);before=read(previous['config_path']);after=read(arm.run_root/'resolved_config.json')
 if p.normalized_config(before,Path(previous['config_path']).parent,before['experiment_uid'])!=p.normalized_config(after,arm.run_root,after['experiment_uid']):raise ValueError('D preflight/actual scientific factory drift')
 try:p.lifecycle_f2(arm)
 except RuntimeError as error:
  if str(error)!='Scientific arm gate FAIL; STOP no retry or replacement' or not (arm.run_root/'arm_science_review.json').exists():raise
  if read(arm.run_root/'arm_science_review.json')['arm_pass']:raise
 sink=arm.run_root/'observation_review';(sink/'evidence').mkdir(parents=True);p.OUT=sink;audit=p.audit_f2_observations(arm)
 if audit['status']!='PASS':raise ValueError('Actual observation audit failure')
 write(arm.run_root/'actual_observation_audit.json',audit)
 manifest=read(arm.run_root/'execution_manifest.json');manifest['artifacts']['actual_observation_audit.json']=sha(arm.run_root/'actual_observation_audit.json');write(arm.run_root/'execution_manifest.json',manifest)
 review=read(arm.run_root/'arm_science_review.json');write(arm.run_root/'D_completion.json',dict(run_id=run_id,technical_pass=True,scientific_pass=review['arm_pass'],binding_sha256=sha(BIND),manifest_sha256=sha(arm.run_root/'execution_manifest.json')))
 return 0 if review['arm_pass'] else 20


def supervise(run_id,ticket,remaining):
 import psutil
 start=time.monotonic();observed={};reason=None;usage=None
 with (SCIENCE/(run_id+'.process.log')).open('x') as f:
  p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',run_id,ticket],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
  while True:
   try:
    parent=psutil.Process(p.pid);members=[parent]+parent.children(recursive=True)
    for member in members:
     try:c=member.cpu_times();key=(member.pid,member.create_time());observed[key]=max(observed.get(key,0),c.user+c.system)
     except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
   except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
   pid,status,usage=os.wait4(p.pid,os.WNOHANG)
   if pid:p.returncode=os.waitstatus_to_exitcode(status);break
   if time.monotonic()-start>=1200 or sum(observed.values())>=remaining:
    reason='WALL_OR_CPU_BUDGET_STOP';os.killpg(p.pid,signal.SIGKILL);_,status,usage=os.wait4(p.pid,0);p.returncode=os.waitstatus_to_exitcode(status);break
   time.sleep(.05)
 for member in locals().get('members',[]):
  if member.pid==p.pid:continue
  try:
   if member.is_running() and member.status()!=psutil.STATUS_ZOMBIE:reason='LIVE_DESCENDANT_CPU_UNCERTAIN';os.killpg(p.pid,signal.SIGKILL)
  except psutil.NoSuchProcess:pass
 wall=time.monotonic()-start;cpu=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
 return dict(returncode=p.returncode,wall_seconds=wall,cpu_seconds=cpu,reason=reason,wait4_cpu=usage.ru_utime+usage.ru_stime,observed_pid_cpu={str(k):v for k,v in observed.items()},CPU_conservative_wall_upper_seconds=wall*(os.cpu_count() or 1))


def execute():
 b=validate(all_data=True);p=read(OUT/'D_preflight.json')
 if p['status']!='PASS' or p['binding_sha256']!=sha(BIND):raise PermissionError('Complete frozen80arm zero-run preflight required')
 if STATE.exists() or SCIENCE.exists():raise PermissionError('No technical retry or ledger reset; existing scientific identity retained')
 SCIENCE.mkdir(parents=True);write(STATE,dict(stage=b['stage'],binding_sha256=sha(BIND),stopped=False,attempts=[],pairs=[],cpu_seconds=0.,cpu_wall_upper_seconds=0.))
 driver_cpu=time.process_time();started=time.monotonic()
 for i,run_id in enumerate(b['run_ids']):
  d=read(STATE);consumed=d['cpu_seconds']+time.process_time()-driver_cpu
  if consumed>=14400:d.update(stopped=True,stop_reason='CPU_BUDGET_STOP');write(STATE,d);break
  if len(d['attempts'])!=i:raise PermissionError('Duplicate/out-of-order scientific run')
  ticket=secrets.token_hex(32);d['attempts'].append(dict(run_id=run_id,status='CLAIMED',ticket=ticket,parent_pid=os.getpid(),index=i+1));write(STATE,d)
  print('D science',i+1,'/80',run_id,flush=True);result=supervise(run_id,ticket,14400-consumed);root=SCIENCE/run_id;technical=False;science=False
  if result['returncode'] in [0,20] and result['reason'] is None and (root/'D_completion.json').exists():
   completion=read(root/'D_completion.json')
   if completion['binding_sha256']==sha(BIND) and completion['manifest_sha256']==sha(root/'execution_manifest.json'):
    technical=all(sha(root/name)==h for name,h in read(root/'execution_manifest.json')['artifacts'].items());science=completion['scientific_pass']
  d=read(STATE);d['cpu_seconds']+=result['cpu_seconds'];d['cpu_wall_upper_seconds']+=result['CPU_conservative_wall_upper_seconds'];d['attempts'][-1].update(**result,status='COMPLETE_PASS' if technical and science else 'SCIENTIFIC_FAIL' if technical else 'TECHNICAL_FAIL')
  if not technical:d.update(stopped=True,stop_reason='TECHNICAL_FAILURE_NO_RETRY');write(STATE,d);break
  spec=b['arms'][i]
  if spec['arm']=='TREATMENT':
   from tools.sdc_v1_execution import temporal
   from tools.tsb_coast_p1r_analysis import analyze_completed_pair
   br=SCIENCE/(spec['pair_id']+'-BASELINE');tr=root;reviews=[read(x/'arm_science_review.json') for x in [br,tr]]
   pair=dict(pair_id=spec['pair_id'],session_id=spec['session_id'],arm_scientific_pass=[x['arm_pass'] for x in reviews],temporal=temporal(br,tr))
   if all(pair['arm_scientific_pass']):pair.update(analyze_completed_pair(br,tr));pair['pair_pass']=pair['pair_pass'] and pair['temporal']['pass_gate']
   else:pair.update(pair_pass=False,F_match=dict(status='NOT_EVALUATED_ARM_GATE_FAILED'))
   write(SCIENCE/(spec['pair_id']+'-pair_review.json'),pair);d['pairs'].append(pair)
  write(STATE,d)
 else:
  d=read(STATE);d.update(stopped=True,stop_reason='FROZEN_D80_COMPLETE');write(STATE,d)
 d=read(STATE);d.update(driver_cpu_seconds=time.process_time()-driver_cpu,scientific_wall_seconds=time.monotonic()-started);write(STATE,d)
 print(d['stop_reason'],len(d['attempts']),'arms',len(d['pairs']),'pairs',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--child',nargs=2);a=p.parse_args()
 if a.child:raise SystemExit(child(*a.child))
 elif a.preflight:preflight()
 elif a.execute:
  try:execute()
  except BaseException as error:
   if STATE.exists():
    d=read(STATE);d.update(stopped=True,stop_reason='UNHANDLED_INFRASTRUCTURE_FAILURE_NO_RETRY',error=repr(error));write(STATE,d)
   raise
 else:p.error('Select preflight or execute')
