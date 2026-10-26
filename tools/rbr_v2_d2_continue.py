#!/usr/bin/env python3
"""Linked infrastructure continuation: only unattempted original IDs; no retry, replacement or budget reset."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,inspect,json,secrets,subprocess,time
from pathlib import Path
from unittest.mock import patch
import tools.rbr_v2_d2_execute as original
from tools.rbr_v2_d2_download import ROOT,OUT,read,write,sha
AUTH=OUT/'D_continuation_authorization.json';STATE=OUT/'science_continuation_ledger.json';SCIENCE=original.SCIENCE
CONTRACT=ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json'


def validate(run_id=None):
 a=read(AUTH);old=read(OUT/'science_ledger.json')
 if sha(OUT/'science_ledger.json')!=a['prior_stopped_ledger_sha256'] or not old['stopped']:raise PermissionError('Original failure ledger must remain stopped and byte-identical')
 b=original.validate(run_id)
 if a['remaining_run_ids']!=b['run_ids'][len(old['attempts']):] or len(old['attempts'])+len(a['remaining_run_ids'])!=80:raise PermissionError('No identity replacement or extra arm budget')
 if run_id is not None and run_id not in a['remaining_run_ids']:raise PermissionError('Failed or attempted IDs cannot be retried')
 if sha(AUTH)!=__import__('hashlib').sha256(subprocess.check_output(['git','show','HEAD:'+str(AUTH.relative_to(ROOT))],cwd=ROOT)).hexdigest():raise PermissionError('Continuation must be committed before unused arm execution')
 for p,h in a['source_sha256'].items():
  if sha(ROOT/p)!=h:raise PermissionError('Committed continuation source changed '+p)
 if sha(CONTRACT)!=a['approved_contract_sha256']:raise PermissionError('Approved F2 science contract changed')
 return a,b,old


def invoke_frozen_lifecycle(lifecycle,arm,spec,contract):
 inspect.signature(lifecycle).bind(arm,spec,contract)
 return lifecycle(arm,spec,contract)


def child(run_id,ticket):
 a,b,old=validate(run_id);d=read(STATE)
 if d['stopped'] or d['attempts'][-1]['run_id']!=run_id:raise PermissionError('No live continuation claim')
 spec=next(s for s in b['arms'] if s['run_id']==run_id)
 import tools.tsb_direct_coast_f2_production as p
 lifecycle=p.lifecycle_f2
 # Exact existing SDC call signature. No controller/model/mechanism/config replacement.
 inspect.signature(lifecycle).bind(None,spec,CONTRACT)
 def dispatch(arm):return invoke_frozen_lifecycle(lifecycle,arm,spec,CONTRACT)
 with patch.object(original,'STATE',STATE),patch.object(p,'lifecycle_f2',dispatch):code=original.child(run_id,ticket)
 write(SCIENCE/run_id/'continuation_completion.json',dict(run_id=run_id,continuation_authorization_sha256=sha(AUTH),original_binding_sha256=sha(original.BIND),attempt_is_first_for_this_run_id=True))
 return code


def execute():
 a,b,prior=validate();priorCPU=prior['cpu_seconds']+prior.get('driver_cpu_seconds',0)
 if STATE.exists():raise PermissionError('Continuation identity already started; no resets')
 write(STATE,dict(stage='RBR_V2_D2_ORIGINAL_REMAINING_FIRST_ARMS',original_stopped_ledger_sha256=sha(OUT/'science_ledger.json'),authorization_sha256=sha(AUTH),
                  stopped=False,attempts=[],pairs=[],cpu_seconds=0.,prior_CPU_seconds=priorCPU,total_arm_budget_including_prior=80,failed_prior_run_ids=[r['run_id'] for r in prior['attempts']]))
 driver=time.process_time();started=time.monotonic()
 for i,run_id in enumerate(a['remaining_run_ids']):
  d=read(STATE);used=priorCPU+d['cpu_seconds']+time.process_time()-driver
  if used>=14400:d.update(stopped=True,stop_reason='CUMULATIVE_CPU_BUDGET_STOP');write(STATE,d);break
  attempted={r['run_id'] for r in prior['attempts']}|{r['run_id'] for r in d['attempts']}
  if run_id in attempted or len(attempted)>=80:raise PermissionError('Duplicate technical retry or extra identity prohibited')
  ticket=secrets.token_hex(32);d['attempts'].append(dict(run_id=run_id,status='CLAIMED',ticket=ticket,parent_pid=os.getpid(),original_ordinal=b['run_ids'].index(run_id)+1));write(STATE,d)
  print('Remaining first execution',i+1,'/79',run_id,flush=True)
  with patch.object(original,'__file__',__file__):result=original.supervise(run_id,ticket,14400-used)
  root=SCIENCE/run_id;technical=False;science=False
  if result['returncode'] in [0,20] and result['reason'] is None and (root/'D_completion.json').exists():
   c=read(root/'D_completion.json');technical=c['binding_sha256']==sha(original.BIND) and c['manifest_sha256']==sha(root/'execution_manifest.json') and all(sha(root/p)==h for p,h in read(root/'execution_manifest.json')['artifacts'].items());science=c['scientific_pass']
  d=read(STATE);d['cpu_seconds']+=result['cpu_seconds'];d['attempts'][-1].update(**result,status='COMPLETE_PASS' if technical and science else 'SCIENTIFIC_FAIL' if technical else 'TECHNICAL_FAIL')
  if not technical:d.update(stopped=True,stop_reason='CONTINUATION_TECHNICAL_FAILURE_NO_RETRY');write(STATE,d);break
  spec=next(s for s in b['arms'] if s['run_id']==run_id)
  if spec['arm']=='TREATMENT':
   br=SCIENCE/(spec['pair_id']+'-BASELINE');tr=root
   if (br/'D_completion.json').exists():
    from tools.sdc_v1_execution import temporal
    from tools.tsb_coast_p1r_analysis import analyze_completed_pair
    reviews=[read(x/'arm_science_review.json') for x in [br,tr]];pair=dict(pair_id=spec['pair_id'],session_id=spec['session_id'],arm_scientific_pass=[r['arm_pass'] for r in reviews],temporal=temporal(br,tr))
    if all(pair['arm_scientific_pass']):pair.update(analyze_completed_pair(br,tr));pair['pair_pass']=pair['pair_pass'] and pair['temporal']['pass_gate']
    else:pair.update(pair_pass=False,F_match=dict(status='NOT_EVALUATED_ARM_GATE_FAILED'))
   else:pair=dict(pair_id=spec['pair_id'],session_id=spec['session_id'],pair_pass=False,status='INCOMPLETE_PRIOR_BASELINE_TECHNICAL_FAILURE_NO_REPLACEMENT',F_match=dict(status='NOT_EVALUATED_MISSING_BASELINE'))
   write(SCIENCE/(spec['pair_id']+'-pair_review.json'),pair);d['pairs'].append(pair)
  write(STATE,d)
 else:d=read(STATE);d.update(stopped=True,stop_reason='ALL_REMAINING79_FIRST_ARMS_COMPLETE');write(STATE,d)
 d=read(STATE);d.update(driver_cpu_seconds=time.process_time()-driver,wall_seconds=time.monotonic()-started,total_attempts_including_prior=len(prior['attempts'])+len(d['attempts']),retry_count=0);write(STATE,d)
 print(d['stop_reason'],d['total_attempts_including_prior'],'total first attempts',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--child',nargs=2);a=p.parse_args()
 if a.child:raise SystemExit(child(*a.child))
 elif a.execute:
  try:execute()
  except BaseException as e:
   if STATE.exists():d=read(STATE);d.update(stopped=True,stop_reason='CONTINUATION_INFRASTRUCTURE_STOP',error=repr(e));write(STATE,d)
   raise
 else:p.error('Select execute or child')
