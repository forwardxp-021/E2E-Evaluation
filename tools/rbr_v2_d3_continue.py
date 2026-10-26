#!/usr/bin/env python3
"""D3 infrastructure-only continuation of exact54 unattempted identities; no rerun, scientific change or budget reset."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,hashlib,secrets,subprocess,time
from unittest.mock import patch
from tools.rbr_v2_d2_download import ROOT,read,write,sha
import tools.rbr_v2_d3_execute as original
OUT=original.OUT;SCIENCE=original.SCIENCE/'SCIENCE';STATE=OUT/'SCIENCE_continuation_ledger.json';AUTH=original.DOC/'SCIENCE_Infrastructure_Continuation_v1.json'


def validate(run_id=None):
 a=read(AUTH);c=original.validate('SCIENCE',run_id);old=read(original.state('SCIENCE'))
 if not old['stopped'] or sha(original.state('SCIENCE'))!=a['original_stopped_ledger_sha256']:raise PermissionError('Original stopped ledger must remain byte-identical')
 if a['remaining_run_ids']!=c['run_ids'][len(old['attempts']):] or len(old['attempts'])+len(a['remaining_run_ids'])!=80:raise PermissionError('No identity change or extra first arm')
 if run_id is not None and run_id not in a['remaining_run_ids']:raise PermissionError('Previously attempted identity refused; no scientific retries')
 if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(AUTH.relative_to(ROOT))],cwd=ROOT)).hexdigest()!=sha(AUTH):raise PermissionError('Committed infrastructure continuation required')
 for p,h in a['source_sha256'].items():
  if sha(ROOT/p)!=h:raise PermissionError('Continuation source drift '+p)
 if sha(OUT/'D13_existing_artifact_recovery.json')!=a['recovery_sha256']:raise PermissionError('Recovery proof changed')
 return a,c,old


def child(rid,ticket):
 validate(rid)
 old_state=original.state
 with patch.object(original,'state',lambda phase:STATE if phase=='SCIENCE' else old_state(phase)):return original.child(rid,ticket)


def execute():
 a,c,old=validate()
 if STATE.exists():raise PermissionError('Continuation already claimed; no reset')
 write(STATE,dict(stage='D3_REMAINING54_FIRST_ARMS_INFRASTRUCTURE_ONLY',stopped=False,attempts=[],pairs=[],authorization_sha256=sha(AUTH),original_stopped_ledger_sha256=sha(original.state('SCIENCE')),
       prior_measured_CPU_seconds=old['cpu_seconds']+old['driver_cpu_seconds'],prior_conservatively_charged_CPU_seconds=a['prior_CPU_budget_charge_seconds'],cpu_seconds=0.,cpu_seconds_budget_charged=0.,retry_count=0,total_scientific_attempt_cap=80))
 from tools.rbr_v2_d3_supervision import supervise
 driver=time.process_time();started=time.monotonic()
 for i,rid in enumerate(a['remaining_run_ids']):
  d=read(STATE);used=a['prior_CPU_budget_charge_seconds']+d['cpu_seconds_budget_charged']+time.process_time()-driver
  if used>=14400:d.update(stopped=True,stop_reason='CUMULATIVE_CONSERVATIVE_CPU_BUDGET_STOP');write(STATE,d);break
  ticket=secrets.token_hex(32);d['attempts'].append(dict(run_id=rid,status='CLAIMED',parent_pid=os.getpid(),ticket=ticket,original_ordinal=c['run_ids'].index(rid)+1));write(STATE,d)
  print('D3 remaining first arm',i+1,'/54',rid,flush=True)
  r=supervise([sys.executable,'-B',str(__file__),'--child',rid,ticket],SCIENCE/(rid+'.process.log'),14400-used);root=SCIENCE/rid;technical=False;science=False
  if r['returncode'] in [0,20] and r['reason'] is None and (root/'completion.json').exists():
   q=read(root/'completion.json');technical=q['contract_sha256']==sha(original.contract('SCIENCE')) and q['manifest_sha256']==sha(root/'execution_manifest.json') and all(sha(root/p)==h for p,h in read(root/'execution_manifest.json')['artifacts'].items());science=q['scientific_pass']
  d=read(STATE);d['cpu_seconds']+=r['cpu_seconds_measured'];d['cpu_seconds_budget_charged']+=r['cpu_seconds_budget_charged'];d['attempts'][-1].update(**r,status='COMPLETE_PASS' if technical and science else 'SCIENTIFIC_FAIL' if technical else 'TECHNICAL_FAIL')
  if not technical:d.update(stopped=True,stop_reason='UNRESOLVED_INTEGRITY_OR_INFRASTRUCTURE_STOP_NO_RETRY');write(STATE,d);break
  spec=next(s for s in c['arms'] if s['run_id']==rid)
  if spec['arm']=='TREATMENT':
   from tools.sdc_v1_execution import temporal
   from tools.tsb_coast_p1r_analysis import analyze_completed_pair
   br=SCIENCE/(spec['pair_id']+'-BASELINE');reports=[read(x/'arm_science_review.json') for x in [br,root]];pair=dict(pair_id=spec['pair_id'],session_id=spec['session_id'],temporal=temporal(br,root),arm_pass=[r['arm_pass'] for r in reports])
   if all(pair['arm_pass']):pair.update(analyze_completed_pair(br,root));pair['pair_pass']=pair['pair_pass'] and pair['temporal']['pass_gate']
   else:pair.update(pair_pass=False,F_match=dict(status='NOT_EVALUATED_ARM_GATE_FAILED'))
   write(SCIENCE/(spec['pair_id']+'-pair_review.json'),pair);d['pairs'].append(pair)
  write(STATE,d)
 else:d=read(STATE);d.update(stopped=True,stop_reason='ALL_ORIGINAL80_FIRST_ATTEMPTS_COMPLETE');write(STATE,d)
 d=read(STATE);d.update(driver_cpu_seconds=time.process_time()-driver,wall_seconds=time.monotonic()-started,total_first_attempts=len(old['attempts'])+len(d['attempts']));write(STATE,d)
 print(d['stop_reason'],d['total_first_attempts'],'cumulative attempts',flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--child',nargs=2);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.child:raise SystemExit(child(*a.child))
 elif a.execute:execute()
 else:p.error('Select child or execute')
