#!/usr/bin/env python3
"""Separate D3 ENGINEERING<=12 and D SCIENCE<=80 first-arm ledgers. No D2 reset or scientific retry."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,copy,hashlib,json,resource,secrets,subprocess,time
from pathlib import Path
from unittest.mock import patch
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1';DOC=ROOT/'docs/stageR/rbr_v2_d3_v1';SCIENCE=ROOT/'outputs/stageR/rbr_v2_d3'


def contract(phase):return DOC/(phase+'_Protocol_v1.json')
def state(phase):return OUT/(phase+'_ledger.json')
def phase_for(run_id):
 if run_id.startswith('D3-ENG-'):return 'ENGINEERING'
 if run_id.startswith('D3-D-'):return 'SCIENCE'
 raise PermissionError('Only D3 ENGINEERING/D identities authorized; historical/V/C identities rejected')


def sources():
 own=list((ROOT/'tools').glob('rbr_v2_d3_*.py'))
 parent=read(ROOT/'outputs/rbr_v2_d2_v1/D_execution_binding.json')['source_sha256']
 return dict(parent,**{str(p.relative_to(ROOT)):sha(p) for p in own})


def prepare_engineering():
 if contract('ENGINEERING').exists():raise FileExistsError('ENGINEERING identities already frozen')
 DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 d=read(ROOT/'outputs/rbr_v2_d2_v1/D_execution_binding.json')['arms'];legacy=read(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/QUALIFICATION_Arm_Specs_v2.json')['arms']
 # Fixed physical diagnostic targets, chosen before new rollouts or detector output reads.
 fit=read(ROOT/'docs/stageR/e2e_scoped_direct_coast_v1/DEVELOPMENT_Arm_Specs_v1.json')['arms']
 fitids={r['run_id'] for r in read(ROOT/'configs/rbr_v2_d1_v1/contract.json')['runs'][:2]};fit=[s for s in fit if s['run_id'] in fitids]
 targets=[d[:2],legacy[16:18],legacy[30:32],fit,d[2:4],d[4:6]]
 if any(len(t)!=2 for t in targets):raise ValueError('Exact six engineering pairs required')
 arms=[]
 for i,pair in enumerate(targets,1):
  for s in pair:
   a=copy.deepcopy(s);a.update(run_id=f'D3-ENG-{i:02d}-'+s['arm'],pair_id=f'D3-ENG-{i:02d}',benchmark_version='SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1',prior_source_run_id=s['run_id'],D3_phase='ENGINEERING');arms.append(a)
 c=dict(stage='RBR_V2_D3_ENGINEERING',phase='ENGINEERING',owner_task_scope='D3 autonomous bounded compatibility repair; never old D2 retry',arms=arms,run_ids=[a['run_id'] for a in arms],max_first_arms=12,
        per_arm_wall_seconds=1200,engineering_CPU_ceiling_seconds=14400,source_sha256=sources(),data_sha256={s['db_path']:sha(s['db_path']) for s in arms},
        gate='Root D01 pair and four affected legacy arms complete; unchanged control/model/measurement/safety; all attempts retained; numerical encoder/detection not used to select repair',
        inherited_F2_contract='docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json',inherited_contract_sha256=sha(ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json'),
        previous_D2_ledgers={p:sha(ROOT/p) for p in ['outputs/rbr_v2_d2_v1/science_ledger.json','outputs/rbr_v2_d2_v1/science_continuation_ledger.json']},download_authorized_bytes=0,Primary_V_C=False)
 write(contract('ENGINEERING'),c);print('Frozen engineering12 identities:',[a['run_id'] for a in arms],flush=True)


def validate(phase,run_id=None):
 c=read(contract(phase))
 committed=subprocess.check_output(['git','show','HEAD:'+str(contract(phase).relative_to(ROOT))],cwd=ROOT)
 if hashlib.sha256(committed).hexdigest()!=sha(contract(phase)):raise PermissionError('Commit exact D3 phase contract before first optimizer-free scientific step')
 if run_id is not None and (phase_for(run_id)!=phase or run_id not in c['run_ids']):raise PermissionError('Wrong phase/run identity')
 if len(c['run_ids'])!=len(set(c['run_ids'])) or len(c['run_ids'])> (12 if phase=='ENGINEERING' else 80):raise PermissionError('First-arm cap exceeded')
 for name,h in c['source_sha256'].items():
  if sha(ROOT/name)!=h:raise PermissionError('Frozen phase source changed '+name)
 for name,h in c['previous_D2_ledgers'].items():
  if sha(ROOT/name)!=h or not read(ROOT/name)['stopped']:raise PermissionError('D2 stopped ledger mutation')
 for name,h in c.get('proof_sha256',{}).items():
  if sha(ROOT/name)!=h:raise PermissionError('Compatibility evidence changed '+name)
 return c


def child(run_id,ticket):
 phase=phase_for(run_id);c=validate(phase,run_id);ledger=read(state(phase));a=ledger['attempts'][-1]
 if ledger['stopped'] or a['run_id']!=run_id or a['status']!='CLAIMED' or a['parent_pid']!=os.getppid() or not secrets.compare_digest(a['ticket'],ticket):raise PermissionError('Live one-use parent claim required')
 spec=next(s for s in c['arms'] if s['run_id']==run_id)
 if sha(spec['db_path'])!=c['data_sha256'][spec['db_path']]:raise ValueError('Source DB changed')
 resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.rbr_v2_d3_production import build,validate as production_check,lifecycle
 arm=build(spec,SCIENCE/phase/run_id);production_check(arm)
 from tools.tsb_direct_coast_f2_production import normalized_config
 before_root=OUT/(phase+'_preflight')/run_id;before=read(before_root/'resolved_config.json');after=read(arm.run_root/'resolved_config.json')
 record=next(r for r in read(OUT/(phase+'_preflight.json'))['rows'] if r['run_id']==run_id)
 if sha(before_root/'resolved_config.json')!=record['config_sha256']:raise ValueError('D3 preflight config bytes changed')
 if normalized_config(before,before_root,before['experiment_uid'])!=normalized_config(after,arm.run_root,after['experiment_uid']):raise ValueError('D3 preflight/production config drift')
 if read(before_root/'canonical_clock_mapping.json')!=read(arm.run_root/'canonical_clock_mapping.json'):raise ValueError('D3 preflight/production clock drift')
 review=lifecycle(arm,spec,contract(phase))
 import tools.tsb_direct_coast_f2_production as p
 sink=arm.run_root/'observation_review';(sink/'evidence').mkdir(parents=True)
 with patch.object(p,'OUT',sink):audit=p.audit_f2_observations(arm)
 if audit['status']!='PASS':raise ValueError('Actual observation identity audit failure')
 write(arm.run_root/'actual_observation_audit.json',audit)
 m=read(arm.run_root/'execution_manifest.json');m['artifacts']['actual_observation_audit.json']=sha(arm.run_root/'actual_observation_audit.json');write(arm.run_root/'execution_manifest.json',m)
 write(arm.run_root/'completion.json',dict(run_id=run_id,phase=phase,technical_pass=True,scientific_pass=review['arm_pass'],contract_sha256=sha(contract(phase)),manifest_sha256=sha(arm.run_root/'execution_manifest.json')))
 return 0 if review['arm_pass'] else 20


def execute(phase):
 c=validate(phase)
 proof=OUT/(phase+'_preflight.json');p=read(proof)
 if p['status']!='PASS' or p['contract_sha256']!=sha(contract(phase)) or [r['run_id'] for r in p['rows']]!=c['run_ids'] or any(p['guard_calls'].values()):raise PermissionError('Complete exact phase preflight required')
 if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(proof.relative_to(ROOT))],cwd=ROOT)).hexdigest()!=sha(proof):raise PermissionError('Preflight evidence must be committed')
 if state(phase).exists() or (SCIENCE/phase).exists():raise PermissionError('Existing phase identity cannot be reset or silently resumed')
 (SCIENCE/phase).mkdir(parents=True);write(state(phase),dict(stage=c['stage'],stopped=False,attempts=[],pairs=[],cpu_seconds=0.,contract_sha256=sha(contract(phase))))
 started=time.monotonic();driver=time.process_time();limit=14400
 import tools.rbr_v2_d2_execute as transport
 for i,spec in enumerate(c['arms']):
  d=read(state(phase));used=d['cpu_seconds']+time.process_time()-driver
  if used>=limit:d.update(stopped=True,stop_reason='CPU_BUDGET_STOP');write(state(phase),d);break
  rid=spec['run_id'];ticket=secrets.token_hex(32);d['attempts'].append(dict(run_id=rid,status='CLAIMED',parent_pid=os.getpid(),ticket=ticket,ordinal=i+1));write(state(phase),d)
  print(phase,'first arm',i+1,'/',len(c['arms']),rid,flush=True)
  with patch.object(transport,'__file__',__file__),patch.object(transport,'SCIENCE',SCIENCE/phase):result=transport.supervise(rid,ticket,limit-used)
  root=SCIENCE/phase/rid;technical=False;science=False
  if result['returncode'] in [0,20] and result['reason'] is None and (root/'completion.json').exists():
   q=read(root/'completion.json');technical=q['contract_sha256']==sha(contract(phase)) and q['manifest_sha256']==sha(root/'execution_manifest.json') and all(sha(root/p)==h for p,h in read(root/'execution_manifest.json')['artifacts'].items());science=q['scientific_pass']
  d=read(state(phase));d['cpu_seconds']+=result['cpu_seconds'];d['attempts'][-1].update(**result,status='COMPLETE_PASS' if technical and science else 'SCIENTIFIC_FAIL' if technical else 'TECHNICAL_FAIL')
  if not technical:d.update(stopped=True,stop_reason='INFRASTRUCTURE_STOP_NO_SCIENTIFIC_RETRY');write(state(phase),d);break
  if spec['arm']=='TREATMENT':
   from tools.sdc_v1_execution import temporal
   from tools.tsb_coast_p1r_analysis import analyze_completed_pair
   br=SCIENCE/phase/(spec['pair_id']+'-BASELINE');tr=root
   reports=[read(x/'arm_science_review.json') for x in [br,tr]];pair=dict(pair_id=spec['pair_id'],session_id=spec['session_id'],temporal=temporal(br,tr),arm_pass=[r['arm_pass'] for r in reports])
   if all(pair['arm_pass']):pair.update(analyze_completed_pair(br,tr));pair['pair_pass']=pair['pair_pass'] and pair['temporal']['pass_gate']
   else:pair.update(pair_pass=False,F_match=dict(status='NOT_EVALUATED_ARM_GATE_FAILED'))
   write(SCIENCE/phase/(spec['pair_id']+'-pair_review.json'),pair);d['pairs'].append(pair)
  write(state(phase),d)
 else:d=read(state(phase));d.update(stopped=True,stop_reason='ALL_FROZEN_FIRST_ARMS_COMPLETE');write(state(phase),d)
 d=read(state(phase));d.update(driver_cpu_seconds=time.process_time()-driver,wall_seconds=time.monotonic()-started);write(state(phase),d)
 print(d['stop_reason'],len(d['attempts']),'arms',len(d['pairs']),'pairs',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-engineering',action='store_true');p.add_argument('--execute',choices=['ENGINEERING','SCIENCE']);p.add_argument('--child',nargs=2);a=p.parse_args()
 if a.child:raise SystemExit(child(*a.child))
 elif a.prepare_engineering:prepare_engineering()
 elif a.execute:
  try:execute(a.execute)
  except BaseException as error:
   if state(a.execute).exists():d=read(state(a.execute));d.update(stopped=True,stop_reason='PRESERVED_INFRASTRUCTURE_FAILURE',error=repr(error));write(state(a.execute),d)
   raise
 else:p.error('Choose prepare-engineering, execute, or child')
