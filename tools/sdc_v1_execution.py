#!/usr/bin/env python3
"""Independent scoped benchmark: fresh process, global budget, fixed full denominator."""
from __future__ import annotations
import argparse,importlib.metadata,json,os,resource,secrets,signal,subprocess,sys,time,math
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tsb_direct_coast_p2_q1 import ROOT,sha,canon,read,write,locked
OUT=ROOT/'docs/stageR/e2e_scoped_direct_coast_v1'
STATE=ROOT/'outputs/stageR/e2e_sdc_v1_state/budget.json'
SCIENCE=ROOT/'outputs/stageR/e2e_sdc_v1'
BUDGET=dict(development_attempts=60,qualification_attempts=40,total_cpu_seconds=10800,per_arm_wall_seconds=1200,qualification_retry=0)


def binding_path(phase):return OUT/(phase+'_Execution_Binding_v1.json')


def validate(phase,run_id):
    auth=read(binding_path(phase))
    if auth.get('stage')!='E2E_SCOPED_DIRECT_COAST_V1' or auth.get('phase')!=phase or auth.get('owner_approved') is not True or auth.get('budget')!=BUDGET or run_id not in auth['run_ids']:
        raise PermissionError('Independent SDC exact phase/identity/Owner scope required; historical credentials rejected')
    committed=subprocess.check_output(['git','show','HEAD:'+str(binding_path(phase).relative_to(ROOT))],cwd=ROOT)
    if __import__('hashlib').sha256(committed).hexdigest()!=sha(binding_path(phase)):raise PermissionError('Uncommitted phase freeze')
    for group in ['source_sha256','proof_sha256','data_sha256']:
        for p,h in auth[group].items():
            if sha(ROOT/p)!=h:raise PermissionError('Frozen '+group+' changed: '+p)
    source=read(ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Execution_Source_Binding_v1.json')
    for p,h in source['source_config_sha256'].items():
        if sha(ROOT/p)!=h:raise PermissionError('Frozen F2 scientific source changed: '+p)
    runtime=source['runtime']
    if sys.executable!=runtime['scientific_runtime_executable'] or sys.version.split()[0]!=runtime['scientific_runtime_python']:raise PermissionError('Wrong scientific Python')
    for name,version in runtime['dependency_versions'].items():
        if importlib.metadata.version(name)!=version:raise PermissionError('Dependency changed: '+name)
    spec=next(x for x in read(OUT/(phase+'_Arm_Specs_v1.json'))['arms'] if x['run_id']==run_id)
    if canon(spec)!=auth['spec_sha256'][run_id]:raise PermissionError('Spec identity changed')
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env();return auth,spec


def preflight(phase):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    from tools.b1_production_preflight import zero_run_guard
    from tools.tsb_direct_coast_f2_production import build_f2_arm,validate_f2_production
    arms=read(OUT/(phase+'_Arm_Specs_v1.json'))['arms'];proof=[];held=[]
    with zero_run_guard() as calls:
        for spec in arms:
            a=build_f2_arm(spec,ROOT/'outputs/stageR/e2e_sdc_v1_preflight_r2'/phase/spec['run_id']);held.append(a)
            p=validate_f2_production(a)
            expected=read(OUT/read(OUT/(phase+'_Arm_Specs_v1.json'))['applicability_path'])
            static=next(x for x in expected['rows'] if x['session_id']==spec['session_id'])
            if {k:v for k,v in p['clock'].items() if k not in ['observer','shared_original_scenario']}!=static['evidence']['clock']:raise ValueError('Static/official factory mapping mismatch')
            proof.append(dict(run_id=spec['run_id'],status='PASS',clock=p['clock'],config_path=str(a.run_root/'resolved_config.json'),config_sha256=sha(a.run_root/'resolved_config.json')))
            print('preflight',phase,len(proof),'/',len(arms),flush=True)
        if any(calls.values()):raise RuntimeError('Zero-run guard violation')
        for name in ['runner','planner','tracker','motion_model','recorder']:
            if len({id(getattr(a,name)) for a in held})!=len(held):raise ValueError('Cross-arm object reuse')
    write(OUT/'evidence'/(phase+'_Production_Preflight_v1.json'),dict(status='PASS',arms=proof,guard_calls=calls),exclusive=True)


def start_phase(phase):
    auth=read(binding_path(phase));validate(phase,auth['run_ids'][0])
    if not STATE.exists():
        if phase!='DEVELOPMENT' or STATE.parent.exists() or SCIENCE.exists():raise PermissionError('Fresh global development ledger required')
        STATE.parent.mkdir(parents=True)
        write(STATE,dict(stage='E2E_SCOPED_DIRECT_COAST_V1',budget=BUDGET,cpu_seconds=0.,cpu_certain=True,stopped=False,attempts=[],pairs=[],phases={}),exclusive=True)
    with locked(STATE):
        d=read(STATE)
        if d['stopped'] or not d['cpu_certain'] or phase in d['phases']:raise PermissionError('Phase already started/interrupted/stopped; no implicit resume')
        if phase=='QUALIFICATION' and d['phases'].get('DEVELOPMENT',{}).get('status')!='COMPLETE':raise PermissionError('Development must be complete')
        d['phases'][phase]=dict(status='RUNNING',binding_sha256=sha(binding_path(phase)),run_ids=auth['run_ids']);write(STATE,d)


def claim(phase,run_id):
    with locked(STATE):
        d=read(STATE);rows=[x for x in d['attempts'] if x['phase']==phase];p=d['phases'][phase]
        if d['stopped'] or not d['cpu_certain'] or p['status']!='RUNNING' or d['cpu_seconds']>=10800:raise PermissionError('Stopped or exhausted budget')
        if p['binding_sha256']!=sha(binding_path(phase)) or len(rows)>=BUDGET['development_attempts' if phase=='DEVELOPMENT' else 'qualification_attempts'] or run_id!=p['run_ids'][len(rows)]:raise PermissionError('Duplicate/out-of-order/unapproved attempt')
        if rows and rows[-1]['status'] not in ['COMPLETE_PASS','SCIENTIFIC_FAILURE']:raise PermissionError('Previous attempt incomplete; fail closed')
        if (SCIENCE/phase/run_id).exists() or (SCIENCE/phase/(run_id+'.process.log')).exists():raise FileExistsError('Existing scientific namespace')
        ticket=secrets.token_hex(32);d['attempts'].append(dict(phase=phase,run_id=run_id,status='CLAIMED_BEFORE_PROCESS_START',token=ticket,parent_pid=os.getpid(),claim_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));write(STATE,d);return ticket


def child(phase,run_id,ticket):
    auth,spec=validate(phase,run_id)
    with locked(STATE):
        d=read(STATE);a=d['attempts'][-1]
        if d['stopped'] or a['run_id']!=run_id or a['phase']!=phase or a['status']!='CLAIMED_BEFORE_PROCESS_START' or a['parent_pid']!=os.getppid() or not secrets.compare_digest(ticket,a['token']):raise PermissionError('Child requires live one-time parent claim')
        a.update(status='CHILD_RUNNING',child_pid=os.getpid());write(STATE,d)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    import tools.tsb_direct_coast_f2_production as p
    arm=p.build_f2_arm(spec,SCIENCE/phase/run_id);p.validate_f2_production(arm)
    old=next(x for x in read(OUT/'evidence'/(phase+'_Production_Preflight_v1.json'))['arms'] if x['run_id']==run_id)
    before=read(old['config_path']);after=read(arm.run_root/'resolved_config.json')
    if p.normalized_config(before,Path(old['config_path']).parent,before['experiment_uid'])!=p.normalized_config(after,arm.run_root,after['experiment_uid']):raise ValueError('Preflight/production configuration mismatch')
    if read(arm.run_root/'canonical_clock_mapping.json')!=old['clock']:raise ValueError('Clock mapping mismatch')
    try:p.lifecycle_f2(arm,spec,ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Protocol_v1.json')
    except RuntimeError as e:
        if str(e)!='Scientific arm gate FAIL; STOP no retry or replacement':raise
        if not (arm.run_root/'arm_science_review.json').exists() or read(arm.run_root/'arm_science_review.json')['arm_pass']:raise
    sink=arm.run_root/'observation_review';(sink/'evidence').mkdir(parents=True)
    original=p.OUT
    try:p.OUT=sink;audit=p.audit_f2_observations(arm)
    finally:p.OUT=original
    if audit['status']!='PASS':raise RuntimeError('Actual observation identity failed')
    write(arm.run_root/'actual_observation_audit.json',audit,exclusive=True)
    manifest=read(arm.run_root/'execution_manifest.json');manifest['artifacts']['actual_observation_audit.json']=sha(arm.run_root/'actual_observation_audit.json');write(arm.run_root/'execution_manifest.json',manifest)
    review=read(arm.run_root/'arm_science_review.json')
    write(arm.run_root/'completion.json',dict(run_id=run_id,phase=phase,technical_pass=True,arm_scientific_pass=review['arm_pass'],spec_sha256=canon(spec),binding_sha256=sha(binding_path(phase)),manifest_sha256=sha(arm.run_root/'execution_manifest.json')),exclusive=True)
    return 0 if review['arm_pass'] else 20


def supervise(phase,run_id,ticket,remaining):
    import psutil
    start=time.monotonic();process=None;usage=None;reason=None;certain=True;code=1;consumed=0.;observed={};active=set()
    try:
        rootdir=SCIENCE/phase;rootdir.mkdir(parents=True,exist_ok=True)
        with (rootdir/(run_id+'.process.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',phase,run_id,ticket],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            root=psutil.Process(process.pid)
            while True:
                try:
                    children=root.children(recursive=True);active={p.pid for p in children if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE}
                    for p in [root,*children]:
                        try:
                            c=p.cpu_times();key=(p.pid,p.create_time());observed[key]=max(observed.get(key,0),c.user+c.system)
                        except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                pid,status,usage=os.wait4(process.pid,os.WNOHANG)
                if pid:process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode;break
                if time.monotonic()-start>=1200 or sum(observed.values())>=remaining:
                    reason='WALL_OR_GLOBAL_CPU_LIMIT';os.killpg(process.pid,signal.SIGKILL);_,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode;break
                time.sleep(.05)
            consumed=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
            for pid in active:
                try:
                    p=psutil.Process(pid)
                    if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:certain=False;reason='LIVE_DESCENDANT_CPU_UNCERTAIN'
                except psutil.NoSuchProcess:pass
            if not certain:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
    except BaseException as e:
        certain=False;reason=repr(e)
        if process is not None and process.returncode is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:
                _,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode;consumed=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
            except ChildProcessError:pass
    if consumed>=remaining:reason='GLOBAL_CPU_EXHAUSTED'
    return dict(returncode=code,cpu_seconds=consumed,wall_seconds=time.monotonic()-start,reason=reason,cpu_certain=certain,
        cpu_accounting=dict(method='max(wait4 child+reaped descendants,sum per-PID sampled maxima)',wait4_cpu=(usage.ru_utime+usage.ru_stime) if usage else None,observed_per_pid={str(k):v for k,v in observed.items()}))


def temporal(broot,troot):
    b=[json.loads(x)['current_ego'] for x in (broot/'trace/realized_current_ego.jsonl').read_text().splitlines()];t=[json.loads(x)['current_ego'] for x in (troot/'trace/realized_current_ego.jsonl').read_text().splitlines()]
    if len(b)!=80 or len(t)!=80 or b[:11]!=t[:11] or [x['time_us'] for x in b]!=[x['time_us'] for x in t]:raise ValueError('Temporal pair identity/clock/precontext invalid')
    import numpy as np
    delta=np.abs(np.array([x['speed_mps'] for x in b])-np.array([x['speed_mps'] for x in t]));distinct=sha(broot/'trace/realized_current_ego.jsonl')!=sha(troot/'trace/realized_current_ego.jsonl')
    bu=[json.loads(x)['longitudinal_requested_acceleration'] for x in (broot/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()];tu=[json.loads(x)['longitudinal_requested_acceleration'] for x in (troot/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
    return dict(max_abs_speed_difference_mps=float(delta.max()),absolute_speed_difference_integral_m=float(np.trapz(delta,np.arange(80)/10)),actual_trace_hash_distinct=distinct,request_sequences_distinct=bu!=tu,pass_gate=distinct and bu!=tu and float(delta.max())>=.5)


def launch(phase,run_id):
    auth,spec=validate(phase,run_id);ticket=claim(phase,run_id);result=supervise(phase,run_id,ticket,10800-read(STATE)['cpu_seconds'])
    root=SCIENCE/phase/run_id;status='INFRASTRUCTURE_FAILURE'
    try:
        if result['returncode'] in [0,20] and result['reason'] is None and result['cpu_certain']:
            c=read(root/'completion.json')
            if c['spec_sha256']!=canon(spec) or c['binding_sha256']!=sha(binding_path(phase)) or not c['technical_pass'] or c['manifest_sha256']!=sha(root/'execution_manifest.json'):raise ValueError('Completion/identity mismatch')
            for p,h in read(root/'execution_manifest.json')['artifacts'].items():
                if sha(root/p)!=h:raise ValueError('Actual artifact changed')
            status='COMPLETE_PASS' if c['arm_scientific_pass'] else 'SCIENTIFIC_FAILURE'
    except Exception as e:result['reason']=repr(e)
    with locked(STATE):
        d=read(STATE);d['cpu_seconds']+=result['cpu_seconds'];d['cpu_certain']=result['cpu_certain'];d['attempts'][-1].update(status=status,**result)
        if status=='INFRASTRUCTURE_FAILURE':d['stopped']=True;d['phases'][phase]['status']='INFRASTRUCTURE_PAUSED'
        write(STATE,d)
    if status=='INFRASTRUCTURE_FAILURE':return False
    if spec['arm']=='TREATMENT':
        b=SCIENCE/phase/(spec['pair_id']+'-BASELINE');t=root;reviews=[read(p/'arm_science_review.json') for p in [b,t]]
        pair=dict(pair_id=spec['pair_id'],phase=phase,session_id=spec['session_id'],arm_scientific_pass=[x['arm_pass'] for x in reviews],temporal=temporal(b,t))
        if all(x['arm_pass'] for x in reviews):
            from tools.tsb_coast_p1r_analysis import analyze_completed_pair
            pair.update(analyze_completed_pair(b,t));pair['pair_pass']=pair['pair_pass'] and pair['temporal']['pass_gate']
        else:pair.update(F_match=dict(status='NOT_EVALUATED_ARM_GATE_FAILED'),pair_pass=False)
        write(SCIENCE/phase/(spec['pair_id']+'-pair_review.json'),pair,exclusive=True)
        with locked(STATE):
            d=read(STATE);d['pairs'].append(pair);write(STATE,d)
    return True


def execute(phase):
    start_phase(phase);auth=read(binding_path(phase))
    try:
        for index,run_id in enumerate(auth['run_ids']):
            print(phase,index+1,'/',len(auth['run_ids']),run_id,flush=True)
            if not launch(phase,run_id):return read(STATE)
        with locked(STATE):
            d=read(STATE);d['phases'][phase]['status']='COMPLETE'
            if phase=='QUALIFICATION':d['stopped']=True
            write(STATE,d)
    except BaseException as e:
        with locked(STATE):
            d=read(STATE);d.update(stopped=True,infrastructure_error=repr(e));d['phases'][phase]['status']='INFRASTRUCTURE_PAUSED';write(STATE,d)
        raise
    return read(STATE)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',choices=['DEVELOPMENT','QUALIFICATION']);p.add_argument('--execute',choices=['DEVELOPMENT','QUALIFICATION']);p.add_argument('--child',nargs=3);a=p.parse_args()
    if a.child:raise SystemExit(child(*a.child))
    elif a.preflight:preflight(a.preflight)
    elif a.execute:print(json.dumps(execute(a.execute),indent=2))
    else:p.error('Choose preflight or execute phase')
