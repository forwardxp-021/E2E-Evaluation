#!/usr/bin/env python3
"""P2-Q1 fixed-identity scheduling only; all scientific operations reuse frozen F2."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/stageR/tsb_direct_coast_p2_q1'
Q0 = ROOT/'docs/stageR/tsb_direct_coast_p2_q0'
F2 = ROOT/'docs/stageR/tsb_direct_coast_f2'
AUTH = OUT/'P2_Q1_Owner_Authorization_v1.json'
BINDING = OUT/'P2_Q1_Execution_Source_Binding_v1.json'
LEDGER = ROOT/'outputs/stageR/tsb_direct_coast_p2_execution_state_v1/budget.json'
SCIENCE = ROOT/'outputs/stageR/tsb_direct_coast_p2_qualification_v1'
OWNER_SHA = '573645e5d1781aefa5ec3b432a7ce4728cb8abba5d5bcf64f34fe7e2fee327c1'
RUNS = [f'DIRECTCOAST-P2-{i:02d}-{arm}' for i in range(1,21) for arm in ('BASELINE','TREATMENT')]
BUDGET = dict(initial_arms=40, cumulative_attempts_ceiling=42, technical_retry_authorized=0,
              cpu_seconds=57600, per_arm_wall_seconds=1200)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def canon(value):
    # Use the exact canonical encoding used by Q0/F2.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if exclusive:
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:
        tmp=path.with_name(path.name+'.atomic-'+secrets.token_hex(8))
        with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextlib.contextmanager
def locked(path):
    with Path(path).with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def specs():return read(Q0/'evidence/P2_Q0_Verified_Arm_Specs_v1.json')['arms']


def validate_scope(auth,run_id):
    if (auth.get('stage')!='TSB_DIRECT_COAST_P2_Q1' or auth.get('authorization_status')!='AUTHORIZED'
        or auth.get('scientific_owner_approved') is not True or auth.get('approved_run_ids')!=RUNS
        or run_id not in RUNS or auth.get('budget')!=BUDGET or auth.get('role')!='B_BENCHMARK_QUALIFICATION'
        or auth.get('owner_instruction_sha256')!=OWNER_SHA
        or auth.get('execution_ledger_path')!=str(LEDGER) or auth.get('scientific_output_root')!=str(SCIENCE)):
        raise PermissionError('P2-Q1 exact new Owner credential, fixed 40 first runs and zero retries required')


def verify_hashes(values):
    for path,digest in values.items():
        if sha(ROOT/path)!=digest:raise PermissionError('Frozen source/data/proof changed: '+path)


def credential(run_id,full_data=True):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    auth=read(AUTH);validate_scope(auth,run_id)
    # Trust only a separately committed credential, never an edited draft.
    committed=subprocess.check_output(['git','show','HEAD:'+str(AUTH.relative_to(ROOT))],cwd=ROOT)
    if hashlib.sha256(committed).hexdigest()!=sha(AUTH):raise PermissionError('Authorization is not independently committed')
    if sha(OUT/'P2_Q1_Owner_Instruction_Original_v1.txt')!=OWNER_SHA or sha(BINDING)!=auth['binding_sha256']:
        raise PermissionError('Owner instruction/source binding changed')
    binding=read(BINDING)
    subprocess.run(['git','merge-base','--is-ancestor',binding['scheduler_source_commit'],'HEAD'],cwd=ROOT,check=True)
    for path,digest in binding['scheduler_source_sha256'].items():
        frozen=subprocess.check_output(['git','show',binding['scheduler_source_commit']+':'+path],cwd=ROOT)
        if hashlib.sha256(frozen).hexdigest()!=digest:raise PermissionError('Scheduler source commit mismatch')
    verify_hashes(binding['scheduler_source_sha256']);verify_hashes(binding['frozen_proof_sha256'])
    verify_hashes(read(F2/'F2_Execution_Source_Binding_v1.json')['source_config_sha256'])
    if full_data:verify_hashes(binding['data_sha256'])
    runtime=read(F2/'F2_Execution_Source_Binding_v1.json')['runtime']
    if sys.executable!=runtime['scientific_runtime_executable'] or sys.version.split()[0]!=runtime['scientific_runtime_python']:
        raise PermissionError('Wrong frozen scientific Python runtime')
    for name,version in runtime['dependency_versions'].items():
        if importlib.metadata.version(name)!=version:raise PermissionError('Frozen dependency mismatch: '+name)
    arms=specs()
    if [s['run_id'] for s in arms]!=RUNS or auth['approved_spec_sha256']!={s['run_id']:canon(s) for s in arms}:
        raise PermissionError('Fixed ordered arm identities changed')
    protocol=read(F2/'F2_Owner_Protocol_v1.json')
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    from tools.tsb_direct_coast_f2_tracker import SEGMENTS
    expected=dict(SCIENCE_CONTRACT);expected['version']='F2_DIRECT_REQUEST_APPROVED_v1'
    if protocol['qualification']!=expected or protocol['request_segments_us']!={k:[list(x) for x in v] for k,v in SEGMENTS.items()}:
        raise PermissionError('Frozen F2 science contract changed')
    for name in ('P2_Q1_Zero_Run_Tests_v1.json','P2_Q1_Final_Preflight_Evidence_v1.json'):
        if read(OUT/'evidence'/name)['status']!='PASS':raise PermissionError('Q1 zero-run proof not PASS')
    return auth,next(s for s in arms if s['run_id']==run_id)


def initialize():
    credential(RUNS[0])
    # A surviving state directory, even without a budget file, is an interruption.
    if LEDGER.parent.exists() or SCIENCE.exists():raise FileExistsError('P2 namespace/state already exists; no reset or implicit resume')
    LEDGER.parent.mkdir(parents=True,exist_ok=False)
    write(LEDGER,dict(stage='TSB_DIRECT_COAST_P2_Q1',authorization_sha256=sha(AUTH),
        binding_sha256=sha(BINDING),budget=BUDGET,approved_run_ids=RUNS,attempts=[],pairs=[],
        stopped=False,status='AUTHORIZED_NOT_STARTED',cpu_seconds_consumed=0.,cpu_accounting_certain=True),exclusive=True)


def check_ledger(ledger,auth_sha):
    if (ledger['stage']!='TSB_DIRECT_COAST_P2_Q1' or ledger['authorization_sha256']!=auth_sha
        or ledger['budget']!=BUDGET or ledger['approved_run_ids']!=RUNS):raise PermissionError('Ledger provenance mismatch')
    if ledger['stopped'] or not ledger['cpu_accounting_certain'] or ledger['cpu_seconds_consumed']>=BUDGET['cpu_seconds']:
        raise PermissionError('P2 stopped, CPU uncertain or exhausted; no retry')


def claim(path,run_id,auth_sha,science_root):
    """Persistent claim occurs before Popen; incomplete previous claims fail closed."""
    with locked(path):
        ledger=read(path);check_ledger(ledger,auth_sha);index=len(ledger['attempts'])
        if index>=40 or index>=42 or run_id!=RUNS[index]:raise PermissionError('Duplicate/out-of-order/unapproved first attempt')
        if index and ledger['attempts'][-1]['status']!='COMPLETE_PASS':raise PermissionError('Previous attempt incomplete or failed; stop')
        if index and index%2==0 and (not ledger['pairs'] or ledger['pairs'][-1]['pair_id']!=RUNS[index-1].rsplit('-',1)[0] or not ledger['pairs'][-1]['pair_pass']):
            raise PermissionError('Previous pair must be formally complete and PASS')
        if (Path(science_root)/run_id).exists() or (Path(science_root)/(run_id+'.process.log')).exists():
            raise FileExistsError('Scientific output namespace is not fresh')
        ticket=secrets.token_hex(32)
        ledger['attempts'].append(dict(run_id=run_id,status='CLAIMED_BEFORE_PROCESS_START',claim_token=ticket,
            monitor_pid=os.getpid(),claim_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        ledger['status']='EXECUTING';write(path,ledger);return ticket


def finish_attempt(path,run_id,status,cpu,elapsed,code,reason,certain=True,accounting=None):
    with locked(path):
        ledger=read(path);row=ledger['attempts'][-1]
        if row['run_id']!=run_id:raise PermissionError('Finishing wrong claim')
        ledger['cpu_seconds_consumed']+=cpu;ledger['cpu_accounting_certain']=certain
        row.update(status=status,cpu_seconds=cpu,wall_seconds=elapsed,returncode=code,stop_reason=reason,cpu_accounting=accounting)
        if status!='COMPLETE_PASS' or not certain:
            ledger['stopped']=True
            ledger['status']='P2_QUALIFICATION_FAILED' if status=='SCIENTIFIC_FAILURE' else 'P2_INFRASTRUCTURE_BLOCKED'
        write(path,ledger)


def record_pair(path,pair):
    with locked(path):
        ledger=read(path)
        if ledger['stopped'] or not ledger['attempts'][-1]['run_id'].endswith('TREATMENT') or ledger['attempts'][-1]['status']!='COMPLETE_PASS':
            raise PermissionError('Formal pair requires two complete PASS arms')
        if any(p['pair_id']==pair['pair_id'] for p in ledger['pairs']):raise PermissionError('Duplicate pair judgment')
        ledger['pairs'].append(pair)
        if not pair['pair_pass']:
            ledger.update(stopped=True,status='P2_QUALIFICATION_FAILED')
        elif len(ledger['pairs'])==20:
            if len(ledger['attempts'])!=40:raise PermissionError('20/20 requires all original 40 arms')
            ledger.update(stopped=True,status='P2_QUALIFICATION_PASS_20_OF_20')
        write(path,ledger)


def stop_infrastructure(reason):
    with locked(LEDGER):
        ledger=read(LEDGER);ledger.update(stopped=True,status='P2_INFRASTRUCTURE_BLOCKED',stop_reason=reason);write(LEDGER,ledger)


def child(run_id,ticket):
    from tools.r1_b2_8_r3_prospective_selector import official_env
    official_env()
    auth,spec=credential(run_id)
    with locked(LEDGER):
        ledger=read(LEDGER);check_ledger(ledger,sha(AUTH));row=ledger['attempts'][-1]
        if (row['run_id']!=run_id or row['status']!='CLAIMED_BEFORE_PROCESS_START' or row['monitor_pid']!=os.getppid()
            or not secrets.compare_digest(row['claim_token'],ticket)):
            raise PermissionError('Child requires its one live parent claim')
        row.update(status='CHILD_RUNNING',child_pid=os.getpid());write(LEDGER,ledger)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    import tools.tsb_direct_coast_f2_production as production
    arm=production.build_f2_arm(spec,SCIENCE/run_id);production.validate_f2_production(arm)
    prior=next(x for x in read(Q0/'evidence/P2_Q0_40Arm_Production_Preflight_v1.json')['arms'] if x['run_id']==run_id)
    before=read(prior['config_path']);after=read(arm.run_root/'resolved_config.json')
    if (sha(prior['config_path'])!=prior['config_sha256'] or
        production.normalized_config(before,Path(prior['config_path']).parent,before['experiment_uid'])!=
        production.normalized_config(after,arm.run_root,after['experiment_uid'])):
        raise ValueError('Resolved configuration differs from exact Q0 preflight beyond output root/timestamp')
    if read(arm.run_root/'canonical_clock_mapping.json')!=prior['clock']:raise ValueError('Frozen per-SESSION clock mapping changed')
    try:production.lifecycle_f2(arm,spec,F2/'F2_Owner_Protocol_v1.json')
    except RuntimeError as error:
        if str(error)!='Scientific arm gate FAIL; STOP no retry or replacement':raise
        if not (arm.run_root/'arm_science_review.json').is_file() or read(arm.run_root/'arm_science_review.json')['arm_pass']:raise
    # The unchanged F2 audit has a hard-coded F2 evidence sink. Redirect only its
    # module output destination to this new per-arm namespace, never old evidence.
    sink=arm.run_root/'observation_review';(sink/'evidence').mkdir(parents=True,exist_ok=False)
    original=production.OUT
    try:
        production.OUT=sink
        audit=production.audit_f2_observations(arm)
    finally:production.OUT=original
    if audit['status']!='PASS':raise ValueError('Actual official observation audit failed')
    write(arm.run_root/'actual_observation_audit.json',audit,exclusive=True)
    manifest=read(arm.run_root/'execution_manifest.json')
    manifest['artifacts']['actual_observation_audit.json']=sha(arm.run_root/'actual_observation_audit.json')
    write(arm.run_root/'execution_manifest.json',manifest)
    science=read(arm.run_root/'arm_science_review.json')
    write(arm.run_root/'P2_Q1_Child_Completion.json',dict(run_id=run_id,technical_status='PASS',
        scientific_arm_pass=science['arm_pass'],authorization_sha256=sha(AUTH),spec_sha256=canon(spec),
        execution_manifest_sha256=sha(arm.run_root/'execution_manifest.json'),
        actual_observation_audit_sha256=sha(arm.run_root/'actual_observation_audit.json')),exclusive=True)
    return 0 if science['arm_pass'] else 20


def supervise(run_id,ticket,remaining):
    """Account wait4 child/reaped descendants plus per-PID live descendant maxima."""
    import psutil
    start=time.monotonic();process=None;code=1;reason=None;certain=True;consumed=0.;usage=None
    observed={};active_descendants=set()
    try:
        SCIENCE.mkdir(parents=True,exist_ok=True)
        with (SCIENCE/(run_id+'.process.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',run_id,ticket],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            root=psutil.Process(process.pid)
            while True:
                try:
                    children=root.children(recursive=True)
                    active_descendants={p.pid for p in children if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE}
                    for p in [root,*children]:
                        try:
                            ct=p.cpu_times();key=(p.pid,p.create_time())
                            observed[key]=max(observed.get(key,0.),ct.user+ct.system)
                        except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                pid,status,usage=os.wait4(process.pid,os.WNOHANG)
                if pid:
                    process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode;break
                if time.monotonic()-start>=1200 or sum(observed.values())>=remaining:
                    reason='WALL_OR_CUMULATIVE_CPU_LIMIT';os.killpg(process.pid,signal.SIGKILL)
                    _,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode;break
                time.sleep(.05)
            consumed=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
            # Any surviving descendant or unknown PID attribution blocks progression.
            alive=[]
            for pid in active_descendants:
                try:
                    p=psutil.Process(pid)
                    if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:alive.append(pid)
                except psutil.NoSuchProcess:pass
            if alive:
                certain=False;reason='SURVIVING_DESCENDANT_CPU_UNCERTAIN'
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
    except BaseException as error:
        certain=False;reason=type(error).__name__+': '+str(error)
        if process is not None and process.returncode is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:
                _,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);code=process.returncode
                consumed=max(sum(observed.values()),usage.ru_utime+usage.ru_stime)
            except ChildProcessError:pass
    if not math.isfinite(consumed) or consumed<0:certain=False;consumed=0.;reason='INVALID_CPU_ACCOUNTING'
    if consumed>=remaining:reason='CUMULATIVE_CPU_LIMIT'
    accounting=dict(method='MAX_WAIT4_CHILD_AND_REAPED_DESCENDANTS_VS_SUM_LIVE_PER_PID_MAXIMA',
        wait4_cpu_seconds=(usage.ru_utime+usage.ru_stime) if usage else None,
        observed_per_pid_cpu_seconds={str(k):v for k,v in observed.items()},
        surviving_descendant_check=True,remaining_budget_at_start=remaining,certain=certain)
    return code,consumed,time.monotonic()-start,reason,certain,accounting


def launch(run_id):
    auth,spec=credential(run_id)
    ticket=claim(LEDGER,run_id,sha(AUTH),SCIENCE)
    remaining=BUDGET['cpu_seconds']-read(LEDGER)['cpu_seconds_consumed']
    code,cpu,elapsed,reason,certain,accounting=supervise(run_id,ticket,remaining)
    complete=SCIENCE/run_id/'P2_Q1_Child_Completion.json'
    status='INFRASTRUCTURE_FAILURE'
    try:
        if code in (0,20) and reason is None and certain and complete.is_file():
            c=read(complete)
            if c['run_id']!=run_id or c['authorization_sha256']!=sha(AUTH) or c['spec_sha256']!=canon(spec) or c['technical_status']!='PASS':
                raise ValueError('Child completion identity/auth mismatch')
            manifest=read(SCIENCE/run_id/'execution_manifest.json')
            if c['execution_manifest_sha256']!=sha(SCIENCE/run_id/'execution_manifest.json'):raise ValueError('Completion manifest changed')
            for path,digest in manifest['artifacts'].items():
                if sha(SCIENCE/run_id/path)!=digest:raise ValueError('Actual artifact changed: '+path)
            status='COMPLETE_PASS' if c['scientific_arm_pass'] and code==0 else 'SCIENTIFIC_FAILURE'
    except Exception as error:reason='ARTIFACT_COMPLETION_ERROR: '+repr(error)
    finish_attempt(LEDGER,run_id,status,cpu,elapsed,code,reason,certain,accounting)
    if status!='COMPLETE_PASS':return False
    if spec['arm']=='TREATMENT':
        try:
            from tools.tsb_coast_p1r_analysis import analyze_completed_pair
            pair=analyze_completed_pair(SCIENCE/(spec['pair_id']+'-BASELINE'),SCIENCE/run_id)
            pair.update(pair_id=spec['pair_id'],session_id=spec['session_id'],scope='FORMAL_FROZEN_P2_PAIR_QUALIFICATION',
                qualification_contract_sha256=sha(F2/'F2_Owner_Protocol_v1.json'),authorization_sha256=sha(AUTH))
            write(SCIENCE/(spec['pair_id']+'-pair_review.json'),pair,exclusive=True)
            record_pair(LEDGER,pair)
        except Exception as error:
            stop_infrastructure('FORMAL_PAIR_FINALIZATION: '+repr(error));return False
    return not read(LEDGER)['stopped']


def execute():
    # This is a one-shot whole-cohort command; interruption is never auto-resumed.
    initialize()
    try:
        for run_id in RUNS:
            print(json.dumps(dict(next_run_id=run_id,attempts=len(read(LEDGER)['attempts'])),ensure_ascii=False),flush=True)
            if not launch(run_id):break
    except BaseException as error:
        stop_infrastructure(type(error).__name__+': '+str(error))
        raise
    return read(LEDGER)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--child',nargs=2,metavar=('RUN_ID','LIVE_TICKET'))
    args=parser.parse_args()
    if args.child:raise SystemExit(child(*args.child))
    if args.validate:
        credential(RUNS[0]);print('PASS: exact frozen authorization/source/data verified; no attempt claimed')
    elif args.execute:print(json.dumps(execute(),indent=2))
    else:parser.error('Choose --validate or one-shot --execute; no resume/retry command exists')
