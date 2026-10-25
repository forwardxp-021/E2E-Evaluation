#!/usr/bin/env python3
"""F2 exact two-arm credential, fresh child process, new fail-closed CPU ledger."""
from __future__ import annotations
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
from tools.s2r_production_executor import ROOT,sha256_file,canonical_sha256,write_manifest
from tools.tsb_direct_coast_f2_production import OUT,fixed_specs,build_f2_arm,validate_f2_production,lifecycle_f2,audit_f2_observations,normalized_config
from tools.tsb_direct_coast_f2_tracker import SEGMENTS

AUTH=OUT/'F2_Owner_Execution_Authorization_v1.json'
LEDGER=ROOT/'outputs/stageR/tsb_direct_coast_f2_execution_state_v1/budget.json'
SCIENCE_ROOT=ROOT/'outputs/stageR/tsb_direct_coast_f2_dev01_v1'


def validate_credential(run_id):
    auth=json.loads(AUTH.read_text());binding=json.loads((OUT/'F2_Execution_Source_Binding_v1.json').read_text());specs=fixed_specs()
    expected=['F2-DEV-01-BASELINE','F2-DEV-01-TREATMENT']
    if auth['stage']!='TSB_DIRECT_COAST_F2' or auth['authorization_status']!='AUTHORIZED' or auth['scientific_owner_approved'] is not True or auth['approved_run_ids']!=expected or run_id not in expected:
        raise PermissionError('F2 only exact two new DEV01 IDs authorized; old C2/DEV02/03 rejected')
    spec=next(s for s in specs if s['run_id']==run_id)
    if canonical_sha256(spec)!=auth['approved_spec_sha256'][run_id] or sha256_file(OUT/'F2_Execution_Source_Binding_v1.json')!=auth['binding_sha256']:
        raise PermissionError('F2 identity/source binding changed')
    protocol=json.loads((OUT/'F2_Owner_Protocol_v1.json').read_text())
    if protocol['request_segments_us']!={k:[list(x) for x in v] for k,v in SEGMENTS.items()} or protocol['other_times_request_mps2']!=0. or not protocol['scientific_owner_approved']:
        raise PermissionError('F2 fixed Owner requests changed')
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    expected_science=dict(SCIENCE_CONTRACT);expected_science['version']='F2_DIRECT_REQUEST_APPROVED_v1'
    if protocol['qualification']!=expected_science or protocol['maximum_new_attempts']!=2 or protocol['scientific_CPU_seconds_max']!=3600 or protocol['per_arm_wall_seconds_max']!=1200 or protocol['automatic_retry']:
        raise PermissionError('F2 qualification/budget changed')
    for name,key in [('F2_Owner_Protocol_v1.json','protocol_sha256'),('F2_Arm_Specs_v1.json','specs_sha256'),('evidence/F2_Isolated_Model_Validation_v1.json','isolated_validation_sha256'),('evidence/F2_Production_Preflight_v1.json','preflight_sha256')]:
        if sha256_file(OUT/name)!=binding[key]:raise PermissionError('F2 contract/proof hash changed: '+name)
    if auth['source_commit']!=binding['source_commit'] or auth['protocol_sha256']!=binding['protocol_sha256'] or auth['execution_ledger_path']!=str(LEDGER) or auth['scientific_output_root']!=str(SCIENCE_ROOT) or auth['maximum_new_attempts']!=2 or auth['budget']!=dict(cpu_seconds=3600,per_arm_wall_seconds=1200) or auth['technical_retry_authorized'] or not auth['disclosed_observation_skew_acknowledged']:
        raise PermissionError('F2 authorization scope/budget mismatch')
    for group in ('source_config_sha256','data_sha256'):
        for path,digest in binding[group].items():
            if sha256_file(ROOT/path)!=digest:raise PermissionError('F2 frozen source/config/data changed: '+path)
    runtime=binding['runtime']
    if sys.executable!=runtime['scientific_runtime_executable'] or sys.version.split()[0]!=runtime['scientific_runtime_python']:raise PermissionError('F2 wrong scientific Python')
    for name,version in runtime['dependency_versions'].items():
        if importlib.metadata.version(name)!=version:raise PermissionError('F2 dependency changed: '+name)
    if any(json.loads((OUT/name).read_text())['status']!='PASS' for name in ['evidence/F2_Isolated_Model_Validation_v1.json','evidence/F2_Production_Preflight_v1.json']):raise PermissionError('F2 prior validation blocked')
    before=json.loads((OUT/'evidence/Protected_Baseline_v1.json').read_text())
    old='outputs/stageR/tsb_coast_p1r_c_execution_state_v1/budget.json'
    if sha256_file(ROOT/old)!=before['protected_sha256'][old]:raise PermissionError('F2 old C2 stopped ledger changed')
    return auth,spec


def claim(run_id):
    with LEDGER.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(LEDGER.read_text());expected=['F2-DEV-01-BASELINE','F2-DEV-01-TREATMENT']
        index=len(ledger['attempts'])
        if ledger['stopped'] or index>=2 or run_id!=expected[index] or ledger['cpu_seconds_consumed']>=3600:
            raise PermissionError('F2 stopped, duplicate, out-of-order or exhausted ledger; no retry')
        if index==1 and ledger['attempts'][0]['status']!='COMPLETE_PASS':raise PermissionError('F2 Baseline must pass all technical/scientific/actual-observation checks')
        ticket=secrets.token_hex(32);ledger['attempts'].append(dict(run_id=run_id,status='CLAIMED_BEFORE_PROCESS_START',claim_token=ticket,monitor_pid=os.getpid()))
        write_manifest(LEDGER,ledger);return ticket


def child(run_id,ticket):
    auth,spec=validate_credential(run_id)
    with LEDGER.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(LEDGER.read_text());row=ledger['attempts'][-1]
        if ledger['stopped'] or row['run_id']!=run_id or row['status']!='CLAIMED_BEFORE_PROCESS_START' or row['monitor_pid']!=os.getppid() or not secrets.compare_digest(row['claim_token'],ticket):raise PermissionError('F2 child requires its live parent claim')
        row.update(status='CHILD_RUNNING',child_pid=os.getpid());write_manifest(LEDGER,ledger)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    arm=build_f2_arm(spec,SCIENCE_ROOT/run_id);validate_f2_production(arm)
    preflight=json.loads((OUT/'evidence/F2_Production_Preflight_v1.json').read_text())
    old=next(row for row in preflight['arms'] if row['run_id']==run_id)
    old_path=Path(old['config_path']);before=json.loads(old_path.read_text());after=json.loads((arm.run_root/'resolved_config.json').read_text())
    if sha256_file(old_path)!=old['config_sha256'] or normalized_config(before,old_path.parent,before['experiment_uid'])!=normalized_config(after,arm.run_root,after['experiment_uid']):
        raise ValueError('F2 resolved config differs from exact preflight beyond output namespace/run timestamp')
    lifecycle_f2(arm,spec,OUT/'F2_Owner_Protocol_v1.json')
    audit=audit_f2_observations(arm)
    if audit['status']!='PASS':raise RuntimeError('F2 actual observation mapping failed; STOP')
    write_manifest(arm.run_root/'actual_observation_audit.json',audit)
    path=arm.run_root/'execution_manifest.json';manifest=json.loads(path.read_text());manifest['artifacts']['actual_observation_audit.json']=sha256_file(arm.run_root/'actual_observation_audit.json');write_manifest(path,manifest)


def launch(run_id):
    import psutil
    validate_credential(run_id)
    if (SCIENCE_ROOT/run_id).exists() or (SCIENCE_ROOT/(run_id+'.process.log')).exists():raise FileExistsError('F2 fresh arm namespace required')
    ticket=claim(run_id);remaining=3600-json.loads(LEDGER.read_text())['cpu_seconds_consumed'];started=time.monotonic();peak=0.;reason=None;code=1;consumed=0.
    SCIENCE_ROOT.mkdir(parents=True,exist_ok=True)
    try:
        with (SCIENCE_ROOT/(run_id+'.process.log')).open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B',__file__,'--child',run_id,ticket],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            root=psutil.Process(process.pid)
            while True:
                pid,status,usage=os.wait4(process.pid,os.WNOHANG)
                if pid:code=os.waitstatus_to_exitcode(status);break
                try:
                    peak=max(peak,sum(p.cpu_times().user+p.cpu_times().system for p in [root,*root.children(recursive=True)] if p.is_running()))
                except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                if time.monotonic()-started>1200 or peak>=remaining:
                    reason='WALL_OR_CUMULATIVE_CPU_CEILING';os.killpg(process.pid,signal.SIGKILL);_,status,usage=os.wait4(process.pid,0);code=os.waitstatus_to_exitcode(status);break
                time.sleep(.25)
            consumed=max(peak,usage.ru_utime+usage.ru_stime)
    except Exception as error:
        reason=type(error).__name__+': '+str(error)
        if 'process' in locals() and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    science_path=SCIENCE_ROOT/run_id/'arm_science_review.json'
    scientific_fail=science_path.is_file() and not json.loads(science_path.read_text())['arm_pass']
    with LEDGER.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(LEDGER.read_text());ledger['cpu_seconds_consumed']+=consumed
        ledger['attempts'][-1].update(status='COMPLETE_PASS' if code==0 else 'SCIENTIFIC_FAILURE' if scientific_fail else 'INFRASTRUCTURE_FAILURE',cpu_seconds=consumed,elapsed_s=time.monotonic()-started,stop_reason=reason,returncode=code)
        ledger['stopped']=code!=0 or run_id.endswith('TREATMENT');write_manifest(LEDGER,ledger)
    if code!=0:raise RuntimeError('F2 attempt failed; STOP. No retry/parameter change/next arm authorized.')
    if run_id.endswith('TREATMENT'):
        from tools.tsb_coast_p1r_analysis import analyze_completed_pair
        pair=analyze_completed_pair(SCIENCE_ROOT/'F2-DEV-01-BASELINE',SCIENCE_ROOT/run_id)
        pair['qualification_contract_sha256']=sha256_file(OUT/'F2_Owner_Protocol_v1.json')
        pair['status']='TSB_DIRECT_COAST_F2_DEVELOPMENT_PASS' if pair['pair_pass'] else 'TSB_DIRECT_COAST_F2_DEVELOPMENT_SCIENTIFIC_FAIL'
        write_manifest(SCIENCE_ROOT/'F2-DEV-01-pair_review.json',pair)
        if not pair['pair_pass']:raise RuntimeError('F2 pair scientific FAIL; STOP, no F3 or parameter tuning')
    return dict(run_id=run_id,status='COMPLETE_PASS',scientific_attempts=len(json.loads(LEDGER.read_text())['attempts']))


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--child':child(sys.argv[2],sys.argv[3])
    elif len(sys.argv)==3 and sys.argv[1]=='--launch':print(json.dumps(launch(sys.argv[2]),indent=2))
    else:raise SystemExit('F2 requires exact authorized --launch F2-DEV-01-{BASELINE,TREATMENT}; no automatic orchestration')
