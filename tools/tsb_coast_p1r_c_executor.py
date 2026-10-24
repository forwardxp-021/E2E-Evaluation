#!/usr/bin/env python3
"""Dormant first-pair entry: explicit NEW scientific credential is mandatory."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.s2r_production_executor import ROOT, write_manifest, sha256_file
from tools.tsb_coast_p1r_c_production import (build_canonical_coast_arm,
    validate_canonical_production, validate_scientific_credential)
from tools.tsb_coast_p1r_production import claim_attempt, validate_child_claim, lifecycle


def credential(auth_path, run_id):
    from tools.tsb_coast_p1r_c_preflight import fixed_specs
    auth=json.loads(Path(auth_path).read_text())
    found=[s for s in fixed_specs() if s['run_id']==run_id]
    if len(found)!=1:
        raise PermissionError('P1R_C original fixed run identity required')
    spec=found[0]
    validate_scientific_credential(auth,spec,ROOT/auth.get('approved_amendment_path','__UNBOUND__'),
        ROOT/auth.get('execution_source_binding_path','__UNBOUND__'),ROOT/auth.get('production_preflight_path','__UNBOUND__'))
    ledger=ROOT/auth.get('execution_ledger_path','__UNBOUND__')
    expected=ROOT/'outputs/stageR/tsb_coast_p1r_c_execution_state_v1/budget.json'
    if ledger.resolve()!=expected.resolve():
        raise PermissionError('P1R_C inherited cumulative budget ledger not bound')
    return auth,spec,ledger


def authorized_child(auth_path, run_id, output_root, ticket):
    auth,spec,ledger=credential(auth_path,run_id)
    validate_child_claim(ledger,run_id,ticket)
    resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    arm=build_canonical_coast_arm(spec,Path(output_root)/run_id)
    validate_canonical_production(arm)
    lifecycle(arm,spec,ROOT/auth['approved_amendment_path'])
    # Reuse unchanged six-step metric/serializer/artifact lifecycle, then add
    # versioned approved interpretation and mandatory clock artifacts.
    science_path=arm.run_root/'arm_science_review.json'
    science=json.loads(science_path.read_text())
    science['mechanism']['contract_version']='P1R_C_APPROVED_ABSTRACT_COAST_PROXY_v1'
    science['observation_time_mapping']='SAME_INDEX_WITH_DISCLOSED_RAW_TIMESTAMP_SKEW'
    from tools.tsb_coast_p1_measurement import stability
    trace=[json.loads(x)['current_ego'] for x in (arm.run_root/'trace/realized_current_ego.jsonl').read_text().splitlines()]
    science['retained_response_stability_diagnostic']=stability([x['time_us'] for x in trace],[x['speed_mps'] for x in trace],spec['arm'])
    write_manifest(science_path,science)
    manifest_path=arm.run_root/'execution_manifest.json';manifest=json.loads(manifest_path.read_text())
    for name in ('canonical_clock_mapping.json','resolved_config.json','coast_binding.json','arm_science_review.json'):
        manifest['artifacts'][name]=sha256_file(arm.run_root/name)
    manifest['canonical_clock_contract']='P1R_C_CANONICAL_CLOCK_v1'
    write_manifest(manifest_path,manifest)


def launch_first_pair_arm(auth_path, run_id, output_root):
    """One monitored process, no automatic retries or next-pair execution."""
    import psutil
    auth,spec,ledger_path=credential(auth_path,run_id)
    if spec['arm']=='TREATMENT':
        previous=json.loads(ledger_path.read_text())
        if not any(a['run_id']==spec['pair_id']+'-BASELINE' and a['status']=='COMPLETE_REVIEW_REQUIRED' for a in previous['attempts']):
            raise PermissionError('P1R_C Baseline must complete and pass first')
    ticket=claim_attempt(ledger_path,run_id)
    remaining=10800-json.loads(ledger_path.read_text())['cpu_seconds_consumed']
    command=[sys.executable,'-B',__file__,'--authorized-child',str(Path(auth_path).resolve()),run_id,str(Path(output_root).resolve()),ticket]
    log=Path(output_root)/(run_id+'.process.log');log.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();peak=0.;reason=None
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        root=psutil.Process(process.pid)
        while True:
            pid,status,usage=os.wait4(process.pid,os.WNOHANG)
            if pid:
                process.returncode=os.waitstatus_to_exitcode(status);break
            try:
                cpu=sum(p.cpu_times().user+p.cpu_times().system for p in [root,*root.children(recursive=True)] if p.is_running())
                peak=max(peak,cpu)
            except (psutil.NoSuchProcess,psutil.ZombieProcess):
                pass
            if time.monotonic()-start>1200 or peak>=remaining:
                reason='TIME_OR_CPU_CEILING';os.killpg(process.pid,signal.SIGKILL)
                _,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);break
            time.sleep(.25)
    consumed=max(peak,usage.ru_utime+usage.ru_stime)
    with ledger_path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(ledger_path.read_text());ledger['cpu_seconds_consumed']+=consumed
        science=Path(output_root)/run_id/'arm_science_review.json'
        scientific_failure=science.is_file() and not json.loads(science.read_text())['arm_pass']
        ledger['attempts'][-1].update(status='COMPLETE_REVIEW_REQUIRED' if process.returncode==0 else ('SCIENTIFIC_FAILURE' if scientific_failure else 'INFRASTRUCTURE_FAILURE'),cpu_seconds=consumed,elapsed_s=time.monotonic()-start,stop_reason=reason)
        ledger['stopped']=process.returncode!=0 or spec['arm']=='TREATMENT'
        write_manifest(ledger_path,ledger)
    if process.returncode!=0:
        raise RuntimeError('P1R_C arm failed; STOP, no automatic retry/replacement')
    if spec['arm']=='TREATMENT':
        from tools.tsb_coast_p1r_analysis import analyze_completed_pair
        pair=analyze_completed_pair(Path(output_root)/(spec['pair_id']+'-BASELINE'),Path(output_root)/run_id)
        write_manifest(Path(output_root)/(spec['pair_id']+'-pair_review.json'),pair)
        if not pair['pair_pass']:
            raise RuntimeError('P1R_C pair failed; STOP, no retry or next-pair execution')


if __name__=='__main__':
    if len(sys.argv)==6 and sys.argv[1]=='--authorized-child':
        authorized_child(*sys.argv[2:])
    else:
        raise SystemExit('P1R-C scientific execution NOT_AUTHORIZED. A new explicit first-pair credential is required.')
