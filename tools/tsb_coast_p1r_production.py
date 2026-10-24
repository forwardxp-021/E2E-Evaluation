#!/usr/bin/env python3
"""New C2 production factory and fail-closed lifecycle; science closed by default."""
from __future__ import annotations
import fcntl
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import secrets
import subprocess
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.s2r_production_executor import (ROOT,CONFIG_ROOT,FreshArm,_validate_spec,verify_scenario_precontext,
    sha256_file,canonical_sha256,resolved_override_contract,write_manifest,PROTECTED,PROTECTED_SHA,SEED)


def build_coast_arm(spec,run_root):
    from tools.r1_b2_8_r3_prospective_selector import official_env,official_count
    official_env();_validate_spec(spec)
    if spec['role']!='B' or spec.get('candidate')!='TSB_COAST_C2_P1R':raise ValueError('New C2 B-only binding required')
    if run_root.exists():raise FileExistsError('Fresh namespace required')
    if sha256_file(PROTECTED)!=PROTECTED_SHA:raise ValueError('Protected CSV mismatch')
    if official_count(spec['db_path'],spec['scenario_token'])!=1:raise ValueError('Exact scenario not uniquely resolved')
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker,build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
    from tools.tsb_coast_p1r_recorder import CoastTransitionRecorderV1
    planner=CoastC2PlannerV1(spec,run_root/'trace',run_root/'telemetry')
    overrides=resolved_override_contract(spec,run_root/'raw')
    overrides=[('planner=tsb_coast_c2_p1r' if x.startswith('planner=') else
                'experiment_name=tsb_coast_p1r_preparation' if x.startswith('experiment_name=') else x) for x in overrides]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg=compose(config_name='default_simulation',overrides=overrides)
    resolved=OmegaConf.to_container(cfg,resolve=True)
    common=set_up_common_builder(cfg,'p1r_c2_zero_run_factory')
    worker=build_callbacks_worker(cfg);callbacks=build_simulation_callbacks(cfg,common.output_dir,worker)
    runners=build_simulations(cfg,common.worker,callbacks,worker,pre_built_planners=[planner])
    if len(runners)!=1:raise ValueError('Runner count must be one')
    runner=runners[0];simulation=runner._simulation
    verify_scenario_precontext(simulation._scenario,spec)
    controller=simulation._ego_controller;base=controller._tracker
    if spec['arm']=='TREATMENT':
        controller._tracker=ZeroLongitudinalRequestTrackerV1(base,int(spec['start_timestamp_us']),[(2100000,3900000)])
    recorder=CoastTransitionRecorderV1(run_root/'telemetry/actual_lqr_controller_telemetry.jsonl',spec['arm'],spec['start_timestamp_us'])
    recorder.install(controller,simulation._time_controller)
    write_manifest(run_root/'resolved_config.json',resolved)
    write_manifest(run_root/'coast_binding.json',{'candidate':spec['candidate'],'arm':spec['arm'],
        'adapter_enabled':spec['arm']=='TREATMENT','episode_anchor_us':spec['start_timestamp_us'],
        'coast_interval_us':[[2100000,3900000]] if spec['arm']=='TREATMENT' else [],
        'adapter_sha256':sha256_file(ROOT/'tools/tsb_coast_zero_command_tracker_v1.py'),
        'planner_sha256':sha256_file(ROOT/'tools/tsb_coast_p1r_planner.py')})
    return FreshArm(spec=spec,runner=runner,planner=planner,simulation=simulation,controller=controller,
        tracker=controller._tracker,motion_model=controller._motion_model,callbacks=callbacks,recorder=recorder,
        random_state=random.Random(SEED),run_root=run_root,config_hash=canonical_sha256(resolved),
        precontext_id=spec['precontext_id'],route_id=spec['route_id'],common_builder=common,cfg=cfg)


def validate_callbacks(arm):
    from nuplan.planning.simulation.callback.metric_callback import MetricCallback
    from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
    from nuplan.planning.metrics.metric_engine import MetricsEngine
    from tools.b1_offline_metric_finalize import EXPECTED_OFFICIAL_METRICS
    from tools.tsb_coast_p1r_recorder import CoastTransitionRecorderV1
    actual=arm.simulation.callback.callbacks
    metrics=[x for x in actual if type(x) is MetricCallback];serial=[x for x in actual if type(x) is SimulationLogCallback]
    final=[x for x in arm.common_builder.multi_main_callback._main_callbacks if type(x) is MetricFileCallback]
    if not (len(metrics)==len(serial)==len(final)==1):raise ValueError('Official callback owners missing')
    metric,serializer,finalizer=metrics[0],serial[0],final[0]
    if type(metric.metric_engine) is not MetricsEngine:raise ValueError('Not official metric engine')
    raw=(arm.run_root/'raw').resolve();metric_path=raw/'metrics'
    if metric.metric_engine._main_save_path.resolve()!=metric_path or finalizer._metric_file_output_path.resolve()!=metric_path or [x.resolve() for x in finalizer._scenario_metric_paths]!=[metric_path]:
        raise ValueError('Metric producer/finalizer path mismatch')
    names=set(x.name for x in metric.metric_engine.metrics)
    if not (set(EXPECTED_OFFICIAL_METRICS)-{'corners_in_drivable_area'}).issubset(names):raise ValueError('Official safety metric engines missing')
    if serializer._serialization_type!='msgpack' or serializer._output_directory.resolve()!=raw/'simulation_log':raise ValueError('Official serializer mismatch')
    if metric._pool is not None or serializer._pool is not None or metric.futures or serializer.futures:raise ValueError('Callbacks not fresh sequential')
    if arm.simulation._history_buffer is not None or len(arm.simulation.history)!=0:raise ValueError('Simulation already initialized')
    if type(arm.recorder) is not CoastTransitionRecorderV1 or arm.recorder.row_count!=0 or arm.recorder._model is not arm.motion_model or arm.recorder._tracker is not arm.tracker:
        raise ValueError('Coast recorder identity mismatch')
    if arm.tracker.track_trajectory is not arm.recorder._bound_track or arm.motion_model.propagate_state is not arm.recorder._bound_propagate:
        raise ValueError('Actual recorder method binding displaced')
    if arm.controller._tracker is not arm.tracker or arm.controller._motion_model is not arm.motion_model:
        raise ValueError('Controller model/tracker binding displaced')
    if arm.runner.planner is not arm.planner or arm.runner.simulation is not arm.simulation:raise ValueError('Runner ownership mismatch')
    if float(arm.motion_model._accel_time_constant)!=.2:raise ValueError('Bound official model time constant changed')
    if not arm.cfg.run_metric or not arm.cfg.disable_callback_parallelization:raise ValueError('Metric/sequential config mismatch')
    return {'status':'PASS','official_metric_engines':sorted(names),'official_metric_file_callback':True,
        'official_serializer':True,'recorder_tracks_final_packet_and_model':True,'simulation_uninitialized':True,
        'runtime_finalization':'NOT_EXECUTED','expected_metric_parquets':len(EXPECTED_OFFICIAL_METRICS)}


def audit_episode_clock(arm):
    """Metadata-only official time-point queries; no controller or runner step."""
    scenario=arm.simulation._scenario
    ts=[int(scenario.get_time_point(i).time_us) for i in range(81)]
    anchor=int(arm.spec['start_timestamp_us'])
    deviations=[t-anchor-i*100000 for i,t in enumerate(ts)]
    crossing=[]
    for boundary in (2100000,3900000):
        crossing.extend({'boundary_us':boundary,'transition':i,'left_elapsed_us':ts[i]-anchor,'right_elapsed_us':ts[i+1]-anchor}
            for i in range(79) if ts[i]-anchor<boundary<ts[i+1]-anchor)
    passed=not any(deviations) and not crossing
    return {'status':'PASS' if passed else 'BLOCKED_PHYSICAL_CLOCK_CONTRACT',
        'scenario_timepoints_us':ts,'deviation_from_anchor_plus_100000i_us':deviations,
        'max_abs_deviation_us':max(abs(x) for x in deviations),'cross_boundary_steps':crossing,
        'min_step_us':min(b-a for a,b in zip(ts,ts[1:])),'max_step_us':max(b-a for a,b in zip(ts,ts[1:])),
        'no_rounding_resampling_or_scenario_replacement':True,'queries_only':True}


def require_exact_episode_clock(arm):
    proof=audit_episode_clock(arm)
    if proof['status']!='PASS':raise ValueError('P1R_PHYSICAL_CLOCK_CONTRACT_BLOCKED: logged timestamps are not an exact 100000us clock; no runner entry')
    return proof


def validate_authorization(auth,spec,contracts_path,roster_path):
    if auth.get('stage')!='TSB_COAST_P1R_FIRST_PAIR' or auth.get('authorization_status')!='AUTHORIZED' or auth.get('scientific_owner_approved') is not True:
        raise PermissionError('New exact P1R Scientific Owner authorization required; old authorizations rejected')
    contract=json.loads(Path(contracts_path).read_text())
    if contract.get('owner_approval')!='APPROVED':raise PermissionError('Prospective scientific amendments not approved')
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    if contract.get('specification')!=SCIENCE_CONTRACT:raise PermissionError('Approved contract differs from the bound analyzer specification')
    if auth.get('contract_sha256')!=sha256_file(Path(contracts_path)) or auth.get('roster_sha256')!=sha256_file(Path(roster_path)):
        raise PermissionError('Contract/roster hash mismatch')
    expected=[f'P1R-C2-DEV-01-{a}' for a in ('BASELINE','TREATMENT')]
    if auth.get('approved_run_ids')!=expected or spec['run_id'] not in expected or spec['role']!='B':raise PermissionError('Only exact first Development pair may be authorized')
    if auth.get('approved_spec_sha256',{}).get(spec['run_id'])!=canonical_sha256(spec):
        raise PermissionError('Exact SESSION/precontext/route/spec binding missing or changed')
    proof_path=Path(auth.get('production_preflight_path','__UNBOUND_PREFLIGHT__'))
    if not proof_path.is_file() or auth.get('production_preflight_sha256')!=sha256_file(proof_path):raise PermissionError('Exact completed production preflight not bound')
    proof=json.loads(proof_path.read_text())
    if proof.get('status')!='PASS' or len(proof.get('arms',[]))!=6 or any(x.get('episode_clock',{}).get('status')!='PASS' for x in proof['arms']):
        raise PermissionError('Physical clock/production entry remains BLOCKED')
    paths=auth.get('source_file_sha256',{})
    required=['tools/tsb_coast_p1r_production.py','tools/tsb_coast_p1r_planner.py','tools/tsb_coast_p1r_recorder.py','tools/tsb_coast_p1r_analysis.py','tools/tsb_coast_zero_command_tracker_v1.py','tools/tsb_coast_p1_measurement.py',
        'tools/s2r_production_executor.py','tools/b1_tsb_qualification_executor.py',
        'tools/r1_b2_8_r3_1_official_safety_adapter.py','tools/r1_official_metric_canonicalizer.py',
        'tools/b1_native_route_infrastructure.py','tools/r1_primary80_scientific_time_controller_v1.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py',
        '../nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py']
    required.extend(os.path.relpath(p,ROOT) for p in CONFIG_ROOT.parent.rglob('*.yaml'))
    required.extend(str(p.relative_to(ROOT)) for p in (ROOT/'configs/r1_official_technical_smoke_hydra').rglob('*.yaml'))
    if not set(required).issubset(paths) or any(sha256_file(ROOT/p)!=h for p,h in paths.items()):raise PermissionError('Executable source binding incomplete/changed')
    data={str((ROOT/p).resolve()):h for p,h in auth.get('data_file_sha256',{}).items()}
    metadata=(ROOT.parent/'nuplan/dataset/maps/nuplan-maps-v1.0.json').resolve()
    version=json.loads(metadata.read_text())[spec['map_name']]['version']
    map_file=metadata.parent/spec['map_name']/version/'map.gpkg'
    required_data={str(Path(spec['db_path']).resolve()),str(metadata),str(map_file)}
    if not required_data.issubset(data) or any(sha256_file(Path(p))!=h for p,h in data.items()):
        raise PermissionError('Full database/map data binding incomplete or changed')


def claim_attempt(ledger_path,run_id):
    ledger_path=Path(ledger_path)
    with (ledger_path.with_suffix('.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text())
        if ledger.get('stopped') or any(x['status'] in ('CLAIMED_BEFORE_PROCESS_START','CHILD_RUNNING') for x in ledger['attempts']) or any(x['run_id']==run_id for x in ledger['attempts']):raise PermissionError('Stopped/inflight ledger or retry requires new Owner decision')
        if ledger['historical_attempts']+len(ledger['attempts'])>=8 or len(ledger['attempts'])>=6 or ledger['cpu_seconds_consumed']>=10800:
            raise PermissionError('Cumulative budget exhausted')
        token=secrets.token_hex(32)
        ledger['attempts'].append({'run_id':run_id,'status':'CLAIMED_BEFORE_PROCESS_START','claim_token':token,'monitor_pid':os.getpid()})
        write_manifest(ledger_path,ledger)
        return token


def validate_child_claim(ledger_path,run_id,ticket):
    """A direct child invocation cannot bypass the persisted parent budget claim."""
    ledger_path=Path(ledger_path)
    with ledger_path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(ledger_path.read_text())
        matches=[a for a in ledger['attempts'] if a['run_id']==run_id]
        if ledger.get('stopped') or len(matches)!=1:
            raise PermissionError('No active monitored budget claim')
        row=matches[0]
        if row['status']!='CLAIMED_BEFORE_PROCESS_START' or not secrets.compare_digest(row.get('claim_token',''),ticket) or row.get('monitor_pid')!=os.getppid():
            raise PermissionError('Child must belong to the live parent budget monitor')
        row.update(status='CHILD_RUNNING',child_pid=os.getpid());write_manifest(ledger_path,ledger)



def lifecycle(arm,spec,approved_contract_path):
    """Only entered by future authorized child. Reuse official safety/serializer validators."""
    from tools.b1_tsb_qualification_executor import validate_official_metric_outputs,validate_serializer_complete,validate_and_hash_artifacts
    from nuplan.planning.script.utils import save_runner_reports
    states=[]
    report=arm.runner.run()
    if not report.succeeded:raise RuntimeError('Runner failed; STOP no automatic retry')
    states.append('RUNNER_COMPLETE');arm.recorder.validate_complete();states.append('RECORDER_COMPLETE')
    save_runner_reports([report],arm.common_builder.output_dir,arm.cfg.runner_report_file)
    arm.common_builder.multi_main_callback.on_run_simulation_end()
    validate_official_metric_outputs(arm);states.append('OFFICIAL_METRICS_COMPLETE')
    validate_serializer_complete(arm);states.append('SERIALIZER_COMPLETE')
    payload={'schema_version':'P1R_C2_ARM_v1','spec':spec,'config_hash':arm.config_hash,'lifecycle':states}
    write_manifest(arm.run_root/'execution_manifest.json',payload);states.append('MANIFEST_COMPLETE')
    payload['artifacts']=validate_and_hash_artifacts(arm);states.append('ARTIFACT_HASHES_VALIDATED')
    write_manifest(arm.run_root/'execution_manifest.json',payload)
    from tools.tsb_coast_p1r_analysis import analyze_completed_arm
    science=analyze_completed_arm(arm.run_root,spec['arm'])
    science['mechanism']['approval']='APPROVED_BOUND_CONTRACT';science['approved_contract_sha256']=sha256_file(Path(approved_contract_path))
    write_manifest(arm.run_root/'arm_science_review.json',science)
    if not science['arm_pass']:raise RuntimeError('Scientific arm gate FAIL; STOP no retry or replacement')


def launch_authorized(auth_path,spec_path,contract_path,roster_path,ledger_path,output_root):
    """A separately authorized future one-arm process; never called by preflight."""
    import psutil
    auth=json.loads(Path(auth_path).read_text());spec=json.loads(Path(spec_path).read_text())
    validate_authorization(auth,spec,contract_path,roster_path)
    if spec['arm']=='TREATMENT':
        prior=json.loads(Path(ledger_path).read_text())
        if not any(x['run_id']==spec['pair_id']+'-BASELINE' and x['status']=='COMPLETE_REVIEW_REQUIRED' for x in prior['attempts']):
            raise PermissionError('Baseline must complete and pass before paired Treatment')
    if auth.get('execution_ledger_path')!=str(Path(ledger_path).resolve()):raise PermissionError('Exact cumulative execution ledger not bound')
    claim_attempt(ledger_path,spec['run_id'])
    ledger=json.loads(Path(ledger_path).read_text());remaining=10800-ledger['cpu_seconds_consumed']
    ticket=ledger['attempts'][-1]['claim_token']
    command=[sys.executable,'-B',__file__,'--authorized-child',str(auth_path),str(spec_path),str(contract_path),str(roster_path),str(output_root),str(ledger_path),ticket]
    log=Path(output_root)/(spec['run_id']+'.process.log');log.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();peak_cpu=0.;reason=None
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        root=psutil.Process(process.pid)
        while True:
            pid,status,usage=os.wait4(process.pid,os.WNOHANG)
            if pid:process.returncode=os.waitstatus_to_exitcode(status);break
            try:
                members=[root,*root.children(recursive=True)]
                cpu=sum(x.cpu_times().user+x.cpu_times().system for x in members if x.is_running())
                peak_cpu=max(peak_cpu,cpu)
            except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
            if time.monotonic()-started>1200 or peak_cpu>=remaining:
                reason='TIME_OR_CPU_CEILING';os.killpg(process.pid,signal.SIGKILL)
                _,status,usage=os.wait4(process.pid,0);process.returncode=os.waitstatus_to_exitcode(status);break
            time.sleep(.25)
    consumed=max(peak_cpu,usage.ru_utime+usage.ru_stime)
    with Path(ledger_path).with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(Path(ledger_path).read_text());ledger['cpu_seconds_consumed']+=consumed
        science_path=Path(output_root)/spec['run_id']/'arm_science_review.json'
        failed_science=science_path.is_file() and not json.loads(science_path.read_text())['arm_pass']
        ledger['attempts'][-1].update(status='COMPLETE_REVIEW_REQUIRED' if process.returncode==0 else ('SCIENTIFIC_FAILURE' if failed_science else 'INFRASTRUCTURE_FAILURE'),cpu_seconds=consumed,elapsed_s=time.monotonic()-started,stop_reason=reason)
        ledger['stopped']=process.returncode!=0;write_manifest(Path(ledger_path),ledger)
    if process.returncode!=0:raise RuntimeError('Attempt failed; Owner review required, no retry')
    if spec['arm']=='TREATMENT':
        from tools.tsb_coast_p1r_analysis import analyze_completed_pair
        with Path(ledger_path).with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(Path(ledger_path).read_text());ledger['stopped']=True;write_manifest(Path(ledger_path),ledger)
        pair=analyze_completed_pair(Path(output_root)/(spec['pair_id']+'-BASELINE'),Path(output_root)/spec['run_id'])
        write_manifest(Path(output_root)/(spec['pair_id']+'-pair_review.json'),pair)
        with Path(ledger_path).with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(Path(ledger_path).read_text());ledger['stopped']=True
            ledger['first_pair_completed']=True;ledger['first_pair_pass']=pair['pair_pass'];write_manifest(Path(ledger_path),ledger)
        if not pair['pair_pass']:raise RuntimeError('Scientific pair gate FAIL; STOP no retry or replacement')


if __name__=='__main__':
    if len(sys.argv)==9 and sys.argv[1]=='--authorized-child':
        auth_path,spec_path,contract,roster,output_root,ledger_path,ticket=sys.argv[2:]
        auth=json.loads(Path(auth_path).read_text());spec=json.loads(Path(spec_path).read_text())
        validate_authorization(auth,spec,contract,roster)
        if auth.get('execution_ledger_path')!=str(Path(ledger_path).resolve()):raise PermissionError('Exact cumulative execution ledger not bound')
        validate_child_claim(ledger_path,spec['run_id'],ticket)
        resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
        arm=build_coast_arm(spec,Path(output_root)/spec['run_id']);validate_callbacks(arm);require_exact_episode_clock(arm);lifecycle(arm,spec,contract)
    else:raise SystemExit('No default execution. P1R scientific execution is NOT_AUTHORIZED.')
