#!/usr/bin/env python3
"""Independent F2 factory, exact actual-command recorder and official lifecycle.

Generic callback/lifecycle checks derived from frozen P1R production validators;
old modules, controller and official vehicle model remain unchanged.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import sys
import types
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.s2r_production_executor import (ROOT,CONFIG_ROOT,FreshArm,_validate_spec,
    verify_scenario_precontext,sha256_file,canonical_sha256,resolved_override_contract,
    write_manifest,PROTECTED,PROTECTED_SHA,SEED)
from tools.tsb_coast_p1r_c_production import validate_clock_binding
from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
from tools.tsb_coast_p1r_recorder import fp_bound
from tools.tsb_direct_coast_f2_tracker import ScriptedLongitudinalRequestTrackerV1,request_at,SEGMENTS
from tools.r2_bj_b0_2_passive_actual_lqr_recorder import _atomic_jsonl
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

OUT=ROOT/'docs/stageR/tsb_direct_coast_f2'


class DirectCoastF2PlannerV1(CoastC2PlannerV1):
    """Preserve the existing native-route geometry/reference generator only."""
    def name(self):return 'TSB_DIRECT_COAST_F2_'+self._coast_arm


class F2TransitionRecorderV1:
    def __init__(self,path,arm,anchor_us):
        self.output_path=Path(path);self.arm=arm;self.anchor_us=int(anchor_us)
        self.rows=[];self.pending=None;self._installed=False;self.forbidden_longitudinal_calls=0

    @property
    def row_count(self):return len(self.rows)

    def install(self,controller):
        if self._installed or type(controller) is not TwoStageController or type(controller._motion_model) is not KinematicBicycleModel:
            raise TypeError('F2 requires fresh official controller and model')
        tracker=controller._tracker;model=controller._motion_model
        if type(tracker) is not ScriptedLongitudinalRequestTrackerV1 or type(tracker._base) is not LQRTracker or tracker.arm!=self.arm or tracker._episode_start_time_us!=self.anchor_us:
            raise ValueError('F2 both arms must use the exact direct adapter/anchor')
        base=tracker._base;rec=self;track=tracker.track_trajectory;propagate=model.propagate_state
        def forbidden(_base,*args,**kwargs):
            rec.forbidden_longitudinal_calls+=1
            raise RuntimeError('F2 longitudinal LQR/stopping/full tracker call forbidden')
        for name in ('track_trajectory','_longitudinal_lqr_controller','_stopping_controller'):
            setattr(base,name,types.MethodType(forbidden,base))
        def wrapped_track(_tracker,current_iteration,next_iteration,initial_state,trajectory):
            i=int(current_iteration.index);now=int(current_iteration.time_point.time_us);nxt=int(next_iteration.time_point.time_us)
            if rec.pending is not None or not 0<=i<79 or i!=len(rec.rows) or now!=rec.anchor_us+i*100000 or nxt!=now+100000 or initial_state.time_us!=now:
                raise ValueError('F2 duplicate/missing/misaligned transition')
            before_calls=tracker.lateral_calls
            result=track(current_iteration,next_iteration,initial_state,trajectory)
            expected=request_at(rec.arm,now-rec.anchor_us);a=float(result.rear_axle_acceleration_2d.x);steer=float(result.tire_steering_rate)
            if a!=expected or a>0 or not math.isfinite(steer) or tracker.lateral_calls!=before_calls+1 or rec.forbidden_longitudinal_calls:
                raise ValueError('F2 request/lateral/longitudinal bypass gate failure')
            ev=tracker.last_command_evidence
            rec.pending=(result,initial_state,dict(iteration=i,time_us=now,next_time_us=nxt,episode_time_us=now-rec.anchor_us,
                tracking_mode=ev['mode'],reference_request_mps2=expected,longitudinal_requested_acceleration=a,
                steering_request=steer,longitudinal_LQR_calls=rec.forbidden_longitudinal_calls,lateral_LQR_calls=tracker.lateral_calls))
            return result
        def wrapped_model(_model,state,ideal_dynamic_state,sampling_time):
            if rec.pending is None:raise ValueError('Propagation without recorded command')
            command,input_state,row=rec.pending
            if command is not ideal_dynamic_state or input_state is not state or sampling_time.time_us!=100000:
                raise ValueError('Command did not reach the bound model or duplicate update')
            before=float(state.dynamic_car_state.rear_axle_acceleration_2d.x);v0=float(state.dynamic_car_state.rear_axle_velocity_2d.x)
            output=propagate(state,ideal_dynamic_state,sampling_time)
            a=float(output.dynamic_car_state.rear_axle_acceleration_2d.x);v1=float(output.dynamic_car_state.rear_axle_velocity_2d.x)
            expected=before+.1/(.1+model._accel_time_constant)*(row['longitudinal_requested_acceleration']-before)
            if output is state or output.time_us!=row['next_time_us'] or not all(math.isfinite(x) for x in (before,a,v0,v1,output.rear_axle.x,output.rear_axle.y)):
                raise ValueError('State propagation missing/nonfinite')
            if abs(a-expected)>fp_bound(a,expected,before) or abs(v1-(v0+.1*a))>fp_bound(v1,v0):
                raise ValueError('Official model equation mismatch')
            row.update(model_realized_acceleration=a,model_acceleration_before=before,realized_speed=v1,
                speed_before=v0,speed_difference_acceleration=(v1-v0)/.1,model_expected_acceleration=expected,
                model_filter_tau_s=float(model._accel_time_constant),coast_command_reached_model=(row['tracking_mode']=='COAST'),
                command_identity_reached_model=True,state_propagation_continues=True,
                displacement_m=float(np.hypot(output.rear_axle.x-state.rear_axle.x,output.rear_axle.y-state.rear_axle.y)),
                realized_position=dict(x=float(output.rear_axle.x),y=float(output.rear_axle.y),heading=float(output.rear_axle.heading)))
            _atomic_jsonl(rec.output_path,[*rec.rows,row]);rec.rows.append(row);rec.pending=None
            return output


        tracker.track_trajectory=types.MethodType(wrapped_track,tracker)
        model.propagate_state=types.MethodType(wrapped_model,model)
        self._tracker=tracker;self._base=base;self._model=model;self._controller=controller;self._installed=True
        self._bound_track=tracker.track_trajectory;self._bound_propagate=model.propagate_state

    def validate_complete(self):
        if len(self.rows)!=79 or self.pending is not None or self.forbidden_longitudinal_calls or self._tracker.lateral_calls!=79:
            raise ValueError('F2 expected 79 complete transitions, lateral calls, and zero longitudinal solver calls')


def fixed_specs():
    specs=json.loads((OUT/'F2_Arm_Specs_v1.json').read_text())['arms']
    if [s['run_id'] for s in specs]!=['F2-DEV-01-BASELINE','F2-DEV-01-TREATMENT'] or any(s['session_id']!='2021.09.17.14.49.23_veh-42' or s['scenario_token']!='9fe700abf47057f0' or s['candidate']!='TSB_DIRECT_COAST_F2' or s['role']!='B' for s in specs):
        raise ValueError('F2 exact original exposed DEV01 only')
    return specs

def build_f2_arm(spec, run_root):
    """Fresh real official objects, never initialize/step/run or read outcomes."""
    from tools.r1_b2_8_r3_prospective_selector import official_env, official_count
    official_env(); _validate_spec(spec)
    run_root = Path(run_root)
    if spec['role'] != 'B' or spec.get('candidate') != 'TSB_DIRECT_COAST_F2':
        raise ValueError('P1R_C requires the unchanged C2 B-development spec')
    if run_root.exists():
        raise FileExistsError('P1R_C fresh namespace required')
    if sha256_file(PROTECTED) != PROTECTED_SHA or official_count(spec['db_path'], spec['scenario_token']) != 1:
        raise ValueError('P1R_C protected data or unique source identity mismatch')
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.tsb_direct_coast_f2_production import DirectCoastF2PlannerV1
    from tools.tsb_direct_coast_f2_tracker import ScriptedLongitudinalRequestTrackerV1
    from tools.tsb_direct_coast_f2_production import F2TransitionRecorderV1
    planner = DirectCoastF2PlannerV1(spec, run_root/'trace', run_root/'telemetry')
    overrides = resolved_override_contract(spec, run_root/'raw')
    overrides = [('planner=tsb_direct_coast_f2' if x.startswith('planner=') else
        'simulation_time_controller=tsb_coast_canonical_clock_v1' if x.startswith('simulation_time_controller.') else
        'experiment_name=tsb_direct_coast_f2' if x.startswith('experiment_name=') else x) for x in overrides]
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name='default_simulation', overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    common = set_up_common_builder(cfg, 'p1r_c_canonical_clock_factory')
    worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, worker)
    runners = build_simulations(cfg, common.worker, callbacks, worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise ValueError('P1R_C expected exactly one official runner')
    runner = runners[0]; simulation = runner._simulation
    verify_scenario_precontext(simulation._scenario, spec)
    controller = simulation._ego_controller
    controller._tracker = ScriptedLongitudinalRequestTrackerV1(controller._tracker,int(spec['start_timestamp_us']),spec['arm'])
    recorder = F2TransitionRecorderV1(run_root/'telemetry/actual_lqr_controller_telemetry.jsonl',
        spec['arm'], int(spec['start_timestamp_us']))
    arm = FreshArm(spec, runner, planner, simulation, controller, controller._tracker,
        controller._motion_model, callbacks, recorder, random.Random(SEED), run_root,
        canonical_sha256(resolved), spec['precontext_id'], spec['route_id'], common, cfg)
    mapping = validate_clock_binding(arm)
    # The legacy optional clock check only recognizes the old class name.
    # Validate the exact new authority above, then retain its untouched strict
    # command/model/anchor/79-transition checks via the clock-free core install.
    recorder.install(controller)
    recorder._canonical_clock_authority = simulation._time_controller
    write_manifest(run_root/'resolved_config.json', resolved)
    write_manifest(run_root/'canonical_clock_mapping.json', mapping)
    write_manifest(run_root/'coast_binding.json', dict(version='F2_DIRECT_REQUEST_STACK_v1',
        candidate=spec['candidate'], arm=spec['arm'], episode_anchor_us=mapping['episode_start_us'],
        clock_source_sha256=sha256_file(ROOT/'tools/tsb_coast_p1r_c_clock.py'),
        adapter_enabled=True, adapter_type='ScriptedLongitudinalRequestTrackerV1',
        direct_request_segments_us=[list(x) for x in SEGMENTS[spec['arm']]],
        base_tracker='OFFICIAL_LQR_LATERAL_ONLY',common_precontext_unchanged=True))
    return arm

def validate_callbacks(arm):
    from nuplan.planning.simulation.callback.metric_callback import MetricCallback
    from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
    from nuplan.planning.metrics.metric_engine import MetricsEngine
    from tools.b1_offline_metric_finalize import EXPECTED_OFFICIAL_METRICS
    from tools.tsb_direct_coast_f2_production import F2TransitionRecorderV1
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
    if type(arm.recorder) is not F2TransitionRecorderV1 or arm.recorder.row_count!=0 or arm.recorder._model is not arm.motion_model or arm.recorder._tracker is not arm.tracker:
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


def validate_f2_production(arm):
    mapping=validate_clock_binding(arm)
    if arm.recorder._canonical_clock_authority is not arm.simulation._time_controller:
        raise ValueError('F2 unique recorder clock authority required')
    if type(arm.planner) is not DirectCoastF2PlannerV1 or type(arm.tracker) is not ScriptedLongitudinalRequestTrackerV1:
        raise ValueError('F2 exact planner/adapter types required')
    return dict(status='PASS',clock=mapping,callbacks=validate_callbacks(arm))


def analyze_f2_arm(root,arm):
    from tools.tsb_coast_p1r_analysis import analyze_completed_arm
    review=analyze_completed_arm(root,arm)
    rows=[json.loads(x) for x in (Path(root)/'telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
    for i,row in enumerate(rows):
        if row['longitudinal_requested_acceleration']!=request_at(arm,i*100000) or row['longitudinal_LQR_calls']!=0 or row['lateral_LQR_calls']!=i+1:
            review['mechanism']['reasons'].append('F2_DIRECT_REQUEST_OR_LATERAL_GATE_FAIL');review['arm_pass']=False
    review['mechanism']['contract_version']='F2_DIRECT_REQUEST_APPROVED_v1'
    review['mechanism']['approval']='SCIENTIFIC_OWNER_APPROVED_F2'
    return review

def lifecycle_f2(arm,spec,approved_contract_path):
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
    payload={'schema_version':'F2_DIRECT_COAST_ARM_v1','spec':spec,'config_hash':arm.config_hash,'lifecycle':states}
    write_manifest(arm.run_root/'execution_manifest.json',payload);states.append('MANIFEST_COMPLETE')
    payload['artifacts']=validate_and_hash_artifacts(arm);states.append('ARTIFACT_HASHES_VALIDATED')
    write_manifest(arm.run_root/'execution_manifest.json',payload)
    science=analyze_f2_arm(arm.run_root,spec['arm'])
    science['mechanism']['approval']='APPROVED_BOUND_CONTRACT';science['approved_contract_sha256']=sha256_file(Path(approved_contract_path))
    write_manifest(arm.run_root/'arm_science_review.json',science)
    for name in ('canonical_clock_mapping.json','resolved_config.json','coast_binding.json','arm_science_review.json'):
        payload['artifacts'][name]=sha256_file(arm.run_root/name)
    write_manifest(arm.run_root/'execution_manifest.json',payload)
    if not science['arm_pass']:raise RuntimeError('Scientific arm gate FAIL; STOP no retry or replacement')


def audit_f2_observations(arm):
    """Inspect trusted, actually produced official serialized history; never run."""
    from nuplan.planning.simulation.simulation_log import SimulationLog
    root=arm.run_root;manifest=json.loads((root/'execution_manifest.json').read_text())
    changed=[p for p,h in manifest['artifacts'].items() if sha256_file(root/p)!=h]
    if changed:raise ValueError('Actual immutable arm artifact changed: '+repr(changed))
    if canonical_sha256(manifest['spec'])!=canonical_sha256(arm.spec):raise ValueError('F2 actual spec mismatch')
    files=list((root/'raw/simulation_log').rglob('*.msgpack.xz'))
    if len(files)!=1:raise ValueError('Expected exactly one official simulation log')
    log=SimulationLog.load_data(files[0]);samples=log.simulation_history.data
    mapping=json.loads((root/'canonical_clock_mapping.json').read_text())['rows']
    if len(samples)!=80:raise ValueError('Official serialized history does not contain exactly 80 frames')
    def objects(observation):
        rows=[];future=0
        for obj in observation.tracked_objects.tracked_objects:
            m=obj.metadata;c=obj.box.center
            rows.append((str(m.token),str(m.track_token),int(m.timestamp_us),float(c.x),float(c.y),float(c.heading)))
            if getattr(obj,'predictions',None):future+=1
        return sorted(rows),future
    def lights(items):
        return sorted((str(x.status),str(x.lane_connector_id),int(x.timestamp)) for x in (items or []))
    evidence=[];unexpected=[]
    for i,sample in enumerate(samples):
        expected=mapping[i];raw=int(log.scenario.get_time_point(i).time_us)
        actual,forecast=objects(sample.observation)
        original,original_forecast=objects(log.scenario.get_tracked_objects_at_iteration(i))
        actual_lights=lights(sample.traffic_light_status)
        original_lights=lights(list(log.scenario.get_traffic_light_status_at_iteration(i)))
        ok=(sample.iteration.index==i and sample.iteration.time_us==expected['simulation_time_us']
            and sample.ego_state.time_us==expected['simulation_time_us'] and raw==expected['raw_log_time_us']
            and str(log.scenario._lidarpc_tokens[i])==expected['lidar_token']
            and actual==original and actual_lights==original_lights and forecast==0 and original_forecast==0)
        if not ok:unexpected.append(i)
        evidence.append(dict(observation_index=i,simulation_time_us=sample.ego_state.time_us,raw_log_time_us=raw,
            raw_minus_sim_us=raw-sample.ego_state.time_us,lidar_token=str(log.scenario._lidarpc_tokens[i]),
            tracked_object_count=len(actual),actual_observation_fingerprint=canonical_sha256(actual),
            source_observation_fingerprint=canonical_sha256(original),future_prediction_objects=forecast,
            original_future_prediction_objects=original_forecast,traffic_light_source_equal=actual_lights==original_lights,
            same_index_observation_values_equal=actual==original,status='PASS' if ok else 'FAIL'))
    result=dict(status='PASS' if not unexpected else 'ACTUAL_OBSERVATION_MAPPING_BLOCKED',arm=arm.spec['arm'],
        scope='ACTUAL_OFFICIAL_SERIALIZED_HISTORY_AND_SAME_INDEX_RAW_SOURCE_COMPARISON',
        scientific_reexecution=False,simulation_log_sha256=sha256_file(files[0]),
        execution_manifest_sha256=sha256_file(root/'execution_manifest.json'),changed_artifacts=changed,
        unexpected_indices=unexpected,frames=evidence,
        claim='DISCLOSED_OFFLINE_SAME_INDEX_REPLAY_NOT_STRICT_EVENT_TIME_CAUSALITY')
    target=OUT/'evidence'/('F2_'+arm.spec['arm']+'_Actual_Observation_Audit_v1.json')
    with target.open('x') as f:f.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


def normalized_config(value,arm_root,uid):
    if isinstance(value,dict):return {k:normalized_config(v,arm_root,uid) for k,v in value.items()}
    if isinstance(value,list):return [normalized_config(v,arm_root,uid) for v in value]
    if isinstance(value,str):
        for prefix in (str(Path(arm_root).resolve()),os.path.relpath(arm_root,ROOT)):
            value=value.replace(prefix,'<ARM_OUTPUT_ROOT>')
        return value.replace(uid,'<RUN_START_TIMESTAMP>')
    return value
