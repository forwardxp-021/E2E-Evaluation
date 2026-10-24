"""Canonical clock and unchanged observations, with no scientific runner."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from tools.tsb_coast_p1r_c_fixture import SyntheticClockScenario,run_fixture,make_state,TimePoint,ANCHOR
from tools.tsb_coast_p1r_c_clock import CoastCanonicalTimeControllerV1
from tools.tsb_coast_p1r_c_production import (ROOT,STAGE,build_canonical_coast_arm,
    validate_canonical_production,validate_scientific_credential,validate_clock_binding)


@pytest.fixture(scope='module')
def response(tmp_path_factory):
    return run_fixture(tmp_path_factory.mktemp('canonical_model')/'transitions.jsonl')


def test_81_interfaces_80_rows_79_transitions_and_boundaries():
    scenario=SyntheticClockScenario();clock=CoastCanonicalTimeControllerV1(scenario)
    raw=[scenario.get_time_point(i).time_us for i in range(81)]
    assert clock.number_of_iterations()==81 and clock.require_mapping()['status']=='PASS'
    for i in range(81):
        assert clock.simulation_time(i).time_us==ANCHOR+i*100000
    used=[];transitions=0
    while not clock.reached_end():
        now=clock.get_iteration();used.append(now)
        nxt=clock.next_iteration()
        if nxt:
            assert nxt.time_us-now.time_us==100000;transitions+=1
    assert [x.index for x in used]==list(range(80)) and transitions==79
    clock.reset();assert clock.get_iteration().time_us==raw[0]
    assert clock.simulation_time(21).time_us-ANCHOR==2100000
    assert clock.simulation_time(39).time_us-ANCHOR==3900000
    assert raw==[scenario.get_time_point(i).time_us for i in range(81)]


@pytest.mark.parametrize('index',[-1,81,1.5,True])
def test_invalid_clock_indices_fail_closed(index):
    with pytest.raises(ValueError):CoastCanonicalTimeControllerV1(SyntheticClockScenario()).simulation_time(index)


def test_same_index_cell_derivation_and_positive_negative_skew_disclosed():
    proof=CoastCanonicalTimeControllerV1(SyntheticClockScenario()).mapping_report()
    assert proof['positive_skew_count']>0 and proof['negative_skew_count']>0
    assert proof['nominal_half_period_us']==50000
    assert not proof['strict_event_time_causal_replay'] and not proof['future_frame_selected']
    assert all(r['unique_same_index_cell'] for r in proof['rows'])


@pytest.mark.parametrize('fault',['reordered','duplicate_token','short','wrong_rate','half_period_offset','noninteger_timestamp'])
def test_clock_mapping_faults_are_not_silently_fixed(fault):
    class Fault(SyntheticClockScenario):
        def get_number_of_iterations(self):return 80 if fault=='short' else 81
        def get_time_point(self,index):
            value=super().get_time_point(index)
            if fault=='noninteger_timestamp' and index==2:return TimePoint(float(value.time_us)+.5)
            if fault=='reordered' and index==2:return super().get_time_point(1)
            if fault=='half_period_offset' and index>0:return TimePoint(ANCHOR+index*100000+50000)
            return value
    if fault=='duplicate_token':Fault._lidarpc_tokens=('duplicate',)*81
    if fault=='wrong_rate':Fault.database_interval=.2
    if fault=='half_period_offset':
        clock=CoastCanonicalTimeControllerV1(Fault())
        with pytest.raises(ValueError,match='CLOCK_MAPPING_NOT_ACCEPTABLE'):clock.require_mapping()
    else:
        with pytest.raises(ValueError):CoastCanonicalTimeControllerV1(Fault())


def test_official_observer_uses_same_index_without_retime_or_future_selection():
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    class SentinelScenario(SyntheticClockScenario):
        def __init__(self):self.requests=[];self.values=[object() for _ in range(80)]
        def get_tracked_objects_at_iteration(self,index):self.requests.append(index);return self.values[index]
    scenario=SentinelScenario();clock=CoastCanonicalTimeControllerV1(scenario);observer=TracksObservation(scenario)
    for i in range(80):
        assert observer.get_observation() is scenario.values[i]
        if i<79:
            current=clock.get_iteration();nxt=clock.next_iteration()
            observer.update_observation(current,nxt,None)
    assert scenario.requests==list(range(80)) and observer.scenario is scenario


def test_real_model_states_follow_single_canonical_authority_and_zero_requests(response):
    rows=response['transitions'];states=response['states'];assert len(rows)==79 and len(states)==80
    for i,row in enumerate(rows):
        assert row['time_us']==ANCHOR+i*100000 and row['next_time_us']==states[i+1]['time_us']
        assert row['command_identity_reached_model'] and row['state_propagation_continues']
    coast=rows[21:39];assert len(coast)==18
    assert all(r['tracking_mode']=='COAST' and r['longitudinal_requested_acceleration']==0 and r['coast_command_reached_model'] for r in coast)
    assert coast[0]['model_acceleration_before']<0
    for r in coast:
        assert r['model_realized_acceleration']==pytest.approx((2/3)*r['model_acceleration_before'],abs=1e-14)
    assert rows[39]['tracking_mode']=='BRAKING' and rows[39]['longitudinal_requested_acceleration']<0
    assert response['scientific_attempts']==0 and response['simulation_runner_calls']==0


def test_official_metric_time_extractor_sees_canonical_states_only(response):
    from nuplan.planning.metrics.utils.state_extractors import extract_ego_time_point
    states=[make_state(time_us=r['time_us'],speed=r['speed_mps'],x=r['x']) for r in response['states']]
    np.testing.assert_array_equal(extract_ego_time_point(states),[ANCHOR+i*100000 for i in range(80)])


def test_unchanged_F_under_canonical_physical_time_and_epoch_translation():
    from tools.tsb_coast_p1_measurement import fixture,f_descriptors,F_LIMITS
    from tools.tsb_coast_p1r_analysis import fmatch_pair
    ts,v,xy=fixture();clock=CoastCanonicalTimeControllerV1(SyntheticClockScenario())
    canonical=[clock.simulation_time(i).time_us for i in range(80)]
    assert f_descriptors(ts,v,xy)==f_descriptors(canonical,v,xy)
    assert F_LIMITS==dict(mean_speed=.10,end_minus_start_speed=.15,path_length=.80,mean_abs_accel=.02)
    result=fmatch_pair(canonical,v,xy,v,xy);assert result['pass'] and all(x==0 for x in result['absolute_delta'].values())


def test_planner_full_future_grid_and_common_precontext_from_authority(tmp_path):
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    clock=CoastCanonicalTimeControllerV1(SyntheticClockScenario())
    planners=[CoastC2PlannerV1(dict(arm=a,start_timestamp_us=ANCHOR,route_roadblock_ids=['fixture']),tmp_path/a/'trace',tmp_path/a/'telemetry') for a in ('BASELINE','TREATMENT')]
    for p in planners:p._initialization=SimpleNamespace(map_api=None)
    route=dict(reference_xy=[[0.,0.],[10000.,0.]],current_route_arc_m=0.)
    with patch('tools.tsb_coast_p1r_planner.build_native_route_reference_v1_2',return_value=route):
        for i in range(80):
            now=clock.get_iteration();ego=make_state(time_us=now.time_us,speed=10.,x=0.)
            current=SimpleNamespace(iteration=now,history=SimpleNamespace(current_state=(ego,None)))
            trajectories=[p.compute_trajectory(current).get_sampled_trajectory() for p in planners]
            for trajectory in trajectories:
                assert trajectory[0] is ego and [s.time_us for s in trajectory]==[now.time_us+j*100000 for j in range(80)]
            if i<11:
                np.testing.assert_array_equal([s.dynamic_car_state.speed for s in trajectories[0]], [s.dynamic_car_state.speed for s in trajectories[1]])
            clock.next_iteration()
    for arm in ('BASELINE','TREATMENT'):
        assert len((tmp_path/arm/'trace/realized_current_ego.jsonl').read_text().splitlines())==80


@pytest.mark.parametrize('auth',[{}, {'stage':'TSB_COAST_P1R_FIRST_PAIR','authorization_status':'AUTHORIZED','scientific_owner_approved':True},
    {'stage':STAGE,'authorization_status':'PROPOSED_ONLY','scientific_owner_approved':False}])
def test_approved_amendment_or_old_auth_cannot_authorize_runner(auth):
    with pytest.raises(PermissionError):validate_scientific_credential(auth,{},'never_read','never_read','never_read')


def test_real_canonical_factory_and_old_clock_or_recorder_authority_rejected(tmp_path):
    from tools.b1_production_preflight import zero_run_guard
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1
    spec=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_entry/evidence/Arm_Specs_v1.json').read_text())['arms'][1]
    with zero_run_guard() as calls,patch.object(CoastC2PlannerV1,'initialize',side_effect=AssertionError('No scientific initialize')):
        arm=build_canonical_coast_arm(spec,tmp_path/'canonical_factory')
        assert validate_canonical_production(arm)['status']=='PASS'
        canonical=arm.simulation._time_controller
        arm.simulation._time_controller=R1Primary80ScientificTimeControllerV1(arm.simulation._scenario)
        with pytest.raises(ValueError,match='OLD_OR_UNBOUND_CLOCK'):validate_canonical_production(arm)
        arm.simulation._time_controller=canonical
        arm.recorder._canonical_clock_authority=object()
        with pytest.raises(ValueError,match='RECORDER_TIME_AUTHORITY'):validate_canonical_production(arm)
        arm.recorder._canonical_clock_authority=canonical
        assert validate_canonical_production(arm)['status']=='PASS' and not any(calls.values())


@pytest.mark.parametrize('code',[0,2])
def test_new_supervisor_with_harmless_non_simulation_process_only(tmp_path,monkeypatch,code):
    import subprocess
    import sys
    import tools.tsb_coast_p1r_c_executor as executor
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps(dict(historical_attempts=0,attempts=[],cpu_seconds_consumed=0,stopped=False)))
    spec=dict(arm='BASELINE',run_id='NON_SCIENTIFIC_FIXTURE')
    monkeypatch.setattr(executor,'credential',lambda *a:({},spec,ledger))
    original=subprocess.Popen
    def harmless(command,**kwargs):
        assert '--authorized-child' in command
        return original([sys.executable,'-c',f'raise SystemExit({code})'],**kwargs)
    monkeypatch.setattr(executor.subprocess,'Popen',harmless)
    if code:
        with pytest.raises(RuntimeError):executor.launch_first_pair_arm('not_read',spec['run_id'],tmp_path/'process')
    else:
        executor.launch_first_pair_arm('not_read',spec['run_id'],tmp_path/'process')
    state=json.loads(ledger.read_text())
    assert len(state['attempts'])==1 and state['cpu_seconds_consumed']>0 and state['stopped']==bool(code)


def test_approved_contract_record_and_proposed_file_are_not_execution_credentials(tmp_path):
    from tools import tsb_coast_p1r_c_executor as executor
    record=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_clock_closure/P1R_C_Owner_Approved_Scientific_Amendment_v1.json').read_text())
    with pytest.raises(PermissionError):validate_scientific_credential(record,{},'never_read','never_read','never_read')
    proposed=tmp_path/'proposed.json';proposed.write_text(json.dumps(dict(stage=STAGE,authorization_status='PROPOSED_ONLY',scientific_owner_approved=False)))
    with patch.object(executor,'claim_attempt',side_effect=AssertionError('Budget must not be claimed')):
        with pytest.raises(PermissionError):executor.launch_first_pair_arm(proposed,'P1R-C2-DEV-01-BASELINE',tmp_path/'unused')
    assert not (tmp_path/'unused').exists()


def test_future_credential_validation_is_hash_bound_and_acknowledges_skew(tmp_path):
    from tools.s2r_production_executor import sha256_file,canonical_sha256
    out=ROOT/'docs/stageR/tsb_coast_p1r_clock_closure'
    contract=out/'P1R_C_Owner_Approved_Scientific_Amendment_v1.json'
    proof=out/'evidence/production_verified/ThreePair_Preflight_v1.json'
    source=json.loads((out/'P1R_C_Execution_Source_Binding_v1.json').read_text())
    # A disposable VALIDATION fixture only; never passed to a scientific runner.
    source['source_config_sha256']={p:sha256_file(ROOT/p) for p in source['source_config_sha256']}
    binding=tmp_path/'source_binding_fixture.json';binding.write_text(json.dumps(source))
    spec=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_entry/evidence/Arm_Specs_v1.json').read_text())['arms'][0]
    auth=dict(stage=STAGE,authorization_status='AUTHORIZED',scientific_owner_approved=True,
        disclosed_observation_skew_acknowledged=True,approved_amendment_sha256=sha256_file(contract),
        execution_source_binding_sha256=sha256_file(binding),production_preflight_sha256=sha256_file(proof),
        approved_run_ids=['P1R-C2-DEV-01-BASELINE','P1R-C2-DEV-01-TREATMENT'],
        approved_spec_sha256={spec['run_id']:canonical_sha256(spec)})
    validate_scientific_credential(auth,spec,contract,binding,proof)
    with pytest.raises(PermissionError,match='acknowledge'):
        validate_scientific_credential({**auth,'disclosed_observation_skew_acknowledged':False},spec,contract,binding,proof)
    with pytest.raises(PermissionError,match='spec changed'):
        validate_scientific_credential(auth,{**spec,'start_timestamp_us':spec['start_timestamp_us']+1},contract,binding,proof)
    source['source_config_sha256'].pop('tools/tsb_coast_p1r_c_clock.py')
    binding.write_text(json.dumps(source));auth['execution_source_binding_sha256']=sha256_file(binding)
    with pytest.raises(PermissionError,match='source binding incomplete'):
        validate_scientific_credential(auth,spec,contract,binding,proof)
