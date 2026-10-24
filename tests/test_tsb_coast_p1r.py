"""Official isolated model, recorder, unchanged F and closed execution gates."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from tools.tsb_coast_p1r_model_fixture import (run_fixture,make_state,make_tracker,KinematicBicycleModel,get_pacifica_parameters,TimePoint)
from tools.tsb_coast_p1r_recorder import fp_bound,CoastTransitionRecorderV1
from tools.tsb_coast_p1r_planner import reference_profile
from tools.tsb_coast_p1r_analysis import metrology_model_internal_bound,fmatch_pair
from tools.tsb_coast_p1r_production import validate_authorization,claim_attempt


@pytest.fixture(scope='module')
def response(tmp_path_factory):
    return run_fixture(tmp_path_factory.mktemp('p1r_isolated')/'transitions.jsonl')


def test_actual_official_model_one_step_retains_negative_accel_and_moves():
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.state_representation import StateVector2D
    state=make_state();model=KinematicBicycleModel(get_pacifica_parameters())
    command=DynamicCarState.build_from_rear_axle(1.,StateVector2D(10.,0.),StateVector2D(0.,0.),tire_steering_rate=.1)
    output=model.propagate_state(state,command,TimePoint(100000))
    assert state.dynamic_car_state.rear_axle_acceleration_2d.x==-1.
    assert output.dynamic_car_state.rear_axle_acceleration_2d.x==pytest.approx(-2/3)
    assert output.dynamic_car_state.speed==pytest.approx(10.-.1*2/3)
    assert output.time_us==state.time_us+100000 and output.rear_axle.x>state.rear_axle.x
    assert np.isfinite(output.dynamic_car_state.tire_steering_rate) and output.dynamic_car_state.tire_steering_rate!=0


def test_real_model_79_transitions_and_18_coast_commands_reach_model(response):
    rows=response['transitions'];assert len(rows)==79 and len(response['states'])==80
    coast=[r for r in rows if r['tracking_mode']=='COAST'];assert len(coast)==18
    assert all(r['longitudinal_requested_acceleration']==0 and r['coast_command_reached_model'] for r in coast)
    assert all(r['state_propagation_continues'] and r['command_identity_reached_model'] for r in rows)
    assert coast[0]['model_acceleration_before']<0
    for r in coast:
        assert r['model_realized_acceleration']==pytest.approx((2/3)*r['model_acceleration_before'],abs=1e-14)
        assert r['realized_speed']<r['speed_before']
    assert abs(coast[-1]['model_realized_acceleration'])<abs(coast[0]['model_realized_acceleration'])


def test_exit_restores_real_longitudinal_braking_and_integer_clock(response):
    rows=response['transitions'];assert rows[20]['tracking_mode']=='BRAKING'
    assert rows[21]['tracking_mode']=='COAST' and rows[38]['tracking_mode']=='COAST'
    assert rows[39]['tracking_mode']=='BRAKING' and rows[39]['longitudinal_requested_acceleration']<0
    assert all(x['next_time_us']-x['time_us']==100000 for x in rows)


def test_model_speed_difference_and_raw_central_gradient_are_distinct(response):
    for r in response['transitions']:
        assert abs(r['speed_difference_acceleration']-r['model_realized_acceleration'])<2e-13
    v=np.array([s['speed_mps'] for s in response['states']]);a=np.gradient(v,.1,edge_order=2)
    assert not np.allclose(a[1:], [r['model_realized_acceleration'] for r in response['transitions']])


def test_isolated_model_is_numerically_deterministic(response,tmp_path):
    repeated=run_fixture(tmp_path/'second.jsonl')
    assert repeated==response


def test_no_adapter_on_baseline_and_strict_type_rejection(tmp_path):
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    model=KinematicBicycleModel(get_pacifica_parameters());base=make_tracker()
    controller=TwoStageController(None,base,model)
    recorder=CoastTransitionRecorderV1(tmp_path/'base.jsonl','BASELINE',0);recorder.install(controller)
    assert controller._tracker is base
    with pytest.raises(ValueError):CoastTransitionRecorderV1(tmp_path/'again.jsonl','BASELINE',0).install(controller)


@pytest.mark.parametrize('arm',['BASELINE','TREATMENT'])
def test_C2_reference_exact_impulse_and_common_future(arm):
    b=reference_profile(10.,0,'BASELINE',True);t=reference_profile(10.,0,'TREATMENT',True)
    np.testing.assert_array_equal(b[0],t[0]);np.testing.assert_array_equal(b[1],t[1])
    speed,distance,command=reference_profile(10.,1100000,arm)
    assert speed[-1]-speed[0]==pytest.approx(-2.61)
    assert np.max(command)<=0


def test_unchanged_F_deterministic_pair_difference_and_actual_numeric_bound(response):
    states=response['states'];times=[s['time_us'] for s in states];v=[s['speed_mps'] for s in states];xy=[[s['x'],s['y']] for s in states]
    result=fmatch_pair(times,v,xy,v,xy);assert result['pass'] and result['absolute_delta']['mean_abs_accel']==0
    bound=metrology_model_internal_bound(v,times)
    assert bound['pair_mean_abs_accel_roundoff_and_rounding_bound_mps2']<1.0001e-6
    assert bound['F_caliper_unchanged']==.02
    serialized=json.loads(json.dumps(v));assert serialized==v


@pytest.mark.parametrize('auth',[{}, {'stage':'B1_FROZEN_TSB_QUALIFICATION','authorization_status':'AUTHORIZED'},
    {'stage':'TSB_COAST_P1R_FIRST_PAIR','authorization_status':'PROPOSED_ONLY','scientific_owner_approved':False}])
def test_old_or_proposed_authorization_rejected_before_model_or_budget(auth):
    with pytest.raises(PermissionError):validate_authorization(auth,{},Path('never_read_contract'),Path('never_read_roster'))


def test_budget_counts_persist_and_no_automatic_retry(tmp_path):
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps({'historical_attempts':0,'attempts':[],'cpu_seconds_consumed':0,'stopped':False}))
    claim_attempt(ledger,'one');stored=json.loads(ledger.read_text());assert len(stored['attempts'])==1
    with pytest.raises(PermissionError):claim_attempt(ledger,'one')
    with pytest.raises(PermissionError):claim_attempt(ledger,'two')
    assert json.loads(ledger.read_text())==stored


def test_child_cannot_bypass_live_parent_monitor(tmp_path):
    from tools.tsb_coast_p1r_production import validate_child_claim
    ledger=tmp_path/'claim.json';ledger.write_text(json.dumps({'historical_attempts':0,'attempts':[],'cpu_seconds_consumed':0,'stopped':False}))
    ticket=claim_attempt(ledger,'one')
    with pytest.raises(PermissionError):validate_child_claim(ledger,'one',ticket)
    with pytest.raises(PermissionError):validate_child_claim(ledger,'one','wrong')


@pytest.mark.parametrize('field,value',[('historical_attempts',8),('cpu_seconds_consumed',10800),('stopped',True)])
def test_cumulative_budget_and_stop_gate(tmp_path,field,value):
    ledger=tmp_path/'budget.json';payload={'historical_attempts':0,'attempts':[],'cpu_seconds_consumed':0,'stopped':False};payload[field]=value
    ledger.write_text(json.dumps(payload))
    with pytest.raises(PermissionError):claim_attempt(ledger,'one')
    assert json.loads(ledger.read_text())==payload


def test_planner_public_dispatch_writes_exactly_one_realized_row_and_rejects_duplicate(tmp_path):
    from types import SimpleNamespace
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from tools.tsb_coast_p1r_model_fixture import ANCHOR
    planner=CoastC2PlannerV1({'arm':'TREATMENT','start_timestamp_us':ANCHOR,'route_roadblock_ids':['fixture']},tmp_path/'trace',tmp_path/'telemetry')
    planner._initialization=SimpleNamespace(map_api=None)
    ego=make_state();current=SimpleNamespace(iteration=SimulationIteration(TimePoint(ANCHOR),0),history=SimpleNamespace(current_state=(ego,None)))
    route={'reference_xy':[[0.,0.],[200.,0.]],'current_route_arc_m':0.}
    with patch('tools.tsb_coast_p1r_planner.build_native_route_reference_v1_2',return_value=route):
        trajectory=planner.compute_trajectory(current)
        assert trajectory.get_sampled_trajectory()[0] is ego
        assert len((tmp_path/'trace/realized_current_ego.jsonl').read_text().splitlines())==1
        with pytest.raises(ValueError):planner.compute_trajectory(current)
        assert len((tmp_path/'trace/realized_current_ego.jsonl').read_text().splitlines())==1


def test_recorder_rejects_unobserved_model_command_and_missing_transitions(tmp_path):
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.state_representation import StateVector2D
    model=KinematicBicycleModel(get_pacifica_parameters());base=make_tracker();controller=TwoStageController(None,base,model)
    rec=CoastTransitionRecorderV1(tmp_path/'fault.jsonl','BASELINE',0);rec.install(controller)
    command=DynamicCarState.build_from_rear_axle(1.,StateVector2D(10.,0.),StateVector2D(0.,0.))
    with pytest.raises(ValueError):model.propagate_state(make_state(),command,TimePoint(100000))
    with pytest.raises(ValueError):rec.validate_complete()
    assert rec.rows==[]


def test_real_uninitialized_factory_callback_and_recorder_faults_fail_closed(tmp_path):
    from tools.tsb_coast_p1r_production import build_coast_arm,validate_callbacks,audit_episode_clock,require_exact_episode_clock,ROOT
    from tools.tsb_coast_p1r_planner import CoastC2PlannerV1
    from tools.b1_production_preflight import zero_run_guard
    spec=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_entry/evidence/Arm_Specs_v1.json').read_text())['arms'][1]
    with zero_run_guard() as calls,patch.object(CoastC2PlannerV1,'initialize',side_effect=AssertionError('No initialize')),patch.object(CoastC2PlannerV1,'compute_planner_trajectory',side_effect=AssertionError('No planner')):
        arm=build_coast_arm(spec,tmp_path/'uninitialized');assert validate_callbacks(arm)['status']=='PASS'
        clock=audit_episode_clock(arm);assert clock['status']=='BLOCKED_PHYSICAL_CLOCK_CONTRACT'
        assert clock['max_abs_deviation_us']>0 and clock['cross_boundary_steps']
        with pytest.raises(ValueError,match='CLOCK_CONTRACT_BLOCKED'):require_exact_episode_clock(arm)
        callback=arm.common_builder.multi_main_callback
        saved=callback._main_callbacks;callback._main_callbacks=[]
        with pytest.raises(ValueError,match='callback owners'):validate_callbacks(arm)
        callback._main_callbacks=saved
        method=arm.motion_model.propagate_state;arm.motion_model.propagate_state=lambda *a:None
        with pytest.raises(ValueError,match='binding displaced'):validate_callbacks(arm)
        arm.motion_model.propagate_state=method
        model=arm.controller._motion_model;arm.controller._motion_model=object()
        with pytest.raises(ValueError,match='binding displaced'):validate_callbacks(arm)
        arm.controller._motion_model=model
        assert validate_callbacks(arm)['status']=='PASS' and not any(calls.values())


def test_pending_scientific_contract_blocks_otherwise_active_authorization(tmp_path):
    contract=tmp_path/'contract.json';contract.write_text(json.dumps({'owner_approval':'PENDING'}))
    active={'stage':'TSB_COAST_P1R_FIRST_PAIR','authorization_status':'AUTHORIZED','scientific_owner_approved':True}
    with pytest.raises(PermissionError,match='amendments not approved'):
        validate_authorization(active,{},contract,tmp_path/'unread_roster')


@pytest.mark.parametrize('return_code',[0,2])
def test_real_process_budget_monitor_on_harmless_fixture_only(tmp_path,monkeypatch,return_code):
    import subprocess
    import sys
    import tools.tsb_coast_p1r_production as production
    auth=tmp_path/'auth.json';auth.write_text(json.dumps({'execution_ledger_path':str((tmp_path/'ledger.json').resolve())}))
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'run_id':'fixture-only','arm':'BASELINE'}))
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'historical_attempts':0,'attempts':[],'cpu_seconds_consumed':0,'stopped':False}))
    monkeypatch.setattr(production,'validate_authorization',lambda *a:None)
    original=subprocess.Popen
    def harmless_process(command,**kwargs):
        assert '--authorized-child' in command
        return original([sys.executable,'-c',f'raise SystemExit({return_code})'],**kwargs)
    monkeypatch.setattr(production.subprocess,'Popen',harmless_process)
    if return_code:
        with pytest.raises(RuntimeError):production.launch_authorized(auth,spec,tmp_path/'unused',tmp_path/'unused',ledger,tmp_path/'process')
    else:production.launch_authorized(auth,spec,tmp_path/'unused',tmp_path/'unused',ledger,tmp_path/'process')
    stored=json.loads(ledger.read_text());assert len(stored['attempts'])==1
    assert stored['cpu_seconds_consumed']>0 and stored['stopped']==bool(return_code)


@pytest.mark.parametrize('fault',['packet','filter','clock','missing'])
def test_layered_proposed_analyzer_rejects_faults_without_a_scientific_run(response,fault):
    from tools.tsb_coast_p1r_analysis import audit_proxy_response
    rows=copy.deepcopy(response['transitions']);states=response['states']
    ts=[s['time_us'] for s in states];v=[s['speed_mps'] for s in states]
    assert audit_proxy_response(ts,v,rows,'TREATMENT')['proposal_diagnostic_pass']
    if fault=='packet':rows[21]['coast_command_reached_model']=False
    elif fault=='filter':rows[21]['model_realized_acceleration']+=.1
    elif fault=='clock':rows[21]['next_time_us']+=1
    else:rows.pop()
    result=audit_proxy_response(ts,v,rows,'TREATMENT')
    assert not result['proposal_diagnostic_pass'] and result['approval']=='PENDING_OWNER'


def test_exact_future_authorization_binding_and_spec_tamper_rejected_without_execution(tmp_path):
    from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
    from tools.s2r_production_executor import ROOT,DEVKIT,sha256_file,canonical_sha256
    contract=tmp_path/'approved_fixture_contract.json';contract.write_text(json.dumps({'owner_approval':'APPROVED','specification':SCIENCE_CONTRACT}))
    roster=tmp_path/'roster_fixture.csv';roster.write_text('fixture only')
    proof=tmp_path/'proof_fixture.json';proof.write_text(json.dumps({'status':'PASS','arms':[{'episode_clock':{'status':'PASS'}} for _ in range(6)]}))
    spec=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_entry/evidence/Arm_Specs_v1.json').read_text())['arms'][0]
    sources=[*ROOT.glob('tools/*.py'),*(ROOT/'configs/r1_official_technical_smoke_hydra').rglob('*.yaml'),*(DEVKIT/'nuplan/planning/script/config').rglob('*.yaml'),*(DEVKIT/'nuplan/planning/simulation/controller').rglob('*.py')]
    import os
    auth={'stage':'TSB_COAST_P1R_FIRST_PAIR','authorization_status':'AUTHORIZED','scientific_owner_approved':True,
        'contract_sha256':sha256_file(contract),'roster_sha256':sha256_file(roster),'production_preflight_path':str(proof),
        'production_preflight_sha256':sha256_file(proof),'approved_run_ids':['P1R-C2-DEV-01-BASELINE','P1R-C2-DEV-01-TREATMENT'],
        'approved_spec_sha256':{spec['run_id']:canonical_sha256(spec)},
        'source_file_sha256':{os.path.relpath(p,ROOT):sha256_file(p) for p in sources}}
    binding=json.loads((ROOT/'docs/stageR/tsb_coast_p1r_entry/evidence/Final_Binding_v1.json').read_text())
    auth['data_file_sha256']={**binding['metadata_data_sha256'],**binding['map_data_sha256']}
    validate_authorization(auth,spec,contract,roster)
    changed={**spec,'scenario_token':'tampered'}
    with pytest.raises(PermissionError,match='spec binding'):validate_authorization(auth,changed,contract,roster)
