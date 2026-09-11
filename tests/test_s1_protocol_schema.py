"""Schema-faithful S1 fixtures: real serializer, recorder and scientific path."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.s1_protocol_schema import (
    SCHEMA, analyze_pair, handcrafted_h, serialize_trace, validate_plan, validate_trace,
)
from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1
from tools.r1_closed_loop_benchmark_v2_1 import (
    calculate_tsb_option_a_v2_timestamp_aware, prospective_primary_f_match,
    trajectory_arrays_timestamp_aware,
)
from tools.r1_context_mechanism_core import qualify_tsb_pair
from tools.r2_bj_b0_2_passive_actual_lqr_recorder import PassiveActualLQRRecorderV1


class LQRTracker:
    _stopping_velocity = 1.0

    def __init__(self):
        self.result = SimpleNamespace(rear_axle_acceleration_2d=SimpleNamespace(x=0.0), tire_steering_rate=0.0)
        self.calls = 0

    def _compute_initial_velocity_and_lateral_state(self, *args):
        return 0.0, [0.0, 0.0, 0.0]

    def _compute_reference_velocity_and_curvature_profile(self, *args):
        return 0.0, [0.0] * 10

    def _stopping_controller(self, *args):
        return 0.0, 0.0

    def track_trajectory(self, *args):
        self.calls += 1
        return self.result


class TwoStageController:
    def __init__(self):
        self._tracker = LQRTracker()


class R1Primary80ScientificTimeControllerV1:
    def number_of_iterations(self):
        return 81


def contract():
    return dict(schema_version='s1_tsb_pair_v0.1', pair_id='SYNTHETIC_PAIR',
                log_id='SYNTHETIC_LOG', scenario_token='SYNTHETIC_TOKEN', family='R-TSB',
                units=json.loads(SCHEMA.read_text())['units'], planner_calls=80,
                runner_report_succeeded=True, pretreatment_clearance=None,
                baseline_context={'pre_context_raw_hash': 'a'*64, 'canonical_context_json_hash': 'b'*64},
                treatment_context={'pre_context_raw_hash': 'a'*64, 'canonical_context_json_hash': 'b'*64})


def trace(arm, initial=10.0, irregular=False):
    time = np.arange(80) / 10
    if irregular:
        time[1::2] += .002
    a = np.zeros(80)
    if arm == 'BASELINE':
        a[(time >= 1.5) & (time < 3.2)] = -1.0
    else:
        a[(time >= 1.5) & (time < 2.5)] = -1.0
        a[(time >= 2.5) & (time < 3.5)] = .3
        a[(time >= 3.5) & (time < 4.5)] = -1.0
    speed = initial + np.r_[0, np.cumsum(a[:-1]*np.diff(time))]
    x = np.r_[0, np.cumsum((speed[:-1]+speed[1:])/2*np.diff(time))]
    result = []
    for i in range(80):
        ego = SimpleNamespace(rear_axle=SimpleNamespace(x=x[i], y=0., heading=0.),
                              dynamic_car_state=SimpleNamespace(speed=speed[i]),
                              time_us=int(1000000 + round(time[i]*1e6)))
        # Exact inherited production current-state serializer, not hand-renamed keys.
        state = R1OfficialTechnicalSmokePlannerV2_1._payload(ego)
        result.append(dict(iteration_index=i, primary_measurement_source='REALIZED_CURRENT_EGO', current_ego=state))
    return result


def write_arm(root, rows, collision=0):
    serialize_trace(root/'trace/realized_current_ego.jsonl', rows)
    metrics = root/'raw/metrics'; metrics.mkdir(parents=True)
    pd.DataFrame({'number_of_all_at_fault_collisions_stat_value': [collision]}).to_parquet(metrics/'no_ego_at_fault_collisions.parquet')
    pd.DataFrame({'drivable_area_compliance_stat_value': [True]}).to_parquet(metrics/'drivable_area_compliance.parquet')
    controller = TwoStageController()
    recorder = PassiveActualLQRRecorderV1(root/'telemetry/actual_lqr_controller_telemetry.jsonl',
                                        {'run_id': 'SYNTHETIC', 'pair_id': 'SYNTHETIC_PAIR', 'arm': 'BASELINE'}, {})
    recorder.install(controller, R1Primary80ScientificTimeControllerV1())
    for i in range(79):
        iteration = SimpleNamespace(index=i, time_point=SimpleNamespace(time_s=i/10))
        returned = controller._tracker.track_trajectory(iteration, None, None, None)
        assert returned is controller._tracker.result
    recorder.validate_complete()
    recorder.uninstall()
    assert controller._tracker.calls == 79


def pair(tmp_path, **kwargs):
    b, t = tmp_path/'b', tmp_path/'t'
    write_arm(b, trace('BASELINE', **kwargs)); write_arm(t, trace('TREATMENT', **kwargs))
    return b, t


def test_end_to_end_production_science_and_h(tmp_path):
    b, t = pair(tmp_path)
    result = analyze_pair(contract(), b, t)
    assert result['evaluation']['mechanism']['pass']
    assert result['evaluation']['f_match']['pass']
    assert result['joint_pass']
    features = handcrafted_h([r['current_ego'] for r in trace('TREATMENT')])
    assert features.shape == (30,) and np.isfinite(features).all()


def test_irregular_physical_time_kept_not_silently_resampled(tmp_path):
    b, t = pair(tmp_path, irregular=True)
    result = analyze_pair(contract(), b, t)
    assert result['evaluation']['mechanism']['pass']
    states = [r['current_ego'] for r in trace('TREATMENT', irregular=True)]
    time, *_ = trajectory_arrays_timestamp_aware(states)
    assert time[1] == pytest.approx(.102)
    h = handcrafted_h(states)
    assert h[-1] == pytest.approx(.002)
    assert np.isfinite(h).all()


@pytest.mark.parametrize('mutation', ['missing', 'wrong', 'nesting', 'timestamp', 'count', 'nan'])
def test_trace_schema_rejects(mutation):
    rows = trace('BASELINE')
    if mutation == 'missing': del rows[0]['current_ego']['speed_mps']
    if mutation == 'wrong': rows[0]['current_ego']['speed'] = rows[0]['current_ego'].pop('speed_mps')
    if mutation == 'nesting': rows[0]['current_ego']['rear_axle'] = [0, 0, 0]
    if mutation == 'timestamp': rows[1]['current_ego']['time_us'] = rows[0]['current_ego']['time_us']
    if mutation == 'count': rows.pop()
    if mutation == 'nan': rows[0]['current_ego']['speed_mps'] = float('nan')
    with pytest.raises(ValueError): validate_trace(rows)



def test_wrong_units_and_missing_context_fail_before_science(tmp_path):
    c = contract(); c['units']['timestamp'] = 's'
    with pytest.raises(ValueError, match='units'): analyze_pair(c, tmp_path, tmp_path)
    c = contract(); c['baseline_context'] = {}
    with pytest.raises(ValueError, match='hash'): analyze_pair(c, tmp_path, tmp_path)


def test_scientific_one_phase_low_speed_safety_fmatch_failures(tmp_path):
    b, t = pair(tmp_path/'one')
    path = t/'trace/realized_current_ego.jsonl'
    path.write_text((b/'trace/realized_current_ego.jsonl').read_text())
    assert not analyze_pair(contract(), b, t)['evaluation']['mechanism']['pass']
    b, t = pair(tmp_path/'low', initial=2.0)
    assert not analyze_pair(contract(), b, t)['joint_pass']
    states = [r['current_ego'] for r in trace('BASELINE', initial=2.0)]
    time, _, _, speed = trajectory_arrays_timestamp_aware(states)
    assert calculate_tsb_option_a_v2_timestamp_aware(time, speed)['status'] == 'LOW_SPEED_ENDSTOP'
    b, t = pair(tmp_path/'unsafe')
    pd.DataFrame({'number_of_all_at_fault_collisions_stat_value': [1]}).to_parquet(t/'raw/metrics/no_ego_at_fault_collisions.parquet')
    assert not analyze_pair(contract(), b, t)['official_safety_pair_pass']
    b, t = pair(tmp_path/'fmatch')
    path = t/'trace/realized_current_ego.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for row in rows: row['current_ego']['speed_mps'] += 2
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    assert not analyze_pair(contract(), b, t)['evaluation']['f_match']['pass']


def test_exact_frozen_numeric_gate_boundaries():
    b = dict(status='OK', brake_phase_count=1)
    t = dict(status='OK', brake_phase_count=2, interstage_release_fraction=.15, second_brake_peak_ratio=.50)
    assert qualify_tsb_pair(b, t)['pass']
    for field in ('interstage_release_fraction', 'second_brake_peak_ratio'):
        bad = dict(t); bad[field] -= .000002
        assert not qualify_tsb_pair(b, bad)['pass']
    # Frozen rounding makes the representable boundary 0.117776, not 0.11777666.
    base = dict(mean_speed=0, end_minus_start_speed=0, path_length=0, mean_abs_accel=0)
    good = dict(base, mean_abs_accel=.117776)
    bad = dict(base, mean_abs_accel=.117777)
    assert prospective_primary_f_match(base, good, 'R-TSB')['pass']
    assert not prospective_primary_f_match(base, bad, 'R-TSB')['pass']


def test_duplicate_budget_and_root(tmp_path):
    c = contract()
    assert validate_plan([c], tmp_path/'fresh', tmp_path, 2)['executed_runs'] == 0
    with pytest.raises(ValueError, match='duplicate'): validate_plan([c, copy.deepcopy(c)], tmp_path/'fresh', tmp_path, 4)
    with pytest.raises(ValueError, match='budget'): validate_plan([c], tmp_path/'fresh', tmp_path, 0)
    with pytest.raises(ValueError, match='output-root'): validate_plan([c], tmp_path.parent, tmp_path, 2)


def test_controller_cardinality_and_production_schema_parity(tmp_path):
    b, t = pair(tmp_path)
    path = b/'telemetry/actual_lqr_controller_telemetry.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError, match='79 transitions'): analyze_pair(contract(), b, t)
    state = trace('BASELINE')[0]['current_ego']
    assert set(state) == set(json.loads(SCHEMA.read_text())['state_keys'])


def test_corrupt_serialized_key_rejected_on_real_analyzer_entry(tmp_path):
    b, t = pair(tmp_path)
    path = t/'trace/realized_current_ego.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[3]['current_ego']['velocity_mps'] = rows[3]['current_ego'].pop('speed_mps')
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError, match='current_ego'):
        analyze_pair(contract(), b, t)


def test_frozen_low_speed_sample_count_boundary():
    time = np.arange(80)/10
    four = np.full(80, 2.0); four[30:34] = .9
    five = np.full(80, 2.0); five[30:35] = .9
    assert calculate_tsb_option_a_v2_timestamp_aware(time, four)['status'] != 'LOW_SPEED_ENDSTOP'
    assert calculate_tsb_option_a_v2_timestamp_aware(time, five)['status'] == 'LOW_SPEED_ENDSTOP'


def test_each_fmatch_rounded_boundary():
    limits = {'mean_speed': .708203939, 'end_minus_start_speed': .978755681,
              'path_length': 5.38423459, 'mean_abs_accel': .11777666}
    base = dict.fromkeys(limits, 0.0)
    for key, limit in limits.items():
        accepted = np.floor(limit*1e6)/1e6
        assert prospective_primary_f_match(base, dict(base, **{key: accepted}), 'R-TSB')['pass']
        assert not prospective_primary_f_match(base, dict(base, **{key: accepted+.000001}), 'R-TSB')['pass']
