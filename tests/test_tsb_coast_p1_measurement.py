"""P1 measurement boundaries and independent error-resolution counterexamples."""
import ast
from pathlib import Path

import numpy as np
import pytest

from tools import tsb_coast_p1_measurement as m


def test_analytic_coast_and_baseline_are_measurable_not_closed_loop_proofs():
    for arm, expected in [('BASELINE', 1), ('TREATMENT', 2)]:
        ts, v, _ = m.fixture(arm)
        result = m.measure(ts, v, arm)
        assert result['status'] == 'PASS'
        assert len(result['phases']) == expected
        assert m.stability(ts, v, arm)['status'] == 'STABLE'
    assert result['gap']['core']['support_s'] == pytest.approx(1.6)
    assert result['gap']['core']['raw_speed_drift_mps'] == pytest.approx(0.)


@pytest.mark.parametrize('drag', [-.10, -.20])
def test_coast_accepts_small_negative_drift_without_speed_recovery(drag):
    ts, v, _ = m.fixture(middle_accel=drag)
    result = m.measure(ts, v, 'TREATMENT')
    assert result['status'] == 'PASS'
    assert result['gap']['core']['raw_speed_drift_mps'] < 0


def test_near_zero_negative_boundary_is_closed_and_outside_is_rejected():
    ts, v, _ = m.fixture(middle_accel=-.200001)
    assert 'COAST_CORE_TOO_SHORT' in m.measure(ts, v, 'TREATMENT')['reasons']


@pytest.mark.parametrize('recovery', [.20, .401, 1.4])
def test_positive_recovery_is_not_relabelled_coast(recovery):
    ts, v, _ = m.fixture(middle_accel=recovery)
    result = m.measure(ts, v, 'TREATMENT')
    assert 'POSITIVE_RECOVERY_EXCEEDED' in result['reasons']
    assert result['gap']['positive_recovery_excluded']


def test_sample_support_is_last_minus_first_not_sample_count_times_dt():
    t = np.arange(80)/10
    assert t[8]-t[5] == pytest.approx(.3)
    assert m.ranges(np.array([False, True, True, True, True, False])) == [(1, 4)]
    # Four samples span .3 seconds and cannot establish a .4-second brake.
    assert (t[8]-t[5]) < .4


def test_epoch_shift_and_one_microsecond_quantization_do_not_create_a_result():
    ts, v, _ = m.fixture()
    nominal = m.measure(ts, v, 'TREATMENT')
    shifted = m.measure([x+987654321000000 for x in ts], v, 'TREATMENT')
    assert nominal == shifted
    jittered = m.measure([x+(i % 2) for i, x in enumerate(ts)], v, 'TREATMENT')
    assert jittered['status'] == nominal['status']
    np.testing.assert_allclose(jittered['acceleration_mps2'], nominal['acceleration_mps2'], atol=3e-5)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'nonfinite', 'cadence', 'float_time'])
def test_invalid_input_fails_closed(kind):
    ts, v, _ = m.fixture()
    if kind == 'missing':
        ts, v = ts[:-1], v[:-1]
    elif kind == 'duplicate':
        ts[10] = ts[9]
    elif kind == 'nonfinite':
        v[30] = np.nan
    elif kind == 'cadence':
        ts[10] += 100
    else:
        ts = [float(x) for x in ts]
    with pytest.raises(ValueError, match='MEASUREMENT_INVALID'):
        m.measure(ts, v, 'TREATMENT')


def test_low_speed_and_floor_flags_cannot_be_coast_success():
    ts, v, _ = m.fixture()
    assert 'LOW_SPEED_ENDSTOP' in m.measure(ts, v, 'TREATMENT', floor_triggered=True)['reasons']
    assert 'LOW_SPEED_ENDSTOP' in m.measure(ts, np.full(80, .5), 'TREATMENT')['reasons']


def test_shallow_oscillation_is_not_a_qualified_brake():
    ts, _, _ = m.fixture()
    t = np.arange(80)/10
    # A small smooth perturbation, with acceleration far below a real brake.
    result = m.measure(ts, 10+.01*np.sin(2*np.pi*t), 'TREATMENT')
    assert len(result['phases']) == 0
    assert 'TWO_BRAKE_PHASES_NOT_REALIZED' in result['reasons']


def test_single_brake_cannot_be_treatment_and_short_gap_cannot_be_coast():
    ts, v, _ = m.fixture('BASELINE')
    assert 'TWO_BRAKE_PHASES_NOT_REALIZED' in m.measure(ts, v, 'TREATMENT')['reasons']
    t = np.arange(80)/10
    segments = [(1.1, 2.1, -1.305), (2.4, 3.4, -1.305)]
    v = np.array([m.speed_at(segments, x) for x in t])
    result = m.measure(ts, v, 'TREATMENT')
    assert result['status'] == 'FAIL'
    assert set(result['reasons']) & {'INTERBRAKE_GAP_TOO_SHORT', 'COAST_CORE_TOO_SHORT'}


def test_half_sample_reference_boundary_does_not_create_extra_brakes():
    ts, _, _ = m.fixture()
    t = np.arange(80)/10
    segments = [(1.15, 2.15, -1.305), (3.95, 4.95, -1.305)]
    v = np.array([m.speed_at(segments, x) for x in t])
    result = m.measure(ts, v, 'TREATMENT')
    assert result['status'] == 'PASS'
    assert len(result['phases']) == 2
    assert m.stability(ts, v, 'TREATMENT')['status'] == 'STABLE'


def test_f_definitions_match_frozen_production_on_synthetic_states():
    from tools.r1_closed_loop_benchmark_v2_1 import trajectory_descriptors_timestamp_aware
    for arm in ('BASELINE', 'TREATMENT'):
        ts, v, xy = m.fixture(arm)
        states = [dict(time_us=ts[i], speed_mps=float(v[i]),
                       rear_axle=dict(x=float(xy[i, 0]), y=0., heading=0.)) for i in range(80)]
        historical = trajectory_descriptors_timestamp_aware(states)
        assert m.f_descriptors(ts, v, xy) == {k:historical[k] for k in m.F_LIMITS}


def test_constant_speed_noise_counterexample_matches_hand_derivative():
    ts, _, _ = m.fixture()
    t = np.arange(80)/10
    xy = np.column_stack([10*t, np.zeros(80)])
    noise = np.tile([.01, .01, -.01, -.01], 20)
    f = m.f_descriptors(ts, 10+noise, xy)
    # Interior centered differences are ±.1 m/s²; edge_order=2 gives the same magnitude.
    assert f['mean_abs_accel'] == pytest.approx(.1)
    assert m.F_LIMITS['mean_abs_accel'] == .02
    assert abs(noise).max() <= .01


def test_resolution_failure_is_recorded_and_not_threshold_adaptation():
    audit = m.resolution_audit()
    assert audit['status'] == 'PREFLIGHT_BLOCKED'
    assert audit['failed_case_count'] == audit['total_case_count'] == 16
    assert audit['max_F_error']['mean_abs_accel'] == pytest.approx(.0825)
    assert audit['runner_calls'] == 0
    assert not audit['actual_nuplan_speed_error_bound_established']


def test_no_runner_or_detector_import_or_invocation_in_measurement_tool():
    tree = ast.parse(Path(m.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert set(imports) <= {'__future__', 'pathlib', 'tools.r1_context_mechanism_core', 'tools.tsb_coast_p0_kinematics'}
    called_attributes = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(called_attributes) & {'run', 'compute_planner_trajectory', 'track_trajectory', 'propagate_state'}
