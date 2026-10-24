"""Zero-runtime source bindings and analytic identities, never model propagation."""
import ast
import math
from pathlib import Path

import pytest

from tools import tsb_coast_c0_actuation_audit as c


def test_production_configuration_and_constructor_defaults_match_source():
    audit = c.audit()
    assert len(audit['production_bindings']) == 6
    assert audit['code']['model_constructor_defaults']['accel_time_constant'] == .2
    for b in audit['production_bindings']:
        assert b['dt_s'] == .1 and b['horizon_s'] == 1.
        assert b['accel_tau_s'] == .2 and b['q'] == 10. and b['r'] == 1.
        assert b['tau_origin'] == 'source constructor default'


def test_zero_command_is_not_zero_acceleration_after_nonzero_history():
    result = c.zero_command_algebra(.1, .2, 1)
    assert result['acceleration_ratio_to_initial'] == pytest.approx(2/3)
    assert result['speed_change_per_initial_acceleration_s'] == pytest.approx(.1*2/3)


def test_closed_form_residual_equals_independent_geometric_sum():
    for n in (0, 1, 18):
        result = c.zero_command_algebra(.1, .2, n)
        expected = math.fsum(.1*(2/3)**j for j in range(1, n+1))
        assert result['speed_change_per_initial_acceleration_s'] == pytest.approx(expected)
    assert c.zero_command_algebra(.1, .2, 18)['acceleration_ratio_to_initial'] > 0


def test_zero_lag_and_zero_history_limits_do_not_invent_drag():
    assert c.zero_command_algebra(.1, 0., 1)['acceleration_ratio_to_initial'] == 0.
    assert c.zero_command_algebra(.1, 0., 1)['speed_change_per_initial_acceleration_s'] == 0.
    # With a0=0 the entire zero-command response remains zero by linearity.
    assert 0.*c.zero_command_algebra(.1, .2, 18)['speed_change_per_initial_acceleration_s'] == 0.


def test_velocity_tracking_can_request_positive_negative_or_zero_control():
    assert c.lqr_algebra(1.,1.,10.,1.) == pytest.approx(10/11)
    assert c.lqr_algebra(-1.,1.,10.,1.) == pytest.approx(-10/11)
    assert c.lqr_algebra(0.,1.,10.,1.) == 0.


def test_two_stage_source_passes_tracker_result_to_model_without_actuator_split():
    node = c.method(c.DEVKIT/c.CONTROLLER,'TwoStageController','update_state')
    calls = {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}
    assert calls == {'self._tracker.track_trajectory','self._motion_model.propagate_state'}
    source = ast.unparse(node)
    assert 'ideal_dynamic_state=dynamic_state' in source


def test_dynamic_state_has_kinematics_but_no_drive_brake_or_torque_channel():
    fields = c.source_contract()['dynamic_state_constructor_fields']
    assert fields == ['rear_axle_to_center_dist','rear_axle_velocity_2d','rear_axle_acceleration_2d',
                      'angular_velocity','angular_acceleration','tire_steering_rate']


def test_recorder_observes_tracker_return_not_post_model_acceleration():
    node = c.method(c.ROOT/'tools/r2_bj_b0_2_passive_actual_lqr_recorder.py','PassiveActualLQRRecorderV1','install')
    source = ast.unparse(node)
    assert 'result.rear_axle_acceleration_2d.x' in source
    calls = {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}
    assert not any('propagate_state' in x for x in calls)
    assert 'return result' in source


@pytest.mark.parametrize('args',[(0.,.2,1),(.1,-.2,1),(.1,.2,-1),(.1,.2,True),(math.nan,.2,1)])
def test_invalid_algebra_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        c.zero_command_algebra(*args)


def test_audit_does_not_import_or_call_any_nuplan_runtime():
    tree = ast.parse(Path(c.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            assert all(not x.name.startswith('nuplan') for x in n.names)
        if isinstance(n,ast.ImportFrom):
            assert n.module in {'__future__','pathlib'}
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls) & {'run','track_trajectory','propagate_state','compute_planner_trajectory','update_state'}
