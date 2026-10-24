"""Synthetic interface tests, no scientific scenario or official model rollout."""
from unittest.mock import Mock

import numpy as np
import pytest

from tools.tsb_coast_zero_command_tracker_v1 import ZeroLongitudinalRequestTrackerV1
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


ANCHOR = 1_650_000_000_000_000
INTERVALS = [(2_100_000, 3_900_000)]  # Historical C2 timing, synthetic tests only.


def tracker():
    return LQRTracker(q_longitudinal=[10.], r_longitudinal=[1.],
                      q_lateral=[1.,10.,0.], r_lateral=[1.], discretization_time=.1,
                      tracking_horizon=10, jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                      stopping_proportional_gain=.5, stopping_velocity=.2)


def state(time_us, velocity=10., accel=-1., x=0.):
    return EgoState.build_from_rear_axle(
        StateSE2(x,0.,0.), StateVector2D(velocity,0.), StateVector2D(accel,0.),
        tire_steering_angle=0., time_point=TimePoint(time_us), vehicle_parameters=get_pacifica_parameters())


def iterations(elapsed):
    now = SimulationIteration(TimePoint(ANCHOR+elapsed), 0)
    nxt = SimulationIteration(TimePoint(ANCHOR+elapsed+100_000), 1)
    return now,nxt


def mock_lateral_inputs(base, future_speed=100., velocity=10.):
    base._compute_initial_velocity_and_lateral_state = Mock(return_value=(velocity, np.zeros(3)))
    base._compute_reference_velocity_and_curvature_profile = Mock(return_value=(future_speed,np.zeros(10)))
    base._lateral_lqr_controller = Mock(return_value=.123)
    base._longitudinal_lqr_controller = Mock(side_effect=AssertionError('Longitudinal solver forbidden in coast'))
    base._stopping_controller = Mock(side_effect=AssertionError('Stopping controller forbidden in coast'))
    base.track_trajectory = Mock(side_effect=AssertionError('Base full tracking forbidden in coast'))


@pytest.mark.parametrize('future_speed', [0., 100.])
def test_coast_bypasses_both_longitudinal_solvers_despite_future_speed(future_speed):
    base=tracker();mock_lateral_inputs(base,future_speed)
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(2_100_000);ego=state(now.time_point.time_us)
    command=adapter.track_trajectory(now,nxt,ego,object())
    assert command.rear_axle_acceleration_2d.x == 0.
    assert command.tire_steering_rate == .123
    for method in [base.track_trajectory,base._longitudinal_lqr_controller,base._stopping_controller]:method.assert_not_called()
    np.testing.assert_array_equal(base._lateral_lqr_controller.call_args[0][1],np.full(10,10.))
    assert ego.dynamic_car_state.rear_axle_acceleration_2d.x == -1.  # No hidden state reset.
    assert adapter.last_command_evidence['longitudinal_tracking_disabled']


def test_coast_disables_stopping_feedback_at_low_speed_too():
    base=tracker();mock_lateral_inputs(base,future_speed=0.,velocity=0.)
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(2_200_000)
    assert adapter.track_trajectory(now,nxt,state(now.time_point.time_us,velocity=0.),object()).rear_axle_acceleration_2d.x == 0.
    base._stopping_controller.assert_not_called()


@pytest.mark.parametrize('elapsed,expected', [(2_099_999,False),(2_100_000,True),(3_899_999,True),(3_900_000,False)])
def test_half_open_mode_is_current_time_not_preview_time(elapsed,expected):
    adapter=ZeroLongitudinalRequestTrackerV1(tracker(),ANCHOR,INTERVALS)
    assert adapter.coast_active(ANCHOR+elapsed) is expected


@pytest.mark.parametrize('elapsed', [2_000_000,3_900_000])
def test_before_and_after_coast_delegate_exact_official_packet(elapsed):
    base=tracker()
    sentinel=DynamicCarState.build_from_rear_axle(1.,StateVector2D(10.,0.),StateVector2D(-1.,0.),tire_steering_rate=.3)
    base.track_trajectory=Mock(return_value=sentinel)
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(elapsed)
    assert adapter.track_trajectory(now,nxt,state(now.time_point.time_us),object()) is sentinel
    base.track_trajectory.assert_called_once()
    assert adapter.last_command_evidence['mode']=='OFFICIAL_TRACKING'


def test_stock_two_stage_still_passes_zero_packet_and_advances_via_model_interface():
    base=tracker();mock_lateral_inputs(base)
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(2_100_000);ego=state(now.time_point.time_us)
    # A test double, not the official vehicle model. It returns a distinct
    # future state solely to verify the official controller's call plumbing.
    next_state=state(nxt.time_point.time_us,x=1.)
    model=Mock();model.propagate_state.return_value=next_state
    controller=TwoStageController(scenario=None,tracker=adapter,motion_model=model)
    controller.update_state(now,nxt,ego,object())
    model.propagate_state.assert_called_once()
    args=model.propagate_state.call_args.kwargs
    assert args['state'] is ego
    assert args['ideal_dynamic_state'].rear_axle_acceleration_2d.x==0.
    assert args['sampling_time'].time_us==100_000
    assert controller.get_state() is next_state


def test_real_route_helper_smoke_uses_synthetic_poses_and_no_vehicle_propagation():
    base=tracker()
    base._longitudinal_lqr_controller=Mock(side_effect=AssertionError('Must remain disabled'))
    base._stopping_controller=Mock(side_effect=AssertionError('Must remain disabled'))
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(2_100_000);ego=state(now.time_point.time_us)
    trajectory=InterpolatedTrajectory([state(now.time_point.time_us+i*100_000,x=float(i)) for i in range(80)])
    command=adapter.track_trajectory(now,nxt,ego,trajectory)
    assert command.rear_axle_acceleration_2d.x==0.
    assert command.tire_steering_rate==pytest.approx(0.,abs=1e-12)
    base._longitudinal_lqr_controller.assert_not_called()


@pytest.mark.parametrize('intervals', [[],[(-1,2)],[(2,2)],[(2,1)],[(0,10),(9,20)],[(0.,10)]])
def test_invalid_schedule_fails_closed(intervals):
    with pytest.raises(ValueError):ZeroLongitudinalRequestTrackerV1(tracker(),ANCHOR,intervals)


def test_clock_mismatch_or_pre_anchor_time_is_rejected():
    adapter=ZeroLongitudinalRequestTrackerV1(tracker(),ANCHOR,INTERVALS)
    now,nxt=iterations(2_100_000)
    with pytest.raises(ValueError):adapter.track_trajectory(now,nxt,state(now.time_point.time_us+1),object())
    with pytest.raises(ValueError):adapter.coast_active(ANCHOR-1)


@pytest.mark.parametrize('elapsed', [2_050_000,3_850_000])
def test_control_step_cannot_hide_a_partial_coast_transition(elapsed):
    adapter=ZeroLongitudinalRequestTrackerV1(tracker(),ANCHOR,INTERVALS)
    now,nxt=iterations(elapsed)
    with pytest.raises(ValueError,match='straddles'):
        adapter.track_trajectory(now,nxt,state(now.time_point.time_us),object())
    assert adapter.last_command_evidence is None


def test_invalid_route_or_nonfinite_packet_cannot_leave_stale_success_evidence():
    base=tracker();mock_lateral_inputs(base)
    adapter=ZeroLongitudinalRequestTrackerV1(base,ANCHOR,INTERVALS)
    now,nxt=iterations(2_100_000);ego=state(now.time_point.time_us)
    adapter.track_trajectory(now,nxt,ego,object())
    base._compute_reference_velocity_and_curvature_profile.return_value=(100.,np.zeros(9))
    with pytest.raises(ValueError,match='shape'):
        adapter.track_trajectory(now,nxt,ego,object())
    assert adapter.last_command_evidence is None
    base._compute_reference_velocity_and_curvature_profile.return_value=(100.,np.zeros(10))
    base._lateral_lqr_controller.return_value=float('nan')
    with pytest.raises(ValueError,match='Nonfinite steering'):
        adapter.track_trajectory(now,nxt,ego,object())
    assert adapter.last_command_evidence is None
