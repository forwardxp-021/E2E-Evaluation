import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import PDMComfortConfig, ego_is_comfortable
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import COMFORTABLE, PROGRESS, TTC, PDMScorer
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleConfig
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTrackerConfig
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def test_default_pdm_closed_constructor_behavior_unchanged():
    planner = PDMClosedPlanner()
    assert planner.scorer.weighted_metrics_weights[PROGRESS] == 5.0
    assert planner.scorer.weighted_metrics_weights[TTC] == 5.0
    assert planner.scorer.weighted_metrics_weights[COMFORTABLE] == 2.0
    assert planner.generator.leading_agent_update_rate == 2


def test_new_scorer_weights_are_used_when_provided():
    scorer = PDMScorer(progress_weight=3.0, ttc_weight=8.0, comfortable_weight=4.0)
    assert scorer.weighted_metrics_weights == {PROGRESS: 3.0, TTC: 8.0, COMFORTABLE: 4.0}


def test_new_comfort_thresholds_are_used_when_provided():
    history = {"lat_accel": [4.0]}
    assert ego_is_comfortable(history) is True
    assert ego_is_comfortable(history, PDMComfortConfig(max_abs_lat_accel=3.5)) is False


def test_tracker_and_motion_model_configs_passed_through_simulator():
    tracker = BatchLQRTrackerConfig(q_lateral=[1.0, 8.0, 1.0], r_lateral=[2.0], curvature_rate_penalty=0.03)
    motion = BatchKinematicBicycleConfig(steering_angle_time_constant=0.08)
    sim = PDMSimulator(tracker_config=tracker, motion_model_config=motion)
    assert sim.tracker.config.q_lateral == [1.0, 8.0, 1.0]
    assert sim.tracker.config.curvature_rate_penalty == 0.03
    assert sim.motion_model.config.steering_angle_time_constant == 0.08


def test_closed_planner_nested_configs_wire_components():
    planner = PDMClosedPlanner(
        scorer={"progress_weight": 8.0},
        comfort={"max_abs_lat_accel": 5.5},
        tracker={"r_lateral": [0.5]},
        motion_model={"steering_angle_time_constant": 0.03},
        emergency_brake={"infraction": "collision"},
        generator={"leading_agent_update_rate": 3},
    )
    assert planner.scorer.weighted_metrics_weights[PROGRESS] == 8.0
    assert planner.comfort_config.max_abs_lat_accel == 5.5
    assert planner.simulator.tracker.config.r_lateral == [0.5]
    assert planner.simulator.motion_model.config.steering_angle_time_constant == 0.03
    assert planner.emergency_brake.config.infraction == "collision"
    assert planner.generator.leading_agent_update_rate == 3
