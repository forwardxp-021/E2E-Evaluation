from __future__ import annotations

from typing import Any, Optional

from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import PDMGenerator
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import PDMComfortConfig
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleConfig
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTrackerConfig
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import PDMEmergencyBrake, PDMEmergencyBrakeConfig


class AbstractPDMClosedPlanner:
    def __init__(
        self,
        *args: Any,
        scorer: Optional[dict[str, Any]] = None,
        comfort: Optional[dict[str, Any]] = None,
        tracker: Optional[dict[str, Any]] = None,
        motion_model: Optional[dict[str, Any]] = None,
        emergency_brake: Optional[dict[str, Any]] = None,
        generator: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.args = args
        self.kwargs = kwargs
        self.comfort_config = PDMComfortConfig(**(comfort or {}))
        scorer_kwargs = dict(scorer or {})
        self.scorer = PDMScorer(**scorer_kwargs, comfort_config=self.comfort_config)
        self.tracker_config = BatchLQRTrackerConfig(**(tracker or {}))
        self.motion_model_config = BatchKinematicBicycleConfig(**(motion_model or {}))
        self.simulator = PDMSimulator(tracker_config=self.tracker_config, motion_model_config=self.motion_model_config)
        self.emergency_brake_config = PDMEmergencyBrakeConfig(**(emergency_brake or {}))
        self.emergency_brake = PDMEmergencyBrake(self.emergency_brake_config)
        self.generator = PDMGenerator(**(generator or {}))


class PDMClosedPlanner(AbstractPDMClosedPlanner):
    pass
