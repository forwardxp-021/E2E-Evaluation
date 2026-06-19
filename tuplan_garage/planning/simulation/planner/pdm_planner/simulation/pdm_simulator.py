from __future__ import annotations

from dataclasses import dataclass

from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleConfig, BatchKinematicBicycleModel
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker, BatchLQRTrackerConfig


@dataclass(frozen=True)
class PDMSimulatorConfig:
    tracker: BatchLQRTrackerConfig = BatchLQRTrackerConfig()
    motion_model: BatchKinematicBicycleConfig = BatchKinematicBicycleConfig()


class PDMSimulator:
    def __init__(self, tracker_config: BatchLQRTrackerConfig | None = None, motion_model_config: BatchKinematicBicycleConfig | None = None, simulator_config: PDMSimulatorConfig | None = None) -> None:
        config = simulator_config or PDMSimulatorConfig(
            tracker=tracker_config or BatchLQRTrackerConfig(),
            motion_model=motion_model_config or BatchKinematicBicycleConfig(),
        )
        self.config = config
        self.motion_model = BatchKinematicBicycleModel(config.motion_model)
        self.tracker = BatchLQRTracker(config.tracker)
