from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchKinematicBicycleConfig:
    max_steering_angle: float = 1.0471975512
    accel_time_constant: float = 0.2
    steering_angle_time_constant: float = 0.05


class BatchKinematicBicycleModel:
    def __init__(self, config: BatchKinematicBicycleConfig | None = None, **kwargs) -> None:
        self.config = config or BatchKinematicBicycleConfig(**kwargs)
