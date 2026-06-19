from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PDMEmergencyBrakeConfig:
    time_to_infraction_threshold: float = 2.0
    max_ego_speed: float = 5.0
    max_long_accel: float = 2.40
    min_long_accel: float = -4.05
    infraction: str = "collision"


class PDMEmergencyBrake:
    def __init__(self, config: PDMEmergencyBrakeConfig | None = None, **kwargs) -> None:
        self.config = config or PDMEmergencyBrakeConfig(**kwargs)
