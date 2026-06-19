from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BatchLQRTrackerConfig:
    q_longitudinal: List[float] = field(default_factory=lambda: [10.0])
    r_longitudinal: List[float] = field(default_factory=lambda: [1.0])
    q_lateral: List[float] = field(default_factory=lambda: [1.0, 10.0, 0.0])
    r_lateral: List[float] = field(default_factory=lambda: [1.0])
    tracking_horizon: int = 10
    jerk_penalty: float = 1.0e-4
    curvature_rate_penalty: float = 1.0e-2
    stopping_proportional_gain: float = 0.5
    stopping_velocity: float = 0.2


class BatchLQRTracker:
    def __init__(self, config: BatchLQRTrackerConfig | None = None, **kwargs) -> None:
        self.config = config or BatchLQRTrackerConfig(**kwargs)
