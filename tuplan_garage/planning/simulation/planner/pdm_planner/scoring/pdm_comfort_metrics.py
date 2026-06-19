from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class PDMComfortConfig:
    max_abs_mag_jerk: float = 8.37
    max_abs_lat_accel: float = 4.89
    max_lon_accel: float = 2.40
    min_lon_accel: float = -4.05
    max_abs_yaw_accel: float = 1.93
    max_abs_lon_jerk: float = 4.13
    max_abs_yaw_rate: float = 0.95


def _get_series(history: Any, name: str) -> Iterable[float]:
    if isinstance(history, dict):
        return history.get(name, [])
    return getattr(history, name, [])


def _max_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return max(vals) if vals else 0.0


def ego_is_comfortable(ego_state_history: Any, comfort_config: Optional[PDMComfortConfig] = None) -> bool:
    config = comfort_config or PDMComfortConfig()
    checks = [
        _max_abs(_get_series(ego_state_history, "mag_jerk")) <= config.max_abs_mag_jerk,
        _max_abs(_get_series(ego_state_history, "lat_accel")) <= config.max_abs_lat_accel,
        max([float(v) for v in _get_series(ego_state_history, "lon_accel")] or [0.0]) <= config.max_lon_accel,
        min([float(v) for v in _get_series(ego_state_history, "lon_accel")] or [0.0]) >= config.min_lon_accel,
        _max_abs(_get_series(ego_state_history, "yaw_accel")) <= config.max_abs_yaw_accel,
        _max_abs(_get_series(ego_state_history, "lon_jerk")) <= config.max_abs_lon_jerk,
        _max_abs(_get_series(ego_state_history, "yaw_rate")) <= config.max_abs_yaw_rate,
    ]
    return all(checks)
