from __future__ import annotations

from typing import Any, Dict, Optional

from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import PDMComfortConfig, ego_is_comfortable

PROGRESS = "progress"
TTC = "ttc"
COMFORTABLE = "comfortable"
WEIGHTED_METRICS_WEIGHTS: Dict[str, float] = {PROGRESS: 5.0, TTC: 5.0, COMFORTABLE: 2.0}


class PDMScorer:
    def __init__(self, progress_weight: float = 5.0, ttc_weight: float = 5.0, comfortable_weight: float = 2.0, comfort_config: Optional[PDMComfortConfig] = None) -> None:
        self.weighted_metrics_weights = dict(WEIGHTED_METRICS_WEIGHTS)
        self.weighted_metrics_weights[PROGRESS] = float(progress_weight)
        self.weighted_metrics_weights[TTC] = float(ttc_weight)
        self.weighted_metrics_weights[COMFORTABLE] = float(comfortable_weight)
        self.comfort_config = comfort_config or PDMComfortConfig()

    def score_weighted_metrics(self, metrics: Dict[str, float]) -> float:
        weight_sum = sum(self.weighted_metrics_weights.values())
        if weight_sum == 0:
            return 0.0
        return sum(float(metrics.get(k, 0.0)) * w for k, w in self.weighted_metrics_weights.items()) / weight_sum

    def ego_is_comfortable(self, ego_state_history: Any) -> bool:
        return ego_is_comfortable(ego_state_history, self.comfort_config)
