from __future__ import annotations

import math

from tools.stage6k_evaluate_realized_dose_curve import descriptive_zero_rows, gate_decisions, json_safe


def test_zero_dose_is_descriptive_only() -> None:
    rows = descriptive_zero_rows()
    assert len(rows) == 7
    assert all(row["analysis_role"] == "descriptive_origin_no_inference" for row in rows)
    assert all(row["mean_delta_A_minus_B"] == 0.0 for row in rows)


def test_gate_requires_both_positive_cluster_bounds() -> None:
    rows = []
    for label, dose in [("dose25", 0.25), ("dose50", 0.5), ("dose75", 0.75), ("dose100", 1.0)]:
        for metric, low in [("delta_mean_speed", 0.1), ("delta_rms_accel", 0.01)]:
            rows.append({"dose_label": label, "nominal_dose": dose, "scope": "overall", "metric": metric, "mean_delta_A_minus_B": low + 0.2, "cluster_bootstrap_one_sided95_low": low, "cluster_bootstrap_ci95_low": low - 0.01, "cluster_bootstrap_ci95_high": low + 0.4})
    rows[3]["cluster_bootstrap_one_sided95_low"] = -0.01
    decisions = gate_decisions(rows)
    assert decisions[0]["kinematic_gate_passed"] is True
    assert decisions[1]["kinematic_gate_passed"] is False


def test_json_safe_converts_undefined_descriptive_statistic_to_null() -> None:
    assert json_safe({"rho": math.nan, "count": 4}) == {"rho": None, "count": 4}
