import json
from pathlib import Path

import pytest

from tools.r1_b2_8_r3_1_official_safety_adapter import MetricCanonicalizationError, _require_payload
from tools.r1_b2_8_r3_1_post_run_evaluator_dispatcher import _require_pair_binding
from tools.r1_closed_loop_benchmark_v2_1 import exact_realized_window_v1_1


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def _trace(count: int) -> list[dict]:
    return [
        {
            "primary_measurement_source": "REALIZED_CURRENT_EGO",
            "iteration_index": index,
            "current_ego": {
                "time_us": 1_000_000 + index * 100_000,
                "rear_axle": {"x": float(index), "y": 0.0, "heading": 0.0},
                "speed_mps": 5.0,
            },
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("count", (80, 81, 100))
def test_primary_trace_uses_exact_first_80_rows_when_raw_trace_is_longer(count: int) -> None:
    states = exact_realized_window_v1_1(_trace(count))
    assert len(states) == 80
    assert states[0]["time_us"] == 1_000_000
    assert states[-1]["time_us"] == 8_900_000


def test_primary_trace_rejects_missing_duplicate_and_nonmonotonic_primary_rows() -> None:
    missing = _trace(80)
    missing[7]["iteration_index"] = 8
    duplicate = _trace(80)
    duplicate[8]["iteration_index"] = 7
    nonmonotonic = _trace(80)
    nonmonotonic[20]["current_ego"]["time_us"] = nonmonotonic[19]["current_ego"]["time_us"]
    for invalid in (missing, duplicate, nonmonotonic):
        with pytest.raises(ValueError):
            exact_realized_window_v1_1(invalid)


def test_safety_adapter_reuses_exact_historical_payload_and_fails_closed() -> None:
    assert _require_payload(
        {
            "collision": {"number_of_all_at_fault_collisions_stat_value": 0},
            "drivable_area": {"drivable_area_compliance_stat_value": True},
        }
    ) == (0, True)
    with pytest.raises(MetricCanonicalizationError):
        _require_payload({"collision": {}, "drivable_area": {}})
    with pytest.raises(MetricCanonicalizationError):
        _require_payload(
            {
                "collision": {"number_of_all_at_fault_collisions_stat_value": 0},
                "drivable_area": {"drivable_area_compliance_stat_value": 0},
            }
        )


def test_dispatcher_requires_hlc_native_references_and_pretreatment_ledger() -> None:
    binding = {
        "family": "R-HLC", "baseline_context": {}, "treatment_context": {},
        "pretreatment_clearance": {"pretreatment_only": True},
    }
    with pytest.raises(ValueError, match="FROZEN_PAIR_BINDING_MISSING"):
        _require_pair_binding(binding)
    binding.update(
        {
            "source_reference_xy": [[0, 0], [1, 0]], "target_reference_xy": [[0, 1], [1, 1]],
            "native_route_reference_xy": [[0, 0], [1, 0]], "native_route_reference_source": "official_map",
        }
    )
    _require_pair_binding(binding)
    binding["pretreatment_clearance"] = {"pretreatment_only": False}
    with pytest.raises(ValueError, match="POSTHOC_CLEARANCE_RECALCULATION_FORBIDDEN"):
        _require_pair_binding(binding)


def test_final_manifest_is_ready_and_zero_run() -> None:
    manifest = json.loads((R1 / "r1_b2_8_r3_1_final_execution_binding_manifest_v1.0.json").read_text())
    regression = json.loads((R1 / "r1_b2_8_r3_1_final_zero_run_regression_v1.0.json").read_text())
    assert manifest["status"] == "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION"
    assert manifest["authorization"] == {
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "RBR_A_B_C_AUTHORIZED": False,
    }
    assert regression["counts"] == {"exact_resolution": 48, "full_hydra": 48, "simulation_runner_construction": 48}
    assert regression["simulation_started"] is False
    assert regression["official_runs"] == 0 and regression["consumed_budget"] == 0
