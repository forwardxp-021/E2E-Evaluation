from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.r1_b2_8_r1_repair_and_repreflight import validate_realized_trace_rows


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def _rows() -> list[dict]:
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
        for index in range(80)
    ]


def test_realized_trace_accepts_exactly_80_timestamp_preserving_rows():
    states = validate_realized_trace_rows(_rows())
    assert len(states) == 80
    assert states[0]["time_us"] == 1_000_000
    assert states[-1]["time_us"] == 8_900_000


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "79", "81", "time", "planned"))
def test_realized_trace_fail_closed_cases(mutation: str):
    rows = _rows()
    if mutation == "duplicate":
        rows = rows[:20] + [rows[19]] + rows[20:]
    elif mutation == "missing":
        rows = rows[:10] + rows[11:]
    elif mutation == "79":
        rows = rows[:79]
    elif mutation == "81":
        rows = rows + [rows[-1]]
    elif mutation == "time":
        rows[20] = {**rows[20], "current_ego": {**rows[20]["current_ego"], "time_us": rows[19]["current_ego"]["time_us"]}}
    else:
        rows[0] = {**rows[0], "primary_measurement_source": "PLANNED"}
    with pytest.raises(ValueError):
        validate_realized_trace_rows(rows)


def test_48_run_bindings_are_exact_schedule_copies():
    schedule = json.loads((R1 / "r1_official_compliant_technical_smoke_schedule_v2.0.json").read_text())
    bindings = json.loads((R1 / "r1_b2_8_r1_execution_bindings_manifest_v1.0.json").read_text())
    fields = ("run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "run_order")
    observed = [{key: row[key] for key in fields} for row in bindings["frozen_run_bindings"]]
    expected = [{key: row[key] for key in fields} for row in schedule["runs"]]
    assert observed == expected


def test_all_hydra_compositions_and_zero_run_preflight_pass():
    hydra = json.loads((R1 / "r1_b2_8_r1_hydra_binding_audit_v1.0.json").read_text())
    preflight = json.loads((R1 / "r1_b2_8_r1_zero_run_complete_path_preflight_v1.0.json").read_text())
    assert hydra["status"] == "48_OF_48_HYDRA_FROZEN_RUN_BINDING_PASS"
    assert len(hydra["rows"]) == 48
    assert preflight["PRE_RUN_INTEGRITY"] == "PASS_COMPLETE_EXECUTION_PATH_ZERO_RUN"
    assert preflight["actual_official_runs"] == 0
    assert preflight["consumed_budget"] == 0
