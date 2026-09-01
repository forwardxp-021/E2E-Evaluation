from __future__ import annotations

import inspect
import json
from pathlib import Path

from tools import r1_b2_9_a_native_reference_coverage_forensic as forensic


ROOT = Path(__file__).resolve().parents[1]
ITERATION = ROOT / "docs/stageR/r1/r1_b2_9_a_iteration_0_33_native_coverage_audit_v1.json"
ALL12 = ROOT / "docs/stageR/r1/r1_b2_9_a_all12_hlc_nominal_replan_coverage_audit_v1.json"


def test_iteration_33_failure_reconstruction_is_exact() -> None:
    payload = json.loads(ITERATION.read_text())
    assert payload["first_invalid_iteration"] == 33
    assert payload["exact_first_raised_reference"] == "source_reference_xy"
    assert payload["simultaneously_invalid_references_at_iteration_33"] == ["source", "target"]
    assert payload["iteration_32_last_valid_row"]["invalid_references"] == []
    assert len(payload["rows"]) == 34


def test_zero_weight_source_and_active_target_are_both_recorded() -> None:
    window = json.loads(ITERATION.read_text())["baseline_iteration_33_output_window"]
    assert window["source_weight_min"] == window["source_weight_max"] == 0.0
    assert window["target_weight_min"] == window["target_weight_max"] == 1.0
    assert len(window["classification"]) == 2


def test_all12_initial_pass_but_nominal_replans_exhaust_before_80() -> None:
    payload = json.loads(ALL12.read_text())
    assert payload["identity_count"] == 12
    assert payload["all_initial_one_shot_pass"] is True
    assert payload["all_predicted_to_exhaust_before_iteration_80"] is True
    for entry in payload["entries"]:
        for arm in ("baseline", "treatment"):
            assert len(entry[arm]["rolling_call_envelope_iteration_0_79"]) == 80
            assert 0 <= entry[arm]["predicted_first_coverage_exhaustion_iteration"] < 80


def test_forensic_tool_contains_no_simulation_execution_entrypoint() -> None:
    source = inspect.getsource(forensic)
    assert "runner" + ".run(" not in source
    assert "simulation" + ".step(" not in source
    assert "run_simulation.py" not in source
