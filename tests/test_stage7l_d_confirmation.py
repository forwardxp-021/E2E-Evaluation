import ast
import json
from copy import deepcopy
from pathlib import Path

from tools.stage7l_evaluate_confirmation_gates import (
    build_completeness,
    mechanism_summary,
    nuisance_summary,
    safety_summary,
)
from tools.stage7l_run_confirmation import (
    DOSES,
    TRANSITION_LENGTH_M,
    build_runtime_manifest_adapter,
    initial_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def frozen_protocol_for_test() -> dict:
    protocol = json.loads((ROOT / "configs/stage7l_c_prospective_confirmation_protocol_v1.json").read_text())
    protocol = deepcopy(protocol)
    protocol["semantic_uncertainty_reporting"]["replicates"] = 40
    return protocol


def roster() -> list[dict]:
    return [
        {
            "collection_order": str(index + 1),
            "scenario_token": f"token-{index:03d}",
            "log_name": "shared-log" if index < 2 else f"log-{index:03d}",
            "direction": "left" if index < 15 else "right",
        }
        for index in range(80)
    ]


def summary(successful_scenarios: int = 80) -> list[dict]:
    rows = []
    for item in roster():
        is_success = int(item["collection_order"]) <= successful_scenarios
        for dose in DOSES:
            rows.append(
                {
                    "cell_id": f"{item['scenario_token']}-{dose}",
                    "scenario_token": item["scenario_token"],
                    "dose": dose,
                    "official_run_status": "SUCCEEDED" if is_success else "PROCESS_FAILED",
                    "trajectory_available": str(is_success),
                    "failure_category": "" if is_success else "INFRASTRUCTURE_RUNTIME",
                }
            )
    return rows


def metrics() -> list[dict]:
    rows = []
    for item in roster():
        for dose_index, dose in enumerate(DOSES):
            rows.append(
                {
                    "cell_id": f"{item['scenario_token']}-{dose}",
                    "scenario_token": item["scenario_token"],
                    "dose": dose,
                    "valid": "True",
                    "lane_change_completion": "True",
                    "offroad": "False",
                    "responsible_collision": "False",
                    "route_failure": "False",
                    "lane_change_duration_s": str(5.0 - 0.1 * dose_index),
                    "rms_lateral_accel_mps2": str(0.2 + 0.02 * dose_index),
                    "peak_yaw_rate_radps": str(0.05 + 0.01 * dose_index),
                    "peak_lateral_accel_mps2": str(0.3 + 0.03 * dose_index),
                    "rms_yaw_rate_radps": str(0.02 + 0.005 * dose_index),
                    "rms_lateral_jerk_mps3": str(0.1 + 0.01 * dose_index),
                    "target_center_settling_time_s": str(6.0 - 0.1 * dose_index),
                    "final_target_lane_center_offset_m": "0.1",
                    "mean_speed_mps": str(5.0 + 0.001 * dose_index),
                    "rms_longitudinal_accel_mps2": str(0.4 + 0.002 * dose_index),
                    "rms_longitudinal_jerk_mps3": str(0.8 + 0.003 * dose_index),
                    "route_progress_m": str(75.0 + 0.01 * dose_index),
                }
            )
    return rows


def pairs() -> list[dict]:
    result = []
    for item in roster():
        for dose in DOSES[1:]:
            dose_number = int(dose.removeprefix("dose"))
            for metric, delta in {
                "lane_change_duration_s": -dose_number / 100.0,
                "rms_lateral_accel_mps2": dose_number / 1000.0,
                "peak_yaw_rate_radps": dose_number / 10000.0,
                "peak_lateral_accel_mps2": dose_number / 500.0,
                "rms_yaw_rate_radps": dose_number / 20000.0,
                "rms_lateral_jerk_mps3": dose_number / 5000.0,
                "target_center_settling_time_s": -dose_number / 100.0,
                "final_target_lane_center_offset_m": 0.0,
                "mean_speed_mps": 0.004,
                "rms_longitudinal_accel_mps2": 0.008,
                "rms_longitudinal_jerk_mps3": 0.012,
                "route_progress_m": 0.04,
            }.items():
                expected = "negative" if metric == "lane_change_duration_s" else (
                    "positive" if metric in {"rms_lateral_accel_mps2", "peak_yaw_rate_radps"} else "descriptive"
                )
                result.append(
                    {
                        "scenario_token": item["scenario_token"],
                        "log_name": item["log_name"],
                        "direction": item["direction"],
                        "dose": dose,
                        "metric": metric,
                        "paired_delta": delta,
                        "directionally_consistent": expected == "negative" and delta < 0 or expected == "positive" and delta > 0,
                    }
                )
    return result


def execution_contract() -> dict:
    return {
        "safety_aggregation": {
            "denominator": "all_80_frozen_scenarios",
            "level": "scenario_level_conservative_across_all_five_frozen_doses",
        }
    }


def test_exact_400_cell_plan_is_roster_order_then_dose_order() -> None:
    frozen = []
    for item in roster():
        frozen.append({**item})
    plan = initial_plan(frozen)
    assert len(plan) == 400
    assert len({row["cell_id"] for row in plan}) == 400
    assert [row["dose"] for row in plan[:5]] == list(DOSES)
    assert [float(row["transition_length_m"]) for row in plan[:5]] == [TRANSITION_LENGTH_M[dose] for dose in DOSES]
    assert all(row["planned"] is True for row in plan)


def test_execution_gate_is_scenario_complete_all_five_and_threshold_76() -> None:
    rows, result = build_completeness(roster(), summary(76))
    assert len(rows) == 80
    assert result["N_complete_all_five_doses"] == 76
    assert result["successful_official_rollout_cells"] == 380
    assert result["execution_gate_pass"] is True
    _, below = build_completeness(roster(), summary(75))
    assert below["execution_gate_pass"] is False


def test_frozen_mechanism_and_nuisance_gates_pass_known_direction() -> None:
    protocol = frozen_protocol_for_test()
    mechanism = mechanism_summary(pairs(), protocol)
    nuisance = nuisance_summary(pairs(), protocol)
    assert mechanism["mechanism_gate_pass"] is True
    assert mechanism["primary"]["lane_change_duration_s"]["directional_consistency"] == 1.0
    assert mechanism["primary"]["rms_lateral_accel_mps2"]["directional_consistency"] == 1.0
    assert mechanism["primary"]["peak_yaw_rate_radps"]["directional_consistency"] == 1.0
    assert nuisance["nuisance_gate_pass"] is True


def test_safety_uses_all_80_scenario_level_conservative_aggregation() -> None:
    protocol = frozen_protocol_for_test()
    metric_rows = metrics()
    # One dose is enough to make a scenario an adverse scenario-level outcome.
    metric_rows[2]["offroad"] = "True"
    metric_rows[8]["responsible_collision"] = "True"
    safety = safety_summary(roster(), summary(), metric_rows, protocol, execution_contract())
    assert safety["denominator"] == 80
    assert safety["counts"]["offroad"] == 1
    assert safety["counts"]["responsible_collision"] == 1
    assert safety["rates"]["offroad_rate"] == 1 / 80
    assert safety["safety_gate_pass"] is True


def test_no_representation_runtime_imports() -> None:
    for name in (
        "tools/stage7l_run_confirmation.py",
        "tools/stage7l_extract_confirmation_metrics.py",
        "tools/stage7l_evaluate_confirmation_gates.py",
    ):
        source = (ROOT / name).read_text()
        tree = ast.parse(source)
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
        }
        assert "torch" not in imported
        assert "tensorflow" not in imported


def test_runtime_manifest_adapter_only_repairs_frozen_planner_interface(tmp_path: Path) -> None:
    source_full = json.loads(
        (ROOT / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_maneuver_manifest.json").read_text()
    )
    source = {**source_full, "maneuvers": source_full["maneuvers"][:1]}
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "runtime.json"
    source_path.write_text(json.dumps(source))
    audit = build_runtime_manifest_adapter(source_path, output_path, frozen_protocol_for_test())
    runtime = json.loads(output_path.read_text())
    original = source["maneuvers"][0]
    adapted = runtime["maneuvers"][0]
    assert audit["status"] == "PASS_CODE_NON_EXECUTABILITY_INTERFACE_REPAIRED"
    assert audit["protocol_changed"] is False
    assert audit["roster_changed"] is False
    assert audit["treatment_changed"] is False
    assert adapted["scenario_token"] == original["scenario_token"]
    assert adapted["source_reference_xy"] == original["source_reference_xy"]
    assert adapted["target_reference_xy"] == original["target_reference_xy"]
    assert adapted["horizon_s"] == 15.0
    assert adapted["planner_profile_ids"] == list(DOSES)
    assert adapted["background_mode"] == "closed_loop_nonreactive_agents"
