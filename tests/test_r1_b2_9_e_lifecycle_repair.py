import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import r1_b2_9_e_execute_frozen_48run_smoke as executor
from tools.r1_b2_9_e_official_run_lifecycle import run_one_with_full_nuplan_lifecycle


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def _load(name: str) -> dict:
    return json.loads((R1 / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_executor_uses_shared_helper_and_has_no_direct_runner_run() -> None:
    source = inspect.getsource(executor)
    tree = ast.parse(source)
    assert "run_one_with_full_nuplan_lifecycle" in source
    assert "runners[0].run" not in source
    direct_run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert direct_run_calls == []


def test_shared_helper_calls_nuplan_run_runners_and_fails_if_parquet_missing(tmp_path, monkeypatch) -> None:
    import nuplan.planning.script.utils as utils

    called = []

    def fake_run_runners(runners, common_builder, profiler_name, cfg):
        called.append((runners, common_builder, profiler_name, cfg))

    monkeypatch.setattr(utils, "run_runners", fake_run_runners)
    cfg = SimpleNamespace(runner_report_file="runner_report.parquet")
    with pytest.raises(RuntimeError, match="no_ego_at_fault_collisions.parquet:0"):
        run_one_with_full_nuplan_lifecycle(
            runners=[object()],
            common_builder=object(),
            profiler_name="test",
            cfg=cfg,
            run_output_root=tmp_path,
        )
    assert len(called) == 1


def test_pair_evaluator_occurs_only_after_full_lifecycle_construction_call() -> None:
    source = inspect.getsource(executor.run_official_package)
    assert source.index("_construct_and_optionally_execute(") < source.index("evaluate_frozen_pair(")
    construction_source = inspect.getsource(executor._construct_and_optionally_execute)
    assert construction_source.index("run_one_with_full_nuplan_lifecycle(") < construction_source.index("return audit")


def test_roster_unchanged_and_schedule_pair_semantics_are_exact() -> None:
    roster = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
    schedule = R1 / "r1_official_compliant_technical_smoke_schedule_v3.1.json"
    pairs = R1 / "r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1.json"
    schedule_audit = _load("r1_b2_9_e_schedule_v3_0_to_v3_1_parity_audit_v1.json")
    pair_audit = _load("r1_b2_9_e_pair_binding_v2_0_to_v2_1_parity_audit_v1.json")
    assert _sha(roster) == "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"
    assert _sha(schedule) == "99f44095c27319b746921376d2549a00186303298b5266ff45dd008a98c08455"
    assert _sha(pairs) == "a606a87b01cd1fdd340070fca7e77170b6e0782aafa1e7c19ab6c91228cc9fa6"
    assert schedule_audit["status"] == "48_OF_48_EXACT_SCIENTIFIC_SEMANTIC_PARITY_RUN_ID_NAMESPACE_ONLY"
    assert len(schedule_audit["rows"]) == 48
    assert all(row["scientific_semantics_exact"] for row in schedule_audit["rows"])
    assert pair_audit["status"] == "24_OF_24_EXACT_SCIENTIFIC_SEMANTIC_PARITY"
    assert len(pair_audit["pairs"]) == 24
    assert all(row["scientific_semantics_exact"] for row in pair_audit["pairs"])


def test_exact_lifecycle_canary_and_zero_run_closures() -> None:
    canary = _load("r1_b2_9_e_exact_lifecycle_canary_run_ledger_v1.0.json")
    zero = _load("r1_b2_9_e_zero_run_final_construction_audit_v1.0.json")
    assert canary["status"] == "4_OF_4_EXACT_LIFECYCLE_CANARY_PASS"
    assert canary["counts"] == {
        "runs": 4,
        "HLC_technical_complete": 2,
        "TSB_technical_complete": 2,
        "exact_80_traces": 4,
        "metric_lifecycle_complete": 4,
        "safety_adapter_complete": 4,
        "dispatcher_complete": 2,
    }
    assert canary["actual_simulation_reruns"] == 0
    assert all(row["run_runners_called"] is True for row in canary["runs"])
    assert all(row["lifecycle"]["runner_report_available"] is True for row in canary["runs"])
    assert all(row["lifecycle"]["temporary_metric_is_only_final_output"] is False for row in canary["runs"])
    assert zero["status"] == "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS"
    assert zero["counts"] == {
        "exact_resolutions": 48,
        "planner_v3_1_bindings": 48,
        "Primary80_controller_bindings": 48,
        "runner_constructions": 48,
        "pair_binding_lookups": 48,
    }
    assert zero["runner_run_calls"] == 0
    assert zero["run_runners_calls"] == 0
    assert zero["simulation_started"] is False
    assert "/var/folders/" not in json.dumps(zero)


def test_final_manifest_closes_callback_dependencies_and_preserves_old_attempt() -> None:
    manifest = _load("r1_b2_9_e_final_execution_binding_manifest_v2.1.json")
    assert manifest["status"] == "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_REAUTHORIZATION"
    assert manifest["callback_transitive_sha_closure"] == "PASS"
    assert manifest["complete_transitive_sha_closure"] == "PASS"
    assert manifest["predecessor"]["old_attempts_consumed"] == 2
    assert manifest["predecessor"]["old_outputs_reused"] is False
    assert manifest["predecessor"]["old_scientific_result_reused"] is False
    components = manifest["complete_transitive_component_sha256"]
    for required in (
        "/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/script/utils.py",
        "/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/runner/executor.py",
        "/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/main_callback/multi_main_callback.py",
        "/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit/nuplan/planning/simulation/main_callback/metric_file_callback.py",
    ):
        assert required in components
    for component, expected in components.items():
        path = Path(component) if Path(component).is_absolute() else ROOT / component
        assert path.is_file(), component
        assert _sha(path) == expected, component
    assert manifest["authorization"] == {
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "RBR_A/B/C": "NOT_AUTHORIZED",
    }
