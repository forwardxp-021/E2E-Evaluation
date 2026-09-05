import hashlib
import json
from pathlib import Path

import pytest

from tools.r2_bj_b0_1_failure_persisting_telemetry_wrapper import R2BJB01FailurePersistingTelemetryWrapper
from tools.r2_bj_b0_1_production_canary_launcher import (
    B01ControlPlaneStop,
    B01_COMPONENT,
    B0_BINDINGS,
    B0_COMPONENT,
    B0_SCHEDULE,
    EXACT_RUN_IDS,
    EXPECTED,
    ProductionRunnerBundle,
    exact_slice,
    run_production_canary,
    validate_real_technical_completion,
)
from tools.r2_bj_b0_hlc_v4_engineering_planner import B0ArchitectureViolation, R2BJB0HLCV4EngineeringPlanner


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_authorization(output_root, control_root):
    return {
        "BJ_B_ENGINEERING_SIMULATION_AUTHORIZED": True,
        "CANARY_AUTHORIZED": True,
        "AUTHORIZED_RUN_ORDERS": [1, 2],
        "AUTHORIZED_RUN_IDS": list(EXACT_RUN_IDS),
        "NEW_RUN_BUDGET": 2,
        "AUTHORIZATION_CONSUMED": False,
        "AUTHORIZED_OUTPUT_ROOT": str(output_root.resolve()),
        "AUTHORIZED_CONTROL_ROOT": str(control_root.resolve()),
        "AUTHORIZED_B0_1_EXECUTION_COMPONENT_MANIFEST_SHA256": file_sha(B01_COMPONENT),
        "authorized": {
            "B0_component_manifest_sha256": file_sha(B0_COMPONENT),
            "B0_schedule_sha256": file_sha(B0_SCHEDULE),
            "B0_pair_binding_sha256": file_sha(B0_BINDINGS),
        },
    }


class MockReport:
    succeeded = True


class MockRunner:
    def __init__(self, run_id, calls, failure=None):
        self.run_id = run_id
        self.calls = calls
        self.failure = failure

    def run(self):
        self.calls.append(self.run_id)
        if self.failure:
            raise self.failure
        return MockReport()


def factory(calls, failures=None, builds=None):
    failures = failures or {}

    def build(run, run_root):
        if builds is not None:
            builds.append(run["run_id"])
        return ProductionRunnerBundle(MockRunner(run["run_id"], calls, failures.get(run["run_id"])), run_root)

    return build


def pass_completion(_bundle, _report):
    return None


def test_historical_b0_bindings_and_exact_slice_are_immutable():
    contract = load("r2_bj_b0_1_production_execution_path_contract_v1.0.json")
    frozen = contract["immutable_B0"]
    assert file_sha(B0_COMPONENT) == frozen["component_manifest_sha256"]
    assert file_sha(B0_SCHEDULE) == frozen["schedule_sha256"]
    assert file_sha(B0_BINDINGS) == frozen["pair_binding_sha256"]
    assert file_sha(ROOT / "tools/r2_bj_b0_execute_frozen_hlc_v4_engineering.py") == frozen["zero_run_executor_sha256"]
    assert file_sha(ROOT / "tools/r2_bj_b0_hlc_v4_engineering_planner.py") == frozen["runtime_planner_sha256"]
    assert tuple(row["run_id"] for row in exact_slice()) == EXACT_RUN_IDS


def test_closed_formal_gate_is_zero_run():
    gate = load("r2_bj_b0_1_closed_authorization_gate_v1.0.json")
    assert gate["BJ_B_ENGINEERING_SIMULATION_AUTHORIZED"] is False
    assert gate["CANARY_AUTHORIZED"] is False
    assert gate["AUTHORIZED_RUN_ORDERS"] == []
    assert gate["NEW_RUN_BUDGET"] == 0
    assert gate["AUTHORIZATION_CONSUMED"] is False


def test_all_prestart_mutations_make_zero_runner_calls(tmp_path):
    schedule = load("r2_bj_b0_hlc_v4_pair_schedule_v1.0.json")["runs"]
    def valid_for(output, control):
        return valid_authorization(output, control)

    cases = []
    o1, c1 = tmp_path / "o1", tmp_path / "c1"; denied = valid_for(o1, c1); denied["CANARY_AUTHORIZED"] = False; cases.append((denied, exact_slice(), o1, c1))
    o2, c2 = tmp_path / "o2", tmp_path / "c2"; zero = valid_for(o2, c2); zero["NEW_RUN_BUDGET"] = 0; cases.append((zero, exact_slice(), o2, c2))
    o3, c3 = tmp_path / "o3", tmp_path / "c3"; bad_component = valid_for(o3, c3); bad_component["authorized"]["B0_component_manifest_sha256"] = "0" * 64; cases.append((bad_component, exact_slice(), o3, c3))
    o4, c4 = tmp_path / "o4", tmp_path / "c4"; bad_b01 = valid_for(o4, c4); bad_b01["AUTHORIZED_B0_1_EXECUTION_COMPONENT_MANIFEST_SHA256"] = "0" * 64; cases.append((bad_b01, exact_slice(), o4, c4))
    o5, c5 = tmp_path / "o5", tmp_path / "c5"; cases.append((valid_for(o5, c5), [schedule[0], schedule[2]], o5, c5))
    o6, c6 = tmp_path / "o6", tmp_path / "c6"; cases.append((valid_for(o6, c6), list(reversed(exact_slice())), o6, c6))
    altered = json.loads(json.dumps(exact_slice()))
    altered[0]["scenario_token"] = "0" * 16
    o6b, c6b = tmp_path / "o6b", tmp_path / "c6b"; cases.append((valid_for(o6b, c6b), altered, o6b, c6b))
    collision, c7 = tmp_path / "o7", tmp_path / "c7"; (collision / EXACT_RUN_IDS[0]).mkdir(parents=True); cases.append((valid_for(collision, c7), exact_slice(), collision, c7))
    o8, c8 = tmp_path / "o8", tmp_path / "c8"; wrong_root = valid_for(o8, c8); wrong_root["AUTHORIZED_CONTROL_ROOT"] = str((tmp_path / "other").resolve()); cases.append((wrong_root, exact_slice(), o8, c8))
    for authorization, requested, output, control in cases:
        calls, builds = [], []
        with pytest.raises(B01ControlPlaneStop):
            run_production_canary(authorization, output, control, factory(calls, builds=builds), pass_completion, requested_runs=requested)
        assert builds == []
        assert calls == []


@pytest.mark.parametrize("frozen_path", tuple(EXPECTED))
def test_every_frozen_input_sha_mismatch_stops_before_runner(monkeypatch, tmp_path, frozen_path):
    calls = []
    monkeypatch.setitem(EXPECTED, frozen_path, "0" * 64)
    output, control = tmp_path / "output", tmp_path / "control"
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(
            valid_authorization(output, control), output, control,
            factory(calls), pass_completion,
        )
    assert error.value.reason.startswith("FROZEN_INPUT_SHA_MISMATCH")
    assert calls == []


def test_two_successful_mock_runners_are_exactly_ordered_and_budget_reaches_zero(tmp_path):
    calls = []
    output, control = tmp_path / "output", tmp_path / "control"
    result = run_production_canary(valid_authorization(output, control), output, control, factory(calls), pass_completion)
    assert calls == list(EXACT_RUN_IDS)
    assert result["runner_run_attempt_count"] == 2 and result["remaining_budget"] == 0
    ledger = json.loads((tmp_path / "control/canary_attempt_ledger.json").read_text())
    assert [row["budget_remaining_after_claim"] for row in ledger["attempts"]] == [1, 0]
    assert ledger["remaining_budget"] == 0 and ledger["runner_run_attempt_count"] == 2
    assert all(row["status"] == "TECHNICAL_COMPLETE" for row in ledger["attempts"])


def test_baseline_architecture_failure_stops_treatment(tmp_path):
    calls = []
    failure = B0ArchitectureViolation(["CURVATURE_LIMIT"], {"classification": "ARCHITECTURE_FAILURE"})
    output, control = tmp_path / "output", tmp_path / "control"
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(valid_authorization(output, control), output, control, factory(calls, {EXACT_RUN_IDS[0]: failure}), pass_completion)
    assert error.value.classification == "ARCHITECTURE_FAILURE"
    assert calls == [EXACT_RUN_IDS[0]]
    ledger = json.loads((tmp_path / "control/canary_attempt_ledger.json").read_text())
    assert ledger["runner_run_attempt_count"] == 1
    assert ledger["attempts"][0]["status"] == "ARCHITECTURE_FAILURE_STOP_ALL"


def test_treatment_architecture_failure_has_two_attempts_and_no_third(tmp_path):
    calls = []
    failure = B0ArchitectureViolation(["TARGET_FRAME_RESIDUAL"], {"classification": "ARCHITECTURE_FAILURE"})
    output, control = tmp_path / "output", tmp_path / "control"
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(valid_authorization(output, control), output, control, factory(calls, {EXACT_RUN_IDS[1]: failure}), pass_completion)
    assert error.value.classification == "ARCHITECTURE_FAILURE"
    assert calls == list(EXACT_RUN_IDS)
    ledger = json.loads((tmp_path / "control/canary_attempt_ledger.json").read_text())
    assert ledger["runner_run_attempt_count"] == 2 and ledger["remaining_budget"] == 0
    assert ledger["attempts"][1]["status"] == "ARCHITECTURE_FAILURE_STOP_ALL"


def test_third_budget_claim_is_impossible(tmp_path):
    from tools.r2_bj_b0_1_production_canary_launcher import AttemptBudgetLedger

    ledger = AttemptBudgetLedger(tmp_path / "ledger.json", "1" * 64)
    ledger.claim_authorization_once()
    rows = exact_slice()
    ledger.claim_run(rows[0])
    ledger.finish("TECHNICAL_COMPLETE")
    ledger.claim_run(rows[1])
    ledger.finish("TECHNICAL_COMPLETE")
    with pytest.raises(B01ControlPlaneStop) as error:
        ledger.claim_run(rows[1])
    assert error.value.reason == "RUN_BUDGET_EXHAUSTED"
    assert ledger.remaining == 0
    assert len(ledger.attempts) == 2


def test_baseline_infrastructure_failure_stops_treatment(tmp_path):
    calls = []
    output, control = tmp_path / "output", tmp_path / "control"
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(valid_authorization(output, control), output, control, factory(calls, {EXACT_RUN_IDS[0]: OSError("synthetic")}), pass_completion)
    assert error.value.classification == "INFRASTRUCTURE_FAILURE"
    assert calls == [EXACT_RUN_IDS[0]]


def test_incomplete_baseline_telemetry_stops_before_treatment_construction(tmp_path):
    calls, builds = [], []
    output, control = tmp_path / "output", tmp_path / "control"

    def incomplete(_bundle, _report):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "PRIMARY80_TELEMETRY_INCOMPLETE")

    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(
            valid_authorization(output, control), output, control,
            factory(calls, builds=builds), incomplete,
        )
    assert error.value.reason == "PRIMARY80_TELEMETRY_INCOMPLETE"
    assert builds == [EXACT_RUN_IDS[0]]
    assert calls == [EXACT_RUN_IDS[0]]


def test_runner_construction_failure_consumes_authorization_and_stops(tmp_path):
    calls = []
    output, control = tmp_path / "output", tmp_path / "control"

    def fail_construction(_run, _run_root):
        raise OSError("synthetic construction failure")

    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(
            valid_authorization(output, control), output, control,
            fail_construction, pass_completion,
        )
    assert error.value.reason == "RUNNER_CONSTRUCTION_EXCEPTION_STOP_ALL"
    assert calls == []
    ledger = json.loads((control / "canary_attempt_ledger.json").read_text())
    assert ledger["AUTHORIZATION_CONSUMED"] is True
    assert ledger["runner_run_attempt_count"] == 0


def test_authorization_ledger_serialization_failure_stops_before_construction(monkeypatch, tmp_path):
    import tools.r2_bj_b0_1_production_canary_launcher as launcher

    builds, calls = [], []
    output, control = tmp_path / "output", tmp_path / "control"
    monkeypatch.setattr(launcher, "atomic_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic serialization failure")))
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(
            valid_authorization(output, control), output, control,
            factory(calls, builds=builds), pass_completion,
        )
    assert error.value.reason == "AUTHORIZATION_LEDGER_PERSISTENCE_FAILURE_BEFORE_RUNNER_CONSTRUCTION"
    assert builds == [] and calls == []


def test_consumed_authorization_cannot_be_invoked_again(tmp_path):
    calls = []
    output, control = tmp_path / "output", tmp_path / "control"
    authorization = valid_authorization(output, control)
    run_production_canary(authorization, output, control, factory(calls), pass_completion)
    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(authorization, output, control, factory(calls), pass_completion)
    assert error.value.reason == "AUTHORIZATION_ALREADY_CONSUMED"
    assert calls == list(EXACT_RUN_IDS)


def test_failure_wrapper_atomically_persists_before_reraise(monkeypatch, tmp_path):
    wrapper = object.__new__(R2BJB01FailurePersistingTelemetryWrapper)
    wrapper._b01_run_id = EXACT_RUN_IDS[0]
    wrapper._b01_pair_id = "R2BJB0-HLC-01"
    wrapper._development_arm = "BASELINE"
    wrapper._parameters = {"frozen": True}
    wrapper._b01_failure_path = tmp_path / "telemetry/architecture_failure_audit.json"
    wrapper._b01_component_manifest_sha256 = file_sha(B0_COMPONENT)
    wrapper._b01_schedule_sha256 = file_sha(B0_SCHEDULE)
    wrapper._b01_pair_binding_sha256 = file_sha(B0_BINDINGS)
    context = {"iteration": 7, "absolute_episode_time_s": 0.7, "realized_current_ego": {"time_us": 7}}
    monkeypatch.setattr(R2BJB01FailurePersistingTelemetryWrapper, "_call_context", lambda self, value: context)
    failure = B0ArchitectureViolation(["YAW_RATE_LIMIT"], {"complete": True})
    monkeypatch.setattr(R2BJB0HLCV4EngineeringPlanner, "compute_planner_trajectory", lambda self, value: (_ for _ in ()).throw(failure))
    with pytest.raises(B0ArchitectureViolation):
        wrapper.compute_planner_trajectory(object())
    record = json.loads(wrapper._b01_failure_path.read_text())
    assert record["failure_codes"] == ["YAW_RATE_LIMIT"]
    assert record["error_audit"] == {"complete": True}
    assert record["STOP_CURRENT_RUN"] is record["STOP_REMAINING_SCHEDULE"] is True
    assert not wrapper._b01_failure_path.with_name("architecture_failure_audit.json.partial").exists()


def test_persisted_architecture_audit_precedes_failed_runner_report_classification(tmp_path):
    failure_path = tmp_path / "telemetry/architecture_failure_audit.json"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text('{"classification":"ARCHITECTURE_FAILURE"}\n', encoding="utf-8")
    report = MockReport()
    report.succeeded = False
    with pytest.raises(B01ControlPlaneStop) as error:
        validate_real_technical_completion(ProductionRunnerBundle(None, tmp_path), report)
    assert error.value.classification == "ARCHITECTURE_FAILURE"
    assert error.value.reason == "PERSISTED_ARCHITECTURE_FAILURE_PRESENT"


def test_secondary_lifecycle_exception_cannot_downgrade_persisted_architecture_failure(tmp_path):
    calls = []
    output, control = tmp_path / "output", tmp_path / "control"

    class PersistThenRaise:
        def __init__(self, run_id, run_root):
            self.run_id = run_id
            self.run_root = run_root

        def run(self):
            calls.append(self.run_id)
            path = self.run_root / "telemetry/architecture_failure_audit.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"classification":"ARCHITECTURE_FAILURE"}\n', encoding="utf-8")
            raise OSError("synthetic secondary lifecycle error")

    def build(run, run_root):
        return ProductionRunnerBundle(PersistThenRaise(run["run_id"], run_root), run_root)

    with pytest.raises(B01ControlPlaneStop) as error:
        run_production_canary(
            valid_authorization(output, control), output, control, build, pass_completion,
        )
    assert error.value.classification == "ARCHITECTURE_FAILURE"
    assert calls == [EXACT_RUN_IDS[0]]
    ledger = json.loads((control / "canary_attempt_ledger.json").read_text())
    assert ledger["attempts"][0]["status"] == "ARCHITECTURE_FAILURE_STOP_ALL"


def test_normal_wrapper_path_returns_exact_parent_object(monkeypatch):
    wrapper = object.__new__(R2BJB01FailurePersistingTelemetryWrapper)
    sentinel = object()
    context = {"iteration": 0, "absolute_episode_time_s": 0.0, "realized_current_ego": {}}
    monkeypatch.setattr(R2BJB01FailurePersistingTelemetryWrapper, "_call_context", lambda self, value: context)
    monkeypatch.setattr(R2BJB01FailurePersistingTelemetryWrapper, "_persist_controller_visible_telemetry", lambda self, value: None)
    monkeypatch.setattr(R2BJB0HLCV4EngineeringPlanner, "compute_planner_trajectory", lambda self, value: sentinel)
    assert wrapper.compute_planner_trajectory(object()) is sentinel


def test_zero_run_audit_and_execution_manifest_close():
    audit = load("r2_bj_b0_1_zero_run_production_path_audit_v1.0.json")
    manifest = load("r2_bj_b0_1_execution_component_sha_manifest_v1.0.json")
    assert audit["status"] == "R2_BJ_B0_1_PRODUCTION_PATH_READY_FOR_OWNER_CANARY_AUTHORIZATION"
    assert audit["RUNNER_RUN"] == audit["NEW_RUN_BUDGET"] == 0
    assert audit["CANARY_AUTHORIZED"] is False
    assert audit["STOP_REMAINING_SCHEDULE"] == "MECHANICALLY_TESTED"
    assert audit["FAILURE_AUDIT_PERSISTENCE"] == "PASS"
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["self_reference"] is False
    assert manifest["owner_authorization_included"] is False
    for row in manifest["components"]:
        assert file_sha(ROOT / row["path"]) == row["sha256"]
    for row in manifest["external_bound_nuplan_1_2_2_runtime_components"]:
        assert file_sha(row["absolute_path"]) == row["sha256"]
