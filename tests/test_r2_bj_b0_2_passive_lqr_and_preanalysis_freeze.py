import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.r2_bj_b0_2_passive_actual_lqr_recorder as recorder_module
import tools.r2_bj_b0_2_production_launcher_adapter as adapter
from tools.r2_bj_b0_1_production_canary_launcher import B01ControlPlaneStop, ProductionRunnerBundle, exact_slice
from tools.r2_bj_b0_2_frozen_canary_pair_analyzer import RESULT_STATES, analyze_frozen_canary_pair, frozen_analysis_binding
from tools.r2_bj_b0_2_passive_actual_lqr_recorder import PassiveActualLQRRecorderV1


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


class FakeResult:
    def __init__(self, acceleration=1.25, steering=-0.4):
        self.rear_axle_acceleration_2d = SimpleNamespace(x=acceleration)
        self.tire_steering_rate = steering


class LQRTracker:
    _stopping_velocity = 0.1
    _tracking_horizon = 4
    _discretization_time = 0.1

    def __init__(self, result=None):
        self.result = result or FakeResult()
        self.calls = 0

    def _compute_initial_velocity_and_lateral_state(self, *_args):
        return 4.0, np.asarray([0.1, -0.2, 0.03])

    def _compute_reference_velocity_and_curvature_profile(self, *_args):
        return 5.0, np.asarray([0.01, 0.02, 0.03, 0.04])

    def _longitudinal_lqr_controller(self, *_args):
        return 1.25

    def _lateral_lqr_controller(self, *_args):
        return -0.4

    def _stopping_controller(self, *_args):
        return 0.0, 0.0

    def track_trajectory(self, *_args):
        self.calls += 1
        return self.result


class TwoStageController:
    def __init__(self, tracker=None):
        self._tracker = tracker or LQRTracker()


class R1Primary80ScientificTimeControllerV1:
    def number_of_iterations(self):
        return 81


def iteration(index):
    return SimpleNamespace(index=index, time_point=SimpleNamespace(time_s=index / 10.0))


def run_row(arm="BASELINE"):
    return {"run_id": f"run-{arm}", "pair_id": "pair", "arm": arm}


def install(tmp_path, monkeypatch, arm="BASELINE"):
    monkeypatch.setattr(
        recorder_module,
        "_frozen_velocity_profile",
        lambda initial, acceleration, tracker: np.asarray([initial + acceleration * 0.1 * i for i in range(4)]),
    )
    controller, time_controller = TwoStageController(), R1Primary80ScientificTimeControllerV1()
    recorder = PassiveActualLQRRecorderV1(tmp_path / "actual.jsonl", run_row(arm), {"B0": "a" * 64})
    original = controller._tracker.track_trajectory
    recorder.install(controller, time_controller)
    return recorder, controller, original


def test_passive_wrapper_preserves_exact_result_identity_and_commands(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    expected = controller._tracker.result
    result = controller._tracker.track_trajectory(iteration(0), iteration(1), object(), object())
    assert result is expected
    assert result.rear_axle_acceleration_2d.x == 1.25
    assert result.tire_steering_rate == -0.4
    row = json.loads((tmp_path / "actual.jsonl").read_text())
    assert row["actual_acceleration_command_mps2"] == row["shadow_acceleration_command_mps2"]
    assert row["actual_tire_steering_rate_command_radps"] == row["shadow_tire_steering_rate_command_radps"]
    assert row["behavior_changed"] is False


def test_install_and_uninstall_outputs_are_exactly_identical(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    installed = controller._tracker.track_trajectory(iteration(0), iteration(1), object(), object())
    recorder.uninstall()
    uninstalled = controller._tracker.track_trajectory(iteration(1), iteration(2), object(), object())
    assert installed is uninstalled is controller._tracker.result


def test_exact_79_rows_pass_and_80th_fails_closed(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    for index in range(79):
        controller._tracker.track_trajectory(iteration(index), iteration(index + 1), object(), object())
    recorder.validate_complete()
    assert recorder.row_count == 79
    with pytest.raises(RuntimeError, match="EXCEEDS_79"):
        controller._tracker.track_trajectory(iteration(79), iteration(80), object(), object())


def test_missing_row_and_nonfinite_fail_closed(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    controller._tracker.track_trajectory(iteration(0), iteration(1), object(), object())
    with pytest.raises(RuntimeError, match="CARDINALITY"):
        recorder.validate_complete()
    controller._tracker.result = FakeResult(float("nan"), 0.0)
    with pytest.raises(ValueError, match="NONFINITE_ACTUAL_ACCELERATION"):
        controller._tracker.track_trajectory(iteration(1), iteration(2), object(), object())


def test_direction_disagreement_is_recorded_without_command_change(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    controller._tracker.result = FakeResult(1.25, 0.4)
    expected = controller._tracker.result
    actual = controller._tracker.track_trajectory(iteration(0), iteration(1), object(), object())
    row = json.loads((tmp_path / "actual.jsonl").read_text())
    assert actual is expected and actual.tire_steering_rate == 0.4
    assert row["steering_direction_agreement"] is False


def test_recorder_persistence_failure_propagates_and_never_retries(monkeypatch, tmp_path):
    recorder, controller, _ = install(tmp_path, monkeypatch)
    monkeypatch.setattr(recorder_module, "_atomic_jsonl", lambda *_: (_ for _ in ()).throw(OSError("synthetic")))
    with pytest.raises(OSError, match="synthetic"):
        controller._tracker.track_trajectory(iteration(0), iteration(1), object(), object())
    assert controller._tracker.calls == 1 and recorder.row_count == 0


def test_wrong_controller_tracker_or_primary80_rejected(tmp_path):
    recorder = PassiveActualLQRRecorderV1(tmp_path / "x", run_row(), {})
    with pytest.raises(TypeError, match="TWO_STAGE"):
        recorder.validate_installability(object(), R1Primary80ScientificTimeControllerV1())
    wrong_tracker = TwoStageController(); wrong_tracker._tracker = object()
    with pytest.raises(TypeError, match="LQR_TRACKER"):
        recorder.validate_installability(wrong_tracker, R1Primary80ScientificTimeControllerV1())
    wrong_time = SimpleNamespace(number_of_iterations=lambda: 81)
    with pytest.raises(TypeError, match="PRIMARY80"):
        recorder.validate_installability(TwoStageController(), wrong_time)


class MockRunner:
    def __init__(self, bundle_ref, calls, transitions=79):
        self._simulation = SimpleNamespace(
            _ego_controller=TwoStageController(),
            _time_controller=R1Primary80ScientificTimeControllerV1(),
        )
        self.bundle_ref = bundle_ref
        self.calls = calls
        self.transitions = transitions

    def run(self):
        bundle = self.bundle_ref[0]
        self.calls.append(bundle.run_root.name)
        for index in range(self.transitions):
            self._simulation._ego_controller._tracker.track_trajectory(
                iteration(index), iteration(index + 1), object(), object()
            )
        for path in (
            bundle.run_root / "trace/realized_current_ego.jsonl",
            bundle.run_root / "telemetry/planner_v4_online_gate.jsonl",
            bundle.run_root / "telemetry/controller_visible_telemetry.jsonl",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" * 80)
        for name in ("no_ego_at_fault_collisions.parquet", "drivable_area_compliance.parquet", "runner_report.parquet"):
            (bundle.run_root / name).write_bytes(b"test")
        return SimpleNamespace(succeeded=True)


def mock_builder(calls, transitions=79):
    def build(_run, run_root):
        holder = []
        bundle = ProductionRunnerBundle(None, run_root, None, SimpleNamespace(runner_report_file="runner_report.parquet"))
        bundle.runner = MockRunner(holder, calls, transitions)
        holder.append(bundle)
        return bundle
    return build


def synthetic_authorization():
    return {"AUTHORIZED_B0_2_EXECUTION_OBSERVABILITY_MANIFEST_SHA256": "b" * 64}


def test_production_adapter_installs_before_claim_and_runs_exactly_two(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "validate_b02_authorization", lambda *_: None)
    monkeypatch.setattr(adapter, "validate_production_control_plane", lambda *_: None)
    monkeypatch.setattr(recorder_module, "_frozen_velocity_profile", lambda initial, acceleration, tracker: np.ones(4))
    calls = []
    result = adapter.run_b02_production_canary(
        synthetic_authorization(), tmp_path / "out", tmp_path / "control",
        runner_builder=mock_builder(calls),
        pair_analyzer=lambda *_: {"result_state": RESULT_STATES[3]},
    )
    assert calls == [row["run_id"] for row in exact_slice()]
    assert result["runner_run_attempt_count"] == 2 and result["remaining_budget"] == 0
    persisted = json.loads((tmp_path / "control/canary_pair_analysis.json").read_text())
    assert persisted["result_state"] == RESULT_STATES[3]


def test_missing_79th_controller_transition_stops_before_treatment(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "validate_b02_authorization", lambda *_: None)
    monkeypatch.setattr(adapter, "validate_production_control_plane", lambda *_: None)
    monkeypatch.setattr(recorder_module, "_frozen_velocity_profile", lambda initial, acceleration, tracker: np.ones(4))
    calls = []
    with pytest.raises(B01ControlPlaneStop) as error:
        adapter.run_b02_production_canary(
            synthetic_authorization(), tmp_path / "out", tmp_path / "control",
            runner_builder=mock_builder(calls, transitions=78), pair_analyzer=lambda *_: {},
        )
    assert error.value.reason == "B0_2_ACTUAL_LQR_TELEMETRY_INCOMPLETE"
    assert len(calls) == 1


def test_architecture_audit_has_priority_over_missing_controller_rows(tmp_path):
    root = tmp_path / "run"
    (root / "telemetry").mkdir(parents=True)
    (root / "telemetry/architecture_failure_audit.json").write_text("{}\n")
    bundle = ProductionRunnerBundle(None, root)
    with pytest.raises(B01ControlPlaneStop) as error:
        adapter.validate_b02_technical_completion(bundle, SimpleNamespace(succeeded=False))
    assert error.value.classification == "ARCHITECTURE_FAILURE"


def test_analyzer_architecture_priority_and_unbound_inputs_fail_closed(tmp_path):
    runs = exact_slice()
    architecture = tmp_path / runs[0]["run_id"] / "telemetry/architecture_failure_audit.json"
    architecture.parent.mkdir(parents=True)
    architecture.write_text("{}\n")
    result = analyze_frozen_canary_pair(tmp_path, runs, evaluator=lambda **_: {})
    assert result["result_state"] == RESULT_STATES[0]
    assert result["ordinary_failure"] is False
    result = analyze_frozen_canary_pair(tmp_path / "missing", runs, evaluator=lambda **_: {})
    assert result["result_state"] == RESULT_STATES[1]


def test_analyzer_nondeterministic_repeat_fails_closed(monkeypatch, tmp_path):
    import tools.r2_bj_b0_2_frozen_canary_pair_analyzer as analyzer

    monkeypatch.setattr(analyzer, "_technical_artifacts", lambda *_: None)
    monkeypatch.setattr(analyzer, "frozen_analysis_binding", lambda: {"pair_id": "R2BJB0-HLC-01"})
    monkeypatch.setattr(analyzer, "_arm_audit", lambda *_: {"actual_lqr_rows": 79})
    calls = {"n": 0}

    def nondeterministic(**_kwargs):
        calls["n"] += 1
        return {"evaluation": {"nonce": calls["n"]}}

    result = analyzer.analyze_frozen_canary_pair(tmp_path, exact_slice(), evaluator=nondeterministic)
    assert result["result_state"] == RESULT_STATES[1]
    assert "NONDETERMINISTIC_REPEAT_ANALYSIS_OUTPUT" in result["reason"]


def test_contracts_keep_formal_gate_closed_and_label_reference_steering():
    technical = json.loads((R2 / "r2_bj_b0_2_technical_completion_contract_v1.0.json").read_text())
    states = json.loads((R2 / "r2_bj_b0_2_canary_result_state_contract_v1.0.json").read_text())
    gate = json.loads((R2 / "r2_bj_b0_2_closed_authorization_gate_v1.0.json").read_text())
    assert technical["primary80"]["actual_lqr_rows"] == 79
    assert technical["telemetry_semantics"]["controller_visible_telemetry.jsonl"]["disposition"] == "NOT_ACTUAL_CONTROLLER_COMMAND"
    assert states["allowed_states"] == list(RESULT_STATES)
    assert gate["CANARY_AUTHORIZED"] is False and gate["NEW_RUN_BUDGET"] == gate["RUNNER_RUN"] == 0


def test_manifest_closure_and_frozen_pair_analysis_binding_are_complete():
    adapter.validate_b02_component_closure(R2 / "r2_bj_b0_2_execution_observability_sha_manifest_v1.0.json")
    binding = frozen_analysis_binding()
    assert binding["pair_id"] == "R2BJB0-HLC-01"
    assert binding["scenario_token"] == "cc1abd3989065d8d"
    assert binding["pretreatment_clearance"]["pretreatment_only"] is True
    assert len(binding["source_reference_xy"]) > 80 and len(binding["target_reference_xy"]) > 80
