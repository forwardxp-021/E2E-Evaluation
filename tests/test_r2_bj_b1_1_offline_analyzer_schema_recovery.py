import hashlib
import json
from pathlib import Path

import pytest

import tools.r2_bj_b1_1_offline_recovery_launcher as launcher


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "tools/r2_bj_b0_2_frozen_canary_pair_analyzer.py"
RECOVERY = ROOT / "tools/r2_bj_b1_1_offline_recovery_analyzer_v1.py"
PARAMETERS = ROOT / "docs/stageR/r2/r2_bj_a_hlc_global_parameter_space_v4.0.json"
GENERATOR = ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py"
PLANNER = ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py"


def test_recovery_analyzer_is_exactly_the_single_preregistered_replacement():
    original, recovery = ORIGINAL.read_text(), RECOVERY.read_text()
    old = 'capture["capture_end_abs_s"]'
    new = 'capture["nominal_capture_end_abs_s"]'
    assert original.count(old) == 1
    assert recovery == original.replace(old, new, 1)
    assert recovery.count(new) == 1


def test_v4_parameter_and_generator_planner_nominal_schema_agree():
    capture = json.loads(PARAMETERS.read_text())["global_parameters"]["capture"]
    assert capture["nominal_capture_end_abs_s"] == 7.4
    assert "capture_end_abs_s" not in capture
    generator = GENERATOR.read_text()
    planner = PLANNER.read_text()
    assert 'capture["nominal_capture_end_abs_s"]' in generator
    assert "compose_kinematic_trajectory" in planner
    assert 'parameters["capture"]' in planner


def test_recovery_tools_have_no_execution_or_selection_path():
    text = RECOVERY.read_text() + (ROOT / "tools/r2_bj_b1_1_offline_recovery_launcher.py").read_text()
    forbidden = (
        ".runner.run(",
        "build_production_runner",
        "run_simulation.py",
        "SimulationRunner",
        "select_roster",
        "roster_selection",
    )
    assert all(token not in text for token in forbidden)


def test_existing_output_rejects_second_invocation(monkeypatch, tmp_path):
    output = tmp_path / "once"
    output.mkdir()
    monkeypatch.setattr(launcher, "validate_b1_artifact_closure", lambda: None)
    monkeypatch.setattr(launcher, "validate_pre_recovery_manifest", lambda *_: None)
    monkeypatch.setattr(launcher, "AUTHORIZATION", tmp_path / "auth.json")
    (tmp_path / "auth.json").write_text(json.dumps({
        "OFFLINE_EXISTING_ARTIFACT_ANALYSIS_AUTHORIZED": True,
        "OFFLINE_ANALYZER_INVOCATION_BUDGET": 1,
        "NEW_RUN_BUDGET": 0,
        "RUNNER_RUN_AUTHORIZED": False,
        "AUTHORIZED_OUTPUT_ROOT": str(output),
    }))
    with pytest.raises(launcher.OfflineRecoveryStop, match="OUTPUT_ALREADY_EXISTS"):
        launcher.run_once(tmp_path / "manifest.json", "a" * 64, output, analyzer=lambda *_: {})


def test_sha_mismatch_stops_before_analyzer(monkeypatch, tmp_path):
    called = {"analyzer": 0}
    monkeypatch.setattr(launcher, "validate_b1_artifact_closure", lambda: (_ for _ in ()).throw(launcher.OfflineRecoveryStop("SHA_MISMATCH")))
    def analyzer(*_):
        called["analyzer"] += 1
        return {}
    with pytest.raises(launcher.OfflineRecoveryStop, match="SHA_MISMATCH"):
        launcher.run_once(tmp_path / "manifest.json", "a" * 64, tmp_path / "out", analyzer=analyzer)
    assert called["analyzer"] == 0


def test_static_frozen_input_hashes_are_exact():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == launcher.EXPECTED_ORIGINAL_ANALYZER_SHA256
    assert hashlib.sha256(PARAMETERS.read_bytes()).hexdigest() == launcher.EXPECTED_V4_PARAMETER_SHA256
    assert hashlib.sha256(launcher.B1_MANIFEST.read_bytes()).hexdigest() == launcher.EXPECTED_B1_MANIFEST_SHA256
