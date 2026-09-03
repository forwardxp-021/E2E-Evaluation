import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_fresh_roster_and_additive_data_firewall_are_closed():
    roster = load("r2_a_controller_id_dev_canary_roster_v1.0.json")
    exclusion = load("r2_a_controller_id_permanent_exclusion_ledger_v1.0.json")
    selected = roster["entries"]
    assert roster["counts"] == {"R-HLC": 8, "R-TSB": 8, "total": 16}
    assert len({row["scenario_token"] for row in selected}) == 16
    assert len({row["log_id"] for row in selected}) == 16
    assert roster["source_universe"]["reused"] is True
    assert roster["outcome_or_safety_or_F_match_used"] is False
    assert roster["confirmatory_roster"] is False
    assert all(row["PERMANENT_R2_ENGINEERING_ONLY"] for row in selected)
    assert all(row["R2_CONFIRMATORY_USE_FORBIDDEN"] for row in selected)
    assert all(row["RBR_SCIENTIFIC_USE_FORBIDDEN"] for row in selected)
    assert exclusion["counts"] == {
        "historical_permanent_exclusions": 45,
        "R1_official_outcome_exposed": 24,
        "R2_A_fresh_engineering_only": 16,
        "effective_unique_identities": 85,
    }
    dev = [row for row in exclusion["entries"] if "R2_A_CONTROLLER_IDENTIFICATION_DEV_IDENTITY" in row["reasons"]]
    assert len(dev) == 16
    assert {(row["scenario_token"], row["log_id"]) for row in dev} == {
        (row["scenario_token"], row["log_id"]) for row in selected
    }


def test_frozen_design_and_zero_run_closure():
    hlc = load("r2_a_hlc_excitation_grid_v1.0.json")
    tsb = load("r2_a_tsb_excitation_grid_v1.0.json")
    zero = load("r2_a_zero_run_construction_audit_v1.0.json")
    assert len(hlc["excitations"]) == 5
    assert len(tsb["excitations"]) == 5
    assert hlc["online_adaptation_allowed"] is False
    assert tsb["online_adaptation_allowed"] is False
    assert zero["status"] == "80_OF_80_ZERO_RUN_CONSTRUCTION_PASS"
    assert zero["counts"]["exact_resolution"] == 80
    assert zero["counts"]["runner_construction"] == 80
    assert zero["counts"]["actual_engineering_runs"] == 0


def test_authorized_engineering_execution_and_telemetry_lifecycle():
    audit = load("r2_a_controller_transfer_execution_audit_v1.0.json")
    assert audit["status"] == "80_OF_80_FROZEN_DEV_RUNS_TECHNICAL_COMPLETE"
    assert audit["counts"] == {
        "planned": 80,
        "executed": 80,
        "technical_reruns": 4,
        "actual_engineering_runs": 84,
    }
    assert audit["actual_runner_run_calls"] == 84
    assert audit["scientific_simulations"] == 0
    assert audit["confirmatory_roster_selected"] is False
    assert audit["RBR_started"] is False
    assert len(audit["effective_runs"]) == 80
    assert len({row["frozen_run_id"] for row in audit["effective_runs"]}) == 80
    for row in audit["effective_runs"]:
        assert row["status"].startswith("TECHNICAL_COMPLETE")
        assert sum(1 for key in ("trace_path", "planner_telemetry_path", "controller_command_path") if row[key]) == 3


def test_identification_and_identity_held_out_validation():
    hlc = load("r2_a_hlc_transfer_identification_v1.json")
    tsb = load("r2_a_tsb_transfer_identification_v1.json")
    surrogate = load("r2_a_controller_transfer_surrogate_v1.json")
    assert hlc["counts"] == {"identities": 8, "effective_runs": 40, "reference_runs": 8, "hesitation_runs": 32}
    assert tsb["counts"] == {"identities": 8, "effective_runs": 40, "reference_runs": 8, "two_pulse_runs": 32}
    assert len(hlc["runs"]) == 40 and len(tsb["runs"]) == 40
    assert all("commanded_monotonic_effect" in row and "realized_monotonic_effect" in row for row in hlc["runs"])
    assert tsb["phase_formation"]["phase_loss_count"] == 32
    assert tsb["phase_formation"]["two_distinct_phases_count"] == 0
    assert surrogate["status"] == "ENGINEERING_MODEL_ONLY"
    assert surrogate["HLC"]["validation"]["held_out_identities"] == 8
    assert surrogate["TSB"]["peak_decel_validation"]["held_out_identities"] == 8
    assert surrogate["complex_black_box_used"] is False
    assert surrogate["R1_official_identity_used"] is False
    assert surrogate["scientific_threshold_changed"] is False
    assert surrogate["final_generator_parameters_frozen"] is False


def test_sha_binding_and_protected_csv():
    manifest = load("r2_a_controller_transfer_identification_binding_manifest_v1.0.json")
    assert manifest["status"] == "R2_A_CONTROLLER_TRANSFER_DIAGNOSTIC_FROZEN_COMPLETE"
    assert manifest["counts"] == {
        "fresh_DEV_identities": 16,
        "frozen_effective_runs": 80,
        "technical_reruns": 4,
        "actual_engineering_runs": 84,
        "bound_effective_telemetry_sets": 80,
    }
    assert len(manifest["effective_runtime_telemetry"]) == 80
    assert len({row["frozen_run_id"] for row in manifest["effective_runtime_telemetry"]}) == 80
    first_planner = None
    first_control = None
    for telemetry in manifest["effective_runtime_telemetry"]:
        assert len(telemetry["files"]) == 3
        for bound in telemetry["files"]:
            path = ROOT / bound["path"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == bound["sha256"]
            if first_planner is None and bound["role"] == "planner_telemetry_path":
                first_planner = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            if first_control is None and bound["role"] == "controller_command_path":
                first_control = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert [row["state_index"] for row in first_planner["controller_lookahead"]["states_0_to_10"]] == list(range(11))
    assert first_control["instrumentation"] == "PASSIVE_RETURN_VALUE_WRAPPER_NO_BEHAVIOR_CHANGE"
    assert "acceleration_command_mps2" in first_control and "steering_rate_command_radps" in first_control
    for component in manifest["components"]:
        path = ROOT / component["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == component["sha256"]
    assert hashlib.sha256(PROTECTED.read_bytes()).hexdigest() == PROTECTED_SHA
    assert manifest["protected_CSV_sha256"] == PROTECTED_SHA
    assert manifest["scientific_threshold_changed"] is False
    assert manifest["final_R2_generator_implemented"] is False
    assert manifest["R2_confirmatory_roster_selected"] is False
    assert manifest["RBR_started"] is False
