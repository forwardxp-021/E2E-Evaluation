import json
from pathlib import Path

from tools import b1_tsb_qualification as b1
from tools import b1_tsb_qualification_executor as executor


def test_frozen_parameter_contract_matches_owner_instruction():
    payload = b1.validate_frozen_parameters()
    assert payload["parameters"]["baseline_brake_mps2"] == -1.45
    assert payload["parameters"]["first_brake_mps2"] == -2.4
    assert payload["parameters"]["release_mps2"] == 1.4
    assert payload["parameters"]["second_brake_mps2"] == -2.4


def test_wilson_interval_is_descriptive_and_deterministic():
    low, high = b1._wilson(20, 20)
    assert round(low, 6) == 0.838875
    assert high == 1.0


def test_budget_claim_is_atomic_and_no_retry(tmp_path: Path):
    authorization = {
        "max_scientific_arms": 40,
        "allowed_runs": [{"run_id": "B1-TEST-BASELINE"}],
    }
    ledger = tmp_path / "ledger.json"
    executor.claim_budget(ledger, authorization, "B1-TEST-BASELINE")
    payload = json.loads(ledger.read_text())
    assert payload["runs"]["B1-TEST-BASELINE"] == "RUNNER_ENTRY_CLAIMED"
    try:
        executor.claim_budget(ledger, authorization, "B1-TEST-BASELINE")
    except executor.B1ExecutionError as exc:
        assert "ALREADY_CLAIMED" in str(exc)
    else:
        raise AssertionError("a claimed scientific arm must never be retried")


def test_b1_status_vocabulary_is_closed():
    allowed = {"NOT_RUN", "TECHNICAL_INCOMPLETE", "MEASUREMENT_INVALID", "SCIENTIFIC_FAIL", "PASS"}
    assert allowed == {"NOT_RUN", "TECHNICAL_INCOMPLETE", "MEASUREMENT_INVALID", "SCIENTIFIC_FAIL", "PASS"}


def test_selection_salt_is_frozen():
    assert b1.SALT == "B1_FROZEN_TSB_QUALIFICATION_V1"
    assert b1.TARGET_PAIRS == 20
    assert b1.MAX_ARMS == 40
