from __future__ import annotations

import json
import hashlib
from pathlib import Path

from tools import r2_bk_verify_family_scope_freeze as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_input_sha_closure_and_zero_run_freeze() -> None:
    result = verifier.verify()
    assert result["status"] == "PASS_R2_BK_ZERO_RUN_FREEZE_VERIFICATION"
    assert result["runner_run"] == 0
    assert result["roster_selected"] is False


def test_protocol_is_zero_run_and_does_not_select_roster() -> None:
    path = ROOT / "docs/stageR/r2/r2_bk_tsb_only_r2c_protocol_amendment_draft_v0.1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "TSB_R2C_SAMPLE_SIZE_REQUIRES_OWNER_DECISION"
    assert payload["capacity_census"]["roster_selection"] is False
    assert payload["forbidden_this_stage"]["runner_run"] is True
    assert payload["paired_design"]["all_pair_joint_success_required"] is True


def test_census_source_contains_no_simulation_or_recovery_invocation() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "build_simulations" not in source
    assert "SimulationRunner" not in source
    assert "r2_bj_b1_1_offline_recovery_analyzer_v1" not in source


def test_family_scope_bifurcation_is_explicit() -> None:
    path = ROOT / "docs/stageR/r2/r2_bk_family_scope_bifurcation_contract_v1.0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["claims"] == {
        "COMBINED_G_R2_CLAIM": "NOT_AVAILABLE",
        "HLC_CLAIM": "DEVELOPMENT_NONCONVERGENCE_NEGATIVE_RESULT",
        "TSB_CLAIM": "INDEPENDENT_FAMILY_CANDIDATE_PENDING_FRESH_VALIDATION",
        "CROSS_FAMILY_POOLING": "PROHIBITED",
    }


def test_component_manifest_has_no_self_reference_and_all_sha_match() -> None:
    manifest_path = ROOT / "docs/stageR/r2/r2_bk_component_sha_binding_manifest_v1.0.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["self_reference"] is False
    for component in payload["components"]:
        path = ROOT / component["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == component["sha256"]
