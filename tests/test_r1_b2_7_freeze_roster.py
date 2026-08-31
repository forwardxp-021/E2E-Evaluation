import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools import r1_b2_7_freeze_official_smoke_roster_v2 as b27


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def test_frozen_rank_payload_is_exact_and_computed_after_eligibility_call():
    salt = "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9"
    expected = hashlib.sha256(f"{salt}|R-HLC|token-a|log-a".encode()).hexdigest()
    assert b27.rank_digest(salt, "R-HLC", "token-a", "log-a") == expected

    source = Path(b27.__file__).read_text(encoding="utf-8")
    function = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_enumerate_db_eligibility"
    )
    call_lines = {
        node.func.id: node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"evaluate_candidate", "rank_digest"}
    }
    assert call_lines["evaluate_candidate"] < call_lines["rank_digest"]
    assert "ranked_work_prefix" not in source


def test_effective_blacklist_is_additive_and_matches_token_or_log():
    audit = b27.resolve_effective_blacklist()
    prior = json.loads((R1 / "r1_official_technical_smoke_roster_v1.0.json").read_text())
    union = json.loads((R1 / "r1_official_technical_smoke_permanent_blacklist_v1.0.json").read_text())
    actual = {(row["scenario_token"], row["log_id"]) for row in audit["entries"]}
    expected = {
        (row["scenario_token"].lower(), row["log_id"])
        for row in union["entries"] + prior["entries"]
    }
    assert expected <= actual
    assert audit["counts"]["unique_scenario_tokens"] >= 40
    assert audit["counts"]["unique_logs"] >= 40
    assert audit["match_rule"] == "EXCLUDE_IF_SCENARIO_TOKEN_OR_LOG_ID_MATCHES"


def _ranked(family, index, log_id=None):
    token = f"{0 if family == 'R-HLC' else 1:x}{index:031x}"
    log_id = log_id or f"log-{family}-{index:02d}"
    entry = {"family": family, "scenario_token": token, "log_id": log_id}
    return {"rank_key": (f"{index:064x}", token, log_id, index), "entry": entry, "audit": dict(entry)}


def test_unique_freeze_skips_cross_family_log_collision_without_manual_choice():
    hlc = [_ranked("R-HLC", index) for index in range(12)]
    tsb = [_ranked("R-TSB", index, "log-R-HLC-00" if index == 0 else None) for index in range(13)]
    enumerations = {
        "R-HLC": {"ranked_best_per_log": hlc, "eligible_count": len(hlc)},
        "R-TSB": {"ranked_best_per_log": tsb, "eligible_count": len(tsb)},
    }
    selected, audits = b27.freeze_unique_roster(enumerations)
    entries = selected["R-HLC"] + selected["R-TSB"]
    assert len(selected["R-HLC"]) == len(selected["R-TSB"]) == 12
    assert len(audits["R-HLC"]) == len(audits["R-TSB"]) == 12
    assert len({row["scenario_token"] for row in entries}) == 24
    assert len({row["log_id"] for row in entries}) == 24
    assert selected["R-TSB"][0]["log_id"] != "log-R-HLC-00"


def test_schedule_has_exact_pair_arms_and_zero_budget_rejects_run_49():
    entries = []
    for family, arms in (("R-HLC", ["HLC_BASELINE", "HLC_TREATMENT"]), ("R-TSB", ["TSB_BASELINE", "TSB_TREATMENT"])):
        for index in range(12):
            entries.append({"family": family, "scenario_token": f"{family}-{index}", "log_id": f"{family}-log-{index}", "arms": arms})
    schedule = b27.build_schedule(entries, {"binding": "sha"}, "roster-sha")
    assert len(schedule["runs"]) == len({row["run_id"] for row in schedule["runs"]}) == 48
    assert len({row["pair_id"] for row in schedule["runs"]}) == 24
    assert schedule["audit"] == {
        "unique_run_ids": 48,
        "unique_pair_ids": 24,
        "duplicate_arms": 0,
        "missing_arms": 0,
        "run_49_pre_call_claim": "REJECTED_ZERO_AUTHORIZED_BUDGET",
    }
    assert schedule["OFFICIAL_SMOKE_AUTHORIZED"] is False
    assert schedule["NEW_RUN_BUDGET"] == 0


def test_artifact_hash_matches_exact_written_bytes(tmp_path):
    payload = {"schema_version": "test", "value": [1, 2, 3]}
    path = tmp_path / "artifact.json"
    b27.write_new(path, payload)
    assert b27.artifact_sha256(payload) == b27.sha256_file(path)
    with pytest.raises(FileExistsError):
        b27.write_new(path, payload)


def test_selector_has_no_simulation_call_or_outcome_input_path():
    source = Path(b27.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any("run_simulation" in name for name in imported)
    assert "run_simulation" not in called_attributes | called_names
    assert "compute_trajectory" not in called_attributes | called_names
    assert "FROM scenario_tag" in source
    assert "JOIN scene s ON s.token=lp.scene_token" in source


def test_historical_blacklisted_replay_sampling_is_complete_and_nonextrapolated():
    roster = json.loads((R1 / "r1_official_technical_smoke_roster_v1.0.json").read_text())
    candidate = dict(roster["entries"][0])
    candidate["timestamp"] = candidate["scenario_anchor_timestamp_us"]
    initial, _ = b27._official_initial(candidate)
    replay = b27._sampled_replay(candidate, initial)
    assert len(replay["tokens"]) == 80
    assert len(set(replay["tokens"])) == 80
    assert len(replay["timestamps_s"]) == 80
    assert replay["timestamps_s"][0] == 0.0
    assert replay["timestamps_s"][-1] >= 7.9
    assert all(b > a for a, b in zip(replay["timestamps_s"], replay["timestamps_s"][1:]))


@pytest.mark.parametrize(
    "report, expected",
    [
        ({"max_abs_lateral_accel_mps2": 6.0, "max_abs_yaw_rate_radps": 1.0, "max_abs_curvature_inv_m": 0.5}, True),
        ({"max_abs_lateral_accel_mps2": 6.000001, "max_abs_yaw_rate_radps": 1.0, "max_abs_curvature_inv_m": 0.5}, False),
    ],
)
def test_frozen_hlc_engineering_boundaries(report, expected):
    assert b27._engineering_pass(report) is expected
