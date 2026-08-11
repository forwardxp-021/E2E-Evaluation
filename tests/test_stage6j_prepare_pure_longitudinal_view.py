from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools import stage6j_prepare_pure_longitudinal_view as prepare


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_planner_metadata_signature_is_row_order_invariant() -> None:
    rows = [
        {"planner_id": "0", "planner_name": "assertive", "parameters_json": '{"x":1}'},
        {"planner_id": "1", "planner_name": "conservative", "parameters_json": '{"x":2}'},
    ]
    assert prepare.planner_metadata_signature(rows) == prepare.planner_metadata_signature(
        list(reversed(rows))
    )


def test_expected_pure_longitudinal_composition_is_frozen() -> None:
    assert sum(prepare.EXPECTED_TASK_COUNTS.values()) == 183
    assert prepare.EXPECTED_TASK_COUNTS == {
        "following_interaction": 60,
        "longitudinal_high_motion": 56,
        "stop_go_control": 67,
    }
    assert prepare.EXPECTED_PLANNERS == [
        "pdm_closed_assertive_longitudinal_v1",
        "pdm_closed_conservative_longitudinal_v1",
    ]


def test_incomplete_batch_is_rejected_before_output_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    locked = tmp_path / "locked.csv"
    batch = tmp_path / "batch.json"
    state = tmp_path / "state.json"
    status = tmp_path / "status.csv"
    freeze.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS",
                "embedding_or_bdd_read": False,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        locked,
        [
            {
                "collection_order": 1,
                "task": "following_interaction",
                "source_task": "following_interaction",
                "scenario_type": "near_long_vehicle",
                "log_name": "log",
                "scenario_token": "token",
                "db_file": "log.db",
            }
        ],
    )
    batch.write_text(
        json.dumps(
            {
                "schema_version": "stage6j_pure_longitudinal_batch_v1",
                "planners": prepare.EXPECTED_PLANNERS,
                "planner_fingerprints": {"frozen": "fingerprint"},
                "locked_scenarios_sha256": prepare.sha256_file(locked),
                "freeze_manifest_sha256": prepare.sha256_file(freeze),
                "full_embedding_or_bdd_read": False,
            }
        ),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "counts": {"succeeded": 182, "failed": 0, "pending": 1},
            }
        ),
        encoding="utf-8",
    )
    write_csv(status, [{"collection_order": 1, "status": "PENDING"}])
    monkeypatch.setattr(
        prepare,
        "current_planner_fingerprints",
        lambda planners: {"frozen": "fingerprint"},
    )
    with pytest.raises(ValueError, match="batch is not complete"):
        prepare.validate_locked_sources(freeze, locked, batch, state, status)
