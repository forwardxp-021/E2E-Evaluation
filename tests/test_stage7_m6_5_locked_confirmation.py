from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tools import stage7_m6_5_prepare_locked_confirmation as prepare
from tools import stage7_m6_5_run_locked_confirmation as confirm


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_development_disjoint_audit_passes_and_rejects_overlap(tmp_path: Path) -> None:
    metadata = tmp_path / "development.csv"
    write_csv(
        metadata,
        [
            {"scenario_token": "dev-token", "log_name": "dev-log"},
            {"scenario_token": "dev-token", "log_name": "dev-log"},
        ],
    )
    result = prepare.validate_development_disjoint(
        [{"scenario_token": "new-token", "log_name": "new-log"}], metadata
    )
    assert result["pass"] is True
    assert result["scenario_token_overlap_count"] == 0
    with pytest.raises(ValueError, match="overlap"):
        prepare.validate_development_disjoint(
            [{"scenario_token": "dev-token", "log_name": "new-log"}], metadata
        )


def test_planner_metadata_signature_is_row_order_invariant() -> None:
    rows = [
        {"planner_id": "0", "planner_name": "a", "parameters_json": '{"x":1}'},
        {"planner_id": "1", "planner_name": "b", "parameters_json": '{"x":2}'},
    ]
    assert prepare.planner_metadata_signature(rows) == prepare.planner_metadata_signature(
        list(reversed(rows))
    )


def test_confirmation_lock_validation_detects_changed_file(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("original", encoding="utf-8")
    tool = Path(confirm.__file__).resolve()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_CONFIRMATION_EMBEDDING_UNBLINDING",
                "pair_count": 310,
                "row_count": 620,
                "locked_files": {
                    "sample": {
                        "path": str(locked),
                        "sha256": confirm.sha256_file(locked),
                    },
                    "confirmation_analysis_tool": {
                        "path": str(tool),
                        "sha256": confirm.sha256_file(tool),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(lock_manifest=lock)
    assert confirm.validate_lock(args)["pair_count"] == 310
    locked.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or disappeared"):
        confirm.validate_lock(args)


def test_locked_counts_and_method_are_exact() -> None:
    assert sum(prepare.EXPECTED_TASK_COUNTS.values()) == 310
    assert set(prepare.EXPECTED_TASK_COUNTS.values()) == {60, 63, 67}
    assert confirm.PERMUTATIONS == 100000
    assert confirm.PLANNER_A == "pdm_closed_assertive_v1"
    assert confirm.PLANNER_B == "pdm_closed_conservative_v1"
    assert np.dtype(np.float32).itemsize == 4
