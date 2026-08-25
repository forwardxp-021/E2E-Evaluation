import json
from pathlib import Path

import pytest

from tools.stage7_m2b_finalize_quality_analysis import (
    CORE_ARRAYS,
    compare_core_arrays,
    validate_bdd,
)


def test_validate_bdd_enforces_paired_count() -> None:
    value = {"mmd2": 0.03, "p_value": 0.7, "n_A": 15, "n_B": 15, "embedding_dim": 64}
    row = validate_bdd("tier_a", value, expected_rows_per_planner=15, alpha=0.05)
    assert row["paired_scenarios"] == 15
    assert row["significant_at_alpha"] is False
    with pytest.raises(ValueError, match="row count mismatch"):
        validate_bdd("tier_a", value, expected_rows_per_planner=16, alpha=0.05)


def test_compare_core_arrays_detects_byte_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    rebuilt = tmp_path / "rebuilt"
    baseline.mkdir()
    rebuilt.mkdir()
    for name in CORE_ARRAYS:
        (baseline / name).write_bytes(b"same")
        (rebuilt / name).write_bytes(b"same")
    assert all(row["byte_identical"] for row in compare_core_arrays(baseline, rebuilt))
    (rebuilt / CORE_ARRAYS[0]).write_bytes(b"different")
    rows = compare_core_arrays(baseline, rebuilt)
    assert [row["byte_identical"] for row in rows].count(False) == 1
