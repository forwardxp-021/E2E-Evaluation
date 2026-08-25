from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.stage6k_prepare_longitudinal_dose_views import validate_sources


ROOT = Path(__file__).resolve().parents[1]


def test_validate_real_stage6k_sources_before_bdd_read() -> None:
    grouped, audit = validate_sources(Namespace(
        addendum_manifest=ROOT / "outputs/stage6k_preanalysis_addendum_freeze_v1/stage6k_preanalysis_addendum_manifest.json",
        rollout_freeze_manifest=ROOT / "outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_freeze_manifest.json",
        locked_jobs_csv=ROOT / "outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_locked_jobs.csv",
        batch_manifest=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_manifest.json",
        batch_state=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_state.json",
        batch_status_csv=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_scenario_status.csv",
    ))
    assert audit["pass"] is True
    assert audit["rollout_count"] == 1098
    assert {label: len(rows) for label, rows in grouped.items()} == {"dose25": 183, "dose50": 183, "dose75": 183}
    tokens = [[row["scenario_token"] for row in grouped[label]] for label in ["dose25", "dose50", "dose75"]]
    assert tokens[0] == tokens[1] == tokens[2]
