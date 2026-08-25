from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.stage6k_freeze_preanalysis_addendum import freeze


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_real_completed_batch_before_new_dose_bdd_read(tmp_path: Path) -> None:
    result = freeze(Namespace(
        design_json=ROOT / "configs/stage6k_preanalysis_addendum.json",
        rollout_freeze_manifest=ROOT / "outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_freeze_manifest.json",
        locked_jobs_csv=ROOT / "outputs/stage6k_longitudinal_dose_freeze_v1/stage6k_locked_jobs.csv",
        batch_manifest=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_manifest.json",
        batch_state=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_state.json",
        batch_status_csv=ROOT / "outputs/stage6k_longitudinal_dose_batch_v1/batch_scenario_status.csv",
        output_dir=tmp_path / "freeze",
        overwrite=False,
    ))
    assert result["status"] == "FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ"
    assert result["new_dose_embedding_or_bdd_read"] is False
    assert result["rollout_audit"]["dose_counts"] == {"dose25": 183, "dose50": 183, "dose75": 183}
    assert result["rollout_audit"]["retry_orders"] == [391]
    assert result["analysis_specification"]["primary_overall_dose_family"]["correction"] == "Holm"
    assert result["analysis_specification"]["secondary_task_dose_family"]["hypothesis_count"] == 12
