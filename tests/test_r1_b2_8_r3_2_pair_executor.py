import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.r1_b2_8_r3_1_official_safety_adapter import MetricCanonicalizationError, adapt_official_safety
from tools.r1_b2_8_r3_2_execute_frozen_48run_smoke import FrozenBudgetLedger
from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import _require_pair_binding, evaluate_frozen_pair


ROOT = Path(__file__).resolve().parents[1]
PAIR_PATH = ROOT / "docs/stageR/r1/r1_b2_8_r3_2_frozen_pair_evaluation_bindings_v1.0.json"
DRY_PATH = ROOT / "docs/stageR/r1/r1_b2_8_r3_2_orchestrator_dry_run_v1.0.json"
FINAL_PATH = ROOT / "docs/stageR/r1/r1_b2_8_r3_2_final_execution_binding_manifest_v1.1.json"


def _metric_fixture(path: Path, *, rows: int = 1, collision: object = 0, drivable: object = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"number_of_all_at_fault_collisions_stat_value": [collision] * rows}).to_parquet(path / "no_ego_at_fault_collisions.parquet")
    pd.DataFrame({"drivable_area_compliance_stat_value": [drivable] * rows}).to_parquet(path / "drivable_area_compliance.parquet")


def _trace(path: Path, family: str, binding: dict) -> None:
    trace = path / "trace"; trace.mkdir(parents=True)
    if family == "R-HLC":
        source, target = np.asarray(binding["source_reference_xy"], dtype=float), np.asarray(binding["target_reference_xy"], dtype=float)
        start, end = source[0], target[min(len(target) - 1, max(1, len(target) // 2))]
    else:
        start, end = np.array([0.0, 0.0]), np.array([80.0, 0.0])
    rows = []
    for index in range(80):
        xy = start + (end - start) * index / 79.0
        rows.append({"primary_measurement_source": "REALIZED_CURRENT_EGO", "iteration_index": index, "current_ego": {"time_us": 1_000_000 + index * 100_000, "rear_axle": {"x": float(xy[0]), "y": float(xy[1]), "heading": 0.0}, "speed_mps": 5.0}})
    (trace / "realized_current_ego.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_tsb_requires_no_hlc_clearance() -> None:
    binding = {"pair_id": "p", "family": "R-TSB", "baseline_context": {}, "treatment_context": {}, "pretreatment_clearance": None}
    _require_pair_binding(binding)
    binding["pretreatment_clearance"] = {"pretreatment_only": True}
    with pytest.raises(ValueError, match="TSB_HLC_CLEARANCE_MUST_BE_NONE"):
        _require_pair_binding(binding)


def test_real_format_safety_adapter_fails_closed(tmp_path: Path) -> None:
    good = tmp_path / "good"; _metric_fixture(good)
    assert adapt_official_safety(good)["canonical_payload"]["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(tmp_path / "missing")
    multi = tmp_path / "multi"; _metric_fixture(multi, rows=2)
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(multi)
    duplicate = tmp_path / "duplicate"; _metric_fixture(duplicate); _metric_fixture(duplicate / "nested")
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(duplicate)
    missing_column = tmp_path / "missing_column"; missing_column.mkdir()
    pd.DataFrame({"wrong": [0]}).to_parquet(missing_column / "no_ego_at_fault_collisions.parquet")
    pd.DataFrame({"drivable_area_compliance_stat_value": [True]}).to_parquet(missing_column / "drivable_area_compliance.parquet")
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(missing_column)
    nan = tmp_path / "nan"; _metric_fixture(nan, collision=float("nan"))
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(nan)
    bad = tmp_path / "bad"; _metric_fixture(bad, collision=-1)
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(bad)
    noninteger = tmp_path / "noninteger"; _metric_fixture(noninteger, collision=1.5)
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(noninteger)
    invalid_drivable = tmp_path / "invalid_drivable"; _metric_fixture(invalid_drivable, drivable=2)
    with pytest.raises(MetricCanonicalizationError): adapt_official_safety(invalid_drivable)


def test_all_24_frozen_bindings_enter_dispatcher_with_synthetic_realized_traces(tmp_path: Path) -> None:
    pairs = json.loads(PAIR_PATH.read_text())["pairs"]
    for index, binding in enumerate(pairs):
        baseline, treatment = tmp_path / f"{index}_b", tmp_path / f"{index}_t"
        _metric_fixture(baseline); _metric_fixture(treatment); _trace(baseline, binding["family"], binding); _trace(treatment, binding["family"], binding)
        result = evaluate_frozen_pair(pair_binding=binding, baseline_run_dir=baseline, treatment_run_dir=treatment)
        assert result["dispatch_status"] == "EVALUATED_NO_POSTHOC_PAIR_DELETION"


def test_budget_48_then_49_fail_closed() -> None:
    ledger = FrozenBudgetLedger()
    for index in range(48): ledger.claim(f"r{index}")
    with pytest.raises(RuntimeError, match="CAP_48"): ledger.claim("r49")


def test_pair_identity_and_final_zero_run_closure() -> None:
    pairs, dry, final = json.loads(PAIR_PATH.read_text()), json.loads(DRY_PATH.read_text()), json.loads(FINAL_PATH.read_text())
    assert len(pairs["pairs"]) == 24 and len({row["pair_id"] for row in pairs["pairs"]}) == 24
    assert all(row["pretreatment_clearance"] is None for row in pairs["pairs"] if row["family"] == "R-TSB")
    assert dry["status"] == "EXECUTION_ORCHESTRATOR_READY" and len(dry["runs"]) == 48
    assert dry["simulation_started"] is False and dry["official_runs"] == 0 and dry["consumed_real_budget"] == 0
    assert "HARD_FAIL_BEFORE_RUNNER_RUN_CAP_48" == dry["claim_49"]
    assert final["status"] == "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION"
