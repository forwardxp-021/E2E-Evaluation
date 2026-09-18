import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import b1_native_route_precheck as route_precheck
from tools import b1_offline_metric_finalize as offline
from tools import b1_tsb_qualification_executor as executor


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "docs/stageR/b1_tsb_qualification/B1_TSB_Qualification_Arm_Specs_v1.json"
B1_OUTPUT = ROOT / "outputs/stageR/b1_tsb_qualification_v1"


def test_official_metric_finalize_is_reproducible(tmp_path: Path):
    from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback

    source = tmp_path / "source"
    source.mkdir()
    source_file = next((B1_OUTPUT / "B1-TSB-01-BASELINE/raw/metrics").glob("*.pickle.temp"))
    shutil.copy2(source_file, source / source_file.name)
    online_dir = tmp_path / "left"
    MetricFileCallback(str(online_dir), [str(source)], delete_scenario_metric_files=False).on_run_simulation_end()
    left_hashes = {path.stem: offline.sha256_file(path) for path in online_dir.glob("*.parquet")}
    right = offline.official_finalize(source, tmp_path / "right")
    assert left_hashes == right["output_sha256"]
    assert offline.REQUIRED_SAFETY_METRICS <= set(left_hashes)


def test_existing_b1_temp_replays_byte_identically(tmp_path: Path):
    source_file = next((B1_OUTPUT / "B1-TSB-01-BASELINE/raw/metrics").glob("*.pickle.temp"))
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(source_file, source / source_file.name)
    left = offline.official_finalize(source, tmp_path / "left")
    right = offline.official_finalize(source, tmp_path / "right")
    assert len(left["output_sha256"]) == 16
    assert left["output_sha256"] == right["output_sha256"]


def test_lifecycle_state_machine_requires_exact_order():
    states = ()
    for expected in executor.LIFECYCLE_ORDER:
        states = executor.advance_lifecycle(states, expected)
    assert states == executor.LIFECYCLE_ORDER
    with pytest.raises(executor.B1ExecutionError, match="LIFECYCLE_ORDER_VIOLATION"):
        executor.advance_lifecycle((), "RECORDER_COMPLETE")


def test_serializer_gate_requires_one_nonempty_official_log(tmp_path: Path):
    arm = SimpleNamespace(run_root=tmp_path)
    with pytest.raises(executor.B1ExecutionError, match="SERIALIZER_OUTPUT_INVALID"):
        executor.validate_serializer_complete(arm)
    output = tmp_path / "raw/simulation_log/planner/type/log/scenario"
    output.mkdir(parents=True)
    serialized = output / "scenario.msgpack.xz"
    serialized.write_bytes(b"official-fixture")
    assert executor.validate_serializer_complete(arm) == serialized


def test_retry_contract_rejects_scientific_rescue_and_requires_owner():
    assert executor.RETRY_CONTRACT["maximum_total_attempts_per_arm"] == 2
    with pytest.raises(executor.B1ExecutionError, match="OWNER_AUTHORIZATION_REQUIRED"):
        executor.validate_retry_request(2, "INFRASTRUCTURE_FAILURE", False)
    with pytest.raises(executor.B1ExecutionError, match="INELIGIBLE"):
        executor.validate_retry_request(2, "SCIENTIFIC_FAIL", True)
    executor.validate_retry_request(2, "INFRASTRUCTURE_FAILURE", True)


def test_b1_18_is_predetected_by_exact_production_route_builder(tmp_path: Path):
    result = route_precheck.precheck_specs(SPECS)
    by_pair = {row["pair_id"]: row for row in result["pairs"]}
    assert result["pair_count"] == 20
    assert result["arm_count"] == 40
    assert by_pair["B1-TSB-18"]["route_precheck_status"] == "INCOMPATIBLE"
    assert by_pair["B1-TSB-18"]["missing_successor_segment"] == "19339"
    assert "no native outgoing successor into 19339" in by_pair["B1-TSB-18"]["failure_code"]
    assert all(row["production_equivalence_version"] == "build_native_route_reference_v1_1" for row in result["pairs"])


def test_b1_artifact_hash_binding_after_offline_finalize():
    metric_dir = B1_OUTPUT / "B1-TSB-01-BASELINE/raw/metrics"
    if not (metric_dir / "no_ego_at_fault_collisions.parquet").exists():
        pytest.skip("offline recovery command has not been run")
    audit = offline.inspect_metric_payload(metric_dir)
    assert len(audit["input_sha256"]) == 64
    assert all(len(offline.sha256_file(path)) == 64 for path in metric_dir.glob("*.parquet"))


def test_remediation_manifest_retry_schema_when_present():
    manifest = ROOT / "docs/stageR/b1_infrastructure_remediation/B1_IR_Manifest_v1.json"
    if not manifest.exists():
        pytest.skip("remediation manifest is generated after evidence collection")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["retry_contract"] == executor.RETRY_CONTRACT
    assert payload["zero_run_proof"]["RUNNER_RUN"] == 0
