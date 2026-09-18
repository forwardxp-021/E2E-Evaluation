import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/s2r_engineering_closure"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_schema_hashes_and_zero_run_state():
    manifest = json.loads((OUT / "S2R_Engineering_Closure_Manifest_v1.json").read_text())
    assert manifest["main_status"] == "S2R_ENGINEERING_CLOSURE_READY_FOR_OWNER_REVIEW"
    assert manifest["contracts"] == {
        "production_execution_binding": "PASS",
        "full_arm_reset": "PASS",
        "precontext_identity": "PASS",
    }
    assert all(value == 0 for value in manifest["zero_run_counters"].values())
    assert not any(manifest["authorization"].values())
    assert manifest["final_primary_roster_materialized"] is False
    assert manifest["sample_size_changed"] is False
    for relative, expected in manifest["artifacts"].items():
        assert file_sha(ROOT / relative) == expected
    for relative, expected in manifest["code"].items():
        assert file_sha(ROOT / relative) == expected


def test_census_capacity_and_required_static_fields():
    with (OUT / "S2R_Static_Eligibility_Census_v1.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 248
    statuses = Counter(row["static_eligibility_status"] for row in rows)
    assert statuses == {"STATIC_ELIGIBLE": 214, "STATIC_INELIGIBLE": 31, "STATIC_AMBIGUOUS": 3}
    eligible = [row for row in rows if row["static_eligibility_status"] == "STATIC_ELIGIBLE"]
    by_class = Counter(row["exposure_class"] for row in eligible)
    assert by_class == {
        "E0_METADATA_ONLY": 1,
        "E1_UNRELATED_HISTORICAL_USE": 147,
        "E2_BENCHMARK_ENGINEERING": 62,
        "E5_UNTOUCHED_CONFIRMATORY": 4,
    }
    for row in eligible:
        assert float(row["execution_initial_speed_mps"]) >= 3.61
        assert row["initial_speed_ge_3_61"] == "true"
        assert row["route_reference_complete"] == "true"
        assert row["production_executor_compatible"] == "true"
        assert row["source_provenance_status"] == "COMPLETE"
        assert row["representative_source_partition"]
        assert len(row["representative_source_database_fingerprint_sha256"]) == 64
        assert len(row["native_route_identity"]) == len(row["precontext_id"]) == 64
        assert row["final_primary_roster_member"] == "false"


def test_capacity_manifest_matches_census():
    manifest = json.loads((OUT / "S2R_Engineering_Closure_Manifest_v1.json").read_text())
    capacity = manifest["capacity"]
    assert capacity["total_sessions"] == 248
    assert capacity["raw_v_candidates"] == 217
    assert capacity["static_certified_v_pool"] == 214
    assert capacity["claim_a_static_candidate_capacity"] == 152
    assert capacity["claim_b_static_v_capacity"] == 214
    assert capacity["strict_c_static_capacity"] == 4
    assert capacity["ambiguous_candidate_sessions"] == 3
