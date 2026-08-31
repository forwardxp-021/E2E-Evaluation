import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def test_r3_zero_run_launch_freeze_is_complete() -> None:
    launch = json.loads((R1 / "r1_b2_8_r3_official_launch_manifest_v1.0.json").read_text())
    roster = json.loads((R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json").read_text())
    assert roster["counts"] == {"R-HLC": 12, "R-TSB": 12, "total": 24, "unique_scenario_tokens": 24, "unique_logs": 24}
    assert launch["status"] == "48_OF_48_READY_TO_CALL_SIMULATION_RUN"
    assert len(launch["runs"]) == 48
    assert all(row["exact_resolution"] == 1 and row["simulation_runner_construction"] == "PASS" for row in launch["runs"])
    assert launch["simulation_started"] is False and launch["official_runs"] == 0 and launch["consumed_budget"] == 0
    assert launch["ledger_dry_run"]["claim_49"] == "HARD_FAIL_BEFORE_SIMULATOR_START"
