from tools.r1_b2_8_r2_official_launch_fail_closed_audit import run_audit


def test_frozen_schedule_fails_closed_on_official_scenario_resolution() -> None:
    audit = run_audit()

    assert audit["successful"] == 40
    assert audit["failed"] == 8
    failed = [row for row in audit["runs"] if row["official_scenario_resolution"] == "FAIL_CLOSED"]
    assert len(failed) == 8
    assert {row["scenario_token"] for row in failed} == {
        "a6e0468e028357de",
        "0198af1831f65977",
        "cf56ddebd44f5372",
        "0f67192c7dd45664",
    }
    assert all(row["full_hydra_composition"] == "NOT_EXECUTED_FAIL_CLOSED" for row in audit["runs"])
    assert all(row["simulation_runner_construction"] == "NOT_EXECUTED_FAIL_CLOSED" for row in audit["runs"])
