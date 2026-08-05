import json

import numpy as np
import pandas as pd
import pytest

from tools.stage7_m6_2_locked_task_bdd import (
    audit_locked_disjointness,
    build_pretreatment_task_masks,
    paired_randomization_test,
    planner_fingerprints,
    validate_frozen_power_justification,
)


def make_metadata(tokens=("s0", "s1"), logs=("l0", "l1")):
    rows = []
    for scenario_index, (token, log_name) in enumerate(zip(tokens, logs)):
        scenario_type = (
            "changing_lane_to_left"
            if scenario_index == 0
            else "following_lane_with_lead"
        )
        for planner_name, parameters in (
            ("planner_b", {"headway": 2.0}),
            ("planner_a", {"headway": 1.0}),
        ):
            rows.append(
                {
                    "global_row": len(rows),
                    "scenario_index": scenario_index,
                    "scenario_token": token,
                    "log_name": log_name,
                    "scenario_type": scenario_type,
                    "planner_name": planner_name,
                    "parameters_json": json.dumps(parameters),
                    "valid_timestep_count": 150,
                }
            )
    return pd.DataFrame(rows)


def test_pretreatment_task_masks_use_equal_scenario_type():
    metadata = make_metadata()
    pairs = np.asarray([[1, 0], [3, 2]], dtype=np.int64)
    masks, table = build_pretreatment_task_masks(metadata, pairs)
    assert masks["lane_change"].tolist() == [True, False]
    assert masks["following_interaction"].tolist() == [False, True]
    assert int(table.loc[table["task"] == "unmapped_scenario_type", "n_pairs"].iloc[0]) == 0


def test_pretreatment_task_masks_reject_post_pair_type_conflict():
    metadata = make_metadata()
    metadata.loc[1, "scenario_type"] = "stationary_in_traffic"
    with pytest.raises(ValueError, match="unequal pre-treatment scenario_type"):
        build_pretreatment_task_masks(
            metadata, np.asarray([[1, 0], [3, 2]], dtype=np.int64)
        )


def test_small_pair_randomization_is_exact_and_reproducible():
    values_a = np.asarray([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]])
    values_b = np.asarray([[-1.0, 0.0], [-1.1, -0.1], [-0.9, 0.1]])
    first, samples_first = paired_randomization_test(
        values_a,
        values_b,
        monte_carlo_repetitions=999,
        seed=1,
        progress_label="test",
    )
    second, samples_second = paired_randomization_test(
        values_a,
        values_b,
        monte_carlo_repetitions=10,
        seed=999,
        progress_label="test",
    )
    assert first["randomization_mode"] == "exact_enumeration"
    assert first["unique_label_assignments"] == 8
    assert first["p_value"] >= 1 / 8
    np.testing.assert_allclose(samples_first, samples_second)


def test_locked_disjointness_requires_new_logs_and_scenarios_but_same_planners():
    development = make_metadata()
    candidate = make_metadata(tokens=("s2", "s3"), logs=("l2", "l3"))
    audit = audit_locked_disjointness(
        development, candidate, planners=("planner_a", "planner_b")
    )
    assert audit["passed"] is True
    overlapping = make_metadata(tokens=("s0", "s3"), logs=("l2", "l3"))
    assert (
        audit_locked_disjointness(
            development, overlapping, planners=("planner_a", "planner_b")
        )["passed"]
        is False
    )


def test_locked_disjointness_rejects_changed_treatment_parameters():
    development = make_metadata()
    candidate = make_metadata(tokens=("s2", "s3"), logs=("l2", "l3"))
    candidate.loc[candidate["planner_name"] == "planner_a", "parameters_json"] = (
        json.dumps({"headway": 0.5})
    )
    audit = audit_locked_disjointness(
        development, candidate, planners=("planner_a", "planner_b")
    )
    assert audit["planner_parameters_identical_to_frozen_treatments"] is False
    assert audit["passed"] is False


def test_planner_fingerprint_rejects_inconsistent_config_within_planner():
    metadata = make_metadata()
    metadata.loc[3, "parameters_json"] = json.dumps({"headway": 0.75})
    with pytest.raises(ValueError, match="multiple parameter configurations"):
        planner_fingerprints(metadata, ("planner_a", "planner_b"))


def test_power_justification_enforces_every_frozen_task(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text('{"lock": true}', encoding="utf-8")
    import hashlib

    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    masks = {
        "task_a": np.asarray([True, True, False]),
        "task_b": np.asarray([False, True, True]),
    }
    payload = {
        "status": "FROZEN_BEFORE_LOCKED_CONFIRMATION",
        "m6_2_lock_spec_sha256": lock_hash,
        "required_complete_pairs_overall": 3,
        "required_complete_pairs_by_task": {"task_a": 2, "task_b": 2},
    }
    result = validate_frozen_power_justification(
        payload, lock_manifest_path=lock, pair_count=3, task_masks=masks
    )
    assert result["passed"] is True
    payload["required_complete_pairs_by_task"]["task_b"] = 3
    with pytest.raises(ValueError, match="sample targets not met"):
        validate_frozen_power_justification(
            payload, lock_manifest_path=lock, pair_count=3, task_masks=masks
        )
