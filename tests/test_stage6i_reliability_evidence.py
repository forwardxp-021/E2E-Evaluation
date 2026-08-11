import json

import numpy as np
import pandas as pd
import pytest

from tools import stage6i_build_reliability_evidence as stage6i


def test_wilson_interval_matches_stage6h_reference():
    low, high = stage6i.wilson_interval(133, 200)
    assert low == pytest.approx(0.5970206277343806)
    assert high == pytest.approx(0.7267601903125736)


def test_primary_reliability_preserves_observed_curve_and_flags_gate_failure():
    rows = []
    for size, fpr, det in [(200, 0.08, 0.30), (250, 0.065, 0.285), (300, 0.035, 0.415), (400, 0.05, 0.665)]:
        rows.append(
            {
                "target_scenarios_per_release": size,
                "scope": "overall",
                "aa_false_positive_rate": fpr,
                "aa_false_positive_wilson95_low": max(0.0, fpr - 0.02),
                "aa_false_positive_wilson95_high": fpr + 0.04,
                "ab_detection_rate": det,
                "ab_detection_wilson95_low": det - 0.06,
                "ab_detection_wilson95_high": det + 0.06,
            }
        )
    result = stage6i.build_primary(pd.DataFrame(rows))
    assert result["ab_detection_rate"].tolist() == [0.30, 0.285, 0.415, 0.665]
    assert result["false_negative_rate"].tolist()[-1] == pytest.approx(0.335)
    assert result["aa_ab_wilson_intervals_separated"].all()
    assert not result["confidence_target_gate_pass"].any()


def test_direction_diagnostics_keeps_both_directions():
    rows = []
    for size in stage6i.EXPECTED_SAMPLE_SIZES:
        for name, count in [
            ("AB_ASSERTIVE_TO_CONSERVATIVE", 62),
            ("AB_CONSERVATIVE_TO_ASSERTIVE", 71),
        ]:
            rows.append(
                {
                    "target_scenarios_per_release": size,
                    "experiment_set": name,
                    "family": "AB_EVALUATION",
                    "scope": "overall",
                    "valid_trials": 100,
                    "exceedance_count": count,
                }
            )
    result = stage6i.build_direction_diagnostics(pd.DataFrame(rows))
    assert len(result) == 4
    assert (result["assertive_to_conservative_detection_rate"] == 0.62).all()
    assert (result["conservative_to_assertive_detection_rate"] == 0.71).all()
    assert np.allclose(result["absolute_direction_gap"], 0.09)
    assert set(result["role"]) == {"DIAGNOSTIC_ONLY_NO_DIRECTION_EQUIVALENCE_GATE"}


def _complete_synthetic_inputs():
    summary = {
        "status": "POWER_CURVE_COMPLETE",
        "config": {
            "sample_sizes_per_release": stage6i.EXPECTED_SAMPLE_SIZES,
            "log_split_strategy": "sequential_full_log_pool_v1",
            "target_detection_rate": 0.8,
            "target_false_positive_rate": 0.05,
        },
        "threshold_audit": {"all_overall_thresholds_pass": True},
        "paired_oracle": {"pair_count": 310, "overall_original_mmd2": 0.004, "overall_original_monte_carlo_p": 1e-5},
        "sufficiency": {"status": "TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS"},
    }
    pool = {
        "status": "EXPANDED_800_PAIR_EMBEDDING_POOL_READY",
        "pair_count": 800,
        "row_count": 1600,
        "cluster_count": 489,
        "all_pairs_complete": True,
        "all_embeddings_finite": True,
    }
    operating_rows = []
    detection_rows = []
    split_rows = []
    trial_rows = []
    for size in stage6i.EXPECTED_SAMPLE_SIZES:
        for scope in stage6i.EXPECTED_SCOPES:
            operating_rows.append({"target_scenarios_per_release": size, "scope": scope})
        for experiment in stage6i.EXPECTED_EXPERIMENTS:
            family = (
                "AA_CALIBRATION" if experiment.startswith("AA_CALIBRATION")
                else "AA_EVALUATION" if experiment.startswith("AA_EVALUATION")
                else "AB_EVALUATION"
            )
            for repetition in range(100):
                split_rows.append(
                    {
                        "target_scenarios_per_release": size,
                        "experiment_set": experiment,
                        "family": family,
                        "repetition": repetition,
                        "selected_scenarios_A": size,
                        "selected_scenarios_B": size,
                        "log_split_strategy": "sequential_full_log_pool_v1",
                        "log_overlap_count": 0,
                        "scenario_overlap_count": 0,
                    }
                )
                for scope in stage6i.EXPECTED_SCOPES:
                    trial_rows.append(
                        {
                            "target_scenarios_per_release": size,
                            "experiment_set": experiment,
                            "repetition": repetition,
                            "scope": scope,
                            "log_overlap_count": 0,
                            "scenario_overlap_count": 0,
                        }
                    )
            for scope in stage6i.EXPECTED_SCOPES:
                detection_rows.append(
                    {
                        "target_scenarios_per_release": size,
                        "experiment_set": experiment,
                        "family": family,
                        "scope": scope,
                    }
                )
    return (
        summary,
        pool,
        pd.DataFrame(operating_rows),
        pd.DataFrame(detection_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(trial_rows),
    )


def test_input_audit_accepts_complete_frozen_design_and_rejects_log_leakage():
    inputs = _complete_synthetic_inputs()
    audit = stage6i.audit_inputs(*inputs)
    assert audit["pass"] is True
    assert audit["split_count"] == 2400

    leaked = list(inputs)
    leaked[4] = leaked[4].copy()
    leaked[4].loc[0, "log_overlap_count"] = 1
    with pytest.raises(ValueError, match="log leakage"):
        stage6i.audit_inputs(*leaked)


def test_claim_matrix_keeps_production_and_oem_claims_unsupported():
    summary = {
        "paired_oracle": {
            "pair_count": 310,
            "overall_original_mmd2": 0.004469,
            "overall_original_monte_carlo_p": 1e-5,
        }
    }
    primary = pd.DataFrame(
        {
            "target_scenarios_per_release": stage6i.EXPECTED_SAMPLE_SIZES,
            "aa_false_positive_rate": [0.08, 0.065, 0.035, 0.05],
            "aa_false_positive_wilson95_high": [0.126, 0.108, 0.070, 0.090],
            "ab_detection_rate": [0.30, 0.285, 0.415, 0.665],
            "ab_detection_wilson95_low": [0.241, 0.227, 0.349, 0.597],
            "aa_ab_wilson_intervals_separated": [True, True, True, True],
        }
    )
    claims = stage6i.build_claim_matrix(summary, primary).set_index("claim_id")
    assert claims.loc["C1_CROSS_DOMAIN_KNOWN_STYLE_SIGNAL", "status"] == "SUPPORTED_WITHIN_PUBLIC_BENCHMARK"
    assert claims.loc["C2_UNPAIRED_DIFFERENT_LOG_DETECTION", "status"] == "SUPPORTED_WITHIN_PUBLIC_RELEASE_EMULATION"
    assert claims.loc["C3_RELIABLE_SINGLE_RELEASE_80_PERCENT", "status"] == "NOT_SUPPORTED"
    assert claims.loc["C5_REAL_OEM_FIELD_VALIDATION", "status"] == "NOT_EVALUATED"


def test_task_classification_uses_scenario_type_and_marks_lane_change_unconfirmed():
    task_types = {
        "following_interaction": ("following_lane_with_lead", 182),
        "lane_change": ("changing_lane_to_left", 71),
        "stop_go_control": ("stationary_in_traffic", 182),
        "high_motion_dynamics": ("high_magnitude_speed", 182),
        "dense_or_vulnerable_interaction": ("near_multiple_vehicles", 183),
    }
    rows = []
    task_specs = []
    token_index = 0
    for task, (scenario_type, count) in task_types.items():
        task_specs.append(
            {
                "name": task,
                "column": "scenario_type",
                "positive_values": [scenario_type],
                "timing": "pre_treatment",
            }
        )
        for _ in range(count):
            token = f"scenario-{token_index}"
            token_index += 1
            for planner in ["assertive", "conservative"]:
                rows.append(
                    {"scenario_token": token, "scenario_type": scenario_type, "planner_name": planner}
                )
    summary = {"config": {"tasks": task_specs}}
    pool = {"task_counts": {task: count for task, (_, count) in task_types.items()}}
    definitions, classifications = stage6i.build_task_classification(summary, pool, pd.DataFrame(rows))
    assert definitions["pool_pair_count"].sum() == 800
    lane = definitions.set_index("task").loc["lane_change"]
    assert lane["semantic_status"] == "SCENARIO_TYPE_SLICE_NOT_CONFIRMED_EGO_LANE_CHANGE"
    assert classifications.loc[
        classifications["task"] == "lane_change", "actual_planner_maneuver_confirmed"
    ].tolist() == [False]


def test_planner_treatment_audit_rejects_mixed_lateral_and_longitudinal_contrast():
    assertive = {
        "idm_policies.headway_time": 1.0,
        "lateral_offsets": [-1.5, 1.5],
        "source": "tuplan_garage",
        "checkpoint_required": False,
    }
    conservative = {
        "idm_policies.headway_time": 2.0,
        "lateral_offsets": [-0.5, 0.5],
        "source": "tuplan_garage",
        "checkpoint_required": False,
    }
    metadata = pd.DataFrame(
        {
            "planner_name": ["assertive", "conservative"],
            "parameters_json": [json.dumps(assertive), json.dumps(conservative)],
        }
    )
    summary = {"config": {"planners": {"assertive": "assertive", "conservative": "conservative"}}}
    audit, pure_longitudinal = stage6i.build_planner_treatment_audit(summary, metadata)
    lateral = audit.set_index("parameter").loc["lateral_offsets"]
    assert lateral["dimension"] == "lateral"
    assert not lateral["same_value"]
    assert pure_longitudinal is False
