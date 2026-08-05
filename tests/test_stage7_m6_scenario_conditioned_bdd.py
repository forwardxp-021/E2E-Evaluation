import numpy as np
import pandas as pd
import pytest

from tools.stage7_m6_scenario_conditioned_bdd import (
    build_pair_quality_audit,
    permutation_bdd,
    scenario_residualize,
    validate_and_build_pairs,
)


def _metadata(n_pairs: int) -> pd.DataFrame:
    rows = []
    for pair in range(n_pairs):
        rows.extend(
            [
                {
                    "global_row": 2 * pair,
                    "scenario_index": pair,
                    "scenario_token": f"scenario-{pair}",
                    "planner_name": "assertive",
                    "valid_timestep_count": 150,
                },
                {
                    "global_row": 2 * pair + 1,
                    "scenario_index": pair,
                    "scenario_token": f"scenario-{pair}",
                    "planner_name": "conservative",
                    "valid_timestep_count": 150,
                },
            ]
        )
    return pd.DataFrame(rows)


def _pair_rows(n_pairs: int):
    return [
        {
            "scenario": f"scenario-{pair}",
            "row_A": str(2 * pair),
            "row_B": str(2 * pair + 1),
        }
        for pair in range(n_pairs)
    ]


def test_pair_validation_requires_exact_scenario_planner_and_row_coverage() -> None:
    pairs, scenarios = validate_and_build_pairs(
        _metadata(3),
        _pair_rows(3),
        6,
        planner_a="assertive",
        planner_b="conservative",
    )
    assert pairs.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert scenarios == ["scenario-0", "scenario-1", "scenario-2"]


def test_pair_validation_rejects_planner_mismatch() -> None:
    metadata = _metadata(2)
    metadata.loc[0, "planner_name"] = "conservative"
    with pytest.raises(ValueError, match="planner mismatch"):
        validate_and_build_pairs(
            metadata,
            _pair_rows(2),
            4,
            planner_a="assertive",
            planner_b="conservative",
        )


def test_pair_validation_rejects_unequal_valid_horizon() -> None:
    metadata = _metadata(2)
    metadata.loc[0, "valid_timestep_count"] = 149
    with pytest.raises(ValueError, match="unequal valid horizon"):
        validate_and_build_pairs(
            metadata,
            _pair_rows(2),
            4,
            planner_a="assertive",
            planner_b="conservative",
        )


def test_scenario_residualization_cancels_pair_midpoints() -> None:
    values = np.asarray([[12.0, 2.0], [10.0, 0.0], [102.0, 4.0], [100.0, 2.0]])
    pairs = np.asarray([[0, 1], [2, 3]])
    residual_a, residual_b = scenario_residualize(values, pairs)
    assert np.allclose(residual_a + residual_b, 0.0)
    assert np.allclose(residual_a, [[1.0, 1.0], [1.0, 1.0]])


def test_paired_bdd_detects_consistent_shift_under_large_scenario_offsets() -> None:
    rng = np.random.default_rng(7)
    scenario_offsets = rng.normal(0.0, 100.0, size=(16, 3))
    shift = np.asarray([1.0, 0.5, -0.25])
    values_a = scenario_offsets + shift
    values_b = scenario_offsets - shift
    paired, _ = permutation_bdd(
        values_a,
        values_b,
        repetitions=1000,
        seed=11,
        paired_swap=True,
        progress_label="test paired",
    )
    pooled, _ = permutation_bdd(
        values_a,
        values_b,
        repetitions=1000,
        seed=11,
        paired_swap=False,
        progress_label="test pooled",
    )
    assert paired["p_value"] < 0.01
    assert paired["exceedance_count"] < 10
    assert paired["bandwidth_fixed_across_observed_and_all_permutations"] is True
    assert paired["subsampling"] == "none"
    assert pooled["p_value"] > 0.1


def test_pair_quality_audit_reports_horizon_and_fallback_balance() -> None:
    metadata = _metadata(2)
    pairs, _ = validate_and_build_pairs(
        metadata,
        _pair_rows(2),
        4,
        planner_a="assertive",
        planner_b="conservative",
    )
    row_quality = metadata[
        ["global_row", "scenario_index", "planner_name"]
    ].copy()
    row_quality["fallback_rate"] = [0.01, 0.02, 0.00, 0.03]
    row_quality["ambiguous_frame_rate"] = [0.00, 0.01, 0.02, 0.00]
    row_quality["quality_tier"] = ["A", "A", "A", "B"]
    pair_quality = pd.DataFrame(
        [
            {
                "scenario_index": 0,
                "pair_quality_tier": "A",
                "tier_a_pair_eligible": True,
                "tier_b_inclusive_pair_eligible": True,
            },
            {
                "scenario_index": 1,
                "pair_quality_tier": "B",
                "tier_a_pair_eligible": False,
                "tier_b_inclusive_pair_eligible": True,
            },
        ]
    )
    embedding = np.arange(12, dtype=float).reshape(4, 3)
    audit, summary = build_pair_quality_audit(
        metadata,
        _pair_rows(2),
        pairs,
        embedding,
        row_quality=row_quality,
        pair_quality=pair_quality,
    )
    assert summary["complete_pairs"] == 2
    assert summary["unequal_valid_horizon_pairs"] == 0
    assert summary["tier_a_pairs"] == 1
    assert summary["tier_b_inclusive_pairs"] == 2
    assert np.isclose(audit.loc[0, "fallback_rate_abs_delta"], 0.01)
