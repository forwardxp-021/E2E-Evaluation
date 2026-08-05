import numpy as np

from tools.stage7_m6_3_simulation_power_analysis import (
    EmpiricalPairedGenerator,
    choose_targets,
    fast_paired_permutation_p,
    paired_kernel_quadratic,
    parse_numeric_grid,
    wilson_interval,
)
from tools.stage7_m6_scenario_conditioned_bdd import (
    biased_mmd2_from_kernel,
    exact_median_bandwidth,
    rbf_kernel,
)


def test_fast_quadratic_matches_frozen_biased_mmd():
    values_a = np.asarray([[1.0, 0.0], [0.8, 0.1], [1.2, -0.1]])
    values_b = np.asarray([[-1.0, 0.0], [-0.8, -0.1], [-1.2, 0.1]])
    quadratic, observed = paired_kernel_quadratic(values_a, values_b)
    pooled = np.vstack([values_a, values_b])
    kernel = rbf_kernel(pooled, exact_median_bandwidth(pooled))
    expected = biased_mmd2_from_kernel(
        kernel, np.arange(3), np.arange(3, 6)
    )
    assert np.isclose(observed, expected)
    assert np.isclose(observed, np.ones(3) @ quadratic @ np.ones(3))


def test_fast_permutation_detects_large_consistent_shift():
    values_a = np.repeat([[2.0, 0.0]], 30, axis=0)
    values_b = np.repeat([[-2.0, 0.0]], 30, axis=0)
    p_value = fast_paired_permutation_p(
        values_a,
        values_b,
        permutations=999,
        rng=np.random.default_rng(7),
    )
    assert p_value <= 0.01


def test_empirical_generator_zero_scale_removes_mean_shift_in_expectation():
    values_a = np.asarray([[2.0], [3.0], [4.0]])
    values_b = np.asarray([[0.0], [0.0], [0.0]])
    generator = EmpiricalPairedGenerator(values_a, values_b)
    sampled_a, sampled_b = generator.sample(
        20000, effect_scale=0.0, rng=np.random.default_rng(5)
    )
    assert abs(float(np.mean(sampled_a - sampled_b))) < 0.03


def test_choose_targets_uses_smallest_eligible_and_attrition():
    rows = []
    for endpoint in ("overall_primary", "task_a"):
        for n_pairs, power in ((20, 0.7), (30, 0.82), (40, 0.9)):
            rows.append(
                {
                    "endpoint": endpoint,
                    "effect_scale_vs_development_pilot_mean_shift": 0.75,
                    "candidate_pairs": n_pairs,
                    "power": power,
                }
            )
    family = [
        {
            "effect_scale_vs_development_pilot_mean_shift": 0.75,
            "candidate_pairs_per_task": 30,
            "power": 0.81,
        }
    ]
    selected = choose_targets(
        rows,
        family,
        target_effect_scale=0.75,
        target_power=0.8,
        attrition_rate=0.2,
    )
    assert selected["endpoint_selections"]["overall_primary"]["candidate_pairs"] == 30
    assert selected["endpoint_selections"]["task_a"]["gross_pairs_with_attrition"] == 38
    assert (
        selected["simultaneous_all_five_task_selection"][
            "gross_total_pairs_for_five_disjoint_task_quotas"
        ]
        == 188
    )
    assert selected["design_ready"] is True


def test_wilson_interval_and_grid_parser():
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high
    assert parse_numeric_grid("30,12,20", cast=int) == [12, 20, 30]
