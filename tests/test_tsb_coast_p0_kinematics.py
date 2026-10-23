"""Physical invariants, independent quadrature, and invalid-domain guards."""
import math

import pytest

from tools.tsb_coast_p0_kinematics import (
    analysis, coast_reference_valid, designs, metrics, moments, speed_at,
)


def test_equal_impulse_moment_and_four_aggregate_designs():
    for name in ('C1_SYMMETRIC_EQUAL_PEAK', 'C2_LONGER_COAST', 'C1_MILD_DRAG_EQUAL_MOMENTS'):
        arms = designs()[name]
        for v0 in (4., 5., 10., 15.):
            b, t = (metrics(arms[x], v0) for x in ('baseline', 'treatment'))
            for key in ('signed_impulse_mps', 'signed_first_moment_m', 'straight_path_length_m',
                        'continuous_mean_speed_mps', 'continuous_mean_abs_accel_mps2', 'terminal_speed_mps'):
                assert t[key] == pytest.approx(b[key], abs=1e-11)
        assert coast_reference_valid(arms['treatment'])


def test_old_and_naive_removal_have_expected_net_speed_loss():
    d = analysis()['designs']
    assert d['OLD_FROZEN']['analytic_metrics_v0_10mps_T_7p9s']['treatment']['signed_impulse_mps'] == pytest.approx(-2.5)
    assert d['NAIVE_REMOVE_RECOVERY']['analytic_metrics_v0_10mps_T_7p9s']['treatment']['signed_impulse_mps'] == pytest.approx(-4.32)
    assert not d['OLD_FROZEN']['treatment_reference_nonpositive']


def test_matching_impulse_without_matching_centroid_leaves_path_difference():
    d = analysis()['designs']['C1_KEEP_OLD_BASELINE_TIMING']['treatment_minus_baseline']
    assert d['signed_impulse_mps'] == pytest.approx(0.)
    assert d['straight_path_length_m'] == pytest.approx(2.61*.65)


def test_integrals_agree_with_independent_midpoint_quadrature():
    # Piecewise velocity is affine: midpoint integration split at all knots is exact.
    for arms in designs().values():
        for segments in arms.values():
            knots = sorted({0., 7.9} | {x for s, e, _ in segments for x in (s, e)})
            independent_distance = sum((r-l)*speed_at(segments, (l+r)/2) for l, r in zip(knots, knots[1:]))
            assert metrics(segments)['straight_path_length_m'] == pytest.approx(independent_distance)


@pytest.mark.parametrize('segments', [[(1., 1., -1.)], [(1., 3., -1.), (2., 4., -1.)],
                                     [(1., 9., -1.)], [(1., 2., math.nan)]])
def test_invalid_reference_domain_rejected(segments):
    with pytest.raises(ValueError):
        moments(segments)


def test_low_speed_is_not_silently_clipped():
    with pytest.raises(ValueError, match='zero speed'):
        metrics(designs()['C1_SYMMETRIC_EQUAL_PEAK']['treatment'], initial_speed=2.)
