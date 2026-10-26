import numpy as np
from tools.rbr_v2_d4_randomization import direct_stat, orbit, all_signs, pvalue, kernel


def test_orbit_independent_direct_formula():
    rng = np.random.RandomState(619)
    for m in [2, 4, 8, 20]:
        x, y = rng.normal(size=(2, m, 5))
        signs = rng.choice([-1., 1.], (80, m))
        got, _ = orbit(x, y, 2., signs)
        expected = [direct_stat(np.where(s[:, None] == 1, x, y), np.where(s[:, None] == 1, y, x), 2.) for s in signs]
        np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)


def test_exhaustive_randomization_size_with_ties():
    rng = np.random.RandomState(620)
    for scale in [1., 20., 1e8]:
        x, y = rng.normal(size=(2, 8, 4))*scale
        signs = all_signs(8)
        scores, _ = orbit(x, y, 2., signs)
        p = np.array([(scores >= t-1e-12).mean() for t in scores])
        assert (p <= .05).mean() <= .05


def test_identical_pair_negative_score_but_no_alarm():
    x = np.arange(40.).reshape(20, 2)
    p, value, _ = pvalue(x, x, 2., all_signs(4)[:, [0]*20], exact=True)
    k = kernel(x, x, 2.)
    expected = 2/20*((k.sum()-np.trace(k))/(20*19)-1)
    np.testing.assert_allclose(value, expected, atol=1e-14)
    assert value < 0 and p == 1


def test_shifted_pairs_and_global_label_swap_invariance():
    x = np.random.RandomState(621).normal(size=(20, 4)); y = x+1.
    signs = np.random.RandomState(622).choice([-1., 1.], (499, 20))
    assert pvalue(x, y, 2., signs)[0] == pvalue(y, x, 2., signs)[0]
    assert pvalue(x, y, 2., signs)[0] <= .05
