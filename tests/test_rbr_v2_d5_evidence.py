import numpy as np
from tools.rbr_v2_d5_evidence import source_example
from tools.rbr_v2_d4_randomization import direct_stat


def test_unpaired_statistic_source_order_and_group_swap():
    rng=np.random.RandomState(15);x,y=rng.normal(size=(2,20,3))
    np.testing.assert_allclose(direct_stat(x,y,1),direct_stat(y[::-1],x[::-1],1),atol=1e-14)


def test_exact_mixture_shift_is_not_marginal_null():
    # No conditional behavior change: Z=C. Different task weights give distinct marginals.
    K=np.array([[1.,np.exp(-4.5)],[np.exp(-4.5),1.]])
    q0=np.array([.9,.1]);q1=np.array([.1,.9]);difference=q0-q1
    assert float(difference@K@difference)>1
    assert float((q0-q0)@K@(q0-q0))==0


def test_saturation_bound_for_all_alarm_tables():
    rng=np.random.RandomState(16)
    for _ in range(100):
        table=rng.dirichlet(np.ones(4)).reshape(2,2) # rows RBR alarm, cols H alarm
        delta=table[1,:].sum()-table[:,1].sum()
        assert delta<=table[:,0].sum()+1e-14
