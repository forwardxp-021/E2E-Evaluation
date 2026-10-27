import numpy as np
from tools.rbr_v2_d6_static import task_cell
from tools.rbr_v2_d6_numeric import analytic_velocity,mean_error_bound


def test_fixed_task_boundaries():
    assert task_cell(5,False)=='5to10_empty'
    assert task_cell(10,True)=='10to15_busy'
    assert task_cell(15,False) is None


def test_closed_form_no_command_decay():
    v=analytic_velocity(10.,.3,np.zeros(79));n=np.arange(80)
    np.testing.assert_allclose(v,10+.06*(1-(2/3)**n),atol=2e-15,rtol=0)


def test_analytic_discrete_filter_equation_without_vehicle_state():
    rng=np.random.RandomState(61);u=rng.uniform(-1.45,0,79);v=analytic_velocity(12,.2,u)
    a=np.r_[.2,np.diff(v)/.1]
    np.testing.assert_allclose(a[1:],(2/3)*a[:-1]+(1/3)*u,atol=4e-14,rtol=0)
    np.testing.assert_allclose(mean_error_bound(1e-6),4e-5,rtol=1e-15)
