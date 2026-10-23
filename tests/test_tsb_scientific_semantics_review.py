import ast
from pathlib import Path
import numpy as np
import pytest
from tools import tsb_scientific_semantics_review as review


def test_positive_duration_uses_physical_crossings():
    t=np.array([0.,1.,2.,4.]);a=np.array([-1.,1.,1.,-1.])
    r=review.component_bounds(t,a,1,0.,False)
    assert r['crossing_start_s']==.5
    assert r['crossing_end_s']==3.
    assert r['crossing_duration_s']==2.5
    assert r['sample_span_s']==1.
    assert not r['edge_censored']


def test_command_anchored_lobe_does_not_relabel_initial_transient():
    t=np.arange(80)*.1;a=np.zeros(80)
    a[0]=-5.;a[11:17]=[-1.,-1.3,-1.2,-1.1,-1.,-.9]
    peak,r=review.brake_lobe(t,a,1.1,2.)
    assert peak==12
    assert r['first_index']==11 and r['last_index']==16
    assert r['crossing_duration_s']>r['sample_span_s']


def test_median3_gradient_preserves_constant_acceleration_on_irregular_time():
    t=np.arange(80)*.1;t[1::2]+=.01
    _,_,sm,a=review.describe(t,5.+.5*t)
    np.testing.assert_allclose(a,.5,atol=1e-12)
    np.testing.assert_array_equal(sm,5.+.5*t)


def test_invalid_trace_and_wrong_threshold_fail_closed():
    t=np.arange(80)*.1
    with pytest.raises(ValueError,match='80 finite states'):review.describe(t[:79],np.ones(79))
    t[4]=t[3]
    with pytest.raises(ValueError,match='increasing'):review.describe(t,np.ones(80))
    with pytest.raises(ValueError,match='threshold'):review.component_bounds(np.arange(4),np.ones(4),1,-.8,True)


def test_no_scientific_adjudicator_or_detector_invocation():
    tree=ast.parse(Path(review.__file__).read_text())
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    forbidden={'calculate_tsb_option_a_v2_timestamp_aware','qualify_tsb_pair','evaluate_frozen_pair',
               'handcrafted_h','ego_kinematic_features','mmd','train','SimulationRunner'}
    assert not calls&forbidden
