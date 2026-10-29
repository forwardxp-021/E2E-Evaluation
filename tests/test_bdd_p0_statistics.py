import numpy as np
import pytest
from tools.bdd_p0_analysis import bank,score,kernel_matrix,cutoff
from tools.rbr_v2_d1_statistics import mmd2


def fixture(n=30):
    return [dict(session_id=str(i),dependency_block='block'+str(i),role='D_P0_FIXTURE',task='LOW' if i<n else 'LANE') for i in range(2*n)]


def test_raw_and_conditioned_draws_disjoint_clusters():
    rows=fixture()
    for design in ['RAW_IID','ACTIVITY_Q','ACTIVITY_STRESS']:
        a,b=bank(rows,design,5,11)
        assert a.shape==b.shape==(5,20)
        for x,y in zip(a,b):assert len(set(x)|set(y))==40
    a,b=bank(rows,'ACTIVITY_Q',3,12)
    assert np.all((a<30).sum(1)==10) and np.all((b<30).sum(1)==10)
    a,b=bank(rows,'ACTIVITY_STRESS',3,12)
    assert np.all((a<30).sum(1)==10) and np.all((b<30).sum(1)==5)


def test_frozen_full_task_capacity_not_relaxed():
    rows=[dict(session_id=str(i),dependency_block=str(i),role='D_P0_FIXTURE',task=t) for i,t in enumerate(['LOW']*28+['LANE']*31+['JUNCTION']*9)]
    with pytest.raises(ValueError,match='INSUFFICIENT_UNIQUE_SUPPORT'):bank(rows,'FULL_TASK_Q',2,3)


def test_no_duplicate_cluster_or_future_role():
    rows=fixture();rows[0]['dependency_block']=rows[1]['dependency_block']
    with pytest.raises(ValueError,match='Repeated'):bank(rows,'RAW_IID',2,3)
    rows=fixture();rows[0]['role']='V_AA'
    with pytest.raises(PermissionError):bank(rows,'RAW_IID',2,3)


def test_cached_unbiased_mmd_matches_independent_direct_and_retains_negative():
    rng=np.random.RandomState(31);x=rng.normal(size=(60,5));fit=dict(mean=[0]*5,scale=[1]*5,width=5,sigma=2.)
    K=kernel_matrix(x,fit);a,b=bank(fixture(),'RAW_IID',12,4);s=score(K,(a,b))
    np.testing.assert_allclose(s,[mmd2(x[i],x[j],2.) for i,j in zip(a,b)],atol=1e-14,rtol=0)
    assert np.any(s<0)
    values=np.arange(1999.)
    assert cutoff(values)==1899
