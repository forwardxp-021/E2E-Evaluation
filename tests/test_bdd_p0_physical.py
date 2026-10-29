import numpy as np
from tools.rbr_v2_canonical83 import resample,canonical
from tools.bdd_p0_data import block


def test_logged_world_clock_to_canonical_motion_oracle():
    # A known northbound world trajectory becomes positive longitudinal progress.
    t=np.arange(161,dtype=np.int64)*50000;seconds=t/1e6
    raw=np.column_stack([np.zeros(len(t)),5*seconds,np.zeros(len(t)),np.full(len(t),5.),np.full(len(t),np.pi/2),np.ones(len(t))])
    ego=resample(raw,t,0);sample=canonical(ego,np.zeros((80,5,6)),np.full((80,5),'',dtype='U64'),np.zeros((80,5),bool))
    np.testing.assert_allclose(sample['x'][:,0],np.arange(80)*.5,atol=1e-6)
    np.testing.assert_allclose(sample['x'][:,2],5.,atol=1e-6)
    assert sample['ego_valid'].all() and not sample['channel_valid'][:,8:].any()


def test_raw_gaps_are_not_bridged_and_source_day_is_not_new_session_count():
    t=np.array([0,50000,400000],np.int64);raw=np.tile([0.,0.,0.,0.,0.,1.],(3,1));out=resample(raw,t,0)
    assert out[0,5]==1 and out[1,5]==0 and out[2,5]==0
    assert block('2021.07.12.08.00.00_veh-37')==block('2021.07.12.19.00.00_veh-37')
