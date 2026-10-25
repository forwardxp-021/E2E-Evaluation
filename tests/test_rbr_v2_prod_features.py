"""Optimization equivalence: masks including holes and exact geometric edge cases."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import numpy as np
from tools.rbr_v2_model import mask_plan
from tools.rbr_v2_prod_features import mask_plan_fast,projections_batch
from tools.waymo_lane_utils import LaneInfo,_build_lane_geom,find_best_lane_for_agent


class OptimizationTests(unittest.TestCase):
    def test_mask_first_visit_bitwise(self):
        rng=np.random.RandomState(127)
        for i in range(100):
            valid=rng.uniform(size=(80,83))>(i%10)/10
            if i%3==0:valid[8:15]=False
            if i%4==0:valid[:,20:35]=False
            np.testing.assert_array_equal(mask_plan(valid,'fixture:'+str(i),3407+i),
                                          mask_plan_fast(valid,'fixture:'+str(i),3407+i))

    def test_projection_boundaries_and_stable_ties(self):
        lanes={}
        for i,y in enumerate([0,0,3,-3,6,-6]):
            pts=np.array([[-100,y],[0,y],[100,y]],np.float32);g=_build_lane_geom(pts)
            lanes[str(i)]=LaneInfo(str(i),pts,*g)
        rng=np.random.RandomState(3407)
        q=np.column_stack([rng.uniform(-120,120,400),rng.uniform(-10,10,400),rng.uniform(-np.pi,np.pi,400)])
        q=np.vstack([q,[[0,3,0],[0,3+1e-10,0],[0,-3,0],[0,0,np.pi/4],
                        [0,0,np.pi/4+1e-10],[100,0,0],[-100,0,0],[0,0,0]]])
        found=projections_batch(q,lanes)
        for x,p in zip(q,found):
            ref=find_best_lane_for_agent(x[:2],x[2],lanes,3,np.pi/4,20,32,False)[0]
            self.assertEqual(p,ref)


if __name__=='__main__':unittest.main()
