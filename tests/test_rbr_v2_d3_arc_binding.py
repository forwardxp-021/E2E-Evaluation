import unittest
import numpy as np
from tools.rbr_v2_d3_arc_binding import bind_arc
from tools.r1_closed_loop_benchmark_v2_1 import _offset_preserving_route_xy


class ArcBindingTests(unittest.TestCase):
 def test_endpoint_scalar_in_different_polyline_coordinate(self):
  points=[[0.,0.],[10.,0.],[20.,0.]];ego={'rear_axle':{'x':11.4,'y':.3,'heading':0.}}
  arc,a=bind_arc(points,ego,10.,10.,[10.,0.]);self.assertEqual(arc,11.4);self.assertTrue(a['correction_applied'])
  __import__('json').dumps(a)
  old=_offset_preserving_route_xy(points,np.array([10.,11.]),ego);new=_offset_preserving_route_xy(points,np.array([arc,arc+1]),ego)
  self.assertLess(old[1,0],old[0,0]);self.assertGreater(new[1,0],new[0,0]);np.testing.assert_allclose(new,[[11.4,.3],[12.4,.3]])
 def test_correct_case_bitwise_preserved(self):
  points=[[0.,0.],[10.,0.],[20.,0.]];ego={'rear_axle':{'x':7.3,'y':.3,'heading':0.}}
  arc,a=bind_arc(points,ego,7.3,10.,[10.,0.]);self.assertEqual(arc.hex(),float(7.3).hex());self.assertFalse(a['correction_applied'])
 def test_rotated_translated_physics_oracle(self):
  theta=.37;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);shift=np.array([332000.,4690000.]);points=np.array([[0.,0.],[10.,0.],[20.,0.]])@rot.T+shift;p=np.array([11.4,.3])@rot.T+shift
  arc,a=bind_arc(points,{'rear_axle':{'x':p[0],'y':p[1],'heading':theta}},10.,10.,points[1]);self.assertAlmostEqual(arc,11.4,places=8);self.assertTrue(a['correction_applied'])
 def test_behind_start_not_artificially_shifted(self):
  arc,a=bind_arc([[0.,0.],[10.,0.],[20.,0.]],{'rear_axle':{'x':-1.,'y':.1,'heading':0.}},0.,10.,[10.,0.]);self.assertEqual(arc,0);self.assertFalse(a['correction_applied'])

if __name__=='__main__':unittest.main()
