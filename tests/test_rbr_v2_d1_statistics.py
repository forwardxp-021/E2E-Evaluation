import unittest
import numpy as np
from tools.rbr_v2_d1_statistics import fit_baseline,mmd2,release_indices,paired_swap,threshold,alarms,pool,draw_statistics,cluster_interval,assert_role_disjoint,sap_decision

class SAPTests(unittest.TestCase):
 def test_negative_unbiased_and_label_swap(self):
  x=np.array([[0.],[1.],[2.]])
  self.assertLess(mmd2(x,x,1),0)
  self.assertAlmostEqual(mmd2(x,x+1,1),mmd2(x+1,x,1),places=14)
  self.assertAlmostEqual(mmd2(x,x+1,1),mmd2(x[::-1],(x+1)[::-1],1))
 def test_constant_pipeline(self):
  self.assertEqual(mmd2(np.ones((20,30)),np.ones((20,30)),1),0)
  with self.assertRaisesRegex(ValueError,'DEGENERATE'):fit_baseline(np.ones((3,30)),['a','b','c'],'D_fit')
  with self.assertRaises(PermissionError):fit_baseline(np.eye(3),['a','b','c'],'V')
 def test_duplicate_support_and_shared_draws(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):release_indices(['x']*40,np.random.RandomState(1))
  with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):release_indices(list(map(str,range(39))),np.random.RandomState(1))
  ids=list(map(str,range(80)));r,t,d=release_indices(ids,np.random.RandomState(1))
  self.assertFalse(set(r)&set(t));self.assertEqual(d.sum(),10)
  with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):release_indices(ids,np.random.RandomState(1),weights=np.r_[np.ones(39),np.zeros(41)])
 def test_threshold_and_ties(self):
  t=threshold(np.arange(1999));self.assertEqual(t['rank_1based'],1900);self.assertEqual(t['value'],1899)
  self.assertEqual(alarms([1899,1900],t).tolist(),[False,True])
 def test_pair_exchange(self):
  x=np.arange(20.)[:,None];y=x+.5;ids=list(map(str,range(20)));s=np.arange(20)%2==0
  self.assertEqual(paired_swap(x,y,ids,s,1),paired_swap(y,x,ids,~s,1))
 def test_synthetic_aa_deterministic_and_firewall(self):
  rng=np.random.RandomState(72);x=rng.normal(size=(80,4));fit={k:fit_baseline(rng.normal(size=(50,4)),['fit'+str(i) for i in range(50)],'D_fit') for k in ['RBR64','H30']}
  p=pool(list(map(str,range(80))),{k:x for k in fit});a=draw_statistics(p,fit,20,123);b=draw_statistics(p,fit,20,123)
  for k in a:np.testing.assert_array_equal(a[k],b[k])
  p['role']='V'
  with self.assertRaises(PermissionError):draw_statistics(p,fit,1,1)
  with self.assertRaisesRegex(ValueError,'coverage'):cluster_interval(p,p,p,fit)

 def test_near_constant_scaling_and_role_disjoint(self):
  fit={k:fit_baseline(np.array([[0.,0.],[1.,1e-13],[2.,2e-13]]),['a','b','c'],'D_fit') for k in ['RBR64','H30']}
  self.assertEqual(fit['H30']['scale'][1],1)
  p=pool(['a','x'],{k:np.eye(2) for k in fit})
  with self.assertRaisesRegex(ValueError,'overlap'):assert_role_disjoint([p],fit)

 def test_original_sap_decisions(self):
  fpr={'RBR64':[.01,.07],'H30':[.02,.075]}
  self.assertEqual(sap_decision(.12,[.01,.2],fpr,True,True)['primary'],'SUCCESS')
  self.assertEqual(sap_decision(.05,[-.01,.09],fpr,True,True)['primary'],'FAIL')
  self.assertEqual(sap_decision(.05,[-.01,.2],fpr,True,True)['primary'],'INCONCLUSIVE')
  self.assertEqual(sap_decision(.12,[.01,.2],fpr,True,False)['primary'],'INCONCLUSIVE')
  fpr['H30']=[.08,.10]
  self.assertEqual(sap_decision(.12,[.01,.2],fpr,True,True)['FPR_gate'],'FAIL')

if __name__=='__main__':unittest.main()
