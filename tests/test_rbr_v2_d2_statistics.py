import unittest
import numpy as np
from tools.rbr_v2_d2_statistics import kernels,index_stat,draws
from tools.rbr_v2_d1_statistics import fit_baseline,transform,mmd2

class PairedTests(unittest.TestCase):
 def test_cached_kernel_is_original_unbiased_mmd(self):
  r=np.random.RandomState(4);x=r.normal(size=(20,3));y=r.normal(size=(20,3));s=fit_baseline(r.normal(size=(50,3)),list(map(str,range(50))),'D_fit');K=kernels(x,y,s)
  a=index_stat(K,np.arange(20),20+np.arange(20));b=mmd2(transform(x,s),transform(y,s),s['sigma']);self.assertAlmostEqual(a,b,places=14)
 def test_same_draws_and_real_unique_support(self):
  r=np.random.RandomState(3);x=r.normal(size=(20,2));y=x+.1;s=fit_baseline(r.normal(size=(50,2)),list(map(str,range(50))),'D_fit');K=kernels(x,y,s);ks={'RBR64':K,'H30':K}
  a=draws(ks,list(map(str,range(20))),50,1,True);b=draws(ks,list(map(str,range(20))),50,1,True)
  np.testing.assert_array_equal(a['RBR64'],a['H30']);np.testing.assert_array_equal(a['RBR64'],b['RBR64'])
  with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):draws(ks,list(map(str,range(19))),50,1,True)
  with self.assertRaisesRegex(ValueError,'Duplicate'):draws(ks,['same']*20,50,1,True)

if __name__=='__main__':unittest.main()
