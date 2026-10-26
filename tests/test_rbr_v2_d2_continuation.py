import unittest
from tools.rbr_v2_d2_continue import invoke_frozen_lifecycle,validate,OUT,read

class ContinuationTests(unittest.TestCase):
 def test_exact_existing_lifecycle_argument_binding_without_simulation(self):
  values=[]
  def fixture(arm,spec,contract):values.append((arm,spec,contract));return 'ok'
  self.assertEqual(invoke_frozen_lifecycle(fixture,'arm',{'id':1},'frozen_contract'),'ok')
  self.assertEqual(values,[('arm',{'id':1},'frozen_contract')])
  with self.assertRaises(TypeError):fixture('arm')
 def test_failed_id_and_primary_identity_refused(self):
  for run in ['RBR-D2-D-01-BASELINE','V-AA-01']:
   with self.assertRaises(PermissionError):validate(run)
 def test_same_frozen_cohort_no_replacement_no_budget_reset(self):
  a=read(OUT/'D_continuation_authorization.json');b=read(OUT/'D_execution_binding.json');old=read(OUT/'science_ledger.json')
  self.assertEqual(a['remaining_run_ids'],b['run_ids'][1:]);self.assertEqual(len(a['remaining_run_ids'])+len(old['attempts']),80);self.assertEqual(a['technical_retry'],0);self.assertTrue(old['stopped']);self.assertFalse(set(a['remaining_run_ids'])&{x['run_id'] for x in old['attempts']})

if __name__=='__main__':unittest.main()
