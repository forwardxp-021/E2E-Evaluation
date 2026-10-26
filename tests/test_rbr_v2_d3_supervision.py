import unittest,sys
from pathlib import Path
from tools.rbr_v2_d2_download import ROOT
from tools.rbr_v2_d3_supervision import supervise

class SupervisorTests(unittest.TestCase):
 def test_post_parent_helper_is_cleaned_without_false_scientific_failure(self):
  import uuid
  p=ROOT/'outputs/rbr_v2_d3_v1/supervision_fixture'/uuid.uuid4().hex;p.mkdir(parents=True)
  log=p/'non_scientific_helper.log'
  self.assertFalse(log.exists(),'Fixture output must not overwrite prior evidence')
  r=supervise([sys.executable,str(ROOT/'tests/fixtures/d3_supervision_fixture.py'),'parent'],log,100,wall_limit=5)
  self.assertEqual(r['returncode'],0);self.assertIsNone(r['reason']);self.assertTrue(r['post_parent_helpers_observed']);self.assertFalse(r['post_parent_cleanup_forced']);self.assertGreaterEqual(r['cpu_seconds_budget_charged'],r['cpu_seconds_measured'])

if __name__=='__main__':unittest.main()
