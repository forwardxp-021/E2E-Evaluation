"""Zero-run permission, persistent claim and failure-injection tests only."""
import concurrent.futures
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import tsb_direct_coast_p2_q1 as q


class P2Q1Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'budget.json';self.science=self.root/'science'
        self.ledger=dict(stage='TSB_DIRECT_COAST_P2_Q1',authorization_sha256='owner',budget=q.BUDGET,
            approved_run_ids=q.RUNS,attempts=[],pairs=[],stopped=False,status='AUTHORIZED_NOT_STARTED',
            cpu_seconds_consumed=0.,cpu_accounting_certain=True)
        q.write(self.path,self.ledger,exclusive=True)

    def claim(self,run=None):return q.claim(self.path,run or q.RUNS[0],'owner',self.science)

    def test_actual_legacy_credentials_all_forty_rejected(self):
        names=['docs/stageR/tsb_direct_coast_f2/F2_Owner_Execution_Authorization_v1.json',
               'docs/stageR/tsb_direct_coast_p2_q0/P2_Q0_Execution_Authorization_PROPOSED_ONLY_v1.json',
               'docs/stageR/tsb_coast_p1r_first_pair/P1R_D_Owner_FirstPair_Authorization_v1.json',
               'docs/stageR/b1_tsb_qualification/B1_TSB_Qualification_Authorization_v1.json']
        names += [str(x.relative_to(q.ROOT)) for x in (q.ROOT/'docs/stageR/b1_resume_execution_r2').glob('*Authorization*.json')]
        names += [str(x.relative_to(q.ROOT)) for x in (q.ROOT/'docs/stageR/tsb_coast_p1r_canonical_clock').glob('*Authorization*.json')]
        count=0
        for name in names:
            for run in q.RUNS:
                with self.assertRaises(PermissionError):q.validate_scope(q.read(q.ROOT/name),run)
                count+=1
        self.assertGreaterEqual(count,200)

    def test_single_persistent_claim_and_concurrent_duplicate(self):
        def contender():
            try:self.claim();return 'claimed'
            except PermissionError:return 'rejected'
        with concurrent.futures.ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:contender(),range(2)))
        self.assertEqual(sorted(results),['claimed','rejected']);self.assertEqual(len(q.read(self.path)['attempts']),1)
        with self.assertRaises(PermissionError):self.claim(q.RUNS[1])

    def test_interrupted_claim_and_stale_running_never_resume(self):
        self.claim();d=q.read(self.path);d['attempts'][0]['status']='CHILD_RUNNING';q.write(self.path,d)
        for run in q.RUNS[:2]:
            with self.assertRaises(PermissionError):self.claim(run)

    def test_science_failure_stops_before_next_arm(self):
        self.claim();q.finish_attempt(self.path,q.RUNS[0],'SCIENTIFIC_FAILURE',1.,2.,20,None)
        d=q.read(self.path);self.assertTrue(d['stopped']);self.assertEqual(d['status'],'P2_QUALIFICATION_FAILED')
        with self.assertRaises(PermissionError):self.claim(q.RUNS[1])

    def test_technical_failure_or_uncertain_cpu_stops(self):
        self.claim();q.finish_attempt(self.path,q.RUNS[0],'INFRASTRUCTURE_FAILURE',2.,3.,1,'injected Popen failure',False)
        d=q.read(self.path);self.assertTrue(d['stopped']);self.assertFalse(d['cpu_accounting_certain'])
        self.assertEqual(d['status'],'P2_INFRASTRUCTURE_BLOCKED')
        with self.assertRaises(PermissionError):self.claim(q.RUNS[1])

    def test_output_collision_and_scope_order_budget_fail_closed(self):
        (self.science/q.RUNS[0]).mkdir(parents=True)
        with self.assertRaises(FileExistsError):self.claim()
        self.assertFalse(q.read(self.path)['attempts'])
        for change in [dict(cpu_seconds_consumed=57600),dict(cpu_accounting_certain=False),dict(authorization_sha256='other')]:
            d=copy.deepcopy(self.ledger);d.update(change);q.write(self.path,d)
            with self.assertRaises(PermissionError):self.claim()

    def test_pair_failure_stops_and_no_next_session(self):
        for run in q.RUNS[:2]:
            self.claim(run);q.finish_attempt(self.path,run,'COMPLETE_PASS',1.,2.,0,None)
        with self.assertRaises(PermissionError):self.claim(q.RUNS[2])
        q.record_pair(self.path,dict(pair_id='DIRECTCOAST-P2-01',pair_pass=False))
        self.assertEqual(q.read(self.path)['status'],'P2_QUALIFICATION_FAILED')
        with self.assertRaises(PermissionError):self.claim(q.RUNS[2])

    def test_all_twenty_required_and_no_41st_attempt(self):
        for i,run in enumerate(q.RUNS):
            self.claim(run);q.finish_attempt(self.path,run,'COMPLETE_PASS',1.,2.,0,None)
            if i%2:q.record_pair(self.path,dict(pair_id=run.rsplit('-',1)[0],pair_pass=True))
            if i<39:self.assertFalse(q.read(self.path)['stopped'])
        d=q.read(self.path);self.assertEqual(d['status'],'P2_QUALIFICATION_PASS_20_OF_20');self.assertEqual(d['cpu_seconds_consumed'],40)
        with self.assertRaises(PermissionError):self.claim(q.RUNS[0])

    def test_one_shot_execute_stops_after_injected_first_failure(self):
        calls=[]
        def fake_launch(run):
            calls.append(run);self.claim(run);q.finish_attempt(self.path,run,'SCIENTIFIC_FAILURE',1.,1.,20,None);return False
        with patch.object(q,'initialize'),patch.object(q,'LEDGER',self.path),patch.object(q,'launch',fake_launch):q.execute()
        self.assertEqual(calls,[q.RUNS[0]])

    def test_canonical_identity_matches_frozen_q0(self):
        auth=q.read(q.Q0/'P2_Q0_Execution_Authorization_PROPOSED_ONLY_v1.json')
        self.assertEqual([s['run_id'] for s in q.specs()],q.RUNS)
        self.assertEqual({s['run_id']:q.canon(s) for s in q.specs()},auth['proposed_spec_sha256'])


if __name__=='__main__':unittest.main()
