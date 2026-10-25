"""Meaningful permission, failure-policy, geometry-envelope and temporal tests."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import sdc_v1_execution as e
from shapely.geometry import LineString,box


class SDCV1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def ledger(self):
        return dict(stage='E2E_SCOPED_DIRECT_COAST_V1',stopped=False,cpu_certain=True,cpu_seconds=0.,budget=e.BUDGET,attempts=[],pairs=[],phases={'QUALIFICATION':dict(status='RUNNING',binding_sha256='frozen',run_ids=['Q-01-BASELINE','Q-01-TREATMENT'])})

    def test_duplicate_claim_rejected_but_scientific_failure_continues_frozen_sequence(self):
        path=self.root/'budget.json';d=self.ledger();e.write(path,d,exclusive=True)
        with patch.object(e,'STATE',path),patch.object(e,'SCIENCE',self.root/'science'),patch.object(e,'sha',return_value='frozen'):
            e.claim('QUALIFICATION','Q-01-BASELINE')
            with self.assertRaises(PermissionError):e.claim('QUALIFICATION','Q-01-BASELINE')
            d=e.read(path);d['attempts'][-1]['status']='SCIENTIFIC_FAILURE';e.write(path,d)
            e.claim('QUALIFICATION','Q-01-TREATMENT')
        self.assertEqual(len(e.read(path)['attempts']),2)

    def test_incomplete_or_infrastructure_failure_cannot_continue(self):
        path=self.root/'budget.json';d=self.ledger();d['attempts']=[dict(phase='QUALIFICATION',run_id='Q-01-BASELINE',status='CHILD_RUNNING')];e.write(path,d)
        with patch.object(e,'STATE',path),patch.object(e,'SCIENCE',self.root/'science'),patch.object(e,'sha',return_value='frozen'):
            with self.assertRaises(PermissionError):e.claim('QUALIFICATION','Q-01-TREATMENT')
            d['stopped']=True;e.write(path,d)
            with self.assertRaises(PermissionError):e.claim('QUALIFICATION','Q-01-TREATMENT')

    def test_global_cpu_budget_is_not_reset_for_qualification(self):
        path=self.root/'budget.json';d=self.ledger();d['cpu_seconds']=10800.;e.write(path,d)
        with patch.object(e,'STATE',path),patch.object(e,'SCIENCE',self.root/'science'),patch.object(e,'sha',return_value='frozen'):
            with self.assertRaises(PermissionError):e.claim('QUALIFICATION','Q-01-BASELINE')
        self.assertEqual(e.read(path)['cpu_seconds'],10800.)

    def test_corridor_cannot_use_expanded_drivable_polygon_to_hide_violation(self):
        road=box(-2,-2,30,2);center=LineString([(0,0),(20,0)])
        self.assertFalse(road.covers(center.buffer(2.1485,cap_style=2)))
        self.assertTrue(road.covers(center.buffer(1.8,cap_style=2)))
        # Production fixes width=half Pacifca width +1m; this fixture confirms
        # the exact predicate rejects a too-narrow road, no polygon repair.

    def test_temporal_gate_rejects_identical_trajectories_and_requests(self):
        for arm in ['B','T']:
            root=self.root/arm;(root/'trace').mkdir(parents=True);(root/'telemetry').mkdir()
            (root/'trace/realized_current_ego.jsonl').write_text('\n'.join(json.dumps({'current_ego':{'time_us':i*100000,'speed_mps':10.}}) for i in range(80)))
            (root/'telemetry/actual_lqr_controller_telemetry.jsonl').write_text('\n'.join(json.dumps({'longitudinal_requested_acceleration':0}) for i in range(79)))
        result=e.temporal(self.root/'B',self.root/'T');self.assertFalse(result['pass_gate']);self.assertFalse(result['actual_trace_hash_distinct'])
        self.assertFalse(result['request_sequences_distinct'])

    def test_historical_credential_cannot_authorize_new_version(self):
        old=e.read(e.ROOT/'docs/stageR/tsb_direct_coast_f2/F2_Owner_Execution_Authorization_v1.json')
        with patch.object(e,'read',return_value=old):
            with self.assertRaises(PermissionError):e.validate('QUALIFICATION','SDC-Q-01-BASELINE')


if __name__=='__main__':unittest.main()
