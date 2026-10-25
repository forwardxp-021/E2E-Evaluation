"""Engineering sampling boundaries; real training success is checked from actual artifacts."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import unittest
from pathlib import Path
from tools.rbr_v2_smoke_data import ordered_candidates,role_for_group
from tools.rbr_v2_train import entry_gate


class SmokeBoundaryTests(unittest.TestCase):
    def test_observed_dependency_groups_and_test_exclusion(self):
        rows=[]
        for i,(group,candidate,legacy) in enumerate([
            ('shared','train','train'),('shared','val','test'),
            ('candidate_test','test','train'),('candidate_test','train','train'),
            ('cleanA','train','train'),('cleanA','train','val'),('cleanB','train','train')]):
            rows.append(dict(scenario_id=str(i),actor_id=str(i),start='0',length='80',
                dependency_group=group,candidate_split=candidate,legacy_split=legacy))
        result,bad=ordered_candidates(rows)
        self.assertEqual(bad,{'shared','candidate_test'})
        selected=[r for rs in result.values() for r in rs]
        self.assertEqual(len(selected),3)
        self.assertFalse({r['dependency_group'] for r in result['train']} &
                         {r['dependency_group'] for r in result['diagnostic']})
        for r in selected:self.assertEqual(r['role'],role_for_group(r['dependency_group']))
        reverse,_=ordered_candidates(list(reversed(rows)))
        self.assertEqual(result,reverse)

    def test_formal_gate_stays_blocked(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/rbr_v2_uonly_v1/seed3407.json').read_text())
        with self.assertRaisesRegex(ValueError,'RAW_SEGMENT_DISJOINTNESS_NOT_ESTABLISHED'):
            entry_gate(root/'outputs/rbr_v2_uonly_v1/data_manifest.json',cfg)
        self.assertFalse(json.loads((root/'outputs/rbr_v2_uonly_v1/data_manifest.json').read_text())['training_certified'])


if __name__=='__main__':unittest.main()
