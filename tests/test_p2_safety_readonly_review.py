"""Frozen-data diagnostic validation; no runner or motion propagation."""
import unittest
from shapely.geometry import shape
from tools import p2_safety_readonly_review as q


class SafetyReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=q.read(q.OUT/'evidence/P2_Safety_Readonly_Diagnostics_v1.json')

    def test_official_aggregate_does_not_supply_collision_event_time(self):
        raw=self.d['official_raw_aggregate']['no_ego_at_fault_collisions']
        self.assertIsNone(raw['time_series_timestamps'])
        event=self.d['collision_events'][0]
        self.assertEqual(event['collision_type'],'STOPPED_TRACK_COLLISION')
        self.assertEqual(event['object_type'],'VEHICLE')
        self.assertEqual(event['delta_v_proxy'],raw['max_collision_energy_with_vehicles_stat_value'])
        self.assertEqual(event['index'],74)

    def test_discrete_overlap_localization_and_no_double_count(self):
        rows=self.d['collision_object_frames'];by_index={x['index']:x for x in rows}
        self.assertGreater(by_index[73]['ego_object_polygon_distance_m'],0)
        self.assertEqual([x['index'] for x in rows if x['intersection']],list(range(74,80)))
        self.assertEqual(len(self.d['collision_events']),1)
        event=self.d['collision_events'][0]
        polygon=shape(event['ego_box']).intersection(shape(event['object_box']))
        self.assertAlmostEqual(polygon.area,event['overlap_area_m2'],places=12)
        self.assertAlmostEqual(polygon.centroid.x,event['overlap_centroid'][0],places=9)

    def test_corner_crossing_and_qualified_violation_are_distinct(self):
        f=self.d['frames'];threshold=self.d['threshold_m']
        self.assertEqual(next(x['index'] for x in f if x['outside']),42)
        self.assertEqual(next(x['index'] for x in f if x['far']),44)
        self.assertLess(f[42]['corners'][3]['nearest_drivable_distance_m'],threshold)
        self.assertGreater(f[44]['corners'][3]['nearest_drivable_distance_m'],threshold)
        self.assertTrue(all(x['center_drivable'] for x in f))
        self.assertFalse(f[74]['outside']);self.assertFalse(f[69]['outside'])
        raw=self.d['official_raw_aggregate']['corners_in_drivable_area']
        self.assertEqual(raw['time_series_values'],[float(not x['outside']) for x in f])

    def test_actual_clock_and_current_vs_outgoing_command(self):
        f=self.d['frames'];anchor=f[0]['time_us']
        self.assertTrue(all(x['time_us']==anchor+i*100000 for i,x in enumerate(f)))
        self.assertEqual(f[74]['raw_log_time_us']-f[74]['time_us'],169)
        self.assertEqual(f[21]['outgoing_request_mps2'],-1.45)
        self.assertEqual(f[39]['outgoing_request_mps2'],0)
        self.assertEqual(f[74]['outgoing_request_mps2'],0)
        self.assertIsNone(f[79]['outgoing_request_mps2'])

    def test_guard_rejects_motion_propagation(self):
        from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
        calls=[]
        with q.no_execution(calls):
            with self.assertRaisesRegex(RuntimeError,'Read-only'):
                KinematicBicycleModel.propagate_state(None,None,None,None)
        self.assertEqual(calls,['FORBIDDEN_EXECUTION_ATTEMPT'])

    def test_original_failed_ledger_remains_stopped(self):
        prior=q.read(q.ROOT/'docs/stageR/tsb_direct_coast_p2_q1/P2_Q1_Manifest_v1.json')
        ledger=q.ROOT/prior['active_stopped_ledger_path']
        self.assertEqual(q.sha(ledger),prior['active_stopped_ledger_sha256'])
        data=q.read(ledger);self.assertTrue(data['stopped']);self.assertEqual(len(data['attempts']),13)
        self.assertEqual(data['status'],'P2_QUALIFICATION_FAILED')


if __name__=='__main__':unittest.main()
