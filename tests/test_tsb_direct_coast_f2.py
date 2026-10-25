"""F2 schedule/interface tests only. No runner or model rollout in unit tests."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from tools.tsb_direct_coast_f2_prepare import fixed_initial,fixed_base,OUT
from tools.tsb_direct_coast_f2_tracker import SEGMENTS,request_at,ScriptedLongitudinalRequestTrackerV1
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint


class DirectCoastF2Tests(unittest.TestCase):
    def test_integer_boundaries(self):
        for arm,segments in SEGMENTS.items():
            for start,end,a in segments:
                self.assertEqual(request_at(arm,start-1),0.)
                self.assertEqual(request_at(arm,start),a);self.assertEqual(request_at(arm,end-1),a);self.assertEqual(request_at(arm,end),0.)
        for arm,time in [('UNKNOWN',0),('BASELINE',2.1),('TREATMENT',-1),('BASELINE',True)]:
            with self.assertRaises(ValueError):request_at(arm,time)

    def test_command_impulse_and_midinterval_centroid(self):
        for arm in SEGMENTS:
            u=np.array([request_at(arm,i*100000) for i in range(79)])
            self.assertAlmostEqual(float(.1*u.sum()),-2.61,places=12)
            self.assertAlmostEqual(float(np.dot(np.arange(79)*.1+.05,-u)/(-u).sum()),3.,places=12)
            self.assertTrue((u<=0).all())
        self.assertEqual([request_at('TREATMENT',i*100000) for i in range(21,39)],[0.]*18)

    def test_both_arms_bypass_all_longitudinal_solvers_keep_lateral_and_state(self):
        for arm in SEGMENTS:
            state=fixed_initial();snapshot=lambda:(state.time_us,state.rear_axle.x,state.rear_axle.y,state.rear_axle.heading,state.dynamic_car_state.rear_axle_velocity_2d.x,state.dynamic_car_state.rear_axle_acceleration_2d.x,state.tire_steering_angle);before=snapshot();base=fixed_base()
            tracker=ScriptedLongitudinalRequestTrackerV1(base,state.time_us-2100000,arm)
            now=SimulationIteration(TimePoint(state.time_us),21);nxt=SimulationIteration(TimePoint(state.time_us+100000),22)
            with patch.object(base,'track_trajectory',side_effect=AssertionError('forbidden')),patch.object(base,'_longitudinal_lqr_controller',side_effect=AssertionError('forbidden')),patch.object(base,'_stopping_controller',side_effect=AssertionError('forbidden')),patch.object(base,'_compute_initial_velocity_and_lateral_state',return_value=(14.,np.zeros(3))),patch.object(base,'_compute_reference_velocity_and_curvature_profile',return_value=(99.,np.zeros(10))),patch.object(base,'_lateral_lqr_controller',return_value=.123) as lateral:
                result=tracker.track_trajectory(now,nxt,state,object());self.assertEqual(result.rear_axle_acceleration_2d.x,request_at(arm,2100000));self.assertEqual(result.tire_steering_rate,.123);self.assertEqual(lateral.call_count,1)
            self.assertEqual(snapshot(),before)

    def test_clock_mismatch_rejected(self):
        state=fixed_initial();tracker=ScriptedLongitudinalRequestTrackerV1(fixed_base(),state.time_us,'BASELINE')
        with self.assertRaises(ValueError):tracker.track_trajectory(SimulationIteration(TimePoint(state.time_us),0),SimulationIteration(TimePoint(state.time_us+99999),1),state,None)

    def test_owner_schedule_and_inherited_qualification_exact(self):
        from tools.tsb_coast_p1r_analysis import SCIENCE_CONTRACT
        protocol=json.loads((OUT/'F2_Owner_Protocol_v1.json').read_text());expected=dict(SCIENCE_CONTRACT);expected['version']='F2_DIRECT_REQUEST_APPROVED_v1'
        self.assertEqual(protocol['qualification'],expected)
        self.assertEqual(protocol['request_segments_us'],{key:[list(x) for x in value] for key,value in SEGMENTS.items()})
        self.assertFalse(protocol['automatic_retry']);self.assertEqual(protocol['maximum_new_attempts'],2)


if __name__=='__main__':unittest.main()
