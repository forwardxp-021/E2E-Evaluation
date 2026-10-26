#!/usr/bin/env python3
"""Deterministic frozen-stack reference/LQR replay; no runner, state propagation or metrics."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json,copy
from pathlib import Path
import numpy as np
from tools.rbr_v2_d2_download import ROOT,read,write,sha
OUT=ROOT/'outputs/rbr_v2_d3_v1'


def diagnose(namespace='zero_run_stack',patched=False):
 from tools.r1_b2_8_r3_prospective_selector import official_env
 official_env()
 from tools.b1_production_preflight import zero_run_guard
 from tools.tsb_direct_coast_f2_production import build_f2_arm,validate_f2_production
 from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization,PlannerInput
 from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
 from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
 from nuplan.planning.simulation.controller.tracker.tracker_utils import get_interpolated_reference_trajectory_poses
 spec=read(ROOT/'outputs/rbr_v2_d2_v1/D_execution_binding.json')['arms'][1]
 with zero_run_guard() as calls:
  if patched:
   from tools.rbr_v2_d3_production import build,validate
   arm=build(spec,OUT/namespace);validate(arm)
  else:arm=build_f2_arm(spec,OUT/namespace);validate_f2_production(arm)
  scenario=arm.simulation.scenario
  ego=scenario.get_ego_state_at_iteration(0);obs=scenario.get_tracked_objects_at_iteration(0)
  arm.planner.initialize(PlannerInitialization(spec['route_roadblock_ids'],scenario.get_mission_goal(),scenario.map_api))
  inp=PlannerInput(SimulationIteration(ego.time_point,0),SimulationHistoryBuffer.initialize_from_list(1,[ego],[obs],.1))
  traj=arm.planner.compute_planner_trajectory(inp);base=arm.tracker._base
  from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
  from tools.r1_closed_loop_benchmark_v2 import sample_native_reference_no_extrapolation
  from shapely.geometry import LineString,Point
  current=arm.planner._payload(ego);route=build_native_route_reference_v1_2(scenario.map_api,spec['route_roadblock_ids'],current,ego.dynamic_car_state.speed*7.9)
  line=LineString(route['reference_xy']);projected=float(line.project(Point(ego.rear_axle.x,ego.rear_axle.y)));oldarc=route['current_route_arc_m']
  centers,_=sample_native_reference_no_extrapolation(route['reference_xy'],[oldarc,oldarc+ego.dynamic_car_state.speed*.1,projected])
  geometry=dict(native_edge_ids=route['native_edge_ids'],route_arc=oldarc,polyline_project_arc=projected,arc_difference=projected-oldarc,centers=centers.tolist(),first_polyline_points=route['reference_xy'][:5].tolist(),reference_length=line.length,initial_point_line_distance=line.distance(Point(ego.rear_axle.x,ego.rear_axle.y)))
  v,lateral=base._compute_initial_velocity_and_lateral_state(inp.iteration,ego,traj)
  rv,curv=base._compute_reference_velocity_and_curvature_profile(inp.iteration,traj)
  vp=np.full(base._tracking_horizon,v);rate=base._lateral_lqr_controller(lateral,vp,curv)
  times,poses=get_interpolated_reference_trajectory_poses(traj,base._discretization_time)
  samples=traj.get_sampled_trajectory()
  records=[dict(i=i,x=s.rear_axle.x,y=s.rear_axle.y,heading=s.rear_axle.heading,tire_steering_angle=s.tire_steering_angle,vx=s.dynamic_car_state.rear_axle_velocity_2d.x,vy=s.dynamic_car_state.rear_axle_velocity_2d.y) for i,s in enumerate(samples)]
  frozen=[json.loads(s) for s in (ROOT/'outputs/stageR/rbr_v2_d2_D/RBR-D2-D-01-TREATMENT/telemetry/actual_lqr_controller_telemetry.jsonl').read_text().splitlines()]
  if not patched:np.testing.assert_allclose(rate,frozen[0]['steering_request'],rtol=0,atol=1e-10)
 if any(calls.values()):raise RuntimeError('Diagnostic invoked science')
 result=dict(run_id=spec['run_id'],session_id=spec['session_id'],patched=patched,scope='REFERENCE_AND_PRIVATE_LQR_ALGEBRA_ONLY_NO_MODEL_PROPAGATION',
       reproduced_first_steering_rate_rad_s=float(rate),recorded_first_rate_rad_s=frozen[0]['steering_request'],initial_lateral_state=lateral.tolist(),initial_velocity=v,
       initial_ego=dict(heading=ego.rear_axle.heading,tire_steering_angle=ego.tire_steering_angle,tire_steering_rate=ego.dynamic_car_state.tire_steering_rate),reference_velocity=rv,curvature_profile=curv.tolist(),
       reference_states=records,interpolated_poses=poses.tolist(),geometry=geometry,tracking_horizon=base._tracking_horizon,tracker_dt=base._discretization_time,zero_run_guard=calls)
 write(OUT/('reference_lqr_diagnosis.json' if namespace=='zero_run_stack' else namespace+'_diagnosis.json'),result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['reference_states','interpolated_poses','zero_run_guard']},indent=2),flush=True)
 print('First reference states:',records[:7],flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--namespace',default='zero_run_stack');p.add_argument('--patched',action='store_true');a=p.parse_args();diagnose(a.namespace,a.patched)
