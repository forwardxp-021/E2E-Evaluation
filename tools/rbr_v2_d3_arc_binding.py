#!/usr/bin/env python3
"""Versioned endpoint arc-coordinate compatibility repair; original route, LQR and requests are unchanged."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
import numpy as np
from shapely.geometry import LineString,Point
from tools.tsb_coast_p1r_recorder import fp_bound
from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2,_initial_native_edge
from tools.tsb_direct_coast_f2_production import DirectCoastF2PlannerV1
from tools.tsb_coast_p1r_planner import reference_profile
from tools.r1_closed_loop_benchmark_v2_1 import _offset_preserving_route_xy,_headings_from_xy
VERSION='SDC_D3_ENDPOINT_ARC_BINDING_COMPAT_V1'


def bind_arc(points,current,old_arc,first_length,first_end):
 """Only repair forward endpoint-clamped coordinates. Interior/correct coordinates retain exact original bits."""
 line=LineString(points);p=np.array([current['rear_axle']['x'],current['rear_axle']['y']],float)
 end=np.array(first_end,float)
 projected=float(line.project(Point(*p)));tol=fp_bound(old_arc,first_length,projected)
 clamped=bool(abs(old_arc-first_length)<=tol)
 corrected=clamped and projected>old_arc+tol
 arc=projected if corrected else old_arc
 return arc,dict(original_arc_m=float(old_arc),bound_arc_m=float(arc),stitched_projection_arc_m=projected,first_edge_length_m=float(first_length),endpoint_clamped=clamped,
       correction_applied=bool(corrected),arc_shift_m=float(arc-old_arc),floating_identity_bound_m=tol,
       raw_endpoint_distance_m=float(np.linalg.norm(p-end)),rule='FIRST_EDGE_ENDPOINT_CLAMP_TO_SAME_STITCHED_REFERENCE_COORDINATE; preserve all non-clamped/original-consistent cases bitwise')


def route_binding(api,route_ids,current,required):
 original=build_native_route_reference_v1_2(api,route_ids,current,required)
 first,_=_initial_native_edge(api,route_ids,current)
 length=float(first.baseline_path.linestring.length);end=list(first.baseline_path.linestring.coords)[-1]
 arc,audit=bind_arc(original['reference_xy'],current,original['current_route_arc_m'],length,end)
 line=LineString(original['reference_xy'])
 if arc+required>line.length+fp_bound(arc,required,line.length):
  # Same frozen roadblocks and original native resolver; extend finite coverage only, never invent geometry.
  expanded=build_native_route_reference_v1_2(api,route_ids,current,required+audit['arc_shift_m'])
  old=np.asarray(original['reference_xy']);new=np.asarray(expanded['reference_xy'])
  if len(new)<len(old) or not np.array_equal(new[:len(old)],old):raise ValueError('D3_ARC_REPAIR_CHANGED_NATIVE_PREFIX')
  original=expanded;arc,audit=bind_arc(new,current,original['current_route_arc_m'],length,end)
 if arc+required>LineString(original['reference_xy']).length+fp_bound(arc,required):raise ValueError('D3_CORRECT_ARC_FINITE_COVERAGE_UNAVAILABLE')
 return dict(original,current_route_arc_m=arc),audit


class ArcBoundDirectCoastPlannerV1(DirectCoastF2PlannerV1):
 """Original F2 reference construction with its coordinate input rebound to the sampled native polyline."""
 def name(self):return VERSION+'_'+self._coast_arm

 def compute_planner_trajectory(self,current_input):
  from nuplan.common.actor_state.ego_state import EgoState
  from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
  from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
  if self._initialization is None:raise RuntimeError('D3 planner not initialized')
  index=int(current_input.iteration.index);ego,_=current_input.history.current_state;elapsed=int(ego.time_us)-self._anchor_us
  if not 0<=index<80 or elapsed!=index*100000 or int(current_input.iteration.time_us)!=ego.time_us:raise ValueError('D3 canonical physical clock mismatch')
  current=self._payload(ego);common=elapsed<1100000
  speed,distance,command=reference_profile(current['speed_mps'],elapsed,self._coast_arm,common)
  route,audit=route_binding(self._initialization.map_api,self._row['route_roadblock_ids'],current,float(distance[-1])+1e-9)
  xy=_offset_preserving_route_xy(route['reference_xy'],route['current_route_arc_m']+distance,current);headings=_headings_from_xy(xy)
  states=[EgoState.build_from_rear_axle(StateSE2(*map(float,(*xy[i],headings[i]))),StateVector2D(float(speed[i]),0.),StateVector2D(0.,0.),0.,TimePoint(int(ego.time_us)+i*100000),ego.car_footprint.vehicle_parameters) for i in range(80)]
  states[0]=ego
  self._telemetry.parent.mkdir(parents=True,exist_ok=True)
  with self._telemetry.open('a') as f:f.write(json.dumps(dict(iteration=index,elapsed_episode_us=elapsed,common_entire_future_reference=common,reference_floor_triggered=bool(np.any(speed<=.2)),reference_acceleration_mps2=command.tolist(),reference_speed_mps=speed.tolist(),native_route_builder='v1.2',evidence='PLANNED_NOT_REALIZED',engineering_version=VERSION,arc_binding=audit))+'\n')
  return InterpolatedTrajectory(states)
