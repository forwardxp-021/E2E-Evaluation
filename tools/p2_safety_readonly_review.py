#!/usr/bin/env python3
"""Read frozen histories only. No runner, controller, planner or model invocation."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
import numpy as np
import pandas as pd
from shapely.geometry import Point,box,mapping
from shapely.ops import unary_union,nearest_points
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
from nuplan.planning.metrics.utils.route_extractor import (get_route,extract_corners_route,
    get_common_or_connected_route_objs_of_corners,get_timestamps_in_common_or_connected_route_objs)
from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import DrivableAreaComplianceStatistics
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import find_new_collisions
from nuplan.planning.metrics.utils.collision_utils import CollisionType
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/stageR/p2_safety_review'
SCIENCE=ROOT/'outputs/stageR/tsb_direct_coast_p2_qualification_v1'
TARGET='DIRECTCOAST-P2-07-BASELINE'
CORNERS=['FL','RL','RR','FR']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [safe(x) for x in value]
    if isinstance(value,np.generic):return value.item()
    return value


def write_new(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(safe(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(path):return json.loads(Path(path).read_text())


def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


@contextlib.contextmanager
def no_execution(calls):
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
    from nuplan.planning.simulation.simulation import Simulation
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.metrics.metric_engine import MetricsEngine
    from tools.tsb_direct_coast_f2_production import DirectCoastF2PlannerV1
    from tools.tsb_direct_coast_f2_tracker import ScriptedLongitudinalRequestTrackerV1
    def forbidden(*args,**kwargs):
        calls.append('FORBIDDEN_EXECUTION_ATTEMPT');raise RuntimeError('Read-only safety review forbids scientific execution')
    with contextlib.ExitStack() as stack:
        for cls,name in [(SimulationRunner,'run'),(Simulation,'initialize'),(Simulation,'propagate'),
            (KinematicBicycleModel,'propagate_state'),(TwoStageController,'update_state'),
            (DirectCoastF2PlannerV1,'compute_trajectory'),(ScriptedLongitudinalRequestTrackerV1,'track_trajectory'),(MetricsEngine,'compute')]:
            stack.enter_context(patch.object(cls,name,forbidden))
        yield


def geometry_diagnosis(states,map_api,threshold):
    centers=get_route(map_api,[s.center for s in states])
    corners=extract_corners_route(map_api,[s.car_footprint for s in states])
    common=get_common_or_connected_route_objs_of_corners(corners)
    connected_ts=set(get_timestamps_in_common_or_connected_route_objs(common,np.array([s.time_us for s in states])))
    metric=DrivableAreaComplianceStatistics('READ_ONLY_GEOMETRY_DIAGNOSIS','DIAGNOSTIC',None,threshold)
    rows=[]
    for i,(s,cr,center) in enumerate(zip(states,corners,centers)):
        detail=[]
        for name,pt,routes in zip(CORNERS,s.car_footprint.all_corners(),cr):
            outside=bool(metric.not_in_drivable_area_with_route_object(pt,routes,map_api))
            nearest_id,distance=map_api.get_distance_to_nearest_map_object(pt,Layer.DRIVABLE_AREA) if outside else (None,0.)
            center_distance=min((x.polygon.distance(Point(pt.x,pt.y)) for x in center),default=None)
            far=bool(metric.is_corner_far_from_drivable_area(map_api,center,pt)) if outside else False
            detail.append(dict(corner=name,x=pt.x,y=pt.y,outside=outside,far=far,nearest_drivable_object_id=nearest_id,
                nearest_drivable_distance_m=distance,center_route_distance_m=center_distance,corner_lane_ids=[x.id for x in routes]))
        rows.append(dict(index=i,time_us=s.time_us,relative_s=(s.time_us-states[0].time_us)/1e6,
            center_x=s.center.x,center_y=s.center.y,rear_x=s.rear_axle.x,rear_y=s.rear_axle.y,heading=s.center.heading,
            speed_mps=s.dynamic_car_state.speed,model_accel_mps2=s.dynamic_car_state.rear_axle_acceleration_2d.x,
            tire_steering_angle=s.tire_steering_angle,
            center_drivable=bool(map_api.is_in_layer(s.center,Layer.DRIVABLE_AREA)),center_lane_ids=[x.id for x in center],
            common_or_connected_route=s.time_us in connected_ts,outside=any(x['outside'] for x in detail),
            far=any(x['far'] for x in detail),corners=detail))
    return rows,connected_ts


def collision_diagnosis(samples,connected_ts):
    collided=set();events=[];overlaps=[]
    for i,sample in enumerate(samples):
        ego=sample.ego_state
        collided,new=find_new_collisions(ego,sample.observation,collided)
        for token,data in new.items():
            obj=next(x for x in sample.observation.tracked_objects if x.track_token==token)
            intersection=ego.car_footprint.geometry.intersection(obj.box.geometry)
            at_fault=data.collision_type in [CollisionType.ACTIVE_FRONT_COLLISION,CollisionType.STOPPED_TRACK_COLLISION] or (data.collision_type==CollisionType.ACTIVE_LATERAL_COLLISION and ego.time_us not in connected_ts)
            events.append(dict(index=i,time_us=ego.time_us,relative_s=(ego.time_us-samples[0].ego_state.time_us)/1e6,
                track_token=token,object_token=obj.token,object_type=data.tracked_object_type.name,collision_type=data.collision_type.name,
                at_fault_by_official_definition=at_fault,common_or_connected_route=ego.time_us in connected_ts,delta_v_proxy=data.collision_ego_delta_v,
                ego_center=[ego.center.x,ego.center.y],ego_rear=[ego.rear_axle.x,ego.rear_axle.y],ego_speed=ego.dynamic_car_state.speed,
                object_center=[obj.center.x,obj.center.y],object_heading=obj.center.heading,object_speed=obj.velocity.magnitude() if hasattr(obj,'velocity') else 0.,
                object_box=mapping(obj.box.geometry),ego_box=mapping(ego.car_footprint.geometry),overlap_area_m2=intersection.area,
                overlap_centroid=[intersection.centroid.x,intersection.centroid.y],source='READ_ONLY_RECONSTRUCTION_OFFICIAL_HELPERS_NOT_EVENT_PARQUET'))
    tokens={x['track_token'] for x in events}
    for i,sample in enumerate(samples):
        ego=sample.ego_state
        for obj in sample.observation.tracked_objects:
            if obj.track_token not in tokens:continue
            ep=ego.car_footprint.geometry;op=obj.box.geometry
            dx=obj.center.x-ego.rear_axle.x;dy=obj.center.y-ego.rear_axle.y;h=ego.rear_axle.heading
            overlaps.append(dict(index=i,track_token=obj.track_token,object_x=obj.center.x,object_y=obj.center.y,
                object_speed=obj.velocity.magnitude() if hasattr(obj,'velocity') else 0.,object_heading=obj.center.heading,
                ego_object_polygon_distance_m=ep.distance(op),intersection=ep.intersects(op),
                overlap_area_m2=ep.intersection(op).area,object_relative_longitudinal_m=dx*np.cos(h)+dy*np.sin(h),
                object_relative_lateral_m=-dx*np.sin(h)+dy*np.cos(h),object_box=mapping(op)))
    return events,overlaps


def analyze_arm(root,full=False):
    manifest=read(root/'execution_manifest.json')
    for p,h in manifest['artifacts'].items():
        if sha(root/p)!=h:raise ValueError('Frozen arm artifact changed: '+str(root/p))
    p=next((root/'raw/simulation_log').rglob('*.msgpack.xz'))
    log=SimulationLog.load_data(p);samples=log.simulation_history.data;states=[x.ego_state for x in samples]
    assert len(states)==80
    config=read(root/'resolved_config.json');threshold=config['simulation_metric']['high_level']['drivable_area_compliance_statistics']['max_violation_threshold']
    rows,connected=geometry_diagnosis(states,log.simulation_history.map_api,threshold)
    events,overlaps=collision_diagnosis(samples,connected)
    raw={name:pd.read_parquet(root/'raw/metrics'/(name+'.parquet')).iloc[0].to_dict() for name in ['no_ego_at_fault_collisions','drivable_area_compliance','corners_in_drivable_area']}
    assert [int(x) for x in raw['corners_in_drivable_area']['time_series_timestamps']]==[s.time_us for s in states]
    assert [float(not x['outside']) for x in rows]==list(raw['corners_in_drivable_area']['time_series_values'])
    assert sum(x['at_fault_by_official_definition'] for x in events)==raw['no_ego_at_fault_collisions']['number_of_all_at_fault_collisions_stat_value']
    assert float(not any(x['far'] for x in rows))==raw['drivable_area_compliance']['drivable_area_compliance_stat_value']
    result=dict(run_id=manifest['spec']['run_id'],spec=manifest['spec'],map_name=log.scenario.map_api.map_name,
        official_raw_aggregate=raw,threshold_m=threshold,collision_events=events,collision_object_frames=overlaps,frames=rows,
        reconstruction_agrees_with_existing_metrics=True,collision_parquet_has_event_timestamps=raw['no_ego_at_fault_collisions']['time_series_timestamps'] is not None,
        source_simulation_log_sha256=sha(p),vehicle_parameters=vars(states[0].car_footprint.vehicle_parameters))
    if full:
        telemetry=jsonl(root/'telemetry/actual_lqr_controller_telemetry.jsonl');trace=jsonl(root/'trace/realized_current_ego.jsonl')
        clock=read(root/'canonical_clock_mapping.json')['rows']
        for i,s in enumerate(states):
            assert s.time_us==trace[i]['current_ego']['time_us'] and s.dynamic_car_state.speed==trace[i]['current_ego']['speed_mps']
            rows[i].update(raw_log_time_us=clock[i]['raw_log_time_us'],raw_minus_sim_us=clock[i]['raw_minus_sim_us'],
                outgoing_request_mps2=telemetry[i]['longitudinal_requested_acceleration'] if i<79 else None,
                incoming_request_mps2=telemetry[i-1]['longitudinal_requested_acceleration'] if i else None,
                steering_rate_request=telemetry[i]['steering_request'] if i<79 else None)
        expert=[log.scenario.get_ego_state_at_iteration(i) for i in range(80)]
        erows,_=geometry_diagnosis(expert,log.scenario.map_api,threshold)
        result['expert_reference_frames']=erows
        result['expert_is_executed_counterfactual']=False
        result['planned_reference_frames']=[]
        for i,sample in enumerate(samples):
            # Existing serialized trajectories only; no planner recomputation.
            planned=sample.trajectory.get_sampled_trajectory()
            near=planned[min(10,len(planned)-1)]
            result['planned_reference_frames'].append(dict(index=i,preview_1s_xy=[near.rear_axle.x,near.rear_axle.y],
                preview_heading=near.rear_axle.heading,preview_speed=near.dynamic_car_state.speed,
                preview_points=[[s.rear_axle.x,s.rear_axle.y] for s in planned[:21]]))
        pts=[Point(s.center.x,s.center.y) for s in states+expert];bounds=unary_union(pts).bounds
        extent=box(bounds[0]-25,bounds[1]-25,bounds[2]+25,bounds[3]+25)
        map_api=log.scenario.map_api
        surfaces=map_api._get_vector_map_layer(Layer.DRIVABLE_AREA)
        local=surfaces[surfaces.geometry.intersects(extent)]
        # No geometry simplification/buffer/repair; full original polygons retained.
        da_union=unary_union(list(local.geometry))
        geometry=dict(extent=list(extent.bounds),drivable_polygons=[dict(id=str(row.get('fid',idx)),geometry=mapping(row.geometry)) for idx,row in local.iterrows()],
            lanes=[],route_ids=manifest['spec']['route_roadblock_ids'])
        lane_objects=map_api.get_proximal_map_objects(states[0].center,120,[Layer.LANE,Layer.LANE_CONNECTOR])
        for layer,objects in lane_objects.items():
            for obj in objects:
                if obj.polygon.intersects(extent):geometry['lanes'].append(dict(id=obj.id,layer=layer.name,roadblock_id=obj.get_roadblock_id(),polygon=mapping(obj.polygon),baseline=mapping(obj.baseline_path.linestring)))
        result['map_geometry']=geometry
        for row,s in zip(rows,states):
            c=Point(s.center.x,s.center.y)
            row['center_signed_drivable_clearance_m']=c.distance(da_union.boundary) if da_union.covers(c) else -c.distance(da_union)
            row['footprint_outside_drivable_area_m2']=s.car_footprint.geometry.difference(da_union).area
            for cr in row['corners']:
                pt=Point(cr['x'],cr['y']);cr['local_union_distance_m']=pt.distance(da_union)
                nearest=nearest_points(pt,da_union)[1];cr['nearest_drivable_xy']=[nearest.x,nearest.y]
    return result


def main():
    calls=[];results=[]
    with no_execution(calls):
        for i in range(1,8):
            for arm in ('BASELINE','TREATMENT'):
                if i==7 and arm=='TREATMENT':continue
                run=f'DIRECTCOAST-P2-{i:02d}-{arm}'
                print('Read frozen history:',run,flush=True)
                result=analyze_arm(SCIENCE/run,run==TARGET)
                results.append(result)
    assert not calls
    target=next(x for x in results if x['run_id']==TARGET)
    write_new(OUT/'evidence/P2_Safety_Readonly_Diagnostics_v1.json',target)
    comparisons=[]
    for x in results:
        f=x['frames'];outside=[r for r in f if r['outside']];far=[r for r in f if r['far']]
        comparisons.append(dict(run_id=x['run_id'],initial_speed_mps=f[0]['speed_mps'],first_corner_outside_s=outside[0]['relative_s'] if outside else None,
            first_corner_far_s=far[0]['relative_s'] if far else None,outside_frames=len(outside),far_frames=len(far),
            max_outside_corner_distance_m=max([c['nearest_drivable_distance_m'] or 0 for row in f for c in row['corners']]),
            reconstructed_first_collision_s=x['collision_events'][0]['relative_s'] if x['collision_events'] else None,
            official_at_fault_collisions=x['official_raw_aggregate']['no_ego_at_fault_collisions']['number_of_all_at_fault_collisions_stat_value'],
            official_drivable_compliance=x['official_raw_aggregate']['drivable_area_compliance']['drivable_area_compliance_stat_value'],
            reconstruction_agrees_with_existing_metrics=x['reconstruction_agrees_with_existing_metrics']))
    write_new(OUT/'evidence/P2_Safety_Existing_13Arm_Interface_Comparison_v1.json',comparisons)
    write_new(OUT/'evidence/P2_Safety_Execution_Guard_v1.json',dict(status='PASS',forbidden_calls=calls,scientific_runner=0,model_propagation=0,metric_engine_compute=0,unrun_arms_loaded=0,old_verdicts_modified=0))
    print(json.dumps(safe(dict(collisions=target['collision_events'],outside=[x['index'] for x in target['frames'] if x['outside']],far=[x['index'] for x in target['frames'] if x['far']],comparison=comparisons)),indent=2))


def supplement():
    calls=[];comparisons=[];target=None
    with no_execution(calls):
        for root in sorted(SCIENCE.glob('DIRECTCOAST-P2-*-*')):
            if not root.is_dir():continue
            path=next((root/'raw/simulation_log').rglob('*.msgpack.xz'))
            log=SimulationLog.load_data(path);samples=log.simulation_history.data
            states=[x.ego_state for x in samples];api=log.scenario.map_api
            bounds=unary_union([Point(x.center.x,x.center.y) for x in states]).bounds
            extent=box(bounds[0]-25,bounds[1]-25,bounds[2]+25,bounds[3]+25)
            surfaces=api._get_vector_map_layer(Layer.DRIVABLE_AREA)
            union=unary_union(list(surfaces[surfaces.geometry.intersects(extent)].geometry))
            def signed(pt):
                pt=Point(pt.x,pt.y)
                return pt.distance(union.boundary) if union.covers(pt) else -pt.distance(union)
            min_corner=min((signed(c),i,n) for i,x in enumerate(states) for n,c in zip(CORNERS,x.car_footprint.all_corners()))
            nearest=[(sample.ego_state.car_footprint.geometry.distance(obj.box.geometry),i,obj.track_token)
                for i,sample in enumerate(samples) for obj in sample.observation.tracked_objects]
            comparisons.append(dict(run_id=root.name,minimum_corner_signed_clearance_m=min_corner[0],minimum_clearance_index=min_corner[1],corner=min_corner[2],
                minimum_any_actor_box_distance_m=min(nearest)[0] if nearest else None,
                minimum_actor_distance_index=min(nearest)[1] if nearest else None,nearest_track_token=min(nearest)[2] if nearest else None))
            if root.name!=TARGET:continue
            expert=[log.scenario.get_ego_state_at_iteration(i) for i in range(80)]
            experts=[];tracking=[];reference=[]
            event_token=read(OUT/'evidence/P2_Safety_Readonly_Diagnostics_v1.json')['collision_events'][0]['track_token']
            positions=[]
            for i,sample in enumerate(samples):
                ego=sample.ego_state;ex=expert[i]
                objs=list(sample.observation.tracked_objects)
                target_obj=next((x for x in objs if x.track_token==event_token),None)
                raw_obj=next((x for x in log.scenario.get_tracked_objects_at_iteration(i).tracked_objects if x.track_token==event_token),None)
                assert target_obj is not None and raw_obj is not None
                assert target_obj.box==raw_obj.box and target_obj.metadata.timestamp_us==raw_obj.metadata.timestamp_us
                positions.append(dict(index=i,x=target_obj.center.x,y=target_obj.center.y,speed_mps=target_obj.velocity.magnitude(),raw_object_timestamp_us=target_obj.metadata.timestamp_us))
                experts.append(dict(index=i,ego_center_distance_m=ego.center.distance_to(ex.center),
                    expert_object_box_distance_m=ex.car_footprint.geometry.distance(target_obj.box.geometry),
                    expert_any_box_overlap_tracks=[x.track_token for x in objs if ex.car_footprint.geometry.intersects(x.box.geometry)],
                    expert_time_us=ex.time_us,simulation_time_us=ego.time_us))
                planned=sample.trajectory.get_sampled_trajectory()
                k=min(10,len(planned)-1);pr=planned[k]
                reference.append(dict(index=i,reference_1s_time_us=pr.time_us,reference_1s_center=[pr.center.x,pr.center.y],
                    reference_1s_corners_outside_drivable=[n for n,c in zip(CORNERS,pr.car_footprint.all_corners()) if not api.is_in_layer(c,Layer.DRIVABLE_AREA)],
                    next_planned_rear=[planned[1].rear_axle.x,planned[1].rear_axle.y]))
                if i<79:
                    nxt=states[i+1];ref=planned[1]
                    dx=nxt.rear_axle.x-ref.rear_axle.x;dy=nxt.rear_axle.y-ref.rear_axle.y;h=ref.rear_axle.heading
                    tracking.append(dict(index=i+1,next_step_position_error_m=float(np.hypot(dx,dy)),
                        next_step_lateral_error_m=float(-dx*np.sin(h)+dy*np.cos(h))))
            target=dict(expert_descriptive_comparison=experts,stored_planned_reference_diagnostics=reference,
                one_step_tracking_diagnostics=tracking,collided_object_full_window=positions,
                collided_object_same_index_raw_source_equal=True,expert_counterfactual_simulation=False,
                initial_expert_model_acceleration=expert[0].dynamic_car_state.rear_axle_acceleration_2d.x)
    assert not calls
    write_new(OUT/'evidence/P2_Safety_Geometric_Margins_v1.json',comparisons)
    write_new(OUT/'evidence/P2_Safety_Expert_Reference_And_Object_Diagnostics_v1.json',target)
    print(json.dumps(safe(dict(margins=comparisons,expert_min_object_gap=min(x['expert_object_box_distance_m'] for x in target['expert_descriptive_comparison']),
        expert_any_overlap_count=sum(bool(x['expert_any_box_overlap_tracks']) for x in target['expert_descriptive_comparison']),
        object_x_range=[min(x['x'] for x in positions),max(x['x'] for x in positions)],object_y_range=[min(x['y'] for x in positions),max(x['y'] for x in positions)],
        max_tracking_lateral_error=max(abs(x['next_step_lateral_error_m']) for x in tracking),guard_calls=calls)),indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--supplement',action='store_true')
    args=parser.parse_args()
    supplement() if args.supplement else main()
