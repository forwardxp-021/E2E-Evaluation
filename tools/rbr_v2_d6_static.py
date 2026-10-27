#!/usr/bin/env python3
"""D6 initial-frame SQLite/map census only. No runner, trajectory integration or protected C access."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv,json,math,sqlite3,time,resource,socket
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import numpy as np
from tools.rbr_v2_d1_interface import sha,write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rbr_v2_d6_v1';PLAN=ROOT/'configs/rbr_v2_d6_v1/analysis_plan.json'


def read(p):return json.loads(Path(p).read_text())


def check():
    p=read(PLAN)
    for f,h in p['protected'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen source changed '+f)
    return p


def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def task_cell(v,near):
    if not 5<=v<15:return None
    return ('5to10' if v<10 else '10to15')+('_busy' if near else '_empty')


def catalog():
    binding={};records={};sources={}
    def load(p):
        sources[str(p.relative_to(ROOT))]=sha(p);return read(p)
    d2=ROOT/'outputs/rbr_v2_d2_v1'
    for p in sorted((d2/'static').glob('*.json')):
        r=load(p)
        if r['status']=='ELIGIBLE':records[r['session_id']]=r['selected']
    for p in sorted((d2/'round2/static').glob('*.json')):
        r=load(p)
        if r['status']=='ELIGIBLE' and r['session_id'] not in records:records[r['session_id']]=r['selected']
    for f in ['docs/stageR/e2e_scoped_direct_coast_v1/evidence/DEVELOPMENT_Applicability_v4.json','docs/stageR/e2e_scoped_direct_coast_v1/evidence/QUALIFICATION_Applicability_v2.json','outputs/rbr_v2_d1_v1/additional_static_results.json']:
        for r in load(ROOT/f)['rows']:
            if r['status']=='ELIGIBLE':
                if r['session_id'] in records:raise ValueError('SESSION source alias')
                records[r['session_id']]=r
    for r in load(ROOT/'outputs/rbr_v2_d1_v1/D_E3_Static_Results.json')['rows']:
        if r['status']=='ELIGIBLE_D_ONLY':
            if r['session_id'] in records:raise ValueError('SESSION source alias')
            records[r['session_id']]=next(v for v in r['anchor_trials'] if v['status']=='ELIGIBLE')
    roles=load(d2/'candidate_role_partition_final.json');C=set(roles['original_C_protected'])|set(roles['new_C_index_reserved'])
    for name in ['D_fit','D_existing_swap_cal','D_new_science','final_calibration_candidates','V_AA_candidates','V_AB_candidates','unallocated_candidates']:
        for r in roles[name]:
            s=r if isinstance(r,str) else r['session_id']
            if s in binding:raise ValueError('Frozen role overlap')
            binding[s]=name
    if set(records)!=set(binding):raise ValueError('Census does not reproduce D2 role denominator '+repr((len(records),len(binding),set(records)^set(binding))))
    if set(records)&C:raise PermissionError('Protected C source access')
    return records,binding,roles,sources


def main():
    plan=check();start=time.monotonic();cpu=time.process_time();records,roles,role_binding,sources=catalog();rows=[];details=[];dbfingerprints={}
    sys.path.insert(0,str(ROOT.parent/'nuplan-devkit'))
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.state_representation import Point2D
    from shapely.geometry import Point
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    def forbidden(*args,**kwargs):raise PermissionError('D6 forbids network and vehicle propagation')
    with patch.object(socket.socket,'connect',forbidden),patch.object(KinematicBicycleModel,'propagate_state',forbidden):
        api=get_maps_api(str(ROOT.parent/'nuplan/dataset/maps'),'nuplan-maps-v1.0',plan['task']['map'])
        for j,(sid,r) in enumerate(sorted(records.items())):
            spec=r['spec'];e=r['evidence'];initial=spec['precontext']['ego_initial_state'];v=float(e.get('initial_speed_mps',initial['initial_speed_mps']))
            item=dict(session_id=sid,prior_role=roles[sid],map_name=spec['map_name'],log_id=spec['log_id'],scenario_token=spec['scenario_token'],initial_speed_mps=v,source_path=spec['db_path'],status='NOT_ASSESSABLE',cell='',reason='',initial_acceleration_mps2=e.get('initial_acceleration_mps2'),initial_heading_error_rad=e.get('initial_heading_error_rad'),centerline_offset_m=e.get('initial_centerline_offset_m'),route_heading_range_rad=e.get('route_heading_range_rad'),offline_future_static_selection=True)
            if spec['map_name']!=plan['task']['map']:
                item.update(status='OUT_OF_Q',reason='MAP');rows.append(item);continue
            path=Path(spec['db_path'])
            if not path.exists():item['reason']='INITIAL_RAW_DB_UNAVAILABLE';rows.append(item);continue
            x,y=float(initial['initial_x']),float(initial['initial_y']);q=Point2D(x,y);point=Point(x,y);tkn=initial['official_simulation_initial_lidar_token'];before=path.stat()
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                db.execute('PRAGMA query_only=ON')
                header=db.execute('SELECT lower(hex(token)),logfile FROM log').fetchall()
                if len(header)!=1 or header[0][1]!=spec['log_id']:raise ValueError('SESSION/LOG header mismatch')
                ts=db.execute('SELECT p.timestamp,e.x,e.y FROM lidar_pc p JOIN ego_pose e ON e.token=p.ego_pose_token WHERE p.token=?',(bytes.fromhex(tkn),)).fetchall()
                if len(ts)!=1 or ts[0][0]!=initial['initial_time_us']:raise ValueError('Initial token/time mismatch')
                # Frozen metadata serializes initial xy to six decimals; retain raw initial-frame xy.
                np.testing.assert_allclose(ts[0][1:],[x,y],atol=5.01e-7,rtol=0)
                rounding_delta=float(np.max(np.abs(np.array(ts[0][1:])-[x,y])))
                x,y=ts[0][1:];q=Point2D(x,y);point=Point(x,y)
                actors=db.execute('SELECT lower(hex(b.track_token)),c.name,b.x,b.y,b.vx,b.vy,b.yaw FROM lidar_box b JOIN track t ON t.token=b.track_token JOIN category c ON c.token=t.category_token WHERE b.lidar_pc_token=? ORDER BY b.track_token',(bytes.fromhex(tkn),)).fetchall()
                total=db.execute('SELECT count(*) FROM lidar_box WHERE lidar_pc_token=?',(bytes.fromhex(tkn),)).fetchone()[0]
                if total!=len(actors):raise ValueError('Unresolved initial actor category')
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Read-only database changed')
            dbfingerprints[sid]=dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,header_token=header[0][0],initial_lidar_token=tkn,initial_timestamp_us=ts[0][0],raw_sha256=e.get('sources',{}).get('db_sha256'),read_only_SQL=True)
            lanes=api.get_all_map_objects(q,L.LANE);connectors=api.get_all_map_objects(q,L.LANE_CONNECTOR)
            dynamic=[a for a in actors if a[1] in plan['task']['traffic_categories']]
            if any(any(w is None or not math.isfinite(w) for w in a[2:4]) for a in dynamic):item['reason']='MISSING_DYNAMIC_POSITION';rows.append(item);continue
            near=[a for a in dynamic if math.hypot(a[2]-x,a[3]-y)<=50]
            lane=lanes[0] if len(lanes)==1 else None
            nearby_connectors=api.get_proximal_map_objects(q,80,[L.LANE_CONNECTOR])[L.LANE_CONNECTOR]
            lane_data=dict(lane_count=len(lanes),connector_count=len(connectors),out_degree=len(lane.outgoing_edges) if lane else None,in_degree=len(lane.incoming_edges) if lane else None,lane_id=str(lane.id) if lane else '',lane_speed_limit_mps=lane.speed_limit_mps if lane else None,lane_mean_width_proxy_m=float(lane.polygon.area/lane.baseline_path.linestring.length) if lane and lane.baseline_path.linestring.length>0 else None,nearest_connector_distance_m=min((c.polygon.distance(point) for c in nearby_connectors),default=None),actor_count50=len(near),initial_actor_count=len(actors),dynamic_actor_count=len(dynamic),nearest_dynamic_distance_m=min((math.hypot(a[2]-x,a[3]-y) for a in dynamic),default=None),initial_categories=dict(Counter(a[1] for a in actors)),initial_velocities_missing=sum(a[4] is None or a[5] is None for a in dynamic))
            item.update({k:v for k,v in lane_data.items() if k!='initial_categories'});item['initial_xy_metadata_rounding_delta_m']=rounding_delta
            reasons=[]
            if not 5<=v<15:reasons.append('SPEED')
            if len(lanes)!=1 or len(connectors)!=0:reasons.append('NOT_UNIQUE_LANE_INTERIOR')
            if lane is not None and len(lane.outgoing_edges)!=1:reasons.append('NOT_SINGLE_SUCCESSOR')
            item.update(status='OUT_OF_Q' if reasons else 'ELIGIBLE_Q',reason=';'.join(reasons),cell=task_cell(v,bool(near)) if not reasons else '')
            rows.append(item);details.append(dict(session_id=sid,initial_timestamp_us=ts[0][0],lane=lane_data,actors_initial_only=[dict(track_token=a[0],category=a[1],x=a[2],y=a[3],vx=a[4],vy=a[5],yaw=a[6]) for a in actors]))
            if len(details)%20==0:print('D6 initial frames read',len(details),'/',sum(v['spec']['map_name']==plan['task']['map'] for v in records.values()),flush=True)
    cells=['5to10_empty','5to10_busy','10to15_empty','10to15_busy'];counts={role:{c:sum(r['prior_role']==role and r['status']=='ELIGIBLE_Q' and r['cell']==c for r in rows) for c in cells} for role in sorted(set(roles.values()))}
    from scipy.stats import ks_2samp
    coarse=[r for r in rows if r['map_name']==plan['task']['map'] and 5<=r['initial_speed_mps']<15 and 'actor_count50' in r];contrast=[]
    for name,predicate in [('topology',lambda r:r['lane_count']==1 and r['connector_count']==0 and r['out_degree']==1),('traffic_presence',lambda r:r['actor_count50']>0)]:
        for lo,hi in [(5,10),(10,15)]:
            a=[r for r in coarse if lo<=r['initial_speed_mps']<hi and predicate(r)];b=[r for r in coarse if lo<=r['initial_speed_mps']<hi and not predicate(r)]
            contrast.append(dict(speed_bin=[lo,hi],split=name,group_n=[len(a),len(b)],descriptive_only=True,covariates={k:dict(ranges_A=[min((r[k] for r in a if r[k] is not None),default=None),max((r[k] for r in a if r[k] is not None),default=None)],ranges_B=[min((r[k] for r in b if r[k] is not None),default=None),max((r[k] for r in b if r[k] is not None),default=None)]) for k in ['actor_count50','route_heading_range_rad','centerline_offset_m','in_degree','nearest_connector_distance_m']}))
    future=[r for r in rows if r['prior_role'] in ['final_calibration_candidates','V_AA_candidates','V_AB_candidates','unallocated_candidates']]
    pending={r['session_id'] for r in role_binding['final_calibration_candidates'] if r['source']=='D1_E3_STATIC_D_ONLY'}
    strict=[r for r in future if r['session_id'] not in pending and r['status']=='ELIGIBLE_Q'];potential=[r for r in future if r['status']=='ELIGIBLE_Q']
    write(OUT/'static_census.json',dict(plan_sha256=sha(PLAN),total_prior_static_eligible=len(rows),q_eligible=sum(r['status']=='ELIGIBLE_Q' for r in rows),counts_by_prior_role=counts,exclusion_counts=dict(Counter(r['reason'] for r in rows if r['status']!='ELIGIBLE_Q')),strict_unexposed_future_q=len(strict),optimistic_including_pending_E3_cal_q=len(potential),strict_future_cells={c:sum(r['cell']==c for r in strict) for c in cells},coarse_city_speed_contrasts=contrast,static_sources=sources,initial_DB_read_provenance=dbfingerprints,protected_C_raw_read=0,V_C_scientific_results_read=0,new_science=0,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu))
    table(OUT/'SESSION_Task_And_Role_Census.csv',rows);write(OUT/'initial_frame_details.json',details)
    check();print('D6 static census complete',len(rows),'sources;Q=',sum(r['status']=='ELIGIBLE_Q' for r in rows),flush=True)

if __name__=='__main__':main()
