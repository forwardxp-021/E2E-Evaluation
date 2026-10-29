#!/usr/bin/env python3
"""Real raw-log BDD-P0 corpus. Source/role freeze, canonical resampling, no vehicle propagation."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse,contextlib,copy,csv,hashlib,json,math,socket,sqlite3,time,zlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
from collections import Counter
from unittest.mock import patch
import numpy as np
from tools.rbr_v2_d1_interface import sha,write
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/bdd_p0_v1';PLAN=ROOT/'configs/bdd_p0_v1/protocol.json'


def read(p):return json.loads(Path(p).read_text())


def check():
    p=read(PLAN)
    for f,h in p['protected'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen history/model changed '+f)
    return p


def digest(*s):return hashlib.sha256('|'.join(map(str,s)).encode()).hexdigest()


def block(s):return s.split('_veh-')[-1]+'|'+s[:10]


def roster():
    p=check();roles=read(ROOT/'outputs/rbr_v2_d2_v1/candidate_role_partition_final.json');old=read(ROOT/'outputs/rbr_v2_d2_v1/acquisition_plan.json')
    excluded=set(old['old_SESSION_excluded'])|set(old['new_C_METADATA_ONLY'])
    for k in ['D_fit','D_existing_swap_cal','D_new_science','final_calibration_candidates','V_AA_candidates','V_AB_candidates','unallocated_candidates','original_C_protected','new_C_index_reserved']:
        excluded|={x if isinstance(x,str) else x['session_id'] for x in roles[k]}
    forbidden=set()
    for s in excluded:
        d=datetime.strptime(s[:10],'%Y.%m.%d');vehicle=s.split('_veh-')[-1]
        forbidden|={vehicle+'|'+(d+timedelta(days=k)).strftime('%Y.%m.%d') for k in [-1,0,1]}
    meta={};sources={}
    for f in sorted((ROOT/'outputs/rbr_v2_d2_v1/static').glob('*.json')):
        r=read(f);meta[r['session_id']]=r;sources[str(f.relative_to(ROOT))]=sha(f)
    candidates=[r for s,r in meta.items() if s not in excluded and block(s) not in forbidden]
    grouped={}
    for r in candidates:grouped.setdefault(block(r['session_id']),[]).append(r)
    samples=[]
    def item(s,source,role):
        path=Path(source['path']);cache=ROOT/'outputs/rbr_v2_d2_v1/cache_transforms'/(path.name+'.json')
        return dict(session_id=s,dependency_block=block(s),role=role,map_name=source['map_name'],source_path=str(path),source_sha256=source['sha256'],archive_receipt=str(cache.relative_to(ROOT)) if cache.exists() else None,source_rank=digest(p['source_salt'],s,path.name))
    fresh=[]
    for b,rs in grouped.items():
        r=min(rs,key=lambda r:digest(p['source_salt'],r['session_id'],r['log']['log_id']))
        fresh.append(item(r['session_id'],dict(path=r['log']['source_file'],sha256=r['download_sha256'],map_name=r['log']['map_version']),'UNASSIGNED'))
    for city in sorted({r['map_name'] for r in fresh}):
        ordered=sorted([r for r in fresh if r['map_name']==city],key=lambda r:digest(p['role_salt'],r['dependency_block']))
        for i,r in enumerate(ordered):r['role']='D_P0_CAL' if i%2==0 else 'D_P0_HELDOUT';samples.append(r)
    # Explicitly old exposed development sources only; no detector outcome values used.
    prior=read(ROOT/'outputs/rbr_v2_d3_v1/interface/input_manifest.json')['rows']+read(ROOT/'outputs/rbr_v2_d1_v1/input_manifest.json')['rows'];dev={}
    for r in prior:
        if r['arm']!='BASELINE':continue
        s=r['session_id']
        if s in meta:
            mr=meta[s];source=dict(path=mr['log']['source_file'],sha256=mr['download_sha256'],map_name=mr['log']['map_version'])
        else:
            manifest=ROOT/r['root']/'execution_manifest.json';e=read(manifest);spec=e['spec'];path=Path(spec['db_path']);source=dict(path=path,sha256=sha(path),map_name=spec['map_name']);sources[str(manifest.relative_to(ROOT))]=sha(manifest)
        row=item(s,source,'D_P0_FIT_EXPOSED');dev.setdefault(block(s),[]).append(row)
    for rs in dev.values():samples.append(min(rs,key=lambda r:r['source_rank']))
    if len({r['dependency_block'] for r in samples})!=len(samples):raise ValueError('FIT/CAL/HELDOUT source block overlap')
    target=OUT/'identity_roster.json'
    if target.exists():raise FileExistsError('Prospective identities already frozen')
    write(target,dict(protocol_sha256=sha(PLAN),source_session_denominator=len(meta),identity_available=sum(s not in excluded for s in meta),after_guard_SESSION=len(candidates),fresh_blocks=len(fresh),development_blocks=len(dev),role_counts=dict(Counter(r['role'] for r in samples)),samples=samples,source_metadata=sources,forbidden_guard_blocks=sorted(forbidden),policy_excluded_identities=sorted(excluded),SDC_status_used_for_selection=False,source_population_limit='Residual role-available first-cost-source frame; exclusion of old reservations can correlate with prior SDC applicability; not all nuPlan.'))
    print('BDD-P0 identities frozen',len(samples),dict(Counter(r['role'] for r in samples)),flush=True)


@contextlib.contextmanager
def source_db(row):
    original=Path(row['source_path']);owned=None
    if original.exists():path=original
    elif row['archive_receipt']:
        receipt=read(ROOT/row['archive_receipt']);archive=Path(receipt['archive_path'])
        if sha(archive)!=receipt['archive_sha256']:raise ValueError('Raw archive changed')
        folder=OUT/'work_cache';folder.mkdir(exist_ok=True);owned=folder/(original.name+'.P0_owned');path=owned
        de=zlib.decompressobj();h=hashlib.sha256()
        with archive.open('rb') as f,path.open('wb') as g:
            for data in iter(lambda:f.read(1024*1024),b''):
                raw=de.decompress(data);h.update(raw);g.write(raw)
            tail=de.flush();h.update(tail);g.write(tail)
        if not de.eof or h.hexdigest()!=row['source_sha256']:raise ValueError('Lossless raw restoration failed')
    else:raise FileNotFoundError('Missing source and certified archive: '+str(original))
    if sha(path)!=row['source_sha256']:raise ValueError('Raw source hash changed')
    try:yield path
    finally:
        if owned is not None and owned.exists():owned.unlink() # Only this new owned scratch copy.


def prepare_window(db,row,api):
    from tools.rbr_v2_canonical83 import resample,assign_slots,nuplan_lane_infos,rotate
    from tools.s1_protocol_schema import handcrafted_h
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
    p=read(PLAN);source=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);source.execute('PRAGMA query_only=ON')
    try:
        header=source.execute('SELECT lower(hex(token)),logfile,vehicle_name FROM log').fetchall()
        if len(header)!=1 or header[0][1].rsplit('_',2)[0]!=row['session_id']:raise ValueError('Source SESSION/header mismatch')
        raw=source.execute('SELECT p.timestamp,lower(hex(p.token)),e.x,e.y,e.qw,e.qx,e.qy,e.qz,e.vx,e.vy,e.angular_rate_z FROM lidar_pc p JOIN ego_pose e ON e.token=p.ego_pose_token ORDER BY p.timestamp,p.token').fetchall()
        times=np.array([r[0] for r in raw],np.int64)
        if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Invalid raw physical clock')
        candidates=[i for i in range(0,len(times),10) if times[-1]-times[i]>=7900000]
        candidates.sort(key=lambda i:digest(p['window_salt'],row['session_id'],int(times[i])))
        if not candidates:raise ValueError('NO_COMPLETE_7P9_SECOND_WINDOW')
        invalid=[];vp=get_pacifica_parameters();d=vp.rear_axle_to_center
        for start in candidates:
            end=int(np.searchsorted(times,times[start]+7900000));local=raw[start:end+1];tt=times[start:end+1];a=np.array([r[2:] for r in local],float)
            valid=np.isfinite(a).all(1);q=a[:,2:6];norm=np.linalg.norm(q,axis=1);valid&=norm>1e-10;q=q/np.maximum(norm[:,None],1e-10)
            qw,qx,qy,qz=q.T;heading=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));body=a[:,6:8];world=rotate(body,-heading);center_xy=a[:,:2]+d*np.column_stack([np.cos(heading),np.sin(heading)]);center_body=body.copy();center_body[:,1]+=a[:,8]*d;center_world=rotate(center_body,-heading)
            rear=resample(np.column_stack([a[:,:2],world,heading,valid]),tt,int(tt[0]));ego=resample(np.column_stack([center_xy,center_world,heading,valid]),tt,int(tt[0]))
            if not (rear[:,5]>.5).all() or not (ego[:,5]>.5).all():invalid.append(dict(origin_us=int(tt[0]),reason='INVALID_EGO_PHYSICAL_INPUT'));continue
            origin=int(tt[0]);break
        else:raise ValueError('NO_QUALITY_VALID_WINDOW '+str(invalid[:3]))
        # Independent official EgoState center/lever-arm oracle on a real logged state.
        state=EgoState.build_from_rear_axle(StateSE2(a[0,0],a[0,1],heading[0]),StateVector2D(*body[0]),StateVector2D(0,0),0.,TimePoint(origin),vehicle_parameters=vp,angular_vel=float(a[0,8]))
        np.testing.assert_allclose([state.center.x,state.center.y],center_xy[0],atol=1e-9,rtol=0)
        cv=state.dynamic_car_state.center_velocity_2d;np.testing.assert_allclose([cv.x,cv.y],center_body[0],atol=1e-12,rtol=0)
        # Neighbor tracks are actual world-frame vehicle detections, no synthesized agents.
        boxes=source.execute('SELECT p.timestamp,lower(hex(b.track_token)),b.x,b.y,b.vx,b.vy,b.yaw FROM lidar_box b JOIN lidar_pc p ON p.token=b.lidar_pc_token JOIN track tr ON tr.token=b.track_token JOIN category c ON c.token=tr.category_token WHERE p.timestamp>=? AND p.timestamp<=? AND c.name=? ORDER BY p.timestamp,b.track_token',(int(tt[0]),int(tt[-1]),'vehicle')).fetchall()
        at={int(t):i for i,t in enumerate(tt)};tracks={}
        for t,token,x,y,vx,vy,yaw in boxes:
            if t not in at:continue
            track=tracks.setdefault(token,np.zeros((len(tt),6)));vals=[x,y,vx,vy,yaw]
            if all(v is not None and math.isfinite(v) for v in vals):track[at[t]]=[*vals,1.]
        tracks={k:resample(v,tt,origin) for k,v in tracks.items()}
        midpoint=ego[:,:2].mean(0);radius=float(np.linalg.norm(ego[:,:2]-midpoint,axis=1).max()+120)
        prox=api.get_proximal_map_objects(Point2D(*midpoint),radius,[L.LANE,L.LANE_CONNECTOR]);infos=nuplan_lane_infos(prox[L.LANE],prox[L.LANE_CONNECTOR])
        sample=assign_slots(ego,tracks,infos)
        states=[dict(rear_axle=dict(x=float(v[0]),y=float(v[1]),heading=float(v[4])),speed_mps=float(np.linalg.norm(v[2:4])),time_us=origin+i*100000) for i,v in enumerate(rear)]
        h=handcrafted_h(states)
        if h.shape!=(30,) or not np.isfinite(h).all():raise ValueError('Frozen H30 invalid')
        speed=float(np.linalg.norm(rear[0,2:4]));point=Point2D(*rear[0,:2]);lanes=api.get_all_map_objects(point,L.LANE);connectors=api.get_all_map_objects(point,L.LANE_CONNECTOR)
        task='LOW' if speed<2 else 'JUNCTION' if connectors else 'LANE' if lanes else 'OTHER'
        observed_world=np.diff(a[:,:2],axis=0)/np.diff(tt)[:,None]*1e6
        midbody=(body[:-1]+body[1:])/2;midworld=(world[:-1]+world[1:])/2
        rmse_world=float(np.sqrt(np.mean((observed_world-midworld)**2)));rmse_wrong_body=float(np.sqrt(np.mean((observed_world-midbody)**2)))
        # World-vs-body diagnostic only; do not infer coordinate conventions by per-window fit.
        return sample,h,dict(origin_us=origin,end_us=origin+7900000,initial_speed_mps=speed,task=task,lane_count=len(lanes),connector_count=len(connectors),valid_channel_fraction=float(sample['channel_valid'].mean()),neighbor_presence_fraction=float(sample['x'][:,[8,23,38,53,68]].mean()),initial_actor_vehicle_count=sum(v[0,5]>.5 for v in tracks.values()),raw_db_header_token=header[0][0],raw_time_span_us=[int(times[0]),int(times[-1])],source_velocity_frame='BODY',agent_velocity_frame='WORLD',center_lever_arm_used_m=d,official_center_oracle_pass=True,velocity_position_RMSE_declared_world=rmse_world,velocity_position_RMSE_wrong_unrotated=rmse_wrong_body,invalid_anchor_attempts=invalid,canonical83_version=str(sample['version']))
    finally:source.close()


def extract():
    check();p=read(OUT/'identity_roster.json');dest=OUT/'samples';dest.mkdir(exist_ok=True);sys.path.insert(0,str(ROOT.parent/'nuplan-devkit'))
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    def forbidden(*args,**kwargs):raise PermissionError('BDD-P0 forbids network/vehicle propagation')
    rows=[];started=time.monotonic()
    with patch.object(socket.socket,'connect',forbidden),patch.object(KinematicBicycleModel,'propagate_state',forbidden):
        for i,r in enumerate(p['samples']):
            file=dest/(r['session_id']+'.npz');receipt=dest/(r['session_id']+'.json')
            if receipt.exists():
                item=read(receipt)
                if item['protocol_sha256']!=sha(PLAN) or item['source_sha256']!=r['source_sha256']:raise ValueError('Resume identity changed')
                if item['status']=='COMPLETE' and sha(file)!=item['input_sha256']:raise ValueError('Certified feature cache changed')
            else:
                try:
                    api=get_maps_api(str(ROOT.parent/'nuplan/dataset/maps'),'nuplan-maps-v1.0',r['map_name'])
                    with source_db(r) as db:sample,h,details=prepare_window(db,r,api)
                    np.savez_compressed(file,**sample,H30=h);item=dict(**r,**details,status='COMPLETE',input_path=str(file.relative_to(ROOT)),input_sha256=sha(file),protocol_sha256=sha(PLAN))
                except ValueError as e:
                    # Input quality errors retained. No model/alarm output can cause replacement.
                    item=dict(**r,status='INVALID_INPUT',reason=str(e),protocol_sha256=sha(PLAN))
                write(receipt,item)
            rows.append(item);print('P0 real raw input',i+1,'/',len(p['samples']),r['role'],item['status'],flush=True)
            write(OUT/'materialization_progress.json',dict(processed=len(rows),total=len(p['samples']),complete=sum(v['status']=='COMPLETE' for v in rows),wall_seconds=time.monotonic()-started))
    # Audit absolute temporal overlap across named-day blocks before inference/any alarms.
    good=[r for r in rows if r['status']=='COMPLETE'];overlaps=[]
    for i,a in enumerate(good):
        for b in good[i+1:]:
            if a['session_id'].split('_veh-')[-1]==b['session_id'].split('_veh-')[-1] and max(a['origin_us'],b['origin_us'])<=min(a['end_us'],b['end_us']):overlaps.append([a['session_id'],b['session_id']])
    if overlaps:raise ValueError('Cross-block raw window temporal dependency: stop before inference '+repr(overlaps))
    write(OUT/'dataset_manifest.json',dict(protocol_sha256=sha(PLAN),roster_sha256=sha(OUT/'identity_roster.json'),rows=rows,counts_by_role=dict(Counter(r['role'] for r in good)),task_counts={role:dict(Counter(r['task'] for r in good if r['role']==role)) for role in sorted({r['role'] for r in good})},cross_block_time_overlap=overlaps,source_group_independence_scope='One per vehicle-name/date block; protected dates guarded +/-1 and actual windows nonoverlapping. Does not establish universal fleet IID.',new_vehicle_propagation=0,new_training=0,new_download_bytes=0,Primary_V_C=0))
    print('BDD-P0 actual raw dataset complete',len(good),'/',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['roster','extract']);a=p.parse_args();{'roster':roster,'extract':extract}[a.mode]()
