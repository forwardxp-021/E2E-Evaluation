#!/usr/bin/env python3
"""Prospective scoped DirectCoast metadata selection; no simulated ego trajectory."""
from __future__ import annotations
import csv,gzip,json,os,sys,sqlite3
from pathlib import Path
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.r1_b2_8_r3_prospective_selector import official_env
official_env()
import numpy as np
from shapely.geometry import Point,LineString,mapping
from shapely.ops import unary_union,substring
from tools.tsb_direct_coast_p2_q1 import sha,canon,read,write,ROOT
OUT=ROOT/'docs/stageR/e2e_scoped_direct_coast_v1'
MAPS=ROOT.parent/'nuplan/dataset/maps'


def csvrows(path):return list(csv.DictReader(Path(path).open()))


def catalog():
    roles={x['session_id']:x for x in csvrows(ROOT/'docs/stageR/s2r_role_aware_preflight/S2R_V_Eligibility_Census_v1.csv')}
    static=csvrows(ROOT/'docs/stageR/s2r_engineering_closure/S2R_Static_Eligibility_Census_v1.csv')
    old=read(ROOT/'docs/stageR/s2_preflight/S2_Preflight_Eligibility_Census_v1.json')['logs'];bylog={x['log_id']:x for x in old}
    oldb1={x['session_id'] for x in csvrows(ROOT/'docs/stageR/b1_tsb_qualification/B1_TSB_Qualification_Roster_v1.csv')}
    olddev={x['session_id'] for x in csvrows(ROOT/'docs/stageR/tsb_coast_p1r_entry/P1R_Development_Candidate_Roster_v1.csv')}
    p2={x['session_id'] for x in csvrows(ROOT/'docs/stageR/tsb_direct_coast_p2_q0/P2_Q0_Independent_Session_Roster_v1.csv')}
    runp2={x['session_id'] for x in csvrows(ROOT/'docs/stageR/tsb_direct_coast_p2_q1/P2_Q1_Exposure_Update_v1.csv')}
    for path in ['docs/stageR/b1_resume_execution_r2/B1_R2_Exposure_Ledger_Update_v1.csv','docs/stageR/tsb_direct_coast_p2_q1/P2_Q1_Exposure_Update_v1.csv']:
        for event in csvrows(ROOT/path):
            sid=event['session_id']
            if not roles[sid]['exposure_class'].startswith(('E3','E4')):roles[sid]['exposure_class']='E2_BENCHMARK_ENGINEERING'
    protected={sid for sid,r in roles.items() if r['exposure_class'].startswith('E5')}
    protected_fps={x['observed_db_fingerprint_sha256'] for x in old if x['session_id'] in protected}
    rule=read(OUT/'Development_Design_v1.json');rows=[]
    for s in static:
        sid=s['session_id'];role=roles[sid];log=bylog.get(s['representative_log_id']);reasons=[]
        if role['exposure_class'].startswith(('E3','E4')):reasons.append('EXCLUDED_E3_E4')
        if sid in protected or log and log['observed_db_fingerprint_sha256'] in protected_fps:reasons.append('PROTECTED_C_E5_SOURCE')
        if role['source_role_provenance_status']!='COMPLETE' or role['source_fingerprint_complete']!='true':reasons.append('UNKNOWN_PROVENANCE')
        if s['static_eligibility_status']!='STATIC_ELIGIBLE':reasons.append('OLD_STATIC_INELIGIBLE')
        if not log or log['session_id']!=sid or not log['frozen_fingerprint_match']:reasons.append('UNKNOWN_LOG')
        if not s.get('execution_initial_speed_mps') or float(s['execution_initial_speed_mps'])<5:reasons.append('INITIAL_SPEED_LT_5')
        rows.append(dict(session_id=sid,static=s,log=log,exposure_class=role['exposure_class'],policy_reasons=reasons,
            split_hash=__import__('hashlib').sha256((rule['split_salt']+'|'+sid).encode()).hexdigest()))
    new=sorted([x for x in rows if not x['policy_reasons'] and x['session_id'] not in oldb1|olddev|p2],key=lambda x:(x['split_hash'],x['session_id']))
    dev=olddev|runp2|{x['session_id'] for x in new[:10]}
    for x in rows:
        sid=x['session_id'];x['pool']='DEVELOPMENT' if sid in dev else 'QUALIFICATION' if sid not in oldb1|olddev|p2 else 'HISTORICAL_EXCLUDED'
    return dict(rows=rows,development_SESSION=sorted(dev),new_development_SESSION=[x['session_id'] for x in new[:10]],protected_C_SESSION=sorted(protected),historical_excluded=sorted(oldb1|olddev|p2))


def source_spec(row):
    from tools.r1_prepare_official_technical_smoke_roster import _official_initial
    from tools.s2r_engineering_closure import precontext_payload
    from tools.s2r_production_executor import route_id,precontext_id
    from tools.tsb_direct_coast_p2_q0 import historical_db_fingerprint
    log=row['log'];db=Path(log['source_file']);shard=ROOT/'docs/stageR/s2_preflight'/log['shard']
    with gzip.open(shard,'rt') as f:data=json.load(f)
    idx=data['record_columns'].index('scenario_token');token=row['static']['representative_scenario_token']
    record=next(x for x in data['records'] if x[idx]==token);anchor=int(record[data['record_columns'].index('tag_anchor_timestamp_us')])
    initial,route=_official_initial(dict(db_path=str(db),scenario_token=token,timestamp=anchor))
    fingerprint=historical_db_fingerprint(db)
    if fingerprint!=log['observed_db_fingerprint_sha256']:raise ValueError('Frozen metadata fingerprint changed')
    base=dict(session_id=row['session_id'],log_id=log['log_id'],scenario_token=token,db_path=str(db),map_name=log['map_version'],source_fingerprint_sha256=fingerprint)
    rid=route_id(base['map_name'],route);pre=precontext_payload(base,initial,route,rid)
    spec=dict(**base,candidate='TSB_DIRECT_COAST_F2',benchmark_version='E2E_SCOPED_DIRECT_COAST_V1',role='B',exposure_class=row['exposure_class'],
        route_roadblock_ids=route,route_id=rid,precontext=pre,precontext_id=precontext_id(pre),start_timestamp_us=initial['official_simulation_initial_timestamp_us'])
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    types=initial['official_scenario_types'];kind=types[0] if types else 'unknown'
    scenario=NuPlanScenario(str(db.parent),str(db),token,anchor,kind,str(MAPS),'nuplan-maps-v1.0',base['map_name'],
        ScenarioExtractionInfo(kind,15.,-3. if types else 0.,.5),get_pacifica_parameters())
    from tools.s2r_production_executor import verify_scenario_precontext
    verify_scenario_precontext(scenario,spec)
    return spec,scenario,dict(db_path=str(db),db_sha256=sha(db),metadata_shard=str(shard.relative_to(ROOT)),metadata_shard_sha256=sha(shard))


def assess(row,rule):
    from tools.b1_native_route_infrastructure import build_native_route_reference_v1_2
    from tools.tsb_coast_p1r_c_clock import CoastCanonicalTimeControllerV1
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    spec,scenario,sources=source_spec(row);ego=scenario.get_ego_state_at_iteration(0);api=scenario.map_api
    clock=CoastCanonicalTimeControllerV1(scenario).require_mapping();domain=rule['physical_domain'];vp=ego.car_footprint.vehicle_parameters
    v=ego.dynamic_car_state.speed;a=ego.dynamic_car_state.rear_axle_acceleration_2d.x
    bound=v+.2*max(a,0.);distance=7.9*bound+vp.front_length
    current=dict(rear_axle=dict(x=ego.rear_axle.x,y=ego.rear_axle.y,heading=ego.rear_axle.heading),speed_mps=v,time_us=ego.time_us)
    route=build_native_route_reference_v1_2(api,spec['route_roadblock_ids'],current,distance+1e-6)
    line=LineString(route['reference_xy']);arc=route['current_route_arc_m'];clip=substring(line,max(0,arc-vp.rear_length),arc+distance)
    points=np.array([clip.interpolate(s).coords[0] for s in np.arange(0,clip.length,.5).tolist()+[clip.length]])
    headings=np.unwrap(np.arctan2(np.diff(points,axis=0)[:,1],np.diff(points,axis=0)[:,0]));h0=line.interpolate(min(line.length,arc+.1));hprev=line.interpolate(max(0,arc-.1))
    tangent=np.arctan2(h0.y-hprev.y,h0.x-hprev.x);herr=abs(np.arctan2(np.sin(ego.rear_axle.heading-tangent),np.cos(ego.rear_axle.heading-tangent)))
    offset=line.distance(Point(ego.rear_axle.x,ego.rear_axle.y));hrange=float(np.ptp(headings))
    evidence=dict(initial_speed_mps=v,initial_acceleration_mps2=a,initial_heading_error_rad=float(herr),initial_centerline_offset_m=offset,route_heading_range_rad=hrange,
        forward_coverage_m=distance,clock=clock,sources=sources,traffic_examined=False,minimum_actor_corridor_gap_m=None)
    reasons=[]
    if v<domain['min_initial_speed_mps']:reasons.append('SPEED_DOMAIN')
    if herr>domain['maximum_initial_heading_error_rad']:reasons.append('INITIAL_HEADING')
    if offset>domain['maximum_initial_centerline_offset_m']:reasons.append('INITIAL_OFFSET')
    if hrange>domain['maximum_route_heading_range_rad']:reasons.append('CURVED_ROUTE')
    corridor=clip.buffer(vp.width/2+domain['corridor_lateral_margin_beyond_half_vehicle_width_m'],cap_style=2)
    surfaces=api._get_vector_map_layer(Layer.DRIVABLE_AREA);local=surfaces[surfaces.geometry.intersects(corridor.envelope)]
    da=unary_union(list(local.geometry));coverage=da.covers(corridor) and corridor.covers(ego.car_footprint.geometry)
    evidence['corridor_drivable_covered']=coverage;evidence['corridor_outside_area_m2']=corridor.difference(da).area
    if not coverage:reasons.append('SWEPT_CORRIDOR_NOT_DRIVABLE')
    # Do not read raw future traffic at all when geometry already rules out the source.
    if not reasons:
        bytrack={};count=0
        for i in range(80):
            for obj in scenario.get_tracked_objects_at_iteration(i).tracked_objects:
                bytrack.setdefault(obj.track_token,[]).append(obj.box.geometry);count+=1
        hulls=[unary_union(polys).convex_hull for polys in bytrack.values()]
        mingap=min((corridor.distance(g) for g in hulls),default=None)
        conflicts=sum(corridor.intersects(g) for g in hulls)
        evidence.update(traffic_examined=True,object_samples=count,unique_tracks=len(bytrack),conflicting_actor_hulls=conflicts,minimum_actor_corridor_gap_m=mingap,
            corridor_geometry=mapping(corridor),offline_future_traffic_used=True)
        if conflicts:reasons.append('OFFLINE_ACTOR_CORRIDOR_CONFLICT')
    return dict(session_id=row['session_id'],status='ELIGIBLE' if not reasons else 'EXCLUDED',reasons=reasons,spec=spec,evidence=evidence)


def census(pool,rule_path):
    cat=read(OUT/'Pool_Split_v1.json');rule=read(rule_path);rows=[]
    dest=OUT/'evidence'/(pool+'_Applicability_v1.json')
    if dest.exists():raise FileExistsError('Immutable census already exists')
    for i,row in enumerate(cat['rows']):
        if row['policy_reasons'] or row['pool']!=pool:
            rows.append(dict(session_id=row['session_id'],status='POLICY_EXCLUDED',reasons=row['policy_reasons'] or ['OUTSIDE_'+pool+'_POOL'],sources_opened=False));continue
        try:
            item=assess(row,rule);item['sources_opened']=True
        except Exception as error:item=dict(session_id=row['session_id'],status='METADATA_BLOCKED',reasons=[type(error).__name__+': '+str(error)],sources_opened=True)
        rows.append(item);print(pool,i+1,'/248',row['session_id'],item['status'],item['reasons'],flush=True)
        write(OUT/'evidence'/(pool+'_Applicability_Progress.json'),dict(rows=rows,rule_sha256=sha(rule_path)))
    write(dest,dict(pool=pool,rule_sha256=sha(rule_path),rows=rows,counts=dict(Counter(x['status'] for x in rows))),exclusive=True)
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['catalog','development','qualification']);a=p.parse_args()
    if a.mode=='catalog':write(OUT/'Pool_Split_v1.json',catalog(),exclusive=True)
    else:census(a.mode.upper(),OUT/('Development_Design_v1.json' if a.mode=='development' else 'Frozen_Qualification_Protocol_v1.json'))
