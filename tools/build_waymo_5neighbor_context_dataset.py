#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, csv, hashlib, json, shutil, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

from tools.interaction_context_features import aggregate_interaction_features
from tools.lane_aware_assignment import SLOT_NAMES, assign_neighbors_lane_aware
from tools.waymo_lane_utils import extract_lane_polylines, find_best_lane_for_agent

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--waymo_dir', type=str, default=None)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--max_files', type=int, default=None)
    p.add_argument('--max_scenarios', type=int, default=None)
    p.add_argument('--max_agents_per_scenario', type=int, default=64)
    p.add_argument('--window_len', type=int, default=80)
    p.add_argument('--stride', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--min_valid_ratio', type=float, default=0.8)
    p.add_argument('--min_speed', type=float, default=1.0)
    p.add_argument('--agent_types', type=str, default='vehicle')
    p.add_argument('--assignment_mode', type=str, default='lane_aware_with_geometric_fallback', choices=['lane_aware_only','lane_aware_with_geometric_fallback','geometric_only'])
    p.add_argument('--lane_max_lateral_distance', type=float, default=3.0)
    p.add_argument('--lane_max_heading_diff_deg', type=float, default=45.0)
    p.add_argument('--adjacent_lane_min_offset', type=float, default=2.0)
    p.add_argument('--adjacent_lane_max_offset', type=float, default=5.5)
    p.add_argument('--adjacent_lane_max_heading_diff_deg', type=float, default=35.0)
    p.add_argument('--ttc_cap', type=float, default=999.0)
    p.add_argument('--thw_cap', type=float, default=999.0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--lane_search_radius', type=float, default=20.0)
    p.add_argument('--lane_topk_candidates', type=int, default=32)
    p.add_argument('--lane_projection_max_candidates', type=int, default=32)
    p.add_argument('--lane_projection_timeout_warning_sec', type=float, default=5.0)
    p.add_argument('--disable_lane_spatial_index', action='store_true')
    p.add_argument('--front_max_distance', type=float, default=120.0)
    p.add_argument('--side_front_max_distance', type=float, default=80.0)
    p.add_argument('--side_rear_max_distance', type=float, default=120.0)
    p.add_argument('--lane_lateral_tolerance', type=float, default=2.0)
    p.add_argument('--slot_heading_diff_deg', type=float, default=45.0)
    p.add_argument('--static_speed_threshold', type=float, default=0.5)
    p.add_argument('--drop_if_no_lane_map', action='store_true')
    p.add_argument('--drop_if_ego_lane_missing', action='store_true')
    p.add_argument('--drop_if_lane_context_bad', action='store_true')
    p.add_argument('--drop_if_lane_context_ambiguous', action='store_true')
    p.add_argument('--allow_empty', action='store_true')
    p.add_argument('--strict_summary_validation', action='store_true')
    return p.parse_args()

def split_of_sid(sid):
    h = int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'train' if h < 0.8 else ('val' if h < 0.9 else 'test')

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def sanitize_track_window(window, dt, name, max_nan_ratio, preserve_valid_mask=True):
    raw = window.astype(np.float64, copy=True); T = raw.shape[0]
    valid = np.isfinite(raw[:, :4]).all(axis=1) & (raw[:, 5] > 0.5)
    heading_valid = np.isfinite(raw[:, 4])
    diag = {'nan_count_raw': int(np.isnan(raw[:, :5]).sum()), 'inf_count_raw': int(np.isinf(raw[:, :5]).sum()), 'repaired_frame_count': 0, 'first_valid_index': -1, 'heading_fallback_count': 0, 'dropped_reason': None}
    if float(np.mean(valid)) < (1.0 - max_nan_ratio):
        diag['dropped_reason'] = 'valid_ratio_below_threshold'; return None, valid.astype(np.float32), diag
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        diag['dropped_reason'] = 'no_valid_frames'; return None, valid.astype(np.float32), diag
    diag['first_valid_index'] = int(idx[0])
    clean = raw.copy(); t = np.arange(T)
    for c in range(4):
        cv = np.isfinite(raw[:, c]) & valid
        if np.sum(cv) == 0: diag['dropped_reason'] = f'no_finite_col_{c}'; return None, valid.astype(np.float32), diag
        clean[:, c] = np.interp(t, t[cv], raw[cv, c])
    if np.sum(heading_valid) >= 2:
        clean[:, 4] = wrap(np.interp(t, t[heading_valid], np.unwrap(raw[heading_valid, 4])))
    elif np.sum(heading_valid) == 1:
        clean[:, 4] = wrap(np.full(T, raw[heading_valid, 4][0]))
    else:
        clean[:, 4] = wrap(np.arctan2(clean[:, 3], clean[:, 2])); diag['heading_fallback_count'] += T
    if np.any(~heading_valid):
        proxy = wrap(np.arctan2(clean[:, 3], clean[:, 2])); clean[~heading_valid, 4] = proxy[~heading_valid]; diag['heading_fallback_count'] += int(np.sum(~heading_valid))
    clean[:, 5] = raw[:, 5] if preserve_valid_mask else valid.astype(np.float32)
    diag['repaired_frame_count'] = int(np.sum(~np.isfinite(raw[:, :5]).all(axis=1)))
    if not np.isfinite(clean[:, :5]).all(): diag['dropped_reason'] = 'sanitize_nonfinite_remains'; return None, valid.astype(np.float32), diag
    return clean.astype(np.float32), valid.astype(np.float32), diag

def localize(xy, origin, heading):
    if not np.isfinite(origin).all() or not np.isfinite(heading) or not np.isfinite(xy).all():
        raise ValueError('localize received non-finite input')
    d = xy - origin[None, :]
    c, s = np.cos(-heading), np.sin(-heading)
    return np.stack([d[:,0]*c - d[:,1]*s, d[:,0]*s + d[:,1]*c], axis=1)

LANE_DEBUG_FIELDS = [
    "scenario_id", "target_agent_id", "start", "slot_name", "assignment_method", "neighbor_id",
    "fallback_used", "fallback_reason", "ego_lane_id", "slot_lane_id", "neighbor_lane_id",
    "ego_s", "neighbor_s", "delta_s", "ego_l", "neighbor_l", "projection_distance",
    "candidate_lateral_offset", "candidate_heading_diff", "neighbor_speed", "neighbor_is_static",
    "distance_threshold_used", "lane_lateral_tolerance", "slot_heading_diff_threshold", "distance",
    "longitudinal_gap", "lateral_gap", "rejection_reason"
]

def normalize_debug_row(row):
    return {k: row.get(k, "") for k in LANE_DEBUG_FIELDS}

def main():
    a = parse_args(); out = Path(a.out_dir)
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t_global=time.perf_counter()
    timing=defaultdict(float)
    scenarios = []; rng = np.random.default_rng(7)
    if a.smoke_test:
        for s in range(3):
            T = max(a.window_len + 10, 100); tracks = {}
            for k in range(8):
                x=np.linspace(0,90,T)+k*6; y=np.full(T,(k-3)*2.0); vx,vy=np.gradient(x,a.dt),np.gradient(y,a.dt); h=np.arctan2(vy,vx); valid=np.ones(T,np.float32)
                arr=np.stack([x,y,vx,vy,h,valid],1).astype(np.float32); arr[:,:5]+=rng.normal(0,0.01,(T,5)).astype(np.float32)
                bad=rng.choice(np.arange(5,T-5),size=3,replace=False); arr[bad,:5]=np.nan; arr[bad,5]=0.0; tracks[f'veh_{k}']=arr
            lane_infos={
                'lane_cur': {'lane_id':'lane_cur','centerline_xy':np.stack([np.linspace(0,120,121),np.zeros(121)],1).astype(np.float32),'seg_heading':np.zeros(120), 'seg_len':np.ones(120), 's_prefix':np.arange(121,dtype=np.float32), 'left_neighbor_lane_ids':['lane_left'],'right_neighbor_lane_ids':['lane_right']},
                'lane_left': {'lane_id':'lane_left','centerline_xy':np.stack([np.linspace(0,120,121),np.full(121,3.5)],1).astype(np.float32),'seg_heading':np.zeros(120), 'seg_len':np.ones(120), 's_prefix':np.arange(121,dtype=np.float32), 'left_neighbor_lane_ids':[],'right_neighbor_lane_ids':['lane_cur']},
                'lane_right': {'lane_id':'lane_right','centerline_xy':np.stack([np.linspace(0,120,121),np.full(121,-3.5)],1).astype(np.float32),'seg_heading':np.zeros(120), 'seg_len':np.ones(120), 's_prefix':np.arange(121,dtype=np.float32), 'left_neighbor_lane_ids':['lane_cur'],'right_neighbor_lane_ids':[]}
            }
            from tools.waymo_lane_utils import LaneInfo, _build_lane_geom
            lane_infos={k:LaneInfo(**v,seg_start_xy=_build_lane_geom(v['centerline_xy'])[3],seg_vec_xy=_build_lane_geom(v['centerline_xy'])[4],seg_den=_build_lane_geom(v['centerline_xy'])[5],bbox_min_xy=_build_lane_geom(v['centerline_xy'])[6],bbox_max_xy=_build_lane_geom(v['centerline_xy'])[7],bbox_center_xy=_build_lane_geom(v['centerline_xy'])[8],entry_lane_ids=[],exit_lane_ids=[],lane_type='driving',topology_source='proto_topology') for k,v in lane_infos.items()}
            scenarios.append((f'smoke_{s}', tracks, lane_infos))
    else:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
        files=sorted([str(p) for p in Path(a.waymo_dir).glob('*.tfrecord*')])
        if a.max_files: files=files[:a.max_files]
        for fp in tqdm(files, desc="Processing TFRecord files"):
            t0=time.perf_counter(); ds=tf.data.TFRecordDataset(fp)
            timing["load_tfrecord"] += time.perf_counter()-t0
            for rec in ds:
                sc=scenario_pb2.Scenario(); sc.ParseFromString(bytes(rec.numpy())); tracks={}
                for tr in sc.tracks:
                    if a.agent_types=='vehicle' and tr.object_type!=1: continue
                    arr=[[st.center_x,st.center_y,st.velocity_x,st.velocity_y,st.heading,1.0] if st.valid else [np.nan]*5+[0.0] for st in tr.states]
                    tracks[str(tr.id)]=np.asarray(arr,np.float32)
                scenarios.append((sc.scenario_id,tracks, extract_lane_polylines(sc)))
                if a.max_scenarios and len(scenarios)>=a.max_scenarios: break
            if a.max_scenarios and len(scenarios)>=a.max_scenarios: break

    ego_feats=["ego_x_local","ego_y_local","ego_vx_local","ego_vy_local","ego_heading_local","ego_speed","ego_accel","ego_yaw_rate"]
    nbr_feats=["valid_mask","dx_ego","dy_ego","rel_vx_ego","rel_vy_ego","distance","longitudinal_gap","lateral_gap","closing_rate","ttc_proxy","thw_proxy","neighbor_speed","neighbor_accel","neighbor_heading_rel","neighbor_yaw_rate"]
    ego_seq=[]; nbr_seq=[]; ctx=[]; cmask=[]; cmask_win=[]; slot_ids=[]; meta=[]; splits=[]; inter=[]; debug_rows=[]; cnt=defaultdict(int); slot_valid_counts=defaultdict(int)
    assignment_method_counts_by_slot={s:{'lane_aware':0,'geometric_fallback':0,'empty':0,'sanitize_failed':0} for s in SLOT_NAMES}
    lane_context_quality_reason_counts = Counter()
    slot_rejection_reason_counts={s:defaultdict(int) for s in SLOT_NAMES}
    scenario_bar=tqdm(scenarios, desc="Processing scenarios")
    for sid,tracks,lane_infos in scenario_bar:
        scenario_bar.set_postfix({"scenario_id":str(sid)[:16],"kept":cnt["kept"]})
        sp=split_of_sid(sid); ids=list(tracks.keys())[:a.max_agents_per_scenario]; cnt['targets']+=len(ids)
        for aid in tqdm(ids, desc="Processing target agents", leave=False):
            tr=tracks[aid]; T=len(tr)
            for st in tqdm(range(0,max(0,T-a.window_len+1),a.stride), desc="Building windows", leave=False):
                cnt['windows_total']+=1
                ew,ego_valid,ediag=sanitize_track_window(tr[st:st+a.window_len],a.dt,f'{sid}:{aid}:{st}:ego',1.0-a.min_valid_ratio)
                cnt['trajectory_nan_count_raw']+=ediag['nan_count_raw']; cnt['trajectory_inf_count_raw']+=ediag['inf_count_raw']
                if ew is None: cnt['f_invalid']+=1; cnt['ego_windows_dropped_sanitize_failed']+=1; continue
                if ediag['repaired_frame_count']>0: cnt['ego_windows_repaired']+=1
                speed=np.hypot(ew[:,2],ew[:,3]);
                if np.nanmean(speed)<a.min_speed: cnt['f_static']+=1; continue
                idx=np.flatnonzero((ego_valid>0.5)&(speed>1e-3)); ref=int(idx[0]) if len(idx)>0 else int(max(0,ediag['first_valid_index']))
                cnt['origin_frame_not_zero_count']+=int(ref!=0); origin=ew[ref,:2]
                if np.isfinite(ew[ref,4]): base_h=float(ew[ref,4]); cnt['base_heading_raw_count']+=1
                else: base_h=float(np.arctan2(ew[ref,3],ew[ref,2])); cnt['base_heading_velocity_fallback_count']+=1
                try:
                    xy_local=localize(ew[:,:2],origin,base_h); v_local=localize(ew[:,2:4],np.array([0.0,0.0],np.float32),base_h)
                except ValueError:
                    cnt['ego_windows_dropped_sanitize_failed']+=1; continue
                heading=np.where(np.isfinite(ew[:,4]),ew[:,4],np.arctan2(ew[:,3],ew[:,2])); cnt['heading_raw']+=int(np.isfinite(ew[:,4]).sum()); cnt['heading_total']+=len(heading)
                cnt['heading_fallback']+=int((~np.isfinite(ew[:,4])).sum()); yaw=wrap(np.diff(heading,prepend=heading[0]))/max(a.dt,1e-6); accel=np.diff(speed,prepend=speed[0])/max(a.dt,1e-6)
                ego=np.stack([xy_local[:,0],xy_local[:,1],v_local[:,0],v_local[:,1],wrap(heading-base_h),speed,accel,yaw],1).astype(np.float32)
                cnt['trajectory_nan_count_after_sanitize']+=int(np.isnan(ew[:,:5]).sum()); cnt['trajectory_inf_count_after_sanitize']+=int(np.isinf(ew[:,:5]).sum())
                candidates={nid:ntr[st:st+a.window_len] for nid,ntr in tracks.items() if nid!=aid and len(ntr)>=st+a.window_len}
                ego_state={'x':float(origin[0]),'y':float(origin[1]),'heading':float(base_h),'velocity_x':float(ew[ref,2]),'velocity_y':float(ew[ref,3])}
                cand_states={k:{'x':float(v[0,0]),'y':float(v[0,1]),'heading':float(v[0,4]) if np.isfinite(v[0,4]) else np.nan,'velocity_x':float(v[0,2]) if np.isfinite(v[0,2]) else 0.0,'velocity_y':float(v[0,3]) if np.isfinite(v[0,3]) else 0.0,'speed':float(np.hypot(v[0,2],v[0,3])) if np.isfinite(v[0,2:4]).all() else 0.0,'valid':bool(v[0,5]>0.5)} for k,v in candidates.items() if np.isfinite(v[0,:2]).all()}
                proj_cache={}
                t_lp=time.perf_counter()
                for ck,cst in cand_states.items():
                    key=(str(ck),int(st))
                    if key in proj_cache: cnt['lane_projection_cache_hits']+=1; continue
                    cnt['lane_projection_cache_misses']+=1
                    pbest,_,cand_n=find_best_lane_for_agent(np.array([cst['x'],cst['y']]), cst.get('heading', np.nan), lane_infos, a.lane_max_lateral_distance, np.deg2rad(a.lane_max_heading_diff_deg), a.lane_search_radius, min(a.lane_topk_candidates,a.lane_projection_max_candidates), a.disable_lane_spatial_index)
                    cnt['lane_projection_candidate_total']+=cand_n; cnt['lane_projection_candidate_max']=max(cnt['lane_projection_candidate_max'],cand_n); cnt['lane_projection_attempt_count']+=1
                    if pbest is not None: cnt['lane_projection_success_count']+=1; proj_cache[key]=pbest
                    else: cnt['lane_projection_failure_count']+=1
                timing['lane_projection'] += time.perf_counter()-t_lp
                cproj={k:v for (k,_),v in proj_cache.items()}
                t_as=time.perf_counter()
                assign=assign_neighbors_lane_aware(ego_state,cand_states,lane_infos=lane_infos,assignment_mode=a.assignment_mode,config={'lane_max_lateral_distance':a.lane_max_lateral_distance,'lane_max_heading_diff_deg':a.lane_max_heading_diff_deg,'adjacent_lane_min_offset':a.adjacent_lane_min_offset,'adjacent_lane_max_offset':a.adjacent_lane_max_offset,'adjacent_lane_max_heading_diff_deg':a.adjacent_lane_max_heading_diff_deg,'lane_search_radius':a.lane_search_radius,'lane_topk_candidates':a.lane_topk_candidates,'disable_lane_spatial_index':a.disable_lane_spatial_index,'front_max_distance':a.front_max_distance,'side_front_max_distance':a.side_front_max_distance,'side_rear_max_distance':a.side_rear_max_distance,'lane_lateral_tolerance':a.lane_lateral_tolerance,'slot_heading_diff_deg':a.slot_heading_diff_deg,'static_speed_threshold':a.static_speed_threshold}, candidate_projections=cproj)
                timing['assignment'] += time.perf_counter()-t_as
                cnt['lane_assignment_success_count_pre_filter']+=int(assign.lane_assignment_available)
                cnt['fallback_assignment_count_pre_filter']+=int(assign.fallback_assignment_used)
                cnt['geometric_only_assignment_count']+=int(a.assignment_mode=='geometric_only')
                cnt['current_lane_found_count_pre_filter']+=int(bool(assign.current_lane_id))
                cnt['left_lane_found_count_pre_filter']+=int(bool(assign.left_lane_id))
                cnt['right_lane_found_count_pre_filter']+=int(bool(assign.right_lane_id))
                cnt[f'adjacency_source::{assign.adjacency_source}']+=1
                nbr=np.zeros((5,a.window_len,len(nbr_feats)),np.float32); sidrow=[]; sample_debug_rows=[]
                for si,sn in enumerate(SLOT_NAMES):
                    nid=assign.slot_to_agent.get(sn,''); sidrow.append(nid if nid else '-1')
                    if not nid: continue
                    nw,n_valid,ndiag=sanitize_track_window(candidates[nid],a.dt,f'{sid}:{aid}:{st}:{nid}',1.0-a.min_valid_ratio)
                    cnt['trajectory_nan_count_raw']+=ndiag['nan_count_raw']; cnt['trajectory_inf_count_raw']+=ndiag['inf_count_raw']
                    if nw is None:
                        cnt['neighbor_windows_dropped_sanitize_failed']+=1
                        sample_debug_rows.append({'scenario_id':str(sid),'target_agent_id':str(aid),'start':int(st),'slot_name':sn,'assignment_method':'sanitize_failed','neighbor_id':str(nid),'fallback_used':True,'fallback_reason':ndiag['dropped_reason'],'distance':'','longitudinal_gap':'','lateral_gap':'','rejection_reason':'sanitize_failed'})
                        continue
                    if ndiag['repaired_frame_count']>0: cnt['neighbor_windows_repaired']+=1
                    cnt['trajectory_nan_count_after_sanitize']+=int(np.isnan(nw[:,:5]).sum()); cnt['trajectory_inf_count_after_sanitize']+=int(np.isinf(nw[:,:5]).sum())
                    n_speed=np.hypot(nw[:,2],nw[:,3]); n_acc=np.diff(n_speed,prepend=n_speed[0])/max(a.dt,1e-6); n_heading=np.where(np.isfinite(nw[:,4]),nw[:,4],np.arctan2(nw[:,3],nw[:,2])); n_yaw=wrap(np.diff(n_heading,prepend=n_heading[0]))/max(a.dt,1e-6)
                    for t in range(a.window_len):
                        if n_valid[t]<=0.5: continue
                        dxy=localize(nw[t:t+1,:2],ew[t,:2],float(heading[t]))[0]; rv=localize((nw[t:t+1,2:4]-ew[t:t+1,2:4]),np.array([0.0,0.0]),float(heading[t]))[0]
                        dist=max(0.0,float(np.hypot(dxy[0],dxy[1]))); closing=float(v_local[t,0]-rv[0]) if np.isfinite(v_local[t,0]-rv[0]) else 0.0
                        ttc=min((dist/max(closing,1e-3)) if closing>1e-3 else a.ttc_cap,a.ttc_cap); thw=min(dist/max(float(speed[t]),1e-3),a.thw_cap)
                        nbr[si,t]=[1,dxy[0],dxy[1],rv[0],rv[1],dist,dxy[0],dxy[1],closing,ttc,thw,n_speed[t],n_acc[t],wrap(n_heading[t]-heading[t]),n_yaw[t]]
                    nbr[si]=np.nan_to_num(nbr[si],nan=0.0,posinf=a.ttc_cap,neginf=-a.ttc_cap)
                for d in assign.per_slot_debug:
                    d=dict(d)
                    d.update(dict(scenario_id=str(sid),target_agent_id=str(aid),start=int(st)))
                    sample_debug_rows.append(d)
                if assign.slot_rejection_reason_counts:
                    for slot, reasons in assign.slot_rejection_reason_counts.items():
                        for reason, rc in reasons.items():
                            slot_rejection_reason_counts[slot][reason] += int(rc)
                context=np.concatenate([ego,nbr.reshape(a.window_len,-1)],1)
                lane_ctx=assign.lane_context_quality
                lane_ctx_reasons = assign.lane_context_quality_reasons or []
                drop_reasons=[]
                if a.drop_if_no_lane_map and not lane_infos:
                    cnt['n_windows_dropped_no_lane_map']+=1; drop_reasons.append('no_lane_map')
                if a.drop_if_ego_lane_missing and not assign.current_lane_id:
                    cnt['n_windows_dropped_ego_lane_missing']+=1; drop_reasons.append('ego_lane_missing')
                if a.drop_if_lane_context_bad and lane_ctx in ('bad','fallback'):
                    cnt['n_windows_dropped_bad_lane_context']+=1; drop_reasons.append('bad_lane_context')
                if a.drop_if_lane_context_ambiguous and lane_ctx == 'ambiguous_intersection':
                    cnt['n_windows_dropped_lane_context_ambiguous'] += 1; drop_reasons.append('lane_context_ambiguous')
                if drop_reasons:
                    cnt['n_windows_dropped_clean_filter_total'] += 1
                    continue
                cnt['lane_assignment_success_count_kept']+=int(assign.lane_assignment_available)
                cnt['fallback_assignment_count_kept']+=int(assign.fallback_assignment_used)
                cnt['current_lane_found_count_kept']+=int(bool(assign.current_lane_id))
                cnt['left_lane_found_count_kept']+=int(bool(assign.left_lane_id))
                cnt['right_lane_found_count_kept']+=int(bool(assign.right_lane_id))
                eidx=len(ego_seq)
                ego_seq.append(ego); nbr_seq.append(nbr); ctx.append(context); cmask.append((nbr[:,:,0]>0.5).T); cmask_win.append(np.max(nbr[:,:,0],axis=1)>0.5); slot_ids.append(sidrow); splits.append(sp)
                cnt[f'lane_context_quality::{lane_ctx}']+=1
                for reason in lane_ctx_reasons:
                    lane_context_quality_reason_counts[reason] += 1
                meta.append((eidx,str(sid),str(aid),int(st),int(a.window_len),sp,a.assignment_mode,assign.lane_assignment_available,assign.fallback_assignment_used,lane_ctx)); inter_feat, inter_names = aggregate_interaction_features(ego,nbr,a.dt); inter.append(np.nan_to_num(inter_feat,nan=0.0,posinf=1e6,neginf=-1e6))
                for sn in SLOT_NAMES:
                    row = next((r for r in sample_debug_rows if r.get('slot_name')==sn and r.get('assignment_method')!='sanitize_failed'), None)
                    method='sanitize_failed' if any(r.get('slot_name')==sn and r.get('assignment_method')=='sanitize_failed' for r in sample_debug_rows) else (row.get('assignment_method') if row else 'empty')
                    if method not in assignment_method_counts_by_slot[sn]:
                        method='empty'
                    assignment_method_counts_by_slot[sn][method]+=1
                    slot_valid_counts[sn]+=int(np.sum(nbr[SLOT_NAMES.index(sn),:,0]>0.5))
                debug_rows.extend(sample_debug_rows)
                cnt['kept']+=1

    n=len(ctx)
    lengths={'ego_seq':len(ego_seq),'nbr_seq':len(nbr_seq),'ctx':len(ctx),'cmask':len(cmask),'cmask_win':len(cmask_win),'slot_ids':len(slot_ids),'splits':len(splits),'meta':len(meta),'inter':len(inter),'kept':cnt['kept']}
    if any(v!=n for v in lengths.values()):
        raise RuntimeError(f"Row alignment assertion failed: {lengths}")
    ego_arr=np.asarray(ego_seq,np.float32); nbr_arr=np.asarray(nbr_seq,np.float32); ctx_arr=np.asarray(ctx,np.float32); cmask_arr=np.asarray(cmask,np.float32); cmw=np.asarray(cmask_win,np.float32)
    inter_raw=np.asarray(inter,np.float32); split_arr=np.asarray(splits,dtype=object)
    if len(inter_raw) != len(split_arr):
        raise RuntimeError(f"Row mismatch: inter_raw={len(inter_raw)}, split={len(split_arr)}, ctx={len(ctx_arr)}")
    if n == 0:
        summary={'warnings':['No windows kept after clean lane filtering.'],'n_windows_kept':0,'n_windows_total':cnt['windows_total']}
        (out/'build_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        if not a.allow_empty:
            raise RuntimeError('No windows kept after clean lane filtering.')
    train=(split_arr=='train')
    mu=np.mean(inter_raw[train],axis=0) if (n>0 and np.any(train)) else (np.mean(inter_raw,axis=0) if n>0 else np.zeros((1,),dtype=np.float32)); sd=np.std(inter_raw[train],axis=0) if (n>0 and np.any(train)) else (np.std(inter_raw,axis=0) if n>0 else np.ones((1,),dtype=np.float32))
    sd=np.where(sd<1e-6,1e-6,sd); inter_std=((inter_raw-mu)/sd).astype(np.float32)
    np.save(out/'ego_seq.npy',ego_arr); np.save(out/'neighbor_seq.npy',nbr_arr); np.save(out/'context_traj.npy',ctx_arr); np.save(out/'context_mask.npy',cmask_arr); np.save(out/'context_mask_window.npy',cmw); np.save(out/'neighbor_slot_ids.npy',np.asarray(slot_ids,dtype=object))
    meta_dtype=np.dtype([('row_index','i4'),('scenario_id','O'),('target_agent_id','O'),('start','i4'),('window_len','i4'),('split','O'),('assignment_mode','O'),('lane_assignment_success','?'),('fallback_used','?'),('lane_context_quality','O')])
    np.save(out/'meta.npy',np.array(meta,dtype=meta_dtype)); np.save(out/'split.npy',split_arr); np.save(out/'interaction_feat_style_raw.npy',inter_raw); np.save(out/'interaction_feat_style.npy',inter_std)
    (out/'ego_feature_names.json').write_text(json.dumps(ego_feats,indent=2),encoding='utf-8'); (out/'neighbor_feature_names.json').write_text(json.dumps(nbr_feats,indent=2),encoding='utf-8'); (out/'neighbor_slot_names.json').write_text(json.dumps(SLOT_NAMES,indent=2),encoding='utf-8')
    (out/'context_feature_names.json').write_text(json.dumps(ego_feats+[f'{s}.{n}' for s in SLOT_NAMES for n in nbr_feats],indent=2),encoding='utf-8'); (out/'interaction_feature_names.json').write_text(json.dumps(inter_names,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'interaction_feature_standardization.json').write_text(json.dumps({'mean':mu.tolist(),'std':sd.tolist(),'feature_names':inter_names,'train_count':int(np.sum(train)),'clip_value':None},indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'neighbor_slot_valid_ratio.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['slot_name','valid_ratio','valid_count','total_count']); w.writeheader(); total=max(1,cnt['kept']*a.window_len)
        for s in SLOT_NAMES: w.writerow({'slot_name':s,'valid_ratio':slot_valid_counts[s]/total,'valid_count':slot_valid_counts[s],'total_count':total})
    with (out/'lane_assignment_debug.csv').open('w',newline='',encoding='utf-8') as f:
        fn=['scenario_id','target_agent_id','start','slot_name','assignment_method','neighbor_id','fallback_used','fallback_reason','ego_lane_id','slot_lane_id','neighbor_lane_id','ego_s','neighbor_s','delta_s','ego_l','neighbor_l','projection_distance','candidate_lateral_offset','candidate_heading_diff','neighbor_speed','neighbor_is_static','distance_threshold_used','lane_lateral_tolerance','slot_heading_diff_threshold']
        w=csv.DictWriter(f,fieldnames=LANE_DEBUG_FIELDS,extrasaction='ignore'); w.writeheader(); w.writerows(normalize_debug_row(r) for r in debug_rows)
    def finite_report(name, arr):
        finite=np.isfinite(arr)
        if finite.all(): return True
        bad=np.argwhere(~finite)[:20].tolist(); fv=arr[finite]
        rep={'name':name,'shape':list(arr.shape),'nan_count':int(np.isnan(arr).sum()),'inf_count':int(np.isinf(arr).sum()),'bad_indices_first20':bad,'min_finite':float(np.min(fv)) if fv.size else None,'max_finite':float(np.max(fv)) if fv.size else None}
        (out/f'nonfinite_debug_{name}.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf-8'); return False
    bad=[k for k,v in {'ego_seq':ego_arr,'neighbor_seq':nbr_arr,'context_traj':ctx_arr,'interaction_feat_style_raw':inter_raw,'interaction_feat_style':inter_std}.items() if not finite_report(k,v)]
    cnt['nonfinite_output_detected']=int(len(bad)>0)
    if bad:
        (out/'nonfinite_debug_summary.json').write_text(json.dumps({'bad_arrays':bad},indent=2),encoding='utf-8')
        raise RuntimeError('Non-finite values remain in context_traj.npy; see nonfinite_debug_context_traj.json')

    slot_ratio={s:float(slot_valid_counts[s]/max(1,cnt['kept']*a.window_len)) for s in SLOT_NAMES}
    adj_counts={k.split('::',1)[1]:v for k,v in cnt.items() if k.startswith('adjacency_source::')}
    lane_context_quality_counts={k.split('::',1)[1]:v for k,v in cnt.items() if k.startswith('lane_context_quality::')}
    lane_map_avail=sum(1 for _,_,li in scenarios if li)
    lane_map_missing=len(scenarios)-lane_map_avail
    lane_assignment_success_rate=float(cnt['lane_assignment_success_count_kept']/max(1,cnt['kept']))
    fallback_rate=float(cnt['fallback_assignment_count_kept']/max(1,cnt['kept']))
    current_lane_found_rate=float(cnt['current_lane_found_count_kept']/max(1,cnt['kept']))
    left_lane_found_rate=float(cnt['left_lane_found_count_kept']/max(1,cnt['kept']))
    right_lane_found_rate=float(cnt['right_lane_found_count_kept']/max(1,cnt['kept']))
    lane_assignment_success_rate_pre_filter=float(cnt['lane_assignment_success_count_pre_filter']/max(1,cnt['windows_total']))
    current_lane_found_rate_pre_filter=float(cnt['current_lane_found_count_pre_filter']/max(1,cnt['windows_total']))
    empty_slot_count_by_slot = {s:int(assignment_method_counts_by_slot[s].get('empty', 0)) for s in SLOT_NAMES}
    empty_slot_ratio_by_slot = {s:float(empty_slot_count_by_slot[s]/max(1,cnt['kept'])) for s in SLOT_NAMES}
    summary={
        'dataset_type':'waymo_5neighbor_context','n_files_processed':0 if a.smoke_test else (a.max_files or -1),
        'n_scenarios_processed':len(scenarios),'n_target_agents_considered':cnt['targets'],'n_windows_total':cnt['windows_total'],'n_windows_kept':cnt['kept'],
        'n_windows_filtered_static':cnt['f_static'],'n_windows_filtered_invalid':cnt['f_invalid'],'n_windows_dropped_no_lane_map':cnt['n_windows_dropped_no_lane_map'],'n_windows_dropped_ego_lane_missing':cnt['n_windows_dropped_ego_lane_missing'],'n_windows_dropped_bad_lane_context':cnt['n_windows_dropped_bad_lane_context'],'n_windows_dropped_lane_context_ambiguous':cnt['n_windows_dropped_lane_context_ambiguous'],'n_windows_dropped_clean_filter_total':cnt['n_windows_dropped_clean_filter_total'],'split_counts':dict(Counter(splits)),'window_len':a.window_len,'dt':a.dt,
        'slot_valid_ratio':slot_ratio,'lane_map_available_scenarios':lane_map_avail,'lane_map_missing_scenarios':lane_map_missing,
        'lane_projection_attempt_count':cnt['lane_projection_attempt_count'],'lane_projection_success_count':cnt['lane_projection_success_count'],'lane_projection_success_rate':float(cnt['lane_projection_success_count']/max(1,cnt['lane_projection_attempt_count'])),'lane_projection_failure_count':cnt['lane_projection_failure_count'],'lane_projection_avg_candidate_lanes':float(cnt['lane_projection_candidate_total']/max(1,cnt['lane_projection_attempt_count'])),'lane_projection_max_candidate_lanes':cnt['lane_projection_candidate_max'],'lane_projection_cache_hits':cnt['lane_projection_cache_hits'],'lane_projection_cache_misses':cnt['lane_projection_cache_misses'],'lane_spatial_index_enabled':bool(not a.disable_lane_spatial_index),'lane_search_radius':a.lane_search_radius,'lane_topk_candidates':a.lane_topk_candidates,
        'lane_assignment_success_count_pre_filter':cnt['lane_assignment_success_count_pre_filter'],
        'current_lane_found_count_pre_filter':cnt['current_lane_found_count_pre_filter'],
        'left_lane_found_count_pre_filter':cnt['left_lane_found_count_pre_filter'],
        'right_lane_found_count_pre_filter':cnt['right_lane_found_count_pre_filter'],
        'fallback_assignment_count_pre_filter':cnt['fallback_assignment_count_pre_filter'],
        'lane_assignment_success_count_kept':cnt['lane_assignment_success_count_kept'],
        'current_lane_found_count_kept':cnt['current_lane_found_count_kept'],
        'left_lane_found_count_kept':cnt['left_lane_found_count_kept'],
        'right_lane_found_count_kept':cnt['right_lane_found_count_kept'],
        'fallback_assignment_count_kept':cnt['fallback_assignment_count_kept'],
        'lane_assignment_success_rate':lane_assignment_success_rate,
        'current_lane_found_rate':current_lane_found_rate,
        'left_lane_found_rate':left_lane_found_rate,
        'right_lane_found_rate':right_lane_found_rate,
        'fallback_assignment_rate':fallback_rate,
        'lane_assignment_success_rate_pre_filter':lane_assignment_success_rate_pre_filter,
        'current_lane_found_rate_pre_filter':current_lane_found_rate_pre_filter,
        'geometric_only_assignment_count':cnt['geometric_only_assignment_count'],
        'adjacency_source_counts':adj_counts,'heading_raw_available_rate':float(cnt['heading_raw']/max(1,cnt['heading_total'])),'heading_proxy_fallback_rate':float(cnt['heading_fallback']/max(1,cnt['heading_total'])),
        'trajectory_nan_count_raw':cnt['trajectory_nan_count_raw'],'trajectory_inf_count_raw':cnt['trajectory_inf_count_raw'],'trajectory_nan_count_after_sanitize':cnt['trajectory_nan_count_after_sanitize'],'trajectory_inf_count_after_sanitize':cnt['trajectory_inf_count_after_sanitize'],
        'ego_windows_repaired':cnt['ego_windows_repaired'],'ego_windows_dropped_sanitize_failed':cnt['ego_windows_dropped_sanitize_failed'],'neighbor_windows_repaired':cnt['neighbor_windows_repaired'],'neighbor_windows_dropped_sanitize_failed':cnt['neighbor_windows_dropped_sanitize_failed'],
        'origin_frame_not_zero_count':cnt['origin_frame_not_zero_count'],'base_heading_raw_count':cnt['base_heading_raw_count'],'base_heading_velocity_fallback_count':cnt['base_heading_velocity_fallback_count'],'nonfinite_output_detected':cnt['nonfinite_output_detected'],
        'assignment_method_counts_by_slot':assignment_method_counts_by_slot,
        'empty_slot_count_by_slot': empty_slot_count_by_slot,
        'empty_slot_ratio_by_slot': empty_slot_ratio_by_slot,
        'slot_thresholds':{'front_max_distance':a.front_max_distance,'side_front_max_distance':a.side_front_max_distance,'side_rear_max_distance':a.side_rear_max_distance,'lane_lateral_tolerance':a.lane_lateral_tolerance,'slot_heading_diff_deg':a.slot_heading_diff_deg,'static_speed_threshold':a.static_speed_threshold},
        'static_neighbor_count_by_slot':{s:int(sum(1 for r in debug_rows if r.get('slot_name')==s and r.get('neighbor_is_static') is True)) for s in SLOT_NAMES},
        'static_front_count':int(sum(1 for r in debug_rows if r.get('slot_name')=='front' and r.get('neighbor_is_static') is True)),
        'static_front_ratio':float(sum(1 for r in debug_rows if r.get('slot_name')=='front' and r.get('neighbor_is_static') is True)/max(1,sum(1 for r in debug_rows if r.get('slot_name')=='front' and r.get('assignment_method')!='empty'))),
        'lane_context_quality_counts':lane_context_quality_counts,'good_lane_context_rate':float(lane_context_quality_counts.get('good',0)/max(1,cnt['kept'])),'ambiguous_intersection_rate':float(lane_context_quality_counts.get('ambiguous_intersection',0)/max(1,cnt['kept'])),'bad_lane_context_rate':float(lane_context_quality_counts.get('bad',0)/max(1,cnt['kept'])),'fallback_lane_context_rate':float(lane_context_quality_counts.get('fallback',0)/max(1,cnt['kept'])),'lane_context_quality_reason_counts':dict(lane_context_quality_reason_counts),
        'mean_projection_distance':float(np.nanmean([r.get('projection_distance',np.nan) for r in debug_rows])) if debug_rows else np.nan,
        'p95_projection_distance':float(np.nanpercentile([r.get('projection_distance',np.nan) for r in debug_rows if np.isfinite(r.get('projection_distance',np.nan))],95)) if any(np.isfinite(r.get('projection_distance',np.nan)) for r in debug_rows) else np.nan,
        'fallback_reason_counts':dict(Counter([r.get('fallback_reason','') for r in debug_rows if r.get('fallback_reason')])), 'slot_rejection_reason_counts':{k:dict(v) for k,v in slot_rejection_reason_counts.items()}, 'progress_enabled':True, 'timing_seconds':{'total':0.0,'load_tfrecord':float(timing['load_tfrecord']),'extract_lanes':float(timing['extract_lanes']),'build_lane_index':float(timing['build_lane_index']),'lane_projection':float(timing['lane_projection']),'assignment':float(timing['assignment']),'feature_build':float(timing['feature_build']),'write_outputs':0.0}, 'warnings':[], 'notes':['Some lane-change features are proxy.']
    }
    if summary['fallback_assignment_rate'] > 0.5:
        summary['warnings'].append('High fallback rate; lane-aware assignment quality may be insufficient for training.')
    if summary['lane_assignment_success_rate'] < 0.5:
        summary['warnings'].append('Lane-aware assignment success rate is low; do not proceed to Stage 5B training until investigated.')
    if summary['ambiguous_intersection_rate'] > 0.9 and summary['fallback_assignment_rate'] == 0:
        summary['warnings'].append('Ambiguous rate is very high despite no fallback; check lane_context_quality definition.')
    for slot in SLOT_NAMES:
        total = sum(assignment_method_counts_by_slot[slot].values())
        if total != cnt['kept']:
            summary['warnings'].append(f"assignment_method_counts_by_slot[{slot}] sum={total} != n_windows_kept={cnt['kept']}")
    main_rates = {
        'lane_assignment_success_rate': summary['lane_assignment_success_rate'],
        'current_lane_found_rate': summary['current_lane_found_rate'],
        'left_lane_found_rate': summary['left_lane_found_rate'],
        'right_lane_found_rate': summary['right_lane_found_rate'],
        'fallback_assignment_rate': summary['fallback_assignment_rate'],
    }
    high_rates = [k for k, v in main_rates.items() if v > 1.0]
    if high_rates:
        msg=f"Main rates exceed 1.0: {high_rates}"
        summary['warnings'].append(msg)
        if a.strict_summary_validation:
            raise RuntimeError(msg)
    if cnt['windows_total'] < (cnt['f_invalid'] + cnt['f_static'] + cnt['n_windows_dropped_clean_filter_total'] + cnt['kept']):
        summary['warnings'].append('Window accounting mismatch detected.')
    summary['timing_seconds']['total']=float(time.perf_counter()-t_global)
    print('Timing summary (s):', summary['timing_seconds'])
    (out/'neighbor_context_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'build_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    report=f"# Stage 5A 构建报告\n\n- 样本数: {cnt['kept']}\n- slot coverage: {slot_ratio}\n- empty_slot_ratio_by_slot: {summary['empty_slot_ratio_by_slot']}\n- lane_context_quality_counts: {summary['lane_context_quality_counts']}\n- lane_assignment_success_rate(kept): {summary['lane_assignment_success_rate']:.4f}\n- current_lane_found_rate(kept): {summary['current_lane_found_rate']:.4f}\n- lane_assignment_success_rate(pre_filter): {summary['lane_assignment_success_rate_pre_filter']:.4f}\n- current_lane_found_rate(pre_filter): {summary['current_lane_found_rate_pre_filter']:.4f}\n- fallback rate(kept): {summary['fallback_assignment_rate']:.4f}\n- heading fallback rate: {summary['heading_proxy_fallback_rate']:.4f}\n- ego windows repaired: {cnt['ego_windows_repaired']}\n- neighbor windows repaired: {cnt['neighbor_windows_repaired']}\n- sanitize failed / dropped count: {cnt['ego_windows_dropped_sanitize_failed'] + cnt['neighbor_windows_dropped_sanitize_failed']}\n- raw NaN count: {cnt['trajectory_nan_count_raw']}\n- after sanitize NaN count: {cnt['trajectory_nan_count_after_sanitize']}\n\n## lane context 解释\n- lane_context_quality 衡量的是 lane/map 语义可靠性，不是五个邻车 slot 是否都有车。\n- empty slot 是正常交通稀疏现象，不应自动判定为 ambiguous_intersection。\n- slot coverage / empty slot ratio 单独报告。\n- 主指标 rate 使用 kept 分母，pre_filter 指标单独报告，避免分母不一致。\n- 如需严格检查主指标 rate<=1.0，可启用 --strict_summary_validation。\n\n## 已知限制\n- lane-change 相关特征部分为 proxy（名称后缀 `_proxy`）。\n- 本阶段仅做数据构建与诊断，不启动训练。\n"
    (out/'build_report.md').write_text(report, encoding='utf-8')

if __name__ == '__main__':
    main()
