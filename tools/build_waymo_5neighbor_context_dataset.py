#!/usr/bin/env python3
import argparse, csv, hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from tools.interaction_context_features import aggregate_interaction_features
from tools.lane_aware_assignment import SLOT_NAMES, assign_neighbors_lane_aware

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
    p.add_argument('--assignment_mode', type=str, default='lane_aware_with_geometric_fallback')
    p.add_argument('--ttc_cap', type=float, default=999.0)
    p.add_argument('--thw_cap', type=float, default=999.0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--smoke_test', action='store_true')
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

def main():
    a = parse_args(); out = Path(a.out_dir)
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = []; rng = np.random.default_rng(7)
    if a.smoke_test:
        for s in range(3):
            T = max(a.window_len + 10, 100); tracks = {}
            for k in range(8):
                x=np.linspace(0,90,T)+k*6; y=np.full(T,(k-3)*2.0); vx,vy=np.gradient(x,a.dt),np.gradient(y,a.dt); h=np.arctan2(vy,vx); valid=np.ones(T,np.float32)
                arr=np.stack([x,y,vx,vy,h,valid],1).astype(np.float32); arr[:,:5]+=rng.normal(0,0.01,(T,5)).astype(np.float32)
                bad=rng.choice(np.arange(5,T-5),size=3,replace=False); arr[bad,:5]=np.nan; arr[bad,5]=0.0; tracks[f'veh_{k}']=arr
            scenarios.append((f'smoke_{s}', tracks))
    else:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
        files=sorted([str(p) for p in Path(a.waymo_dir).glob('*.tfrecord*')])
        if a.max_files: files=files[:a.max_files]
        for fp in files:
            ds=tf.data.TFRecordDataset(fp)
            for rec in ds:
                sc=scenario_pb2.Scenario(); sc.ParseFromString(bytes(rec.numpy())); tracks={}
                for tr in sc.tracks:
                    if a.agent_types=='vehicle' and tr.object_type!=1: continue
                    arr=[[st.center_x,st.center_y,st.velocity_x,st.velocity_y,st.heading,1.0] if st.valid else [np.nan]*5+[0.0] for st in tr.states]
                    tracks[str(tr.id)]=np.asarray(arr,np.float32)
                scenarios.append((sc.scenario_id,tracks))
                if a.max_scenarios and len(scenarios)>=a.max_scenarios: break
            if a.max_scenarios and len(scenarios)>=a.max_scenarios: break

    ego_feats=["ego_x_local","ego_y_local","ego_vx_local","ego_vy_local","ego_heading_local","ego_speed","ego_accel","ego_yaw_rate"]
    nbr_feats=["valid_mask","dx_ego","dy_ego","rel_vx_ego","rel_vy_ego","distance","longitudinal_gap","lateral_gap","closing_rate","ttc_proxy","thw_proxy","neighbor_speed","neighbor_accel","neighbor_heading_rel","neighbor_yaw_rate"]
    ego_seq=[]; nbr_seq=[]; ctx=[]; cmask=[]; cmask_win=[]; slot_ids=[]; meta=[]; splits=[]; inter=[]; debug_rows=[]; cnt=defaultdict(int); slot_valid_counts=defaultdict(int)
    for sid,tracks in scenarios:
        sp=split_of_sid(sid); ids=list(tracks.keys())[:a.max_agents_per_scenario]; cnt['targets']+=len(ids)
        for aid in ids:
            tr=tracks[aid]; T=len(tr)
            for st in range(0,max(0,T-a.window_len+1),a.stride):
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
                assign=assign_neighbors_lane_aware(np.array([origin[0],origin[1],base_h],np.float32),{k:v[0,:2] for k,v in candidates.items() if np.isfinite(v[0,:2]).all()},a.assignment_mode)
                nbr=np.zeros((5,a.window_len,len(nbr_feats)),np.float32); sidrow=[]
                for si,sn in enumerate(SLOT_NAMES):
                    nid=assign.slot_to_agent.get(sn,''); sidrow.append(nid if nid else '-1')
                    if not nid: continue
                    nw,n_valid,ndiag=sanitize_track_window(candidates[nid],a.dt,f'{sid}:{aid}:{st}:{nid}',1.0-a.min_valid_ratio)
                    cnt['trajectory_nan_count_raw']+=ndiag['nan_count_raw']; cnt['trajectory_inf_count_raw']+=ndiag['inf_count_raw']
                    if nw is None:
                        cnt['neighbor_windows_dropped_sanitize_failed']+=1
                        debug_rows.append({'scenario_id':str(sid),'target_agent_id':str(aid),'start':int(st),'slot_name':sn,'assignment_method':'sanitize_failed','neighbor_id':str(nid),'fallback_used':True,'fallback_reason':ndiag['dropped_reason'],'distance':'','longitudinal_gap':'','lateral_gap':''})
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
                    nbr[si]=np.nan_to_num(nbr[si],nan=0.0,posinf=a.ttc_cap,neginf=-a.ttc_cap); slot_valid_counts[sn]+=int(np.sum(nbr[si,:,0]>0.5))
                for d in assign.per_slot_debug: d.update(dict(scenario_id=str(sid),target_agent_id=str(aid),start=int(st))); debug_rows.append(d)
                context=np.concatenate([ego,nbr.reshape(a.window_len,-1)],1); eidx=len(ego_seq)
                ego_seq.append(ego); nbr_seq.append(nbr); ctx.append(context); cmask.append((nbr[:,:,0]>0.5).T); cmask_win.append(np.max(nbr[:,:,0],axis=1)>0.5); slot_ids.append(sidrow); splits.append(sp)
                meta.append((eidx,str(sid),str(aid),int(st),int(a.window_len),sp,a.assignment_mode,False,True)); inter_feat, inter_names = aggregate_interaction_features(ego,nbr,a.dt); inter.append(np.nan_to_num(inter_feat,nan=0.0,posinf=1e6,neginf=-1e6)); cnt['kept']+=1

    ego_arr=np.asarray(ego_seq,np.float32); nbr_arr=np.asarray(nbr_seq,np.float32); ctx_arr=np.asarray(ctx,np.float32); cmask_arr=np.asarray(cmask,np.float32); cmw=np.asarray(cmask_win,np.float32)
    inter_raw=np.asarray(inter,np.float32); split_arr=np.asarray(splits,dtype=object)
    train=(split_arr=='train'); mu=np.mean(inter_raw[train],axis=0) if np.any(train) else np.mean(inter_raw,axis=0); sd=np.std(inter_raw[train],axis=0) if np.any(train) else np.std(inter_raw,axis=0)
    sd=np.where(sd<1e-6,1e-6,sd); inter_std=((inter_raw-mu)/sd).astype(np.float32)
    np.save(out/'ego_seq.npy',ego_arr); np.save(out/'neighbor_seq.npy',nbr_arr); np.save(out/'context_traj.npy',ctx_arr); np.save(out/'context_mask.npy',cmask_arr); np.save(out/'context_mask_window.npy',cmw); np.save(out/'neighbor_slot_ids.npy',np.asarray(slot_ids,dtype=object))
    meta_dtype=np.dtype([('row_index','i4'),('scenario_id','O'),('target_agent_id','O'),('start','i4'),('window_len','i4'),('split','O'),('assignment_mode','O'),('lane_assignment_success','?'),('fallback_used','?')])
    np.save(out/'meta.npy',np.array(meta,dtype=meta_dtype)); np.save(out/'split.npy',split_arr); np.save(out/'interaction_feat_style_raw.npy',inter_raw); np.save(out/'interaction_feat_style.npy',inter_std)
    (out/'ego_feature_names.json').write_text(json.dumps(ego_feats,indent=2),encoding='utf-8'); (out/'neighbor_feature_names.json').write_text(json.dumps(nbr_feats,indent=2),encoding='utf-8'); (out/'neighbor_slot_names.json').write_text(json.dumps(SLOT_NAMES,indent=2),encoding='utf-8')
    (out/'context_feature_names.json').write_text(json.dumps(ego_feats+[f'{s}.{n}' for s in SLOT_NAMES for n in nbr_feats],indent=2),encoding='utf-8'); (out/'interaction_feature_names.json').write_text(json.dumps(inter_names,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'interaction_feature_standardization.json').write_text(json.dumps({'mean':mu.tolist(),'std':sd.tolist(),'feature_names':inter_names,'train_count':int(np.sum(train)),'clip_value':None},indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'neighbor_slot_valid_ratio.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['slot_name','valid_ratio','valid_count','total_count']); w.writeheader(); total=max(1,cnt['kept']*a.window_len)
        for s in SLOT_NAMES: w.writerow({'slot_name':s,'valid_ratio':slot_valid_counts[s]/total,'valid_count':slot_valid_counts[s],'total_count':total})
    with (out/'lane_assignment_debug.csv').open('w',newline='',encoding='utf-8') as f:
        fn=['scenario_id','target_agent_id','start','slot_name','assignment_method','neighbor_id','fallback_used','fallback_reason','distance','longitudinal_gap','lateral_gap']
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(debug_rows)
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
    summary={'dataset_type':'waymo_5neighbor_context','n_files_processed':0 if a.smoke_test else (a.max_files or -1),'n_scenarios_processed':len(scenarios),'n_target_agents_considered':cnt['targets'],'n_windows_total':cnt['windows_total'],'n_windows_kept':cnt['kept'],'n_windows_filtered_static':cnt['f_static'],'n_windows_filtered_invalid':cnt['f_invalid'],'split_counts':dict(Counter(splits)),'window_len':a.window_len,'dt':a.dt,'slot_valid_ratio':slot_ratio,'lane_assignment_success_rate':0.0,'fallback_assignment_rate':1.0,'heading_raw_available_rate':float(cnt['heading_raw']/max(1,cnt['heading_total'])),'heading_proxy_fallback_rate':float(cnt['heading_fallback']/max(1,cnt['heading_total'])),'trajectory_nan_count_raw':cnt['trajectory_nan_count_raw'],'trajectory_inf_count_raw':cnt['trajectory_inf_count_raw'],'trajectory_nan_count_after_sanitize':cnt['trajectory_nan_count_after_sanitize'],'trajectory_inf_count_after_sanitize':cnt['trajectory_inf_count_after_sanitize'],'ego_windows_repaired':cnt['ego_windows_repaired'],'ego_windows_dropped_sanitize_failed':cnt['ego_windows_dropped_sanitize_failed'],'neighbor_windows_repaired':cnt['neighbor_windows_repaired'],'neighbor_windows_dropped_sanitize_failed':cnt['neighbor_windows_dropped_sanitize_failed'],'origin_frame_not_zero_count':cnt['origin_frame_not_zero_count'],'base_heading_raw_count':cnt['base_heading_raw_count'],'base_heading_velocity_fallback_count':cnt['base_heading_velocity_fallback_count'],'nonfinite_output_detected':cnt['nonfinite_output_detected'],'warnings':['Lane-aware map projection placeholder is active; geometric fallback used.','Some lane-change features are proxy.']}
    (out/'neighbor_context_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8'); (out/'build_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    report=f"# Stage 5A 构建报告\n\n- 样本数: {cnt['kept']}\n- slot coverage: {slot_ratio}\n- fallback rate: 1.0（当前为几何 fallback）\n- heading fallback rate: {summary['heading_proxy_fallback_rate']:.4f}\n- ego windows repaired: {cnt['ego_windows_repaired']}\n- neighbor windows repaired: {cnt['neighbor_windows_repaired']}\n- sanitize failed / dropped count: {cnt['ego_windows_dropped_sanitize_failed'] + cnt['neighbor_windows_dropped_sanitize_failed']}\n- raw NaN count: {cnt['trajectory_nan_count_raw']}\n- after sanitize NaN count: {cnt['trajectory_nan_count_after_sanitize']}\n\n## 已知限制\n- lane-aware map 投影尚未完成，`assign_neighbors_lane_aware` 当前返回几何 fallback。\n- lane-change 相关特征部分为 proxy（名称后缀 `_proxy`）。\n- 本阶段仅做数据构建与诊断，不启动训练。\n"
    (out/'build_report.md').write_text(report,encoding='utf-8')

if __name__ == '__main__':
    main()
