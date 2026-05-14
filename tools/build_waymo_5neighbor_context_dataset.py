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
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--smoke_test', action='store_true')
    return p.parse_args()

def split_of_sid(sid):
    h = int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'train' if h < 0.8 else ('val' if h < 0.9 else 'test')

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def localize(xy, origin, heading):
    d = xy - origin[None, :]
    c, s = np.cos(-heading), np.sin(-heading)
    return np.stack([d[:,0]*c - d[:,1]*s, d[:,0]*s + d[:,1]*c], axis=1)

def main():
    a = parse_args(); out = Path(a.out_dir)
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = []
    rng = np.random.default_rng(7)
    if a.smoke_test:
        for s in range(3):
            T = max(a.window_len + 10, 100)
            tracks = {}
            for k in range(8):
                x = np.linspace(0, 90, T) + k * 6
                y = np.full(T, (k - 3) * 2.0)
                vx, vy = np.gradient(x, a.dt), np.gradient(y, a.dt)
                heading = np.arctan2(vy, vx)
                valid = np.ones(T, dtype=np.float32)
                tracks[f'veh_{k}'] = np.stack([x,y,vx,vy,heading,valid],1).astype(np.float32) + rng.normal(0,0.01,(T,6)).astype(np.float32)
            scenarios.append((f'smoke_{s}', tracks))
    else:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
        files = sorted([str(p) for p in Path(a.waymo_dir).glob('*.tfrecord*')])
        if a.max_files: files = files[:a.max_files]
        for fp in files:
            ds = tf.data.TFRecordDataset(fp)
            for rec in ds:
                sc = scenario_pb2.Scenario(); sc.ParseFromString(bytes(rec.numpy()))
                tracks = {}
                for tr in sc.tracks:
                    if a.agent_types == 'vehicle' and tr.object_type != 1: continue
                    arr = []
                    for st in tr.states:
                        if st.valid: arr.append([st.center_x, st.center_y, st.velocity_x, st.velocity_y, st.heading, 1.0])
                        else: arr.append([np.nan]*5 + [0.0])
                    tracks[str(tr.id)] = np.asarray(arr, np.float32)
                scenarios.append((sc.scenario_id, tracks))
                if a.max_scenarios and len(scenarios) >= a.max_scenarios: break
            if a.max_scenarios and len(scenarios) >= a.max_scenarios: break

    ego_feats = ["ego_x_local","ego_y_local","ego_vx_local","ego_vy_local","ego_heading_local","ego_speed","ego_accel","ego_yaw_rate"]
    nbr_feats = ["valid_mask","dx_ego","dy_ego","rel_vx_ego","rel_vy_ego","distance","longitudinal_gap","lateral_gap","closing_rate","ttc_proxy","thw_proxy","neighbor_speed","neighbor_accel","neighbor_heading_rel","neighbor_yaw_rate"]

    ego_seq=[]; nbr_seq=[]; ctx=[]; cmask=[]; cmask_win=[]; slot_ids=[]; meta=[]; splits=[]; inter=[]; debug_rows=[]
    cnt=defaultdict(int); slot_valid_counts=defaultdict(int)

    for sid, tracks in scenarios:
        sp=split_of_sid(sid)
        ids=list(tracks.keys())[:a.max_agents_per_scenario]; cnt['targets']+=len(ids)
        for aid in ids:
            tr=tracks[aid]; T=len(tr)
            for st in range(0, max(0, T-a.window_len+1), a.stride):
                cnt['windows_total']+=1
                ew=tr[st:st+a.window_len]
                valid=np.isfinite(ew[:,:5]).all(axis=1) & (ew[:,5] > 0.5)
                vr=float(np.mean(valid))
                if vr<a.min_valid_ratio: cnt['f_invalid']+=1; continue
                speed=np.hypot(ew[:,2],ew[:,3]);
                if np.nanmean(speed)<a.min_speed: cnt['f_static']+=1; continue

                origin=ew[0,:2]; base_h=float(np.arctan2(ew[0,3],ew[0,2]))
                xy_local=localize(ew[:,:2], origin, base_h)
                v_local=localize(ew[:,2:4], np.array([0.0,0.0],dtype=np.float32), base_h)
                heading_raw=ew[:,4]
                heading_vel=np.arctan2(ew[:,3],ew[:,2])
                heading=np.where(np.isfinite(heading_raw), heading_raw, heading_vel)
                cnt['heading_raw'] += int(np.isfinite(heading_raw).sum())
                cnt['heading_total'] += len(heading_raw)
                yaw=wrap(np.diff(heading,prepend=heading[0]))/max(a.dt,1e-6)
                fallback=(~np.isfinite(heading_raw)); cnt['heading_fallback'] += int(fallback.sum())
                accel=np.diff(speed,prepend=speed[0])/max(a.dt,1e-6)
                ego=np.stack([xy_local[:,0],xy_local[:,1],v_local[:,0],v_local[:,1],wrap(heading-base_h),speed,accel,yaw],1).astype(np.float32)

                candidates={}
                for nid,ntr in tracks.items():
                    if nid==aid or len(ntr)<st+a.window_len: continue
                    w=ntr[st:st+a.window_len]
                    if not np.isfinite(w[0,:2]).all(): continue
                    candidates[nid]=w
                assign=assign_neighbors_lane_aware(np.array([ew[0,0],ew[0,1],base_h],dtype=np.float32), {k:v[0,:2] for k,v in candidates.items()}, a.assignment_mode)

                nbr=np.zeros((5,a.window_len,len(nbr_feats)),dtype=np.float32); sidrow=[]
                for si,sn in enumerate(SLOT_NAMES):
                    nid=assign.slot_to_agent.get(sn,'')
                    sidrow.append(nid if nid else '-1')
                    if nid:
                        nw=candidates[nid]
                        n_speed=np.hypot(nw[:,2],nw[:,3]); n_acc=np.diff(n_speed,prepend=n_speed[0])/max(a.dt,1e-6)
                        n_heading=np.where(np.isfinite(nw[:,4]),nw[:,4],np.arctan2(nw[:,3],nw[:,2]))
                        n_yaw=wrap(np.diff(n_heading,prepend=n_heading[0]))/max(a.dt,1e-6)
                        for t in range(a.window_len):
                            if not np.isfinite(nw[t,:5]).all(): continue
                            dxy=localize(nw[t:t+1,:2], ew[t,:2], float(heading[t]))[0]
                            rv=localize((nw[t:t+1,2:4]-ew[t:t+1,2:4]), np.array([0.0,0.0]), float(heading[t]))[0]
                            dist=np.hypot(dxy[0],dxy[1]); closing=float(v_local[t,0]-rv[0])
                            ttc=dist/max(closing,1e-3) if closing>0 else 999.0; thw=dist/max(speed[t],1e-3)
                            nbr[si,t]=[1,dxy[0],dxy[1],rv[0],rv[1],dist,dxy[0],dxy[1],closing,ttc,thw,n_speed[t],n_acc[t],wrap(n_heading[t]-heading[t]),n_yaw[t]]
                        slot_valid_counts[sn]+=int(np.sum(nbr[si,:,0]>0.5))

                for d in assign.per_slot_debug:
                    d.update(dict(scenario_id=str(sid),target_agent_id=str(aid),start=int(st)))
                    debug_rows.append(d)

                context=np.concatenate([ego, nbr.reshape(a.window_len,-1)], axis=1)
                eidx=len(ego_seq)
                ego_seq.append(ego); nbr_seq.append(nbr); ctx.append(context); cmask.append((nbr[:,:,0]>0.5).T); cmask_win.append(np.max(nbr[:,:,0],axis=1)>0.5)
                slot_ids.append(sidrow); splits.append(sp)
                meta.append((eidx,str(sid),str(aid),int(st),int(a.window_len),sp,a.assignment_mode,False,True))
                inter_feat, inter_names = aggregate_interaction_features(ego, nbr, a.dt); inter.append(inter_feat)
                cnt['kept']+=1

    ego_arr=np.asarray(ego_seq,np.float32); nbr_arr=np.asarray(nbr_seq,np.float32); ctx_arr=np.asarray(ctx,np.float32)
    cmask_arr=np.asarray(cmask,np.float32); cmw=np.asarray(cmask_win,np.float32)
    inter_raw=np.asarray(inter,np.float32); split_arr=np.asarray(splits,dtype=object)
    train=(split_arr=='train')
    mu=np.mean(inter_raw[train],axis=0) if np.any(train) else np.mean(inter_raw,axis=0)
    sd=np.std(inter_raw[train],axis=0) if np.any(train) else np.std(inter_raw,axis=0)
    sd=np.where(sd<1e-6,1e-6,sd); inter_std=((inter_raw-mu)/sd).astype(np.float32)

    np.save(out/'ego_seq.npy',ego_arr); np.save(out/'neighbor_seq.npy',nbr_arr); np.save(out/'context_traj.npy',ctx_arr)
    np.save(out/'context_mask.npy',cmask_arr); np.save(out/'context_mask_window.npy',cmw); np.save(out/'neighbor_slot_ids.npy',np.asarray(slot_ids,dtype=object))
    meta_dtype=np.dtype([('row_index','i4'),('scenario_id','O'),('target_agent_id','O'),('start','i4'),('window_len','i4'),('split','O'),('assignment_mode','O'),('lane_assignment_success','?'),('fallback_used','?')])
    np.save(out/'meta.npy',np.array(meta,dtype=meta_dtype)); np.save(out/'split.npy',split_arr)
    np.save(out/'interaction_feat_style_raw.npy',inter_raw); np.save(out/'interaction_feat_style.npy',inter_std)

    (out/'ego_feature_names.json').write_text(json.dumps(ego_feats,indent=2),encoding='utf-8')
    (out/'neighbor_feature_names.json').write_text(json.dumps(nbr_feats,indent=2),encoding='utf-8')
    (out/'neighbor_slot_names.json').write_text(json.dumps(SLOT_NAMES,indent=2),encoding='utf-8')
    (out/'context_feature_names.json').write_text(json.dumps(ego_feats+[f'{s}.{n}' for s in SLOT_NAMES for n in nbr_feats],indent=2),encoding='utf-8')
    (out/'interaction_feature_names.json').write_text(json.dumps(inter_names,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'interaction_feature_standardization.json').write_text(json.dumps({'mean':mu.tolist(),'std':sd.tolist(),'feature_names':inter_names,'train_count':int(np.sum(train)),'clip_value':None},indent=2,ensure_ascii=False),encoding='utf-8')

    with (out/'neighbor_slot_valid_ratio.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['slot_name','valid_ratio','valid_count','total_count']); w.writeheader()
        total=max(1,cnt['kept']*a.window_len)
        for s in SLOT_NAMES: w.writerow({'slot_name':s,'valid_ratio':slot_valid_counts[s]/total,'valid_count':slot_valid_counts[s],'total_count':total})
    with (out/'lane_assignment_debug.csv').open('w',newline='',encoding='utf-8') as f:
        fn=['scenario_id','target_agent_id','start','slot_name','assignment_method','neighbor_id','fallback_used','fallback_reason','distance','longitudinal_gap','lateral_gap']
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(debug_rows)

    slot_ratio={s:float(slot_valid_counts[s]/max(1,cnt['kept']*a.window_len)) for s in SLOT_NAMES}
    summary={
        'dataset_type':'waymo_5neighbor_context','n_files_processed':0 if a.smoke_test else (a.max_files or -1),'n_scenarios_processed':len(scenarios),
        'n_target_agents_considered':cnt['targets'],'n_windows_total':cnt['windows_total'],'n_windows_kept':cnt['kept'],'n_windows_filtered_static':cnt['f_static'],
        'n_windows_filtered_invalid':cnt['f_invalid'],'split_counts':dict(Counter(splits)),'window_len':a.window_len,'dt':a.dt,'slot_valid_ratio':slot_ratio,
        'lane_assignment_success_rate':0.0,'fallback_assignment_rate':1.0,'heading_raw_available_rate':float(cnt['heading_raw']/max(1,cnt['heading_total'])),
        'heading_proxy_fallback_rate':float(cnt['heading_fallback']/max(1,cnt['heading_total'])),'warnings':['Lane-aware map projection placeholder is active; geometric fallback used.','Some lane-change features are proxy.']
    }
    (out/'neighbor_context_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'build_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    report=f"# Stage 5A 构建报告\n\n- 样本数: {cnt['kept']}\n- slot coverage: {slot_ratio}\n- fallback rate: 1.0（当前为几何 fallback）\n- heading fallback rate: {summary['heading_proxy_fallback_rate']:.4f}\n\n## 限制\n- lane-aware map 投影尚未完成，`assign_neighbors_lane_aware` 当前返回几何 fallback。\n- lane-change 相关特征部分为 proxy（名称后缀 `_proxy`）。\n- 本阶段仅做数据构建与诊断，不启动训练。\n"
    (out/'build_report.md').write_text(report,encoding='utf-8')

    assert np.isfinite(ctx_arr).all()
    assert np.isfinite(inter_std).all()

if __name__ == '__main__':
    main()
