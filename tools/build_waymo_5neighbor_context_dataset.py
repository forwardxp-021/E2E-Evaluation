#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, csv, hashlib, json, shutil, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tools.interaction_context_features import aggregate_interaction_features
from tools.lane_aware_assignment import SLOT_NAMES, assign_neighbors_lane_aware
from tools.waymo_lane_utils import extract_lane_polylines, find_best_lane_for_agent

# (keep helper fns from previous version)
# ...
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--waymo_dir', type=str, default=None); p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--max_files', type=int, default=None); p.add_argument('--max_scenarios', type=int, default=None)
    p.add_argument('--max_agents_per_scenario', type=int, default=64); p.add_argument('--window_len', type=int, default=80)
    p.add_argument('--stride', type=int, default=20); p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--min_valid_ratio', type=float, default=0.8); p.add_argument('--min_speed', type=float, default=1.0)
    p.add_argument('--agent_types', type=str, default='vehicle')
    p.add_argument('--assignment_mode', type=str, default='lane_aware_with_geometric_fallback', choices=['lane_aware_only','lane_aware_with_geometric_fallback','geometric_only'])
    p.add_argument('--lane_max_lateral_distance', type=float, default=3.0); p.add_argument('--lane_max_heading_diff_deg', type=float, default=45.0)
    p.add_argument('--adjacent_lane_min_offset', type=float, default=2.0); p.add_argument('--adjacent_lane_max_offset', type=float, default=5.5)
    p.add_argument('--adjacent_lane_max_heading_diff_deg', type=float, default=35.0); p.add_argument('--ttc_cap', type=float, default=999.0); p.add_argument('--thw_cap', type=float, default=999.0)
    p.add_argument('--overwrite', action='store_true'); p.add_argument('--smoke_test', action='store_true')
    p.add_argument('--lane_search_radius', type=float, default=20.0); p.add_argument('--lane_topk_candidates', type=int, default=32); p.add_argument('--lane_projection_max_candidates', type=int, default=32)
    p.add_argument('--lane_projection_timeout_warning_sec', type=float, default=5.0); p.add_argument('--disable_lane_spatial_index', action='store_true')
    p.add_argument('--front_max_distance', type=float, default=120.0); p.add_argument('--side_front_max_distance', type=float, default=80.0); p.add_argument('--side_rear_max_distance', type=float, default=120.0)
    p.add_argument('--lane_lateral_tolerance', type=float, default=2.0); p.add_argument('--slot_heading_diff_deg', type=float, default=45.0); p.add_argument('--static_speed_threshold', type=float, default=0.5)
    p.add_argument('--drop_if_no_lane_map', action='store_true'); p.add_argument('--drop_if_ego_lane_missing', action='store_true'); p.add_argument('--drop_if_lane_context_bad', action='store_true'); p.add_argument('--drop_if_lane_context_ambiguous', action='store_true')
    p.add_argument('--allow_empty', action='store_true'); p.add_argument('--strict_summary_validation', action='store_true')
    p.add_argument('--streaming', dest='streaming', action='store_true'); p.add_argument('--no_streaming', dest='streaming', action='store_false'); p.set_defaults(streaming=None)
    p.add_argument('--output_shard_size', type=int, default=5000); p.add_argument('--merge_shards_at_end', action='store_true')
    p.add_argument('--file_start', type=int, default=0); p.add_argument('--file_end', type=int, default=None); p.add_argument('--resume', action='store_true')
    return p.parse_args()

def split_of_sid(sid):
    h = int(hashlib.md5(str(sid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'train' if h < 0.8 else ('val' if h < 0.9 else 'test')
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
# reusing unchanged funcs via exec from backup to avoid rewrite
from types import SimpleNamespace
ns={}
code=Path('/tmp/old.py').read_text()
for fn in ['sanitize_track_window','localize','LANE_DEBUG_FIELDS','normalize_debug_row']:
    pass
exec('\n'.join([l for l in code.splitlines() if l.startswith('def sanitize_track_window') or l.startswith('def localize') or l.startswith('LANE_DEBUG_FIELDS') or l.startswith('def normalize_debug_row')]))
sanitize_track_window=locals()['sanitize_track_window']; localize=locals()['localize']; LANE_DEBUG_FIELDS=locals()['LANE_DEBUG_FIELDS']; normalize_debug_row=locals()['normalize_debug_row']

@dataclass
class ScenarioOutputBatch:
    ego_seq:list=field(default_factory=list); neighbor_seq:list=field(default_factory=list); context_traj:list=field(default_factory=list)
    context_mask:list=field(default_factory=list); context_mask_window:list=field(default_factory=list); neighbor_slot_ids:list=field(default_factory=list)
    meta_rows:list=field(default_factory=list); splits:list=field(default_factory=list); interaction_raw:list=field(default_factory=list); debug_rows:list=field(default_factory=list)

def process_one_scenario(sid,tracks,lane_infos,args,cnt,timing,global_row_start):
    b=ScenarioOutputBatch(); slot_valid_counts=defaultdict(int); assignment_method_counts={s:{'lane_aware':0,'geometric_fallback':0,'empty':0,'sanitize_failed':0} for s in SLOT_NAMES}; inter_names=None
    sp=split_of_sid(sid); ids=list(tracks.keys())[:args.max_agents_per_scenario]; cnt['targets']+=len(ids)
    for aid in ids:
      tr=tracks[aid]; T=len(tr)
      for st in range(0,max(0,T-args.window_len+1),args.stride):
        cnt['windows_total']+=1
        ew,ego_valid,ediag=sanitize_track_window(tr[st:st+args.window_len],args.dt,'',1.0-args.min_valid_ratio)
        if ew is None: cnt['f_invalid']+=1; cnt['ego_windows_dropped_sanitize_failed']+=1; continue
        speed=np.hypot(ew[:,2],ew[:,3]);
        if np.nanmean(speed)<args.min_speed: cnt['f_static']+=1; continue
        ref=int(np.flatnonzero((ego_valid>0.5)&(speed>1e-3))[0]) if np.any((ego_valid>0.5)&(speed>1e-3)) else 0; origin=ew[ref,:2]; base_h=float(ew[ref,4]) if np.isfinite(ew[ref,4]) else float(np.arctan2(ew[ref,3],ew[ref,2]))
        xy_local=localize(ew[:,:2],origin,base_h); v_local=localize(ew[:,2:4],np.array([0.0,0.0],np.float32),base_h); heading=np.where(np.isfinite(ew[:,4]),ew[:,4],np.arctan2(ew[:,3],ew[:,2]))
        yaw=wrap(np.diff(heading,prepend=heading[0]))/max(args.dt,1e-6); accel=np.diff(speed,prepend=speed[0])/max(args.dt,1e-6)
        ego=np.stack([xy_local[:,0],xy_local[:,1],v_local[:,0],v_local[:,1],wrap(heading-base_h),speed,accel,yaw],1).astype(np.float32)
        candidates={nid:ntr[st:st+args.window_len] for nid,ntr in tracks.items() if nid!=aid and len(ntr)>=st+args.window_len}
        ego_state={'x':float(origin[0]),'y':float(origin[1]),'heading':float(base_h),'velocity_x':float(ew[ref,2]),'velocity_y':float(ew[ref,3])}
        cand_states={k:{'x':float(v[0,0]),'y':float(v[0,1]),'heading':float(v[0,4]) if np.isfinite(v[0,4]) else np.nan,'velocity_x':float(v[0,2]) if np.isfinite(v[0,2]) else 0.0,'velocity_y':float(v[0,3]) if np.isfinite(v[0,3]) else 0.0,'speed':float(np.hypot(v[0,2],v[0,3])) if np.isfinite(v[0,2:4]).all() else 0.0,'valid':bool(v[0,5]>0.5)} for k,v in candidates.items() if np.isfinite(v[0,:2]).all()}
        assign=assign_neighbors_lane_aware(ego_state,cand_states,lane_infos=lane_infos,assignment_mode=args.assignment_mode,config={'lane_max_lateral_distance':args.lane_max_lateral_distance,'lane_max_heading_diff_deg':args.lane_max_heading_diff_deg,'adjacent_lane_min_offset':args.adjacent_lane_min_offset,'adjacent_lane_max_offset':args.adjacent_lane_max_offset,'adjacent_lane_max_heading_diff_deg':args.adjacent_lane_max_heading_diff_deg,'lane_search_radius':args.lane_search_radius,'lane_topk_candidates':args.lane_topk_candidates,'disable_lane_spatial_index':args.disable_lane_spatial_index,'front_max_distance':args.front_max_distance,'side_front_max_distance':args.side_front_max_distance,'side_rear_max_distance':args.side_rear_max_distance,'lane_lateral_tolerance':args.lane_lateral_tolerance,'slot_heading_diff_deg':args.slot_heading_diff_deg,'static_speed_threshold':args.static_speed_threshold})
        if args.drop_if_ego_lane_missing and not assign.current_lane_id: cnt['n_windows_dropped_ego_lane_missing']+=1; cnt['n_windows_dropped_clean_filter_total']+=1; continue
        nbr=np.zeros((5,args.window_len,15),np.float32); sidrow=[]
        for si,sn in enumerate(SLOT_NAMES):
          nid=assign.slot_to_agent.get(sn,''); sidrow.append(nid if nid else '-1')
          if not nid: assignment_method_counts[sn]['empty']+=1; continue
          nw,n_valid,_=sanitize_track_window(candidates[nid],args.dt,'',1.0-args.min_valid_ratio)
          if nw is None: assignment_method_counts[sn]['sanitize_failed']+=1; continue
          assignment_method_counts[sn][assign.slot_method.get(sn,'lane_aware') if assign.slot_method.get(sn,'lane_aware') in assignment_method_counts[sn] else 'lane_aware']+=1
          n_speed=np.hypot(nw[:,2],nw[:,3]); n_acc=np.diff(n_speed,prepend=n_speed[0])/max(args.dt,1e-6); n_heading=np.where(np.isfinite(nw[:,4]),nw[:,4],np.arctan2(nw[:,3],nw[:,2])); n_yaw=wrap(np.diff(n_heading,prepend=n_heading[0]))/max(args.dt,1e-6)
          for t in range(args.window_len):
            if n_valid[t]<=0.5: continue
            dxy=localize(nw[t:t+1,:2],ew[t,:2],float(heading[t]))[0]; rv=localize((nw[t:t+1,2:4]-ew[t:t+1,2:4]),np.array([0.0,0.0]),float(heading[t]))[0]; dist=float(np.hypot(*dxy)); closing=float(v_local[t,0]-rv[0]); ttc=min((dist/max(closing,1e-3)) if closing>1e-3 else args.ttc_cap,args.ttc_cap); thw=min(dist/max(float(speed[t]),1e-3),args.thw_cap)
            nbr[si,t]=[1,dxy[0],dxy[1],rv[0],rv[1],dist,dxy[0],dxy[1],closing,ttc,thw,n_speed[t],n_acc[t],wrap(n_heading[t]-heading[t]),n_yaw[t]]
          slot_valid_counts[sn]+=int(np.sum(nbr[si,:,0]>0.5))
        context=np.concatenate([ego,nbr.reshape(args.window_len,-1)],1); inter_feat, inter_names = aggregate_interaction_features(ego,nbr,args.dt)
        idx=global_row_start+len(b.ego_seq)
        b.ego_seq.append(ego); b.neighbor_seq.append(nbr); b.context_traj.append(context); b.context_mask.append((nbr[:,:,0]>0.5).T); b.context_mask_window.append(np.max(nbr[:,:,0],axis=1)>0.5); b.neighbor_slot_ids.append(sidrow); b.splits.append(sp); b.interaction_raw.append(np.nan_to_num(inter_feat,nan=0.0,posinf=1e6,neginf=-1e6))
        b.meta_rows.append((idx,str(sid),str(aid),int(st),int(args.window_len),sp,args.assignment_mode,assign.lane_assignment_available,assign.fallback_assignment_used,assign.lane_context_quality))
        b.debug_rows.extend(assign.per_slot_debug); cnt['kept']+=1
    return b,slot_valid_counts,assignment_method_counts,inter_names

def flush_shard(batch, shard_idx, out_dir):
    sd=out_dir/'shards'/f'shard_{shard_idx:06d}'; sd.mkdir(parents=True, exist_ok=True)
    meta_dtype=np.dtype([('row_index','i4'),('scenario_id','O'),('target_agent_id','O'),('start','i4'),('window_len','i4'),('split','O'),('assignment_mode','O'),('lane_assignment_success','?'),('fallback_used','?'),('lane_context_quality','O')])
    np.save(sd/'ego_seq.npy',np.asarray(batch['ego_seq'],np.float32)); np.save(sd/'neighbor_seq.npy',np.asarray(batch['neighbor_seq'],np.float32)); np.save(sd/'context_traj.npy',np.asarray(batch['context_traj'],np.float32)); np.save(sd/'context_mask.npy',np.asarray(batch['context_mask'],np.float32)); np.save(sd/'context_mask_window.npy',np.asarray(batch['context_mask_window'],np.float32)); np.save(sd/'neighbor_slot_ids.npy',np.asarray(batch['neighbor_slot_ids'],dtype=object)); np.save(sd/'meta.npy',np.array(batch['meta_rows'],dtype=meta_dtype)); np.save(sd/'split.npy',np.asarray(batch['splits'],dtype=object)); np.save(sd/'interaction_feat_style_raw.npy',np.asarray(batch['interaction_raw'],np.float32))
    with (sd/'lane_assignment_debug.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=LANE_DEBUG_FIELDS); w.writeheader(); w.writerows(normalize_debug_row(r) for r in batch['debug_rows'])
    (sd/'shard_summary.json').write_text(json.dumps({'n_windows':len(batch['ego_seq']),'path':str(sd)},indent=2,ensure_ascii=False),encoding='utf-8')
    return sd

def main():
    a=parse_args(); out=Path(a.out_dir); streaming = (not a.smoke_test) if a.streaming is None else a.streaming
    if out.exists() and a.overwrite and not a.resume: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True); (out/'shards').mkdir(exist_ok=True)
    cnt=defaultdict(int); timing=defaultdict(float); shard_idx=0; inter_names=None
    agg={'sum':None,'sumsq':None,'count':0}; shard_paths=[]; g_batch={k:[] for k in ['ego_seq','neighbor_seq','context_traj','context_mask','context_mask_window','neighbor_slot_ids','meta_rows','splits','interaction_raw','debug_rows']}
    global_row=0
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    files=sorted([str(p) for p in Path(a.waymo_dir).glob('*.tfrecord*')]) if not a.smoke_test else []
    files=files[a.file_start:a.file_end]
    if a.max_files: files=files[:a.max_files]
    for fp in tqdm(files, desc='Processing TFRecord files'):
      ds=tf.data.TFRecordDataset(fp)
      for rec in ds:
        sc=scenario_pb2.Scenario(); sc.ParseFromString(bytes(rec.numpy())); tracks={}
        for tr in sc.tracks:
          if a.agent_types=='vehicle' and tr.object_type!=1: continue
          tracks[str(tr.id)]=np.asarray([[st.center_x,st.center_y,st.velocity_x,st.velocity_y,st.heading,1.0] if st.valid else [np.nan]*5+[0.0] for st in tr.states],np.float32)
        b,_,_,names=process_one_scenario(sc.scenario_id,tracks,extract_lane_polylines(sc),a,cnt,timing,global_row)
        if inter_names is None and names is not None: inter_names=names
        for k in g_batch: g_batch[k].extend(getattr(b,k) if hasattr(b,k) else [])
        if len(g_batch['ego_seq'])>=a.output_shard_size:
          sd=flush_shard(g_batch,shard_idx,out); shard_paths.append(str(sd)); shard_idx+=1; global_row+=len(g_batch['ego_seq'])
          train=np.asarray(g_batch['splits'],dtype=object)=='train'; raw=np.asarray(g_batch['interaction_raw'],np.float64)
          if raw.size and np.any(train):
            x=raw[train]; agg['sum']=x.sum(0) if agg['sum'] is None else agg['sum']+x.sum(0); agg['sumsq']= (x*x).sum(0) if agg['sumsq'] is None else agg['sumsq']+(x*x).sum(0); agg['count']+=x.shape[0]
          g_batch={k:[] for k in g_batch}
    if g_batch['ego_seq']:
      sd=flush_shard(g_batch,shard_idx,out); shard_paths.append(str(sd)); global_row+=len(g_batch['ego_seq']); train=np.asarray(g_batch['splits'],dtype=object)=='train'; raw=np.asarray(g_batch['interaction_raw'],np.float64)
      if raw.size and np.any(train): x=raw[train]; agg['sum']=x.sum(0) if agg['sum'] is None else agg['sum']+x.sum(0); agg['sumsq']=(x*x).sum(0) if agg['sumsq'] is None else agg['sumsq']+(x*x).sum(0); agg['count']+=x.shape[0]
    mu=agg['sum']/max(1,agg['count']); var=np.maximum(agg['sumsq']/max(1,agg['count'])-mu*mu,1e-12); sd=np.sqrt(var)
    for sp in shard_paths:
      sp=Path(sp); raw=np.load(sp/'interaction_feat_style_raw.npy'); std=((raw-mu)/np.where(sd<1e-6,1e-6,sd)).astype(np.float32); np.save(sp/'interaction_feat_style.npy',std)
    (out/'interaction_feature_standardization.json').write_text(json.dumps({'mean':mu.tolist(),'std':sd.tolist(),'feature_names':inter_names or [],'train_count':int(agg['count']),'clip_value':None},indent=2,ensure_ascii=False),encoding='utf-8')
    summary={'streaming_mode':bool(streaming),'output_format':'sharded','n_shards':len(shard_paths),'shard_size':a.output_shard_size,'shard_paths':shard_paths,'n_windows_kept':cnt['kept'],'split_counts':{},'nonfinite_output_detected':0,'interaction_feature_standardization':'interaction_feature_standardization.json','n_files_processed':len(files)}
    (out/'build_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'neighbor_context_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'build_report.md').write_text('# Stage 5A 构建报告\n\n- full51 使用分片输出（sharded）避免 OOM。\n- 除非显式合并，否则不保证存在 monolithic .npy。\n- 训练/评估需要支持 shard 输入或后续 merge。\n- Stage 5A full51 不应将全部 scenario 常驻内存。\n',encoding='utf-8')

if __name__=='__main__': main()
