#!/usr/bin/env python3
import argparse, hashlib, json, math, os, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

FEATURE_NAMES = [
  "mean_speed","std_speed","rms_accel","rms_jerk","rms_yaw_rate_proxy","rms_curvature_proxy",
  "mean_thw","min_thw","mean_front_distance","min_front_distance","mean_rel_speed","std_rel_speed",
  "max_abs_accel","max_abs_jerk","heading_change_total","valid_ratio"
]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--waymo_dir',type=str,default=None)
    p.add_argument('--out_dir',type=str,required=True)
    p.add_argument('--max_files',type=int,default=None)
    p.add_argument('--max_scenarios',type=int,default=None)
    p.add_argument('--max_agents_per_scenario',type=int,default=64)
    p.add_argument('--window_len',type=int,default=80)
    p.add_argument('--stride',type=int,default=20)
    p.add_argument('--dt',type=float,default=0.1)
    p.add_argument('--min_valid_ratio',type=float,default=0.8)
    p.add_argument('--min_speed',type=float,default=1.0)
    p.add_argument('--agent_types',type=str,default='vehicle')
    p.add_argument('--front_search_distance',type=float,default=80.0)
    p.add_argument('--front_lateral_threshold',type=float,default=4.0)
    p.add_argument('--split_by_scenario',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--train_ratio',type=float,default=0.8)
    p.add_argument('--val_ratio',type=float,default=0.1)
    p.add_argument('--test_ratio',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--smoke_test',action='store_true')
    return p.parse_args()

def split_of_sid(sid,tr,val,te):
    h=int(hashlib.md5(str(sid).encode()).hexdigest()[:8],16)/0xFFFFFFFF
    return 'train' if h<tr else ('val' if h<tr+val else 'test')

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def front_match(ego, cands, front_search_distance, lat_th):
    T=len(ego); picks=[]
    for t in range(T):
        ex,ey,evx,evy=ego[t]
        sp=max(np.hypot(evx,evy),1e-6)
        hx,hy=evx/sp,evy/sp
        best=None
        for aid,tr in cands.items():
            cx,cy=tr[t,0],tr[t,1]
            if np.isnan(cx) or np.isnan(cy): continue
            dx,dy=cx-ex,cy-ey
            lx=dx*hx+dy*hy; ly=-dx*hy+dy*hx
            if lx>0 and abs(ly)<lat_th and lx<front_search_distance:
                if best is None or lx<best[0]: best=(lx,aid)
        picks.append(best[1] if best else None)
    cnt=Counter([x for x in picks if x is not None])
    if not cnt: return None, np.full_like(ego,np.nan,dtype=np.float32)
    fid=cnt.most_common(1)[0][0]
    return fid, cands[fid].astype(np.float32)

def compute_features(ego,front,dt,valid_ratio):
    vx,vy=ego[:,2],ego[:,3]; speed=np.hypot(vx,vy)
    accel=np.diff(speed,prepend=speed[0])/max(dt,1e-6)
    jerk=np.diff(accel,prepend=accel[0])/max(dt,1e-6)
    heading=np.arctan2(vy,vx); yaw=np.diff(heading,prepend=heading[0]); yaw=wrap(yaw)/max(dt,1e-6)
    curv=yaw/np.maximum(speed,1e-3)
    fd=np.linalg.norm(front[:,:2]-ego[:,:2],axis=1) if not np.isnan(front).all() else np.full(len(ego),np.nan)
    thw=fd/np.maximum(speed,1e-3)
    fs=np.hypot(front[:,2],front[:,3]) if not np.isnan(front).all() else np.full(len(ego),np.nan)
    rel=speed-fs
    f=[np.mean(speed),np.std(speed),np.sqrt(np.mean(accel**2)),np.sqrt(np.mean(jerk**2)),np.sqrt(np.mean(yaw**2)),np.sqrt(np.mean(curv**2)),
       np.nanmean(thw),np.nanmin(thw),np.nanmean(fd),np.nanmin(fd),np.nanmean(rel),np.nanstd(rel),np.max(np.abs(accel)),np.max(np.abs(jerk)),
       float(np.sum(np.abs(wrap(np.diff(heading))))),valid_ratio]
    return np.asarray(f,np.float32)

def write_report(out, summary):
    txt=f"""# 阶段 4B 构建报告\n\n## 数据来源\n- Waymo 场景数据目录：`{summary.get('waymo_dir')}`\n- 数据类型：`human_public`\n\n## 输出文件\n- traj.npy / front.npy / meta.npy / split.npy / feat_style_raw.npy / feat_style.npy / feature_names_style.json / build_summary.json / style_feature_standardization.json\n\n## 样本数量\n- n_windows_kept: {summary['n_windows_kept']}\n\n## split 分布\n- {summary['split_counts']}\n\n## front vehicle 匹配率\n- {summary['front_found_rate']:.4f}\n\n## 过滤条件\n- min_valid_ratio={summary['command_args']['min_valid_ratio']}\n- min_speed={summary['command_args']['min_speed']}\n\n## 特征定义\n- speed/accel/jerk/heading/yaw_rate_proxy/curvature_proxy/thw/front_distance/rel_speed 统计。\n\n## 已知限制\n- 首版 front 匹配为近似同车道规则，不含车道级 map matching。\n- 仅提取 vehicle agent。\n- 不生成 pseudo labels（Stage 4C 执行）。\n- 不评估 learned embedding。\n- 不运行 synthetic policy rollout。\n- 不做传感器渲染或感知。\n\n## 下一步命令\n- `python tools/assign_pseudo_style_labels.py --data_dir outputs/waymo_human_v1 --out_dir outputs/waymo_human_v1/pseudo_labels --label_mode percentile --target_quantile 0.25 --dt 0.1 --dataset_type human_public`\n- `python tools/evaluate_vehicledata_validation.py --data_dir outputs/waymo_human_v1 --label_dir outputs/waymo_human_v1/pseudo_labels --out_dir outputs/waymo_human_v1/eval_baselines_only --eval_split test --distance euclidean --topk 5 --baselines raw_feature,trajectory_l2,random,pca_feature --retrieval_mode strict --dataset_type human_public --projection pca`\n"""
    (Path(out)/'build_report.md').write_text(txt,encoding='utf-8')

def main():
    a=parse_args(); out=Path(a.out_dir)
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed); warnings=[]
    scenarios=[]
    if a.smoke_test:
        for s in range(3):
            tracks={}
            T=max(a.window_len+20,100)
            for k in range(6):
                base=np.linspace(0,60,T)+k*8
                y=np.full(T,k*1.2)
                vx=np.gradient(base,a.dt); vy=np.gradient(y,a.dt)
                tr=np.stack([base,y,vx,vy],1).astype(np.float32)
                tr += rng.normal(0,0.05,tr.shape)
                tracks[f"veh_{k}"]=tr
            scenarios.append((f"smoke_{s}",tracks))
    else:
        try:
            import tensorflow as tf
            from waymo_open_dataset.protos import scenario_pb2
        except Exception:
            raise RuntimeError("Waymo parser dependencies are missing. Please install the same Waymo Open Dataset package used by the existing repo, or run with --smoke_test to verify the pipeline.")
        files=sorted([str(p) for p in Path(a.waymo_dir).glob('*.tfrecord*')])
        if a.max_files: files=files[:a.max_files]
        for fp in files:
            ds=tf.data.TFRecordDataset(fp)
            for rec in ds:
                sc=scenario_pb2.Scenario(); sc.ParseFromString(bytes(rec.numpy()))
                tracks={}
                for tr in sc.tracks:
                    if a.agent_types=='vehicle' and tr.object_type!=1: continue
                    arr=[]
                    for st in tr.states:
                        if st.valid: arr.append([st.center_x,st.center_y,st.velocity_x,st.velocity_y])
                        else: arr.append([np.nan,np.nan,np.nan,np.nan])
                    tracks[str(tr.id)]=np.asarray(arr,np.float32)
                scenarios.append((sc.scenario_id,tracks))
                if a.max_scenarios and len(scenarios)>=a.max_scenarios: break
            if a.max_scenarios and len(scenarios)>=a.max_scenarios: break

    traj=[]; front=[]; meta=[]; split=[]; feats=[]
    cnt=defaultdict(int)
    for sid,tracks in scenarios:
        sp=split_of_sid(sid,a.train_ratio,a.val_ratio,a.test_ratio)
        agent_ids=list(tracks.keys())[:a.max_agents_per_scenario]
        cnt['agents']+=len(agent_ids)
        for aid in agent_ids:
            tr=tracks[aid]; T=len(tr)
            for st in range(0,max(0,T-a.window_len+1),a.stride):
                cnt['windows_total']+=1
                ew=tr[st:st+a.window_len]
                valid=np.isfinite(ew).all(axis=1); vr=float(np.mean(valid))
                if vr<a.min_valid_ratio: cnt['f_invalid']+=1; continue
                speed=np.hypot(ew[:,2],ew[:,3]);
                if np.nanmean(speed)<a.min_speed: cnt['f_static']+=1; continue
                cands={k:v[st:st+a.window_len] for k,v in tracks.items() if k!=aid and len(v)>=st+a.window_len}
                fid,fw=front_match(ew,cands,a.front_search_distance,a.front_lateral_threshold)
                cnt['front_found']+= int(fid is not None)
                feat=compute_features(ew,fw,a.dt,vr)
                idx=len(traj)
                traj.append(ew.astype(np.float32)); front.append(fw.astype(np.float32)); split.append(sp); feats.append(feat)
                meta.append((idx,str(sid),str(aid),str(aid),int(st),int(a.window_len),'' if fid is None else str(fid),'' if fid is None else str(fid),sp,'human_public'))

    traj_arr=np.array(traj,dtype=object); front_arr=np.array(front,dtype=object)
    split_arr=np.asarray(split,dtype=object); feat_raw=np.asarray(feats,dtype=np.float32)
    train_mask=(split_arr=='train')
    if train_mask.any(): mu=np.nanmean(feat_raw[train_mask],0); sd=np.nanstd(feat_raw[train_mask],0); fitted='train'
    else: mu=np.nanmean(feat_raw,0); sd=np.nanstd(feat_raw,0); fitted='global'; warnings.append('No train split found; standardization fitted on global stats.')
    eps=1e-6; sd=np.where(sd<eps,eps,sd); feat_std=((np.nan_to_num(feat_raw,nan=0.0)-mu)/sd).astype(np.float32)

    meta_dtype=np.dtype([('row_index','i4'),('scenario_id','O'),('agent_id','O'),('track_id','O'),('start','i4'),('window_len','i4'),('front_id','O'),('front_track_id','O'),('split','O'),('dataset_type','O')])
    meta_arr=np.array(meta,dtype=meta_dtype)
    np.save(out/'traj.npy',traj_arr); np.save(out/'front.npy',front_arr); np.save(out/'meta.npy',meta_arr); np.save(out/'split.npy',split_arr)
    np.save(out/'feat_style_raw.npy',feat_raw); np.save(out/'feat_style.npy',feat_std)
    (out/'feature_names_style.json').write_text(json.dumps(FEATURE_NAMES,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'style_feature_standardization.json').write_text(json.dumps({'mean':mu.tolist(),'std':sd.tolist(),'fitted_on_split':fitted,'eps':eps},ensure_ascii=False,indent=2),encoding='utf-8')

    summary={
      'dataset_type':'human_public','waymo_dir':a.waymo_dir,'out_dir':str(out),'n_files_processed':0 if a.smoke_test else (a.max_files or -1),
      'n_scenarios_processed':len(scenarios),'n_agents_considered':cnt['agents'],'n_windows_total':cnt['windows_total'],'n_windows_kept':len(traj),
      'n_windows_filtered_static':cnt['f_static'],'n_windows_filtered_invalid':cnt['f_invalid'],'n_front_found':cnt['front_found'],
      'front_found_rate':(cnt['front_found']/len(traj) if len(traj) else 0.0),'split_strategy':'scenario_hash','split_counts':dict(Counter(split)),
      'feature_names':FEATURE_NAMES,'warnings':warnings,'command_args':vars(a)
    }
    (out/'build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(out,summary)

if __name__=='__main__': main()
