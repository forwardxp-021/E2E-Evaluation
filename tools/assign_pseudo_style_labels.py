#!/usr/bin/env python3
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

LABEL_MAP={-1:'unlabeled',0:'conservative_like',1:'aggressive_like',2:'lateral_stable_like'}

def _load_optional(path):
    return np.load(path, allow_pickle=True) if path and os.path.exists(path) else None

def _infer_split(split,n):
    if split is None: return np.array(['all']*n)
    arr=np.asarray(split)
    return arr.astype(str)

def _compute_signals(traj, front, dt):
    # traj: [N,T,D], D>=4 with x,y,vx,vy
    xy=traj[:,:,:2]; vel=traj[:,:,2:4]
    speed=np.linalg.norm(vel,axis=2)
    acc=np.diff(speed,axis=1,prepend=speed[:,[0]])/max(dt,1e-6)
    jerk=np.diff(acc,axis=1,prepend=acc[:,[0]])/max(dt,1e-6)
    dx=np.diff(xy[:,:,0],axis=1,prepend=xy[:,[0],0]); dy=np.diff(xy[:,:,1],axis=1,prepend=xy[:,[0],1])
    heading=np.arctan2(dy,dx); yaw=np.diff(heading,axis=1,prepend=heading[:,[0]])/max(dt,1e-6)
    seg=np.sqrt(np.diff(xy[:,:,0],axis=1)**2+np.diff(xy[:,:,1],axis=1)**2)+1e-6
    dhead=np.abs(np.diff(heading,axis=1)); curv=np.zeros_like(yaw); curv[:,1:]=dhead/seg
    out={
      'mean_speed':speed.mean(1),'rms_accel':np.sqrt((acc**2).mean(1)),'rms_jerk':np.sqrt((jerk**2).mean(1)),
      'rms_yaw_rate_proxy':np.sqrt((yaw**2).mean(1)),'rms_curvature_proxy':np.sqrt((curv**2).mean(1)),
      'mean_thw':np.full(traj.shape[0],np.nan),'min_thw':np.full(traj.shape[0],np.nan),
    }
    if front is not None:
      d=np.linalg.norm(front[:,:,:2]-xy,axis=2); thw=d/(speed+1e-3); out['mean_thw']=np.nanmean(thw,1); out['min_thw']=np.nanmin(thw,1)
    return out

def _q(x,q): return np.nanquantile(x,q)

def assign_labels(sig,q=0.25,mode='percentile',unl=-1):
    n=len(sig['mean_speed'])
    ag=(sig['mean_speed']>_q(sig['mean_speed'],1-q)).astype(float)+(sig['mean_thw']<_q(sig['mean_thw'],q)).astype(float)+(sig['rms_accel']>_q(sig['rms_accel'],1-q)).astype(float)+(sig['rms_jerk']>_q(sig['rms_jerk'],1-q)).astype(float)
    co=(sig['mean_speed']<_q(sig['mean_speed'],q)).astype(float)+(sig['mean_thw']>_q(sig['mean_thw'],1-q)).astype(float)+(sig['rms_accel']<_q(sig['rms_accel'],q)).astype(float)+(sig['rms_jerk']<_q(sig['rms_jerk'],q)).astype(float)
    la=(sig['rms_yaw_rate_proxy']<_q(sig['rms_yaw_rate_proxy'],q)).astype(float)+(sig['rms_curvature_proxy']<_q(sig['rms_curvature_proxy'],q)).astype(float)
    scores=np.vstack([co,ag,la]).T
    labels=np.full(n,unl,int); reasons=np.array(['ambiguous']*n,object)
    mx=scores.max(1); arg=scores.argmax(1)
    for i in range(n):
      top=np.where(scores[i]==mx[i])[0]
      if mx[i]>=2 and len(top)==1:
        labels[i]=int(top[0]); reasons[i]=f'high_{LABEL_MAP[labels[i]]}_score'
    return labels,ag,co,la,reasons

def run(args):
    np.random.seed(args.seed)
    data=Path(args.data_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    traj=np.load(args.traj_path or data/'traj.npy')
    front=_load_optional(args.front_path or data/'front.npy')
    split=_load_optional(args.split_path or data/'split.npy')
    sig=_compute_signals(traj,front,args.dt)
    labels,ag,co,la,reasons=assign_labels(sig,args.target_quantile,args.label_mode,args.unlabeled_value)
    n=len(labels); splitv=_infer_split(split,n)
    names=np.array([LABEL_MAP[int(x)] for x in labels],dtype=object)
    df=pd.DataFrame({'index':np.arange(n),'split':splitv,'pseudo_label':labels,'pseudo_label_name':names,'aggressive_score':ag,'conservative_score':co,'lateral_stable_score':la,**sig,'label_reason':reasons})
    df.to_csv(out/'pseudo_label_scores.csv',index=False)
    dist=df['pseudo_label_name'].value_counts().rename_axis('pseudo_label_name').reset_index(name='count'); dist.to_csv(out/'pseudo_label_distribution.csv',index=False)
    np.save(out/'pseudo_label.npy',labels); np.save(out/'pseudo_label_name.npy',names)
    summary={'n_total':int(n),'n_labeled':int(np.sum(labels!=-1)),'n_unlabeled':int(np.sum(labels==-1)),'label_counts':{k:int(v) for k,v in df['pseudo_label_name'].value_counts().to_dict().items()},'label_percentages':{k:float(v/n) for k,v in df['pseudo_label_name'].value_counts().to_dict().items()},'thresholds_used':{'target_quantile':args.target_quantile},'label_mode':args.label_mode,'target_quantile':args.target_quantile,'warnings':['Pseudo labels are weak labels, not ground truth.']}
    (out/'pseudo_label_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'pseudo_label_report.md').write_text('# Pseudo Label Report\n\nPseudo labels are rule-based weak labels and not ground truth.\n')

def smoke_test():
    with tempfile.TemporaryDirectory() as td:
      d=Path(td)/'d'; o=Path(td)/'o'; d.mkdir()
      n,t=64,20
      traj=np.zeros((n,t,4),dtype=np.float32); front=np.zeros((n,t,4),dtype=np.float32)
      for i in range(n):
        v=6+4*np.sin(np.linspace(0,1,t)*np.pi)+(i%3)
        traj[i,:,2]=v; traj[i,:,0]=np.cumsum(v*0.1); front[i,:,2]=v-0.5; front[i,:,0]=traj[i,:,0]+20+(i%5)
      np.save(d/'traj.npy',traj); np.save(d/'front.npy',front); np.save(d/'split.npy',np.array(['train']*20+['val']*20+['test']*24))
      run(argparse.Namespace(data_dir=str(d),out_dir=str(o),traj_path=None,front_path=None,split_path=None,label_mode='percentile',target_quantile=0.25,min_class_count=10,allow_overlap=False,unlabeled_value=-1,dt=0.1,seed=42))
      assert (o/'pseudo_label_summary.json').exists()
      print('smoke_test_pass')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir'); p.add_argument('--out_dir')
    p.add_argument('--feat_path',default=None); p.add_argument('--feature_names_path',default=None)
    p.add_argument('--traj_path',default=None); p.add_argument('--front_path',default=None); p.add_argument('--split_path',default=None)
    p.add_argument('--label_mode',choices=['percentile','rule'],default='percentile'); p.add_argument('--target_quantile',type=float,default=0.25)
    p.add_argument('--min_class_count',type=int,default=50); p.add_argument('--allow_overlap',action='store_true'); p.add_argument('--unlabeled_value',type=int,default=-1)
    p.add_argument('--dt',type=float,default=0.1); p.add_argument('--seed',type=int,default=42); p.add_argument('--smoke_test',action='store_true')
    a=p.parse_args(); smoke_test() if a.smoke_test else run(a)
