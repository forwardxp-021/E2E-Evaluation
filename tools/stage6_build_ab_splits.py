#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json
from pathlib import Path
import numpy as np


def load_schema(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    feats=obj.get('features',[])
    names=[f['name'] for f in sorted(feats,key=lambda x:int(x['index']))] if feats else obj.get('feature_names',[])
    return names

def fmap(names): return {n:i for i,n in enumerate(names)}

def col(arr,m,keys,warns):
    for k in keys:
        if k in m: return arr[:,m[k]],k
    warns.append(f'missing feature candidates: {keys}')
    return None,None

def main(a):
    out=Path(a.output_dir)/a.experiment_name; out.mkdir(parents=True,exist_ok=True)
    feat=np.load(a.feature_path,mmap_mode='r')
    split=np.load(a.split_path,allow_pickle=True)
    split=split.astype(str) if split.dtype.kind in {'U','S','O'} else np.array([['train','val','test'][int(x)] if int(x) in [0,1,2] else str(int(x)) for x in split],dtype=object)
    test_idx=np.flatnonzero(split=='test')
    names=load_schema(a.feature_schema_path); m=fmap(names)
    warns=[]; rng=np.random.default_rng(a.seed)
    ql,qh=a.q_low,a.q_high

    if a.mode=='negative_control_random':
        sel=rng.permutation(test_idx); mid=len(sel)//2; A,B=sel[:mid],sel[mid:]
        criteria=['random split on test indices']
    elif a.mode=='pseudo_style_aggressive_vs_conservative':
        score=np.zeros(len(test_idx),dtype=float); used=[]
        rules=[(['mean_thw'],+1),(['min_thw'],+1),(['rms_jerk'],-1),(['assertiveness_score_proxy','assertiveness_proxy'],-1)]
        for keys,sign in rules:
            v,n=col(feat[test_idx],m,keys,warns)
            if v is None: continue
            s=(v-np.nanmedian(v))/(np.nanpercentile(v,75)-np.nanpercentile(v,25)+1e-6)
            score+=sign*s; used.append((n,sign))
        lo,hi=np.quantile(score,[ql,qh]); A=test_idx[score>=hi]; B=test_idx[score<=lo]
        criteria=[f'conservative score high>=q{qh}, aggressive score low<=q{ql}',f'used={used}']
    else:
        s,_=col(feat[test_idx],m,['ego_speed_mean','speed_mean'],warns)
        d,_=col(feat[test_idx],m,['interaction_density','neighbor_count'],warns)
        if s is None: s=np.zeros(len(test_idx))
        if d is None: d=np.zeros(len(test_idx))
        s_lo,s_hi=np.quantile(s,[ql,qh]); d_lo,d_hi=np.quantile(d,[ql,qh])
        A=test_idx[(s<=s_lo)&(d>=d_hi)]
        B=test_idx[(s>=s_hi)&(d<=d_lo)]
        criteria=[f'A: low_speed<=q{ql} & dense>=q{qh}',f'B: high_speed>=q{qh} & sparse<=q{ql}']

    np.save(out/'a_indices.npy',A); np.save(out/'b_indices.npy',B)
    means={}
    for n in ['speed_mean','mean_thw','min_thw','rms_jerk','interaction_density']:
        if n in m:
            means[n]={'A':float(np.nanmean(feat[A,m[n]])) if len(A) else None,'B':float(np.nanmean(feat[B,m[n]])) if len(B) else None}
    summary={'mode':a.mode,'n_A':int(len(A)),'n_B':int(len(B)),'criteria':criteria,'warnings':warns,'feature_means':means}
    (out/'split_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Stage6 A/B Split Summary',f'- mode: {a.mode}',f'- n_A: {len(A)}',f'- n_B: {len(B)}','## criteria']+[f'- {c}' for c in criteria]+['## warnings']+[f'- {w}' for w in (warns or ['none'])]
    (out/'split_summary.md').write_text('\n'.join(md),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();
    p.add_argument('--mode',choices=['negative_control_random','pseudo_style_aggressive_vs_conservative','scene_confounding_control'],required=True)
    p.add_argument('--feature_path',required=True); p.add_argument('--feature_schema_path',required=True); p.add_argument('--split_path',required=True)
    p.add_argument('--output_dir',default='outputs/stage6A_splits'); p.add_argument('--experiment_name',required=True)
    p.add_argument('--q_low',type=float,default=0.3); p.add_argument('--q_high',type=float,default=0.7); p.add_argument('--seed',type=int,default=42)
    main(p.parse_args())
