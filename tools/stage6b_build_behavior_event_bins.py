#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from tools.stage6_compare_unpaired_style import load_schema, _resolve_alias

def load_all(shard_manifest):
    sm=json.loads(Path(shard_manifest).read_text()); base=Path(shard_manifest).parent
    shards=sm.get('shards',sm.get('shard_infos',[])); sps=[s['shard_path'] for s in shards] if shards else sm.get('shard_paths',[])
    feats=[]; metas=[]; g=0
    for i,sp in enumerate(sps):
        sd=base/sp; f=np.load(sd/'interaction_feat_style.npy', mmap_mode='r'); feats.append(np.asarray(f))
        mpath = sd/'metadata.csv'
        if mpath.exists(): m=pd.read_csv(mpath)
        else: m=pd.DataFrame({'_row':np.arange(f.shape[0])})
        metas.append(pd.DataFrame({'global_row':np.arange(g,g+f.shape[0]),'shard_id':i,'local_row':np.arange(f.shape[0])}))
        g += f.shape[0]
    return np.concatenate(feats,0), pd.concat(metas,ignore_index=True)

def tertile(v, labels):
    q=np.quantile(v[np.isfinite(v)], [1/3,2/3]) if np.isfinite(v).sum()>=3 else [np.nan,np.nan]
    out=np.array(['unknown']*len(v),dtype=object)
    if np.isfinite(q).all() and q[0]<q[1]:
        out[v<q[0]]=labels[0]; out[(v>=q[0])&(v<q[1])]=labels[1]; out[v>=q[1]]=labels[2]
    return out

def main(a):
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite: raise FileExistsError
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    X,M=load_all(a.shard_manifest)
    names=load_schema(a.feature_schema_path); fmap={n:i for i,n in enumerate(names)}
    warn={'resolved':{},'missing':[]}
    def get(cands):
        k=_resolve_alias(fmap,cands); warn['resolved'][cands[0]]=k
        if not k: warn['missing'].append(cands)
        return np.asarray(X[:,fmap[k]],float) if k else None
    thw=get(['mean_thw','min_thw']); frontd=get(['mean_front_distance','min_front_distance']); fp=get(['front_pressure_score'])
    lc=get(['lane_change_count_proxy']); yaw=get(['rms_yaw_rate']); spd=get(['speed_mean','ego_speed_mean','speed_norm_mean'])
    yld=get(['yielding_score_proxy'])
    df=M.copy()
    df['event_following_bin']=np.where((thw is not None)&(frontd is not None)&np.isfinite(thw)&np.isfinite(frontd),'following_proxy','no_following_proxy')
    cut_proxy = (fp if fp is not None else np.zeros(len(df)))
    df['event_cut_in_bin']=np.where(cut_proxy>np.nanmedian(cut_proxy),'cut_in_proxy','no_cut_in_proxy')
    if lc is not None: df['event_lane_change_bin']=np.where(lc>0,'lane_change','no_lane_change')
    else: df['event_lane_change_bin']='unknown'
    if spd is not None:
        sb=tertile(spd,['low_speed','mid_speed','high_speed'])
        df['event_low_speed_bin']=np.where(sb=='low_speed','low_speed','not_low_speed')
        df['event_high_speed_bin']=np.where(sb=='high_speed','high_speed','not_high_speed')
    else:
        df['event_low_speed_bin']='unknown'; df['event_high_speed_bin']='unknown'
    if yld is not None: df['event_yielding_bin']=np.where(yld>np.nanmedian(yld),'yielding_like','non_yielding_like')
    else: df['event_yielding_bin']='unknown'
    lat_score=(lc if lc is not None else 0)+(yaw if yaw is not None else 0)
    df['event_lateral_activity_bin']=tertile(np.asarray(lat_score,float),['low','mid','high'])
    df.to_csv(out/'behavior_event_bins.csv', index=False)
    np.save(out/'behavior_event_bins.npy', df.to_records(index=False))
    (out/'behavior_event_bin_schema.json').write_text(json.dumps({'columns':df.columns.tolist()},indent=2,ensure_ascii=False), encoding='utf-8')
    (out/'behavior_event_bin_warnings.json').write_text(json.dumps(warn,indent=2,ensure_ascii=False), encoding='utf-8')
    count_cols=['event_following_bin','event_cut_in_bin','event_lane_change_bin','event_low_speed_bin','event_high_speed_bin','event_yielding_bin','event_lateral_activity_bin']
    counts={c: df[c].value_counts(dropna=False).to_dict() for c in count_cols}
    report='# Stage6B behavior-event bins\n\n'
    report += '## 事件分箱计数\n\n' + json.dumps(counts, ensure_ascii=False, indent=2)
    report += '\n\n> 说明: event_lateral_activity_bin 含行为污染（behavior-contaminated），仅用于行为报告，不得作为 map ODD。\n'
    (out/'behavior_event_bin_report.md').write_text(report, encoding='utf-8')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--shard_manifest',required=True); p.add_argument('--feature_schema_path',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--overwrite',action='store_true'); main(p.parse_args())
