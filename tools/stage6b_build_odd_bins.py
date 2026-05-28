#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

def qbin(vals, labels):
    ok = np.isfinite(vals)
    if ok.sum() < len(labels):
        return np.array(['unknown']*len(vals),dtype=object), {'collapsed': True}
    qs = np.quantile(vals[ok], np.linspace(0,1,len(labels)+1)[1:-1])
    if len(np.unique(qs)) < len(qs):
        return np.array(['unknown']*len(vals),dtype=object), {'collapsed': True, 'quantiles': qs.tolist()}
    out = np.array(['unknown']*len(vals), dtype=object)
    edges = [-np.inf] + list(qs) + [np.inf]
    for i,l in enumerate(labels):
        out[ok & (vals>=edges[i]) & (vals<edges[i+1])] = l
    return out, {'collapsed': False, 'quantiles': qs.tolist()}

def main(a):

    out=Path(a.output_dir)
    if out.exists() and not a.overwrite: raise FileExistsError('output_dir exists')
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    mm = json.loads(Path(a.map_odd_manifest).read_text(encoding='utf-8'))
    root = Path(a.map_odd_manifest).parent
    rows=[]; feats=[]
    match_rates = []

    for s in mm['shards']:
        feat=np.load(root/s['feature_path'])
        meta=pd.read_csv(root/s['meta_path'])
        feats.append(feat); rows.append(meta[['global_row','shard_id','local_row']])
        if feat.shape[1] <= 15:
            raise ValueError('map_odd_feat.npy 缺少 map_match_valid 列(索引15)')
        match_rates.append(float((feat[:,15] > 0.5).mean()))
    X=np.concatenate(feats,0); M=pd.concat(rows, ignore_index=True)
    global_match_rate = float((X[:,15] > 0.5).mean())
    if global_match_rate < a.min_map_match_rate and not a.allow_low_match_rate:
        raise RuntimeError(f'map_match_valid 比例过低: {global_match_rate:.4f} < {a.min_map_match_rate}，拒绝构建 ODD bins。')
    cross=np.where(X[:,1]>0.5,'crosswalk_near','no_crosswalk_near')
    stop=np.where(X[:,3]>0.5,'stop_sign_near','no_stop_sign_near')
    curv,w1=qbin(X[:,5], ['straight','mild_curve','sharp_curve'])
    inter=np.where(X[:,14]>0.5,'intersection_like','non_intersection_like')
    comp,w2=qbin(X[:,13], ['low','mid','high'])
    lane,w3=qbin(X[:,7], ['simple_lane_context','multi_lane_context','dense_lane_context'])
    df=M.copy()
    df['odd_crosswalk_bin']=cross; df['odd_stop_sign_bin']=stop; df['odd_curvature_bin']=curv
    df['odd_intersection_bin']=inter; df['odd_map_complexity_bin']=comp; df['odd_lane_count_bin']=lane
    df['map_match_valid']=X[:,15].astype(int); df['fallback_full_scenario_path']=X[:,16].astype(int)
    df.to_csv(out/'odd_bins.csv', index=False)
    np.save(out/'odd_bins.npy', df.to_records(index=False))
    counts={c: df[c].value_counts(dropna=False).to_dict() for c in ['odd_crosswalk_bin','odd_stop_sign_bin','odd_curvature_bin','odd_intersection_bin','odd_map_complexity_bin','odd_lane_count_bin']}
    (out/'odd_bin_warnings.json').write_text(json.dumps({'curvature':w1,'complexity':w2,'lane_count':w3},indent=2,ensure_ascii=False), encoding='utf-8')
    (out/'odd_bin_schema.json').write_text(json.dumps({'columns':df.columns.tolist()}, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'odd_bin_report.md').write_text('# Stage6B ODD bins\n\n'+json.dumps(counts,ensure_ascii=False,indent=2), encoding='utf-8')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--map_odd_manifest',required=True); p.add_argument('--shard_manifest',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--min_map_match_rate', type=float, default=0.1); p.add_argument('--allow_low_match_rate', action='store_true'); p.add_argument('--overwrite',action='store_true'); main(p.parse_args())
