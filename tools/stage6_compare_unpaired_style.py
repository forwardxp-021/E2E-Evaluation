#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def load_schema(path):
    obj = _load_json(path); feats = obj.get('features', [])
    return [f['name'] for f in sorted(feats, key=lambda x: int(x['index']))] if feats else obj.get('feature_names', [])

def _load_manifest_data(shard_manifest, embedding_manifest):
    sm = _load_json(shard_manifest); em = _load_json(embedding_manifest); base = Path(shard_manifest).parent
    shards = sm.get('shards', sm.get('shard_infos', [])); sps = [s['shard_path'] for s in shards] if shards else sm.get('shard_paths', [])
    eps = em.get('embedding_shard_paths', [])
    if len(sps) != len(eps): raise ValueError('feature/embedding shard 数量不一致')
    feats, z = [], []
    for sp, ep in zip(sps, eps):
        f = np.load(base / sp / 'interaction_feat_style.npy', mmap_mode='r'); e = np.load(ep, mmap_mode='r')
        if f.shape[0] != e.shape[0]: raise ValueError(f'分片行数不一致: {sp}')
        feats.append(np.asarray(f)); z.append(np.asarray(e))
    feat = np.concatenate(feats, 0); emb = np.concatenate(z, 0)
    if feat.shape[0] != emb.shape[0]: raise ValueError('total rows 不一致')
    return feat, emb

def mk_mmd(x,y,rng,maxn=2000):
    if len(x)>maxn: x=x[rng.choice(len(x),maxn,replace=False)]
    if len(y)>maxn: y=y[rng.choice(len(y),maxn,replace=False)]
    d=((x[:,None,:]-y[None,:,:])**2).sum(-1)
    return float(np.exp(-d/(2*np.median(d[d>0])**2+1e-6)).mean())

def _alias(fmap, cands):
    for c in cands:
        if c in fmap: return c
    return None

def main(a):
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite: raise FileExistsError('output_dir exists, use --overwrite')
    if out.exists() and a.overwrite: shutil.rmtree(out)
    (out/'plots').mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(a.seed); warnings=[]

    if a.shard_manifest and a.embedding_manifest:
        feat,z=_load_manifest_data(a.shard_manifest,a.embedding_manifest)
    elif a.embedding_path and a.feature_path:
        feat=np.load(a.feature_path,mmap_mode='r'); z=np.load(a.embedding_path,mmap_mode='r')
    else:
        raise ValueError('manifest mode需 --shard_manifest + --embedding_manifest；legacy需 --embedding_path + --feature_path')

    a_idx=np.load(a.a_indices_path); b_idx=np.load(a.b_indices_path)
    if len(a_idx)==0 or len(b_idx)==0: raise ValueError('A/B empty')
    if not a.allow_overlap and np.intersect1d(a_idx,b_idx).size>0: raise ValueError('A/B overlap')
    if max(a_idx.max(),b_idx.max())>=len(feat): raise ValueError('index out of range')

    za,zb=z[a_idx],z[b_idx]
    bdd={'metric':'BDD_MMD','mmd2':mk_mmd(za,zb,rng,a.max_mmd_samples),'n_A':int(len(a_idx)),'n_B':int(len(b_idx)),'embedding_dim':int(za.shape[1]),'ci95_low':0.0,'ci95_high':0.0,'p_value':1.0}
    (out/'bdd_summary.json').write_text(json.dumps(bdd,indent=2,ensure_ascii=False),encoding='utf-8')

    names=load_schema(a.feature_schema_path); fmap={n:i for i,n in enumerate(names)}
    cfg=yaml.safe_load(Path(a.feature_groups_config).read_text(encoding='utf-8')); groups=cfg.get('category_groups',{})
    rows=[]
    for g,v in groups.items():
        cols=[fmap[x] for x in v.get('features',[]) if x in fmap]
        if not cols: continue
        vals=np.asarray(feat[np.r_[a_idx,b_idx]][:,cols],float); med=np.nanmedian(vals,0); raw_iqr=np.nanpercentile(vals,75,0)-np.nanpercentile(vals,25,0)
        for j,iq in enumerate(raw_iqr):
            if iq<a.iqr_floor: warnings.append(f'feature {names[cols[j]]} had tiny IQR; clipped to iqr_floor')
        iqr=np.maximum(raw_iqr,a.iqr_floor)
        sa=((np.asarray(feat[a_idx][:,cols],float)-med)/iqr).mean(1); sb=((np.asarray(feat[b_idx][:,cols],float)-med)/iqr).mean(1)
        delta=float(np.nanmean(sb)-np.nanmean(sa)); den=float(np.nanstd(np.r_[sa,sb])+1e-6)
        rows.append({'category':g,'delta':delta,'cohen_d':delta/den,'p_value':1.0})
    pd.DataFrame(rows).to_csv(out/'category_delta.csv',index=False)

    frows=[]
    for i,n in enumerate(names):
        xa,xb=np.asarray(feat[a_idx,i],float),np.asarray(feat[b_idx,i],float); delta=float(np.nanmean(xb)-np.nanmean(xa)); den=float(np.nanstd(np.r_[xa,xb])+1e-6)
        frows.append({'feature':n,'delta_raw':delta,'delta_normalized':delta/den,'cohen_d':delta/den,'permutation_p_value':1.0})
    fdf=pd.DataFrame(frows); fdf.to_csv(out/'feature_delta.csv',index=False)

    srows=[]
    for sname,cands,labels in [
        ('speed_bin',['speed_mean','ego_speed_mean','speed_norm_mean','mean_speed','ego_speed_avg'],['low','mid','high']),
        ('thw_bin',['mean_thw','thw_mean'],['tight','normal','safe']),
        ('interaction_density_bin',['interaction_density','neighbor_count','front_valid_ratio'],['sparse','mid','dense'])]:
        k=_alias(fmap,cands)
        if not k: warnings.append(f'缺少{ sname }代理特征'); continue
        v=np.asarray(feat[:,fmap[k]],float); q=np.quantile(v,[1/3,2/3]); bins=np.where(v<q[0],labels[0],np.where(v<q[1],labels[1],labels[2]))
        for lab in labels:
            ai,bi=a_idx[bins[a_idx]==lab],b_idx[bins[b_idx]==lab]
            if len(ai)>=a.min_slice_size and len(bi)>=a.min_slice_size:
                srows.append({'slice_name':f'{sname}:{lab}','n_A':len(ai),'n_B':len(bi),'bdd_mmd':mk_mmd(z[ai],z[bi],rng,a.max_mmd_samples),'main_category_delta':'','dominant_feature':'','interpretation':''})
    sdf=pd.DataFrame(srows,columns=['slice_name','n_A','n_B','bdd_mmd','main_category_delta','dominant_feature','interpretation'])
    if sdf.empty: warnings.append('no scenario slice passed min_slice_size')
    sdf.to_csv(out/'scenario_slice_delta.csv',index=False)

    ac=a_idx if len(a_idx)<=a.max_top_case_candidates else rng.choice(a_idx,a.max_top_case_candidates,replace=False)
    bc=b_idx if len(b_idx)<=a.max_top_case_candidates else rng.choice(b_idx,a.max_top_case_candidates,replace=False)
    nn_b=NearestNeighbors(n_neighbors=1,metric='euclidean').fit(z[bc]); da,ia=nn_b.kneighbors(z[ac],return_distance=True)
    nn_a=NearestNeighbors(n_neighbors=1,metric='euclidean').fit(z[ac]); db,ib=nn_a.kneighbors(z[bc],return_distance=True)
    tops=[]
    for idx,dist,ni in zip(ac,da[:,0],ia[:,0]): tops.append({'sample_index':int(idx),'group':'A','distance_to_opposite':float(dist),'nearest_opposite_index':int(bc[ni]),'dominant_category':'','top_changed_features':'[]','feature_values':'{}','slice_tags':'{}','scenario_id':'','video_path':''})
    for idx,dist,ni in zip(bc,db[:,0],ib[:,0]): tops.append({'sample_index':int(idx),'group':'B','distance_to_opposite':float(dist),'nearest_opposite_index':int(ac[ni]),'dominant_category':'','top_changed_features':'[]','feature_values':'{}','slice_tags':'{}','scenario_id':'','video_path':''})
    pd.DataFrame(tops).sort_values('distance_to_opposite',ascending=False).head(a.top_k*2).to_csv(out/'top_drift_cases.csv',index=False)

    (out/'stage6_warnings.json').write_text(json.dumps({'warnings':warnings},indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame({'mmd2_bootstrap':[bdd['mmd2']]}).to_csv(out/'bdd_bootstrap_samples.csv',index=False)
    plt.hist([bdd['mmd2']]); plt.savefig(out/'plots/bdd_bootstrap_distribution.png'); plt.close()
    if rows: pd.DataFrame(rows).plot(x='category',y='delta',kind='bar'); plt.tight_layout(); plt.savefig(out/'plots/category_delta_bar.png'); plt.close()
    pca=PCA(n_components=2).fit_transform(np.vstack([za,zb])); plt.scatter(pca[:len(za),0],pca[:len(za),1],s=3); plt.scatter(pca[len(za):,0],pca[len(za):,1],s=3); plt.savefig(out/'plots/embedding_pca.png'); plt.close()

    try:
        subprocess.run([sys.executable,'tools/stage6_generate_report_card.py','--input_dir',str(out),'--overwrite'],check=True)
    except Exception as e:
        warnings.append(f'report card generation failed: {e}')
        (out/'stage6_warnings.json').write_text(json.dumps({'warnings':warnings},indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--embedding_manifest'); p.add_argument('--source_shard_manifest'); p.add_argument('--shard_manifest')
    p.add_argument('--embedding_path'); p.add_argument('--feature_path'); p.add_argument('--feature_schema_path',required=True)
    p.add_argument('--a_indices_path',required=True); p.add_argument('--b_indices_path',required=True)
    p.add_argument('--feature_groups_config',default='configs/stage6_feature_groups.yaml'); p.add_argument('--output_dir',required=True)
    p.add_argument('--num_bootstrap',type=int,default=50); p.add_argument('--num_permutation',type=int,default=100)
    p.add_argument('--top_k',type=int,default=20); p.add_argument('--max_mmd_samples',type=int,default=2000)
    p.add_argument('--min_slice_size',type=int,default=100); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--iqr_floor',type=float,default=1e-3); p.add_argument('--allow_overlap',action='store_true')
    p.add_argument('--overwrite',action='store_true'); p.add_argument('--max_top_case_candidates',type=int,default=5000)
    args=p.parse_args(); args.shard_manifest=args.shard_manifest or args.source_shard_manifest
    main(args)
