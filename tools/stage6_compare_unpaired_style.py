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

def _rbf_mean_sqdist(a, b):
    d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return d

def _safe_bandwidth_from_pairwise(x, y):
    n = min(512, len(x), len(y))
    xs = x[np.random.choice(len(x), n, replace=False)] if len(x) > n else x
    ys = y[np.random.choice(len(y), n, replace=False)] if len(y) > n else y
    dxx = _rbf_mean_sqdist(xs, xs)
    dyy = _rbf_mean_sqdist(ys, ys)
    dxy = _rbf_mean_sqdist(xs, ys)
    vec = np.concatenate([dxx.ravel(), dyy.ravel(), dxy.ravel()])
    vec = vec[np.isfinite(vec) & (vec > 0)]
    if vec.size == 0:
        return 1.0
    med = float(np.median(np.sqrt(vec)))
    return med if np.isfinite(med) and med > 1e-8 else 1.0

def compute_mmd2(x, y, rng, max_samples):
    if len(x) > max_samples: x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples: y = y[rng.choice(len(y), max_samples, replace=False)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    base_bw = _safe_bandwidth_from_pairwise(x, y)
    bws = [base_bw * s for s in [0.25, 0.5, 1.0, 2.0, 4.0]]
    dxx = _rbf_mean_sqdist(x, x)
    dyy = _rbf_mean_sqdist(y, y)
    dxy = _rbf_mean_sqdist(x, y)
    mmd2 = 0.0
    for bw in bws:
        bw = max(float(bw), 1e-6)
        gamma = 1.0 / (2.0 * (bw ** 2))
        kxx = np.exp(-gamma * dxx).mean()
        kyy = np.exp(-gamma * dyy).mean()
        kxy = np.exp(-gamma * dxy).mean()
        mmd2 += (kxx + kyy - 2.0 * kxy)
    return float(mmd2 / len(bws))

def two_sided_permutation_pvalue(x_a, x_b, num_permutation, rng):
    x_a = np.asarray(x_a, float); x_b = np.asarray(x_b, float)
    x_a = x_a[np.isfinite(x_a)]; x_b = x_b[np.isfinite(x_b)]
    if len(x_a) == 0 or len(x_b) == 0 or num_permutation <= 0:
        return 1.0
    na = len(x_a)
    obs = float(np.mean(x_b) - np.mean(x_a))
    comb = np.concatenate([x_a, x_b])
    cnt = 0
    for _ in range(num_permutation):
        perm = rng.permutation(comb)
        pa = perm[:na]; pb = perm[na:]
        d = float(np.mean(pb) - np.mean(pa))
        if abs(d) >= abs(obs): cnt += 1
    return float((cnt + 1) / (num_permutation + 1))

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
    obs_mmd2 = compute_mmd2(za, zb, rng, a.max_mmd_samples)

    b_samples=[]
    for _ in range(a.num_bootstrap):
        ia=rng.choice(len(za),len(za),replace=True); ib=rng.choice(len(zb),len(zb),replace=True)
        b_samples.append(compute_mmd2(za[ia],zb[ib],rng,a.max_mmd_samples))
    bdf = pd.DataFrame({'mmd2_bootstrap':b_samples})
    bdf.to_csv(out/'bdd_bootstrap_samples.csv',index=False)

    perm_samples=[]
    zz=np.vstack([za,zb]); na=len(za)
    for _ in range(a.num_permutation):
        pidx=rng.permutation(len(zz)); pa=zz[pidx[:na]]; pb=zz[pidx[na:]]
        perm_samples.append(compute_mmd2(pa,pb,rng,a.max_mmd_samples))
    pdf = pd.DataFrame({'mmd2_permutation':perm_samples})
    pdf.to_csv(out/'bdd_permutation_samples.csv',index=False)
    pval=float((np.sum(np.asarray(perm_samples)>=obs_mmd2)+1)/(a.num_permutation+1)) if a.num_permutation>0 else 1.0
    ci_low=float(np.quantile(b_samples,0.025)) if b_samples else float('nan')
    ci_high=float(np.quantile(b_samples,0.975)) if b_samples else float('nan')

    bdd={'metric':'BDD_MMD','mmd2':obs_mmd2,'n_A':int(len(a_idx)),'n_B':int(len(b_idx)),'embedding_dim':int(za.shape[1]),'ci95_low':ci_low,'ci95_high':ci_high,'p_value':pval}
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
        p = two_sided_permutation_pvalue(sa, sb, a.num_permutation, rng)
        rows.append({'category':g,'delta':delta,'cohen_d':delta/den,'p_value':p})
    pd.DataFrame(rows).to_csv(out/'category_delta.csv',index=False)

    frows=[]
    for i,n in enumerate(names):
        xa,xb=np.asarray(feat[a_idx,i],float),np.asarray(feat[b_idx,i],float); delta=float(np.nanmean(xb)-np.nanmean(xa)); den=float(np.nanstd(np.r_[xa,xb])+1e-6)
        p = two_sided_permutation_pvalue(xa, xb, a.num_permutation, rng)
        frows.append({'feature':n,'delta_raw':delta,'delta_normalized':delta/den,'cohen_d':delta/den,'permutation_p_value':p})
    fdf=pd.DataFrame(frows); fdf.to_csv(out/'feature_delta.csv',index=False)

    srows=[]
    slice_bins = {}
    for sname,cands,labels in [
        ('speed_bin',['speed_mean','ego_speed_mean','speed_norm_mean','mean_speed','ego_speed_avg'],['low','mid','high']),
        ('thw_bin',['mean_thw','thw_mean'],['tight','normal','safe']),
        ('interaction_density_bin',['interaction_density','neighbor_count','front_valid_ratio'],['sparse','mid','dense'])]:
        k=_alias(fmap,cands)
        if not k: warnings.append(f'缺少{ sname }代理特征'); continue
        v=np.asarray(feat[:,fmap[k]],float); q=np.quantile(v,[1/3,2/3]); bins=np.where(v<q[0],labels[0],np.where(v<q[1],labels[1],labels[2]))
        slice_bins[sname]=bins
        for lab in labels:
            ai,bi=a_idx[bins[a_idx]==lab],b_idx[bins[b_idx]==lab]
            if len(ai)>=a.min_slice_size and len(bi)>=a.min_slice_size:
                srows.append({'slice_name':f'{sname}:{lab}','n_A':len(ai),'n_B':len(bi),'bdd_mmd':compute_mmd2(z[ai],z[bi],rng,a.max_mmd_samples),'main_category_delta':'','dominant_feature':'','interpretation':''})
    sdf=pd.DataFrame(srows,columns=['slice_name','n_A','n_B','bdd_mmd','main_category_delta','dominant_feature','interpretation'])
    if sdf.empty: warnings.append('no scenario slice passed min_slice_size')
    sdf.to_csv(out/'scenario_slice_delta.csv',index=False)

    pool=np.r_[a_idx,b_idx]
    pool_vals=np.asarray(feat[pool],float)
    feat_med=np.nanmedian(pool_vals,axis=0)
    feat_iqr=np.nanpercentile(pool_vals,75,axis=0)-np.nanpercentile(pool_vals,25,axis=0)
    for j,iq in enumerate(feat_iqr):
        if iq<a.iqr_floor: warnings.append(f'top_drift zscore feature {names[j]} had tiny IQR; clipped to iqr_floor')
    feat_scale=np.maximum(feat_iqr,a.iqr_floor)
    za_feat=(np.asarray(feat[a_idx],float)-feat_med)/feat_scale
    zb_feat=(np.asarray(feat[b_idx],float)-feat_med)/feat_scale
    med_za=np.nanmedian(za_feat,axis=0); med_zb=np.nanmedian(zb_feat,axis=0)

    cat_to_cols={g:[fmap[x] for x in v.get('features',[]) if x in fmap] for g,v in groups.items()}
    ac=a_idx if len(a_idx)<=a.max_top_case_candidates else rng.choice(a_idx,a.max_top_case_candidates,replace=False)
    bc=b_idx if len(b_idx)<=a.max_top_case_candidates else rng.choice(b_idx,a.max_top_case_candidates,replace=False)
    nn_b=NearestNeighbors(n_neighbors=1,metric='euclidean').fit(z[bc]); da,ia=nn_b.kneighbors(z[ac],return_distance=True)
    nn_a=NearestNeighbors(n_neighbors=1,metric='euclidean').fit(z[ac]); db,ib=nn_a.kneighbors(z[bc],return_distance=True)
    tops=[]

    def _mk_case(idx, group, dist, near_idx):
        zi=((np.asarray(feat[idx],float)-feat_med)/feat_scale)
        opp_med=med_zb if group=='A' else med_za
        dev=np.abs(zi-opp_med)
        topk=np.argsort(-dev)[:3]
        top_features=[names[i] for i in topk if np.isfinite(dev[i])]
        feature_values={names[i]: float(np.asarray(feat[idx,i],float)) for i in topk if np.isfinite(dev[i])}
        dominant=''
        best=-np.inf
        for g,cols in cat_to_cols.items():
            if not cols: continue
            score=float(np.nanmean(dev[cols]))
            if score>best:
                best=score; dominant=g
        tags={k:str(v[idx]) for k,v in slice_bins.items()} if slice_bins else {}
        return {'sample_index':int(idx),'group':group,'distance_to_opposite':float(dist),'nearest_opposite_index':int(near_idx),'dominant_category':dominant,'top_changed_features':json.dumps(top_features,ensure_ascii=False),'feature_values':json.dumps(feature_values,ensure_ascii=False),'slice_tags':json.dumps(tags,ensure_ascii=False),'scenario_id':'','video_path':''}

    for idx,dist,ni in zip(ac,da[:,0],ia[:,0]): tops.append(_mk_case(idx,'A',dist,bc[ni]))
    for idx,dist,ni in zip(bc,db[:,0],ib[:,0]): tops.append(_mk_case(idx,'B',dist,ac[ni]))
    pd.DataFrame(tops).sort_values('distance_to_opposite',ascending=False).head(a.top_k*2).to_csv(out/'top_drift_cases.csv',index=False)

    (out/'stage6_warnings.json').write_text(json.dumps({'warnings':warnings},indent=2,ensure_ascii=False),encoding='utf-8')
    plt.hist(b_samples if b_samples else [obs_mmd2]); plt.savefig(out/'plots/bdd_bootstrap_distribution.png'); plt.close()
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
