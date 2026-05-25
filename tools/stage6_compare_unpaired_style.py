#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil, subprocess, time, resource
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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

def _squared_l2_chunk(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)

def _safe_bandwidth_from_samples(x, y, rng, max_pairs=20000):
    comb = np.vstack([x, y])
    if len(comb) < 2:
        return 1.0
    n = len(comb)
    num_pairs = min(max_pairs, max(1, n * (n - 1) // 2))
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    neq = i != j
    if not np.any(neq):
        return 1.0
    i = i[neq]
    j = j[neq]
    pairwise = np.sqrt(np.sum((comb[i] - comb[j]) ** 2, axis=1))
    pairwise = pairwise[np.isfinite(pairwise) & (pairwise > 0)]
    if pairwise.size == 0:
        return 1.0
    med = float(np.median(pairwise))
    return med if np.isfinite(med) and med > 1e-8 else 1.0

def _rbf_kernel_mean(x, y, gamma, block_size):
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    total = 0.0
    count = 0
    for i in range(0, len(x), block_size):
        xb = x[i:i + block_size]
        d2 = _squared_l2_chunk(xb, y)
        total += float(np.exp(-gamma * d2).sum())
        count += d2.size
    return total / max(1, count)

def compute_mmd2(x, y, rng, max_samples, kernel_block_size):
    if len(x) > max_samples: x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples: y = y[rng.choice(len(y), max_samples, replace=False)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    bw = max(float(_safe_bandwidth_from_samples(x, y, rng)), 1e-6)
    gamma = 1.0 / (2.0 * (bw ** 2))
    kxx = _rbf_kernel_mean(x, x, gamma, kernel_block_size)
    kyy = _rbf_kernel_mean(y, y, gamma, kernel_block_size)
    kxy = _rbf_kernel_mean(x, y, gamma, kernel_block_size)
    return float(kxx + kyy - 2.0 * kxy)

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

def _resolve_alias(feature_map, candidates):
    for c in candidates:
        if c in feature_map:
            return c
    return None

def _build_quantile_bins(values, quantiles, labels):
    vv = np.asarray(values, float)
    ok = np.isfinite(vv)
    if ok.sum() < len(labels):
        return None, {'reason': 'insufficient_valid_rows', 'valid_rows': int(ok.sum())}
    qs = np.quantile(vv[ok], quantiles)
    if not np.all(np.isfinite(qs)):
        return None, {'reason': 'degenerate_quantiles', 'valid_rows': int(ok.sum())}
    if len(np.unique(qs)) < len(qs):
        return None, {'reason': 'degenerate_quantiles', 'valid_rows': int(ok.sum()), 'quantiles': [float(x) for x in qs]}
    bins = np.array(['unknown'] * len(vv), dtype=object)
    lo = -np.inf
    for q,lab in zip(list(qs)+[np.inf], labels):
        mask = ok & (vv >= lo) & (vv < q)
        bins[mask] = lab
        lo = q
    uniq = [u for u in np.unique(bins[ok]) if u != 'unknown']
    if len(uniq) < 2:
        return None, {'reason': 'all_samples_in_one_bin', 'valid_rows': int(ok.sum()), 'unique_bins': uniq}
    return bins, {'reason': 'ok', 'valid_rows': int(ok.sum())}

def _build_tertile_bins(values, labels):
    return _build_quantile_bins(values, [1/3, 2/3], labels)


def main(a):
    t0 = time.perf_counter()
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
    obs_mmd2 = compute_mmd2(za, zb, rng, a.max_mmd_samples, a.kernel_block_size)

    b_samples=[]
    for _ in tqdm(range(a.num_bootstrap), desc='Bootstrap sampling', leave=False):
        ia=rng.choice(len(za),len(za),replace=True); ib=rng.choice(len(zb),len(zb),replace=True)
        b_samples.append(compute_mmd2(za[ia],zb[ib],rng,a.max_mmd_samples,a.kernel_block_size))
    bdf = pd.DataFrame({'mmd2_bootstrap':b_samples})
    bdf.to_csv(out/'bdd_bootstrap_samples.csv',index=False)

    perm_samples=[]
    zz=np.vstack([za,zb]); na=len(za)
    for _ in tqdm(range(a.num_permutation), desc='Permutation testing', leave=False):
        pidx=rng.permutation(len(zz)); pa=zz[pidx[:na]]; pb=zz[pidx[na:]]
        perm_samples.append(compute_mmd2(pa,pb,rng,a.max_mmd_samples,a.kernel_block_size))
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
    for g,v in tqdm(groups.items(), desc='Processing category groups', leave=False):
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
    for i,n in tqdm(enumerate(names), desc='Processing features', total=len(names), leave=False):
        xa,xb=np.asarray(feat[a_idx,i],float),np.asarray(feat[b_idx,i],float); delta=float(np.nanmean(xb)-np.nanmean(xa)); den=float(np.nanstd(np.r_[xa,xb])+1e-6)
        p = two_sided_permutation_pvalue(xa, xb, a.num_permutation, rng)
        frows.append({'feature':n,'delta_raw':delta,'delta_normalized':delta/den,'cohen_d':delta/den,'permutation_p_value':p})
    fdf=pd.DataFrame(frows); fdf.to_csv(out/'feature_delta.csv',index=False)

    srows=[]
    slice_bins = {}
    slice_resolution = {'available_feature_count': int(len(names))}
    slice_cfgs = [
        ('speed_bin',['speed_mean','ego_speed_mean','speed_norm_mean','mean_speed','ego_speed_avg','speed_std','speed_norm_std'],['low','mid','high'],None),
        ('thw_bin',['mean_thw','thw_mean','min_thw','thw_min'],['tight','normal','safe'],([0.3,0.7],['tight','normal','safe'])),
        ('interaction_bin',['interaction_density','neighbor_count','neighbor_valid_count','front_valid_ratio','front_vehicle_valid_ratio','front_pressure_score'],['sparse','mid','dense'],None),
        ('lateral_activity_bin',['lane_change_count_proxy','lane_change_left_count_proxy','lane_change_right_count_proxy','rms_yaw_rate','rms_curvature','heading_change_total'],['low','mid','high'],None),
    ]
    for sname,cands,labels,fallback in slice_cfgs:
        k=_resolve_alias(fmap,cands)
        info={'resolved_feature':k,'candidates_tried':cands,'status':'pending','attempted_bins':{}}
        if not k:
            info['status']='missing'
            warnings.append(f'missing {sname} proxy feature')
            slice_resolution[sname]=info
            continue
        v=np.asarray(feat[:,fmap[k]],float)
        bins, meta = _build_tertile_bins(v, labels)
        if bins is None and fallback is not None:
            bins, meta = _build_quantile_bins(v, fallback[0], fallback[1])
            labels = fallback[1]
        if bins is None:
            info.update(meta)
            info['status']=meta.get('reason','failed')
            warnings.append(f'{sname} skipped: {info["status"]}')
            slice_resolution[sname]=info
            continue
        info['status']='ok'
        slice_bins[sname]=bins
        for lab in labels:
            ai,bi=a_idx[bins[a_idx]==lab],b_idx[bins[b_idx]==lab]
            info['attempted_bins'][lab]={'n_A':int(len(ai)),'n_B':int(len(bi))}
            if len(ai)>=a.min_slice_size and len(bi)>=a.min_slice_size:
                srows.append({'slice_name':f'{sname}:{lab}','n_A':len(ai),'n_B':len(bi),'bdd_mmd':compute_mmd2(z[ai],z[bi],rng,a.max_mmd_samples,a.kernel_block_size),'main_category_delta':'','dominant_feature':'','interpretation':''})
            else:
                warnings.append(f'{sname}:{lab} insufficient samples, A={len(ai)} B={len(bi)}')
        slice_resolution[sname]=info
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
        tags={k:str(v[idx]) for k,v in slice_bins.items() if idx < len(v) and v[idx] != 'unknown'}
        return {'sample_index':int(idx),'group':group,'distance_to_opposite':float(dist),'nearest_opposite_index':int(near_idx),'dominant_category':dominant,'top_changed_features':json.dumps(top_features,ensure_ascii=False),'feature_values':json.dumps(feature_values,ensure_ascii=False),'slice_tags':json.dumps(tags,ensure_ascii=False),'scenario_id':'','video_path':''}

    for idx,dist,ni in zip(ac,da[:,0],ia[:,0]): tops.append(_mk_case(idx,'A',dist,bc[ni]))
    for idx,dist,ni in zip(bc,db[:,0],ib[:,0]): tops.append(_mk_case(idx,'B',dist,ac[ni]))
    pd.DataFrame(tops).sort_values('distance_to_opposite',ascending=False).head(a.top_k*2).to_csv(out/'top_drift_cases.csv',index=False)

    (out/'stage6_warnings.json').write_text(json.dumps({'warnings':warnings,'slice_resolution':slice_resolution},indent=2,ensure_ascii=False),encoding='utf-8')
    plt.hist(b_samples if b_samples else [obs_mmd2]); plt.savefig(out/'plots/bdd_bootstrap_distribution.png'); plt.close()
    if rows: pd.DataFrame(rows).plot(x='category',y='delta',kind='bar'); plt.tight_layout(); plt.savefig(out/'plots/category_delta_bar.png'); plt.close()
    pca=PCA(n_components=2).fit_transform(np.vstack([za,zb])); plt.scatter(pca[:len(za),0],pca[:len(za),1],s=3); plt.scatter(pca[len(za):,0],pca[len(za):,1],s=3); plt.savefig(out/'plots/embedding_pca.png'); plt.close()

    try:
        subprocess.run([sys.executable,'tools/stage6_generate_report_card.py','--input_dir',str(out),'--overwrite'],check=True)
    except Exception as e:
        warnings.append(f'report card generation failed: {e}')
        (out/'stage6_warnings.json').write_text(json.dumps({'warnings':warnings,'slice_resolution':slice_resolution},indent=2,ensure_ascii=False),encoding='utf-8')

    runtime = time.perf_counter() - t0
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'runtime_stats.json').write_text(
        json.dumps({
            'runtime_seconds': float(runtime),
            'max_rss_kb': int(max_rss_kb),
            'max_rss_mb': float(max_rss_kb / 1024.0),
            'num_bootstrap': int(a.num_bootstrap),
            'num_permutation': int(a.num_permutation),
            'kernel_block_size': int(a.kernel_block_size),
        }, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

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
    p.add_argument('--kernel_block_size',type=int,default=512)
    args=p.parse_args(); args.shard_manifest=args.shard_manifest or args.source_shard_manifest
    main(args)
