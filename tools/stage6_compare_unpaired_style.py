#!/usr/bin/env python3
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tools.train_context_behavior_embedding import ContextFlattenGRUEncoder

def load_schema(p):
    o=json.loads(Path(p).read_text(encoding='utf-8')); feats=o.get('features',[])
    return [f['name'] for f in sorted(feats,key=lambda x:int(x['index']))] if feats else o.get('feature_names',[])

def mk_mmd(X,Y,maxn=5000,seed=42):
    rng=np.random.default_rng(seed)
    if len(X)>maxn: X=X[rng.choice(len(X),maxn,replace=False)]
    if len(Y)>maxn: Y=Y[rng.choice(len(Y),maxn,replace=False)]
    Z=np.vstack([X,Y]); d=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=-1); med=np.median(d[d>0]) if np.any(d>0) else 1.0
    bws=np.clip(med*np.array([0.25,0.5,1,2,4]),1e-6,None)
    def k(A,B):
        D=((A[:,None,:]-B[None,:,:])**2).sum(-1); return sum(np.exp(-D/(2*b*b)) for b in bws)/len(bws)
    Kxx,Kyy,Kxy=k(X,X),k(Y,Y),k(X,Y)
    mmd=float(Kxx.mean()+Kyy.mean()-2*Kxy.mean())
    return mmd

def emb(context,ckpt,device,bz):
    c=torch.load(ckpt,map_location='cpu'); model=ContextFlattenGRUEncoder(context.shape[-1],embedding_dim=int(c.get('embedding_dim',64))); model.load_state_dict(c['model'],strict=False)
    dev=torch.device(device if (device!='cuda' or torch.cuda.is_available()) else 'cpu'); model.to(dev).eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(context),bz): out.append(model(torch.from_numpy(context[i:i+bz]).float().to(dev)).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)

def main(a):
    out=Path(a.output_dir); (out/'plots').mkdir(parents=True,exist_ok=True)
    feat=np.load(a.feature_path,mmap_mode='r'); ctx=np.load(a.context_traj_path,mmap_mode='r')
    A=np.load(a.a_indices_path); B=np.load(a.b_indices_path)
    names=load_schema(a.feature_schema_path); m={n:i for i,n in enumerate(names)}
    cfg=yaml.safe_load(Path(a.feature_groups_config).read_text(encoding='utf-8')); aliases=cfg.get('feature_aliases',{}); groups=cfg.get('category_groups',{})
    ZA,ZB=emb(np.asarray(ctx[A],dtype=np.float32),a.encoder_ckpt,a.device,a.batch_size),emb(np.asarray(ctx[B],dtype=np.float32),a.encoder_ckpt,a.device,a.batch_size)
    mmd=mk_mmd(ZA,ZB,a.max_mmd_samples,a.seed)
    bs=[]; rng=np.random.default_rng(a.seed)
    for _ in range(a.num_bootstrap): bs.append(mk_mmd(ZA[rng.choice(len(ZA),len(ZA),replace=True)],ZB[rng.choice(len(ZB),len(ZB),replace=True)],a.max_mmd_samples,a.seed))
    mix=np.vstack([ZA,ZB]); nA=len(ZA); perm=[]
    for _ in range(a.num_permutation):
        p=rng.permutation(len(mix)); perm.append(mk_mmd(mix[p[:nA]],mix[p[nA:]],a.max_mmd_samples,a.seed))
    bdd={'metric':'BDD_MMD','mmd2':float(mmd),'ci95_low':float(np.percentile(bs,2.5)),'ci95_high':float(np.percentile(bs,97.5)),'p_value':float((np.sum(np.array(perm)>=mmd)+1)/(len(perm)+1)),'n_A':int(len(A)),'n_B':int(len(B)),'embedding_dim':int(ZA.shape[1])}
    (out/'bdd_summary.json').write_text(json.dumps(bdd,indent=2),encoding='utf-8')
    pd.DataFrame({'mmd2_bootstrap':bs}).to_csv(out/'bdd_bootstrap_samples.csv',index=False)

    rows=[]; group_map={}
    for g,v in groups.items():
        fs=[]
        for f in v.get('features',[]):
            cands=[f]+aliases.get(f,[]); hit=next((x for x in cands if x in m),None)
            if hit: fs.append((f,hit,m[hit])); group_map[hit]=g
        if not fs: continue
        vals=np.asarray(feat[np.r_[A,B]][:,[x[2] for x in fs]],dtype=float); med=np.nanmedian(vals,0); iqr=np.nanpercentile(vals,75,0)-np.nanpercentile(vals,25,0)+1e-6
        SA=np.asarray(feat[A][:,[x[2] for x in fs]],dtype=float); SB=np.asarray(feat[B][:,[x[2] for x in fs]],dtype=float)
        ZA1=(SA-med)/iqr; ZB1=(SB-med)/iqr
        for j,(f,hit,_) in enumerate(fs):
            if f in v.get('lower_is_better',[]): ZA1[:,j]*=-1; ZB1[:,j]*=-1
        sA,sB=ZA1.mean(1),ZB1.mean(1); delta=float(np.mean(sB)-np.mean(sA))
        rows.append({'category':g,'n_features':len(fs),'mean_A':float(np.mean(sA)),'mean_B':float(np.mean(sB)),'delta':delta,'p_value':float((np.sum(np.array([np.mean(np.random.permutation(np.r_[sA,sB])[len(sA):])-np.mean(np.random.permutation(np.r_[sA,sB])[:len(sA)]) for _ in range(100)])>=delta)+1)/101)})
    cdf=pd.DataFrame(rows); cdf.to_csv(out/'category_delta.csv',index=False)

    frows=[]
    for i,n in enumerate(names):
        a1,b1=np.asarray(feat[A,i],float),np.asarray(feat[B,i],float)
        den=np.nanstd(np.r_[a1,b1])+1e-6
        frows.append({'feature':n,'mean_A':np.nanmean(a1),'mean_B':np.nanmean(b1),'median_A':np.nanmedian(a1),'median_B':np.nanmedian(b1),'delta_raw':np.nanmean(b1)-np.nanmean(a1),'delta_normalized':(np.nanmean(b1)-np.nanmean(a1))/den,'relative_change_percent':100*(np.nanmean(b1)-np.nanmean(a1))/(abs(np.nanmean(a1))+1e-6),'cohen_d':(np.nanmean(b1)-np.nanmean(a1))/den,'permutation_p_value':1.0,'group':group_map.get(n,'')})
    fdf=pd.DataFrame(frows); fdf.to_csv(out/'feature_delta.csv',index=False)

    srows=[]
    if 'speed_mean' in m:
        sp=np.asarray(feat[:,m['speed_mean']],float); q=np.quantile(sp,[1/3,2/3]); bins=np.where(sp<q[0],'low',np.where(sp<q[1],'mid','high'))
        for b in ['low','mid','high']:
            aidx=A[bins[A]==b]; bidx=B[bins[B]==b]
            if len(aidx)>=a.min_slice_size and len(bidx)>=a.min_slice_size:
                srows.append({'slice_name':f'speed_bin:{b}','n_A':len(aidx),'n_B':len(bidx),'bdd_mmd':mk_mmd(emb(np.asarray(ctx[aidx],np.float32),a.encoder_ckpt,a.device,a.batch_size),emb(np.asarray(ctx[bidx],np.float32),a.encoder_ckpt,a.device,a.batch_size),2000,a.seed),'main_category_delta':cdf.sort_values('delta',key=np.abs,ascending=False).iloc[0]['category'] if not cdf.empty else '','dominant_feature_delta':fdf.sort_values('delta_normalized',key=np.abs,ascending=False).iloc[0]['feature'] if not fdf.empty else ''})
    pd.DataFrame(srows).to_csv(out/'scenario_slice_delta.csv',index=False)

    D=np.linalg.norm(ZA[:,None,:]-ZB[None,:,:],axis=-1)
    ta=np.argsort(D.min(1))[-a.top_k:]; tb=np.argsort(D.min(0))[-a.top_k:]
    tops=[]
    for i in ta: tops.append({'sample_index':int(A[i]),'group':'A','distance_to_opposite':float(D[i].min()),'nearest_opposite_index':int(B[D[i].argmin()]),'dominant_category':'','top_changed_features':'','feature_values':'','slice_tags':'','scenario_id':'','video_path':''})
    for j in tb: tops.append({'sample_index':int(B[j]),'group':'B','distance_to_opposite':float(D[:,j].min()),'nearest_opposite_index':int(A[D[:,j].argmin()]),'dominant_category':'','top_changed_features':'','feature_values':'','slice_tags':'','scenario_id':'','video_path':''})
    pd.DataFrame(tops).sort_values('distance_to_opposite',ascending=False).to_csv(out/'top_drift_cases.csv',index=False)

    if not cdf.empty: cdf.plot(x='category',y='delta',kind='bar'); plt.tight_layout(); plt.savefig(out/'plots/category_delta_bar.png'); plt.close()
    if not fdf.empty: t=fdf.reindex(fdf.delta_normalized.abs().sort_values(ascending=False).head(20).index); t.plot(x='feature',y='delta_normalized',kind='bar'); plt.tight_layout(); plt.savefig(out/'plots/feature_delta_bar_top20.png'); plt.close()
    plt.hist(bs,bins=30); plt.axvline(mmd,color='r'); plt.tight_layout(); plt.savefig(out/'plots/bdd_bootstrap_distribution.png'); plt.close()
    pca=PCA(n_components=2).fit_transform(np.vstack([ZA,ZB])); plt.scatter(pca[:len(ZA),0],pca[:len(ZA),1],s=4,label='A'); plt.scatter(pca[len(ZA):,0],pca[len(ZA):,1],s=4,label='B'); plt.legend(); plt.tight_layout(); plt.savefig(out/'plots/embedding_pca.png'); plt.close()
    subprocess.run([sys.executable,'tools/stage6_generate_report_card.py','--input_dir',str(out)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--context_traj_path',required=True); p.add_argument('--context_mask_path'); p.add_argument('--context_mask_window_path')
    p.add_argument('--feature_path',required=True); p.add_argument('--feature_schema_path',required=True); p.add_argument('--encoder_ckpt',required=True)
    p.add_argument('--a_indices_path',required=True); p.add_argument('--b_indices_path',required=True)
    p.add_argument('--feature_groups_config',default='configs/stage6_feature_groups.yaml'); p.add_argument('--output_dir',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--batch_size',type=int,default=512); p.add_argument('--num_bootstrap',type=int,default=200)
    p.add_argument('--num_permutation',type=int,default=500); p.add_argument('--top_k',type=int,default=20); p.add_argument('--max_mmd_samples',type=int,default=5000)
    p.add_argument('--min_slice_size',type=int,default=100); p.add_argument('--seed',type=int,default=42)
    main(p.parse_args())
