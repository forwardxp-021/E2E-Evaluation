#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

EPS=1e-8

def load_optional(path):
    return np.load(path, allow_pickle=True) if path.exists() else None

def parse_meta_row(r):
    if isinstance(r, np.void) and r.dtype.names:
        d={k:r[k].item() if hasattr(r[k],'item') else r[k] for k in r.dtype.names}
    elif isinstance(r, dict): d=r
    else:
        vals=list(r) if isinstance(r,(list,tuple,np.ndarray)) else [r]
        d={"scenario_id":vals[0] if len(vals)>0 else '', "start":vals[1] if len(vals)>1 else 0, "window_len":vals[2] if len(vals)>2 else 0, "front_id":vals[3] if len(vals)>3 else ''}
    return {"scenario_id":str(d.get('scenario_id','')), "start":int(d.get('start',0)), "window_len":int(d.get('window_len',0)), "front_id":str(d.get('front_id',''))}

def pairdist(a,b,metric):
    if metric=='euclidean': return float(np.linalg.norm(a-b))
    den=max(np.linalg.norm(a)*np.linalg.norm(b),EPS); return float(1.0-np.dot(a,b)/den)

def style_metrics(traj, front, dt):
    x,y,vx,vy = traj[:,0],traj[:,1],traj[:,2],traj[:,3]
    speed=np.sqrt(vx*vx+vy*vy)
    accel=np.diff(speed,prepend=speed[0])/dt
    jerk=np.diff(accel,prepend=accel[0])/dt
    heading=np.unwrap(np.arctan2(vy,vx+EPS))
    yaw=np.diff(heading,prepend=heading[0])/dt
    curv=yaw/np.maximum(speed,EPS)
    out={"mean_speed":float(np.mean(speed)),"std_speed":float(np.std(speed)),"rms_accel":float(np.sqrt(np.mean(accel**2))),"rms_jerk":float(np.sqrt(np.mean(jerk**2))),"rms_yaw_rate_proxy":float(np.sqrt(np.mean(yaw**2))),"rms_curvature_proxy":float(np.sqrt(np.mean(curv**2))),"mean_gap":np.nan,"min_gap":np.nan,"mean_thw":np.nan,"min_thw":np.nan}
    if front is not None and np.asarray(front).ndim==2 and front.shape[1]>=2:
        gap=np.sqrt((front[:,0]-x)**2+(front[:,1]-y)**2)
        thw=gap/np.maximum(speed,EPS)
        out.update({"mean_gap":float(np.mean(gap)),"min_gap":float(np.min(gap)),"mean_thw":float(np.mean(thw)),"min_thw":float(np.min(thw))})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',required=True); ap.add_argument('--out_dir',required=True)
    ap.add_argument('--embedding',default='feat_style',choices=['feat_style','feat_style_raw','feat','feat_legacy'])
    ap.add_argument('--split',default='test',choices=['train','val','test','all'])
    ap.add_argument('--distance',default='euclidean',choices=['euclidean','cosine'])
    ap.add_argument('--topk',type=int,default=5); ap.add_argument('--dt',type=float,default=0.1)
    ap.add_argument('--expected_num_policies',type=int,default=3); ap.add_argument('--policy_of_interest',type=int,default=2)
    ap.add_argument('--save_per_source_reports',action='store_true'); ap.add_argument('--max_sources_for_debug',type=int,default=None)
    ap.add_argument('--projection',default='pca',choices=['none','pca','umap','both'])
    ap.add_argument('--allow_incomplete_metadata',action='store_true')
    args=ap.parse_args()
    d=Path(args.data_dir); o=Path(args.out_dir); o.mkdir(parents=True,exist_ok=True)

    emb=load_optional(d/f'{args.embedding}.npy'); traj=load_optional(d/'traj.npy'); front=load_optional(d/'front.npy')
    meta=load_optional(d/'meta.npy'); split=load_optional(d/'split.npy')
    source_index=load_optional(d/'source_index.npy'); source_key=load_optional(d/'source_key.npy')
    policy_id=load_optional(d/'policy_id.npy'); policy_name=load_optional(d/'policy_name.npy')
    if emb is None or traj is None: raise ValueError('Missing embedding or traj file')
    n=len(emb)
    if split is None: split=np.array(['all']*n)
    if policy_id is None and not args.allow_incomplete_metadata: raise ValueError('policy_id.npy required (or --allow_incomplete_metadata)')
    if source_index is None and source_key is None and not args.allow_incomplete_metadata: raise ValueError('source_index.npy required (or source_key/--allow_incomplete_metadata)')
    if policy_id is None: policy_id=np.array([-1]*n)
    if source_index is None:
        if source_key is None:
            source_key=np.array([f'row_{i}' for i in range(n)])
        _,inv=np.unique(source_key.astype(str),return_inverse=True); source_index=inv
    if source_key is None:
        if meta is not None: source_key=np.array(['{scenario_id}|{start}|{window_len}|{front_id}'.format(**parse_meta_row(r)) for r in meta])
        else: source_key=np.array([f'src_{int(s)}' for s in source_index])
    if policy_name is None: policy_name=np.array([f'policy_{int(p)}' for p in policy_id])

    df=pd.DataFrame({'index':np.arange(n),'source_index':source_index.astype(int),'source_key':source_key.astype(str),'policy_id':policy_id.astype(int),'policy_name':policy_name.astype(str),'split':split.astype(str)})
    if meta is not None:
        md=[parse_meta_row(r) for r in meta]; mdf=pd.DataFrame(md); df=pd.concat([df,mdf],axis=1)
    else:
        for c,v in [('scenario_id',''),('start',0),('window_len',0),('front_id','')]: df[c]=v

    mask=np.ones(n,dtype=bool) if args.split=='all' else (df['split'].values==args.split)
    eval_df=df[mask].copy(); eval_emb=emb[mask]

    hist_total=df.groupby('source_index').size().value_counts().sort_index().to_dict()
    hist_eval=eval_df.groupby('source_index').size().value_counts().sort_index().to_dict()

    complete=[]; incomplete=[]; per_rows=[]
    for sid,g in eval_df.groupby('source_index'):
        pset=set(g.policy_id.tolist())
        if len(pset)!=args.expected_num_policies: incomplete.append(int(sid)); continue
        if args.max_sources_for_debug and len(complete)>=args.max_sources_for_debug: break
        complete.append(int(sid)); r=g.iloc[0]
        idx_by={int(row.policy_id):int(row['index']) for _,row in g.iterrows()}
        pids=sorted(idx_by)
        if len(pids)!=3: continue
        d01=pairdist(emb[idx_by[pids[0]]],emb[idx_by[pids[1]]],args.distance); d02=pairdist(emb[idx_by[pids[0]]],emb[idx_by[pids[2]]],args.distance); d12=pairdist(emb[idx_by[pids[1]]],emb[idx_by[pids[2]]],args.distance)
        p2min=min(d02,d12)
        per_rows.append(dict(source_index=int(sid),source_key=r.source_key,scenario_id=r.scenario_id,start=int(r.start),window_len=int(r.window_len),front_id=r.front_id,split=args.split,idx_p0=idx_by.get(0,-1),idx_p1=idx_by.get(1,-1),idx_p2=idx_by.get(2,-1),d_p0_p1=d01,d_p0_p2=d02,d_p1_p2=d12,p2_min_distance_to_others=p2min,p2_mean_distance_to_others=float((d02+d12)/2),p2_separation_margin=float(p2min-d01),p2_farthest=bool(d02>d01 and d12>d01)))
    pair_df=pd.DataFrame(per_rows); pair_df.to_csv(o/'per_source_pairwise_distances.csv',index=False)

    # style per row
    style=[]
    for i in eval_df['index']:
        style.append({'index':int(i),**style_metrics(np.asarray(traj[i]),None if front is None else np.asarray(front[i]),args.dt)})
    style_df=eval_df.merge(pd.DataFrame(style),on='index');
    cols=['source_index','source_key','index','policy_id','policy_name','split','mean_speed','std_speed','rms_accel','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_gap','min_gap','mean_thw','min_thw']
    style_df[cols].to_csv(o/'per_source_style_summary.csv',index=False)

    # centroid classification
    train_mask=df['split'].values=='train' if args.split in ('test','val') else mask
    centroids={pid:np.mean(emb[(df.policy_id.values==pid)&train_mask],axis=0) for pid in sorted(df.policy_id.unique())}
    crec=[]
    for i,row in eval_df.iterrows():
        dists={pid:pairdist(emb[int(row['index'])],c,args.distance) for pid,c in centroids.items()}
        pred=min(dists,key=dists.get)
        rec={'index':int(row['index']),'source_index':int(row.source_index),'policy_id':int(row.policy_id),'policy_name':row.policy_name,'predicted_policy_id':int(pred),'predicted_policy_name':str(df[df.policy_id==pred].policy_name.iloc[0]),'correct':int(pred==row.policy_id)}
        for pid,v in dists.items(): rec[f'distance_to_p{int(pid)}_centroid']=v
        crec.append(rec)
    cdf=pd.DataFrame(crec); cdf.to_csv(o/'centroid_classification.csv',index=False)
    labels=sorted(df.policy_id.unique()); cm=confusion_matrix(cdf.policy_id,cdf.predicted_policy_id,labels=labels)
    pd.DataFrame(cm,index=[f'true_{x}' for x in labels],columns=[f'pred_{x}' for x in labels]).to_csv(o/'centroid_confusion_matrix.csv')

    # retrieval
    topk_rows=[]; qs=[]
    eval_idx=eval_df['index'].to_numpy()
    for qi in eval_idx:
        q=df.iloc[qi]; cands=eval_df[eval_df['index']!=qi]
        cands=cands[cands['source_index']!=q.source_index]
        if q.scenario_id!='': cands=cands[cands['scenario_id']!=q.scenario_id]
        if len(cands)==0: continue
        dvec=cdist(emb[[qi]],emb[cands['index'].to_numpy()],metric='euclidean' if args.distance=='euclidean' else 'cosine')[0]
        ord=np.argsort(dvec)[:args.topk]; sel=cands.iloc[ord]
        same=(sel.policy_id.values==q.policy_id)
        for rank,(ri,rv,sd) in enumerate(zip(sel['index'].values,dvec[ord],same),1):
            rr=df.iloc[int(ri)]; topk_rows.append({'query_index':int(qi),'query_source_index':int(q.source_index),'query_policy_id':int(q.policy_id),'query_policy_name':q.policy_name,'rank':rank,'retrieved_index':int(ri),'retrieved_source_index':int(rr.source_index),'retrieved_policy_id':int(rr.policy_id),'retrieved_policy_name':rr.policy_name,'distance':float(rv),'same_policy':bool(sd),'same_source':False,'same_scenario':bool(rr.scenario_id==q.scenario_id)})
        qs.append({'query_index':int(qi),'query_policy_id':int(q.policy_id),'hit_at_1_same_policy':int(same[0]) if len(same)>0 else 0,'hit_at_k_same_policy':int(np.any(same)),'num_same_policy_in_topk':int(np.sum(same)),'same_policy_fraction_topk':float(np.mean(same))})
    pd.DataFrame(topk_rows).to_csv(o/'global_retrieval_topk.csv',index=False)
    qdf=pd.DataFrame(qs); qdf.to_csv(o/'global_retrieval_summary.csv',index=False)

    # plots minimal
    if len(pair_df):
        plt.figure(); plt.boxplot([pair_df.d_p0_p1,pair_df.d_p0_p2,pair_df.d_p1_p2],labels=['d_p0_p1','d_p0_p2','d_p1_p2']); plt.savefig(o/'pairwise_distance_boxplot.png'); plt.close()
        plt.figure(); plt.hist(pair_df.p2_separation_margin,bins=30); plt.axvline(0,color='r'); plt.savefig(o/'p2_separation_margin_hist.png'); plt.close()
        plt.figure(); rate=float(np.mean(pair_df.p2_farthest)); plt.bar(['p2_farthest_rate'],[rate]); plt.ylim(0,1); plt.savefig(o/'p2_farthest_rate_bar.png'); plt.close()
    plt.figure(); plt.imshow(cm); plt.colorbar(); plt.savefig(o/'centroid_confusion_matrix.png'); plt.close()
    if len(qdf):
        plt.figure(); plt.bar(['hit@1','hit@k','mean_same_frac'],[qdf.hit_at_1_same_policy.mean(),qdf.hit_at_k_same_policy.mean(),qdf.same_policy_fraction_topk.mean()]); plt.ylim(0,1); plt.savefig(o/'retrieval_hit_at_k_bar.png'); plt.close()
    plt.figure(figsize=(10,6));
    for i,m in enumerate(['mean_speed','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw','min_thw'],1):
        plt.subplot(2,3,i); style_df.boxplot(column=m,by='policy_name'); plt.title(m); plt.suptitle('')
    plt.tight_layout(); plt.savefig(o/'policy_style_fingerprint_boxplot.png'); plt.close()
    if len(pair_df):
        sgrp=style_df.groupby(['source_index','policy_id'])[['rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw','mean_speed']].mean().reset_index()
        scat=[]; cor={}
        for metric in ['mean_speed','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw']:
            xs=[]; ys=[]
            for sid,g in sgrp.groupby('source_index'):
                mp={int(r.policy_id):r[metric] for _,r in g.iterrows()}
                if not all(k in mp for k in [0,1,2]): continue
                d=[pair_df[pair_df.source_index==sid].d_p0_p1.values[0],pair_df[pair_df.source_index==sid].d_p0_p2.values[0],pair_df[pair_df.source_index==sid].d_p1_p2.values[0]]
                sd=[abs(mp[0]-mp[1]),abs(mp[0]-mp[2]),abs(mp[1]-mp[2])]
                xs.extend(d); ys.extend(sd)
            cor[metric+'_delta_spearman']=float(spearmanr(xs,ys,nan_policy='omit').correlation) if len(xs)>2 else np.nan
        p2=pair_df.merge(sgrp[sgrp.policy_id==2][['source_index','rms_jerk']],on='source_index',how='left')
        plt.figure(); plt.scatter(p2.p2_mean_distance_to_others,p2.rms_jerk,s=10); plt.xlabel('p2 mean embedding distance'); plt.ylabel('p2 rms_jerk'); plt.savefig(o/'p2_distance_vs_style_delta_scatter.png'); plt.close()
    else: cor={}

    if args.projection in ('pca','both'):
        p=PCA(n_components=2).fit(eval_emb); z=p.transform(eval_emb)
        plt.figure();
        for pid,g in eval_df.groupby('policy_id'):
            m=(eval_df.policy_id.values==pid); plt.scatter(z[m,0],z[m,1],s=8,label=str(pid))
        plt.legend(); plt.xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.1f}%)'); plt.ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.1f}%)'); plt.title('Visualization only'); plt.savefig(o/'embedding_2d_population_pca.png'); plt.close()

    summary={
        'n_total_rows':n,'n_rows_after_split':int(mask.sum()),'n_unique_sources_total':int(df.source_index.nunique()),'n_unique_sources_after_split':int(eval_df.source_index.nunique()),
        'source_group_size_histogram_total':{str(k):int(v) for k,v in hist_total.items()},'source_group_size_histogram_after_split':{str(k):int(v) for k,v in hist_eval.items()},
        'n_complete_sources':len(complete),'n_incomplete_sources':len(incomplete),'expected_num_policies':args.expected_num_policies,
        'observed_policy_ids':sorted([int(x) for x in df.policy_id.unique()]),'policy_mapping':{str(int(pid)):str(name) for pid,name in df.groupby('policy_id').policy_name.first().items()},
        'policy_counts_total':{str(int(k)):int(v) for k,v in df.policy_id.value_counts().to_dict().items()},'policy_counts_after_split':{str(int(k)):int(v) for k,v in eval_df.policy_id.value_counts().to_dict().items()},
        'embedding_shape':list(emb.shape),'traj_shape':list(traj.shape),'front_shape':None if front is None else list(front.shape),'split_shape':list(np.asarray(split).shape),
        'warnings':[],'style_distance_correlation':cor,
        'centroid_accuracy_overall':float(cdf.correct.mean()) if len(cdf) else np.nan,'centroid_accuracy_by_policy':{str(int(k)):float(v) for k,v in cdf.groupby('policy_id').correct.mean().to_dict().items()} if len(cdf) else {},'centroid_chance_level':1.0/max(len(labels),1),
        'retrieval_hit_at_1':float(qdf.hit_at_1_same_policy.mean()) if len(qdf) else np.nan,'retrieval_hit_at_k':float(qdf.hit_at_k_same_policy.mean()) if len(qdf) else np.nan,'retrieval_mean_same_policy_count_topk':float(qdf.num_same_policy_in_topk.mean()) if len(qdf) else np.nan,'retrieval_mean_same_policy_fraction_topk':float(qdf.same_policy_fraction_topk.mean()) if len(qdf) else np.nan
    }
    if len(pair_df):
        for c in ['d_p0_p1','d_p0_p2','d_p1_p2']:
            a=pair_df[c].values; summary[c+'_stats']={'mean':float(np.mean(a)),'median':float(np.median(a)),'std':float(np.std(a)),'p25':float(np.percentile(a,25)),'p75':float(np.percentile(a,75))}
        summary.update({'p2_farthest_rate':float(np.mean(pair_df.p2_farthest)),'mean_p2_separation_margin':float(np.mean(pair_df.p2_separation_margin)),'median_p2_separation_margin':float(np.median(pair_df.p2_separation_margin)),'pct_p2_separation_margin_gt_0':float(np.mean(pair_df.p2_separation_margin>0))})

    (o/'population_summary.json').write_text(json.dumps(summary,indent=2))
    (o/'population_report.md').write_text('# Population Policy Separation Report\n\nPCA/UMAP are visualization only. Main evidence comes from high-dimensional distances, aligned within-source stats, classification, and retrieval.\n\nPolicies p0/p1/p2 are synthetic labels; human/public validation is future work.\n')

if __name__=='__main__': main()
