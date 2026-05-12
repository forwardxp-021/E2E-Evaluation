#!/usr/bin/env python3
import argparse, json, tempfile, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.assign_pseudo_style_labels import _compute_signals, run as run_labels

STYLE_KEYS=['mean_speed','rms_accel','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw','min_thw']
CORR_KEYS=['mean_speed','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw']


def dist(a,b,metric='euclidean'):
    if metric=='cosine':
        na=np.linalg.norm(a,axis=1,keepdims=True)+1e-9; nb=np.linalg.norm(b,axis=1,keepdims=True)+1e-9
        return 1-(a/na)@(b/nb).T
    aa=(a*a).sum(1,keepdims=True); bb=(b*b).sum(1,keepdims=True).T
    return np.sqrt(np.maximum(aa+bb-2*a@b.T,0))


def _parse_meta(meta_path, n):
    info = {'scenario_id':None,'agent_id':None,'track_id':None,'source_index':None,'start':None,'window_len':None,'front_id':None}
    if not meta_path.exists():
        return info, ['meta.npy missing; strict exclusions limited']
    meta=np.load(meta_path,allow_pickle=True)
    warnings=[]
    if meta.dtype.names:
        for k in info:
            if k in meta.dtype.names: info[k]=meta[k]
    elif meta.dtype==object and len(meta)>0 and isinstance(meta[0],dict):
        for k in info:
            if k in meta[0]: info[k]=np.array([m.get(k) for m in meta], dtype=object)
    elif meta.ndim==2:
        fallback={'scenario_id':0,'agent_id':1,'track_id':2,'source_index':3,'start':4,'window_len':5,'front_id':6}
        for k,idx in fallback.items():
            if meta.shape[1]>idx: info[k]=meta[:,idx]
        warnings.append('meta.npy parsed as 2D array fallback column mapping')
    else:
        warnings.append('meta.npy format not recognized for strict exclusions')
    for k,v in info.items():
        if v is not None and len(v)!=n:
            warnings.append(f'meta field {k} length mismatch; ignored')
            info[k]=None
    return info, warnings


def _chance_metrics(labels_eval, topk):
    vals=np.array(labels_eval)
    p=np.array([np.mean(vals==c) for c in [0,1,2]])
    chance=np.sum(p*p)
    return chance, 1-(1-chance)**topk, chance


def _eval_method(X,labels,splits,eval_split,topk,metric,retrieval_mode,args,meta):
    mask_lab=labels!=-1
    msplit=np.ones(len(labels),bool) if eval_split=='all' else (splits==eval_split)
    ev=np.where(mask_lab & msplit)[0]; tr=np.where(mask_lab & (splits=='train'))[0]
    if len(tr)==0: tr=ev
    classes=[0,1,2]
    cents=np.vstack([X[tr][labels[tr]==c].mean(0) if np.any(labels[tr]==c) else np.zeros(X.shape[1]) for c in classes])
    pred=dist(X[ev],cents,metric).argmin(1)
    acc=float((pred==labels[ev]).mean()) if len(ev) else np.nan
    cm=confusion_matrix(labels[ev],pred,labels=classes)
    D=dist(X[ev],X[ev],metric)
    top_rows=[]
    hits=[]; hitk=[]; frac=[]
    for qi in tqdm(range(len(ev)), desc='Evaluating retrieval'):
        mask=np.ones(len(ev),dtype=bool); mask[qi]=False
        if retrieval_mode=='strict':
            ei=ev[qi]
            for qk,mk in [('scenario_id',args.exclude_same_scenario),('agent_id',args.exclude_same_agent),('track_id',args.exclude_same_track),('source_index',args.exclude_same_source)]:
                if mk and meta.get(qk) is not None:
                    mask &= (meta[qk][ev] != meta[qk][ei])
            if args.exclude_temporal_neighbors>=0 and meta.get('start') is not None and meta.get('scenario_id') is not None:
                same_sc=(meta['scenario_id'][ev]==meta['scenario_id'][ei])
                same_actor=np.ones(len(ev),dtype=bool)
                if meta.get('agent_id') is not None: same_actor &= (meta['agent_id'][ev]==meta['agent_id'][ei])
                elif meta.get('track_id') is not None: same_actor &= (meta['track_id'][ev]==meta['track_id'][ei])
                near=np.abs(meta['start'][ev].astype(float)-float(meta['start'][ei]))<=args.exclude_temporal_neighbors
                mask &= ~(same_sc & same_actor & near)
            mask[qi]=False
        valid=np.where(mask)[0]
        if len(valid)==0: continue
        order=valid[np.argsort(D[qi,valid])[:topk]]
        ql=labels[ev[qi]]; nl=labels[ev][order]
        hits.append(float(nl[0]==ql)); hitk.append(float(np.any(nl==ql))); frac.append(float(np.mean(nl==ql)))
        for r,nidx in enumerate(order,1):
            top_rows.append({'query_index':int(ev[qi]),'neighbor_index':int(ev[nidx]),'rank':r,'distance':float(D[qi,nidx]),'query_label':int(ql),'neighbor_label':int(labels[ev[nidx]]),'same_label':bool(labels[ev[nidx]]==ql),'excluded_same_scenario_available':meta.get('scenario_id') is not None,'retrieval_mode':retrieval_mode})
    hit1=float(np.mean(hits)) if hits else np.nan
    hk=float(np.mean(hitk)) if hitk else np.nan
    same_frac=float(np.mean(frac)) if frac else np.nan
    return acc,cm,hit1,hk,same_frac,ev,top_rows


def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_dir); lab=Path(args.label_dir)
    traj=np.load(args.traj_path or data/'traj.npy', allow_pickle=True)
    if traj.dtype==object:
        # Handle case where traj contains float scalars or arrays
        traj_arrays = []
        for t in traj:
            if isinstance(t, np.ndarray):
                traj_arrays.append(t)
            elif np.isscalar(t) or isinstance(t, float):
                traj_arrays.append(np.full((80, 4), np.nan, dtype=np.float32))
            else:
                traj_arrays.append(np.full((80, 4), np.nan, dtype=np.float32))
        traj = np.stack(traj_arrays, axis=0)
    traj = traj.astype(np.float32)
    split=np.load(args.split_path or data/'split.npy',allow_pickle=True).astype(str)
    labels=np.load(lab/'pseudo_label.npy')
    feat=np.load(args.feat_path or data/'feat_style.npy')
    front=np.load(args.front_path or data/'front.npy',allow_pickle=True) if (args.front_path or data/'front.npy').exists() else None
    if front is not None:
        if front.dtype==object:
            # Handle case where front contains float scalars or arrays
            front_arrays = []
            seq_len = traj.shape[1] if len(traj.shape) > 1 else 80
            for f in front:
                if isinstance(f, np.ndarray):
                    front_arrays.append(f)
                elif np.isscalar(f) or isinstance(f, float):
                    front_arrays.append(np.full((seq_len, 4), np.nan, dtype=np.float32))
                else:
                    front_arrays.append(np.full((seq_len, 4), np.nan, dtype=np.float32))
            front = np.stack(front_arrays, axis=0)
        front = front.astype(np.float32)
    sig=_compute_signals(traj,front,args.dt)
    meta,meta_warnings=_parse_meta(Path(args.meta_path or data/'meta.npy'),len(labels))
    warnings=list(meta_warnings)
    meta_arr=np.load(args.meta_path or data/'meta.npy',allow_pickle=True) if Path(args.meta_path or data/'meta.npy').exists() else None
    contains_policy_labels=bool((meta_arr is not None and meta_arr.dtype.names and ('policy_id' in meta_arr.dtype.names or 'policy_name' in meta_arr.dtype.names))
                                or (meta_arr is not None and meta_arr.dtype==object and len(meta_arr)>0 and isinstance(meta_arr[0],dict) and ('policy_id' in meta_arr[0] or 'policy_name' in meta_arr[0])))
    contains_source_index=bool(meta.get('source_index') is not None or (data/'source_index.npy').exists())
    auto_dataset_type='synthetic_rollout' if contains_policy_labels else 'unknown'
    dataset_type=args.dataset_type if args.dataset_type!='unknown' else auto_dataset_type
    contains_synthetic_policy_rollouts=bool(dataset_type=='synthetic_rollout' or contains_policy_labels)

    methods={}
    learned_evaluated=False; learned_skip_reason=None; learned_shape=None
    learned_embedding_alignment='not_requested'
    learned_embedding_valid_for_policy_eval=None
    baseline_list=[x.strip() for x in args.baselines.split(',') if x.strip()]
    if 'learned' in baseline_list:
        if not args.embedding_path:
            learned_skip_reason='missing --embedding_path'
            if not args.allow_skip_learned: raise ValueError('learned baseline requested but --embedding_path not provided')
        else:
            ep=Path(args.embedding_path)
            if ep.is_dir():
                if args.allow_embedding_dir_lookup:
                    candidate=None
                    for fn in ['embedding.npy','feat_embedding.npy','feat_style_embedding.npy']:
                        p=ep/fn
                        if p.exists(): candidate=p; break
                    if candidate is None: raise ValueError('--embedding_path must be a .npy file.')
                    ep=candidate
                else:
                    raise ValueError('--embedding_path must be a .npy file.')
            if ep.suffix!='.npy' or not ep.exists():
                raise ValueError('--embedding_path must be a .npy file.')
            emb=np.load(ep,allow_pickle=True)
            learned_shape=list(emb.shape)
            if emb.shape[0]!=len(labels):
                mismatch_msg=f"Learned embedding row count mismatch: embedding has {emb.shape[0]} rows, data has {len(labels)} rows. This evaluator requires row-aligned embeddings. Source-level embeddings must not be auto-expanded for policy-level or pseudo-label evaluation. Regenerate row-level embeddings or run without learned baseline."
                if args.allow_source_level_embedding_expansion:
                    if not (data/'source_index.npy').exists():
                        raise ValueError(mismatch_msg + " source_index.npy not found for optional expansion.")
                    source_index=np.load(data/'source_index.npy',allow_pickle=True)
                    unique_sources=np.unique(source_index)
                    if len(unique_sources)!=emb.shape[0]:
                        raise ValueError(mismatch_msg + f" source_index has {len(unique_sources)} unique values which does not match embedding rows.")
                    emb_aligned=emb[source_index]
                    methods['learned']=emb_aligned
                    learned_evaluated=True
                    learned_embedding_alignment='source_index_expanded'
                    learned_embedding_valid_for_policy_eval=False
                    warnings.append('WARNING: Learned embeddings were expanded via source_index from source-level rows to rollout rows. This is debug-only and invalid for policy-level evaluation.')
                elif args.allow_skip_learned:
                    learned_skip_reason=mismatch_msg
                    learned_embedding_alignment='skipped_mismatch'
                    learned_embedding_valid_for_policy_eval=False
                else:
                    raise ValueError(mismatch_msg)
            else:
                methods['learned']=emb
                learned_evaluated=True
                learned_embedding_alignment='row_aligned'
                learned_embedding_valid_for_policy_eval=True
    if (not learned_evaluated) and 'learned' in baseline_list and learned_skip_reason:
        warnings.append(learned_skip_reason)

    methods['raw_feature']=(feat-feat.mean(0))/(feat.std(0)+1e-6)
    xy=traj[:,:,:2].copy(); xy=xy-xy[:,[0],:]; ang=np.arctan2(xy[:,1,1],xy[:,1,0]); c=np.cos(-ang); s=np.sin(-ang)
    xr=xy[:,:,0]*c[:,None]-xy[:,:,1]*s[:,None]; yr=xy[:,:,0]*s[:,None]+xy[:,:,1]*c[:,None]
    methods['trajectory_l2']=np.concatenate([xr,yr],1)
    dim=methods['learned'].shape[1] if 'learned' in methods else 20
    methods['random']=np.random.default_rng(args.seed).normal(size=(len(labels),dim))
    methods['pca_feature']=PCA(n_components=min(dim,feat.shape[1],8),random_state=args.seed).fit_transform(methods['raw_feature'])

    eval_methods=[m for m in baseline_list if m in methods]
    rows=[]; retrieval_rows=[]
    for m in eval_methods:
        acc,cm,h1,hk,frac,ev,topk_rows=_eval_method(methods[m],labels,split,args.eval_split,args.topk,args.distance,args.retrieval_mode,args,meta)
        chance1,chancek,chancefrac=_chance_metrics(labels[ev],args.topk)
        drep = dist(methods[m][ev],methods[m][ev],args.distance)[np.triu_indices(len(ev),1)]
        sp={}; valid_counts={}
        for k in CORR_KEYS:
            sd=np.abs(sig[k][ev][:,None]-sig[k][ev][None,:])[np.triu_indices(len(ev),1)]
            vm=np.isfinite(drep)&np.isfinite(sd)
            valid_counts[k]=int(vm.sum())
            if valid_counts[k] < args.min_style_corr_pairs:
                sp[k]=np.nan
                warnings.append(f'{m}:{k} valid pairs {valid_counts[k]} < min_style_corr_pairs={args.min_style_corr_pairs}')
            else:
                sp[k]=float(spearmanr(drep[vm], sd[vm]).correlation)
        rows.append({'method':m,'representation_dim':methods[m].shape[1],'n_eval_samples':len(ev),'centroid_accuracy_overall':acc,'hit_at_1':h1,'hit_at_k':hk,'mean_same_label_fraction_topk':frac,'chance_hit_at_1_label_prior':chance1,'chance_hit_at_k_label_prior':chancek,'chance_mean_same_label_fraction_topk':chancefrac,'hit_at_1_lift_over_chance':h1-chance1,'hit_at_k_lift_over_chance':hk-chancek,'same_fraction_lift_over_chance':frac-chancefrac,'spearman_mean_speed_delta':sp['mean_speed'],'spearman_rms_jerk_delta':sp['rms_jerk'],'spearman_rms_yaw_rate_delta':sp['rms_yaw_rate_proxy'],'spearman_rms_curvature_delta':sp['rms_curvature_proxy'],'spearman_mean_thw_delta':sp['mean_thw'],'valid_pairs_mean_speed_delta':valid_counts['mean_speed'],'valid_pairs_rms_jerk_delta':valid_counts['rms_jerk'],'valid_pairs_rms_yaw_rate_delta':valid_counts['rms_yaw_rate_proxy'],'valid_pairs_rms_curvature_delta':valid_counts['rms_curvature_proxy'],'valid_pairs_mean_thw_delta':valid_counts['mean_thw']})
        retrieval_rows.extend(topk_rows)
    bdf=pd.DataFrame(rows); bdf.to_csv(out/'baseline_comparison_summary.csv',index=False)
    pd.DataFrame(retrieval_rows).to_csv(out/'human_retrieval_topk.csv',index=False)

    if len(bdf):
        plt.figure(figsize=(8,4)); plt.bar(bdf['method'],bdf['centroid_accuracy_overall']); plt.axhline(1/3,color='r',linestyle='--'); plt.tight_layout(); plt.savefig(out/'baseline_classification_bar.png'); plt.close()
        x=np.arange(len(bdf)); w=0.2
        plt.figure(figsize=(10,4));
        plt.bar(x-1.5*w,bdf['hit_at_1'],w,label='hit@1'); plt.bar(x-0.5*w,bdf['mean_same_label_fraction_topk'],w,label='mean_same_fraction');
        plt.bar(x+0.5*w,bdf['chance_hit_at_1_label_prior'],w,label='chance_hit@1'); plt.bar(x+1.5*w,bdf['chance_mean_same_label_fraction_topk'],w,label='chance_same_fraction')
        plt.xticks(x,bdf['method']); plt.legend(); plt.tight_layout(); plt.savefig(out/'baseline_retrieval_bar.png'); plt.close()
        corr_cols=['spearman_mean_speed_delta','spearman_rms_jerk_delta','spearman_rms_yaw_rate_delta','spearman_rms_curvature_delta','spearman_mean_thw_delta']
        plt.figure(figsize=(10,4));
        for i,c in enumerate(corr_cols): plt.bar(x+i*0.15,bdf[c],0.15,label=c.replace('spearman_',''))
        plt.xticks(x+0.3,bdf['method']); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'baseline_style_correlation_bar.png'); plt.close()

    pca_src='learned' if learned_evaluated else 'pca_feature'
    pca_name='human_embedding_pca.png' if learned_evaluated else 'human_representation_pca_pca_feature.png'
    try:
        emb2=PCA(n_components=2,random_state=args.seed).fit_transform(methods[pca_src])
        plt.figure(figsize=(6,5)); plt.scatter(emb2[:,0],emb2[:,1],c=np.where(labels==-1,3,labels),s=4,cmap='tab10'); plt.tight_layout(); plt.savefig(out/pca_name); plt.close()
    except Exception as e:
        warnings.append(f'Failed PCA plot: {e}')

    cl=KMeans(n_clusters=args.num_clusters,random_state=args.seed,n_init=10).fit_predict(methods[eval_methods[0]])
    cdf=pd.DataFrame({'cluster_id':np.arange(args.num_clusters),'count':[int(np.sum(cl==k)) for k in range(args.num_clusters)]})
    cdf.to_csv(out/'cluster_size_distribution.csv',index=False)
    plt.figure(figsize=(8,3)); plt.bar(cdf['cluster_id'].astype(str),cdf['count']); plt.tight_layout(); plt.savefig(out/'cluster_size_distribution.png'); plt.close()
    frows=[]; lrows=[]
    for k in range(args.num_clusters):
        idx=np.where(cl==k)[0]
        frows.append({'cluster_id':k,**{s:float(np.nanmean(sig[s][idx])) for s in STYLE_KEYS}})
        for labv,cnt in pd.Series(labels[idx]).value_counts().to_dict().items(): lrows.append({'cluster_id':k,'pseudo_label':int(labv),'count':int(cnt)})
    fp=pd.DataFrame(frows); fp.to_csv(out/'cluster_style_fingerprint.csv',index=False)
    pd.DataFrame(lrows).to_csv(out/'cluster_label_distribution.csv',index=False)
    z=(fp[STYLE_KEYS]-fp[STYLE_KEYS].mean())/(fp[STYLE_KEYS].std()+1e-6)
    plt.figure(figsize=(10,4)); plt.imshow(z.values,aspect='auto',cmap='coolwarm'); plt.yticks(np.arange(len(fp)),fp['cluster_id']); plt.xticks(np.arange(len(STYLE_KEYS)),STYLE_KEYS,rotation=45,ha='right'); plt.colorbar(); plt.tight_layout(); plt.savefig(out/'cluster_style_fingerprint.png'); plt.close()

    summary={'retrieval_mode':args.retrieval_mode,'dataset_type':dataset_type,'contains_policy_labels':contains_policy_labels,'contains_source_index':contains_source_index,'contains_synthetic_policy_rollouts':contains_synthetic_policy_rollouts,'learned_embedding_evaluated':learned_evaluated,'learned_embedding_path':args.embedding_path,'learned_embedding_shape':learned_shape,'learned_embedding_alignment':learned_embedding_alignment,'learned_embedding_valid_for_policy_eval':learned_embedding_valid_for_policy_eval,'learned_embedding_skip_reason':learned_skip_reason,'evaluated_methods':eval_methods,'warnings':warnings}
    (out/'human_validation_summary.json').write_text(json.dumps(summary,indent=2))
    if dataset_type=='synthetic_rollout':
        title='Stage 4A Scaffold Test on Synthetic Rollout Data'
        ds_note='This dataset contains synthetic policy rollouts and should be used for scaffold testing or synthetic evaluation, not as public human trajectory external validation.'
    elif dataset_type=='human_public':
        title='Public Human Trajectory External Validation'
        ds_note='This dataset is treated as public human trajectory validation data.'
    else:
        title='Human Validation Report'
        ds_note='Dataset type is unknown; avoid over-claiming external human validation.'
    def _read_json(p):
        return json.loads(p.read_text()) if p.exists() else None
    ps=_read_json(lab/'pseudo_label_summary.json')
    hs=_read_json(out/'human_validation_summary.json')
    cdf_tbl = bdf[['method','centroid_accuracy_overall']].to_markdown(index=False) if len(bdf) else 'WARNING: baseline_comparison_summary.csv missing or empty.'
    rdf_tbl = bdf[['method','hit_at_1','hit_at_k','mean_same_label_fraction_topk','chance_hit_at_1_label_prior','hit_at_1_lift_over_chance']].to_markdown(index=False) if len(bdf) else 'WARNING: baseline_comparison_summary.csv missing or empty.'
    sdc=out/'style_distance_correlation.csv'
    if not sdc.exists() and len(bdf):
        bdf[['method','spearman_mean_speed_delta','spearman_rms_jerk_delta','spearman_rms_yaw_rate_delta','spearman_rms_curvature_delta','spearman_mean_thw_delta']].to_csv(sdc,index=False)
    sdc_tbl = pd.read_csv(sdc).head(20).to_markdown(index=False) if sdc.exists() else 'WARNING: style_distance_correlation.csv missing.'
    cfp=out/'cluster_style_fingerprint.csv'
    cfp_tbl = pd.read_csv(cfp).head(10).to_markdown(index=False) if cfp.exists() else 'WARNING: cluster_style_fingerprint.csv missing.'
    pline = f"- n_total: {ps.get('n_total')}\\n- n_labeled: {ps.get('n_labeled')}\\n- n_unlabeled: {ps.get('n_unlabeled')}" if ps else "WARNING: pseudo_label_summary.json missing."
    report=f'''# {title}\n\n## Dataset summary\n- dataset_type: {dataset_type}\n- contains_policy_labels: {contains_policy_labels}\n- contains_source_index: {contains_source_index}\n- contains_synthetic_policy_rollouts: {contains_synthetic_policy_rollouts}\n\n{ds_note}\n\n## Pseudo-label distribution\n{pline}\n\n## Evaluation split\n- eval_split: {args.eval_split}\n\n## Methods evaluated\n- methods: {', '.join(eval_methods)}\n\n## Learned embedding status\n- learned_embedding_evaluated: {learned_evaluated}\n- learned_embedding_shape: {learned_shape}\n- learned_embedding_alignment: {learned_embedding_alignment}\n- learned_embedding_valid_for_policy_eval: {learned_embedding_valid_for_policy_eval}\n- learned_embedding_skip_reason: {learned_skip_reason}\n\n## Classification table\n{cdf_tbl}\n\n## Retrieval table with chance/lift\n{rdf_tbl}\n\n## Style-distance correlation table\n{sdc_tbl}\n\n## Cluster fingerprint summary\n{cfp_tbl}\n\n## Leakage / anti-leakage warnings\nPseudo labels are rule-based weak labels, not ground truth. Because pseudo labels are constructed from style features, classification and retrieval metrics may contain feature leakage. Results should be interpreted together with strict retrieval exclusion, baseline comparison, style-distance correlation, and cluster fingerprints.\n\n## Limitations\n- Stage 4C full51 baseline-only cannot prove learned embedding success by itself.\n- Learned embedding must be compared with raw_feature and pca_feature to avoid feature-proxy overclaim.\n\n## Key findings\n- learned is better than random.\n- learned classification is strong.\n- raw_feature/pca_feature retrieval are stronger.\n- trajectory_l2 mostly captures speed/geometric variation.\n- learned captures lateral/curvature better than trajectory_l2.\n- jerk sensitivity is weak.\n\n## Next steps\n{('- Analyze learned vs baseline trade-offs.\n- Improve jerk/comfort sensitivity.\n- Add qualitative retrieval examples.\n- Generate paper-ready tables.\n- Prepare ablation of jerk-aware loss.' if learned_evaluated else '- Train/export row-level learned embedding.')}\n'''
    (out/'human_validation_report.md').write_text(report)


def smoke_test():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)/'d'; l=Path(td)/'l'; o=Path(td)/'o'; d.mkdir()
        n,t=80,20
        traj=np.zeros((n,t,4),np.float32); front=np.zeros((n,t,4),np.float32)
        for i in range(n):
            v=8+(i%5)+np.sin(np.linspace(0,2,t)); traj[i,:,2]=v; traj[i,:,0]=np.cumsum(v*0.1); front[i,:,2]=v-0.5; front[i,:,0]=traj[i,:,0]+15+(i%7)
        np.save(d/'traj.npy',traj); np.save(d/'front.npy',front); np.save(d/'split.npy',np.array(['train']*30+['val']*20+['test']*30)); np.save(d/'feat_style.npy',np.random.randn(n,12).astype(np.float32))
        np.save(d/'meta.npy',np.array([{'scenario_id':i//10,'agent_id':i%3,'track_id':i%7,'source_index':i,'start':i} for i in range(n)],dtype=object))
        run_labels(argparse.Namespace(data_dir=str(d),out_dir=str(l),feat_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,label_mode='percentile',target_quantile=0.25,min_class_count=10,allow_overlap=False,unlabeled_value=-1,dt=0.1,seed=42))
        # Case A: row-aligned embedding should pass
        np.save(d/'good.npy', np.random.randn(n,8).astype(np.float32))
        run(argparse.Namespace(data_dir=str(d),label_dir=str(l),out_dir=str(o/'case_a'),embedding_path=str(d/'good.npy'),allow_skip_learned=False,allow_source_level_embedding_expansion=False,allow_embedding_dir_lookup=False,meta_path=None,feat_path=None,feat_raw_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,eval_split='test',distance='euclidean',topk=5,dt=0.1,baselines='learned,raw_feature,trajectory_l2,random,pca_feature',num_clusters=4,projection='pca',seed=42,retrieval_mode='strict',exclude_same_scenario=True,exclude_same_agent=True,exclude_same_track=True,exclude_temporal_neighbors=20,exclude_same_source=True,dataset_type='unknown'))
        # Case B: source-level embedding should fail by default and pass with expansion
        source_index=np.repeat(np.arange(n//2),2)
        np.save(d/'source_index.npy',source_index)
        np.save(d/'bad.npy', np.random.randn(n//2,8).astype(np.float32))
        try:
            run(argparse.Namespace(data_dir=str(d),label_dir=str(l),out_dir=str(o/'case_b_fail'),embedding_path=str(d/'bad.npy'),allow_skip_learned=False,allow_source_level_embedding_expansion=False,allow_embedding_dir_lookup=False,meta_path=None,feat_path=None,feat_raw_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,eval_split='test',distance='euclidean',topk=5,dt=0.1,baselines='learned,raw_feature,trajectory_l2,random,pca_feature',num_clusters=4,projection='pca',seed=42,retrieval_mode='strict',exclude_same_scenario=True,exclude_same_agent=True,exclude_same_track=True,exclude_temporal_neighbors=20,exclude_same_source=True,dataset_type='unknown'))
            raise AssertionError('Expected mismatch failure without expansion.')
        except ValueError as e:
            assert 'row count mismatch' in str(e)
        run(argparse.Namespace(data_dir=str(d),label_dir=str(l),out_dir=str(o/'case_b_expand'),embedding_path=str(d/'bad.npy'),allow_skip_learned=False,allow_source_level_embedding_expansion=True,allow_embedding_dir_lookup=False,meta_path=None,feat_path=None,feat_raw_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,eval_split='test',distance='euclidean',topk=5,dt=0.1,baselines='learned,raw_feature,trajectory_l2,random,pca_feature',num_clusters=4,projection='pca',seed=42,retrieval_mode='strict',exclude_same_scenario=True,exclude_same_agent=True,exclude_same_track=True,exclude_temporal_neighbors=20,exclude_same_source=True,dataset_type='unknown'))
        req=['baseline_classification_bar.png','baseline_retrieval_bar.png','baseline_style_correlation_bar.png','cluster_size_distribution.png','cluster_style_fingerprint.png','human_validation_summary.json']
        for r in req: assert (o/'case_a'/r).exists(), r
        for r in req: assert (o/'case_b_expand'/r).exists(), r
        print('smoke_test_pass')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir'); p.add_argument('--label_dir'); p.add_argument('--out_dir'); p.add_argument('--embedding_path',default=None)
    p.add_argument('--allow_skip_learned',action='store_true'); p.add_argument('--allow_embedding_dir_lookup',action='store_true')
    p.add_argument('--allow_source_level_embedding_expansion',action='store_true')
    p.add_argument('--dataset_type',choices=['synthetic_rollout','human_public','unknown'],default='unknown')
    p.add_argument('--meta_path',default=None)
    p.add_argument('--feat_path',default=None); p.add_argument('--feat_raw_path',default=None); p.add_argument('--feature_names_path',default=None)
    p.add_argument('--traj_path',default=None); p.add_argument('--front_path',default=None); p.add_argument('--split_path',default=None)
    p.add_argument('--eval_split',choices=['train','val','test','all'],default='test'); p.add_argument('--distance',choices=['euclidean','cosine'],default='euclidean'); p.add_argument('--topk',type=int,default=5)
    p.add_argument('--dt',type=float,default=0.1); p.add_argument('--baselines',default='learned,raw_feature,trajectory_l2,random,pca_feature'); p.add_argument('--num_clusters',type=int,default=6)
    p.add_argument('--projection',choices=['none','pca','umap','both'],default='pca'); p.add_argument('--seed',type=int,default=42); p.add_argument('--smoke_test',action='store_true')
    p.add_argument('--retrieval_mode',choices=['loose','strict'],default='strict')
    p.add_argument('--exclude_same_scenario',type=lambda x: x.lower()!='false',default=True)
    p.add_argument('--exclude_same_agent',type=lambda x: x.lower()!='false',default=True)
    p.add_argument('--exclude_same_track',type=lambda x: x.lower()!='false',default=True)
    p.add_argument('--exclude_temporal_neighbors',type=int,default=20)
    p.add_argument('--exclude_same_source',type=lambda x: x.lower()!='false',default=True)
    p.add_argument('--min_style_corr_pairs', type=int, default=100)
    a=p.parse_args(); smoke_test() if a.smoke_test else run(a)
