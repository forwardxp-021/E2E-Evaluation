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

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.assign_pseudo_style_labels import _compute_signals, run as run_labels

def dist(a,b,metric='euclidean'):
    if metric=='cosine':
      na=np.linalg.norm(a,axis=1,keepdims=True)+1e-9; nb=np.linalg.norm(b,axis=1,keepdims=True)+1e-9
      return 1-(a/na)@(b/nb).T
    aa=(a*a).sum(1,keepdims=True); bb=(b*b).sum(1,keepdims=True).T
    return np.sqrt(np.maximum(aa+bb-2*a@b.T,0))

def eval_method(X,labels,splits,eval_split='test',topk=5,metric='euclidean'):
    mask_lab=labels!=-1; msplit=np.ones(len(labels),bool) if eval_split=='all' else (splits==eval_split)
    ev=np.where(mask_lab & msplit)[0]; tr=np.where(mask_lab & (splits=='train'))[0]
    if len(tr)==0: tr=ev
    classes=[0,1,2]
    cents=np.vstack([X[tr][labels[tr]==c].mean(0) if np.any(labels[tr]==c) else np.zeros(X.shape[1]) for c in classes])
    pred=dist(X[ev],cents,metric).argmin(1)
    acc=float((pred==labels[ev]).mean()) if len(ev) else np.nan
    cm=confusion_matrix(labels[ev],pred,labels=classes)
    D=dist(X[ev],X[ev],metric); np.fill_diagonal(D,np.inf)
    nn=np.argsort(D,1)[:,:topk]; lq=labels[ev][:,None]; ln=labels[ev][nn]
    hit1=float((ln[:,0]==lq[:,0]).mean()); hitk=float(np.any(ln==lq,1).mean()); frac=float((ln==lq).mean())
    # correlations
    return acc,cm,hit1,hitk,frac,ev

def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_dir); lab=Path(args.label_dir)
    traj=np.load(args.traj_path or data/'traj.npy', allow_pickle=True)
    if traj.dtype == object:
        traj = np.stack(traj, axis=0)
    front=np.load(args.front_path or data/'front.npy', allow_pickle=True) if (args.front_path or data/'front.npy').exists() else None
    if front is not None and front.dtype == object:
        front = np.stack(front, axis=0)
    split=np.load(args.split_path or data/'split.npy',allow_pickle=True).astype(str)
    labels=np.load(lab/'pseudo_label.npy')
    sig=_compute_signals(traj,front,args.dt)
    feat=np.load(args.feat_path or data/'feat_style.npy')
    methods={}
    warnings=[]
    if 'learned' in args.baselines.split(','):
      ep=args.embedding_path
      if ep:
        ep_path=Path(ep)
        if ep_path.is_file(): emb=np.load(ep)
        elif ep_path.is_dir() and (ep_path/'embeddings.npy').exists(): emb=np.load(str(ep_path/'embeddings.npy'))
        else: warnings.append('learned embedding missing; skipped')
        if 'emb' in locals():
          if len(emb)==len(labels): methods['learned']=emb
          else: warnings.append(f'learned embedding length mismatch: {len(emb)} vs {len(labels)}; skipped')
    methods['raw_feature']=(feat-feat.mean(0))/(feat.std(0)+1e-6)
    xy=traj[:,:,:2].copy(); xy=xy-xy[:,[0],:]; ang=np.arctan2(xy[:,1,1],xy[:,1,0]); c=np.cos(-ang); s=np.sin(-ang)
    xr=xy[:,:,0]*c[:,None]-xy[:,:,1]*s[:,None]; yr=xy[:,:,0]*s[:,None]+xy[:,:,1]*c[:,None]; methods['trajectory_l2']=np.concatenate([xr,yr],1)
    dim=methods.get('learned',np.zeros((len(labels),20))).shape[1] if 'learned' in methods else 20
    rng=np.random.default_rng(args.seed); methods['random']=rng.normal(size=(len(labels),dim))
    p=min(dim,feat.shape[1],8); methods['pca_feature']=PCA(n_components=p,random_state=args.seed).fit_transform(methods['raw_feature'])
    rows=[]; best_c,best_r,best_s=None,None,None
    for m in [x for x in args.baselines.split(',') if x in methods]:
      X=methods[m]
      acc,cm,h1,hk,frac,ev=eval_method(X,labels,split,args.eval_split,args.topk,args.distance)
      pd.DataFrame(cm,index=['true0','true1','true2'],columns=['pred0','pred1','pred2']).to_csv(out/'pseudo_label_confusion_matrix.csv')
      plt.figure(); plt.imshow(cm); plt.colorbar(); plt.savefig(out/'pseudo_label_confusion_matrix.png'); plt.close()
      D=dist(X[ev],X[ev],args.distance); tri=np.triu_indices(len(ev),1); dvec=D[tri]
      sp={k:float(spearmanr(dvec,np.abs(v[ev][:,None]-v[ev][None,:])[tri]).correlation) for k,v in sig.items() if k in ['mean_speed','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw','min_thw']}
      rows.append({'method':m,'representation_dim':X.shape[1],'n_eval_samples':len(ev),'centroid_accuracy_overall':acc,'centroid_accuracy_by_label':'{}','hit_at_1':h1,'hit_at_k':hk,'mean_same_label_fraction_topk':frac,'spearman_mean_speed_delta':sp.get('mean_speed'),'spearman_rms_jerk_delta':sp.get('rms_jerk'),'spearman_rms_yaw_rate_delta':sp.get('rms_yaw_rate_proxy'),'spearman_rms_curvature_delta':sp.get('rms_curvature_proxy'),'spearman_mean_thw_delta':sp.get('mean_thw'),'notes':'weak-label evaluation'})
    bdf=pd.DataFrame(rows); bdf.to_csv(out/'baseline_comparison_summary.csv',index=False)
    bdf[['method','centroid_accuracy_overall']].to_csv(out/'pseudo_label_classification.csv',index=False)
    bdf[['method','hit_at_1','hit_at_k','mean_same_label_fraction_topk']].to_csv(out/'human_retrieval_summary.csv',index=False)
    bdf[[c for c in bdf.columns if c.startswith('spearman_') or c=='method']].to_csv(out/'style_distance_correlation.csv',index=False)
    # cluster fingerprint
    m=list(methods.keys())[0]; X=methods[m]; km=KMeans(n_clusters=args.num_clusters,random_state=args.seed,n_init=10).fit(X); cl=km.labels_
    crows=[]
    for k in range(args.num_clusters):
      idx=np.where(cl==k)[0]
      crows.append({'cluster':k,'cluster_size':len(idx),'pseudo_label_distribution':str(pd.Series(labels[idx]).value_counts().to_dict()),**{s:float(np.nanmean(sig[s][idx])) for s in ['mean_speed','rms_jerk','rms_yaw_rate_proxy','rms_curvature_proxy','mean_thw','min_thw']}})
    cdf=pd.DataFrame(crows); cdf.to_csv(out/'cluster_style_fingerprint.csv',index=False)
    plt.figure(figsize=(8,3)); plt.bar(cdf['cluster'].astype(str),cdf['cluster_size']); plt.savefig(out/'cluster_style_fingerprint.png'); plt.close()
    summary={'data_dir':args.data_dir,'label_dir':args.label_dir,'embedding_path':args.embedding_path,'n_total':int(len(labels)),'n_labeled':int(np.sum(labels!=-1)),'label_counts':{str(k):int(v) for k,v in pd.Series(labels).value_counts().to_dict().items()},'eval_split':args.eval_split,'baselines':args.baselines,'best_method_by_classification':bdf.sort_values('centroid_accuracy_overall',ascending=False)['method'].iloc[0] if len(bdf) else None,'best_method_by_retrieval':bdf.sort_values('hit_at_1',ascending=False)['method'].iloc[0] if len(bdf) else None,'best_method_by_style_correlation':bdf.sort_values('spearman_mean_speed_delta',ascending=False)['method'].iloc[0] if len(bdf) else None,'warnings':warnings+['Pseudo-label validation may contain feature leakage; interpret results together with retrieval visualization, cluster fingerprints, and baselines.']}
    (out/'human_validation_summary.json').write_text(json.dumps(summary,indent=2))
    txt='Pseudo labels are rule-based weak labels, not ground truth.\nThey are used for external validation only.\nLabel-defining features may bias classification metrics.\n'
    (out/'human_validation_report.md').write_text(txt)
    (out/'baseline_comparison_report.md').write_text(txt)

def smoke_test():
    with tempfile.TemporaryDirectory() as td:
      d=Path(td)/'d'; l=Path(td)/'l'; o=Path(td)/'o'; d.mkdir();
      n,t=80,20
      traj=np.zeros((n,t,4),np.float32); front=np.zeros((n,t,4),np.float32)
      for i in range(n):
        v=8+(i%5)+np.sin(np.linspace(0,2,t)); traj[i,:,2]=v; traj[i,:,0]=np.cumsum(v*0.1); front[i,:,2]=v-0.5; front[i,:,0]=traj[i,:,0]+15+(i%7)
      np.save(d/'traj.npy',traj); np.save(d/'front.npy',front); np.save(d/'split.npy',np.array(['train']*30+['val']*20+['test']*30)); np.save(d/'feat_style.npy',np.random.randn(n,12).astype(np.float32))
      run_labels(argparse.Namespace(data_dir=str(d),out_dir=str(l),feat_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,label_mode='percentile',target_quantile=0.25,min_class_count=10,allow_overlap=False,unlabeled_value=-1,dt=0.1,seed=42))
      run(argparse.Namespace(data_dir=str(d),label_dir=str(l),out_dir=str(o),embedding_path=None,feat_path=None,feat_raw_path=None,feature_names_path=None,traj_path=None,front_path=None,split_path=None,eval_split='test',distance='euclidean',topk=5,dt=0.1,baselines='raw_feature,trajectory_l2,random,pca_feature',num_clusters=4,projection='pca',seed=42))
      assert (o/'human_validation_summary.json').exists(); assert (o/'baseline_comparison_summary.csv').exists(); print('smoke_test_pass')

if __name__=='__main__':
  p=argparse.ArgumentParser()
  p.add_argument('--data_dir'); p.add_argument('--label_dir'); p.add_argument('--out_dir'); p.add_argument('--embedding_path',default=None)
  p.add_argument('--feat_path',default=None); p.add_argument('--feat_raw_path',default=None); p.add_argument('--feature_names_path',default=None)
  p.add_argument('--traj_path',default=None); p.add_argument('--front_path',default=None); p.add_argument('--split_path',default=None)
  p.add_argument('--eval_split',choices=['train','val','test','all'],default='test'); p.add_argument('--distance',choices=['euclidean','cosine'],default='euclidean'); p.add_argument('--topk',type=int,default=5)
  p.add_argument('--dt',type=float,default=0.1); p.add_argument('--baselines',default='learned,raw_feature,trajectory_l2,random,pca_feature'); p.add_argument('--num_clusters',type=int,default=6)
  p.add_argument('--projection',choices=['none','pca','umap','both'],default='pca'); p.add_argument('--seed',type=int,default=42); p.add_argument('--smoke_test',action='store_true')
  a=p.parse_args(); smoke_test() if a.smoke_test else run(a)
